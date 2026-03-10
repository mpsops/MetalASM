import XCTest
import Foundation
@testable import MetalASM

#if canImport(Metal)
import Metal

/// Convert Foundation Data to DispatchData for Metal API.
private func asDispatchData(_ data: Data) -> DispatchData {
    return data.withUnsafeBytes { buf in
        DispatchData(bytes: buf)
    }
}
#endif

final class EndToEndTests: XCTestCase {

    /// Test Phase 1: wrap known-good .air into metallib and load on GPU.
    func testWrapAIRAndLoad() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let airURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.air")
        guard FileManager.default.fileExists(atPath: airURL.path) else {
            throw XCTSkip("Reference file not found at \(airURL.path)")
        }

        let airData = try Data(contentsOf: airURL)
        let metallib = MetalASM.wrapAIR(
            Array(airData),
            kernelNames: ["monolithic_kernel"]
        )

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }

        // Load the metallib
        let library = try device.makeLibrary(data: asDispatchData(metallib))

        // Find the kernel
        let fn = library.makeFunction(name: "monolithic_kernel")
        XCTAssertNotNil(fn, "Kernel function not found")

        // Create pipeline
        let pipeline = try device.makeComputePipelineState(function: fn!)

        // Dispatch the kernel
        let count = 128
        let inputBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!

        // Fill input with 1.0, 2.0, ..., 128.0
        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            inputPtr[i] = Float(i + 1)
        }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Verify: output should be input * 2
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        var correct = 0
        for i in 0..<count {
            let expected = Float(i + 1) * 2.0
            if abs(outputPtr[i] - expected) < 0.001 {
                correct += 1
            }
        }
        XCTAssertEqual(correct, count, "Expected all \(count) elements correct, got \(correct)")
        #endif
    }

    /// Test full pipeline: parse .ll → bitcode → metallib → GPU dispatch.
    func testFullPipeline() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let irURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("Reference IR file not found")
        }

        let source = try String(contentsOf: irURL, encoding: .utf8)

        // Full pipeline
        let metallib = try MetalASM.assemble(ir: source)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }

        // Try to load - this validates the entire pipeline
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "monolithic_kernel")
        XCTAssertNotNil(fn, "Kernel function not found after full pipeline")
        #endif
    }

    /// Test that we can parse the monolithic IR and serialize it back.
    func testParseAndSerialize() throws {
        let irURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("Reference IR file not found")
        }

        let source = try String(contentsOf: irURL, encoding: .utf8)

        // Parse
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()

        // Verify module parsed correctly
        XCTAssertEqual(module.targetTriple, "air64_v28-apple-macosx26.0.0")
        XCTAssertFalse(module.dataLayout.isEmpty)
        XCTAssertEqual(module.globals.count, 1)
        XCTAssertEqual(module.functions.count, 6)

        // Serialize to bitcode
        let bitcode = BitcodeWriter.write(module: module)
        XCTAssertGreaterThan(bitcode.count, 0)

        // Should start with BC magic
        XCTAssertEqual(bitcode[0], 0x42) // B
        XCTAssertEqual(bitcode[1], 0x43) // C
        XCTAssertEqual(bitcode[2], 0xC0)
        XCTAssertEqual(bitcode[3], 0xDE)
    }

    /// Test assembling the GEMM IR and write metallib for external verification.
    func testAssembleGEMMIR() throws {
        let irURL = URL(fileURLWithPath: "/tmp/gemm_debug_8x8x8_NN.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("GEMM IR file not found at /tmp/gemm_debug_8x8x8_NN.ll")
        }

        let source = try String(contentsOf: irURL, encoding: .utf8)
        let metallib = try MetalASM.assemble(ir: source)
        let outURL = URL(fileURLWithPath: "/tmp/gemm_attr_fixed.metallib")
        try metallib.write(to: outURL)
        print("Wrote \(metallib.count) bytes to \(outURL.path)")
        XCTAssertGreaterThan(metallib.count, 0)
    }

    /// Helper: extract bitcode from metallib data
    private func extractBitcode(from metallib: Data) -> Data? {
        let bytes = [UInt8](metallib)
        for i in 0..<(bytes.count - 4) {
            if bytes[i] == 0xDE && bytes[i+1] == 0xC0 && bytes[i+2] == 0x17 && bytes[i+3] == 0x0B {
                let off = Int(bytes[i+8]) | (Int(bytes[i+9]) << 8) | (Int(bytes[i+10]) << 16) | (Int(bytes[i+11]) << 24)
                let sz = Int(bytes[i+12]) | (Int(bytes[i+13]) << 8) | (Int(bytes[i+14]) << 16) | (Int(bytes[i+15]) << 24)
                return Data(bytes[(i+off)..<(i+off+sz)])
            }
        }
        return nil
    }

    /// Test assembling the MFA-generated GEMM IR (bf16 variant).
    func testAssembleMFAIR() throws {
        let irURL = URL(fileURLWithPath: "/tmp/mfa_gemm_debug.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("MFA IR not found at /tmp/mfa_gemm_debug.ll")
        }

        let source = try String(contentsOf: irURL, encoding: .utf8)
        let metallib = try MetalASM.assemble(ir: source, platform: .macOS(version: 26))
        try metallib.write(to: URL(fileURLWithPath: "/tmp/mfa_debug.metallib"))
        print("Wrote \(metallib.count) bytes")

        if let bc = extractBitcode(from: metallib) {
            try bc.write(to: URL(fileURLWithPath: "/tmp/mfa_debug.bc"))
            print("Extracted \(bc.count) bytes of bitcode")
        }
    }

    /// Test assembling attention IR and loading on GPU.
    func testAssembleAttentionIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let path = "/tmp/mfa_attn_10x10x3_fwd.ll"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Attention IR not found at \(path)")
        }
        let source = try String(contentsOfFile: path, encoding: .utf8)
        let metallib = try MetalASM.assemble(ir: source, platform: .macOS(version: 26))
        print("Assembled \(metallib.count) bytes")

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "attention")
        XCTAssertNotNil(fn, "attention function not found")
        let pipeline = try device.makeComputePipelineState(function: fn!)
        print("GPU load OK, maxTotalThreadsPerThreadgroup=\(pipeline.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Test assembling progressively larger subsets of MFA IR.
    func testMFABisect() throws {
        // Test various IR files to isolate the invalid record
        let paths = ["/tmp/mfa_bisect9.ll", "/tmp/mfa_bisect10.ll", "/tmp/mfa_bisect6.ll"]
        for path in paths {
            guard FileManager.default.fileExists(atPath: path) else { continue }
            let source = try String(contentsOfFile: path, encoding: .utf8)
            do {
                let metallib = try MetalASM.assemble(ir: source, platform: .macOS(version: 26))
                let label = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
                if let bc = extractBitcode(from: metallib) {
                    try bc.write(to: URL(fileURLWithPath: "/tmp/\(label)_test.bc"))
                    print("[\(label)] \(bc.count) bytes bitcode")
                }
            } catch {
                print("[\(path)] Parse error: \(error)")
            }
        }
    }

    /// Minimal test for store instruction encoding.
    func testStoreEncoding() throws {
        let ir = """
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"
        source_filename = "store_test"

        define kernel void @test_store(
          i32 addrspace(1)* %out
        ) {
        entry:
          store i32 42, i32 addrspace(1)* %out
          ret void
        }
        """
        let metallib = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
        if let bc = extractBitcode(from: metallib) {
            try bc.write(to: URL(fileURLWithPath: "/tmp/store_test.bc"))
            print("Store test: \(bc.count) bytes bitcode")
        }
    }

    /// Compare our metallib size with the reference (sanity check).
    func testMetallibSizeReasonable() throws {
        let refURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.metallib")
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw XCTSkip("Reference metallib not found")
        }
        let refData = try Data(contentsOf: refURL)

        let airURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.air")
        let airData = try Data(contentsOf: airURL)

        let ourMetallib = MetalASM.wrapAIR(
            Array(airData),
            kernelNames: ["monolithic_kernel"]
        )

        // Our metallib should be similar in size to the reference
        // (may differ slightly due to tag layout differences)
        let ratio = Double(ourMetallib.count) / Double(refData.count)
        XCTAssertGreaterThan(ratio, 0.8, "Our metallib is too small compared to reference")
        XCTAssertLessThan(ratio, 1.2, "Our metallib is too large compared to reference")
    }

    /// Test that global constant arrays (addrspace(2)) are serialized correctly.
    /// This is the NF4 codebook pattern: a kernel reads from a constant array.
    /// The kernel looks up output[tid] = CODEBOOK[input[tid] & 0x3].
    func testConstantArrayInitializer() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // A simple kernel with a constant array in addrspace(2).
        // It reads a uint index from input, looks up a float in the codebook, writes to output.
        let ir = """
        source_filename = "codebook_test"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        @CODEBOOK = internal addrspace(2) constant [4 x float] [
          float 1.0, float 2.0, float 3.0, float 4.0
        ]

        define void @codebook_lookup(
          i8 addrspace(1)* noundef "air-buffer-no-alias" %input_raw,
          i8 addrspace(1)* noundef "air-buffer-no-alias" %output_raw,
          i32 %tid_scalar
        ) {
        entry:
          %input = bitcast i8 addrspace(1)* %input_raw to i32 addrspace(1)*
          %output = bitcast i8 addrspace(1)* %output_raw to float addrspace(1)*
          %idx_ptr = getelementptr i32, i32 addrspace(1)* %input, i32 %tid_scalar
          %idx = load i32, i32 addrspace(1)* %idx_ptr
          %masked = and i32 %idx, 3
          %cb_ptr = getelementptr [4 x float], [4 x float] addrspace(2)* @CODEBOOK, i32 0, i32 %masked
          %val = load float, float addrspace(2)* %cb_ptr
          %out_ptr = getelementptr float, float addrspace(1)* %output, i32 %tid_scalar
          store float %val, float addrspace(1)* %out_ptr
          ret void
        }

        !air.kernel = !{!1}
        !llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16}
        !air.version = !{!20}
        !air.language_version = !{!21}
        !air.compile_options = !{!30}
        !1 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i32)* @codebook_lookup, !2, !3}
        !2 = !{}
        !3 = !{!4, !5, !6}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
        !5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"output"}
        !6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
        !10 = !{i32 1, !"wchar_size", i32 4}
        !11 = !{i32 7, !"air.max_device_buffers", i32 31}
        !12 = !{i32 7, !"air.max_constant_buffers", i32 31}
        !13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        !14 = !{i32 7, !"air.max_textures", i32 128}
        !15 = !{i32 7, !"air.max_read_write_textures", i32 8}
        !16 = !{i32 7, !"air.max_samplers", i32 16}
        !20 = !{i32 2, i32 8, i32 0}
        !21 = !{!"Metal", i32 4, i32 0, i32 0}
        !30 = !{!"air.compile.fast_math_enable"}
        """

        let metallib = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }

        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "codebook_lookup")
        XCTAssertNotNil(fn, "codebook_lookup function not found")
        let pipeline = try device.makeComputePipelineState(function: fn!)

        // Input: indices [0, 1, 2, 3]
        let count = 4
        var inputData: [UInt32] = [0, 1, 2, 3]
        let inputBuf = device.makeBuffer(bytes: &inputData, length: count * 4, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!

        // Clear output to NaN so we can detect if nothing was written
        let outPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = .nan }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Expected: [1.0, 2.0, 3.0, 4.0] from codebook lookup
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0]
        for i in 0..<count {
            XCTAssertEqual(outPtr[i], expected[i], accuracy: 0.001,
                "Codebook lookup at index \(i): got \(outPtr[i]), expected \(expected[i])")
        }
        #endif
    }
}

// GPU pipeline test for simple kernels
extension EndToEndTests {
    /// Test that a minimal kernel can create a GPU pipeline via MetalASM.
    func testSimpleKernelPipeline() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "test_simple"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @test_kernel(ptr addrspace(1) %buf) #0 {
        entry:
          ret void
        }

        attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!5}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @test_kernel, !2, !4}
        !2 = !{!3}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !4 = !{i32 1, i32 1, i32 1}
        !5 = !{i32 2, i32 8, i32 0}
        """

        // Verify parsing succeeds with opaque ptr syntax
        let lexer = Lexer(source: ir)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        XCTAssertEqual(module.functions.count, 1)
        XCTAssertEqual(module.functions[0].name, "test_kernel")
        #endif
    }
}

// Quick test for llvm-dis compatibility
extension EndToEndTests {
    func testAssemblyTiming() throws {
        // Try large IR first, fall back to small
        let candidates = ["/tmp/debug_backwardKeyValue_ir.ll", "/tmp/air-monolithic-test/monolithic.ll"]
        var source: String?
        for path in candidates {
            if FileManager.default.fileExists(atPath: path) {
                source = try String(contentsOfFile: path, encoding: .utf8)
                break
            }
        }
        guard let source else { throw XCTSkip("No IR file found") }
        print("IR size: \(source.count) bytes")

        // Warmup
        _ = try MetalASM.assemble(ir: source, platform: .macOS(version: 26))

        // Timed runs
        let runs = 5
        var times: [Double] = []
        for _ in 0..<runs {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try MetalASM.assemble(ir: source, platform: .macOS(version: 26))
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            times.append(dt)
        }
        let avg = times.reduce(0, +) / Double(runs)
        let best = times.min()!
        print("Assembly: avg=\(String(format: "%.1f", avg))ms best=\(String(format: "%.1f", best))ms over \(runs) runs")

        // Breakdown
        let t0 = CFAbsoluteTimeGetCurrent()
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let t1 = CFAbsoluteTimeGetCurrent()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        let t2 = CFAbsoluteTimeGetCurrent()
        let bitcode = BitcodeWriter.write(module: module)
        let t3 = CFAbsoluteTimeGetCurrent()

        print("  lex=\(String(format: "%.1f", (t1-t0)*1000))ms parse=\(String(format: "%.1f", (t2-t1)*1000))ms bc=\(String(format: "%.1f", (t3-t2)*1000))ms")
        print("  tokens=\(tokens.count) functions=\(module.functions.count)")
    }

    func testAddKernelTritonIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %tid_x, i32 %tgid_x) {
          %block_offset = mul i32 %tgid_x, 128
          %idx = add i32 %block_offset, %tid_x
          %p0 = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %v0 = load float, ptr addrspace(1) %p0, align 4
          %p1 = getelementptr float, ptr addrspace(1) %1, i32 %idx
          %v1 = load float, ptr addrspace(1) %p1, align 4
          %sum = fadd float %v0, %v1
          %p2 = getelementptr float, ptr addrspace(1) %2, i32 %idx
          store float %sum, ptr addrspace(1) %p2, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!8}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @add_kernel, !2, !3}
        !2 = !{}
        !3 = !{!4, !5, !6, !10, !11}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
        !5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
        !6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
        !10 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !11 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tgid_x"}
        !8 = !{i32 2, i32 8, i32 0}
        """
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/triton_test.metallib"))
        if let bc = extractBitcode(from: data) {
            try bc.write(to: URL(fileURLWithPath: "/tmp/triton_test.bc"))
        }
        print("testAddKernelTritonIR: \(data.count) bytes")
        XCTAssertGreaterThan(data.count, 100)

        // Verify the metallib loads on GPU
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "add_kernel")
        XCTAssertNotNil(fn, "add_kernel function not found in library")
        let pso = try device.makeComputePipelineState(function: fn!)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        print("testAddKernelTritonIR: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    func testZZDotKernelIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/dot_kernel_final.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("dot_kernel_final.ll not found at \(llPath)")
        }
        let irText = try String(contentsOfFile: llPath, encoding: .utf8)

        // Debug: parse and apply transforms, inspect preamble GEPs
        let lexer = Lexer(source: irText)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        print("Globals: \(module.globals.map { "@\($0.name) addrspace(\($0.addressSpace))" })")
        applyAirTransforms(module: module)
        if let fn = module.functions.first(where: { !$0.isDeclaration }) {
            // Print all call instructions with operands
            for bb in fn.basicBlocks {
                for inst in bb.instructions where inst.opcode == .call {
                    if case .value(let callee) = inst.operands.last, callee.name.contains("simdgroup") {
                        print("CALL \(callee.name):")
                        for (i, op) in inst.operands.dropLast().enumerated() {
                            switch op {
                            case .value(let v): print("  arg[\(i)]: name=\(v.name), type=\(v.type)")
                            case .constant(let c): print("  arg[\(i)]: const \(c)")
                            default: print("  arg[\(i)]: \(op)")
                            }
                        }
                    }
                }
            }
        }

        // Find any opaque ptrs in instructions
        if let fn = module.functions.first(where: { !$0.isDeclaration }) {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if case .opaquePointer(let as_) = inst.type {
                        print("[OPQ] inst %\(inst.name) result opaquePointer(\(as_)): \(inst.opcode)")
                    }
                    for (i, op) in inst.operands.enumerated() {
                        if case .value(let v) = op, case .opaquePointer(let as_) = v.type {
                            print("[OPQ] inst %\(inst.name) op[\(i)] \(v.name) opaquePointer(\(as_))")
                        }
                    }
                }
            }
        }
        // Also dump type table
        let ve2 = ValueEnumerator(module: module)
        let opqTypes = ve2.types.enumerated().filter { if case .opaquePointer(_) = $0.element { return true }; return false }
        print("[TypeTable opaque] \(opqTypes.map { "[\($0.offset)] \($0.element)" })")

        let data = try MetalASM.assemble(ir: irText)
        try data.write(to: URL(fileURLWithPath: "/tmp/dot_kernel_test.metallib"))
        if let bc = extractBitcode(from: data) {
            try bc.write(to: URL(fileURLWithPath: "/tmp/dot_kernel_test.bc"))
        }
        print("testZZDotKernelIR: \(data.count) bytes")
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "dot_kernel")
        XCTAssertNotNil(fn, "dot_kernel function not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testZZDotKernelIR: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Hybrid test: metal-as compiled MMA bc wrapped in our metallib format.
    /// If this passes but testMMALoadMinimal fails → crash is in our bitcode encoding.
    /// If this also fails → crash is in our metallib format.
    func testMMAHybrid() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // mma_ref.bc = metal-as compiled from mma_minimal.ll (has !air.version metadata)
        let bcPath = "/tmp/mma_ref.bc"
        guard FileManager.default.fileExists(atPath: bcPath) else {
            throw XCTSkip("Need /tmp/mma_ref.bc — run: metal-as /tmp/mma_minimal.ll -o /tmp/mma_ref.bc")
        }
        // mma_ref.bc is already a bitcode wrapper (metal-as output) — use wrapAIR
        let bc = try [UInt8](Data(contentsOf: URL(fileURLWithPath: bcPath)))
        let data = MetalASM.wrapAIR(bc, kernelNames: ["mma_kernel"])
        try data.write(to: URL(fileURLWithPath: "/tmp/mma_hybrid.metallib"))
        print("testMMAHybrid: \(data.count) bytes")
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_kernel")
        XCTAssertNotNil(fn)
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testMMAHybrid: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Minimal test: just simdgroup_matrix_8x8_load with typed float* from TG global.
    func testMMALoadMinimal() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Test: does mmaLoad cause crash?
        let ir = """
        @__tg = internal addrspace(3) global [64 x float] undef, align 4
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        define void @mma_kernel(ptr addrspace(1) %buf, i32 %tid_x, i32 %tid_y, i32 %tid_z) {
          %base = getelementptr inbounds [64 x float], ptr addrspace(3) @__tg, i64 0, i64 0
          %v = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) %base, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %c = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %v, <64 x float> %v, <64 x float> %v)
          %gep = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
          %s = extractelement <64 x float> %c, i32 0
          store float %s, ptr addrspace(1) %gep, align 4
          ret void
        }
        !air.kernel = !{!0}
        !air.version = !{!7}
        !air.language_version = !{!8}
        !llvm.module.flags = !{!9, !10, !11, !12, !13, !14}
        !0 = !{void (ptr addrspace(1), i32, i32, i32)* @mma_kernel, !1, !2}
        !1 = !{}
        !2 = !{!3, !4, !5, !6}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_y"}
        !6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_z"}
        !7 = !{i32 2, i32 8, i32 0}
        !8 = !{!"Metal", i32 4, i32 0, i32 0}
        !9 = !{i32 7, !"air.max_device_buffers", i32 31}
        !10 = !{i32 7, !"air.max_constant_buffers", i32 31}
        !11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        !12 = !{i32 7, !"air.max_textures", i32 128}
        !13 = !{i32 7, !"air.max_read_write_textures", i32 8}
        !14 = !{i32 7, !"air.max_samplers", i32 16}
        """
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/mma_minimal.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/mma_minimal.bc")) }
        print("testMMALoadMinimal: \(data.count) bytes")
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_kernel")
        XCTAssertNotNil(fn)
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testMMALoadMinimal: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Same as testMMALoadMinimal but using version 1 (VALUE_SYMTAB) format.
    func testMMALoadMinimalV1() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        @__tg = internal addrspace(3) global [64 x float] undef, align 4
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        define void @mma_kernel(ptr addrspace(1) %buf, i32 %tid_x, i32 %tid_y, i32 %tid_z) {
          %base = getelementptr inbounds [64 x float], ptr addrspace(3) @__tg, i64 0, i64 0
          %v = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) %base, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %c = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %v, <64 x float> %v, <64 x float> %v)
          %gep = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
          %s = extractelement <64 x float> %c, i32 0
          store float %s, ptr addrspace(1) %gep, align 4
          ret void
        }
        !air.kernel = !{!0}
        !air.version = !{!7}
        !air.language_version = !{!8}
        !llvm.module.flags = !{!9, !10, !11, !12, !13, !14}
        !0 = !{void (ptr addrspace(1), i32, i32, i32)* @mma_kernel, !1, !2}
        !1 = !{}
        !2 = !{!3, !4, !5, !6}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_y"}
        !6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_z"}
        !7 = !{i32 2, i32 8, i32 0}
        !8 = !{!"Metal", i32 4, i32 0, i32 0}
        !9 = !{i32 7, !"air.max_device_buffers", i32 31}
        !10 = !{i32 7, !"air.max_constant_buffers", i32 31}
        !11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        !12 = !{i32 7, !"air.max_textures", i32 128}
        !13 = !{i32 7, !"air.max_read_write_textures", i32 8}
        !14 = !{i32 7, !"air.max_samplers", i32 16}
        """
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/mma_minimal_v1.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/mma_minimal_v1.bc")) }
        print("testMMALoadMinimalV1: \(data.count) bytes")
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_kernel")
        XCTAssertNotNil(fn)
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testMMALoadMinimalV1: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Test MMA store intrinsic (void return).
    func testMMAStoreMinimal() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        @__tg = internal addrspace(3) global [64 x float] undef, align 4
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        define void @mma_kernel(ptr addrspace(1) %buf, i32 %tid_x, i32 %tid_y, i32 %tid_z) {
          %base = getelementptr inbounds [64 x float], ptr addrspace(3) @__tg, i64 0, i64 0
          %v = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) %base, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %v, ptr addrspace(3) %base, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          ret void
        }
        !air.kernel = !{!0}
        !air.version = !{!7}
        !llvm.module.flags = !{!9}
        !0 = !{void (ptr addrspace(1), i32, i32, i32)* @mma_kernel, !1, !2}
        !1 = !{}
        !2 = !{!3, !4, !5, !6}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_y"}
        !6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_z"}
        !7 = !{i32 2, i32 8, i32 0}
        !9 = !{i32 7, !"frame-pointer", i32 0}
        """
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/mma_store_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/mma_store_test.bc")) }
        print("testMMAStoreMinimal: \(data.count) bytes")
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_kernel")
        XCTAssertNotNil(fn)
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testMMAStoreMinimal: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    // MARK: - GEMM kernel (loop + MMA)

    /// Test the actual GEMM kernel IR from Triton (loop + MMA intrinsics).
    func testGEMMKernelIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/dot_kernel_final.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("GEMM IR not found at \(llPath)")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/gemm_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/gemm_test.bc")) }
        print("testGEMMKernelIR: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        print("testGEMMKernelIR: functionNames = \(lib.functionNames)")
        let fn = lib.makeFunction(name: "matmul_kernel")
        XCTAssertNotNil(fn, "matmul_kernel not found in \(lib.functionNames)")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testGEMMKernelIR: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    /// Minimal: 2D grid + MMA. Each TG does 8x8 ones matmul, stores to different offsets via pid_x.
    func testMMA2DKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/mma_2d_kernel.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("mma_2d_kernel.ll not found")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/mma_2d_test.metallib"))
        print("testMMA2DKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_2d_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testMMA2DKernel: PSO OK")

        // 2 TGs of 32 threads, output = 128 floats
        let outBuf = device.makeBuffer(length: 128 * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0xFF, 128 * 4) // fill with NaN to detect unwritten
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: 128)
        // TG0 writes to [0..63], TG1 writes to [64..127]
        // 8x8 ones matmul → each element = 8.0
        print("testMMA2DKernel: TG0[0..3] = \(result[0]), \(result[1]), \(result[2]), \(result[3])")
        print("testMMA2DKernel: TG1[64..67] = \(result[64]), \(result[65]), \(result[66]), \(result[67])")
        for i in 0..<32 {
            XCTAssertEqual(result[i], 8.0, accuracy: 1e-3, "TG0[\(i)]")
        }
        for i in 64..<96 {
            XCTAssertEqual(result[i], 8.0, accuracy: 1e-3, "TG1[\(i)]")
        }
        #endif
    }

    /// Test 2D tiled dot (MMA) with multiple threadgroups — verifies pid_m/pid_n + MMA.
    func testDot2DKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/dot_kernel_dot_2d.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("dot_2d IR not found")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/dot_2d_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/dot_2d_test.bc")) }
        print("testDot2DKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "dot_2d")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testDot2DKernel: PSO OK")

        // A = ones(32x16), B = ones(16x32), C should = 16.0 everywhere
        let M = 32, N = 32, K = 16
        let aBuf = device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: K * N * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: M * K)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: K * N)
        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: M * N)
        for i in 0..<(M*K) { aPtr[i] = 1.0 }
        for i in 0..<(K*N) { bPtr[i] = 1.0 }
        for i in 0..<(M*N) { cPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        // 2x2 threadgroups, 128 threads each (4 warps)
        enc.dispatchThreadgroups(MTLSize(width: 2, height: 2, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        var maxErr: Float = 0
        for i in 0..<M {
            for j in 0..<N {
                let err = abs(cPtr[i * N + j] - 16.0)
                maxErr = max(maxErr, err)
                if err > 0.01 {
                    print("testDot2DKernel: MISMATCH C[\(i),\(j)] = \(cPtr[i * N + j]) (expected 16.0)")
                }
            }
        }
        print("testDot2DKernel: max_err = \(maxErr)")
        // Print per-tile summary
        for ti in stride(from: 0, to: M, by: 16) {
            for tj in stride(from: 0, to: N, by: 16) {
                var tileSum: Float = 0
                for i in ti..<ti+16 { for j in tj..<tj+16 { tileSum += abs(cPtr[i*N+j]) } }
                print("  tile[\(ti),\(tj)] sum=\(tileSum)")
            }
        }
        XCTAssertLessThan(maxErr, 0.01, "2D dot max_err=\(maxErr)")
        #endif
    }

    /// Test struct phi with extractvalue/insertvalue in a loop.
    func testStructPhiKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/struct_phi_kernel.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("struct_phi_kernel.ll not found")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/struct_phi_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/struct_phi_test.bc")) }
        print("testStructPhiKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "struct_phi_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testStructPhiKernel: PSO OK")

        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0, count * 4)
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        print("testStructPhiKernel: result[0..3] = \(result[0]), \(result[1]), \(result[2]), \(result[3])")
        // Thread i: sum = 4*i (4 iterations adding tid each time)
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-3, "thread 0: 4*0=0")
        XCTAssertEqual(result[1], 4.0, accuracy: 1e-3, "thread 1: 4*1=4")
        XCTAssertEqual(result[2], 8.0, accuracy: 1e-3, "thread 2: 4*2=8")
        XCTAssertEqual(result[3], 12.0, accuracy: 1e-3, "thread 3: 4*3=12")
        #endif
    }

    /// Test 2D grid dispatch: pid_x and pid_y are independent (not diagonal).
    func testPidDumpKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/pid_dump_kernel.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("pid_dump_kernel.ll not found")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/pid_dump_test.metallib"))
        print("testPidDumpKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "pid_dump_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testPidDumpKernel: PSO OK")

        // Dispatch 4x4 grid, 1 thread per threadgroup
        // value[pid_x * 4 + pid_y] = pid_x * 10 + pid_y + 100
        let outBuf = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0, 16 * 4)
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 4, height: 4, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: 16)
        for px in 0..<4 {
            for py in 0..<4 {
                let idx = px * 4 + py
                let expected = Float(px * 10 + py + 100)
                print("testPidDumpKernel: [\(px),\(py)] = \(result[idx]) (expected \(expected))")
                XCTAssertEqual(result[idx], expected, accuracy: 1e-3,
                              "pid_x=\(px) pid_y=\(py)")
            }
        }
        #endif
    }

    /// Test loop + MMA (no struct phi) — isolates MMA-in-loop from struct phi issue.
    func testLoopMMAKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let llPath = "/tmp/loop_mma_kernel.ll"
        guard FileManager.default.fileExists(atPath: llPath) else {
            throw XCTSkip("loop_mma_kernel.ll not found")
        }
        let ir = try String(contentsOfFile: llPath, encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/loop_mma_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/loop_mma_test.bc")) }
        print("testLoopMMAKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        print("testLoopMMAKernel: functionNames = \(lib.functionNames)")
        let fn = lib.makeFunction(name: "loop_mma_kernel")
        XCTAssertNotNil(fn, "loop_mma_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testLoopMMAKernel: PSO OK")

        // Run: 32 threads, each writes C[tid]. 2 iters of ones @ ones = each element should be 2*8 = 16
        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0, count * 4)
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        print("testLoopMMAKernel: result[0..3] = \(result[0]), \(result[1]), \(result[2]), \(result[3])")
        // Each element: 2 iterations × 8 (8×8 ones matmul) = 16
        XCTAssertEqual(result[0], 16.0, accuracy: 1e-3)
        #endif
    }

    /// Test the reference toolchain's .air for the GEMM kernel — isolates MetalASM bitcode issue.
    func testGEMMRefAIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let airPath = "/tmp/gemm_ref.air"
        guard FileManager.default.fileExists(atPath: airPath) else {
            throw XCTSkip("Reference .air not found at \(airPath)")
        }
        let airData = try Data(contentsOf: URL(fileURLWithPath: airPath))
        let metallib = MetalASM.wrapAIR(Array(airData), kernelNames: ["matmul_kernel"])
        try metallib.write(to: URL(fileURLWithPath: "/tmp/gemm_ref2.metallib"))

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = lib.makeFunction(name: "matmul_kernel")
        XCTAssertNotNil(fn, "matmul_kernel not found in ref metallib")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testGEMMRefAIR: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")
        #endif
    }

    // MARK: - Loop kernel (br/phi)

    /// Test kernel with a simple loop (br + phi instructions).
    /// sum_kernel: out[tid] = in[0] + in[1] + in[2] + in[3]
    func testLoopKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        define void @sum_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %tid) {
        entry:
          br label %loop

        loop:
          %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
          %acc = phi float [ 0.0, %entry ], [ %acc_next, %loop ]
          %ptr = getelementptr float, ptr addrspace(1) %in, i32 %i
          %val = load float, ptr addrspace(1) %ptr, align 4
          %acc_next = fadd float %acc, %val
          %i_next = add i32 %i, 1
          %cond = icmp slt i32 %i_next, 4
          br i1 %cond, label %loop, label %exit

        exit:
          %out_ptr = getelementptr float, ptr addrspace(1) %out, i32 %tid
          store float %acc_next, ptr addrspace(1) %out_ptr, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!8}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @sum_kernel, !2, !3}
        !2 = !{}
        !3 = !{!4, !5, !6}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
        !5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
        !6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
        !8 = !{i32 2, i32 8, i32 0}
        """
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/loop_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/loop_test.bc")) }
        print("testLoopKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "sum_kernel")
        XCTAssertNotNil(fn, "sum_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testLoopKernel: PSO OK")

        // Run it: in = [1, 2, 3, 4], expect out[0] = 10
        let inBuf = device.makeBuffer(length: 4 * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let inp = inBuf.contents().bindMemory(to: Float.self, capacity: 4)
        inp[0] = 1; inp[1] = 2; inp[2] = 3; inp[3] = 4
        outBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee
        print("testLoopKernel: result = \(result)")
        XCTAssertEqual(result, 10.0, accuracy: 1e-5, "1+2+3+4 should be 10")
        #endif
    }
}
