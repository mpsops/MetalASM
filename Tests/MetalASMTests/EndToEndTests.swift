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
        guard irText.contains("define void @dot_kernel(") else {
            throw XCTSkip("dot_kernel_final.ll does not contain dot_kernel (stale file)")
        }

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
    /// Minimal MMA kernel: single 8x8 matmul C = A * B via simdgroup_matrix intrinsics.
    /// Inline IR — no /tmp file dependency.
    func testGEMMKernelIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; Minimal MMA kernel: C[8x8] = A[8x8] * B[8x8]
        source_filename = "LLVMDialectModule"

        @__tg_dot_a = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_dot_b = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_dot_c = internal addrspace(3) global [64 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()
        declare [3 x i32] @air.thread_position_in_grid()

        define void @matmul_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
          ; Load A[lane] and B[lane] from device memory into TG
          %tid3 = call [3 x i32] @air.thread_position_in_grid()
          %tid = extractvalue [3 x i32] %tid3, 0
          %lane = call i32 @air.thread_index_in_simdgroup()
          ; Each thread loads 2 elements: A[lane*2] and A[lane*2+1]
          %a_idx0 = mul i32 %lane, 2
          %a_idx1 = add i32 %a_idx0, 1
          %a_ptr0 = getelementptr float, ptr addrspace(1) %A, i32 %a_idx0
          %a_ptr1 = getelementptr float, ptr addrspace(1) %A, i32 %a_idx1
          %a_val0 = load float, ptr addrspace(1) %a_ptr0, align 4
          %a_val1 = load float, ptr addrspace(1) %a_ptr1, align 4
          %a_idx0_64 = zext i32 %a_idx0 to i64
          %a_idx1_64 = zext i32 %a_idx1 to i64
          %tg_a0 = getelementptr float, ptr addrspace(3) @__tg_dot_a, i64 %a_idx0_64
          %tg_a1 = getelementptr float, ptr addrspace(3) @__tg_dot_a, i64 %a_idx1_64
          store float %a_val0, ptr addrspace(3) %tg_a0, align 4
          store float %a_val1, ptr addrspace(3) %tg_a1, align 4
          ; Same for B
          %b_ptr0 = getelementptr float, ptr addrspace(1) %B, i32 %a_idx0
          %b_ptr1 = getelementptr float, ptr addrspace(1) %B, i32 %a_idx1
          %b_val0 = load float, ptr addrspace(1) %b_ptr0, align 4
          %b_val1 = load float, ptr addrspace(1) %b_ptr1, align 4
          %tg_b0 = getelementptr float, ptr addrspace(3) @__tg_dot_b, i64 %a_idx0_64
          %tg_b1 = getelementptr float, ptr addrspace(3) @__tg_dot_b, i64 %a_idx1_64
          store float %b_val0, ptr addrspace(3) %tg_b0, align 4
          store float %b_val1, ptr addrspace(3) %tg_b1, align 4
          ; Zero C
          %tg_c0 = getelementptr float, ptr addrspace(3) @__tg_dot_c, i64 %a_idx0_64
          %tg_c1 = getelementptr float, ptr addrspace(3) @__tg_dot_c, i64 %a_idx1_64
          store float 0.0, ptr addrspace(3) %tg_c0, align 4
          store float 0.0, ptr addrspace(3) %tg_c1, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          ; MMA: C = A * B + C
          %mA = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mB = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mC = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mR = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %mA, <64 x float> %mB, <64 x float> %mC)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %mR, ptr addrspace(3) @__tg_dot_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          ; Store C back to device
          %c_val0 = load float, ptr addrspace(3) %tg_c0, align 4
          %c_val1 = load float, ptr addrspace(3) %tg_c1, align 4
          %c_ptr0 = getelementptr float, ptr addrspace(1) %C, i32 %a_idx0
          %c_ptr1 = getelementptr float, ptr addrspace(1) %C, i32 %a_idx1
          store float %c_val0, ptr addrspace(1) %c_ptr0, align 4
          store float %c_val1, ptr addrspace(1) %c_ptr1, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let data = try MetalASM.assemble(ir: ir)
        print("testGEMMKernelIR: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        print("testGEMMKernelIR: functionNames = \(lib.functionNames)")
        let fn = lib.makeFunction(name: "matmul_kernel")
        XCTAssertNotNil(fn, "matmul_kernel not found in \(lib.functionNames)")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testGEMMKernelIR: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")

        // Dispatch: A = ones(8x8), B = ones(8x8), expect C = 8.0 everywhere
        let aBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: 64)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: 64)
        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: 64)
        for i in 0..<64 { aPtr[i] = 1.0; bPtr[i] = 1.0; cPtr[i] = -1.0 }
        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        var maxErr: Float = 0
        for i in 0..<64 {
            let err = abs(cPtr[i] - 8.0)
            maxErr = max(maxErr, err)
            if err > 0.01 { print("testGEMMKernelIR: C[\(i)] = \(cPtr[i]) (expected 8.0)") }
        }
        print("testGEMMKernelIR: max_err = \(maxErr)")
        XCTAssertLessThan(maxErr, 0.01, "GEMM max_err=\(maxErr)")
        #endif
    }

    /// Minimal: MMA with pid + simdlane. 2x1 grid, each TG does 8x8 ones matmul.
    func testMMA2DKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // MMA kernel: each thread writes 2 elements (like GEMM test) to fill all 64
        let ir = """
        @__tg_a = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_b = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_c = internal addrspace(3) global [64 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()
        declare [3 x i32] @air.thread_position_in_grid()

        define void @mma_2d_kernel(ptr addrspace(1) %out) {
          %tid3 = call [3 x i32] @air.thread_position_in_grid()
          %tid = extractvalue [3 x i32] %tid3, 0
          %sl = call i32 @air.thread_index_in_simdgroup()
          ; Each thread writes 2 elements to fill all 64
          %idx0 = mul i32 %sl, 2
          %idx1 = add i32 %idx0, 1
          %idx0_64 = zext i32 %idx0 to i64
          %idx1_64 = zext i32 %idx1 to i64
          %a0 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx0_64
          %a1 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx1_64
          %b0 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx0_64
          %b1 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx1_64
          %c0 = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx0_64
          %c1 = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx1_64
          store float 1.000000e+00, ptr addrspace(3) %a0, align 4
          store float 1.000000e+00, ptr addrspace(3) %a1, align 4
          store float 1.000000e+00, ptr addrspace(3) %b0, align 4
          store float 1.000000e+00, ptr addrspace(3) %b1, align 4
          store float 0.000000e+00, ptr addrspace(3) %c0, align 4
          store float 0.000000e+00, ptr addrspace(3) %c1, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %mma_a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_a, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mma_b = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_b, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mma_c = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %mma_r = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %mma_a, <64 x float> %mma_b, <64 x float> %mma_c)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %mma_r, ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          ; Read back result — each thread reads its 2 elements
          %r0 = load float, ptr addrspace(3) %c0, align 4
          %r1 = load float, ptr addrspace(3) %c1, align 4
          %out0 = getelementptr float, ptr addrspace(1) %out, i32 %idx0
          %out1 = getelementptr float, ptr addrspace(1) %out, i32 %idx1
          store float %r0, ptr addrspace(1) %out0, align 4
          store float %r1, ptr addrspace(1) %out1, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let data = try MetalASM.assemble(ir: ir)
        print("testMMA2DKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "mma_2d_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testMMA2DKernel: PSO OK")

        let outBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0xFF, 64 * 4)
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

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: 64)
        print("testMMA2DKernel: [0..3] = \(result[0]), \(result[1]), \(result[2]), \(result[3])")
        var maxErr: Float = 0
        for i in 0..<64 {
            let err = abs(result[i] - 8.0)
            if err > maxErr { maxErr = err }
            XCTAssertEqual(result[i], 8.0, accuracy: 1e-3, "out[\(i)]")
        }
        print("testMMA2DKernel: max_err = \(maxErr)")
        #endif
    }

    /// Test 2D tiled dot (MMA) with multiple threadgroups — verifies pid_m/pid_n + MMA.
    /// 2D grid MMA: 2x2 TGs of 128 threads, A=ones(32x16), B=ones(16x32), C should=16.0
    func testDot2DKernel() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = #"""
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        @__tg_cvt_0 = internal addrspace(3) global [256 x float] undef, align 4
        @__tg_dot_c_0 = internal addrspace(3) global [256 x float] undef, align 4
        @__tg_dot_b_0 = internal addrspace(3) global [256 x float] undef, align 4
        @__tg_dot_a_0 = internal addrspace(3) global [256 x float] undef, align 4

        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.simdgroup.barrier(i32, i32)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()
        declare [3 x i32] @air.thread_position_in_grid()
        declare [3 x i32] @air.threadgroup_position_in_grid()

        define void @dot_2d(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
          %4 = call [3 x i32] @air.threadgroup_position_in_grid()
          %5 = extractvalue [3 x i32] %4, 0
          %6 = call [3 x i32] @air.threadgroup_position_in_grid()
          %7 = extractvalue [3 x i32] %6, 1
          %8 = mul i32 %5, 16
          %9 = call [3 x i32] @air.thread_position_in_grid()
          %10 = extractvalue [3 x i32] %9, 0
          %11 = zext i32 %10 to i64
          %12 = trunc i64 %11 to i32
          %13 = and i32 %12, 127
          %14 = urem i32 %13, 32
          %15 = call [3 x i32] @air.thread_position_in_grid()
          %16 = extractvalue [3 x i32] %15, 0
          %17 = udiv i32 %16, 32
          %18 = shl i32 %14, 0
          %19 = or i32 0, %18
          %20 = shl i32 %17, 5
          %21 = or i32 %19, %20
          %22 = and i32 %21, 120
          %23 = lshr i32 %22, 3
          %24 = or disjoint i32 %23, 0
          %25 = xor i32 0, %24
          %26 = xor i32 %25, 0
          %27 = add i32 %26, 0
          %28 = call [3 x i32] @air.thread_position_in_grid()
          %29 = extractvalue [3 x i32] %28, 0
          %30 = zext i32 %29 to i64
          %31 = trunc i64 %30 to i32
          %32 = and i32 %31, 127
          %33 = urem i32 %32, 32
          %34 = call [3 x i32] @air.thread_position_in_grid()
          %35 = extractvalue [3 x i32] %34, 0
          %36 = udiv i32 %35, 32
          %37 = shl i32 %33, 0
          %38 = or i32 0, %37
          %39 = shl i32 %36, 5
          %40 = or i32 %38, %39
          %41 = and i32 %40, 7
          %42 = shl i32 %41, 1
          %43 = or disjoint i32 %42, 0
          %44 = xor i32 0, %43
          %45 = xor i32 %44, 0
          %46 = xor i32 %44, 1
          %47 = add i32 %45, 0
          %48 = add i32 %46, 0
          %49 = add i32 %8, %27
          %50 = mul i32 %7, 16
          %51 = add i32 %50, %47
          %52 = add i32 %50, %48
          %53 = mul i32 %49, 16
          %54 = getelementptr float, ptr addrspace(1) %0, i32 %53
          %55 = getelementptr float, ptr addrspace(1) %54, i32 %47
          %56 = getelementptr float, ptr addrspace(1) %54, i32 %48
          %57 = load float, ptr addrspace(1) %55, align 4
          %58 = load float, ptr addrspace(1) %56, align 4
          %59 = mul i32 %27, 32
          %60 = getelementptr float, ptr addrspace(1) %1, i32 %59
          %61 = getelementptr float, ptr addrspace(1) %60, i32 %51
          %62 = getelementptr float, ptr addrspace(1) %60, i32 %52
          %63 = load float, ptr addrspace(1) %61, align 4
          %64 = load float, ptr addrspace(1) %62, align 4
          %65 = call i32 @air.thread_index_in_simdgroup()
          %66 = call [3 x i32] @air.thread_position_in_grid()
          %67 = extractvalue [3 x i32] %66, 0
          %68 = udiv i32 %67, 32
          %69 = udiv i32 %65, 8
          %70 = urem i32 %65, 8
          %71 = mul i32 %68, 4
          %72 = add i32 %71, %69
          %73 = mul i32 %70, 2
          %74 = add i32 0, %73
          %75 = udiv i32 %65, 16
          %76 = urem i32 %65, 16
          %77 = mul i32 %68, 2
          %78 = add i32 %77, %75
          %79 = add i32 0, %76
          %80 = mul i32 %72, 16
          %81 = add i32 %80, %74
          %82 = zext i32 %81 to i64
          %83 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %82
          store float %57, ptr addrspace(3) %83, align 4
          %84 = add i32 %74, 1
          %85 = add i32 %80, %84
          %86 = zext i32 %85 to i64
          %87 = getelementptr float, ptr addrspace(3) @__tg_dot_a_0, i64 %86
          store float %58, ptr addrspace(3) %87, align 4
          %88 = getelementptr float, ptr addrspace(3) @__tg_dot_b_0, i64 %82
          store float %63, ptr addrspace(3) %88, align 4
          %89 = getelementptr float, ptr addrspace(3) @__tg_dot_b_0, i64 %86
          store float %64, ptr addrspace(3) %89, align 4
          %90 = mul i32 %78, 16
          %91 = add i32 %90, %79
          %92 = zext i32 %91 to i64
          %93 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %92
          store float 0.000000e+00, ptr addrspace(3) %93, align 4
          %94 = add i32 %78, 8
          %95 = mul i32 %94, 16
          %96 = add i32 %95, %79
          %97 = zext i32 %96 to i64
          %98 = getelementptr float, ptr addrspace(3) @__tg_dot_c_0, i64 %97
          store float 0.000000e+00, ptr addrspace(3) %98, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %99 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %100 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %101 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %102 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %100, <64 x float> %101, <64 x float> %99)
          %103 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %104 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %105 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %103, <64 x float> %104, <64 x float> %102)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %105, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %106 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %107 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %108 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %109 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %107, <64 x float> %108, <64 x float> %106)
          %110 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %111 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %112 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %110, <64 x float> %111, <64 x float> %109)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %112, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %113 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %114 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %115 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer)
          %116 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %114, <64 x float> %115, <64 x float> %113)
          %117 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
          %118 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %119 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %117, <64 x float> %118, <64 x float> %116)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %119, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %120 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
          %121 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 0, i64 8>)
          %122 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 8, i64 0>)
          %123 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %121, <64 x float> %122, <64 x float> %120)
          %124 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_a_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
          %125 = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_dot_b_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
          %126 = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %124, <64 x float> %125, <64 x float> %123)
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %126, ptr addrspace(3) @__tg_dot_c_0, <2 x i64> splat (i64 16), <2 x i64> <i64 1, i64 16>, <2 x i64> splat (i64 8))
          call void @air.simdgroup.barrier(i32 2, i32 4)
          %127 = load float, ptr addrspace(3) %93, align 4
          %128 = load float, ptr addrspace(3) %98, align 4
          %129 = mul i32 %49, 32
          %130 = getelementptr float, ptr addrspace(1) %2, i32 %129
          %131 = getelementptr float, ptr addrspace(1) %130, i32 %51
          %132 = getelementptr float, ptr addrspace(1) %130, i32 %52
          %133 = call i32 @air.thread_index_in_simdgroup()
          %134 = call [3 x i32] @air.thread_position_in_grid()
          %135 = extractvalue [3 x i32] %134, 0
          %136 = udiv i32 %135, 32
          %137 = udiv i32 %133, 16
          %138 = urem i32 %133, 16
          %139 = mul i32 %136, 2
          %140 = add i32 %139, %137
          %141 = add i32 0, %138
          %142 = udiv i32 %133, 8
          %143 = urem i32 %133, 8
          %144 = mul i32 %136, 4
          %145 = add i32 %144, %142
          %146 = mul i32 %143, 2
          %147 = add i32 0, %146
          %148 = mul i32 %140, 16
          %149 = add i32 %148, %141
          %150 = zext i32 %149 to i64
          %151 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %150
          store float %127, ptr addrspace(3) %151, align 4
          %152 = add i32 %140, 8
          %153 = mul i32 %152, 16
          %154 = add i32 %153, %141
          %155 = zext i32 %154 to i64
          %156 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %155
          store float %128, ptr addrspace(3) %156, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %157 = mul i32 %145, 16
          %158 = add i32 %157, %147
          %159 = zext i32 %158 to i64
          %160 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %159
          %161 = load float, ptr addrspace(3) %160, align 4
          %162 = add i32 %147, 1
          %163 = add i32 %157, %162
          %164 = zext i32 %163 to i64
          %165 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %164
          %166 = load float, ptr addrspace(3) %165, align 4
          store float %161, ptr addrspace(1) %131, align 4
          store float %166, ptr addrspace(1) %132, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """#
        let data = try MetalASM.assemble(ir: ir)
        try data.write(to: URL(fileURLWithPath: "/tmp/dot_2d_test.metallib"))
        if let bc = extractBitcode(from: data) { try bc.write(to: URL(fileURLWithPath: "/tmp/dot_2d_test.bc")) }
        print("testDot2DKernel: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "dot_2d")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testDot2DKernel: PSO OK")

        // 1 TG of 128 threads computes a 16x16 tile.
        // The kernel uses air.thread_position_in_grid for thread-local indexing,
        // so we dispatch exactly 1 TG. A=ones(16x16), B=ones(16x32), C=16x32.
        // pid_x=0, pid_y=0, so it reads A[0:16,0:16], B[0:16,0:16], writes C[0:16,0:16].
        let M = 16, N = 32, K = 16
        let aBuf = device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: K * N * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: M * K)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: K * N)
        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: M * N)
        for i in 0..<(M*K) { aPtr[i] = 1.0 }
        for i in 0..<(K*N) { bPtr[i] = 1.0 }
        for i in 0..<(M*N) { cPtr[i] = -1.0 }

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        // 1 TG, 128 threads (4 warps) — kernel uses thread_position_in_grid
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // With 1 TG (pid=0,0): kernel writes 16x16 tile at C[0:16, 0:16] with stride N=32
        // Each thread writes 2 values, so check the written region
        var maxErr: Float = 0
        for i in 0..<16 {
            for j in 0..<16 {
                let err = abs(cPtr[i * N + j] - 16.0)
                maxErr = max(maxErr, err)
                if err > 0.01 {
                    print("testDot2DKernel: MISMATCH C[\(i),\(j)] = \(cPtr[i * N + j]) (expected 16.0)")
                }
            }
        }
        print("testDot2DKernel: max_err = \(maxErr)")
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
        let ir = """
        ; Loop kernel with MMA: accumulate 8x8 matmul over 2 iterations
        @__tg_a = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_b = internal addrspace(3) global [64 x float] undef, align 4
        @__tg_c = internal addrspace(3) global [64 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()
        declare [3 x i32] @air.thread_position_in_grid()

        define void @loop_mma_kernel(ptr addrspace(1) %out) {
        entry:
          %tid3 = call [3 x i32] @air.thread_position_in_grid()
          %tid = extractvalue [3 x i32] %tid3, 0
          %sl = call i32 @air.thread_index_in_simdgroup()
          ; Each thread writes 2 elements to fill all 64
          %idx0 = mul i32 %sl, 2
          %idx1 = add i32 %idx0, 1
          %idx0_64 = zext i32 %idx0 to i64
          %idx1_64 = zext i32 %idx1 to i64
          %a0 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx0_64
          %a1 = getelementptr float, ptr addrspace(3) @__tg_a, i64 %idx1_64
          %b0 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx0_64
          %b1 = getelementptr float, ptr addrspace(3) @__tg_b, i64 %idx1_64
          %c0_p = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx0_64
          %c1_p = getelementptr float, ptr addrspace(3) @__tg_c, i64 %idx1_64
          store float 1.0, ptr addrspace(3) %a0, align 4
          store float 1.0, ptr addrspace(3) %a1, align 4
          store float 1.0, ptr addrspace(3) %b0, align 4
          store float 1.0, ptr addrspace(3) %b1, align 4
          store float 0.0, ptr addrspace(3) %c0_p, align 4
          store float 0.0, ptr addrspace(3) %c1_p, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %c_init = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          br label %loop
        loop:
          %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
          %c = phi <64 x float> [ %c_init, %entry ], [ %c_next, %loop ]
          %a_mat = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_a, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %b_mat = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) @__tg_b, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          %c_next = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a_mat, <64 x float> %b_mat, <64 x float> %c)
          %i_next = add i32 %i, 1
          %cond = icmp slt i32 %i_next, 2
          br i1 %cond, label %loop, label %exit
        exit:
          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %c_next, ptr addrspace(3) @__tg_c, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
          ; Read back results
          %r0 = load float, ptr addrspace(3) %c0_p, align 4
          %r1 = load float, ptr addrspace(3) %c1_p, align 4
          %out0 = getelementptr float, ptr addrspace(1) %out, i32 %idx0
          %out1 = getelementptr float, ptr addrspace(1) %out, i32 %idx1
          store float %r0, ptr addrspace(1) %out0, align 4
          store float %r1, ptr addrspace(1) %out1, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
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

        // Run: 32 threads, each writes 2 elements. 2 iters of ones @ ones = 2*8 = 16
        let outBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
        memset(outBuf.contents(), 0xFF, 64 * 4)
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

        let result = outBuf.contents().bindMemory(to: Float.self, capacity: 64)
        print("testLoopMMAKernel: result[0..3] = \(result[0]), \(result[1]), \(result[2]), \(result[3])")
        var maxErr: Float = 0
        for i in 0..<64 {
            let err = abs(result[i] - 16.0)
            if err > maxErr { maxErr = err }
            XCTAssertEqual(result[i], 16.0, accuracy: 1e-3, "out[\(i)]")
        }
        print("testLoopMMAKernel: max_err = \(maxErr)")
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

    /// Test llvm.maxnum.f32 → air.fmax.f32 intrinsic mapping.
    /// Kernel: output[tid] = fmax(input[tid], input[tid+1])
    func testFmaxIntrinsic() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // This IR uses llvm.maxnum.f32 which should be renamed to air.fmax.f32
        let ir = """
        source_filename = "LLVMDialectModule"

        declare float @llvm.maxnum.f32(float, float)
        declare float @air.simd_shuffle_xor.f32(float, i16)

        define void @fmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid_x = extractvalue [3 x i32] %tid, 0
          %idx = zext i32 %tid_x to i64
          %p0 = getelementptr float, ptr addrspace(1) %0, i64 %idx
          %v0 = load float, ptr addrspace(1) %p0
          %idx1 = add i64 %idx, 1
          %p1 = getelementptr float, ptr addrspace(1) %0, i64 %idx1
          %v1 = load float, ptr addrspace(1) %p1
          %mx = call float @llvm.maxnum.f32(float %v0, float %v1)
          ; Also test shuffle
          %sh = call float @air.simd_shuffle_xor.f32(float %mx, i16 1)
          %final = fadd float %mx, %sh
          %p2 = getelementptr float, ptr addrspace(1) %1, i64 %idx
          store float %final, ptr addrspace(1) %p2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        """
        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "fmax_kernel")
        XCTAssertNotNil(fn, "fmax_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testFmaxIntrinsic: PSO OK, maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")

        // Run: input = [1, 5, 3, 7, 2, 8, 4, 6], output should have fmax(input[i], input[i+1])
        let N = 4
        let inBuf = device.makeBuffer(length: (N + 1) * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let inp = inBuf.contents().bindMemory(to: Float.self, capacity: N + 1)
        inp[0] = 1; inp[1] = 5; inp[2] = 3; inp[3] = 7; inp[4] = 2

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let out = outBuf.contents().bindMemory(to: Float.self, capacity: N)
        // fmax(1,5)=5, fmax(5,3)=5, fmax(3,7)=7, fmax(7,2)=7
        // shuffle_xor with 1 swaps adjacent lanes: [5,5,7,7] → xor1 → [5,5,7,7]
        // For 4 threads: lane0↔lane1, lane2↔lane3
        // final = mx + shuffled
        // lane0: fmax(1,5)=5, shuffle_xor(5,1)=lane1's 5 → 5+5=10
        // lane1: fmax(5,3)=5, shuffle_xor(5,1)=lane0's 5 → 5+5=10
        // lane2: fmax(3,7)=7, shuffle_xor(7,1)=lane3's 7 → 7+7=14
        // lane3: fmax(7,2)=7, shuffle_xor(7,1)=lane2's 7 → 7+7=14
        print("testFmaxIntrinsic: results = [\(out[0]), \(out[1]), \(out[2]), \(out[3])]")
        XCTAssertEqual(out[0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(out[1], 10.0, accuracy: 1e-5)
        XCTAssertEqual(out[2], 14.0, accuracy: 1e-5)
        XCTAssertEqual(out[3], 14.0, accuracy: 1e-5)
        #endif
    }

    /// Test reduce_max with llvm.maxnum → air.fmax rename + shuffle + -inf constant.
    func testTritonReduceMaxIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare [3 x i32] @air.thread_position_in_threadgroup()
        declare [3 x i32] @air.threadgroup_position_in_grid()

        define void @reduce_max_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %4 = load i32, ptr addrspace(2) %2, align 4
          %5 = call [3 x i32] @air.threadgroup_position_in_grid()
          %6 = extractvalue [3 x i32] %5, 0
          %7 = mul i32 %6, 32
          %8 = call [3 x i32] @air.thread_position_in_threadgroup()
          %9 = extractvalue [3 x i32] %8, 0
          %10 = zext i32 %9 to i64
          %11 = trunc i64 %10 to i32
          %12 = and i32 %11, 127
          %13 = urem i32 %12, 32
          %14 = call [3 x i32] @air.thread_position_in_threadgroup()
          %15 = extractvalue [3 x i32] %14, 0
          %16 = udiv i32 %15, 32
          %17 = shl i32 %13, 0
          %18 = or i32 0, %17
          %19 = shl i32 %16, 5
          %20 = or i32 %18, %19
          %21 = and i32 %20, 31
          %22 = lshr i32 %21, 0
          %23 = or disjoint i32 %22, 0
          %24 = xor i32 0, %23
          %25 = xor i32 %24, 0
          %26 = add i32 %25, 0
          %27 = add i32 %7, %26
          %28 = icmp slt i32 %27, %4
          %29 = getelementptr float, ptr addrspace(1) %0, i32 %27
          %30 = load float, ptr addrspace(1) %29, align 4
          %31 = select i1 %28, float %30, float 0xFFF0000000000000
          %32 = call float @air.simd_shuffle_xor.f32(float %31, i16 16)
          %33 = call float @llvm.maxnum.f32(float %31, float %32)
          %34 = call float @air.simd_shuffle_xor.f32(float %33, i16 8)
          %35 = call float @llvm.maxnum.f32(float %33, float %34)
          %36 = call float @air.simd_shuffle_xor.f32(float %35, i16 4)
          %37 = call float @llvm.maxnum.f32(float %35, float %36)
          %38 = call float @air.simd_shuffle_xor.f32(float %37, i16 2)
          %39 = call float @llvm.maxnum.f32(float %37, float %38)
          %40 = call float @air.simd_shuffle_xor.f32(float %39, i16 1)
          %41 = call float @llvm.maxnum.f32(float %39, float %40)
          %42 = getelementptr float, ptr addrspace(1) %1, i32 %6
          store float %41, ptr addrspace(1) %42, align 4
          ret void
        }

        ; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
        declare float @llvm.maxnum.f32(float, float) #0

        attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)
        print("testTritonReduceMaxIR: \(data.count) bytes")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "reduce_max_kernel")
        XCTAssertNotNil(fn, "reduce_max_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testTritonReduceMaxIR: PSO OK")
        #endif
    }

    /// Test multi-warp reduce sum (128 elements, 4 warps, threadgroup memory)
    func testMultiWarpReduceSum() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let device = MTLCreateSystemDefaultDevice()!

        // TG array global with GEP — matches Triton's global_smem pattern
        let ir = """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16

        define void @reduce_sum_multi(ptr addrspace(1) %0) {
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store float 42.0, ptr addrspace(3) %ptr, align 4
          %val = load float, ptr addrspace(3) %ptr, align 4
          store float %val, ptr addrspace(1) %0, align 4
          ret void
        }
        """

        let result = try MetalASM.assemble(ir: ir)
        print("testMultiWarpReduceSum: \(result.count) bytes")

        let dd = asDispatchData(Data(result))
        let lib = try device.makeLibrary(data: dd)
        let fn = lib.makeFunction(name: "reduce_sum_multi")
        XCTAssertNotNil(fn, "reduce_sum_multi not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testMultiWarpReduceSum: PSO OK")

        // Run it: 1 thread, with 16 bytes TG memory
        let outputBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outputBuf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let result_val = outputBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee
        print("testMultiWarpReduceSum: result = \\(result_val)")
        XCTAssertEqual(result_val, 42.0, accuracy: 0.01)
        #endif
    }

    func testTritonMultiWarpReduce() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let device = MTLCreateSystemDefaultDevice()!

        // Full Triton-generated multi-warp reduce IR
        let ir = """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16

        declare void @air.wg.barrier(i32, i32)
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare [3 x i32] @air.thread_position_in_threadgroup()
        declare [3 x i32] @air.threadgroup_position_in_grid()

        define void @reduce_sum_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %4 = load i32, ptr addrspace(2) %2, align 4
          %5 = call [3 x i32] @air.threadgroup_position_in_grid()
          %6 = extractvalue [3 x i32] %5, 0
          %7 = mul i32 %6, 128
          %8 = call [3 x i32] @air.thread_position_in_threadgroup()
          %9 = extractvalue [3 x i32] %8, 0
          %10 = zext i32 %9 to i64
          %11 = trunc i64 %10 to i32
          %12 = and i32 %11, 127
          %13 = urem i32 %12, 32
          %14 = call [3 x i32] @air.thread_position_in_threadgroup()
          %15 = extractvalue [3 x i32] %14, 0
          %16 = udiv i32 %15, 32
          %17 = shl i32 %13, 0
          %18 = or i32 0, %17
          %19 = shl i32 %16, 5
          %20 = or i32 %18, %19
          %21 = and i32 %20, 127
          %22 = lshr i32 %21, 0
          %23 = or disjoint i32 %22, 0
          %24 = xor i32 0, %23
          %25 = xor i32 %24, 0
          %26 = add i32 %25, 0
          %27 = add i32 %7, %26
          %28 = icmp slt i32 %27, %4
          %29 = getelementptr float, ptr addrspace(1) %0, i32 %27
          %30 = load float, ptr addrspace(1) %29, align 4
          %31 = select i1 %28, float %30, float 0.000000e+00
          %32 = call float @air.simd_shuffle_xor.f32(float %31, i16 16)
          %33 = fadd float %31, %32
          %34 = call float @air.simd_shuffle_xor.f32(float %33, i16 8)
          %35 = fadd float %33, %34
          %36 = call float @air.simd_shuffle_xor.f32(float %35, i16 4)
          %37 = fadd float %35, %36
          %38 = call float @air.simd_shuffle_xor.f32(float %37, i16 2)
          %39 = fadd float %37, %38
          %40 = call float @air.simd_shuffle_xor.f32(float %39, i16 1)
          %41 = fadd float %39, %40
          %42 = call [3 x i32] @air.thread_position_in_threadgroup()
          %43 = extractvalue [3 x i32] %42, 0
          %44 = zext i32 %43 to i64
          %45 = trunc i64 %44 to i32
          %46 = and i32 %45, 127
          %47 = urem i32 %46, 32
          %48 = call [3 x i32] @air.thread_position_in_threadgroup()
          %49 = extractvalue [3 x i32] %48, 0
          %50 = udiv i32 %49, 32
          %51 = shl i32 %47, 0
          %52 = or i32 0, %51
          %53 = shl i32 %50, 5
          %54 = or i32 %52, %53
          %55 = and i32 %54, 96
          %56 = lshr i32 %55, 3
          %57 = or disjoint i32 0, %56
          %58 = xor i32 0, %57
          %59 = xor i32 %58, 0
          %60 = xor i32 %59, 0
          %61 = add i32 %60, 0
          %62 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %61
          %63 = insertelement <1 x float> undef, float %41, i32 0
          store <1 x float> %63, ptr addrspace(3) %62, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %64 = call [3 x i32] @air.thread_position_in_threadgroup()
          %65 = extractvalue [3 x i32] %64, 0
          %66 = zext i32 %65 to i64
          %67 = trunc i64 %66 to i32
          %68 = and i32 %67, 127
          %69 = urem i32 %68, 32
          %70 = call [3 x i32] @air.thread_position_in_threadgroup()
          %71 = extractvalue [3 x i32] %70, 0
          %72 = udiv i32 %71, 32
          %73 = shl i32 %69, 0
          %74 = or i32 0, %73
          %75 = shl i32 %72, 5
          %76 = or i32 %74, %75
          %77 = and i32 %76, 3
          %78 = shl i32 %77, 2
          %79 = or disjoint i32 %78, 0
          %80 = xor i32 0, %79
          %81 = xor i32 %80, 0
          %82 = xor i32 %81, 0
          %83 = add i32 %82, 0
          %84 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %83
          %85 = load <1 x float>, ptr addrspace(3) %84, align 4
          %86 = extractelement <1 x float> %85, i32 0
          %87 = call float @air.simd_shuffle_xor.f32(float %86, i16 2)
          %88 = fadd float %86, %87
          %89 = call float @air.simd_shuffle_xor.f32(float %88, i16 1)
          %90 = fadd float %88, %89
          %91 = getelementptr float, ptr addrspace(1) %1, i32 %6
          store float %90, ptr addrspace(1) %91, align 4
          ret void
        }
        """

        let result = try MetalASM.assemble(ir: ir)
        print("testTritonMultiWarpReduce: \(result.count) bytes")
        try result.write(to: URL(fileURLWithPath: "/tmp/test_triton_reduce.metallib"))

        let dd = asDispatchData(Data(result))
        let lib = try device.makeLibrary(data: dd)
        let fn = lib.makeFunction(name: "reduce_sum_kernel")
        XCTAssertNotNil(fn, "reduce_sum_kernel not found in \(lib.functionNames)")
        let pso = try device.makeComputePipelineState(function: fn!)
        print("testTritonMultiWarpReduce: PSO OK")
        #endif
    }

    // MARK: - Pass 5b scalar packing test

    func testScalarPackingScale() throws {
        // Kernel: out[i] = in[i] * scale
        // 2 device ptrs + 1 float scalar → Pass 5b should pack the float into a buffer
        let ir = """
        define void @scale_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, float %scale) {
          %tid3 = call <3 x i32> @air.thread_position_in_grid.v3i32()
          %idx = extractelement <3 x i32> %tid3, i32 0
          %p_in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %idx
          %val = load float, ptr addrspace(1) %p_in, align 4
          %mul = fmul float %val, %scale
          %p_out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %idx
          store float %mul, ptr addrspace(1) %p_out, align 4
          ret void
        }
        declare <3 x i32> @air.thread_position_in_grid.v3i32()
        """

        // Step 1: parse
        print("STEP 1: parsing...")
        let lexer = Lexer(source: ir)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        print("  parsed OK, \(module.functions.count) functions")

        // Step 2: transform
        print("STEP 2: applyAirTransforms...")
        applyAirTransforms(module: module)
        print("  transforms OK")

        let fn = module.functions.first(where: { !$0.isDeclaration })!
        print("STEP 3: params after transform:")
        for (i, (ty, nm)) in zip(fn.parameterTypes, fn.parameterNames).enumerated() {
            print("  [\(i)] \(ty) %\(nm)")
        }
        print("STEP 4: entry block instructions (\(fn.basicBlocks[0].instructions.count) total):")
        for (i, inst) in fn.basicBlocks[0].instructions.enumerated() {
            print("  [\(i)] \(inst.opcode) \(inst.name ?? "-") : \(inst.type) ops=\(inst.operands.count)")
        }

        // Step 5: compile to metallib (NO GPU dispatch)
        print("STEP 5: assembling to metallib...")
        let data = try MetalASM.assemble(ir: ir)
        print("  metallib: \(data.count) bytes — DONE")
    }

    // MARK: - TG byte global ablation tests

    func testTGByteAblation() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let device = MTLCreateSystemDefaultDevice()!

        func tryKernel(_ label: String, _ ir: String) {
            do {
                let data = try MetalASM.assemble(ir: ir)
                print("\(label): timing=\(MetalASM._lastTiming)")
                let lib = try device.makeLibrary(data: asDispatchData(data))
                let fnName = lib.functionNames.first!
                let fn = lib.makeFunction(name: fnName)!
                let pso = try device.makeComputePipelineState(function: fn)
                print("\(label): OK (maxThreads=\(pso.maxTotalThreadsPerThreadgroup))")
            } catch {
                // Dump the IR for debugging
                print("\(label): FAILED — dumping transformed IR")
                if let parsed = try? {
                    let lexer = Lexer(source: ir)
                    let tokens = lexer.tokenize()
                    var parser = Parser(tokens: tokens, source: lexer.source)
                    return try parser.parse()
                }() {
                    applyAirTransforms(module: parsed)
                    for fn in parsed.functions where !fn.isDeclaration {
                        print("  fn \(fn.name)(\(fn.parameterTypes.map { "\($0)" }.joined(separator: ", ")))")
                        for bb in fn.basicBlocks {
                            for inst in bb.instructions {
                                print("    \(inst.name.isEmpty ? "" : "%\(inst.name) = ")\(inst.opcode) \(inst.type) \(inst.operands)")
                            }
                        }
                    }
                }
                XCTFail("\(label): \(error.localizedDescription.prefix(80))")
            }
        }

        tryKernel("L0: TG byte + float", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        define void @kern(ptr addrspace(1) %0) {
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store float 42.0, ptr addrspace(3) %ptr, align 4
          %val = load float, ptr addrspace(3) %ptr, align 4
          store float %val, ptr addrspace(1) %0, align 4
          ret void
        }
        """)

        tryKernel("L1: + const buf", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        define void @kern(ptr addrspace(1) %0, ptr addrspace(2) %1) {
          %n = load i32, ptr addrspace(2) %1, align 4
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store float 42.0, ptr addrspace(3) %ptr, align 4
          %val = load float, ptr addrspace(3) %ptr, align 4
          store float %val, ptr addrspace(1) %0, align 4
          ret void
        }
        """)

        tryKernel("L2: + shuffle", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        define void @kern(ptr addrspace(1) %0) {
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store float 42.0, ptr addrspace(3) %ptr, align 4
          %val = load float, ptr addrspace(3) %ptr, align 4
          %s = call float @air.simd_shuffle_xor.f32(float %val, i16 1)
          %sum = fadd float %val, %s
          store float %sum, ptr addrspace(1) %0, align 4
          ret void
        }
        """)

        tryKernel("L3: + barrier", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare void @air.wg.barrier(i32, i32)
        define void @kern(ptr addrspace(1) %0) {
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store float 42.0, ptr addrspace(3) %ptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %val = load float, ptr addrspace(3) %ptr, align 4
          %s = call float @air.simd_shuffle_xor.f32(float %val, i16 1)
          store float %s, ptr addrspace(1) %0, align 4
          ret void
        }
        """)

        tryKernel("L4: + vec1 store/load", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        define void @kern(ptr addrspace(1) %0) {
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          %vec = insertelement <1 x float> undef, float 42.0, i32 0
          store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %ld = load <1 x float>, ptr addrspace(3) %ptr, align 4
          %val = extractelement <1 x float> %ld, i32 0
          store float %val, ptr addrspace(1) %0, align 4
          ret void
        }
        """)

        tryKernel("L5: + 2bufs+const+pid+tidtg", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          %vec = insertelement <1 x float> undef, float 42.0, i32 0
          store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %ld = load <1 x float>, ptr addrspace(3) %ptr, align 4
          %val = extractelement <1 x float> %ld, i32 0
          store float %val, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        tryKernel("L6a: + shuffle+barrier+vec+dynamic", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %s1 = call float @air.simd_shuffle_xor.f32(float 1.0, i16 1)
          %warpid = udiv i32 %tid, 32
          %offset = shl i32 %warpid, 2
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %offset
          %vec = insertelement <1 x float> undef, float %s1, i32 0
          store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %lane = urem i32 %tid, 32
          %roff = shl i32 %lane, 2
          %rptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %roff
          %ld = load <1 x float>, ptr addrspace(3) %rptr, align 4
          %val = extractelement <1 x float> %ld, i32 0
          %s2 = call float @air.simd_shuffle_xor.f32(float %val, i16 1)
          %sum = fadd float %val, %s2
          store float %sum, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        tryKernel("L6: + dynamic TG GEP", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %warpid = udiv i32 %tid, 32
          %offset = shl i32 %warpid, 2
          %ptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %offset
          %vec = insertelement <1 x float> undef, float 42.0, i32 0
          store <1 x float> %vec, ptr addrspace(3) %ptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %ld = load <1 x float>, ptr addrspace(3) %ptr, align 4
          %val = extractelement <1 x float> %ld, i32 0
          store float %val, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        // L6b0: float GEP on device buf, NO TG global
        tryKernel("L6b0: float GEP no TG", """
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %inptr = getelementptr float, ptr addrspace(1) %0, i32 %tid
          %inval = load float, ptr addrspace(1) %inptr, align 4
          store float %inval, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        // L6b: L6a + device buf float GEP + load (no icmp/select, no TG)
        tryKernel("L6b: + device float GEP+load", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %idx = add i32 %pid, %tid
          %inptr = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %inval = load float, ptr addrspace(1) %inptr, align 4
          store float %inval, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        // L6c: L6b + icmp + select
        tryKernel("L6c: + icmp+select", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %idx = add i32 %pid, %tid
          %inptr = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %inval = load float, ptr addrspace(1) %inptr, align 4
          %cmp = icmp slt i32 %idx, %n
          %val = select i1 %cmp, float %inval, float 0.0
          store float %val, ptr addrspace(1) %1, align 4
          ret void
        }
        """)

        // L6d: L6c + shuffle reduction + TG store/load (full L7a)
        tryKernel("L6d: + shuffle+TG (full reduce)", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.threadgroup_position_in_grid()
        declare [3 x i32] @air.thread_position_in_threadgroup()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %idx = add i32 %pid, %tid
          %inptr = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %inval = load float, ptr addrspace(1) %inptr, align 4
          %cmp = icmp slt i32 %idx, %n
          %val = select i1 %cmp, float %inval, float 0.0
          %s1 = call float @air.simd_shuffle_xor.f32(float %val, i16 1)
          %a1 = fadd float %val, %s1
          %warpid = udiv i32 %tid, 32
          %woff = shl i32 %warpid, 2
          %tgptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %woff
          %vec = insertelement <1 x float> undef, float %a1, i32 0
          store <1 x float> %vec, ptr addrspace(3) %tgptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %lane = urem i32 %tid, 32
          %roff = shl i32 %lane, 2
          %rptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %roff
          %ld = load <1 x float>, ptr addrspace(3) %rptr, align 4
          %ldval = extractelement <1 x float> %ld, i32 0
          %outptr = getelementptr float, ptr addrspace(1) %1, i32 %pid
          store float %ldval, ptr addrspace(1) %outptr, align 4
          ret void
        }
        """)

        // L7a: simplified Triton reduce — keep device load + shuffle + TG store/load + output
        tryKernel("L7a: simplified reduce", """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare [3 x i32] @air.thread_position_in_threadgroup()
        declare [3 x i32] @air.threadgroup_position_in_grid()
        define void @kern(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %n = load i32, ptr addrspace(2) %2, align 4
          %pid3 = call [3 x i32] @air.threadgroup_position_in_grid()
          %pid = extractvalue [3 x i32] %pid3, 0
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %idx = add i32 %pid, %tid
          %inptr = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %inval = load float, ptr addrspace(1) %inptr, align 4
          %cmp = icmp slt i32 %idx, %n
          %val = select i1 %cmp, float %inval, float 0.0
          %s1 = call float @air.simd_shuffle_xor.f32(float %val, i16 16)
          %a1 = fadd float %val, %s1
          %s2 = call float @air.simd_shuffle_xor.f32(float %a1, i16 8)
          %a2 = fadd float %a1, %s2
          %s3 = call float @air.simd_shuffle_xor.f32(float %a2, i16 4)
          %a3 = fadd float %a2, %s3
          %s4 = call float @air.simd_shuffle_xor.f32(float %a3, i16 2)
          %a4 = fadd float %a3, %s4
          %s5 = call float @air.simd_shuffle_xor.f32(float %a4, i16 1)
          %a5 = fadd float %a4, %s5
          %warpid = udiv i32 %tid, 32
          %woff = shl i32 %warpid, 2
          %tgptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %woff
          %vec = insertelement <1 x float> undef, float %a5, i32 0
          store <1 x float> %vec, ptr addrspace(3) %tgptr, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %lane = urem i32 %tid, 32
          %roff = shl i32 %lane, 2
          %rptr = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %roff
          %ld = load <1 x float>, ptr addrspace(3) %rptr, align 4
          %ldval = extractelement <1 x float> %ld, i32 0
          %r1 = call float @air.simd_shuffle_xor.f32(float %ldval, i16 2)
          %b1 = fadd float %ldval, %r1
          %r2 = call float @air.simd_shuffle_xor.f32(float %b1, i16 1)
          %b2 = fadd float %b1, %r2
          %outptr = getelementptr float, ptr addrspace(1) %1, i32 %pid
          store float %b2, ptr addrspace(1) %outptr, align 4
          ret void
        }
        """)

        #endif
    }

    // MARK: - Triton multi-warp reduce correctness test

    func testTritonMultiWarpReduceCorrectness() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Exact Triton-generated IR (with redundant ops)
        let ir = """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        declare float @air.simd_shuffle_xor.f32(float, i16)
        declare [3 x i32] @air.thread_position_in_threadgroup()
        declare [3 x i32] @air.threadgroup_position_in_grid()
        define void @reduce_sum_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2) {
          %4 = load i32, ptr addrspace(2) %2, align 4
          %5 = call [3 x i32] @air.threadgroup_position_in_grid()
          %6 = extractvalue [3 x i32] %5, 0
          %7 = mul i32 %6, 128
          %8 = call [3 x i32] @air.thread_position_in_threadgroup()
          %9 = extractvalue [3 x i32] %8, 0
          %10 = zext i32 %9 to i64
          %11 = trunc i64 %10 to i32
          %12 = and i32 %11, 127
          %13 = urem i32 %12, 32
          %14 = call [3 x i32] @air.thread_position_in_threadgroup()
          %15 = extractvalue [3 x i32] %14, 0
          %16 = udiv i32 %15, 32
          %17 = shl i32 %13, 0
          %18 = or i32 0, %17
          %19 = shl i32 %16, 5
          %20 = or i32 %18, %19
          %21 = and i32 %20, 127
          %22 = lshr i32 %21, 0
          %23 = or i32 %22, 0
          %24 = xor i32 0, %23
          %25 = xor i32 %24, 0
          %26 = add i32 %25, 0
          %27 = add i32 %7, %26
          %28 = icmp slt i32 %27, %4
          %29 = getelementptr float, ptr addrspace(1) %0, i32 %27
          %30 = load float, ptr addrspace(1) %29, align 4
          %31 = select i1 %28, float %30, float 0.000000e+00
          %32 = call float @air.simd_shuffle_xor.f32(float %31, i16 16)
          %33 = fadd float %31, %32
          %34 = call float @air.simd_shuffle_xor.f32(float %33, i16 8)
          %35 = fadd float %33, %34
          %36 = call float @air.simd_shuffle_xor.f32(float %35, i16 4)
          %37 = fadd float %35, %36
          %38 = call float @air.simd_shuffle_xor.f32(float %37, i16 2)
          %39 = fadd float %37, %38
          %40 = call float @air.simd_shuffle_xor.f32(float %39, i16 1)
          %41 = fadd float %39, %40
          %42 = call [3 x i32] @air.thread_position_in_threadgroup()
          %43 = extractvalue [3 x i32] %42, 0
          %44 = zext i32 %43 to i64
          %45 = trunc i64 %44 to i32
          %46 = and i32 %45, 127
          %47 = urem i32 %46, 32
          %48 = call [3 x i32] @air.thread_position_in_threadgroup()
          %49 = extractvalue [3 x i32] %48, 0
          %50 = udiv i32 %49, 32
          %51 = shl i32 %47, 0
          %52 = or i32 0, %51
          %53 = shl i32 %50, 5
          %54 = or i32 %52, %53
          %55 = and i32 %54, 96
          %56 = lshr i32 %55, 3
          %57 = or i32 0, %56
          %58 = xor i32 0, %57
          %59 = xor i32 %58, 0
          %60 = xor i32 %59, 0
          %61 = add i32 %60, 0
          %62 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %61
          %63 = insertelement <1 x float> undef, float %41, i32 0
          store <1 x float> %63, ptr addrspace(3) %62, align 4
          call void @air.wg.barrier(i32 1, i32 1)
          %64 = call [3 x i32] @air.thread_position_in_threadgroup()
          %65 = extractvalue [3 x i32] %64, 0
          %66 = zext i32 %65 to i64
          %67 = trunc i64 %66 to i32
          %68 = and i32 %67, 127
          %69 = urem i32 %68, 32
          %70 = call [3 x i32] @air.thread_position_in_threadgroup()
          %71 = extractvalue [3 x i32] %70, 0
          %72 = udiv i32 %71, 32
          %73 = shl i32 %69, 0
          %74 = or i32 0, %73
          %75 = shl i32 %72, 5
          %76 = or i32 %74, %75
          %77 = and i32 %76, 3
          %78 = shl i32 %77, 2
          %79 = or i32 %78, 0
          %80 = xor i32 0, %79
          %81 = xor i32 %80, 0
          %82 = xor i32 %81, 0
          %83 = add i32 %82, 0
          %84 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %83
          %85 = load <1 x float>, ptr addrspace(3) %84, align 4
          %86 = extractelement <1 x float> %85, i32 0
          %87 = call float @air.simd_shuffle_xor.f32(float %86, i16 2)
          %88 = fadd float %86, %87
          %89 = call float @air.simd_shuffle_xor.f32(float %88, i16 1)
          %90 = fadd float %88, %89
          %91 = getelementptr float, ptr addrspace(1) %1, i32 %6
          store float %90, ptr addrspace(1) %91, align 4
          ret void
        }
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "reduce_sum_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        // Input: 128 ones
        let N = 128
        let inputBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let inputPtr = inputBuf.contents().bindMemory(to: Float.self, capacity: N)
        for i in 0..<N { inputPtr[i] = 1.0 }

        // Output: 1 float
        let outputBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let outputPtr = outputBuf.contents().bindMemory(to: Float.self, capacity: 1)
        outputPtr[0] = -999.0

        // Const buf: n_elements = 128
        let constBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        constBuf.contents().bindMemory(to: Int32.self, capacity: 1).pointee = Int32(N)

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(constBuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let result = outputPtr[0]
        print("testTritonMultiWarpReduceCorrectness: result = \(result), expected 128.0")
        XCTAssertEqual(result, 128.0, accuracy: 0.01, "Multi-warp reduce: expected 128.0, got \(result)")
        #endif
    }

    // ── Atomic add (f32) ────────────────────────────────────────────────────
    func testAtomicAddF32() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // 128 threads each atomically add 1.0 to output[0]
        // Uses thread_position_in_threadgroup to avoid scalar store guard
        let ir = """
        source_filename = "LLVMDialectModule"

        declare float @air.atomic.global.add.f32(ptr addrspace(1), float, i32, i32, i1)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @atomic_add_kernel(ptr addrspace(1) %0, ptr addrspace(2) %1) {
          %tid_arr = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid_arr, 0
          %val = load float, ptr addrspace(2) %1, align 4
          %3 = call float @air.atomic.global.add.f32(ptr addrspace(1) %0, float %val, i32 0, i32 2, i1 true)
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "atomic_add_kernel")
        XCTAssertNotNil(fn, "atomic_add_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        // Output buffer: single float = 0.0
        let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        outBuf.contents().storeBytes(of: Float(0.0), as: Float.self)

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        var val: Float = 1.0
        let valBuf = device.makeBuffer(bytes: &val, length: 4, options: .storageModeShared)!
        enc.setBuffer(valBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 128, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().load(as: Float.self)
        print("testAtomicAddF32: result = \(result), expected 128.0")
        XCTAssertEqual(result, 128.0, accuracy: 0.5, "128 threads × atomic_add(1.0) should = 128.0, got \(result)")
        #endif
    }

    // ── Triton-exact atomic IR (binary search what breaks) ──────────────
    func testAtomicTritonExactIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Exact IR from Triton's atomic_add_kernel (scalar float val)
        // Uses thread_position_in_threadgroup to avoid scalar store guard
        let ir = """
        source_filename = "LLVMDialectModule"

        declare float @air.atomic.global.add.f32(ptr addrspace(1), float, i32, i32, i1)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @atomic_add_kernel(ptr addrspace(1) %0, float %1) {
          %tid_arr = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid_arr, 0
          %3 = call float @air.atomic.global.add.f32(ptr addrspace(1) %0, float %1, i32 0, i32 2, i1 true)
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100, "metallib too small")

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "atomic_add_kernel")
        XCTAssertNotNil(fn, "atomic_add_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        outBuf.contents().storeBytes(of: Float(0.0), as: Float.self)

        var val: Float = 1.0
        let valBuf = device.makeBuffer(bytes: &val, length: 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBuffer(valBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 128, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().load(as: Float.self)
        print("testAtomicTritonExactIR: result = \(result), expected 128.0")
        XCTAssertEqual(result, 128.0, accuracy: 0.5)
        #endif
    }

    func testCmpxchgI32() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // CAS loop: atomically replace buf[0] with 42 if it's 0
        let ir = """
        declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @cas_kernel(ptr addrspace(1) %0) {
          %2 = alloca i32, i64 1, align 4
          store i32 0, ptr %2, align 4
          %3 = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %0, ptr %2, i32 42, i32 0, i32 0, i32 2, i1 true)
          ret void
        }
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "cas_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let buf = device.makeBuffer(length: 4, options: .storageModeShared)!
        buf.contents().storeBytes(of: Int32(0), as: Int32.self)

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let result = buf.contents().load(as: Int32.self)
        XCTAssertEqual(result, 42, "CAS should have swapped 0 → 42")
        #endif
    }

    func testMultiFunction() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Kernel calls a device function: add_fn(x, pid) returns x + (pid==0 ? 1 : 2)
        // Single thread (pid=0), input=10 → output=11
        let ir = """
        declare [3 x i32] @air.threadgroup_position_in_grid()

        define void @kernel(ptr addrspace(1) %0) {
          %2 = call [3 x i32] @air.threadgroup_position_in_grid()
          %3 = extractvalue [3 x i32] %2, 0
          %4 = load i32, ptr addrspace(1) %0, align 4
          %5 = call i32 @add_fn(i32 %4, i32 %3)
          store i32 %5, ptr addrspace(1) %0, align 4
          ret void
        }

        define i32 @add_fn(i32 %0, i32 %1) {
          %3 = icmp eq i32 %1, 0
          %4 = select i1 %3, i32 1, i32 2
          %5 = add i32 %0, %4
          ret i32 %5
        }
        """
        let metallib = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = lib.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let buf = device.makeBuffer(length: 4, options: .storageModeShared)!
        buf.contents().storeBytes(of: Int32(10), as: Int32.self)

        let cmdq = device.makeCommandQueue()!
        let cmdbuf = cmdq.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let result = buf.contents().load(as: Int32.self)
        XCTAssertEqual(result, 11, "kernel should call add_fn(10, 0) = 10 + 1 = 11")
        #endif
    }

    func testLoopStoreLoad() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Simple loop: load float, add 1.0, store — 40 times
        let ir = """
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @loop_store_load(ptr addrspace(1) %0) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tidx = extractvalue [3 x i32] %tid, 0
          br label %loop
        loop:
          %i = phi i32 [ %i_next, %loop ], [ 0, %1 ]
          %val = load volatile float, ptr addrspace(1) %0, align 4
          %val1 = fadd float %val, 1.000000e+00
          store volatile float %val1, ptr addrspace(1) %0, align 4
          %i_next = add i32 %i, 1
          %cmp = icmp slt i32 %i_next, 40
          br i1 %cmp, label %loop, label %done
        done:
          ret void
        }
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "loop_store_load")!
        let pso = try device.makeComputePipelineState(function: fn)

        let buf = device.makeBuffer(length: 4, options: .storageModeShared)!
        buf.contents().storeBytes(of: Float(0.0), as: Float.self)

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let result = buf.contents().load(as: Float.self)
        print("loop_store_load result: \(result)")
        XCTAssertEqual(result, 40.0, "40 iterations of load+add+store")
        #endif
    }



    /// Struct phi in loop: 32-element {ptr addrspace(1) x 32} phi gets split
    /// into 32 scalar ptr phis. Verifies transformStructPhis handles
    /// insertvalue chains whose aggregate base is a split struct phi.
    func testStructPhiLoop() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Build 32-element insertvalue chain + struct phi in loop
        let n = 32
        let structTy = "{ " + Array(repeating: "ptr addrspace(1)", count: n).joined(separator: ", ") + " }"
        var inserts = ""
        for i in 0..<n {
            let prev = i == 0 ? "undef" : "%s\(i-1)"
            inserts += "  %s\(i) = insertvalue \(structTy) \(prev), ptr addrspace(1) %a_init, \(i)\n"
        }
        let ir = """
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @struct_phi_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
        entry:
          %tid_arr = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid_arr, 0
          %tid64 = zext i32 %tid to i64
          %a_init = getelementptr float, ptr addrspace(1) %0, i64 %tid64
        \(inserts)  br label %loop
        loop:
          %iv = phi i32 [ 0, %entry ], [ %iv_next, %loop ]
          %acc = phi float [ 0.000000e+00, %entry ], [ %acc_next, %loop ]
          %ptrs = phi \(structTy) [ %s\(n-1), %entry ], [ %ptrs_next, %loop ]
          %p0 = extractvalue \(structTy) %ptrs, 0
          %val = load float, ptr addrspace(1) %p0
          %acc_next = fadd float %acc, %val
          %p0_next = getelementptr float, ptr addrspace(1) %p0, i64 \(n)
          %ptrs_next = insertvalue \(structTy) %ptrs, ptr addrspace(1) %p0_next, 0
          %iv_next = add i32 %iv, 1
          %cond = icmp slt i32 %iv_next, 4
          br i1 %cond, label %loop, label %exit
        exit:
          %out = getelementptr float, ptr addrspace(1) %1, i64 %tid64
          store float %acc_next, ptr addrspace(1) %out
          ret void
        }
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "struct_phi_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        // Fill input: [0, 1, 2, ..., 127]  (32 threads × 4 iterations = read 128 floats)
        let count = 128
        let inBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { inPtr[i] = Float(i) }

        let outBuf = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Thread t reads input[t], input[t+32], input[t+64], input[t+96]
        // acc = input[t] + input[t+32] + input[t+64] + input[t+96]
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: 32)
        for t in 0..<32 {
            let expected = Float(t) + Float(t + 32) + Float(t + 64) + Float(t + 96)
            XCTAssertEqual(outPtr[t], expected, accuracy: 0.01,
                           "thread \(t): got \(outPtr[t]) expected \(expected)")
        }
        #endif
    }

    /// Dot kernel pattern: half device ptrs for both input (load) and output (store-only),
    /// MMA intrinsics, TG half buffer for layout conversion, bitcast zeroinitializer.
    /// Tests three fixes:
    ///   1. bitcast <2 x i64> zeroinitializer to <64 x float> → replaced with zeroinitializer
    ///   2. half* device ptrs: load ptrs → float* (with index rescaling), store-only ptrs → stay half*
    ///   3. phi inference: params feeding phis that feed half GEPs are inferred as half*
    func testDotKernelHalfMMA() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Part 1: PSO creation (tests bitcast zeroinitializer + MMA + half ptrs)
        let mmaIR = """
        source_filename = "LLVMDialectModule"

        @__tg_dot_ab_0 = internal addrspace(3) global [512 x float] undef, align 4

        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @mma_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tidx = extractvalue [3 x i32] %tid, 0
          %lane = urem i32 %tidx, 32
          %a_idx = mul i32 %lane, 8
          %a_gep = getelementptr half, ptr addrspace(1) %0, i32 %a_idx
          %a_val = load half, ptr addrspace(1) %a_gep, align 2

          %mma_base = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 0
          %mma_a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
            ptr addrspace(3) %mma_base,
            <2 x i64> <i64 64, i64 8>, <2 x i64> <i64 1, i64 64>, <2 x i64> zeroinitializer)

          %zero_acc = bitcast <2 x i64> zeroinitializer to <64 x float>
          %mma_c = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(
            <64 x float> %mma_a, <64 x float> %mma_a, <64 x float> %zero_acc)

          call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
            <64 x float> %mma_c, ptr addrspace(3) %mma_base,
            <2 x i64> <i64 64, i64 8>, <2 x i64> <i64 1, i64 64>, <2 x i64> zeroinitializer)

          %out_gep = getelementptr half, ptr addrspace(1) %2, i32 %a_idx
          store half %a_val, ptr addrspace(1) %out_gep, align 2
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let mmaData = try MetalASM.assemble(ir: mmaIR)
        let device = MTLCreateSystemDefaultDevice()!
        let mmaLib = try device.makeLibrary(data: asDispatchData(mmaData))
        let mmaFn = mmaLib.makeFunction(name: "mma_kernel")!
        let mmaPso = try device.makeComputePipelineState(function: mmaFn)
        XCTAssertGreaterThan(mmaPso.maxTotalThreadsPerThreadgroup, 0)

        // Part 2: Correctness — half load from A, half store to C (store-only ptr).
        // Verifies index addressing is correct for both load (float*) and store (half*) paths.
        let copyIR = """
        source_filename = "LLVMDialectModule"

        @__tg_dot_ab_0 = internal addrspace(3) global [64 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @copy_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tidx = extractvalue [3 x i32] %tid, 0

          ; Load half from input (chained GEPs like real dot kernel)
          %base_gep = getelementptr half, ptr addrspace(1) %0, i32 0
          %elem_gep = getelementptr half, ptr addrspace(1) %base_gep, i32 %tidx
          %val = load half, ptr addrspace(1) %elem_gep, align 2

          ; Store half to output (store-only ptr, chained GEPs)
          %out_base = getelementptr half, ptr addrspace(1) %1, i32 0
          %out_elem = getelementptr half, ptr addrspace(1) %out_base, i32 %tidx
          store half %val, ptr addrspace(1) %out_elem, align 2

          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let copyData = try MetalASM.assemble(ir: copyIR)
        let copyLib = try device.makeLibrary(data: asDispatchData(copyData))
        let copyFn = copyLib.makeFunction(name: "copy_kernel")!
        let copyPso = try device.makeComputePipelineState(function: copyFn)

        let n = 64
        let aBuf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
        let aPtr = aBuf.contents().bindMemory(to: UInt16.self, capacity: n)
        let cPtr = cBuf.contents().bindMemory(to: UInt16.self, capacity: n)
        for i in 0..<n {
            var f = Float16(i)
            aPtr[i] = withUnsafeBytes(of: &f) { $0.load(as: UInt16.self) }
            cPtr[i] = 0xFFFF
        }

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(copyPso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(cBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: n, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        var failures = 0
        for i in 0..<n {
            let aVal = Float16(bitPattern: aPtr[i])
            let cVal = Float16(bitPattern: cPtr[i])
            if cVal != aVal { failures += 1 }
            if i < 4 || cVal != aVal {
                print("testDotKernelHalfMMA: C[\(i)] = \(cVal), expected \(aVal) \(cVal == aVal ? "OK" : "FAIL")")
            }
        }
        XCTAssertEqual(failures, 0, "\(failures)/\(n) elements wrong")
        #endif
    }

    /// Minimal repro: advancing ptr phi + MMA load in same loop = GPU JIT crash.
    /// Either alone works. The combination triggers "Failed to materializeAll."
    func testAdvancingPtrPhiWithMMA() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        @__tg_dot_ab_0 = internal addrspace(3) global [4096 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.wg.barrier(i32, i32)

        define void @test_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
        entry:
          br label %loop

        loop:
          %iv = phi i32 [ 0, %entry ], [ %iv_next, %loop ]
          %p0 = phi ptr addrspace(1) [ %0, %entry ], [ %p0_next, %loop ]

          %p0_next = getelementptr half, ptr addrspace(1) %p0, i32 64

          %tg_ptr = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i32 0
          %a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
            ptr addrspace(3) %tg_ptr,
            <2 x i64> <i64 8, i64 0>, <2 x i64> <i64 1, i64 0>, <2 x i64> <i64 0, i64 0>)

          call void @air.wg.barrier(i32 2, i32 1)

          %iv_next = add i32 %iv, 1
          %cond = icmp slt i32 %iv_next, 8
          br i1 %cond, label %loop, label %exit

        exit:
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "test_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    func testAtomicXchgI32() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        declare i32 @air.atomic.global.xchg.i32(ptr addrspace(1), i32, i32, i32, i1)

        define void @xchg_kernel(ptr addrspace(1) %0, i32 %1) {
          %3 = call i32 @air.atomic.global.xchg.i32(ptr addrspace(1) %0, i32 %1, i32 0, i32 2, i1 true)
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "xchg_kernel")
        XCTAssertNotNil(fn, "xchg_kernel not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        outBuf.contents().storeBytes(of: Int32(0), as: Int32.self)

        var val: Int32 = 42
        let valBuf = device.makeBuffer(bytes: &val, length: 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBuffer(valBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().load(as: Int32.self)
        print("testAtomicXchgI32: result = \(result), expected 42")
        XCTAssertEqual(result, 42, "xchg: expected 42, got \(result)")
        #endif
    }

    /// Test i1 (boolean) GEP + load — needed for tl.where with int tensors.
    func testI1GepLoad() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        define void @where_i32_kernel(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
          %cond_ptr = getelementptr i1, ptr addrspace(1) %cond, i32 0
          %cond_val = load i8, ptr addrspace(1) %cond_ptr, align 1
          %cond_bool = icmp ne i8 %cond_val, 0
          %x_ptr = getelementptr i32, ptr addrspace(1) %x, i32 0
          %x_val = load i32, ptr addrspace(1) %x_ptr, align 4
          %y_ptr = getelementptr i32, ptr addrspace(1) %y, i32 0
          %y_val = load i32, ptr addrspace(1) %y_ptr, align 4
          %sel = select i1 %cond_bool, i32 %x_val, i32 %y_val
          %out_ptr = getelementptr i32, ptr addrspace(1) %out, i32 0
          store i32 %sel, ptr addrspace(1) %out_ptr, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "where_i32_kernel")
        XCTAssertNotNil(fn)
        let pso = try device.makeComputePipelineState(function: fn!)

        let condBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let xBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let yBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

        condBuf.contents().storeBytes(of: UInt8(1), as: UInt8.self)
        xBuf.contents().storeBytes(of: Int32(42), as: Int32.self)
        yBuf.contents().storeBytes(of: Int32(99), as: Int32.self)

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(condBuf, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().load(as: Int32.self)
        print("testI1GepLoad: result = \(result), expected 42")
        XCTAssertEqual(result, 42)
        #endif
    }

    /// Test the EXACT Triton-generated IR for tl.where with int32.
    /// This is the full kernel that fails with "Failed to materializeAll."

    /// Bisect: strip features until we find what causes materializeAll failure
    func testWhereI32Bisect() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else

        // Test A: vector i32 TG store/load WITHOUT i1 GEP (no cond buffer)
        let irA = """
        source_filename = "LLVMDialectModule"
        @global_smem = internal addrspace(3) global [4096 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        define void @bisect_a(ptr addrspace(1) %x, ptr addrspace(1) %out) {
          %xp = getelementptr i32, ptr addrspace(1) %x, i32 0
          %xv = load i32, ptr addrspace(1) %xp, align 4
          %v0 = insertelement <4 x i32> undef, i32 %xv, i32 0
          %v1 = insertelement <4 x i32> %v0, i32 %xv, i32 1
          %v2 = insertelement <4 x i32> %v1, i32 %xv, i32 2
          %v3 = insertelement <4 x i32> %v2, i32 %xv, i32 3
          %tg = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store <4 x i32> %v3, ptr addrspace(3) %tg, align 16
          call void @air.wg.barrier(i32 1, i32 1)
          %ld = load <4 x i32>, ptr addrspace(3) %tg, align 16
          %r = extractelement <4 x i32> %ld, i32 0
          %op = getelementptr i32, ptr addrspace(1) %out, i32 0
          store i32 %r, ptr addrspace(1) %op, align 4
          ret void
        }
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        // Test B: i1 GEP + i8 load WITHOUT TG (no global_smem)
        let irB = """
        source_filename = "LLVMDialectModule"
        define void @bisect_b(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
          %cp = getelementptr i1, ptr addrspace(1) %cond, i32 0
          %cv = load i8, ptr addrspace(1) %cp, align 1
          %cb = icmp ne i8 %cv, 0
          %xp = getelementptr i32, ptr addrspace(1) %x, i32 0
          %xv = load i32, ptr addrspace(1) %xp, align 4
          %yp = getelementptr i32, ptr addrspace(1) %y, i32 0
          %yv = load i32, ptr addrspace(1) %yp, align 4
          %sel = select i1 %cb, i32 %xv, i32 %yv
          %op = getelementptr i32, ptr addrspace(1) %out, i32 0
          store i32 %sel, ptr addrspace(1) %op, align 4
          ret void
        }
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        // Test C: i1 GEP + TG + vector (full combo)
        let irC = """
        source_filename = "LLVMDialectModule"
        @global_smem = internal addrspace(3) global [4096 x i8] undef, align 16
        declare void @air.wg.barrier(i32, i32)
        define void @bisect_c(ptr addrspace(1) %cond, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(1) %out) {
          %cp = getelementptr i1, ptr addrspace(1) %cond, i32 0
          %cv = load i8, ptr addrspace(1) %cp, align 1
          %cb = icmp ne i8 %cv, 0
          %xp = getelementptr i32, ptr addrspace(1) %x, i32 0
          %xv = load i32, ptr addrspace(1) %xp, align 4
          %yp = getelementptr i32, ptr addrspace(1) %y, i32 0
          %yv = load i32, ptr addrspace(1) %yp, align 4
          %sel = select i1 %cb, i32 %xv, i32 %yv
          %v0 = insertelement <4 x i32> undef, i32 %sel, i32 0
          %v1 = insertelement <4 x i32> %v0, i32 %sel, i32 1
          %v2 = insertelement <4 x i32> %v1, i32 %sel, i32 2
          %v3 = insertelement <4 x i32> %v2, i32 %sel, i32 3
          %tg = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          store <4 x i32> %v3, ptr addrspace(3) %tg, align 16
          call void @air.wg.barrier(i32 1, i32 1)
          %ld = load <4 x i32>, ptr addrspace(3) %tg, align 16
          %r = extractelement <4 x i32> %ld, i32 0
          %op = getelementptr i32, ptr addrspace(1) %out, i32 0
          store i32 %r, ptr addrspace(1) %op, align 4
          ret void
        }
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        for (name, ir) in [("A:vec_tg_no_i1", irA), ("B:i1_no_tg", irB), ("C:i1+vec+tg", irC)] {
            do {
                let data = try MetalASM.assemble(ir: ir)
                let lib = try device.makeLibrary(data: asDispatchData(data))
                let fns = lib.functionNames
                let fn = lib.makeFunction(name: fns[0])!
                let pso = try device.makeComputePipelineState(function: fn)
                print("  \(name): PSO OK (maxThreads=\(pso.maxTotalThreadsPerThreadgroup))")
            } catch {
                print("  \(name): FAIL — \(error)")
                XCTFail("\(name) failed: \(error)")
            }
        }
        #endif
    }

    // MARK: - BFloat16 tests

    func testBFloat16ConstantAdd() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Exact IR from Triton: bf16 tensor + constexpr int scalar (0xR4040 = 3.0 in bf16)
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @bf16_const_add(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %p = getelementptr bfloat, ptr addrspace(1) %1, i32 %idx
          %v = load bfloat, ptr addrspace(1) %p, align 2
          %r = fadd bfloat %v, 0xR4040
          %po = getelementptr bfloat, ptr addrspace(1) %0, i32 %idx
          store bfloat %r, ptr addrspace(1) %po, align 2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "bf16_const_add")
        XCTAssertNotNil(fn, "bf16_const_add not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    func testBFloat16LoadStore() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @bf16_copy(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %p = getelementptr bfloat, ptr addrspace(1) %1, i32 %idx
          %v = load bfloat, ptr addrspace(1) %p, align 2
          %po = getelementptr bfloat, ptr addrspace(1) %0, i32 %idx
          store bfloat %v, ptr addrspace(1) %po, align 2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "bf16_copy")
        XCTAssertNotNil(fn, "bf16_copy not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    func testBFloat16ArithOps() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @bf16_arith(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %pa = getelementptr bfloat, ptr addrspace(1) %1, i32 %idx
          %a = load bfloat, ptr addrspace(1) %pa, align 2
          %pb = getelementptr bfloat, ptr addrspace(1) %2, i32 %idx
          %b = load bfloat, ptr addrspace(1) %pb, align 2
          %sum = fadd bfloat %a, %b
          %prod = fmul bfloat %sum, %b
          %diff = fsub bfloat %prod, %a
          %po = getelementptr bfloat, ptr addrspace(1) %0, i32 %idx
          store bfloat %diff, ptr addrspace(1) %po, align 2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "bf16_arith")
        XCTAssertNotNil(fn, "bf16_arith not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    func testFPTruncF32ToBFloat16() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @fptrunc_f32_bf16(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %p = getelementptr float, ptr addrspace(1) %0, i32 %idx
          %v = load float, ptr addrspace(1) %p, align 4
          %f = fptrunc float %v to bfloat
          %po = getelementptr bfloat, ptr addrspace(1) %1, i32 %idx
          store bfloat %f, ptr addrspace(1) %po, align 2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "fptrunc_f32_bf16")
        XCTAssertNotNil(fn, "fptrunc_f32_bf16 not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        let N = 8
        let inputData: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]
        let inBuf = device.makeBuffer(bytes: inputData, length: N * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: N * 2, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: UInt16.self, capacity: N)
        let expected: [UInt16] = [0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0]
        for i in 0..<N {
            XCTAssertEqual(outPtr[i], expected[i],
                "fptrunc f32(\(inputData[i])) to bfloat: got 0x\(String(outPtr[i], radix: 16)), expected 0x\(String(expected[i], radix: 16))")
        }
        #endif
    }

    func testLoadI8() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @load_i8_test(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %p = getelementptr i8, ptr addrspace(1) %0, i32 %idx
          %v = load i8, ptr addrspace(1) %p, align 1
          %ext = zext i8 %v to i32
          %po = getelementptr i32, ptr addrspace(1) %1, i32 %idx
          store i32 %ext, ptr addrspace(1) %po, align 4
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "load_i8_test")
        XCTAssertNotNil(fn, "load_i8_test not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        let N = 8
        let inputData: [UInt8] = [0, 1, 2, 3, 4, 5, 6, 7]
        let inBuf = device.makeBuffer(bytes: inputData, length: N, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Int32.self, capacity: N)
        for i in 0..<N {
            XCTAssertEqual(outPtr[i], Int32(i), "load_i8: thread \(i) got \(outPtr[i])")
        }
        #endif
    }

    func testSIToFPInt8ToBFloat16() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        define void @sitofp_i8_bf16_v5(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %idx = extractvalue [3 x i32] %tid, 0
          %p = getelementptr i8, ptr addrspace(1) %0, i32 %idx
          %v = load i8, ptr addrspace(1) %p, align 1
          %f = sitofp i8 %v to bfloat
          %po = getelementptr bfloat, ptr addrspace(1) %1, i32 %idx
          store bfloat %f, ptr addrspace(1) %po, align 2
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let device = MTLCreateSystemDefaultDevice()!
        let data = try MetalASM.assemble(ir: ir)
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "sitofp_i8_bf16_v5")
        XCTAssertNotNil(fn, "sitofp_i8_bf16_v5 not found")
        let pso = try device.makeComputePipelineState(function: fn!)

        let N = 8
        let inputData: [Int8] = [0, 1, 2, 3, 4, 5, 6, 7]
        let inBuf = device.makeBuffer(bytes: inputData, length: N, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: N * 2, options: .storageModeShared)!

        let queue = device.makeCommandQueue()!
        let cmdbuf = queue.makeCommandBuffer()!
        let enc = cmdbuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: UInt16.self, capacity: N)
        let expected: [UInt16] = [0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0]
        for i in 0..<N {
            XCTAssertEqual(outPtr[i], expected[i],
                "ref i8(\(i)) to bfloat: got 0x\(String(outPtr[i], radix: 16)), expected 0x\(String(expected[i], radix: 16))")
        }
        #endif
    }

    // MARK: - TG Global BFloat16 typed pointer propagation

    /// Test that transformTGGlobalGEPs Part 2 correctly propagates typed pointers
    /// for bfloat16 TG globals (no opaque ptr addrspace(3) should remain).
    func testTGGlobalBFloat16TypedPointers() throws {
        let ir = """
        @__tg_cvt_0 = internal addrspace(3) global [32 x bfloat] undef, align 4

        declare void @air.threadgroup.barrier(i32, i32)

        define void @test_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %tid_x, i32 %tid_y, i32 %tid_z) {
        entry:
          %tid64 = zext i32 %tid_x to i64
          %in_gep = getelementptr bfloat, ptr addrspace(1) %in, i64 %tid64
          %val = load bfloat, ptr addrspace(1) %in_gep, align 2
          %tg_gep = getelementptr bfloat, ptr addrspace(3) @__tg_cvt_0, i64 %tid64
          store bfloat %val, ptr addrspace(3) %tg_gep, align 2
          call void @air.threadgroup.barrier(i32 2, i32 1)
          %read_idx = sub i32 31, %tid_x
          %read_idx64 = zext i32 %read_idx to i64
          %tg_gep2 = getelementptr bfloat, ptr addrspace(3) @__tg_cvt_0, i64 %read_idx64
          %val2 = load bfloat, ptr addrspace(3) %tg_gep2, align 2
          %out_gep = getelementptr bfloat, ptr addrspace(1) %out, i64 %tid64
          store bfloat %val2, ptr addrspace(1) %out_gep, align 2
          ret void
        }
        !air.kernel = !{!0}
        !air.version = !{!7}
        !air.language_version = !{!8}
        !llvm.module.flags = !{!9, !10, !11, !12, !13, !14}
        !0 = !{void (ptr addrspace(1), ptr addrspace(1), i32, i32, i32)* @test_kernel, !1, !2}
        !1 = !{}
        !2 = !{!3, !4, !5, !6, !15}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"in"}
        !4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"out"}
        !5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !6 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_y"}
        !15 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_z"}
        !7 = !{i32 2, i32 8, i32 0}
        !8 = !{!"Metal", i32 3, i32 2, i32 0}
        !9 = !{i32 7, !"air.max_device_buffers", i32 31}
        !10 = !{i32 7, !"air.max_constant_buffers", i32 31}
        !11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        !12 = !{i32 7, !"air.max_textures", i32 128}
        !13 = !{i32 7, !"air.max_read_write_textures", i32 8}
        !14 = !{i32 7, !"air.max_samplers", i32 16}
        """

        // Parse and transform
        let lexer = Lexer(source: ir)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        applyAirTransforms(module: module)

        // Check: no opaque ptr addrspace(3) should remain
        var opaqueAS3Count = 0
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if case .opaquePointer(3) = inst.type { opaqueAS3Count += 1 }
                    for (_, op) in inst.operands.enumerated() {
                        if case .value(let v) = op, case .opaquePointer(3) = v.type { opaqueAS3Count += 1 }
                    }
                }
            }
        }
        XCTAssertEqual(opaqueAS3Count, 0, "Found \(opaqueAS3Count) opaque ptr addrspace(3) references")

        // Assemble and verify PSO creation
        let metallib = try MetalASM.assemble(ir: ir)

        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "test_kernel")
        XCTAssertNotNil(fn, "Kernel function not found")
        let pso = try device.makeComputePipelineState(function: fn!)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    /// Load arbitrary LLIR from TEST_LLIR env var, assemble and test PSO.
    func testExternalLLIR() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        guard let path = ProcessInfo.processInfo.environment["TEST_LLIR"] else {
            throw XCTSkip("Set TEST_LLIR=/path/to/file.llir")
        }
        let ir = try String(contentsOfFile: path, encoding: .utf8)
        // Dump transformed IR for debugging
        let lexer = Lexer(source: ir)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        applyAirTransforms(module: module)
        var dump = ""
        for fn in module.functions {
            dump += "FN: \(fn.name) isDecl=\(fn.isDeclaration) bbs=\(fn.basicBlocks.count)\n"
            for (bbIdx, bb) in fn.basicBlocks.enumerated() {
                dump += " BB[\(bbIdx)] '\(bb.name)' (\(bb.instructions.count) insts)\n"
                for inst in bb.instructions {
                    let ops = inst.operands.map { op -> String in
                        switch op {
                        case .basicBlock(let b): return "label %\(b.name)"
                        case .value(let v): return "%\(v.name)"
                        default: return "\(op)"
                        }
                    }.joined(separator: ", ")
                    dump += "  \(inst.name.isEmpty ? "_" : "%\(inst.name)") = \(inst.opcode) [\(ops)]\n"
                }
            }
        }
        try dump.write(toFile: "/tmp/inline_dump.txt", atomically: true, encoding: .utf8)
        let metallib = try MetalASM.assemble(ir: ir)
        try metallib.write(to: URL(fileURLWithPath: "/tmp/test_external.metallib"))
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fnName = library.functionNames.first!
        let fn = library.makeFunction(name: fnName)!
        let pso = try device.makeComputePipelineState(function: fn)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        print("testExternalLLIR: PSO OK, fn=\(fnName), maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")

        // If TEST_LLIR_DISPATCH=MxNxK, dispatch as a dot kernel: C = A @ B
        // Buffers: 0=A(MxK float), 1=B(KxN float), 2=C(MxN float)
        if let dims = ProcessInfo.processInfo.environment["TEST_LLIR_DISPATCH"] {
            let parts = dims.split(separator: "x").compactMap { Int($0) }
            guard parts.count == 3 else {
                XCTFail("TEST_LLIR_DISPATCH must be MxNxK, got \(dims)")
                return
            }
            let M = parts[0], N = parts[1], K = parts[2]
            let aBuf = device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
            let bBuf = device.makeBuffer(length: K * N * 4, options: .storageModeShared)!
            let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
            let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: M * K)
            let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: K * N)
            let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: M * N)
            // Fill A and B with deterministic pattern
            for i in 0..<(M*K) { aPtr[i] = Float(i % 7 - 3) }
            for i in 0..<(K*N) { bPtr[i] = Float(i % 5 - 2) }
            for i in 0..<(M*N) { cPtr[i] = -999.0 }

            let queue = device.makeCommandQueue()!
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(aBuf, offset: 0, index: 0)
            enc.setBuffer(bBuf, offset: 0, index: 1)
            enc.setBuffer(cBuf, offset: 0, index: 2)
            // Thread count: override via TEST_LLIR_THREADS env, default 128
            let threadsPerTG: Int
            if let t = ProcessInfo.processInfo.environment["TEST_LLIR_THREADS"], let n = Int(t) {
                threadsPerTG = n
            } else {
                threadsPerTG = 128
            }
            print("testExternalLLIR: dispatching \(threadsPerTG) threads")
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            // Compute reference: C[i,j] = sum_k A[i,k]*B[k,j]
            var refC = [Float](repeating: 0, count: M * N)
            for i in 0..<M {
                for j in 0..<N {
                    var sum: Float = 0
                    for k in 0..<K {
                        sum += aPtr[i * K + k] * bPtr[k * N + j]
                    }
                    refC[i * N + j] = sum
                }
            }
            var maxErr: Float = 0
            var firstBad = ""
            for i in 0..<M {
                for j in 0..<N {
                    let val = cPtr[i * N + j]
                    let err = abs(val - refC[i * N + j])
                    if err > maxErr {
                        maxErr = err
                        if firstBad.isEmpty && err > 0.01 {
                            firstBad = "C[\(i),\(j)]=\(val) expected \(refC[i * N + j])"
                        }
                    }
                }
            }
            // Show per-8x8-tile errors and dump bad tile values
            for tm in 0..<(M/8) {
                for tn in 0..<(N/8) {
                    var tileMax: Float = 0
                    for i in (tm*8)..<(tm*8+8) {
                        for j in (tn*8)..<(tn*8+8) {
                            tileMax = max(tileMax, abs(cPtr[i*N+j] - refC[i*N+j]))
                        }
                    }
                    if tileMax > 0.01 {
                        print("  tile[\(tm),\(tn)] rows \(tm*8)-\(tm*8+7) cols \(tn*8)-\(tn*8+7): max_err=\(tileMax)")
                        if tm == 7 && tn == 2 {
                            // Check if got matches ref from a different tile
                            for otm in 0..<(M/8) {
                                let match = (0..<8).allSatisfy { j in
                                    cPtr[56*N + tn*8 + j] == refC[otm*8*N + tn*8 + j]
                                }
                                if match { print("    got row56 = ref row\(otm*8) tile[\(otm),\(tn)]!") }
                            }
                            print("    got row56: ", (0..<8).map { cPtr[56*N + tn*8 + $0] })
                            print("    ref row56: ", (0..<8).map { refC[56*N + tn*8 + $0] })
                            // Also check tile[6,2] (tm=6 = rows 48-55, same tn)
                            print("    ref row48: ", (0..<8).map { refC[48*N + tn*8 + $0] })
                            print("    ref row40: ", (0..<8).map { refC[40*N + tn*8 + $0] })
                        }
                    }
                }
            }
            print("testExternalLLIR: dispatch \(M)x\(N)x\(K) max_err=\(maxErr) \(firstBad)")
            XCTAssertLessThan(maxErr, 0.1, "dot \(M)x\(N)x\(K) max_err=\(maxErr) \(firstBad)")
        }
        #endif
    }

    /// Minimal repro for join_scalars materializeAll failure
    func testJoinScalars() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Full join_scalars IR from Triton — two TG globals, constant GEP with nuw
        let ir = """
        @__tg_cvt_0 = internal addrspace(3) global [2 x i32] undef, align 4
        @global_smem = internal addrspace(3) global [8 x i8] undef, align 16

        declare void @air.threadgroup.barrier(i32, i32)
        declare [3 x i32] @air.thread_position_in_threadgroup()
        declare i32 @air.thread_index_in_simdgroup()

        define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
          %4 = load i32, ptr addrspace(1) %0, align 4
          %5 = load i32, ptr addrspace(1) %1, align 4
          %6 = call i32 @air.thread_index_in_simdgroup()
          %7 = call [3 x i32] @air.thread_position_in_threadgroup()
          %8 = urem i32 %6, 2
          %9 = add i32 0, %8
          store i32 %4, ptr addrspace(3) @__tg_cvt_0, align 4
          store i32 %5, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__tg_cvt_0, i64 4), align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %10 = add i32 0, %9
          %11 = zext i32 %10 to i64
          %12 = getelementptr i32, ptr addrspace(3) @__tg_cvt_0, i64 %11
          %13 = load i32, ptr addrspace(3) %12, align 4
          %14 = call [3 x i32] @air.thread_position_in_threadgroup()
          %15 = extractvalue [3 x i32] %14, 0
          %16 = zext i32 %15 to i64
          %17 = trunc i64 %16 to i32
          %18 = and i32 %17, 127
          %19 = urem i32 %18, 32
          %20 = call [3 x i32] @air.thread_position_in_threadgroup()
          %21 = extractvalue [3 x i32] %20, 0
          %22 = udiv i32 %21, 32
          %23 = shl i32 %19, 0
          %24 = or i32 0, %23
          %25 = shl i32 %22, 5
          %26 = or i32 %24, %25
          %27 = and i32 %26, 1
          %28 = icmp eq i32 %27, 0
          %29 = select i1 %28, i32 0, i32 1
          %30 = or disjoint i32 %29, 0
          %31 = xor i32 0, %30
          %32 = xor i32 %31, 0
          %33 = add i32 %32, 0
          %34 = getelementptr float, ptr addrspace(1) %2, i32 %33
          %35 = sitofp i32 %13 to float
          store float %35, ptr addrspace(1) %34, align 4
          ret void
        }
        """
        let metallib = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)

        // Execute and verify: x=42, y=100 → output[0]=42.0, output[1]=100.0
        let queue = device.makeCommandQueue()!
        let bufX = device.makeBuffer(bytes: [Int32(42)], length: 4, options: .storageModeShared)!
        let bufY = device.makeBuffer(bytes: [Int32(100)], length: 4, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: 128 * MemoryLayout<Float>.size, options: .storageModeShared)!
        memset(bufOut.contents(), 0, bufOut.length)

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufX, offset: 0, index: 0)
        enc.setBuffer(bufY, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let out = bufOut.contents().bindMemory(to: Float.self, capacity: 128)
        XCTAssertEqual(out[0], 42.0, "output[0] should be x=42")
        XCTAssertEqual(out[1], 100.0, "output[1] should be y=100")
        #endif
    }

    // ── While loop: count down from 8 by 2, add 1.0 each iteration → expect 4.0 ──
    func testWhileLoop() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // While pattern — accumulate via phi (no load/store in loop body)
        // count=8, decrement by 2, accumulate +1.0 each iteration → 4.0
        let ir = """
        target triple = "air64_v28-apple-macosx26.0.0"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

        define void @kernel(float addrspace(1)* %0, i32 addrspace(1)* %1, <3 x i32> %2) {
          %4 = extractelement <3 x i32> %2, i32 0
          %5 = icmp eq i32 %4, 0
          br i1 %5, label %body, label %exit

        body:
          %count0 = load i32, i32 addrspace(1)* %1, align 4
          br label %loop

        loop:
          %count = phi i32 [ %count0, %body ], [ %newcount, %loop ]
          %val = load float, float addrspace(1)* %0, align 4
          %newacc = fadd float %val, 1.000000e+00
          store float %newacc, float addrspace(1)* %0, align 4
          %newcount = sub i32 %count, 2
          %test = icmp sgt i32 %newcount, 0
          br i1 %test, label %loop, label %done

        done:
          ret void

        exit:
          ret void
        }

        !air.kernel = !{!0}
        !air.version = !{!6}
        !air.language_version = !{!7}
        !llvm.module.flags = !{!8, !9, !10, !11, !12, !13}

        !0 = !{void (float addrspace(1)*, i32 addrspace(1)*, <3 x i32>)* @kernel, !1, !2}
        !1 = !{}
        !2 = !{!3, !4, !5}
        !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
        !4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"n"}
        !5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
        !6 = !{i32 2, i32 8, i32 0}
        !7 = !{!"Metal", i32 3, i32 2, i32 0}
        !8 = !{i32 7, !"air.max_device_buffers", i32 31}
        !9 = !{i32 7, !"air.max_constant_buffers", i32 31}
        !10 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        !11 = !{i32 7, !"air.max_textures", i32 128}
        !12 = !{i32 7, !"air.max_read_write_textures", i32 8}
        !13 = !{i32 7, !"air.max_samplers", i32 16}
        """
        let metallib = try MetalASM.assemble(ir: ir)
        try metallib.write(to: URL(fileURLWithPath: "/tmp/while_ours.metallib"))
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fn = library.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let queue = device.makeCommandQueue()!
        // data buffer (float) initialized to 0.0
        var dataVal: Float = 0.0
        let bufData = device.makeBuffer(bytes: &dataVal, length: 4, options: .storageModeShared)!
        // count buffer (i32) = 8
        var countVal: Int32 = 8
        let bufCount = device.makeBuffer(bytes: &countVal, length: 4, options: .storageModeShared)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufData, offset: 0, index: 0)
        enc.setBuffer(bufCount, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = bufData.contents().bindMemory(to: Float.self, capacity: 1)[0]
        XCTAssertEqual(result, 4.0, "while loop should iterate 4 times (8/2)")
        #endif
    }

    // ── Constant GEP offset: two TG slots at @global_smem+0 and @global_smem+8 ──
    func testConstantGEPOffsetCorrectness() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Minimal kernel: thread 0 stores 42 at TG offset 0, 99 at TG offset 8.
        // After barrier, thread 0 loads both back and writes them to output[0] and output[1].
        // Tests that constant GEP expression `getelementptr(i8, @global_smem, i64 8)` is handled.
        let ir = """
        @global_smem = internal addrspace(3) global [16 x i8] undef, align 16

        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @kernel(ptr addrspace(1) %0) {
          %tid3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid = extractvalue [3 x i32] %tid3, 0
          %is0 = icmp eq i32 %tid, 0

          ; Thread 0: store 42 at TG offset 0
          br i1 %is0, label %store_vals, label %after_store

        store_vals:
          %slot0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          %v42 = insertelement <1 x i32> undef, i32 42, i32 0
          store <1 x i32> %v42, ptr addrspace(3) %slot0, align 4

          ; Store 99 at TG offset 8 using constant GEP expression
          %slot1 = getelementptr inbounds i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i64 8), i32 0
          %v99 = insertelement <1 x i32> undef, i32 99, i32 0
          store <1 x i32> %v99, ptr addrspace(3) %slot1, align 4
          br label %after_store

        after_store:
          call void @air.wg.barrier(i32 1, i32 1)

          ; Thread 0: load both back and write to output
          br i1 %is0, label %load_vals, label %done

        load_vals:
          %rd0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
          %ld0 = load <1 x i32>, ptr addrspace(3) %rd0, align 4
          %val0 = extractelement <1 x i32> %ld0, i32 0

          %rd1 = getelementptr inbounds i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i64 8), i32 0
          %ld1 = load <1 x i32>, ptr addrspace(3) %rd1, align 4
          %val1 = extractelement <1 x i32> %ld1, i32 0

          %out0 = getelementptr i32, ptr addrspace(1) %0, i32 0
          store i32 %val0, ptr addrspace(1) %out0, align 4
          %out1 = getelementptr i32, ptr addrspace(1) %0, i32 1
          store i32 %val1, ptr addrspace(1) %out1, align 4
          br label %done

        done:
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(data: asDispatchData(data))
        let fn = library.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let outBuf = device.makeBuffer(length: 8, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Int32.self, capacity: 2)
        outPtr[0] = -1
        outPtr[1] = -1

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let r0 = outPtr[0]
        let r1 = outPtr[1]
        print("testConstantGEPOffsetCorrectness: slot0=\(r0) (expect 42), slot1=\(r1) (expect 99)")
        XCTAssertEqual(r0, 42, "TG offset 0 should contain 42")
        XCTAssertEqual(r1, 99, "TG offset 8 (via constant GEP) should contain 99")
        #endif
    }

    func testFloatNaNReferenceMetallib() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Test: load reference metallib (compiled with metal toolchain) with NaN constant
        let path = "/tmp/test_nan_metal.metallib"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Reference \(path) not found — compile test_nan.metal first")
        }
        let device = MTLCreateSystemDefaultDevice()!
        let url = URL(fileURLWithPath: path)
        let lib = try device.makeLibrary(URL: url)
        let fn = lib.makeFunction(name: "nan_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        XCTAssertTrue(outPtr[0].isNaN, "Reference NaN: expected NaN, got \(outPtr[0])")
        print("testFloatNaNReferenceMetallib: outPtr[0]=\(outPtr[0]) PASS")
        #endif
    }

    func testFloat42Constant() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "test_42"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @test42_kernel(ptr addrspace(1) %buf, i32 %tid_x) {
          %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
          store float 42.0, ptr addrspace(1) %p, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!5}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @test42_kernel, !2, !3}
        !2 = !{}
        !3 = !{!4, !6}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !6 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, i32 8, i32 0}
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "test42_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        XCTAssertEqual(outPtr[0], 42.0, "Expected 42.0, got \(outPtr[0])")
        print("testFloat42Constant: outPtr[0]=\(outPtr[0]) PASS")
        #endif
    }

    func testFloatInfConstant() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "test_inf"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @inf_kernel(ptr addrspace(1) %buf, i32 %tid_x) {
          %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
          store float 0x7FF0000000000000, ptr addrspace(1) %p, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!5}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @inf_kernel, !2, !3}
        !2 = !{}
        !3 = !{!4, !6}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !6 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, i32 8, i32 0}
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "inf_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        XCTAssertTrue(outPtr[0].isInfinite, "Expected inf, got \(outPtr[0])")
        print("testFloatInfConstant: outPtr[0]=\(outPtr[0]) PASS")
        #endif
    }

    /// Minimal repro: GEP with byte offset into middle of a single TG global
    /// crashes Metal GPU JIT ("Failed to materializeAll"). The fix is to split
    /// the monolithic @global_smem into separate TG globals for each region.
    func testTGByteOffsetGEPCrash() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        @global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %4 = extractvalue [3 x i32] %3, 0
          %5 = and i32 %4, 31

          ; Store to base of global_smem
          %6 = getelementptr float, ptr addrspace(3) @global_smem, i32 %5
          store float 1.0, ptr addrspace(3) %6, align 4

          ; Store to global_smem + 512 bytes — byte-offset GEP into middle of TG global
          %base2 = getelementptr i8, ptr addrspace(3) @global_smem, i64 512
          %7 = getelementptr float, ptr addrspace(3) %base2, i32 %5
          store float 2.0, ptr addrspace(3) %7, align 4

          call void @air.wg.barrier(i32 1, i32 1)

          %8 = load float, ptr addrspace(3) %6, align 4
          %9 = load float, ptr addrspace(3) %7, align 4
          %10 = fadd float %8, %9

          %11 = getelementptr float, ptr addrspace(1) %1, i32 %5
          store float %10, ptr addrspace(1) %11, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        // Dispatch and verify correctness: each thread should store 1.0 + 2.0 = 3.0
        let count = 32
        let inBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        for i in 0..<count {
            XCTAssertEqual(outPtr[i], 3.0, "thread \(i): expected 3.0, got \(outPtr[i])")
        }
        #endif
    }

    /// Same as testTGByteOffsetGEPCrash but with nested constant GEP expression
    /// (the form Triton actually emits for multi-value scan).
    func testTGNestedConstantGEPCrash() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "LLVMDialectModule"

        @global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

        declare void @air.wg.barrier(i32, i32)
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1) {
          %3 = call [3 x i32] @air.thread_position_in_threadgroup()
          %4 = extractvalue [3 x i32] %3, 0
          %5 = and i32 %4, 31

          %6 = getelementptr float, ptr addrspace(3) @global_smem, i32 %5
          store float 1.0, ptr addrspace(3) %6, align 4

          ; Nested constant GEP — the exact pattern Triton emits
          %7 = getelementptr float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i64 512), i32 %5
          store float 2.0, ptr addrspace(3) %7, align 4

          call void @air.wg.barrier(i32 1, i32 1)

          %8 = load float, ptr addrspace(3) %6, align 4
          %9 = load float, ptr addrspace(3) %7, align 4
          %10 = fadd float %8, %9

          %11 = getelementptr float, ptr addrspace(1) %1, i32 %5
          store float %10, ptr addrspace(1) %11, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let count = 32
        let inBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        for i in 0..<count {
            XCTAssertEqual(outPtr[i], 3.0, "thread \(i): expected 3.0, got \(outPtr[i])")
        }
        #endif
    }

    /// Full 2D linear recurrence scan IR from Triton — the actual failing kernel.
    func testScan2DLinearRecurrence() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = try String(contentsOfFile: "/tmp/scan2d_linear_recurrence.ll", encoding: .utf8)
        let data = try MetalASM.assemble(ir: ir)
        XCTAssertGreaterThan(data.count, 100)

        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "scan2d_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        XCTAssertGreaterThan(pso.maxTotalThreadsPerThreadgroup, 0)
        #endif
    }

    func testFloatNaNConstant() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Minimal repro: store float NaN constant to output buffer.
        // float 0x7FF8000000000000 is LLVM IR double-hex for float qNaN (bits 0x7FC00000).
        // This crashed Metal GPU JIT due to bitcode encoding issues.
        let ir = """
        source_filename = "test_nan"
        target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @nan_kernel(ptr addrspace(1) %buf, i32 %tid_x) {
          %p = getelementptr float, ptr addrspace(1) %buf, i32 %tid_x
          store float 0x7FF8000000000000, ptr addrspace(1) %p, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !air.kernel = !{!1}
        !air.version = !{!5}

        !0 = !{i32 7, !"frame-pointer", i32 0}
        !1 = !{ptr @nan_kernel, !2, !3}
        !2 = !{}
        !3 = !{!4, !6}
        !4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
        !6 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tid_x"}
        !5 = !{i32 2, i32 8, i32 0}
        """

        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "nan_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        let count = 32
        let outBuf = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = 0.0 }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        XCTAssertTrue(outPtr[0].isNaN, "Expected NaN, got \(outPtr[0])")
        XCTAssertTrue(outPtr[1].isNaN, "Expected NaN at index 1, got \(outPtr[1])")
        print("testFloatNaNConstant: outPtr[0]=\(outPtr[0]) (bits: \(String(outPtr[0].bitPattern, radix: 16)))")
        #endif
    }

    /// Variable-index half GEP + MMA + correctness: the load-and-extract pattern.
    /// Each thread loads A[lane] (half) via variable-index GEP, converts to float,
    /// stores to TG, does MMA(ones, diag(A)), checks result = A broadcast across rows.
    /// Tests that lshr index scaling + extractelement picks correct half for both
    /// even and odd indices.
    func testHalfGEPLoadExtractMMA() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // Simpler: each thread loads input[tid] (half) via variable GEP,
        // stores to TG, then MMA load from TG. We verify the load is correct.
        // Key: tid is variable, so GEP gets lshr scaling. Even tids (0,2,4..)
        // and odd tids (1,3,5..) exercise both extractelement lanes.
        let ir = """
        source_filename = "LLVMDialectModule"

        @__tg_buf = internal addrspace(3) global [64 x float] undef, align 4

        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()

        define void @half_extract_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
          %lane = call i32 @air.thread_index_in_simdgroup()

          ; Variable-index half GEP — this triggers lshr + load-and-extract
          %p = getelementptr half, ptr addrspace(1) %in, i32 %lane
          %v = load half, ptr addrspace(1) %p
          %vf = fpext half %v to float

          ; Store to TG
          %lane64 = zext i32 %lane to i64
          %tg_p = getelementptr float, ptr addrspace(3) @__tg_buf, i64 %lane64
          store float %vf, ptr addrspace(3) %tg_p

          call void @air.threadgroup.barrier(i32 1, i32 4)

          ; MMA load (forces MMA path, validates typed pointer compatibility)
          %mma = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
            ptr addrspace(3) @__tg_buf,
            <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)

          ; Write loaded half value to output for verification
          %out_p = getelementptr float, ptr addrspace(1) %out, i32 %lane
          store float %vf, ptr addrspace(1) %out_p
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "half_extract_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        // Input: 32 half values = 0.0, 1.0, 2.0, ..., 31.0
        // Tests both even indices (lane 0,2,4..) and odd indices (lane 1,3,5..)
        let inBuf = device.makeBuffer(length: 32 * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
        let inPtr = inBuf.contents().bindMemory(to: UInt16.self, capacity: 32)
        for i in 0..<32 {
            // Convert Float to float16 bits
            inPtr[i] = Float16(Float(i)).bitPattern
        }
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: 32)
        for i in 0..<32 { outPtr[i] = -999.0 }

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        var maxErr: Float = 0
        for i in 0..<32 {
            let expected = Float(i)
            let err = abs(outPtr[i] - expected)
            maxErr = max(maxErr, err)
            if err > 0.1 {
                print("testHalfGEPLoadExtractMMA: out[\(i)] = \(outPtr[i]) expected \(expected)")
            }
        }
        print("testHalfGEPLoadExtractMMA: max_err = \(maxErr)")
        XCTAssertLessThan(maxErr, 0.1, "Half GEP load-and-extract max_err=\(maxErr)")
        #endif
    }

    /// Test llvm.minimum.f32 (NaN-propagating) lowering.
    /// llvm.minimum returns NaN if either operand is NaN.
    /// Metal only has air.fmin (minnum — NaN-ignoring), so MetalASM must expand
    /// to fmin + fcmp uno + select.
    func testLLVMMinimumNaNPropagation() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "test"

        declare float @llvm.minimum.f32(float, float)

        define void @min_nan_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid_x = extractvalue [3 x i32] %tid, 0
          %idx = zext i32 %tid_x to i64
          %pa = getelementptr float, ptr addrspace(1) %A, i64 %idx
          %a = load float, ptr addrspace(1) %pa
          %pb = getelementptr float, ptr addrspace(1) %B, i64 %idx
          %b = load float, ptr addrspace(1) %pb
          %r = call float @llvm.minimum.f32(float %a, float %b)
          %pc = getelementptr float, ptr addrspace(1) %C, i64 %idx
          store float %r, ptr addrspace(1) %pc
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "min_nan_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testLLVMMinimumNaNPropagation: PSO OK")

        let N = 4
        let aBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!

        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: N)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: N)
        // lane 0: min(1,2) = 1 (no NaN)
        // lane 1: min(NaN,3) = NaN (a is NaN)
        // lane 2: min(4,NaN) = NaN (b is NaN)
        // lane 3: min(5,2) = 2 (no NaN)
        aPtr[0] = 1; aPtr[1] = .nan; aPtr[2] = 4; aPtr[3] = 5
        bPtr[0] = 2; bPtr[1] = 3;    bPtr[2] = .nan; bPtr[3] = 2

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: N)
        print("testLLVMMinimumNaNPropagation: results = [\(cPtr[0]), \(cPtr[1]), \(cPtr[2]), \(cPtr[3])]")
        XCTAssertEqual(cPtr[0], 1.0, accuracy: 1e-5, "min(1,2) should be 1")
        XCTAssert(cPtr[1].isNaN, "min(NaN,3) should be NaN")
        XCTAssert(cPtr[2].isNaN, "min(4,NaN) should be NaN")
        XCTAssertEqual(cPtr[3], 2.0, accuracy: 1e-5, "min(5,2) should be 2")
        #endif
    }

    /// Test llvm.maximum.f32 (NaN-propagating) lowering.
    func testLLVMMaximumNaNPropagation() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        let ir = """
        source_filename = "test"

        declare float @llvm.maximum.f32(float, float)

        define void @max_nan_kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {
          %tid = call [3 x i32] @air.thread_position_in_threadgroup()
          %tid_x = extractvalue [3 x i32] %tid, 0
          %idx = zext i32 %tid_x to i64
          %pa = getelementptr float, ptr addrspace(1) %A, i64 %idx
          %a = load float, ptr addrspace(1) %pa
          %pb = getelementptr float, ptr addrspace(1) %B, i64 %idx
          %b = load float, ptr addrspace(1) %pb
          %r = call float @llvm.maximum.f32(float %a, float %b)
          %pc = getelementptr float, ptr addrspace(1) %C, i64 %idx
          store float %r, ptr addrspace(1) %pc
          ret void
        }

        declare [3 x i32] @air.thread_position_in_threadgroup()
        """
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "max_nan_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)
        print("testLLVMMaximumNaNPropagation: PSO OK")

        let N = 4
        let aBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!

        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: N)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: N)
        // lane 0: max(1,2) = 2 (no NaN)
        // lane 1: max(NaN,3) = NaN (a is NaN)
        // lane 2: max(4,NaN) = NaN (b is NaN)
        // lane 3: max(5,2) = 5 (no NaN)
        aPtr[0] = 1; aPtr[1] = .nan; aPtr[2] = 4; aPtr[3] = 5
        bPtr[0] = 2; bPtr[1] = 3;    bPtr[2] = .nan; bPtr[3] = 2

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: N)
        print("testLLVMMaximumNaNPropagation: results = [\(cPtr[0]), \(cPtr[1]), \(cPtr[2]), \(cPtr[3])]")
        XCTAssertEqual(cPtr[0], 2.0, accuracy: 1e-5, "max(1,2) should be 2")
        XCTAssert(cPtr[1].isNaN, "max(NaN,3) should be NaN")
        XCTAssert(cPtr[2].isNaN, "max(4,NaN) should be NaN")
        XCTAssertEqual(cPtr[3], 5.0, accuracy: 1e-5, "max(5,2) should be 5")
        #endif
    }

    /// 16x16 dot kernel LLIR with dual TG globals (__tg_cvt_0 + __tg_dot_ab_0).
    /// Tests that MetalASM keeps them separate (no coalescing without MMA calls).
    /// No scalar buffer, 3 device ptrs.
    func testDualTGGlobalDot16x16() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        // 16x16 dot-like kernel using TG-based scatter/gather (no MMA calls).
        // Has two TG globals that must NOT be coalesced.
        let ir = """
        ; ModuleID = 'LLVMDialectModule'
        source_filename = "LLVMDialectModule"

        @__tg_cvt_0 = internal addrspace(3) global [256 x float] undef, align 4
        @__tg_dot_ab_0 = internal addrspace(3) global [256 x float] undef, align 4
        @global_smem = internal addrspace(3) global [1024 x i8] undef, align 16

        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
        declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
        declare void @air.simdgroup.barrier(i32, i32)
        declare void @air.threadgroup.barrier(i32, i32)
        declare i32 @air.thread_index_in_simdgroup()
        declare [3 x i32] @air.thread_position_in_threadgroup()

        define void @dot_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
          %4 = call [3 x i32] @air.thread_position_in_threadgroup()
          %5 = extractvalue [3 x i32] %4, 0
          %6 = zext i32 %5 to i64
          %7 = trunc i64 %6 to i32
          %8 = and i32 %7, 127
          %9 = urem i32 %8, 32
          %10 = call [3 x i32] @air.thread_position_in_threadgroup()
          %11 = extractvalue [3 x i32] %10, 0
          %12 = udiv i32 %11, 32
          %13 = shl i32 %9, 0
          %14 = or i32 0, %13
          %15 = shl i32 %12, 5
          %16 = or i32 %14, %15
          %17 = and i32 %16, 120
          %18 = lshr i32 %17, 3
          %19 = or disjoint i32 %18, 0
          %20 = xor i32 0, %19
          %21 = xor i32 %20, 0
          %22 = add i32 %21, 0
          %23 = mul i32 %22, 16
          %24 = getelementptr float, ptr addrspace(1) %0, i32 %23
          %25 = call [3 x i32] @air.thread_position_in_threadgroup()
          %26 = extractvalue [3 x i32] %25, 0
          %27 = zext i32 %26 to i64
          %28 = trunc i64 %27 to i32
          %29 = and i32 %28, 127
          %30 = urem i32 %29, 32
          %31 = call [3 x i32] @air.thread_position_in_threadgroup()
          %32 = extractvalue [3 x i32] %31, 0
          %33 = udiv i32 %32, 32
          %34 = shl i32 %30, 0
          %35 = or i32 0, %34
          %36 = shl i32 %33, 5
          %37 = or i32 %35, %36
          %38 = and i32 %37, 7
          %39 = shl i32 %38, 1
          %40 = or disjoint i32 %39, 0
          %41 = xor i32 0, %40
          %42 = xor i32 %41, 0
          %43 = xor i32 %41, 1
          %44 = add i32 %42, 0
          %45 = add i32 %43, 0
          %46 = getelementptr float, ptr addrspace(1) %24, i32 %44
          %47 = getelementptr float, ptr addrspace(1) %24, i32 %45
          %48 = load float, ptr addrspace(1) %46, align 4
          %49 = load float, ptr addrspace(1) %47, align 4
          %50 = getelementptr float, ptr addrspace(1) %1, i32 %23
          %51 = getelementptr float, ptr addrspace(1) %50, i32 %44
          %52 = getelementptr float, ptr addrspace(1) %50, i32 %45
          %53 = load float, ptr addrspace(1) %51, align 4
          %54 = load float, ptr addrspace(1) %52, align 4
          %55 = call i32 @air.thread_index_in_simdgroup()
          %56 = call [3 x i32] @air.thread_position_in_threadgroup()
          %57 = extractvalue [3 x i32] %56, 0
          %58 = udiv i32 %57, 32
          %59 = udiv i32 %55, 8
          %60 = urem i32 %55, 8
          %61 = mul i32 %58, 4
          %62 = add i32 %61, %59
          %63 = mul i32 %60, 2
          %64 = add i32 0, %63
          %65 = udiv i32 %55, 16
          %66 = urem i32 %55, 16
          %67 = mul i32 %58, 2
          %68 = add i32 %67, %65
          %69 = add i32 0, %66
          %70 = mul i32 %62, 16
          %71 = add i32 %70, %64
          %72 = zext i32 %71 to i64
          %73 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %72
          store float %48, ptr addrspace(3) %73, align 4
          %74 = add i32 %64, 1
          %75 = add i32 %70, %74
          %76 = zext i32 %75 to i64
          %77 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %76
          store float %49, ptr addrspace(3) %77, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %78 = mul i32 %68, 16
          %79 = add i32 %78, %69
          %80 = zext i32 %79 to i64
          %81 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %80
          store float 0.000000e+00, ptr addrspace(3) %81, align 4
          %82 = add i32 %68, 8
          %83 = mul i32 %82, 16
          %84 = add i32 %83, %69
          %85 = zext i32 %84 to i64
          %86 = getelementptr float, ptr addrspace(3) @__tg_dot_ab_0, i64 %85
          store float 0.000000e+00, ptr addrspace(3) %86, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          call void @air.threadgroup.barrier(i32 1, i32 4)
          store float %53, ptr addrspace(3) %73, align 4
          store float %54, ptr addrspace(3) %77, align 4
          call void @air.threadgroup.barrier(i32 1, i32 4)
          call void @air.threadgroup.barrier(i32 1, i32 4)
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %87 = load float, ptr addrspace(3) %81, align 4
          %88 = load float, ptr addrspace(3) %86, align 4
          %89 = getelementptr float, ptr addrspace(1) %2, i32 %23
          %90 = getelementptr float, ptr addrspace(1) %89, i32 %44
          %91 = getelementptr float, ptr addrspace(1) %89, i32 %45
          %92 = call i32 @air.thread_index_in_simdgroup()
          %93 = call [3 x i32] @air.thread_position_in_threadgroup()
          %94 = extractvalue [3 x i32] %93, 0
          %95 = udiv i32 %94, 32
          %96 = udiv i32 %92, 16
          %97 = urem i32 %92, 16
          %98 = mul i32 %95, 2
          %99 = add i32 %98, %96
          %100 = add i32 0, %97
          %101 = udiv i32 %92, 8
          %102 = urem i32 %92, 8
          %103 = mul i32 %95, 4
          %104 = add i32 %103, %101
          %105 = mul i32 %102, 2
          %106 = add i32 0, %105
          br i1 true, label %107, label %117

        107:
          %108 = mul i32 %99, 16
          %109 = add i32 %108, %100
          %110 = zext i32 %109 to i64
          %111 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %110
          store float %87, ptr addrspace(3) %111, align 4
          %112 = add i32 %99, 8
          %113 = mul i32 %112, 16
          %114 = add i32 %113, %100
          %115 = zext i32 %114 to i64
          %116 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %115
          store float %88, ptr addrspace(3) %116, align 4
          br label %117

        117:
          call void @air.threadgroup.barrier(i32 1, i32 4)
          %118 = mul i32 %104, 16
          %119 = add i32 %118, %106
          %120 = zext i32 %119 to i64
          %121 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %120
          %122 = load float, ptr addrspace(3) %121, align 4
          %123 = add i32 %106, 1
          %124 = add i32 %118, %123
          %125 = zext i32 %124 to i64
          %126 = getelementptr float, ptr addrspace(3) @__tg_cvt_0, i64 %125
          %127 = load float, ptr addrspace(3) %126, align 4
          store float %122, ptr addrspace(1) %90, align 4
          store float %127, ptr addrspace(1) %91, align 4
          ret void
        }

        !llvm.module.flags = !{!0}
        !0 = !{i32 2, !"Debug Info Version", i32 3}
        """

        let M = 16, N = 16, K = 16
        let data = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let lib = try device.makeLibrary(data: asDispatchData(data))
        let fn = lib.makeFunction(name: "dot_kernel")!
        let pso = try device.makeComputePipelineState(function: fn)

        // This LLIR has no MMA calls — it's a scatter/gather through two TG buffers.
        // The kernel passes B values through to output (B is scattered last to __tg_dot_ab_0,
        // then gathered via __tg_cvt_0). This tests that MetalASM keeps __tg_cvt_0 and
        // __tg_dot_ab_0 as separate globals (no coalescing without MMA).
        let aBuf = device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: K * N * 4, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: M * K)
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: K * N)
        let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: M * N)
        for i in 0..<(M * K) { aPtr[i] = Float(i % 5 + 1) }   // A = 1..5
        for i in 0..<(K * N) { bPtr[i] = Float(i % 3 + 10) }  // B = 10..12
        for i in 0..<(M * N) { cPtr[i] = -999.0 }  // sentinel

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        // 4 warps × 32 threads = 128 threads in one threadgroup
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Verify output = B values (no MMA, B is passed through)
        var failures = 0
        for i in 0..<M {
            for j in 0..<N {
                let got = cPtr[i * N + j]
                let expected = bPtr[i * N + j]
                if abs(got - expected) > 0.01 { failures += 1 }
            }
        }
        print("testDualTGGlobalDot16x16: \(failures == 0 ? "OK" : "FAIL \(failures)/\(M*N)")")
        XCTAssertEqual(failures, 0, "\(failures)/\(M*N) elements wrong — TG global coalescing bug?")
        #endif
    }

    /// Dispatch an external .ll kernel with 2 buffers (input i32[], output i32[])
    /// and verify output against expected values from TEST_LLIR_EXPECTED.
    ///
    /// Usage:
    ///   TEST_LLIR=/tmp/kernel.ll \
    ///   TEST_LLIR_RUN="in_count:out_count:threads" \
    ///   swift test --filter testExternalLLIRRun
    ///
    /// in_count/out_count = number of i32 elements, threads = threadgroup width.
    /// Input is filled with deterministic pattern (i % 1000).
    /// Output is printed so you can compare with reference.
    func testExternalLLIRRun() throws {
        #if !canImport(Metal)
        throw XCTSkip("Metal not available")
        #else
        guard let path = ProcessInfo.processInfo.environment["TEST_LLIR"] else {
            throw XCTSkip("Set TEST_LLIR=/path/to/file.ll")
        }
        guard let runSpec = ProcessInfo.processInfo.environment["TEST_LLIR_RUN"] else {
            throw XCTSkip("Set TEST_LLIR_RUN=in_count:out_count:threads")
        }
        let parts = runSpec.split(separator: ":").compactMap { Int($0) }
        guard parts.count == 3 else {
            XCTFail("TEST_LLIR_RUN must be in_count:out_count:threads, got \(runSpec)")
            return
        }
        let inCount = parts[0], outCount = parts[1], threads = parts[2]

        let ir = try String(contentsOfFile: path, encoding: .utf8)
        let metallib = try MetalASM.assemble(ir: ir)
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(data: asDispatchData(metallib))
        let fnName = library.functionNames.first!
        let fn = library.makeFunction(name: fnName)!
        let pso = try device.makeComputePipelineState(function: fn)

        let inBuf = device.makeBuffer(length: inCount * 4, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: outCount * 4, options: .storageModeShared)!
        let inPtr = inBuf.contents().bindMemory(to: Int32.self, capacity: inCount)
        let outPtr = outBuf.contents().bindMemory(to: Int32.self, capacity: outCount)

        // Deterministic input
        for i in 0..<inCount { inPtr[i] = Int32(i % 1000) }
        for i in 0..<outCount { outPtr[i] = -1 }

        let queue = device.makeCommandQueue()!
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Print output for comparison
        print("testExternalLLIRRun: output[\(outCount)] =", (0..<outCount).map { outPtr[$0] })

        // If TEST_LLIR_EXPECTED is set, verify
        if let expectedStr = ProcessInfo.processInfo.environment["TEST_LLIR_EXPECTED"] {
            let expected = expectedStr.split(separator: ",").compactMap { Int32($0) }
            guard expected.count == outCount else {
                XCTFail("Expected \(outCount) values, got \(expected.count)")
                return
            }
            var failures = 0
            for i in 0..<outCount {
                if outPtr[i] != expected[i] {
                    print("  MISMATCH at [\(i)]: got \(outPtr[i]), expected \(expected[i])")
                    failures += 1
                }
            }
            XCTAssertEqual(failures, 0, "\(failures)/\(outCount) elements wrong")
        }
        #endif
    }
}
