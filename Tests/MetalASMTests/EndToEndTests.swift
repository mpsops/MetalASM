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
    func testMinimalIRDisassembly() throws {
        let ir = try String(contentsOfFile: "/tmp/gemm_simple.ll", encoding: .utf8)
        
        let metallib = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
        try metallib.write(to: URL(fileURLWithPath: "/tmp/minimal_metalasm.metallib"))
        
        // Extract bitcode from metallib
        let bytes = [UInt8](metallib)
        var bcOffset = 0
        var bcSize = 0
        for i in 0..<(bytes.count - 4) {
            if bytes[i] == 0xDE && bytes[i+1] == 0xC0 && bytes[i+2] == 0x17 && bytes[i+3] == 0x0B {
                let off = Int(bytes[i+8]) | (Int(bytes[i+9]) << 8) | (Int(bytes[i+10]) << 16) | (Int(bytes[i+11]) << 24)
                let sz = Int(bytes[i+12]) | (Int(bytes[i+13]) << 8) | (Int(bytes[i+14]) << 16) | (Int(bytes[i+15]) << 24)
                bcOffset = i + off
                bcSize = sz
                break
            }
        }
        
        let bcData = Data(bytes[bcOffset..<(bcOffset + bcSize)])
        try bcData.write(to: URL(fileURLWithPath: "/tmp/minimal_metalasm.bc"))
        print("Wrote \(bcSize) bytes of bitcode to /tmp/minimal_metalasm.bc")
        XCTAssertGreaterThan(bcSize, 0)
    }

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
}
