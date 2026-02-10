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
        var parser = Parser(tokens: tokens)
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
}
