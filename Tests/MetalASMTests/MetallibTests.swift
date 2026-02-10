import XCTest
@testable import MetalASM
import Foundation

final class MetallibTests: XCTestCase {

    /// Test wrapping known-good bitcode from metal-as into a metallib.
    func testWrapKnownBitcode() throws {
        // Read the reference .air file (which is a bitcode wrapper)
        let airURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.air")
        guard FileManager.default.fileExists(atPath: airURL.path) else {
            throw XCTSkip("Reference file not found at \(airURL.path)")
        }
        let airData = try Data(contentsOf: airURL)
        let airBytes = Array(airData)

        // Wrap into metallib
        let metallib = MetalASM.wrapAIR(
            airBytes,
            kernelNames: ["monolithic_kernel"]
        )

        // Verify the header
        XCTAssertEqual(metallib[0], 0x4D) // M
        XCTAssertEqual(metallib[1], 0x54) // T
        XCTAssertEqual(metallib[2], 0x4C) // L
        XCTAssertEqual(metallib[3], 0x42) // B

        // Verify file size matches
        let fileSize = metallib.withUnsafeBytes { buf in
            buf.load(fromByteOffset: 16, as: UInt64.self)
        }
        XCTAssertEqual(Int(fileSize), metallib.count)

        // Verify NAME tag contains the kernel name
        XCTAssertTrue(metallib.contains("monolithic_kernel"))

        // Verify the bitcode is present
        let wrapperMagic: [UInt8] = [0xDE, 0xC0, 0x17, 0x0B]
        XCTAssertTrue(containsSubsequence(Array(metallib), wrapperMagic))

        // Verify HASH is computed (not all zeros)
        let hashTag = Array("HASH".utf8)
        if let hashIdx = findSubsequence(Array(metallib), hashTag) {
            let hashDataStart = hashIdx + 6  // skip tag(4) + size(2)
            let hashSlice = Array(metallib[hashDataStart..<hashDataStart+32])
            XCTAssertFalse(hashSlice.allSatisfy { $0 == 0 }, "HASH should not be all zeros")
        } else {
            XCTFail("HASH tag not found")
        }
    }

    /// Test that our metallib matches the reference structure.
    func testMetallibStructure() throws {
        let airURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.air")
        guard FileManager.default.fileExists(atPath: airURL.path) else {
            throw XCTSkip("Reference file not found")
        }
        let airData = try Data(contentsOf: airURL)
        let airBytes = Array(airData)

        let metallib = MetalASM.wrapAIR(
            airBytes,
            kernelNames: ["monolithic_kernel"]
        )
        let bytes = Array(metallib)

        // Parse section descriptors
        let s0Off = readU64(bytes, 24)
        let s0Size = readU64(bytes, 32)
        let s1Off = readU64(bytes, 40)
        let s1Size = readU64(bytes, 48)
        let s2Off = readU64(bytes, 56)
        let s2Size = readU64(bytes, 64)
        let s3Off = readU64(bytes, 72)
        let s3Size = readU64(bytes, 80)

        // Section 0 should start at 88 (header size = 4+12+8+64)
        XCTAssertEqual(s0Off, 88)

        // Section 1 and 2 should be 8 bytes each (u32(4) + "ENDT")
        XCTAssertEqual(s1Size, 8)
        XCTAssertEqual(s2Size, 8)

        // Section 3 size should match the wrapped bitcode size
        XCTAssertEqual(Int(s3Size), airBytes.count)

        // Sections should be contiguous
        XCTAssertEqual(s1Off, s0Off + s0Size)
        XCTAssertEqual(s2Off, s1Off + s1Size)
        XCTAssertEqual(s3Off, s2Off + s2Size)
        XCTAssertEqual(Int(s3Off + s3Size), bytes.count)
    }

    // GPU loading test is in EndToEndTests.swift (requires Metal framework)

    // MARK: - Helpers

    func containsSubsequence(_ data: [UInt8], _ sub: [UInt8]) -> Bool {
        return findSubsequence(data, sub) != nil
    }

    func findSubsequence(_ data: [UInt8], _ sub: [UInt8]) -> Int? {
        guard sub.count <= data.count else { return nil }
        for i in 0...(data.count - sub.count) {
            if Array(data[i..<i+sub.count]) == sub {
                return i
            }
        }
        return nil
    }

    func readU64(_ data: [UInt8], _ offset: Int) -> UInt64 {
        var value: UInt64 = 0
        for i in 0..<8 {
            value |= UInt64(data[offset + i]) << (i * 8)
        }
        return value
    }
}

// Extension to check if Data contains a string
extension Data {
    func contains(_ string: String) -> Bool {
        let searchBytes = Array(string.utf8)
        let dataBytes = Array(self)
        guard searchBytes.count <= dataBytes.count else { return false }
        for i in 0...(dataBytes.count - searchBytes.count) {
            if Array(dataBytes[i..<i+searchBytes.count]) == searchBytes {
                return true
            }
        }
        return false
    }
}
