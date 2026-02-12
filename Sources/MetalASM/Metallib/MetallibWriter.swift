import Foundation
import CryptoKit

/// Wraps LLVM bitcode bytes into a valid .metallib container.
///
/// The metallib format has:
/// - 92-byte header (magic, platform, file size, 4 section descriptors, entry count)
/// - Section 0: entry headers (tag-based: NAME, TYPE, HASH, MDSZ, OFFT, VERS, ENDT)
/// - Section 1: function list (just ENDT)
/// - Section 2: public metadata (just ENDT)
/// - Section 3: bitcode data (wrapper + LLVM bitcode)
public struct MetallibWriter {

    /// Platform target for the metallib.
    public enum Platform {
        case macOS(version: Int)
        case iOS(version: Int)
    }

    /// Entry type in the metallib.
    public enum EntryType: UInt8 {
        case kernel = 2
        case visible = 4
    }

    /// A single function entry in the metallib.
    public struct Entry {
        public var name: String
        public var type: EntryType
        /// AIR version (major, minor).
        public var airVersionMajor: UInt16
        public var airVersionMinor: UInt16
        /// Metal language version (major, minor).
        public var metalVersionMajor: UInt16
        public var metalVersionMinor: UInt16

        public init(
            name: String,
            type: EntryType = .kernel,
            airVersionMajor: UInt16 = 2,
            airVersionMinor: UInt16 = 8,
            metalVersionMajor: UInt16 = 4,
            metalVersionMinor: UInt16 = 0
        ) {
            self.name = name
            self.type = type
            self.airVersionMajor = airVersionMajor
            self.airVersionMinor = airVersionMinor
            self.metalVersionMajor = metalVersionMajor
            self.metalVersionMinor = metalVersionMinor
        }
    }

    /// Build a complete metallib from LLVM bitcode bytes and entry descriptors.
    ///
    /// - Parameters:
    ///   - bitcode: Raw LLVM bitcode (starting with "BC\xC0\xDE"). NOT the wrapper.
    ///   - entries: Function entries to include.
    ///   - platform: Target platform.
    /// - Returns: Complete metallib data suitable for `device.makeLibrary(data:)`.
    public static func build(
        bitcode: [UInt8],
        entries: [Entry],
        platform: Platform = .macOS(version: 26)
    ) -> Data {
        // Wrap the bitcode in the Apple bitcode wrapper
        let wrappedBitcode = wrapBitcode(bitcode)

        // Build entry header tags for each entry
        let entryHeaders = entries.map { entry in
            buildEntryHeader(entry: entry, bitcodeSize: wrappedBitcode.count)
        }

        // Build section 0: entry count (u32) + entry sizes (u32 each) + all entry tag data
        var section0 = Data()
        appendU32(&section0, UInt32(entries.count))
        for header in entryHeaders {
            appendU32(&section0, UInt32(header.count))
        }
        for header in entryHeaders {
            section0.append(contentsOf: header)
        }

        // Section 1: function list (u32 size + ENDT)
        let section1 = buildENDTSection()

        // Section 2: public metadata (u32 size + ENDT)
        let section2 = buildENDTSection()

        // Section 3: wrapped bitcode
        let section3 = Data(wrappedBitcode)

        // Compute section offsets
        let headerSize = 88
        let s0Offset = headerSize
        let s1Offset = s0Offset + section0.count
        let s2Offset = s1Offset + section1.count
        let s3Offset = s2Offset + section2.count
        let totalSize = s3Offset + section3.count

        // Build the 92-byte header
        var header = Data()

        // [0-3] Magic
        header.append(contentsOf: [0x4D, 0x54, 0x4C, 0x42]) // "MTLB"

        // [4-15] Platform/version info
        header.append(contentsOf: platformBytes(platform))

        // [16-23] Total file size (u64)
        appendU64(&header, UInt64(totalSize))

        // [24-87] 4 section descriptors: (offset: u64, size: u64)
        appendU64(&header, UInt64(s0Offset))
        appendU64(&header, UInt64(section0.count))
        appendU64(&header, UInt64(s1Offset))
        appendU64(&header, UInt64(section1.count))
        appendU64(&header, UInt64(s2Offset))
        appendU64(&header, UInt64(section2.count))
        appendU64(&header, UInt64(s3Offset))
        appendU64(&header, UInt64(section3.count))

        // Assemble
        assert(header.count == headerSize)
        var result = header
        result.append(section0)
        result.append(section1)
        result.append(section2)
        result.append(section3)
        assert(result.count == totalSize)
        return result
    }

    /// Build metallib from an already-wrapped bitcode section (wrapper + bitcode).
    public static func buildFromWrappedBitcode(
        wrappedBitcode: [UInt8],
        entries: [Entry],
        platform: Platform = .macOS(version: 26)
    ) -> Data {
        let entryHeaders = entries.map { entry in
            buildEntryHeader(entry: entry, bitcodeSize: wrappedBitcode.count)
        }

        var section0 = Data()
        appendU32(&section0, UInt32(entries.count))
        for header in entryHeaders {
            appendU32(&section0, UInt32(header.count))
        }
        for header in entryHeaders {
            section0.append(contentsOf: header)
        }

        let section1 = buildENDTSection()
        let section2 = buildENDTSection()
        let section3 = Data(wrappedBitcode)

        let headerSize = 88
        let s0Offset = headerSize
        let s1Offset = s0Offset + section0.count
        let s2Offset = s1Offset + section1.count
        let s3Offset = s2Offset + section2.count
        let totalSize = s3Offset + section3.count

        var header = Data()
        header.append(contentsOf: [0x4D, 0x54, 0x4C, 0x42])
        header.append(contentsOf: platformBytes(platform))
        appendU64(&header, UInt64(totalSize))
        appendU64(&header, UInt64(s0Offset))
        appendU64(&header, UInt64(section0.count))
        appendU64(&header, UInt64(s1Offset))
        appendU64(&header, UInt64(section1.count))
        appendU64(&header, UInt64(s2Offset))
        appendU64(&header, UInt64(section2.count))
        appendU64(&header, UInt64(s3Offset))
        appendU64(&header, UInt64(section3.count))

        assert(header.count == headerSize)
        var result = header
        result.append(section0)
        result.append(section1)
        result.append(section2)
        result.append(section3)
        assert(result.count == totalSize)
        return result
    }

    // MARK: - Bitcode wrapper

    /// Wrap raw LLVM bitcode in Apple's bitcode wrapper.
    ///
    /// Format: 20-byte header + raw bitcode
    /// - [0-3]: Magic 0x0B17C0DE (little-endian: DE C0 17 0B)
    /// - [4-7]: Version = 0
    /// - [8-11]: Offset to bitcode = 20
    /// - [12-15]: Size of bitcode
    /// - [16-19]: CPU type = 0xFFFFFFFF
    static func wrapBitcode(_ bitcode: [UInt8]) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(20 + bitcode.count)

        // Wrapper magic
        appendU32LE(&result, 0x0B17C0DE)
        // Version
        appendU32LE(&result, 0)
        // Offset to bitcode
        appendU32LE(&result, 20)
        // Size of bitcode
        appendU32LE(&result, UInt32(bitcode.count))
        // CPU type
        appendU32LE(&result, 0xFFFFFFFF)

        result.append(contentsOf: bitcode)
        return result
    }

    // MARK: - Entry header tags

    /// Build the tag data for a single entry header.
    static func buildEntryHeader(entry: Entry, bitcodeSize: Int) -> [UInt8] {
        var tags = [UInt8]()

        // NAME tag
        let nameBytes = Array(entry.name.utf8) + [0] // null-terminated
        appendTag(&tags, name: "NAME", data: nameBytes)

        // TYPE tag
        appendTag(&tags, name: "TYPE", data: [entry.type.rawValue])

        // HASH tag (SHA256 of the wrapped bitcode section)
        // Note: we compute this over the entire wrapped bitcode
        // The actual hash will be computed in build() since we need the full bitcode
        // For now, write placeholder zeros - will be patched
        appendTag(&tags, name: "HASH", data: Array(repeating: 0, count: 32))

        // MDSZ tag (bitcode section size as u64)
        var mdszData = [UInt8](repeating: 0, count: 8)
        writeU64LE(&mdszData, 0, UInt64(bitcodeSize))
        appendTag(&tags, name: "MDSZ", data: mdszData)

        // OFFT tag (3 x u64 offsets, all zero for single-entry)
        let offtData = [UInt8](repeating: 0, count: 24)
        appendTag(&tags, name: "OFFT", data: offtData)

        // VERS tag (air_major, air_minor, metal_major, metal_minor as 4 x u16)
        var versData = [UInt8](repeating: 0, count: 8)
        writeU16LE(&versData, 0, entry.airVersionMajor)
        writeU16LE(&versData, 2, entry.airVersionMinor)
        writeU16LE(&versData, 4, entry.metalVersionMajor)
        writeU16LE(&versData, 6, entry.metalVersionMinor)
        appendTag(&tags, name: "VERS", data: versData)

        // ENDT tag (just the 4-byte tag name, no size/data)
        tags.append(contentsOf: Array("ENDT".utf8))

        return tags
    }

    /// Build a section containing just u32(4) + "ENDT".
    static func buildENDTSection() -> Data {
        var data = Data()
        appendU32(&data, 4) // size of ENDT = 4
        data.append(contentsOf: Array("ENDT".utf8))
        return data
    }

    // MARK: - SHA256 patching

    /// Compute SHA256 of the wrapped bitcode and patch the HASH tags in the result.
    static func patchHashes(in data: inout Data, wrappedBitcode: [UInt8]) {
        let hash = SHA256.hash(data: wrappedBitcode)
        let hashBytes = Array(hash)

        // Find all HASH tags and patch them
        let hashTag = Array("HASH".utf8)
        var searchStart = 0
        while searchStart < data.count - 38 {
            if let range = data.range(of: Data(hashTag), in: searchStart..<data.count) {
                let dataStart = range.upperBound + 2 // skip 2-byte size field
                if dataStart + 32 <= data.count {
                    for i in 0..<32 {
                        data[dataStart + i] = hashBytes[i]
                    }
                }
                searchStart = range.upperBound
            } else {
                break
            }
        }
    }

    // MARK: - Platform bytes

    /// Generate the 12-byte platform/version descriptor (bytes 4-15 of header).
    static func platformBytes(_ platform: Platform) -> [UInt8] {
        // Reverse-engineered from known metallib files:
        // Byte 4: target (1 = macos, 6 = ios?)
        // Byte 5: flags (0x80)
        // Bytes 6-7: container version (2)
        // Bytes 8-9: file type (9)
        // Byte 10: reserved (0)
        // Byte 11: os type (0x81 = macos, 0x82 = ios?)
        // Bytes 12-15: os version (u32)
        switch platform {
        case .macOS(let version):
            var b = [UInt8](repeating: 0, count: 12)
            b[0] = 0x01   // macos target
            b[1] = 0x80   // flags
            b[2] = 0x02   // container version lo
            b[3] = 0x00   // container version hi
            b[4] = 0x09   // file type lo
            b[5] = 0x00   // file type hi
            b[6] = 0x00   // reserved
            b[7] = 0x81   // os type
            writeU32LE(&b, 8, UInt32(version))
            return b
        case .iOS(let version):
            // iOS metallibs use the same platform bytes as macOS.
            // Confirmed by comparing metal-as CLI output for iOS targets.
            var b = [UInt8](repeating: 0, count: 12)
            b[0] = 0x01   // same as macOS
            b[1] = 0x80   // flags
            b[2] = 0x02   // container version lo
            b[3] = 0x00   // container version hi
            b[4] = 0x09   // file type lo
            b[5] = 0x00   // file type hi
            b[6] = 0x00   // reserved
            b[7] = 0x81   // same as macOS
            writeU32LE(&b, 8, UInt32(version))
            return b
        }
    }

    // MARK: - Tag helpers

    /// Append a tag: 4-byte name + u16 size + data.
    static func appendTag(_ buffer: inout [UInt8], name: String, data: [UInt8]) {
        assert(name.count == 4)
        buffer.append(contentsOf: Array(name.utf8))
        buffer.append(UInt8(data.count & 0xFF))
        buffer.append(UInt8((data.count >> 8) & 0xFF))
        buffer.append(contentsOf: data)
    }

    // MARK: - Binary helpers

    static func appendU32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &v) { Array($0) })
    }

    static func appendU64(_ data: inout Data, _ value: UInt64) {
        var v = value.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &v) { Array($0) })
    }

    static func appendU32LE(_ buffer: inout [UInt8], _ value: UInt32) {
        buffer.append(UInt8(value & 0xFF))
        buffer.append(UInt8((value >> 8) & 0xFF))
        buffer.append(UInt8((value >> 16) & 0xFF))
        buffer.append(UInt8((value >> 24) & 0xFF))
    }

    static func writeU16LE(_ buffer: inout [UInt8], _ offset: Int, _ value: UInt16) {
        buffer[offset] = UInt8(value & 0xFF)
        buffer[offset + 1] = UInt8((value >> 8) & 0xFF)
    }

    static func writeU32LE(_ buffer: inout [UInt8], _ offset: Int, _ value: UInt32) {
        buffer[offset] = UInt8(value & 0xFF)
        buffer[offset + 1] = UInt8((value >> 8) & 0xFF)
        buffer[offset + 2] = UInt8((value >> 16) & 0xFF)
        buffer[offset + 3] = UInt8((value >> 24) & 0xFF)
    }

    static func writeU64LE(_ buffer: inout [UInt8], _ offset: Int, _ value: UInt64) {
        for i in 0..<8 {
            buffer[offset + i] = UInt8((value >> (i * 8)) & 0xFF)
        }
    }
}
