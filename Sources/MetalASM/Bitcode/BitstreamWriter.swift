/// Low-level LLVM bitstream writer.
///
/// Emits bits LSB-first within each byte, supporting VBR encoding,
/// block enter/exit, and unabbreviated records per the LLVM bitcode format.
public final class BitstreamWriter {
    /// The buffer accumulating output bytes.
    private(set) var bytes: [UInt8] = []

    /// Current byte being accumulated (not yet flushed).
    private var currentByte: UInt8 = 0

    /// Number of bits written into `currentByte` (0..7).
    private var bitOffset: Int = 0

    /// Stack of (blockStartByteIndex, outerAbbrevLen, outerNextAbbrevID, outerAbbrevDefs) for nested blocks.
    private var blockStack: [(startIndex: Int, outerAbbrevLen: Int, outerNextAbbrevID: UInt64, outerAbbrevDefs: [UInt64: [(type: UInt8, data: UInt64)]])] = []

    /// Current abbreviation length (bits per abbrev id).
    private var abbrevLen: Int = 2

    // MARK: - Built-in abbrev IDs

    /// END_BLOCK abbreviation ID.
    static let endBlockAbbrevID: UInt64 = 0
    /// ENTER_SUBBLOCK abbreviation ID.
    static let enterSubblockAbbrevID: UInt64 = 1
    /// DEFINE_ABBREV abbreviation ID.
    static let defineAbbrevAbbrevID: UInt64 = 2
    /// UNABBREV_RECORD abbreviation ID.
    static let unabbrevRecordAbbrevID: UInt64 = 3

    /// Total number of bits written so far.
    var totalBits: Int {
        bytes.count * 8 + bitOffset
    }

    public init() {}

    /// Initialize with a capacity hint (bytes).
    public init(capacity: Int) {
        bytes.reserveCapacity(capacity)
    }

    // MARK: - Bit-level emission

    /// Emit a single bit (0 or 1).
    func emitBit(_ bit: UInt64) {
        if (bit & 1) != 0 {
            currentByte |= UInt8(1 << bitOffset)
        }
        bitOffset += 1
        if bitOffset == 8 {
            bytes.append(currentByte)
            currentByte = 0
            bitOffset = 0
        }
    }

    /// Emit `numBits` low bits of `value`, LSB first.
    @inline(__always)
    func emit(_ value: UInt64, _ numBits: Int) {
        // Fast path: if everything fits in the current byte
        if numBits <= 8 - bitOffset {
            currentByte |= UInt8(truncatingIfNeeded: value) << bitOffset
            bitOffset += numBits
            if bitOffset == 8 {
                bytes.append(currentByte)
                currentByte = 0
                bitOffset = 0
            }
            return
        }

        // General path: pack into a 64-bit accumulator, flush whole bytes
        var acc = UInt64(currentByte) | (value << bitOffset)
        var totalBits = bitOffset + numBits

        while totalBits >= 8 {
            bytes.append(UInt8(truncatingIfNeeded: acc))
            acc >>= 8
            totalBits -= 8
        }

        currentByte = UInt8(truncatingIfNeeded: acc)
        bitOffset = totalBits
    }

    /// Emit a value using Variable Bit Rate encoding with `numBits` chunk size.
    /// Each chunk has (numBits-1) data bits + 1 continuation bit.
    @inline(__always)
    func emitVBR(_ value: UInt64, _ numBits: Int) {
        let dataMask: UInt64 = (1 << (numBits - 1)) - 1
        // Fast path: value fits in one chunk (very common)
        if value <= dataMask {
            emit(value, numBits)
            return
        }
        // General path
        var val = value
        let dataBits = numBits - 1
        while true {
            let chunk = val & dataMask
            val >>= dataBits
            if val == 0 {
                emit(chunk, numBits)
                return
            }
            emit(chunk | (1 << dataBits), numBits)
        }
    }

    /// Emit a signed value using VBR encoding.
    /// Encodes sign in the LSB: positive n → 2n, negative n → (-2n) + 1.
    func emitSignedVBR(_ value: Int64, _ numBits: Int) {
        let encoded: UInt64
        if value >= 0 {
            encoded = UInt64(value) << 1
        } else {
            encoded = (UInt64(bitPattern: -value) << 1) | 1
        }
        emitVBR(encoded, numBits)
    }

    // MARK: - Alignment

    /// Pad to 32-bit boundary with zero bits.
    func alignTo32Bits() {
        let totalBits = bytes.count * 8 + bitOffset
        let remainder = totalBits % 32
        if remainder != 0 {
            let padding = 32 - remainder
            emit(0, padding)
        }
    }

    // MARK: - Block operations

    /// Enter a sub-block. Emits ENTER_SUBBLOCK with the given block ID and
    /// abbreviation length, then writes a placeholder for the block length.
    func enterSubblock(blockID: UInt64, abbrevLen newAbbrevLen: Int) {
        // Emit ENTER_SUBBLOCK abbreviation ID
        emit(Self.enterSubblockAbbrevID, abbrevLen)
        // Block ID as VBR8
        emitVBR(blockID, 8)
        // New abbreviation length as VBR4
        emitVBR(UInt64(newAbbrevLen), 4)
        // Align to 32-bit boundary before writing block length placeholder
        alignTo32Bits()
        // Save state: remember where the block length word will go
        let startIndex = bytes.count
        // Write placeholder for block length (in 32-bit words, excluding this word itself)
        bytes.append(contentsOf: [0, 0, 0, 0])
        blockStack.append((startIndex: startIndex, outerAbbrevLen: abbrevLen, outerNextAbbrevID: nextAbbrevID, outerAbbrevDefs: abbrevDefs))
        abbrevLen = newAbbrevLen
        nextAbbrevID = 4
        abbrevDefs = [:]
    }

    /// Exit the current sub-block. Writes END_BLOCK, aligns to 32 bits,
    /// and patches the block length placeholder.
    func exitBlock() {
        guard let (startIndex, outerAbbrevLen, outerNextAbbrevID, outerAbbrevDefs) = blockStack.popLast() else {
            fatalError("exitBlock called with no matching enterSubblock")
        }
        // Emit END_BLOCK abbreviation ID
        emit(Self.endBlockAbbrevID, abbrevLen)
        alignTo32Bits()
        // Patch the block length: number of 32-bit words AFTER the length field
        let blockContentBytes = bytes.count - startIndex - 4
        assert(blockContentBytes % 4 == 0)
        let blockLengthWords = UInt32(blockContentBytes / 4)
        bytes[startIndex + 0] = UInt8(blockLengthWords & 0xFF)
        bytes[startIndex + 1] = UInt8((blockLengthWords >> 8) & 0xFF)
        bytes[startIndex + 2] = UInt8((blockLengthWords >> 16) & 0xFF)
        bytes[startIndex + 3] = UInt8((blockLengthWords >> 24) & 0xFF)
        abbrevLen = outerAbbrevLen
        nextAbbrevID = outerNextAbbrevID
        abbrevDefs = outerAbbrevDefs
    }

    // MARK: - Records

    /// Emit an unabbreviated record with the given code and operands.
    /// All values are encoded as VBR6.
    func emitUnabbrevRecord(code: UInt64, operands: [UInt64]) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(UInt64(operands.count), 6)
        for op in operands {
            emitVBR(op, 6)
        }
    }

    // MARK: - Inline unabbreviated record emission (no array allocation)

    @inline(__always)
    func emitUnabbrevRecord(code: UInt64) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(0, 6)
    }

    @inline(__always)
    func emitUnabbrevRecord(code: UInt64, _ a: UInt64) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(1, 6)
        emitVBR(a, 6)
    }

    @inline(__always)
    func emitUnabbrevRecord(code: UInt64, _ a: UInt64, _ b: UInt64) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(2, 6)
        emitVBR(a, 6)
        emitVBR(b, 6)
    }

    @inline(__always)
    func emitUnabbrevRecord(code: UInt64, _ a: UInt64, _ b: UInt64, _ c: UInt64) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(3, 6)
        emitVBR(a, 6)
        emitVBR(b, 6)
        emitVBR(c, 6)
    }

    @inline(__always)
    func emitUnabbrevRecord(code: UInt64, _ a: UInt64, _ b: UInt64, _ c: UInt64, _ d: UInt64) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(4, 6)
        emitVBR(a, 6)
        emitVBR(b, 6)
        emitVBR(c, 6)
        emitVBR(d, 6)
    }

    /// Emit an unabbreviated record whose operands are UTF-8 bytes of a string.
    /// Avoids creating an intermediate [UInt64] array.
    @inline(__always)
    func emitUnabbrevStringRecord(code: UInt64, _ str: String) {
        let utf8 = str.utf8
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(UInt64(utf8.count), 6)
        for b in utf8 {
            emitVBR(UInt64(b), 6)
        }
    }

    /// Emit an unabbreviated record with a leading operand followed by UTF-8 bytes.
    @inline(__always)
    func emitUnabbrevStringRecord(code: UInt64, leading: UInt64, _ str: String) {
        let utf8 = str.utf8
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        emitVBR(UInt64(1 + utf8.count), 6)
        emitVBR(leading, 6)
        for b in utf8 {
            emitVBR(UInt64(b), 6)
        }
    }

    /// Emit an unabbreviated record with a trailing blob (array of chars/bytes).
    /// Used for string records.
    func emitUnabbrevRecordWithBlob(code: UInt64, operands: [UInt64], blob: [UInt8]) {
        emit(Self.unabbrevRecordAbbrevID, abbrevLen)
        emitVBR(code, 6)
        // Total operands = operands.count + blob.count
        emitVBR(UInt64(operands.count + blob.count), 6)
        for op in operands {
            emitVBR(op, 6)
        }
        for b in blob {
            emitVBR(UInt64(b), 6)
        }
    }

    // MARK: - Raw data emission

    /// Write raw bytes directly (must be 32-bit aligned before calling).
    func emitRawBytes(_ data: [UInt8]) {
        assert(bitOffset == 0, "Must be byte-aligned to emit raw bytes")
        bytes.append(contentsOf: data)
    }

    /// Emit the LLVM bitcode magic "BC\xC0\xDE" (4 bytes).
    func emitBitcodeMagic() {
        assert(bitOffset == 0)
        bytes.append(contentsOf: [0x42, 0x43, 0xC0, 0xDE])
    }

    // MARK: - Abbreviation support

    /// Stored abbreviation operand encodings, keyed by abbrevID.
    /// Each entry is the list of non-literal operand encodings: (type, data).
    /// Literal operands are omitted (emitted automatically by the definition).
    private var abbrevDefs: [UInt64: [(type: UInt8, data: UInt64)]] = [:]

    /// Next abbreviation ID to assign in the current block.
    private var nextAbbrevID: UInt64 = 4

    /// Emit a record using a defined abbreviation.
    /// `abbrevID` is the abbreviation ID (>= 4).
    /// `operands` are the non-literal operand values in definition order.
    func emitAbbreviatedRecord(abbrevID: UInt64, operands: [UInt64]) {
        emit(abbrevID, abbrevLen)
        guard let encodings = abbrevDefs[abbrevID] else {
            // Fallback: VBR6 for all (shouldn't happen if abbreviation was defined)
            for op in operands {
                emitVBR(op, 6)
            }
            return
        }
        var opIdx = 0
        for (encIdx, enc) in encodings.enumerated() {
            if enc.type == 3 {
                // Array: next encoding is the element encoding, consume remaining operands
                let elemEnc = encodings[encIdx + 1]
                let remaining = operands.count - opIdx
                emitVBR(UInt64(remaining), 6)  // array count
                while opIdx < operands.count {
                    let val = operands[opIdx]
                    switch elemEnc.type {
                    case 1: emit(val, Int(elemEnc.data))
                    case 2: emitVBR(val, Int(elemEnc.data))
                    case 4: emit(val, 6)
                    default: emitVBR(val, 6)
                    }
                    opIdx += 1
                }
                return  // Array consumes everything remaining
            }
            let val = opIdx < operands.count ? operands[opIdx] : 0
            switch enc.type {
            case 1: // Fixed
                emit(val, Int(enc.data))
            case 2: // VBR
                emitVBR(val, Int(enc.data))
            case 4: // Char6
                emit(val, 6)
            default:
                emitVBR(val, 6)
            }
            opIdx += 1
        }
    }

    /// Emit a DEFINE_ABBREV record.
    /// `operandEncodings` is a list of (encoding_type, optional_data) pairs.
    ///
    /// Encoding types:
    /// - 1: Fixed(N) - N bits
    /// - 2: VBR(N) - VBR with N-bit chunks
    /// - 3: Array - followed by element encoding
    /// - 4: Char6 - 6-bit char
    /// - 5: Blob - raw bytes
    func emitDefineAbbrev(operandEncodings: [(type: UInt8, data: UInt64?)]) {
        let thisID = nextAbbrevID
        nextAbbrevID += 1

        emit(Self.defineAbbrevAbbrevID, abbrevLen)
        emitVBR(UInt64(operandEncodings.count), 5)

        // Collect non-literal encodings for emitAbbreviatedRecord
        var nonLiterals: [(type: UInt8, data: UInt64)] = []

        for (encType, encData) in operandEncodings {
            if encType == 0 {
                // Literal encoding
                emitBit(1)
                emitVBR(encData ?? 0, 8)
                // Literals are not passed as operands — skip
            } else {
                emitBit(0)
                emit(UInt64(encType), 3)
                if encType == 1 || encType == 2 {
                    emitVBR(encData ?? 0, 5)
                }
                nonLiterals.append((type: encType, data: encData ?? 0))
            }
        }

        abbrevDefs[thisID] = nonLiterals
    }

    /// Emit a record using a defined abbreviation with a trailing blob.
    func emitAbbreviatedRecordWithBlob(abbrevID: UInt64, operands: [UInt64], blob: [UInt8]) {
        emit(abbrevID, abbrevLen)
        for op in operands {
            emitVBR(op, 6)
        }
        // Blob: emit length as VBR6, align to 32 bits, emit raw bytes, align again
        emitVBR(UInt64(blob.count), 6)
        alignTo32Bits()
        emitRawBytes(blob)
        alignTo32Bits()
    }

    // MARK: - Output

    /// Finalize and return the bitstream bytes.
    /// Flushes any remaining partial byte.
    func finalize() -> [UInt8] {
        var result = bytes
        if bitOffset > 0 {
            result.append(currentByte)
        }
        return result
    }
}
