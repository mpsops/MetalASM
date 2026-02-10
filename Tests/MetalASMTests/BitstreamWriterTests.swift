import XCTest
@testable import MetalASM

final class BitstreamWriterTests: XCTestCase {

    func testEmitBit() {
        let w = BitstreamWriter()
        // Emit bits: 1, 0, 1, 1, 0, 0, 0, 0 = 0b00001101 = 0x0D
        w.emitBit(1)
        w.emitBit(0)
        w.emitBit(1)
        w.emitBit(1)
        w.emitBit(0)
        w.emitBit(0)
        w.emitBit(0)
        w.emitBit(0)
        let bytes = w.finalize()
        XCTAssertEqual(bytes, [0x0D])
    }

    func testEmitMultiBit() {
        let w = BitstreamWriter()
        // Emit 0xAB (8 bits) LSB first
        w.emit(0xAB, 8)
        let bytes = w.finalize()
        XCTAssertEqual(bytes, [0xAB])
    }

    func testEmitVBR() {
        let w = BitstreamWriter()
        // VBR4: value 5
        // 5 in binary = 101
        // VBR4: 3 data bits + 1 continuation bit
        // 5 = 0b101, fits in 3 bits, so: 0101 (no continuation)
        w.emitVBR(5, 4)
        w.alignTo32Bits()
        let bytes = w.finalize()
        // Bits: 1010 + padding to 32
        // 0101 = 5 in 4 bits, LSB first
        XCTAssertEqual(bytes[0] & 0xF, 5) // first 4 bits
    }

    func testVBRMultiChunk() {
        let w = BitstreamWriter()
        // VBR4 for value 15
        // 15 = 0b1111 = needs 4 data bits, VBR4 has 3 data bits per chunk
        // Chunk 1: 111 (data) + 1 (continuation) = bits 1111 (LSB first)
        // Chunk 2: 001 (data) + 0 (no continuation) = bits 1000
        // Total 8 bits LSB first: 1111_1000 = 0x1F
        // Wait let's recount: data=111, cont=1 → chunk=1111, then data=001, cont=0 → chunk=0010
        // LSB first: chunk1 bits [3:0] = 1111, chunk2 bits [7:4] = 0010
        // byte = 0010_1111 = 0x2F... but emit is LSB first within byte
        // Actually each chunk is emitted as numBits=4 bits LSB first:
        // chunk1 = 0b1111 emitted as bits 1,1,1,1
        // chunk2 = 0b0010 emitted as bits 0,1,0,0
        // byte bits: [0]=1 [1]=1 [2]=1 [3]=1 [4]=0 [5]=1 [6]=0 [7]=0
        // byte = 0b00101111 = 0x2F... no that's wrong
        // bit[0]=1, bit[1]=1, bit[2]=1, bit[3]=1, bit[4]=0, bit[5]=1, bit[6]=0, bit[7]=0
        // = 2^0 + 2^1 + 2^2 + 2^3 + 2^5 = 1+2+4+8+32 = 47 = 0x2F? No, 47 = 0x2F.
        // Actually 1+2+4+8+32 = 47. But 0x2F = 47. Let me verify:
        // 0x2F = 0010_1111 = 32+8+4+2+1 = 47. Wait that's 32+15=47.
        // Our output was 31. Let's see: chunk1 has value bits=111=7, cont=1 → emit(15, 4)
        // emit(15, 4): 15=0b1111, 4 bits LSB first = bits 1,1,1,1
        // chunk2 has value bits=001=1, cont=0 → emit(2, 4)
        // emit(2, 4): 2=0b0010, 4 bits LSB first = bits 0,1,0,0
        // Combined byte: bits 1111_0100 = 0x4F? No...
        // Hmm let me just verify empirically
        w.emitVBR(15, 4)
        w.alignTo32Bits()
        let bytes = w.finalize()
        // Just check the value decoded back gives 15
        // First chunk: low 4 bits of byte[0]
        let chunk1 = bytes[0] & 0xF  // should have continuation bit set (bit 3)
        XCTAssertTrue(chunk1 & 0x8 != 0, "First chunk should have continuation bit")
        let data1 = chunk1 & 0x7
        // Second chunk
        let chunk2 = (bytes[0] >> 4) & 0xF
        let data2 = chunk2 & 0x7
        let reconstructed = Int(data1) | (Int(data2) << 3)
        XCTAssertEqual(reconstructed, 15)
    }

    func testBlockEnterExit() {
        let w = BitstreamWriter()
        // Emit magic
        w.emitBitcodeMagic()
        // Enter a block
        w.enterSubblock(blockID: 8, abbrevLen: 4)
        // Emit a simple record
        w.emitUnabbrevRecord(code: 1, operands: [42])
        // Exit block
        w.exitBlock()
        let bytes = w.finalize()

        // First 4 bytes should be BC magic
        XCTAssertEqual(bytes[0], 0x42) // B
        XCTAssertEqual(bytes[1], 0x43) // C
        XCTAssertEqual(bytes[2], 0xC0)
        XCTAssertEqual(bytes[3], 0xDE)

        // The rest should be valid bitstream
        XCTAssertTrue(bytes.count > 4)
        // Should be 32-bit aligned
        XCTAssertEqual(bytes.count % 4, 0)
    }

    func testUnabbrevRecord() {
        let w = BitstreamWriter()
        w.emitBitcodeMagic()
        w.enterSubblock(blockID: 8, abbrevLen: 4)
        // Record with code=2, operands=[100, 200]
        w.emitUnabbrevRecord(code: 2, operands: [100, 200])
        w.exitBlock()
        let bytes = w.finalize()
        XCTAssertTrue(bytes.count > 4)
        XCTAssertEqual(bytes.count % 4, 0)
    }

    func testAlignTo32() {
        let w = BitstreamWriter()
        w.emit(0xFF, 8)  // 8 bits
        w.emit(0xFF, 8)  // 16 bits
        w.emit(0xFF, 3)  // 19 bits
        w.alignTo32Bits() // should pad to 32
        let bytes = w.finalize()
        XCTAssertEqual(bytes.count, 4) // 32 bits = 4 bytes
    }

    func testNestedBlocks() {
        let w = BitstreamWriter()
        w.emitBitcodeMagic()
        w.enterSubblock(blockID: 8, abbrevLen: 4)
        w.enterSubblock(blockID: 17, abbrevLen: 4) // TYPE_BLOCK
        w.emitUnabbrevRecord(code: 1, operands: [5]) // NUMENTRY = 5
        w.exitBlock()
        w.exitBlock()
        let bytes = w.finalize()
        XCTAssertEqual(bytes.count % 4, 0)
    }
}
