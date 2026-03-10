/// Tokenizer for LLVM IR text.
///
/// Converts a string of LLVM IR text into a sequence of `Token` values.
/// Operates on raw UTF-8 bytes for performance — IR is pure ASCII.
/// Uses UnsafeBufferPointer internally to eliminate bounds checking in hot loops.
public final class Lexer {
    public let source: [UInt8]
    private let count: Int

    public init(source: String) {
        self.source = Array(source.utf8)
        self.count = self.source.count
    }

    /// Build String from byte range in the source buffer.
    @inline(__always)
    public func text(_ start: Int, _ end: Int) -> String {
        String(decoding: source[start..<end], as: UTF8.self)
    }

    /// Get the text of a token.
    @inline(__always)
    public func text(_ token: Token) -> String {
        String(decoding: source[token.start..<token.end], as: UTF8.self)
    }

    // MARK: - Lookup table for identifier characters

    /// 256-byte lookup table: 1 if byte is valid in an identifier, 0 otherwise.
    /// Valid: [a-zA-Z0-9_.$-]
    private static let identTable: [UInt8] = {
        var t = [UInt8](repeating: 0, count: 256)
        for b in UInt8(ascii: "a")...UInt8(ascii: "z") { t[Int(b)] = 1 }
        for b in UInt8(ascii: "A")...UInt8(ascii: "Z") { t[Int(b)] = 1 }
        for b in UInt8(ascii: "0")...UInt8(ascii: "9") { t[Int(b)] = 1 }
        t[Int(UInt8(ascii: "_"))] = 1
        t[Int(UInt8(ascii: "."))] = 1
        t[Int(UInt8(ascii: "$"))] = 1
        t[Int(UInt8(ascii: "-"))] = 1
        return t
    }()

    // MARK: - Public API

    /// Tokenize the entire source into an array of tokens.
    public func tokenize() -> [Token] {
        source.withUnsafeBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return [] }
            var tokens: [Token] = []
            tokens.reserveCapacity(count / 6)
            var pos = 0
            let count = self.count
            let ident = Self.identTable

            while true {
                let tok = Self.nextToken(ptr: ptr, pos: &pos, count: count, ident: ident)
                tokens.append(tok)
                if tok.kind == .eof { break }
            }
            return tokens
        }
    }

    // MARK: - Token production (all static, operating on raw pointer)

    @inline(__always)
    private static func nextToken(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int,
        ident: [UInt8]
    ) -> Token {
        skipWhitespaceAndComments(ptr: ptr, pos: &pos, count: count)

        guard pos < count else {
            return Token.eof
        }

        let ch = ptr[pos]

        switch ch {
        case 0x0A: // \n
            pos += 1
            return Token(kind: .newline, start: pos - 1, end: pos)

        case 0x28: pos += 1; return Token(kind: .leftParen, start: pos - 1, end: pos)   // (
        case 0x29: pos += 1; return Token(kind: .rightParen, start: pos - 1, end: pos)  // )
        case 0x7B: pos += 1; return Token(kind: .leftBrace, start: pos - 1, end: pos)   // {
        case 0x7D: pos += 1; return Token(kind: .rightBrace, start: pos - 1, end: pos)  // }
        case 0x5B: pos += 1; return Token(kind: .leftBracket, start: pos - 1, end: pos) // [
        case 0x5D: pos += 1; return Token(kind: .rightBracket, start: pos - 1, end: pos)// ]
        case 0x3C: pos += 1; return Token(kind: .leftAngle, start: pos - 1, end: pos)   // <
        case 0x3E: pos += 1; return Token(kind: .rightAngle, start: pos - 1, end: pos)  // >
        case 0x2C: pos += 1; return Token(kind: .comma, start: pos - 1, end: pos)       // ,
        case 0x3D: pos += 1; return Token(kind: .equals, start: pos - 1, end: pos)      // =
        case 0x2A: pos += 1; return Token(kind: .star, start: pos - 1, end: pos)        // *

        case 0x2E: // .
            if pos + 2 < count && ptr[pos+1] == 0x2E && ptr[pos+2] == 0x2E {
                let start = pos; pos += 3
                return Token(kind: .dotDotDot, start: start, end: pos)
            }
            return lexNumber(ptr: ptr, pos: &pos, count: count)

        case 0x22: // "
            return lexString(ptr: ptr, pos: &pos, count: count)

        case 0x40: // @
            return lexGlobalIdent(ptr: ptr, pos: &pos, count: count, ident: ident)

        case 0x25: // %
            return lexLocalIdent(ptr: ptr, pos: &pos, count: count, ident: ident)

        case 0x21: // !
            return lexMetadata(ptr: ptr, pos: &pos, count: count, ident: ident)

        case 0x23: // #
            return lexAttrGroupRef(ptr: ptr, pos: &pos, count: count)

        case 0x24: // $
            return lexComdatRef(ptr: ptr, pos: &pos, count: count, ident: ident)

        case 0x2D, 0x30...0x39: // - or 0-9
            return lexNumber(ptr: ptr, pos: &pos, count: count)

        case 0x63: // c
            if pos + 1 < count && ptr[pos+1] == 0x22 {
                let start = pos; pos += 1
                _ = lexString(ptr: ptr, pos: &pos, count: count)
                return Token(kind: .string, start: start, end: pos)
            }
            return lexKeywordOrIdent(ptr: ptr, pos: &pos, count: count, ident: ident)

        default:
            if (ch >= 0x61 && ch <= 0x7A) || (ch >= 0x41 && ch <= 0x5A) || ch == 0x5F {
                return lexKeywordOrIdent(ptr: ptr, pos: &pos, count: count, ident: ident)
            }
            pos += 1
            return Token(kind: .keyword, start: pos-1, end: pos)
        }
    }

    // MARK: - Lexing helpers

    @inline(__always)
    private static func skipWhitespaceAndComments(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int
    ) {
        while pos < count {
            let ch = ptr[pos]
            if ch == 0x20 || ch == 0x09 || ch == 0x0D { // space, tab, CR
                pos += 1
            } else if ch == 0x3B { // ;
                pos += 1
                while pos < count && ptr[pos] != 0x0A { pos += 1 }
            } else {
                break
            }
        }
    }

    @inline(__always)
    @discardableResult
    private static func lexString(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int
    ) -> Token {
        let start = pos
        pos += 1 // skip opening quote
        while pos < count && ptr[pos] != 0x22 {
            if ptr[pos] == 0x5C { // backslash
                pos += 1
                if pos < count { pos += 1 }
            } else {
                pos += 1
            }
        }
        if pos < count { pos += 1 } // skip closing quote
        return Token(kind: .string, start: start, end: pos)
    }

    @inline(__always)
    private static func lexGlobalIdent(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int, ident: [UInt8]
    ) -> Token {
        let start = pos
        pos += 1 // skip @
        if pos < count && ptr[pos] == 0x22 {
            _ = lexString(ptr: ptr, pos: &pos, count: count)
            return Token(kind: .globalIdent, start: start, end: pos)
        }
        while pos < count && ident[Int(ptr[pos])] != 0 { pos += 1 }
        return Token(kind: .globalIdent, start: start, end: pos)
    }

    @inline(__always)
    private static func lexLocalIdent(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int, ident: [UInt8]
    ) -> Token {
        let start = pos
        pos += 1 // skip %
        if pos < count && ptr[pos] == 0x22 {
            _ = lexString(ptr: ptr, pos: &pos, count: count)
            return Token(kind: .localIdent, start: start, end: pos)
        }
        while pos < count && ident[Int(ptr[pos])] != 0 { pos += 1 }
        return Token(kind: .localIdent, start: start, end: pos)
    }

    @inline(__always)
    private static func lexMetadata(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int, ident: [UInt8]
    ) -> Token {
        let start = pos
        pos += 1 // skip !
        if pos < count && ptr[pos] == 0x22 {
            _ = lexString(ptr: ptr, pos: &pos, count: count)
            return Token(kind: .metadataString, start: start, end: pos)
        } else if pos < count && ptr[pos] == 0x7B {
            return Token(kind: .exclamation, start: start, end: start + 1)
        } else {
            while pos < count && ident[Int(ptr[pos])] != 0 { pos += 1 }
            return Token(kind: .metadataIdent, start: start, end: pos)
        }
    }

    @inline(__always)
    private static func lexAttrGroupRef(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int
    ) -> Token {
        let start = pos
        pos += 1 // skip #
        while pos < count && ptr[pos] >= 0x30 && ptr[pos] <= 0x39 { pos += 1 }
        return Token(kind: .attrGroupRef, start: start, end: pos)
    }

    @inline(__always)
    private static func lexComdatRef(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int, ident: [UInt8]
    ) -> Token {
        let start = pos
        pos += 1 // skip $
        while pos < count && ident[Int(ptr[pos])] != 0 { pos += 1 }
        return Token(kind: .comdatRef, start: start, end: pos)
    }

    @inline(__always)
    private static func lexNumber(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int
    ) -> Token {
        let start = pos

        if pos < count && ptr[pos] == 0x2D { pos += 1 } // minus

        // Check for hex: 0x
        if pos + 1 < count && ptr[pos] == 0x30 && (ptr[pos+1] == 0x78 || ptr[pos+1] == 0x58) {
            pos += 2
            var isTypedHexFloat = false
            if pos < count {
                let ch = ptr[pos]
                if ch == 0x48 || ch == 0x4B || ch == 0x4C || ch == 0x4D { // H K L M
                    pos += 1; isTypedHexFloat = true
                }
            }
            while pos < count {
                let ch = ptr[pos]
                if (ch >= 0x30 && ch <= 0x39) || (ch >= 0x61 && ch <= 0x66) || (ch >= 0x41 && ch <= 0x46) {
                    pos += 1
                } else { break }
            }
            return Token(kind: isTypedHexFloat ? .float_ : .integer, start: start, end: pos)
        }

        // Decimal, possibly float
        var isFloat = false
        var prevWasExponent = false
        while pos < count {
            let ch = ptr[pos]
            if ch >= 0x30 && ch <= 0x39 { pos += 1; prevWasExponent = false }
            else if ch == 0x2E { isFloat = true; pos += 1; prevWasExponent = false }
            else if ch == 0x65 || ch == 0x45 { isFloat = true; pos += 1; prevWasExponent = true }
            else if prevWasExponent && (ch == 0x2B || ch == 0x2D) { pos += 1; prevWasExponent = false } // +/- after e/E
            else { break }
        }
        // Check for numeric label (e.g. "65:" in basic block labels)
        if !isFloat && pos < count && ptr[pos] == 0x3A /* ':' */ {
            pos += 1
            return Token(kind: .label, start: start, end: pos)
        }
        return Token(kind: isFloat ? .float_ : .integer, start: start, end: pos)
    }

    @inline(__always)
    private static func lexKeywordOrIdent(
        ptr: UnsafePointer<UInt8>, pos: inout Int, count: Int, ident: [UInt8]
    ) -> Token {
        let start = pos
        while pos < count && ident[Int(ptr[pos])] != 0 { pos += 1 }

        // Check if this is a label (followed by ':')
        if pos < count && ptr[pos] == 0x3A {
            pos += 1
            return Token(kind: .label, start: start, end: pos)
        }
        return Token(kind: .keyword, start: start, end: pos)
    }
}
