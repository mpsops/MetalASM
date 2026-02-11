/// Tokenizer for LLVM IR text.
///
/// Converts a string of LLVM IR text into a sequence of `Token` values.
/// Operates on raw UTF-8 bytes for performance — IR is pure ASCII.
public final class Lexer {
    private let source: [UInt8]
    private var pos: Int = 0

    public init(source: String) {
        self.source = Array(source.utf8)
    }

    // Byte constants
    private static let NL:    UInt8 = 0x0A  // \n
    private static let CR:    UInt8 = 0x0D  // \r
    private static let TAB:   UInt8 = 0x09  // \t
    private static let SPACE: UInt8 = 0x20  //
    private static let SEMI:  UInt8 = 0x3B  // ;
    private static let QUOTE: UInt8 = 0x22  // "
    private static let BSLASH:UInt8 = 0x5C  // \
    private static let AT:    UInt8 = 0x40  // @
    private static let PCT:   UInt8 = 0x25  // %
    private static let BANG:  UInt8 = 0x21  // !
    private static let HASH:  UInt8 = 0x23  // #
    private static let DOLLAR:UInt8 = 0x24  // $
    private static let LPAREN:UInt8 = 0x28  // (
    private static let RPAREN:UInt8 = 0x29  // )
    private static let LBRACE:UInt8 = 0x7B  // {
    private static let RBRACE:UInt8 = 0x7D  // }
    private static let LBRACK:UInt8 = 0x5B  // [
    private static let RBRACK:UInt8 = 0x5D  // ]
    private static let LANGLE:UInt8 = 0x3C  // <
    private static let RANGLE:UInt8 = 0x3E  // >
    private static let COMMA: UInt8 = 0x2C  // ,
    private static let EQ:    UInt8 = 0x3D  // =
    private static let STAR:  UInt8 = 0x2A  // *
    private static let DOT:   UInt8 = 0x2E  // .
    private static let MINUS: UInt8 = 0x2D  // -
    private static let USCORE:UInt8 = 0x5F  // _
    private static let COLON: UInt8 = 0x3A  // :
    private static let _0:    UInt8 = 0x30
    private static let _9:    UInt8 = 0x39
    private static let _a:    UInt8 = 0x61
    private static let _c:    UInt8 = 0x63
    private static let _e:    UInt8 = 0x65
    private static let _f:    UInt8 = 0x66
    private static let _x:    UInt8 = 0x78
    private static let _z:    UInt8 = 0x7A
    private static let _A:    UInt8 = 0x41
    private static let _E:    UInt8 = 0x45
    private static let _F:    UInt8 = 0x46
    private static let _H:    UInt8 = 0x48
    private static let _K:    UInt8 = 0x4B
    private static let _L:    UInt8 = 0x4C
    private static let _M:    UInt8 = 0x4D
    private static let _X:    UInt8 = 0x58
    private static let _Z:    UInt8 = 0x5A

    @inline(__always)
    private static func isDigit(_ b: UInt8) -> Bool { b >= _0 && b <= _9 }

    @inline(__always)
    private static func isLetter(_ b: UInt8) -> Bool {
        (b >= _a && b <= _z) || (b >= _A && b <= _Z)
    }

    @inline(__always)
    private static func isHexDigit(_ b: UInt8) -> Bool {
        isDigit(b) || (b >= _a && b <= _f) || (b >= _A && b <= _F)
    }

    @inline(__always)
    private static func isIdentChar(_ b: UInt8) -> Bool {
        isLetter(b) || isDigit(b) || b == USCORE || b == DOT || b == DOLLAR || b == MINUS
    }

    private var count: Int { source.count }

    @inline(__always)
    private func byte(_ i: Int) -> UInt8 { source[i] }

    // Build String from byte range
    @inline(__always)
    private func text(_ start: Int, _ end: Int) -> String {
        String(decoding: source[start..<end], as: UTF8.self)
    }

    // MARK: - Public API

    /// Tokenize the entire source into an array of tokens.
    public func tokenize() -> [Token] {
        var tokens: [Token] = []
        tokens.reserveCapacity(count / 6)  // rough estimate
        while true {
            let tok = nextToken()
            tokens.append(tok)
            if tok.kind == .eof { break }
        }
        return tokens
    }

    // MARK: - Token production

    func nextToken() -> Token {
        skipWhitespaceAndComments()

        guard pos < count else {
            return Token(kind: .eof, text: "")
        }

        let ch = byte(pos)

        switch ch {
        case Self.NL:
            pos += 1
            return Token(kind: .newline, text: "\n")

        case Self.LPAREN:  pos += 1; return Token(kind: .leftParen, text: "(")
        case Self.RPAREN:  pos += 1; return Token(kind: .rightParen, text: ")")
        case Self.LBRACE:  pos += 1; return Token(kind: .leftBrace, text: "{")
        case Self.RBRACE:  pos += 1; return Token(kind: .rightBrace, text: "}")
        case Self.LBRACK:  pos += 1; return Token(kind: .leftBracket, text: "[")
        case Self.RBRACK:  pos += 1; return Token(kind: .rightBracket, text: "]")
        case Self.LANGLE:  pos += 1; return Token(kind: .leftAngle, text: "<")
        case Self.RANGLE:  pos += 1; return Token(kind: .rightAngle, text: ">")
        case Self.COMMA:   pos += 1; return Token(kind: .comma, text: ",")
        case Self.EQ:      pos += 1; return Token(kind: .equals, text: "=")
        case Self.STAR:    pos += 1; return Token(kind: .star, text: "*")

        case Self.DOT:
            if pos + 2 < count && byte(pos+1) == Self.DOT && byte(pos+2) == Self.DOT {
                pos += 3
                return Token(kind: .dotDotDot, text: "...")
            }
            return lexNumber()

        case Self.QUOTE:
            return lexString()

        case Self.AT:
            return lexGlobalIdent()

        case Self.PCT:
            return lexLocalIdent()

        case Self.BANG:
            return lexMetadata()

        case Self.HASH:
            return lexAttrGroupRef()

        case Self.DOLLAR:
            return lexComdatRef()

        case Self.MINUS, Self._0...Self._9:
            return lexNumber()

        case Self._c:
            if pos + 1 < count && byte(pos+1) == Self.QUOTE {
                pos += 1  // skip 'c'
                let str = lexString()
                return Token(kind: .string, text: "c" + str.text)
            }
            return lexKeywordOrIdent()

        default:
            if Self.isLetter(ch) || ch == Self.USCORE {
                return lexKeywordOrIdent()
            }
            pos += 1
            return Token(kind: .keyword, text: text(pos-1, pos))
        }
    }

    // MARK: - Lexing helpers

    private func skipWhitespaceAndComments() {
        while pos < count {
            let ch = byte(pos)
            if ch == Self.SPACE || ch == Self.TAB || ch == Self.CR {
                pos += 1
            } else if ch == Self.SEMI {
                pos += 1
                while pos < count && byte(pos) != Self.NL {
                    pos += 1
                }
            } else {
                break
            }
        }
    }

    private func lexString() -> Token {
        let start = pos
        pos += 1  // skip opening quote
        while pos < count && byte(pos) != Self.QUOTE {
            if byte(pos) == Self.BSLASH {
                pos += 1
                if pos < count { pos += 1 }
            } else {
                pos += 1
            }
        }
        if pos < count {
            pos += 1  // skip closing quote
        }
        return Token(kind: .string, text: text(start, pos))
    }

    private func lexGlobalIdent() -> Token {
        let start = pos
        pos += 1  // skip @
        if pos < count && byte(pos) == Self.QUOTE {
            _ = lexString()  // advances past the quoted name
            return Token(kind: .globalIdent, text: text(start, pos))
        }
        while pos < count && Self.isIdentChar(byte(pos)) {
            pos += 1
        }
        return Token(kind: .globalIdent, text: text(start, pos))
    }

    private func lexLocalIdent() -> Token {
        let start = pos
        pos += 1  // skip %
        if pos < count && byte(pos) == Self.QUOTE {
            _ = lexString()
            return Token(kind: .localIdent, text: text(start, pos))
        }
        while pos < count && Self.isIdentChar(byte(pos)) {
            pos += 1
        }
        return Token(kind: .localIdent, text: text(start, pos))
    }

    private func lexMetadata() -> Token {
        let start = pos
        pos += 1  // skip !

        if pos < count && byte(pos) == Self.QUOTE {
            _ = lexString()
            return Token(kind: .metadataString, text: text(start, pos))
        } else if pos < count && byte(pos) == Self.LBRACE {
            return Token(kind: .exclamation, text: "!")
        } else {
            while pos < count && Self.isIdentChar(byte(pos)) {
                pos += 1
            }
            return Token(kind: .metadataIdent, text: text(start, pos))
        }
    }

    private func lexAttrGroupRef() -> Token {
        let start = pos
        pos += 1  // skip #
        while pos < count && Self.isDigit(byte(pos)) {
            pos += 1
        }
        return Token(kind: .attrGroupRef, text: text(start, pos))
    }

    private func lexComdatRef() -> Token {
        let start = pos
        pos += 1  // skip $
        while pos < count && Self.isIdentChar(byte(pos)) {
            pos += 1
        }
        return Token(kind: .comdatRef, text: text(start, pos))
    }

    private func lexNumber() -> Token {
        let start = pos

        // Handle negative sign
        if pos < count && byte(pos) == Self.MINUS {
            pos += 1
        }

        // Check for hex: 0x
        if pos + 1 < count && byte(pos) == Self._0 && (byte(pos+1) == Self._x || byte(pos+1) == Self._X) {
            pos += 2
            var isTypedHexFloat = false
            if pos < count {
                let ch = byte(pos)
                if ch == Self._H || ch == Self._K || ch == Self._L || ch == Self._M {
                    pos += 1
                    isTypedHexFloat = true
                }
            }
            while pos < count && Self.isHexDigit(byte(pos)) {
                pos += 1
            }
            let kind: Token.Kind = isTypedHexFloat ? .float_ : .integer
            return Token(kind: kind, text: text(start, pos))
        }

        // Decimal number, possibly float
        var isFloat = false
        while pos < count {
            let ch = byte(pos)
            if Self.isDigit(ch) {
                pos += 1
            } else if ch == Self.DOT || ch == Self._e || ch == Self._E {
                isFloat = true
                pos += 1
            } else {
                break
            }
        }

        return Token(kind: isFloat ? .float_ : .integer, text: text(start, pos))
    }

    private func lexKeywordOrIdent() -> Token {
        let start = pos
        while pos < count && Self.isIdentChar(byte(pos)) {
            pos += 1
        }

        // Check if this is a label (followed by ':')
        if pos < count && byte(pos) == Self.COLON {
            pos += 1
            return Token(kind: .label, text: text(start, pos))
        }

        return Token(kind: .keyword, text: text(start, pos))
    }
}
