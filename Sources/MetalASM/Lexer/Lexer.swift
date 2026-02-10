/// Tokenizer for LLVM IR text.
///
/// Converts a string of LLVM IR text into a sequence of `Token` values.
/// Handles identifiers, keywords, numbers, strings, punctuation, and comments.
public final class Lexer {
    private let source: [Character]
    private var pos: Int = 0
    private var line: Int = 1
    private var column: Int = 1

    public init(source: String) {
        self.source = Array(source)
    }

    // MARK: - Public API

    /// Tokenize the entire source into an array of tokens.
    public func tokenize() -> [Token] {
        var tokens: [Token] = []
        while true {
            let tok = nextToken()
            tokens.append(tok)
            if tok.kind == .eof { break }
        }
        return tokens
    }

    // MARK: - Token production

    /// Return the next token.
    func nextToken() -> Token {
        skipWhitespaceAndComments()

        guard pos < source.count else {
            return Token(kind: .eof, text: "", line: line, column: column)
        }

        let startLine = line
        let startCol = column
        let ch = source[pos]

        switch ch {
        case "\n":
            advance()
            return Token(kind: .newline, text: "\n", line: startLine, column: startCol)

        case "(":
            advance()
            return Token(kind: .leftParen, text: "(", line: startLine, column: startCol)
        case ")":
            advance()
            return Token(kind: .rightParen, text: ")", line: startLine, column: startCol)
        case "{":
            advance()
            return Token(kind: .leftBrace, text: "{", line: startLine, column: startCol)
        case "}":
            advance()
            return Token(kind: .rightBrace, text: "}", line: startLine, column: startCol)
        case "[":
            advance()
            return Token(kind: .leftBracket, text: "[", line: startLine, column: startCol)
        case "]":
            advance()
            return Token(kind: .rightBracket, text: "]", line: startLine, column: startCol)
        case "<":
            advance()
            return Token(kind: .leftAngle, text: "<", line: startLine, column: startCol)
        case ">":
            advance()
            return Token(kind: .rightAngle, text: ">", line: startLine, column: startCol)
        case ",":
            advance()
            return Token(kind: .comma, text: ",", line: startLine, column: startCol)
        case "=":
            advance()
            return Token(kind: .equals, text: "=", line: startLine, column: startCol)
        case "*":
            advance()
            return Token(kind: .star, text: "*", line: startLine, column: startCol)

        case ".":
            // Check for ...
            if pos + 2 < source.count && source[pos+1] == "." && source[pos+2] == "." {
                advance(); advance(); advance()
                return Token(kind: .dotDotDot, text: "...", line: startLine, column: startCol)
            }
            // Otherwise, part of a float or identifier
            return lexNumber(startLine: startLine, startCol: startCol)

        case "\"":
            return lexString(startLine: startLine, startCol: startCol)

        case "@":
            return lexGlobalIdent(startLine: startLine, startCol: startCol)

        case "%":
            return lexLocalIdent(startLine: startLine, startCol: startCol)

        case "!":
            return lexMetadata(startLine: startLine, startCol: startCol)

        case "#":
            return lexAttrGroupRef(startLine: startLine, startCol: startCol)

        case "$":
            return lexComdatRef(startLine: startLine, startCol: startCol)

        case "-", "0"..."9":
            return lexNumber(startLine: startLine, startCol: startCol)

        case "c":
            // Could be c"string" (constant string) or keyword
            if pos + 1 < source.count && source[pos+1] == "\"" {
                advance() // skip 'c'
                let str = lexString(startLine: startLine, startCol: startCol)
                return Token(kind: .string, text: "c" + str.text, line: startLine, column: startCol)
            }
            return lexKeywordOrIdent(startLine: startLine, startCol: startCol)

        default:
            if ch.isLetter || ch == "_" {
                return lexKeywordOrIdent(startLine: startLine, startCol: startCol)
            }
            // Unknown character - skip it
            advance()
            return Token(kind: .keyword, text: String(ch), line: startLine, column: startCol)
        }
    }

    // MARK: - Lexing helpers

    private func skipWhitespaceAndComments() {
        while pos < source.count {
            let ch = source[pos]
            if ch == " " || ch == "\t" || ch == "\r" {
                advance()
            } else if ch == ";" {
                // Line comment - skip to end of line
                while pos < source.count && source[pos] != "\n" {
                    advance()
                }
            } else {
                break
            }
        }
    }

    private func lexString(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "\"")
        advance() // skip opening quote
        var text = "\""
        while pos < source.count && source[pos] != "\"" {
            if source[pos] == "\\" {
                text.append(source[pos])
                advance()
                if pos < source.count {
                    text.append(source[pos])
                    advance()
                }
            } else {
                text.append(source[pos])
                advance()
            }
        }
        if pos < source.count {
            text.append("\"")
            advance() // skip closing quote
        }
        return Token(kind: .string, text: text, line: startLine, column: startCol)
    }

    private func lexGlobalIdent(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "@")
        advance() // skip @
        var text = "@"

        if pos < source.count && source[pos] == "\"" {
            // Quoted name: @"name"
            let str = lexString(startLine: startLine, startCol: startCol)
            text += str.text
        } else {
            // Unquoted name
            while pos < source.count && isIdentChar(source[pos]) {
                text.append(source[pos])
                advance()
            }
        }
        return Token(kind: .globalIdent, text: text, line: startLine, column: startCol)
    }

    private func lexLocalIdent(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "%")
        advance() // skip %
        var text = "%"

        if pos < source.count && source[pos] == "\"" {
            let str = lexString(startLine: startLine, startCol: startCol)
            text += str.text
        } else {
            while pos < source.count && isIdentChar(source[pos]) {
                text.append(source[pos])
                advance()
            }
        }
        return Token(kind: .localIdent, text: text, line: startLine, column: startCol)
    }

    private func lexMetadata(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "!")
        advance() // skip !

        if pos < source.count && source[pos] == "\"" {
            // Metadata string: !"string"
            let str = lexString(startLine: startLine, startCol: startCol)
            return Token(kind: .metadataString, text: "!" + str.text, line: startLine, column: startCol)
        } else if pos < source.count && source[pos] == "{" {
            // !{ is just ! followed by {
            return Token(kind: .exclamation, text: "!", line: startLine, column: startCol)
        } else {
            // Metadata identifier: !name or !0
            var text = "!"
            while pos < source.count && isIdentChar(source[pos]) {
                text.append(source[pos])
                advance()
            }
            return Token(kind: .metadataIdent, text: text, line: startLine, column: startCol)
        }
    }

    private func lexAttrGroupRef(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "#")
        advance()
        var text = "#"
        while pos < source.count && source[pos].isNumber {
            text.append(source[pos])
            advance()
        }
        return Token(kind: .attrGroupRef, text: text, line: startLine, column: startCol)
    }

    private func lexComdatRef(startLine: Int, startCol: Int) -> Token {
        assert(source[pos] == "$")
        advance()
        var text = "$"
        while pos < source.count && isIdentChar(source[pos]) {
            text.append(source[pos])
            advance()
        }
        return Token(kind: .comdatRef, text: text, line: startLine, column: startCol)
    }

    private func lexNumber(startLine: Int, startCol: Int) -> Token {
        var text = ""

        // Handle negative sign
        if pos < source.count && source[pos] == "-" {
            text.append("-")
            advance()
        }

        // Check for hex: 0x, 0xH (half), 0xK (fp80), 0xL (fp128), 0xM (ppc_fp128)
        if pos + 1 < source.count && source[pos] == "0" && (source[pos+1] == "x" || source[pos+1] == "X") {
            text.append(source[pos]); advance()
            text.append(source[pos]); advance()
            // LLVM IR uses 0xH for half, 0xK for fp80, 0xL for fp128, 0xM for ppc_fp128
            var isTypedHexFloat = false
            if pos < source.count {
                let ch = source[pos]
                if ch == "H" || ch == "K" || ch == "L" || ch == "M" {
                    text.append(ch); advance()
                    isTypedHexFloat = true
                }
            }
            while pos < source.count && isHexChar(source[pos]) {
                text.append(source[pos])
                advance()
            }
            let kind: Token.Kind = isTypedHexFloat ? .float_ : (text.contains(".") ? .float_ : .integer)
            return Token(kind: kind, text: text, line: startLine, column: startCol)
        }

        // Decimal number, possibly float
        var isFloat = false
        while pos < source.count && (source[pos].isNumber || source[pos] == "." || source[pos] == "e" || source[pos] == "E") {
            if source[pos] == "." || source[pos] == "e" || source[pos] == "E" {
                isFloat = true
            }
            text.append(source[pos])
            advance()
        }

        return Token(kind: isFloat ? .float_ : .integer, text: text, line: startLine, column: startCol)
    }

    private func lexKeywordOrIdent(startLine: Int, startCol: Int) -> Token {
        var text = ""
        while pos < source.count && isIdentChar(source[pos]) {
            text.append(source[pos])
            advance()
        }

        // Check if this is a label (followed by ':')
        if pos < source.count && source[pos] == ":" {
            advance()
            return Token(kind: .label, text: text + ":", line: startLine, column: startCol)
        }

        return Token(kind: .keyword, text: text, line: startLine, column: startCol)
    }

    // MARK: - Character classification

    private func isIdentChar(_ ch: Character) -> Bool {
        return ch.isLetter || ch.isNumber || ch == "_" || ch == "." || ch == "$" || ch == "-"
    }

    private func isHexChar(_ ch: Character) -> Bool {
        return ch.isHexDigit
    }

    private func advance() {
        if pos < source.count {
            if source[pos] == "\n" {
                line += 1
                column = 1
            } else {
                column += 1
            }
            pos += 1
        }
    }

    private func peek() -> Character? {
        return pos < source.count ? source[pos] : nil
    }
}
