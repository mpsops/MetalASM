/// A token produced by the LLVM IR lexer.
public struct Token {
    public var kind: Kind
    public var text: String
    public var line: Int
    public var column: Int

    public init(kind: Kind, text: String, line: Int = 0, column: Int = 0) {
        self.kind = kind
        self.text = text
        self.line = line
        self.column = column
    }

    public enum Kind: Equatable {
        // Literals
        case integer          // 42, -1, 0xFF
        case float_           // 0x40000000, 2.0
        case string           // "hello"
        case metadataString   // !"hello"
        case label            // entry:, bb0:

        // Identifiers
        case localIdent       // %foo, %0
        case globalIdent      // @foo
        case metadataIdent    // !foo, !0
        case attrGroupRef     // #0, #1
        case comdatRef        // $foo
        case typeIdent        // %struct.name

        // Keywords (a subset - the parser handles most contextually)
        case keyword          // define, declare, target, etc.

        // Punctuation
        case leftParen        // (
        case rightParen       // )
        case leftBrace        // {
        case rightBrace       // }
        case leftBracket      // [
        case rightBracket     // ]
        case leftAngle        // <
        case rightAngle       // >
        case comma            // ,
        case equals           // =
        case star             // *
        case exclamation      // !
        case dotDotDot        // ...

        // Special
        case newline
        case eof
    }
}
