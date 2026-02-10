import XCTest
@testable import MetalASM

final class LexerTests: XCTestCase {

    func testBasicTokens() {
        let source = "define void @main() {\n  ret void\n}\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()

        // Filter out newlines and eof for easier testing
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].kind, .keyword)
        XCTAssertEqual(significant[0].text, "define")

        XCTAssertEqual(significant[1].kind, .keyword)
        XCTAssertEqual(significant[1].text, "void")

        XCTAssertEqual(significant[2].kind, .globalIdent)
        XCTAssertEqual(significant[2].text, "@main")

        XCTAssertEqual(significant[3].kind, .leftParen)
        XCTAssertEqual(significant[4].kind, .rightParen)
        XCTAssertEqual(significant[5].kind, .leftBrace)
        XCTAssertEqual(significant[6].kind, .keyword)
        XCTAssertEqual(significant[6].text, "ret")
        XCTAssertEqual(significant[7].kind, .keyword)
        XCTAssertEqual(significant[7].text, "void")
        XCTAssertEqual(significant[8].kind, .rightBrace)
    }

    func testComments() {
        let source = "; this is a comment\ndefine void @f()\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }
        XCTAssertEqual(significant[0].text, "define")
    }

    func testMetadataTokens() {
        let source = "!0 = !{!\"hello\", i32 42}\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].kind, .metadataIdent)
        XCTAssertEqual(significant[0].text, "!0")
        XCTAssertEqual(significant[1].kind, .equals)
        XCTAssertEqual(significant[2].kind, .exclamation)
        XCTAssertEqual(significant[3].kind, .leftBrace)
        XCTAssertEqual(significant[4].kind, .metadataString)
        XCTAssertEqual(significant[4].text, "!\"hello\"")
    }

    func testTypes() {
        let source = "float addrspace(1)* i32 [128 x float]\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].text, "float")
        XCTAssertEqual(significant[1].text, "addrspace")
        XCTAssertEqual(significant[2].kind, .leftParen)
        XCTAssertEqual(significant[3].text, "1")
        XCTAssertEqual(significant[4].kind, .rightParen)
        XCTAssertEqual(significant[5].kind, .star)
    }

    func testLocalAndGlobalIdents() {
        let source = "%ev = alloca %event_t addrspace(3)*, @global_var\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].kind, .localIdent)
        XCTAssertEqual(significant[0].text, "%ev")
        XCTAssertEqual(significant[1].kind, .equals)
        XCTAssertEqual(significant[2].text, "alloca")
    }

    func testAttrGroupRef() {
        let source = "attributes #0 = { nounwind }\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].text, "attributes")
        XCTAssertEqual(significant[1].kind, .attrGroupRef)
        XCTAssertEqual(significant[1].text, "#0")
    }

    func testNumbers() {
        let source = "42 -1 0xFF 2.0\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].kind, .integer)
        XCTAssertEqual(significant[0].text, "42")
        XCTAssertEqual(significant[1].kind, .integer)
        XCTAssertEqual(significant[1].text, "-1")
        XCTAssertEqual(significant[2].kind, .integer)
        XCTAssertEqual(significant[2].text, "0xFF")
        XCTAssertEqual(significant[3].kind, .float_)
        XCTAssertEqual(significant[3].text, "2.0")
    }

    func testStringTokens() {
        let source = "\"hello world\" \"air-buffer-no-alias\"\n"
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        let significant = tokens.filter { $0.kind != .newline && $0.kind != .eof }

        XCTAssertEqual(significant[0].kind, .string)
        XCTAssertEqual(significant[0].text, "\"hello world\"")
    }

    func testMonolithicIR() throws {
        // Test lexing the actual monolithic IR file
        let irURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("Reference IR file not found")
        }
        let source = try String(contentsOf: irURL, encoding: .utf8)
        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()

        // Should have a reasonable number of tokens
        XCTAssertGreaterThan(tokens.count, 100)

        // Should end with EOF
        XCTAssertEqual(tokens.last?.kind, .eof)

        // Should contain key identifiers
        let texts = Set(tokens.map { $0.text })
        XCTAssertTrue(texts.contains("define"))
        XCTAssertTrue(texts.contains("declare"))
        XCTAssertTrue(texts.contains("@monolithic_kernel"))
        XCTAssertTrue(texts.contains("ret"))
    }
}
