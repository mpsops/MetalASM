import XCTest
@testable import MetalASM

final class ParserTests: XCTestCase {

    func testTrivialModule() throws {
        let source = """
        source_filename = "test"
        target datalayout = "e-p:64:64:64"
        target triple = "air64_v28-apple-macosx26.0.0"

        define void @trivial() {
        entry:
          ret void
        }
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertEqual(module.sourceFilename, "test")
        XCTAssertEqual(module.targetTriple, "air64_v28-apple-macosx26.0.0")
        XCTAssertEqual(module.functions.count, 1)
        XCTAssertEqual(module.functions[0].name, "trivial")
        XCTAssertFalse(module.functions[0].isDeclaration)
        XCTAssertEqual(module.functions[0].basicBlocks.count, 1)
        XCTAssertEqual(module.functions[0].basicBlocks[0].instructions.count, 1)
        XCTAssertEqual(module.functions[0].basicBlocks[0].instructions[0].opcode, .ret)
    }

    func testFunctionDeclaration() throws {
        let source = """
        declare void @air.wg.barrier(i32, i32)
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertEqual(module.functions.count, 1)
        XCTAssertEqual(module.functions[0].name, "air.wg.barrier")
        XCTAssertTrue(module.functions[0].isDeclaration)
        XCTAssertEqual(module.functions[0].parameterTypes.count, 2)
    }

    func testGlobalVariable() throws {
        let source = """
        @tg_buf = internal addrspace(3) global [128 x float] undef, align 4
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertEqual(module.globals.count, 1)
        XCTAssertEqual(module.globals[0].name, "tg_buf")
        XCTAssertEqual(module.globals[0].linkage, .internal)
    }

    func testOpaqueStructType() throws {
        let source = """
        %event_t = type opaque
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertEqual(module.structTypes.count, 1)
        XCTAssertEqual(module.structTypes[0].name, "event_t")
        if case .opaque(let name) = module.structTypes[0].type {
            XCTAssertEqual(name, "event_t")
        } else {
            XCTFail("Expected opaque type")
        }
    }

    func testAttributeGroup() throws {
        let source = """
        attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" }
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertEqual(module.attributeGroups.count, 1)
        XCTAssertEqual(module.attributeGroups[0].index, 0)
        XCTAssertTrue(module.attributeGroups[0].attributes.contains(.convergent))
        XCTAssertTrue(module.attributeGroups[0].attributes.contains(.noUnwind))
    }

    func testMetadata() throws {
        let source = """
        !0 = !{i32 42, !"hello"}
        !air.kernel = !{!0}
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        XCTAssertGreaterThanOrEqual(module.metadataNodes.count, 1)
        XCTAssertEqual(module.namedMetadata.count, 1)
        XCTAssertEqual(module.namedMetadata[0].name, "air.kernel")
    }

    func testMonolithicIR() throws {
        let irURL = URL(fileURLWithPath: "/tmp/air-monolithic-test/monolithic.ll")
        guard FileManager.default.fileExists(atPath: irURL.path) else {
            throw XCTSkip("Reference IR file not found")
        }
        let source = try String(contentsOf: irURL, encoding: .utf8)

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        // Verify key properties
        XCTAssertEqual(module.targetTriple, "air64_v28-apple-macosx26.0.0")
        XCTAssertFalse(module.dataLayout.isEmpty)

        // Should have 1 struct type (%event_t = type opaque)
        XCTAssertEqual(module.structTypes.count, 1)

        // Should have 1 global (@tg_buf)
        XCTAssertEqual(module.globals.count, 1)
        XCTAssertEqual(module.globals[0].name, "tg_buf")

        // Should have 6 functions (1 definition + 5 declarations)
        XCTAssertEqual(module.functions.count, 6)

        // The first function should be the kernel definition
        let kernel = module.functions.first { $0.name == "monolithic_kernel" }
        XCTAssertNotNil(kernel)
        XCTAssertFalse(kernel?.isDeclaration ?? true)
        XCTAssertEqual(kernel?.parameterTypes.count, 3)

        // Declarations
        let declarations = module.functions.filter { $0.isDeclaration }
        XCTAssertEqual(declarations.count, 5)

        // Should have metadata
        XCTAssertGreaterThan(module.metadataNodes.count, 0)
        XCTAssertGreaterThan(module.namedMetadata.count, 0)

        // Should have air.kernel named metadata
        let kernelMD = module.namedMetadata.first { $0.name == "air.kernel" }
        XCTAssertNotNil(kernelMD)

        // Should have attribute groups
        XCTAssertGreaterThan(module.attributeGroups.count, 0)
    }

    func testCallInstruction() throws {
        let source = """
        declare void @air.wg.barrier(i32, i32)

        define void @test() {
        entry:
          call void @air.wg.barrier(i32 2, i32 1)
          ret void
        }
        """

        let lexer = Lexer(source: source)
        let tokens = lexer.tokenize()
        var parser = Parser(tokens: tokens)
        let module = try parser.parse()

        let testFn = module.functions.first { $0.name == "test" }
        XCTAssertNotNil(testFn)
        XCTAssertEqual(testFn?.basicBlocks.count, 1)

        let bb = testFn?.basicBlocks[0]
        XCTAssertEqual(bb?.instructions.count, 2)
        XCTAssertEqual(bb?.instructions[0].opcode, .call)
        XCTAssertEqual(bb?.instructions[1].opcode, .ret)
    }
}
