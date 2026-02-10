/// Recursive descent parser for LLVM IR text.
///
/// Parses a token stream into an `IRModule` in-memory representation.
/// Supports the subset of LLVM IR used by Metal AIR, including typed pointers.
public struct Parser {
    private var tokens: [Token]
    private var pos: Int = 0

    /// Named values in the current function scope.
    private var localValues: [String: IRValue] = [:]

    /// Global named values.
    private var globalValues: [String: IRValue] = [:]

    /// Struct type definitions.
    private var structTypes: [String: IRType] = [:]

    public init(tokens: [Token]) {
        self.tokens = tokens
    }

    // MARK: - Error handling

    enum ParseError: Error {
        case unexpected(String, line: Int, column: Int)
        case expected(String, got: String, line: Int, column: Int)
    }

    // MARK: - Token navigation

    private var current: Token {
        guard pos < tokens.count else {
            return Token(kind: .eof, text: "")
        }
        return tokens[pos]
    }

    private mutating func advance() -> Token {
        let tok = current
        pos += 1
        return tok
    }

    private mutating func skipNewlines() {
        while pos < tokens.count && tokens[pos].kind == .newline {
            pos += 1
        }
    }

    private func peek(offset: Int = 0) -> Token {
        let idx = pos + offset
        guard idx < tokens.count else {
            return Token(kind: .eof, text: "")
        }
        return tokens[idx]
    }

    private mutating func expect(_ kind: Token.Kind, _ context: String = "") throws -> Token {
        skipNewlines()
        guard current.kind == kind else {
            throw ParseError.expected(
                "\(kind) \(context)",
                got: "\(current.kind)(\(current.text))",
                line: current.line,
                column: current.column
            )
        }
        return advance()
    }

    private mutating func expectKeyword(_ keyword: String) throws -> Token {
        skipNewlines()
        guard current.kind == .keyword && current.text == keyword else {
            throw ParseError.expected(
                "keyword '\(keyword)'",
                got: current.text,
                line: current.line,
                column: current.column
            )
        }
        return advance()
    }

    private mutating func consumeIfKeyword(_ keyword: String) -> Bool {
        if current.kind == .keyword && current.text == keyword {
            pos += 1
            return true
        }
        return false
    }

    private mutating func consumeIf(_ kind: Token.Kind) -> Bool {
        if current.kind == kind {
            pos += 1
            return true
        }
        return false
    }

    // MARK: - Top-level parsing

    /// Parse the entire module.
    public mutating func parse() throws -> IRModule {
        let module = IRModule()

        while current.kind != .eof {
            skipNewlines()
            if current.kind == .eof { break }

            switch current.kind {
            case .keyword:
                try parseTopLevelKeyword(module)
            case .globalIdent:
                try parseGlobalDefinition(module)
            case .localIdent:
                // Type definition: %name = type ...
                try parseTypeDefinition(module)
            case .metadataIdent:
                try parseNamedMetadata(module)
            case .exclamation:
                // Named metadata starting with !
                try parseNamedMetadata(module)
            case .newline:
                skipNewlines()
            default:
                // Skip unknown tokens
                _ = advance()
            }
        }

        return module
    }

    // MARK: - Top-level keyword handling

    private mutating func parseTopLevelKeyword(_ module: IRModule) throws {
        let keyword = current.text

        switch keyword {
        case "source_filename":
            _ = advance()
            _ = try expect(.equals)
            let str = try expect(.string)
            module.sourceFilename = unquote(str.text)

        case "target":
            _ = advance()
            let what = try expect(.keyword)
            _ = try expect(.equals)
            let str = try expect(.string)
            if what.text == "datalayout" {
                module.dataLayout = unquote(str.text)
            } else if what.text == "triple" {
                module.targetTriple = unquote(str.text)
            }

        case "define":
            let fn = try parseFunctionDefinition()
            module.functions.append(fn)
            globalValues[fn.name] = IRValue(type: fn.type, name: fn.name)

        case "declare":
            let fn = try parseFunctionDeclaration()
            module.functions.append(fn)
            globalValues[fn.name] = IRValue(type: fn.type, name: fn.name)

        case "attributes":
            let group = try parseAttributeGroup()
            module.attributeGroups.append(group)

        default:
            // Skip unknown top-level directives
            skipToNextLine()
        }
    }

    // MARK: - Type definitions

    private mutating func parseTypeDefinition(_ module: IRModule) throws {
        // %name = type { ... } or %name = type opaque
        let nameTok = try expect(.localIdent)
        let name = String(nameTok.text.dropFirst()) // remove %
        _ = try expect(.equals)
        _ = try expectKeyword("type")

        skipNewlines()
        if current.kind == .keyword && current.text == "opaque" {
            _ = advance()
            let opaqueType = IRType.opaque(name: name)
            structTypes[name] = opaqueType
            module.structTypes.append((name, opaqueType))
        } else {
            let ty = try parseStructBody(name: name)
            structTypes[name] = ty
            module.structTypes.append((name, ty))
        }
    }

    private mutating func parseStructBody(name: String?) throws -> IRType {
        let isPacked = current.kind == .leftAngle
        if isPacked {
            _ = advance() // <
        }
        _ = try expect(.leftBrace)

        var elements: [IRType] = []
        skipNewlines()
        if current.kind != .rightBrace {
            elements.append(try parseType())
            while consumeIf(.comma) {
                skipNewlines()
                elements.append(try parseType())
            }
        }
        _ = try expect(.rightBrace)
        if isPacked {
            _ = try expect(.rightAngle)
        }

        return .structure(name: name, elements: elements, isPacked: isPacked)
    }

    // MARK: - Global definitions

    private mutating func parseGlobalDefinition(_ module: IRModule) throws {
        let nameTok = try expect(.globalIdent)
        let name = String(nameTok.text.dropFirst()) // remove @
        _ = try expect(.equals)

        skipNewlines()

        // Parse linkage
        var linkage: IRFunction.Linkage = .external
        if let l = tryParseLinkage() {
            linkage = l
        }

        // Parse optional modifiers
        var localUnnamedAddr = false
        var unnamedAddr = false
        var addressSpace = 0
        while true {
            if consumeIfKeyword("local_unnamed_addr") {
                localUnnamedAddr = true
            } else if consumeIfKeyword("unnamed_addr") {
                unnamedAddr = true
            } else if current.kind == .keyword && current.text == "addrspace" {
                _ = advance()
                _ = try expect(.leftParen)
                let spaceTok = try expect(.integer)
                addressSpace = Int(spaceTok.text) ?? 0
                _ = try expect(.rightParen)
            } else if consumeIfKeyword("externally_initialized") {
                // skip
            } else {
                break
            }
        }

        // Expect 'global' or 'constant'
        let isConstant: Bool
        if consumeIfKeyword("constant") {
            isConstant = true
        } else if consumeIfKeyword("global") {
            isConstant = false
        } else {
            // Might be an alias or other definition - skip
            skipToNextLine()
            return
        }

        // Parse the value type
        let valueType = try parseType()

        // Parse optional initializer
        skipNewlines()
        var initializer: IRConstant? = nil
        if current.kind != .comma && current.kind != .newline && current.kind != .eof {
            initializer = try parseConstantValue(type: valueType)
        }

        // Parse optional alignment and other attributes
        var alignment: Int? = nil
        while consumeIf(.comma) {
            skipNewlines()
            if consumeIfKeyword("align") {
                let alignTok = try expect(.integer)
                alignment = Int(alignTok.text) ?? 0
            } else {
                // Skip other attributes
                _ = advance()
            }
        }

        let global = IRGlobal(
            name: name,
            valueType: valueType,
            addressSpace: addressSpace,
            initializer: initializer
        )
        global.linkage = linkage
        global.isConstant = isConstant
        global.alignment = alignment
        global.localUnnamedAddr = localUnnamedAddr
        global.unnamedAddr = unnamedAddr

        module.globals.append(global)
        globalValues[name] = IRValue(type: global.type, name: name)
    }

    // MARK: - Function parsing

    private mutating func parseFunctionDefinition() throws -> IRFunction {
        _ = try expectKeyword("define")
        return try parseFunctionHeader(isDeclaration: false)
    }

    private mutating func parseFunctionDeclaration() throws -> IRFunction {
        _ = try expectKeyword("declare")
        return try parseFunctionHeader(isDeclaration: true)
    }

    private mutating func parseFunctionHeader(isDeclaration: Bool) throws -> IRFunction {
        skipNewlines()

        // Parse optional linkage
        var linkage: IRFunction.Linkage = .external
        if let l = tryParseLinkage() {
            linkage = l
        }

        // Parse return attributes and modifiers
        var localUnnamedAddr = false
        while true {
            if consumeIfKeyword("local_unnamed_addr") {
                localUnnamedAddr = true
            } else {
                break
            }
        }

        // Parse return type
        let returnType = try parseType()

        // Parse function name
        let nameTok = try expect(.globalIdent)
        let name = String(nameTok.text.dropFirst())

        // Parse parameter list
        _ = try expect(.leftParen)
        var paramTypes: [IRType] = []
        var paramNames: [String] = []
        var paramStringAttrs: [[String: String]] = []
        var paramAttrs: [[IRAttribute]] = []
        var isVarArg = false

        skipNewlines()
        if current.kind != .rightParen {
            if current.kind == .dotDotDot {
                isVarArg = true
                _ = advance()
            } else {
                let (ty, nm, sattrs, attrs) = try parseParameter()
                paramTypes.append(ty)
                paramNames.append(nm)
                paramStringAttrs.append(sattrs)
                paramAttrs.append(attrs)

                while consumeIf(.comma) {
                    skipNewlines()
                    if current.kind == .dotDotDot {
                        isVarArg = true
                        _ = advance()
                        break
                    }
                    let (ty, nm, sattrs, attrs) = try parseParameter()
                    paramTypes.append(ty)
                    paramNames.append(nm)
                    paramStringAttrs.append(sattrs)
                    paramAttrs.append(attrs)
                }
            }
        }
        _ = try expect(.rightParen)

        // Parse optional function attributes
        var attrGroupIndex: Int? = nil
        skipNewlines()
        while true {
            if current.kind == .attrGroupRef {
                let ref = advance()
                attrGroupIndex = Int(String(ref.text.dropFirst()))
                skipNewlines()
            } else if current.kind == .keyword && isAttributeKeyword(current.text) {
                _ = advance()
                skipNewlines()
            } else {
                break
            }
        }

        let fn = IRFunction(
            name: name,
            returnType: returnType,
            parameterTypes: paramTypes,
            isDeclaration: isDeclaration
        )
        fn.linkage = linkage
        fn.localUnnamedAddr = localUnnamedAddr
        fn.parameterNames = paramNames
        fn.parameterStringAttributes = paramStringAttrs
        fn.parameterAttributes = paramAttrs
        fn.attributeGroupIndex = attrGroupIndex
        fn.type = .function(ret: returnType, params: paramTypes, isVarArg: isVarArg)

        // Set up parameter IRValues
        fn.parameters = paramTypes.enumerated().map { i, ty in
            let n = i < paramNames.count ? paramNames[i] : ""
            return IRValue(type: ty, name: n)
        }

        if !isDeclaration {
            // Parse function body
            localValues = [:]
            // Register parameters in local scope
            for param in fn.parameters {
                if !param.name.isEmpty {
                    localValues[param.name] = param
                }
            }

            _ = try expect(.leftBrace)
            fn.basicBlocks = try parseFunctionBody()
            _ = try expect(.rightBrace)
            localValues = [:]
        }

        return fn
    }

    private mutating func parseParameter() throws -> (IRType, String, [String: String], [IRAttribute]) {
        var type = try parseType()
        var name = ""
        var stringAttrs: [String: String] = [:]
        var attrs: [IRAttribute] = []

        // Parse attributes between type and name
        skipNewlines()
        while true {
            if current.kind == .keyword && isParamAttributeKeyword(current.text) {
                let attr = parseParamAttribute()
                if let a = attr { attrs.append(a) }
            } else if current.kind == .string {
                // String attribute like "air-buffer-no-alias"
                let key = unquote(advance().text)
                stringAttrs[key] = ""
            } else {
                break
            }
            skipNewlines()
        }

        // Parse pointer suffixes (addrspace(N)*)
        while current.kind == .star {
            _ = advance()
            type = .pointer(pointee: type, addressSpace: 0)
        }

        // Parse name if present
        if current.kind == .localIdent {
            name = String(advance().text.dropFirst())
        }

        return (type, name, stringAttrs, attrs)
    }

    private mutating func parseParamAttribute() -> IRAttribute? {
        let text = current.text
        _ = advance()
        switch text {
        case "noundef": return .noundef
        case "nocapture": return .noCapture
        case "nonnull": return .nonNull
        case "readonly": return .readOnly
        case "writeonly": return .writeOnly
        case "signext": return .signExt
        case "zeroext": return .zeroExt
        case "inreg": return .inReg
        case "noalias": return .noAlias
        case "immarg": return .immArg
        case "returned": return .returned
        default: return nil
        }
    }

    // MARK: - Function body

    private mutating func parseFunctionBody() throws -> [IRBasicBlock] {
        var blocks: [IRBasicBlock] = []
        skipNewlines()

        // First block may have an implicit label
        var currentBlock = IRBasicBlock(name: "entry")
        var isFirstBlock = true

        while current.kind != .rightBrace && current.kind != .eof {
            skipNewlines()
            if current.kind == .rightBrace || current.kind == .eof { break }

            // Check for label
            if current.kind == .label {
                // Save current block if it has instructions
                if !isFirstBlock || !currentBlock.instructions.isEmpty {
                    blocks.append(currentBlock)
                }
                let labelName = String(advance().text.dropFirst()) // remove ':'
                currentBlock = IRBasicBlock(name: labelName)
                isFirstBlock = false
                skipNewlines()
                continue
            }

            // Parse instruction
            if let inst = try parseInstruction() {
                currentBlock.instructions.append(inst)
            }
            skipNewlines()
        }

        if !currentBlock.instructions.isEmpty || blocks.isEmpty {
            blocks.append(currentBlock)
        }

        return blocks
    }

    // MARK: - Instruction parsing

    private mutating func parseInstruction() throws -> IRInstruction? {
        skipNewlines()
        if current.kind == .rightBrace || current.kind == .eof { return nil }

        // Check for result assignment: %name = ...
        var resultName = ""
        if current.kind == .localIdent {
            // Look ahead to see if this is an assignment
            let saved = pos
            let name = advance()
            if current.kind == .equals {
                _ = advance() // consume =
                resultName = String(name.text.dropFirst())
            } else {
                // Not an assignment - put back
                pos = saved
            }
        }

        skipNewlines()

        // Parse the instruction opcode
        guard current.kind == .keyword else {
            // Skip to next line
            skipToNextLine()
            return nil
        }

        let opcode = current.text

        switch opcode {
        case "ret":
            return try parseRet()
        case "br":
            return try parseBr()
        case "call", "tail", "musttail", "notail":
            return try parseCall(resultName: resultName)
        case "alloca":
            return try parseAlloca(resultName: resultName)
        case "load":
            return try parseLoad(resultName: resultName)
        case "store":
            return try parseStore()
        case "getelementptr":
            return try parseGEP(resultName: resultName)
        case "bitcast":
            return try parseCast(opcode: .bitcast, resultName: resultName)
        case "zext":
            return try parseCast(opcode: .zext, resultName: resultName)
        case "sext":
            return try parseCast(opcode: .sext, resultName: resultName)
        case "trunc":
            return try parseCast(opcode: .trunc, resultName: resultName)
        case "fptoui":
            return try parseCast(opcode: .fpToUI, resultName: resultName)
        case "fptosi":
            return try parseCast(opcode: .fpToSI, resultName: resultName)
        case "uitofp":
            return try parseCast(opcode: .uiToFP, resultName: resultName)
        case "sitofp":
            return try parseCast(opcode: .siToFP, resultName: resultName)
        case "fptrunc":
            return try parseCast(opcode: .fpTrunc, resultName: resultName)
        case "fpext":
            return try parseCast(opcode: .fpExt, resultName: resultName)
        case "ptrtoint":
            return try parseCast(opcode: .ptrToInt, resultName: resultName)
        case "inttoptr":
            return try parseCast(opcode: .intToPtr, resultName: resultName)
        case "addrspacecast":
            return try parseCast(opcode: .addrSpaceCast, resultName: resultName)
        case "add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
             "shl", "lshr", "ashr", "and", "or", "xor",
             "fadd", "fsub", "fmul", "fdiv", "frem":
            return try parseBinOp(resultName: resultName)
        case "icmp":
            return try parseCmp(isFP: false, resultName: resultName)
        case "fcmp":
            return try parseCmp(isFP: true, resultName: resultName)
        case "phi":
            return try parsePhi(resultName: resultName)
        case "select":
            return try parseSelect(resultName: resultName)
        case "fneg":
            return try parseFNeg(resultName: resultName)
        case "extractvalue":
            return try parseExtractValue(resultName: resultName)
        case "insertvalue":
            return try parseInsertValue(resultName: resultName)
        case "extractelement":
            return try parseExtractElement(resultName: resultName)
        case "insertelement":
            return try parseInsertElement(resultName: resultName)
        case "shufflevector":
            return try parseShuffleVector(resultName: resultName)
        case "unreachable":
            _ = advance()
            return IRInstruction(opcode: .unreachable)
        case "switch":
            return try parseSwitch()
        case "fence":
            skipToNextLine()
            return nil
        default:
            // Unknown instruction - skip line
            skipToNextLine()
            return nil
        }
    }

    // MARK: - Individual instruction parsers

    private mutating func parseRet() throws -> IRInstruction {
        _ = try expectKeyword("ret")
        skipNewlines()

        if consumeIfKeyword("void") {
            return IRInstruction(opcode: .ret, type: .void)
        }

        let type = try parseType()
        let operand = try parseOperand(type: type)
        let inst = IRInstruction(opcode: .ret, type: .void, operands: [operand])
        return inst
    }

    private mutating func parseBr() throws -> IRInstruction {
        _ = try expectKeyword("br")
        skipNewlines()

        // Unconditional: br label %bb
        // Conditional: br i1 %cond, label %true, label %false
        if consumeIfKeyword("label") {
            let dest = try expect(.localIdent)
            let bb = IRBasicBlock(name: String(dest.text.dropFirst()))
            return IRInstruction(opcode: .br, operands: [.basicBlock(bb)])
        }

        // Conditional branch
        let condType = try parseType()
        let cond = try parseOperand(type: condType)
        _ = try expect(.comma)
        _ = try expectKeyword("label")
        let trueDest = try expect(.localIdent)
        _ = try expect(.comma)
        _ = try expectKeyword("label")
        let falseDest = try expect(.localIdent)

        let trueBB = IRBasicBlock(name: String(trueDest.text.dropFirst()))
        let falseBB = IRBasicBlock(name: String(falseDest.text.dropFirst()))
        return IRInstruction(opcode: .br, operands: [cond, .basicBlock(trueBB), .basicBlock(falseBB)])
    }

    private mutating func parseCall(resultName: String) throws -> IRInstruction {
        // Parse optional tail call modifier
        var tailKind: IRInstruction.TailCallKind? = nil
        if consumeIfKeyword("tail") {
            tailKind = .tail
        } else if consumeIfKeyword("musttail") {
            tailKind = .mustTail
        } else if consumeIfKeyword("notail") {
            tailKind = .noTail
        }

        _ = try expectKeyword("call")
        skipNewlines()

        // Parse calling convention
        // Skip fast math flags
        while current.kind == .keyword && isFastMathFlag(current.text) {
            _ = advance()
        }

        // Parse return type
        let returnType = try parseType()

        // Parse function reference
        let fnTok = try expect(.globalIdent)
        let fnName = String(fnTok.text.dropFirst())

        // Parse argument list
        _ = try expect(.leftParen)
        var argOperands: [IRInstruction.Operand] = []
        skipNewlines()
        if current.kind != .rightParen {
            let (_, argOp) = try parseTypedOperand()
            argOperands.append(argOp)
            while consumeIf(.comma) {
                skipNewlines()
                let (_, argOp) = try parseTypedOperand()
                argOperands.append(argOp)
            }
        }
        _ = try expect(.rightParen)

        // Parse optional function attributes
        while current.kind == .attrGroupRef {
            _ = advance()
        }

        // Build function value
        let fnValue = resolveGlobalValue(fnName, type: .pointer(
            pointee: .function(ret: returnType, params: argOperands.map { _ in .void }, isVarArg: false),
            addressSpace: 0
        ))

        // All args + function as last operand
        var operands = argOperands
        operands.append(.value(fnValue))

        let inst = IRInstruction(
            opcode: .call,
            type: returnType,
            name: resultName,
            operands: operands
        )
        inst.attributes.tailCall = tailKind

        if !resultName.isEmpty {
            let val = IRValue(type: returnType, name: resultName)
            localValues[resultName] = val
        }

        return inst
    }

    private mutating func parseAlloca(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("alloca")
        skipNewlines()

        let allocType = try parseType()

        // Parse optional comma-separated attributes
        var alignment: Int? = nil
        while consumeIf(.comma) {
            skipNewlines()
            if consumeIfKeyword("align") {
                let tok = try expect(.integer)
                alignment = Int(tok.text)
            } else {
                // Skip unknown
                _ = advance()
            }
        }

        let ptrType = IRType.pointer(pointee: allocType, addressSpace: 0)
        let inst = IRInstruction(
            opcode: .alloca,
            type: ptrType,
            name: resultName
        )
        inst.attributes.allocaType = allocType
        inst.attributes.alignment = alignment

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: ptrType, name: resultName)
        }

        return inst
    }

    private mutating func parseLoad(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("load")
        skipNewlines()

        // Parse optional volatile
        let isVolatile = consumeIfKeyword("volatile")

        let resultType = try parseType()
        _ = try expect(.comma)
        let ptrType = try parseType()
        let ptr = try parseOperand(type: ptrType)

        // Parse optional alignment
        var alignment: Int? = nil
        if consumeIf(.comma) {
            skipNewlines()
            if consumeIfKeyword("align") {
                let tok = try expect(.integer)
                alignment = Int(tok.text)
            }
        }

        let inst = IRInstruction(
            opcode: .load,
            type: resultType,
            name: resultName,
            operands: [ptr]
        )
        inst.attributes.alignment = alignment
        inst.attributes.isVolatile = isVolatile

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: resultType, name: resultName)
        }

        return inst
    }

    private mutating func parseStore() throws -> IRInstruction {
        _ = try expectKeyword("store")
        skipNewlines()

        let isVolatile = consumeIfKeyword("volatile")

        let valueType = try parseType()
        let value = try parseOperand(type: valueType)
        _ = try expect(.comma)
        let ptrType = try parseType()
        let ptr = try parseOperand(type: ptrType)

        var alignment: Int? = nil
        if consumeIf(.comma) {
            skipNewlines()
            if consumeIfKeyword("align") {
                let tok = try expect(.integer)
                alignment = Int(tok.text)
            }
        }

        let inst = IRInstruction(
            opcode: .store,
            type: .void,
            operands: [value, ptr]
        )
        inst.attributes.alignment = alignment
        inst.attributes.isVolatile = isVolatile
        return inst
    }

    private mutating func parseGEP(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("getelementptr")
        skipNewlines()

        var inBounds = false
        if consumeIfKeyword("inbounds") {
            inBounds = true
        }

        // Source element type
        let sourceType = try parseType()
        _ = try expect(.comma)

        // Base pointer
        let basePtrType = try parseType()
        let basePtr = try parseOperand(type: basePtrType)

        // Indices
        var operands: [IRInstruction.Operand] = [basePtr]
        while consumeIf(.comma) {
            skipNewlines()
            let idxType = try parseType()
            let idx = try parseOperand(type: idxType)
            operands.append(idx)
        }

        // Compute result type (simplified: just pointer with same addrspace)
        let resultType: IRType
        if case .pointer = basePtrType {
            // GEP result is always a pointer
            resultType = basePtrType // simplified
        } else {
            resultType = basePtrType
        }

        let inst = IRInstruction(
            opcode: .getelementptr,
            type: resultType,
            name: resultName,
            operands: operands
        )
        inst.attributes.inBounds = inBounds
        inst.attributes.gepSourceType = sourceType

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: resultType, name: resultName)
        }

        return inst
    }

    private mutating func parseCast(opcode: IRInstruction.Opcode, resultName: String) throws -> IRInstruction {
        _ = advance() // consume opcode keyword
        skipNewlines()

        let srcType = try parseType()
        let src = try parseOperand(type: srcType)
        _ = try expectKeyword("to")
        let destType = try parseType()

        let inst = IRInstruction(
            opcode: opcode,
            type: destType,
            name: resultName,
            operands: [src]
        )

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: destType, name: resultName)
        }

        return inst
    }

    private mutating func parseBinOp(resultName: String) throws -> IRInstruction {
        let opcodeStr = advance().text
        skipNewlines()

        // Parse optional flags (nuw, nsw, exact, fast, nnan, etc.)
        while current.kind == .keyword && isBinOpFlag(current.text) {
            _ = advance()
        }

        let type = try parseType()
        let lhs = try parseOperand(type: type)
        _ = try expect(.comma)
        let rhs = try parseOperand(type: type)

        let opcode = binOpFromString(opcodeStr)
        let inst = IRInstruction(
            opcode: opcode,
            type: type,
            name: resultName,
            operands: [lhs, rhs]
        )

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: type, name: resultName)
        }

        return inst
    }

    private mutating func parseCmp(isFP: Bool, resultName: String) throws -> IRInstruction {
        _ = advance() // consume icmp/fcmp
        skipNewlines()

        // Parse predicate
        let predTok = try expect(.keyword)
        let predicate = cmpPredicate(predTok.text, isFP: isFP)

        let type = try parseType()
        let lhs = try parseOperand(type: type)
        _ = try expect(.comma)
        let rhs = try parseOperand(type: type)

        let inst = IRInstruction(
            opcode: isFP ? .fcmp : .icmp,
            type: .i1,
            name: resultName,
            operands: [lhs, rhs]
        )
        inst.attributes.predicate = predicate

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: .i1, name: resultName)
        }

        return inst
    }

    private mutating func parsePhi(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("phi")
        skipNewlines()

        let type = try parseType()
        var operands: [IRInstruction.Operand] = []

        // Parse [value, label] pairs
        repeat {
            skipNewlines()
            _ = try expect(.leftBracket)
            let val = try parseOperand(type: type)
            _ = try expect(.comma)
            let bbTok = try expect(.localIdent)
            let bb = IRBasicBlock(name: String(bbTok.text.dropFirst()))
            _ = try expect(.rightBracket)
            operands.append(val)
            operands.append(.basicBlock(bb))
        } while consumeIf(.comma)

        let inst = IRInstruction(
            opcode: .phi,
            type: type,
            name: resultName,
            operands: operands
        )

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: type, name: resultName)
        }

        return inst
    }

    private mutating func parseSelect(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("select")
        skipNewlines()

        let condType = try parseType()
        let cond = try parseOperand(type: condType)
        _ = try expect(.comma)
        let trueType = try parseType()
        let trueVal = try parseOperand(type: trueType)
        _ = try expect(.comma)
        let falseType = try parseType()
        let falseVal = try parseOperand(type: falseType)

        let inst = IRInstruction(
            opcode: .select,
            type: trueType,
            name: resultName,
            operands: [cond, trueVal, falseVal]
        )

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: trueType, name: resultName)
        }

        return inst
    }

    private mutating func parseFNeg(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("fneg")
        skipNewlines()

        let type = try parseType()
        let operand = try parseOperand(type: type)

        let inst = IRInstruction(
            opcode: .fneg,
            type: type,
            name: resultName,
            operands: [operand]
        )

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: type, name: resultName)
        }

        return inst
    }

    private mutating func parseExtractValue(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("extractvalue")
        skipNewlines()
        let aggType = try parseType()
        let agg = try parseOperand(type: aggType)
        var indices: [IRInstruction.Operand] = [agg]
        while consumeIf(.comma) {
            let tok = try expect(.integer)
            indices.append(.intLiteral(Int64(tok.text)!))
        }
        let inst = IRInstruction(opcode: .extractValue, type: .i32, name: resultName, operands: indices)
        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: .i32, name: resultName)
        }
        return inst
    }

    private mutating func parseInsertValue(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("insertvalue")
        skipNewlines()
        let aggType = try parseType()
        let agg = try parseOperand(type: aggType)
        _ = try expect(.comma)
        let elemType = try parseType()
        let elem = try parseOperand(type: elemType)
        var operands: [IRInstruction.Operand] = [agg, elem]
        while consumeIf(.comma) {
            let tok = try expect(.integer)
            operands.append(.intLiteral(Int64(tok.text)!))
        }
        let inst = IRInstruction(opcode: .insertValue, type: aggType, name: resultName, operands: operands)
        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: aggType, name: resultName)
        }
        return inst
    }

    private mutating func parseExtractElement(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("extractelement")
        skipNewlines()
        let vecType = try parseType()
        let vec = try parseOperand(type: vecType)
        _ = try expect(.comma)
        let idxType = try parseType()
        let idx = try parseOperand(type: idxType)
        let elemType: IRType
        if case .vector(let elem, _) = vecType { elemType = elem } else { elemType = .i32 }
        let inst = IRInstruction(opcode: .extractElement, type: elemType, name: resultName, operands: [vec, idx])
        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: elemType, name: resultName)
        }
        return inst
    }

    private mutating func parseInsertElement(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("insertelement")
        skipNewlines()
        let vecType = try parseType()
        let vec = try parseOperand(type: vecType)
        _ = try expect(.comma)
        let elemType = try parseType()
        let elem = try parseOperand(type: elemType)
        _ = try expect(.comma)
        let idxType = try parseType()
        let idx = try parseOperand(type: idxType)
        let inst = IRInstruction(opcode: .insertElement, type: vecType, name: resultName, operands: [vec, elem, idx])
        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: vecType, name: resultName)
        }
        return inst
    }

    private mutating func parseShuffleVector(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("shufflevector")
        skipNewlines()
        let vecType = try parseType()
        let v1 = try parseOperand(type: vecType)
        _ = try expect(.comma)
        let _ = try parseType()
        let v2 = try parseOperand(type: vecType)
        _ = try expect(.comma)
        let maskType = try parseType()
        let mask = try parseOperand(type: maskType)
        let inst = IRInstruction(opcode: .shuffleVector, type: vecType, name: resultName, operands: [v1, v2, mask])
        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: vecType, name: resultName)
        }
        return inst
    }

    private mutating func parseSwitch() throws -> IRInstruction {
        _ = try expectKeyword("switch")
        skipToNextLine() // simplified
        return IRInstruction(opcode: .switchInst)
    }

    // MARK: - Type parsing

    mutating func parseType() throws -> IRType {
        skipNewlines()

        var baseType: IRType

        switch current.kind {
        case .keyword:
            let text = current.text
            switch text {
            case "void": _ = advance(); baseType = .void
            case "i1": _ = advance(); baseType = .i1
            case "i8": _ = advance(); baseType = .i8
            case "i16": _ = advance(); baseType = .i16
            case "i32": _ = advance(); baseType = .i32
            case "i64": _ = advance(); baseType = .i64
            case "half": _ = advance(); baseType = .float16
            case "float": _ = advance(); baseType = .float32
            case "double": _ = advance(); baseType = .float64
            case "label": _ = advance(); baseType = .label
            case "metadata": _ = advance(); baseType = .metadata
            case "token": _ = advance(); baseType = .token
            case "opaque": _ = advance(); baseType = .opaque(name: "opaque")
            default:
                // Check for iN (arbitrary width integer)
                if text.hasPrefix("i"), let bits = Int(text.dropFirst()) {
                    _ = advance()
                    baseType = .int(bits: bits)
                } else {
                    throw ParseError.unexpected("type keyword '\(text)'", line: current.line, column: current.column)
                }
            }

        case .localIdent:
            // Struct type reference: %struct_name
            let name = String(advance().text.dropFirst())
            if let ty = structTypes[name] {
                baseType = ty
            } else {
                // Forward reference to opaque type
                baseType = .opaque(name: name)
            }

        case .leftBrace:
            // Anonymous struct: { T1, T2, ... }
            baseType = try parseStructBody(name: nil)

        case .leftAngle:
            // Could be <{ packed struct }> or <N x T> vector
            _ = advance()
            skipNewlines()
            if current.kind == .leftBrace {
                // Packed struct: <{ ... }>
                let structTy = try parseStructBody(name: nil)
                if case .structure(let n, let elems, _) = structTy {
                    baseType = .structure(name: n, elements: elems, isPacked: true)
                } else {
                    baseType = structTy
                }
                _ = try expect(.rightAngle)
            } else {
                // Vector: <N x T>
                let countTok = try expect(.integer)
                let count = Int(countTok.text) ?? 0
                _ = try expectKeyword("x")
                let elemType = try parseType()
                _ = try expect(.rightAngle)
                baseType = .vector(element: elemType, count: count)
            }

        case .leftBracket:
            // Array: [N x T]
            _ = advance()
            let countTok = try expect(.integer)
            let count = Int(countTok.text) ?? 0
            _ = try expectKeyword("x")
            let elemType = try parseType()
            _ = try expect(.rightBracket)
            baseType = .array(element: elemType, count: count)

        default:
            throw ParseError.unexpected(
                "in type position: \(current.kind)(\(current.text))",
                line: current.line, column: current.column
            )
        }

        // Check for function type: retType (paramTypes...)
        if current.kind == .leftParen {
            baseType = try parseFunctionType(returnType: baseType)
        }

        // Parse addrspace and pointer suffixes
        baseType = try parsePointerSuffix(baseType)

        return baseType
    }

    private mutating func parseFunctionType(returnType: IRType) throws -> IRType {
        _ = try expect(.leftParen)
        var paramTypes: [IRType] = []
        var isVarArg = false

        skipNewlines()
        if current.kind != .rightParen {
            if current.kind == .dotDotDot {
                isVarArg = true
                _ = advance()
            } else {
                paramTypes.append(try parseType())
                // Skip param attributes
                while current.kind == .keyword && isParamAttributeKeyword(current.text) {
                    _ = advance()
                }
                while current.kind == .string {
                    _ = advance()
                }
                while consumeIf(.comma) {
                    skipNewlines()
                    if current.kind == .dotDotDot {
                        isVarArg = true
                        _ = advance()
                        break
                    }
                    paramTypes.append(try parseType())
                    // Skip param attributes
                    while current.kind == .keyword && isParamAttributeKeyword(current.text) {
                        _ = advance()
                    }
                    while current.kind == .string {
                        _ = advance()
                    }
                }
            }
        }
        _ = try expect(.rightParen)

        return .function(ret: returnType, params: paramTypes, isVarArg: isVarArg)
    }

    private mutating func parsePointerSuffix(_ baseType: IRType) throws -> IRType {
        var ty = baseType

        // Check for addrspace(N) qualifier
        while true {
            skipNewlines()
            if current.kind == .keyword && current.text == "addrspace" {
                _ = advance()
                _ = try expect(.leftParen)
                let spaceTok = try expect(.integer)
                let addrSpace = Int(spaceTok.text) ?? 0
                _ = try expect(.rightParen)
                _ = try expect(.star)
                ty = .pointer(pointee: ty, addressSpace: addrSpace)
            } else if current.kind == .star {
                _ = advance()
                ty = .pointer(pointee: ty, addressSpace: 0)
            } else {
                break
            }
        }

        return ty
    }

    // MARK: - Operand parsing

    private mutating func parseOperand(type: IRType) throws -> IRInstruction.Operand {
        skipNewlines()

        switch current.kind {
        case .localIdent:
            let name = String(advance().text.dropFirst())
            if let val = localValues[name] {
                return .value(val)
            }
            let val = IRValue(type: type, name: name)
            localValues[name] = val
            return .value(val)

        case .globalIdent:
            let name = String(advance().text.dropFirst())
            let val = resolveGlobalValue(name, type: type)
            return .value(val)

        case .integer:
            let text = advance().text
            let value = Int64(text) ?? 0
            return .constant(.integer(type, value))

        case .float_:
            let text = advance().text
            if text.hasPrefix("0x") {
                // Hex float
                let hexStr = String(text.dropFirst(2))
                if let bits = UInt64(hexStr, radix: 16) {
                    if case .float32 = type {
                        return .constant(.float32(Float(bitPattern: UInt32(bits & 0xFFFFFFFF))))
                    } else if case .float64 = type {
                        return .constant(.float64(Double(bitPattern: bits)))
                    }
                }
            }
            if let d = Double(text) {
                if case .float32 = type {
                    return .constant(.float32(Float(d)))
                }
                return .constant(.float64(d))
            }
            return .constant(.float64(0))

        case .keyword:
            let text = current.text
            if text == "null" {
                _ = advance()
                return .constant(.null(type))
            } else if text == "undef" {
                _ = advance()
                return .constant(.undef(type))
            } else if text == "zeroinitializer" {
                _ = advance()
                return .constant(.zeroInitializer(type))
            } else if text == "true" {
                _ = advance()
                return .constant(.integer(.i1, 1))
            } else if text == "false" {
                _ = advance()
                return .constant(.integer(.i1, 0))
            } else if text == "getelementptr" {
                return try parseConstantGEP(type: type)
            } else if text == "bitcast" {
                return try parseConstantCast(type: type)
            }
            throw ParseError.unexpected("operand '\(text)'", line: current.line, column: current.column)

        default:
            throw ParseError.unexpected(
                "operand: \(current.kind)(\(current.text))",
                line: current.line, column: current.column
            )
        }
    }

    private mutating func parseTypedOperand() throws -> (IRType, IRInstruction.Operand) {
        let type = try parseType()

        // Parse parameter attributes between type and value
        while current.kind == .keyword && isParamAttributeKeyword(current.text) {
            _ = advance()
        }
        while current.kind == .string {
            _ = advance() // skip string attributes
        }

        let operand = try parseOperand(type: type)
        return (type, operand)
    }

    private mutating func parseConstantGEP(type: IRType) throws -> IRInstruction.Operand {
        _ = try expectKeyword("getelementptr")
        skipNewlines()
        _ = consumeIfKeyword("inbounds")
        _ = try expect(.leftParen)
        // Parse the GEP expression
        _ = try parseType()
        _ = try expect(.comma)
        let ptrType = try parseType()
        let ptr = try parseOperand(type: ptrType)
        var indices: [IRInstruction.Operand] = []
        while consumeIf(.comma) {
            let idxType = try parseType()
            let idx = try parseOperand(type: idxType)
            indices.append(idx)
        }
        _ = try expect(.rightParen)
        // Return as a constant GEP (simplified)
        return ptr
    }

    private mutating func parseConstantCast(type: IRType) throws -> IRInstruction.Operand {
        _ = advance() // consume 'bitcast'
        _ = try expect(.leftParen)
        let srcType = try parseType()
        let src = try parseOperand(type: srcType)
        _ = try expectKeyword("to")
        _ = try parseType()
        _ = try expect(.rightParen)
        return src
    }

    private mutating func parseConstantValue(type: IRType) throws -> IRConstant {
        skipNewlines()
        switch current.kind {
        case .integer:
            let text = advance().text
            return .integer(type, Int64(text) ?? 0)
        case .float_:
            let text = advance().text
            if let d = Double(text) {
                return .float64(d)
            }
            return .float64(0)
        case .keyword:
            let text = current.text
            if text == "undef" { _ = advance(); return .undef(type) }
            if text == "zeroinitializer" { _ = advance(); return .zeroInitializer(type) }
            if text == "null" { _ = advance(); return .null(type) }
            // Skip unknown
            skipToNextLine()
            return .undef(type)
        default:
            return .undef(type)
        }
    }

    // MARK: - Metadata parsing

    private mutating func parseNamedMetadata(_ module: IRModule) throws {
        // !name = !{!0, !1} or !0 = !{...} or !0 = distinct !{...}
        if current.kind == .exclamation {
            _ = advance() // consume !
        }

        if current.kind == .metadataIdent {
            let ident = advance()
            let text = String(ident.text.dropFirst()) // remove !

            if let index = Int(text) {
                // Numbered metadata: !0 = !{...}
                _ = try expect(.equals)
                skipNewlines()
                let isDistinct = consumeIfKeyword("distinct")
                let node = try parseMetadataNode(index: index)
                node.isDistinct = isDistinct

                // Ensure we have enough nodes
                while module.metadataNodes.count <= index {
                    module.metadataNodes.append(IRMetadataNode(index: module.metadataNodes.count))
                }
                module.metadataNodes[index] = node
            } else {
                // Named metadata: !air.kernel = !{!0}
                let name = text
                _ = try expect(.equals)
                skipNewlines()
                _ = try expect(.exclamation)
                _ = try expect(.leftBrace)

                var operands: [Int] = []
                skipNewlines()
                if current.kind != .rightBrace {
                    let ref = try parseMetadataRef()
                    if case .index(let idx) = ref {
                        operands.append(idx)
                    }
                    while consumeIf(.comma) {
                        skipNewlines()
                        let ref = try parseMetadataRef()
                        if case .index(let idx) = ref {
                            operands.append(idx)
                        }
                    }
                }
                _ = try expect(.rightBrace)

                module.namedMetadata.append(IRNamedMetadata(name: name, operands: operands))
            }
        } else {
            // Skip unknown metadata
            skipToNextLine()
        }
    }

    private mutating func parseMetadataNode(index: Int) throws -> IRMetadataNode {
        _ = try expect(.exclamation)
        _ = try expect(.leftBrace)

        var operands: [IRMetadataOperand] = []
        skipNewlines()

        if current.kind != .rightBrace {
            let op = try parseMetadataOperand()
            operands.append(op)
            while consumeIf(.comma) {
                skipNewlines()
                let op = try parseMetadataOperand()
                operands.append(op)
            }
        }

        _ = try expect(.rightBrace)
        return IRMetadataNode(index: index, operands: operands)
    }

    private mutating func parseMetadataOperand() throws -> IRMetadataOperand {
        skipNewlines()

        // Metadata string: !"string"
        if current.kind == .metadataString {
            let text = advance().text
            // Remove ! and quotes: !"foo" → foo
            let inner = String(text.dropFirst()) // remove !
            return .string(unquote(inner))
        }

        // Metadata reference: !N
        if current.kind == .metadataIdent {
            let text = advance().text
            let numStr = String(text.dropFirst()) // remove !
            if let idx = Int(numStr) {
                return .metadata(idx)
            }
            return .string(numStr)
        }

        // Null/empty
        if current.kind == .keyword && current.text == "null" {
            _ = advance()
            return .null
        }

        // Typed value: type value (e.g., i32 42 or void (...)* @fn)
        let savedPos = pos
        if let type = try? parseType() {
            skipNewlines()
            // Check for a value
            if current.kind == .integer {
                let valText = advance().text
                let value = Int64(valText) ?? 0
                return .constant(type, .integer(type, value))
            } else if current.kind == .globalIdent {
                let name = String(advance().text.dropFirst())
                return .value(type, name)
            } else if current.kind == .keyword && current.text == "null" {
                _ = advance()
                return .constant(type, .null(type))
            } else if current.kind == .keyword && current.text == "undef" {
                _ = advance()
                return .constant(type, .undef(type))
            }
        }

        // If type parsing consumed tokens but didn't find a value, backtrack
        if pos != savedPos {
            // Already consumed, can't easily backtrack. Return null.
            return .null
        }

        // Skip unknown
        _ = advance()
        return .null
    }

    private mutating func parseMetadataRef() throws -> IRMetadataRef {
        if current.kind == .metadataIdent {
            let text = advance().text
            let numStr = String(text.dropFirst())
            if let idx = Int(numStr) {
                return .index(idx)
            }
            return .string(numStr)
        }
        _ = try expect(.exclamation)
        if current.kind == .integer {
            let idx = Int(advance().text) ?? 0
            return .index(idx)
        }
        throw ParseError.unexpected("metadata ref", line: current.line, column: current.column)
    }

    // MARK: - Attribute groups

    private mutating func parseAttributeGroup() throws -> IRAttributeGroup {
        _ = try expectKeyword("attributes")
        let refTok = try expect(.attrGroupRef)
        let index = Int(String(refTok.text.dropFirst())) ?? 0
        _ = try expect(.equals)
        _ = try expect(.leftBrace)

        var attributes: [IRAttribute] = []
        skipNewlines()
        while current.kind != .rightBrace && current.kind != .eof {
            if current.kind == .string {
                // String attribute: "key" or "key"="value"
                let key = unquote(advance().text)
                if consumeIf(.equals) {
                    let value = unquote((try expect(.string)).text)
                    attributes.append(.stringAttr(key: key, value: value))
                } else {
                    attributes.append(.stringAttr(key: key, value: nil))
                }
            } else if current.kind == .keyword {
                if let attr = attributeFromKeyword(current.text) {
                    attributes.append(attr)
                }
                _ = advance()
            } else {
                _ = advance()
            }
            skipNewlines()
        }
        _ = try expect(.rightBrace)

        return IRAttributeGroup(index: index, attributes: attributes)
    }

    // MARK: - Helper methods

    private func resolveGlobalValue(_ name: String, type: IRType) -> IRValue {
        if let val = globalValues[name] {
            return val
        }
        return IRValue(type: type, name: name)
    }

    private mutating func skipToNextLine() {
        while pos < tokens.count && tokens[pos].kind != .newline && tokens[pos].kind != .eof {
            pos += 1
        }
        if pos < tokens.count && tokens[pos].kind == .newline {
            pos += 1
        }
    }

    private func unquote(_ s: String) -> String {
        var str = s
        if str.hasPrefix("\"") && str.hasSuffix("\"") {
            str = String(str.dropFirst().dropLast())
        }
        return str
    }

    private mutating func tryParseLinkage() -> IRFunction.Linkage? {
        guard current.kind == .keyword else { return nil }
        switch current.text {
        case "private", "internal", "linkonce", "linkonce_odr", "weak", "weak_odr",
             "common", "appending", "extern_weak", "available_externally":
            let text = advance().text
            switch text {
            case "private": return .private
            case "internal": return .internal
            case "linkonce": return .linkonce
            case "linkonce_odr": return .linkonceODR
            case "weak": return .weak
            case "weak_odr": return .weakODR
            case "common": return .common
            case "appending": return .appending
            case "extern_weak": return .externWeak
            case "available_externally": return .available_externally
            default: return nil
            }
        default:
            return nil
        }
    }

    private func isAttributeKeyword(_ text: String) -> Bool {
        let attrs = ["nounwind", "convergent", "mustprogress", "willreturn", "nofree",
                     "nosync", "nocallback", "argmemonly", "readnone", "readonly",
                     "writeonly", "noreturn", "noinline", "alwaysinline", "optnone",
                     "optsize", "minsize", "local_unnamed_addr", "unnamed_addr"]
        return attrs.contains(text)
    }

    private func isParamAttributeKeyword(_ text: String) -> Bool {
        let attrs = ["noundef", "nocapture", "nonnull", "readonly", "writeonly",
                     "signext", "zeroext", "inreg", "noalias", "immarg",
                     "returned", "nofree", "nosync"]
        return attrs.contains(text)
    }

    private func isBinOpFlag(_ text: String) -> Bool {
        let flags = ["nuw", "nsw", "exact", "nnan", "ninf", "nsz",
                     "arcp", "contract", "afn", "reassoc", "fast"]
        return flags.contains(text)
    }

    private func isFastMathFlag(_ text: String) -> Bool {
        return isBinOpFlag(text)
    }

    private func binOpFromString(_ text: String) -> IRInstruction.Opcode {
        switch text {
        case "add": return .add
        case "fadd": return .fadd
        case "sub": return .sub
        case "fsub": return .fsub
        case "mul": return .mul
        case "fmul": return .fmul
        case "udiv": return .udiv
        case "sdiv": return .sdiv
        case "fdiv": return .fdiv
        case "urem": return .urem
        case "srem": return .srem
        case "frem": return .frem
        case "shl": return .shl
        case "lshr": return .lshr
        case "ashr": return .ashr
        case "and": return .and
        case "or": return .or
        case "xor": return .xor
        default: return .add
        }
    }

    private func cmpPredicate(_ text: String, isFP: Bool) -> Int {
        if isFP {
            switch text {
            case "false": return 0
            case "oeq": return 1
            case "ogt": return 2
            case "oge": return 3
            case "olt": return 4
            case "ole": return 5
            case "one": return 6
            case "ord": return 7
            case "uno": return 8
            case "ueq": return 9
            case "ugt": return 10
            case "uge": return 11
            case "ult": return 12
            case "ule": return 13
            case "une": return 14
            case "true": return 15
            default: return 0
            }
        } else {
            switch text {
            case "eq": return 32
            case "ne": return 33
            case "ugt": return 34
            case "uge": return 35
            case "ult": return 36
            case "ule": return 37
            case "sgt": return 38
            case "sge": return 39
            case "slt": return 40
            case "sle": return 41
            default: return 32
            }
        }
    }

    private func attributeFromKeyword(_ text: String) -> IRAttribute? {
        switch text {
        case "nounwind": return .noUnwind
        case "convergent": return .convergent
        case "mustprogress": return .mustProgress
        case "willreturn": return .willReturn
        case "nofree": return .noFree
        case "nosync": return .noSync
        case "nocallback": return .noCallback
        case "argmemonly": return .argMemOnly
        case "readnone": return .readNone
        case "readonly": return .readOnly
        case "noreturn": return .noReturn
        case "noinline": return .noInline
        case "alwaysinline": return .alwaysInline
        case "nocapture": return .noCapture
        case "noalias": return .noAlias
        case "immarg": return .immArg
        default: return nil
        }
    }
}
