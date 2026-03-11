/// Recursive descent parser for LLVM IR text.
///
/// Parses a token stream into an `IRModule` in-memory representation.
/// Supports the subset of LLVM IR used by Metal AIR, including typed pointers.
public struct Parser {
    private var tokens: [Token]
    private let source: [UInt8]
    private var pos: Int = 0

    /// Named values in the current function scope.
    private var localValues: [String: IRValue] = [:]

    /// Global named values.
    private var globalValues: [String: IRValue] = [:]

    /// Struct type definitions.
    private var structTypes: [String: IRType] = [:]

    public init(tokens: [Token], source: [UInt8]) {
        self.tokens = tokens
        self.source = source
    }

    /// Get the text of a token from the source buffer.
    @inline(__always)
    private func text(_ t: Token) -> String {
        String(decoding: source[t.start..<t.end], as: UTF8.self)
    }

    /// Compare token text to a string without allocating (for hot paths).
    @inline(__always)
    private func textEquals(_ t: Token, _ s: String) -> Bool {
        let len = t.end - t.start
        guard len == s.utf8.count else { return false }
        var i = t.start
        for b in s.utf8 {
            if source[i] != b { return false }
            i += 1
        }
        return true
    }

    /// Compare token text to a static string without allocating.
    @inline(__always)
    private func textEquals(_ t: Token, _ s: StaticString) -> Bool {
        let len = t.end - t.start
        guard len == s.utf8CodeUnitCount else { return false }
        return s.withUTF8Buffer { buf in
            for i in 0..<len {
                if source[t.start + i] != buf[i] { return false }
            }
            return true
        }
    }

    /// Parse an integer from token bytes without String allocation.
    @inline(__always)
    private func tokenInt(_ t: Token) -> Int? {
        var i = t.start
        let end = t.end
        guard i < end else { return nil }
        var neg = false
        if source[i] == 0x2D { neg = true; i += 1 } // '-'
        guard i < end && source[i] >= 0x30 && source[i] <= 0x39 else { return nil }
        var val = 0
        while i < end {
            let b = source[i]
            guard b >= 0x30 && b <= 0x39 else { return nil }
            val = val * 10 + Int(b - 0x30)
            i += 1
        }
        return neg ? -val : val
    }

    /// Build a String from token bytes, skipping `drop` bytes from the start.
    @inline(__always)
    private func tokenText(_ t: Token, dropFirst drop: Int = 0) -> String {
        String(decoding: source[(t.start + drop)..<t.end], as: UTF8.self)
    }

    /// Parse hex digits from token bytes (after skipping `drop` prefix bytes) into UInt64.
    @inline(__always)
    private func tokenHexUInt64(_ t: Token, dropFirst drop: Int) -> UInt64? {
        var i = t.start + drop
        let end = t.end
        guard i < end else { return nil }
        var val: UInt64 = 0
        while i < end {
            let b = source[i]
            if b >= 0x30 && b <= 0x39 { val = val &* 16 &+ UInt64(b - 0x30) }
            else if b >= 0x61 && b <= 0x66 { val = val &* 16 &+ UInt64(b - 0x61 + 10) }
            else if b >= 0x41 && b <= 0x46 { val = val &* 16 &+ UInt64(b - 0x41 + 10) }
            else { return nil }
            i += 1
        }
        return val
    }

    /// Check if token matches any string in a set (byte-level, no allocation).
    @inline(__always)
    private func tokenInSet(_ t: Token, _ set: Set<String>) -> Bool {
        let len = t.end - t.start
        for s in set {
            if s.utf8.count == len {
                var match = true
                var i = t.start
                for b in s.utf8 {
                    if source[i] != b { match = false; break }
                    i += 1
                }
                if match { return true }
            }
        }
        return false
    }

    // MARK: - Error handling

    enum ParseError: Error {
        case unexpected(String, line: Int, column: Int)
        case expected(String, got: String, line: Int, column: Int)
    }

    // MARK: - Token navigation

    private var current: Token {
        guard pos < tokens.count else {
            return Token.eof
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
            return Token.eof
        }
        return tokens[idx]
    }

    private mutating func expect(_ kind: Token.Kind, _ context: String = "") throws -> Token {
        skipNewlines()
        guard current.kind == kind else {
            throw ParseError.expected(
                "\(kind) \(context)",
                got: "\(current.kind)(\(text(current)))",
                line: 0,
                column: 0
            )
        }
        return advance()
    }

    private mutating func expectKeyword(_ keyword: String) throws -> Token {
        skipNewlines()
        guard current.kind == .keyword && textEquals(current, keyword) else {
            throw ParseError.expected(
                "keyword '\(keyword)'",
                got: text(current),
                line: 0,
                column: 0
            )
        }
        return advance()
    }

    private mutating func consumeIfKeyword(_ keyword: String) -> Bool {
        if current.kind == .keyword && textEquals(current, keyword) {
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
        if textEquals(current, "source_filename") {
            _ = advance()
            _ = try expect(.equals)
            let str = try expect(.string)
            module.sourceFilename = unquote(text(str))
        } else if textEquals(current, "target") {
            _ = advance()
            let what = try expect(.keyword)
            _ = try expect(.equals)
            let str = try expect(.string)
            if textEquals(what, "datalayout") {
                module.dataLayout = unquote(text(str))
            } else if textEquals(what, "triple") {
                module.targetTriple = unquote(text(str))
            }
        } else if textEquals(current, "define") {
            let fn = try parseFunctionDefinition()
            // Create attribute group for inline function attributes
            if !fn.functionAttributes.isEmpty && fn.attributeGroupIndex == nil {
                let newIdx = (module.attributeGroups.map(\.index).max() ?? -1) + 1
                module.attributeGroups.append(IRAttributeGroup(index: newIdx, attributes: fn.functionAttributes))
                fn.attributeGroupIndex = newIdx
            }
            module.functions.append(fn)
            globalValues[fn.name] = IRValue(type: fn.type, name: fn.name)
        } else if textEquals(current, "declare") {
            let fn = try parseFunctionDeclaration()
            module.functions.append(fn)
            globalValues[fn.name] = IRValue(type: fn.type, name: fn.name)
        } else if textEquals(current, "attributes") {
            let group = try parseAttributeGroup()
            module.attributeGroups.append(group)
        } else {
            skipToNextLine()
        }
    }

    // MARK: - Type definitions

    private mutating func parseTypeDefinition(_ module: IRModule) throws {
        // %name = type { ... } or %name = type opaque
        let nameTok = try expect(.localIdent)
        let name = tokenText(nameTok, dropFirst: 1) // remove %
        _ = try expect(.equals)
        _ = try expectKeyword("type")

        skipNewlines()
        if current.kind == .keyword && textEquals(current, "opaque") {
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
        let name = tokenText(nameTok, dropFirst: 1) // remove @
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
            } else if current.kind == .keyword && textEquals(current, "addrspace") {
                _ = advance()
                _ = try expect(.leftParen)
                let spaceTok = try expect(.integer)
                addressSpace = tokenInt(spaceTok) ?? 0
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
                alignment = tokenInt(alignTok) ?? 0
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

        // Parse optional calling convention
        var callingConvention: IRFunction.CallingConvention = .c
        if consumeIfKeyword("kernel") || consumeIfKeyword("spir_kernel") {
            callingConvention = .spirKernel
        } else if consumeIfKeyword("spir_func") {
            callingConvention = .spirFunc
        } else if consumeIfKeyword("ccc") {
            callingConvention = .c
        } else if consumeIfKeyword("fastcc") {
            callingConvention = .fast
        } else if consumeIfKeyword("coldcc") {
            callingConvention = .cold
        } else if consumeIfKeyword("swiftcc") {
            callingConvention = .swiftcc
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
        let name = tokenText(nameTok, dropFirst: 1)

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
        var inlineAttrs: [IRAttribute] = []
        skipNewlines()
        while true {
            if current.kind == .attrGroupRef {
                let ref = advance()
                attrGroupIndex = tokenInt(Token(kind: ref.kind, start: ref.start + 1, end: ref.end))
                skipNewlines()
            } else if current.kind == .keyword && tokenInSet(current, Self.attributeKeywords) {
                if let attr = attributeFromToken(current) {
                    inlineAttrs.append(attr)
                }
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
        fn.callingConvention = callingConvention
        fn.localUnnamedAddr = localUnnamedAddr
        fn.parameterNames = paramNames
        fn.parameterStringAttributes = paramStringAttrs
        fn.parameterAttributes = paramAttrs
        fn.functionAttributes = inlineAttrs
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
            fn.basicBlocks = try parseFunctionBody(numParams: fn.parameters.count)
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
            if current.kind == .keyword && tokenInSet(current, Self.paramAttributeKeywords) {
                let attr = parseParamAttribute()
                if let a = attr { attrs.append(a) }
            } else if current.kind == .string {
                // String attribute like "air-buffer-no-alias"
                let key = unquote(text(advance()))
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
            name = tokenText(advance(), dropFirst: 1)
        }

        return (type, name, stringAttrs, attrs)
    }

    private mutating func parseParamAttribute() -> IRAttribute? {
        let t = current
        _ = advance()
        if textEquals(t, "noundef") { return .noundef }
        if textEquals(t, "nocapture") { return .noCapture }
        if textEquals(t, "nonnull") { return .nonNull }
        if textEquals(t, "readonly") { return .readOnly }
        if textEquals(t, "writeonly") { return .writeOnly }
        if textEquals(t, "signext") { return .signExt }
        if textEquals(t, "zeroext") { return .zeroExt }
        if textEquals(t, "inreg") { return .inReg }
        if textEquals(t, "noalias") { return .noAlias }
        if textEquals(t, "immarg") { return .immArg }
        if textEquals(t, "returned") { return .returned }
        return nil
    }

    // MARK: - Function body

    private mutating func parseFunctionBody(numParams: Int = 0) throws -> [IRBasicBlock] {
        var blocks: [IRBasicBlock] = []
        skipNewlines()

        // First block may have an implicit label — LLVM numbers it after the last param
        var currentBlock = IRBasicBlock(name: "\(numParams)")
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
                let labelTok = advance()
                let labelName = String(decoding: source[labelTok.start..<(labelTok.end - 1)], as: UTF8.self) // remove trailing ':'
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
                resultName = tokenText(name, dropFirst: 1)
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

        // Dispatch by first byte + textEquals to avoid String allocation per instruction.
        let t = current
        let fb = source[t.start]
        switch fb {
        case 0x72: // r
            if textEquals(t, "ret") { return try parseRet() }
            break
        case 0x62: // b
            if textEquals(t, "br") { return try parseBr() }
            if textEquals(t, "bitcast") { return try parseCast(opcode: .bitcast, resultName: resultName) }
            break
        case 0x63: // c
            if textEquals(t, "call") { return try parseCall(resultName: resultName) }
            break
        case 0x74: // t
            if textEquals(t, "tail") || textEquals(t, "trunc") {
                if textEquals(t, "tail") { return try parseCall(resultName: resultName) }
                return try parseCast(opcode: .trunc, resultName: resultName)
            }
            break
        case 0x6D: // m
            if textEquals(t, "musttail") { return try parseCall(resultName: resultName) }
            if textEquals(t, "mul") { return try parseBinOp(resultName: resultName) }
            break
        case 0x6E: // n
            if textEquals(t, "notail") { return try parseCall(resultName: resultName) }
            break
        case 0x61: // a
            if textEquals(t, "atomicrmw") { return try parseAtomicRMW(resultName: resultName) }
            if textEquals(t, "alloca") { return try parseAlloca(resultName: resultName) }
            if textEquals(t, "add") || textEquals(t, "and") || textEquals(t, "ashr") {
                return try parseBinOp(resultName: resultName)
            }
            if textEquals(t, "addrspacecast") { return try parseCast(opcode: .addrSpaceCast, resultName: resultName) }
            break
        case 0x6C: // l
            if textEquals(t, "load") { return try parseLoad(resultName: resultName) }
            if textEquals(t, "lshr") { return try parseBinOp(resultName: resultName) }
            break
        case 0x73: // s
            if textEquals(t, "store") { return try parseStore() }
            if textEquals(t, "select") { return try parseSelect(resultName: resultName) }
            if textEquals(t, "sub") || textEquals(t, "sdiv") || textEquals(t, "srem") ||
               textEquals(t, "shl") { return try parseBinOp(resultName: resultName) }
            if textEquals(t, "sext") { return try parseCast(opcode: .sext, resultName: resultName) }
            if textEquals(t, "sitofp") { return try parseCast(opcode: .siToFP, resultName: resultName) }
            if textEquals(t, "shufflevector") { return try parseShuffleVector(resultName: resultName) }
            if textEquals(t, "switch") { return try parseSwitch() }
            break
        case 0x67: // g
            if textEquals(t, "getelementptr") { return try parseGEP(resultName: resultName) }
            break
        case 0x7A: // z
            if textEquals(t, "zext") { return try parseCast(opcode: .zext, resultName: resultName) }
            break
        case 0x66: // f
            if textEquals(t, "fptoui") { return try parseCast(opcode: .fpToUI, resultName: resultName) }
            if textEquals(t, "fptosi") { return try parseCast(opcode: .fpToSI, resultName: resultName) }
            if textEquals(t, "fptrunc") { return try parseCast(opcode: .fpTrunc, resultName: resultName) }
            if textEquals(t, "fpext") { return try parseCast(opcode: .fpExt, resultName: resultName) }
            if textEquals(t, "fadd") || textEquals(t, "fsub") || textEquals(t, "fmul") ||
               textEquals(t, "fdiv") || textEquals(t, "frem") { return try parseBinOp(resultName: resultName) }
            if textEquals(t, "fcmp") { return try parseCmp(isFP: true, resultName: resultName) }
            if textEquals(t, "fneg") { return try parseFNeg(resultName: resultName) }
            if textEquals(t, "fence") { skipToNextLine(); return nil }
            break
        case 0x75: // u
            if textEquals(t, "uitofp") { return try parseCast(opcode: .uiToFP, resultName: resultName) }
            if textEquals(t, "udiv") || textEquals(t, "urem") { return try parseBinOp(resultName: resultName) }
            if textEquals(t, "unreachable") { _ = advance(); return IRInstruction(opcode: .unreachable) }
            break
        case 0x70: // p
            if textEquals(t, "ptrtoint") { return try parseCast(opcode: .ptrToInt, resultName: resultName) }
            if textEquals(t, "phi") { return try parsePhi(resultName: resultName) }
            break
        case 0x69: // i
            if textEquals(t, "inttoptr") { return try parseCast(opcode: .intToPtr, resultName: resultName) }
            if textEquals(t, "icmp") { return try parseCmp(isFP: false, resultName: resultName) }
            if textEquals(t, "insertvalue") { return try parseInsertValue(resultName: resultName) }
            if textEquals(t, "insertelement") { return try parseInsertElement(resultName: resultName) }
            break
        case 0x65: // e
            if textEquals(t, "extractvalue") { return try parseExtractValue(resultName: resultName) }
            if textEquals(t, "extractelement") { return try parseExtractElement(resultName: resultName) }
            break
        case 0x6F: // o
            if textEquals(t, "or") || textEquals(t, "xor") { return try parseBinOp(resultName: resultName) }
            break
        case 0x78: // x
            if textEquals(t, "xor") { return try parseBinOp(resultName: resultName) }
            break
        default:
            break
        }
        // Unknown instruction - skip line
        skipToNextLine()
        return nil
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
            let bb = IRBasicBlock(name: tokenText(dest, dropFirst: 1))
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

        let trueBB = IRBasicBlock(name: tokenText(trueDest, dropFirst: 1))
        let falseBB = IRBasicBlock(name: tokenText(falseDest, dropFirst: 1))
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
        while current.kind == .keyword && tokenInSet(current, Self.binOpFlags) {
            _ = advance()
        }

        // Parse return type
        let returnType = try parseType()

        // Parse function reference
        let fnTok = try expect(.globalIdent)
        let fnName = tokenText(fnTok, dropFirst: 1)

        // Parse argument list
        _ = try expect(.leftParen)
        var argOperands: [IRInstruction.Operand] = []
        var argTypes: [IRType] = []
        skipNewlines()
        if current.kind != .rightParen {
            let (argTy, argOp) = try parseTypedOperand()
            argOperands.append(argOp)
            argTypes.append(argTy)
            while consumeIf(.comma) {
                skipNewlines()
                let (argTy, argOp) = try parseTypedOperand()
                argOperands.append(argOp)
                argTypes.append(argTy)
            }
        }
        _ = try expect(.rightParen)

        // Parse optional function attributes
        while current.kind == .attrGroupRef {
            _ = advance()
        }

        // Build function value — use actual argument types for forward references
        let fnValue = resolveGlobalValue(fnName, type: .pointer(
            pointee: .function(ret: returnType, params: argTypes, isVarArg: false),
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
                alignment = tokenInt(tok)
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
                alignment = tokenInt(tok)
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
                alignment = tokenInt(tok)
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

    // atomicrmw add ptr addrspace(3) %ptr, i32 1 monotonic, align 4
    private mutating func parseAtomicRMW(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("atomicrmw")
        skipNewlines()

        // Parse optional volatile
        _ = consumeIfKeyword("volatile")

        // Parse operation: add, xchg, max, min, umax, umin, and, or, xor, sub
        guard current.kind == .keyword else {
            throw ParseError.unexpected("expected atomic operation", line: 0, column: 0)
        }
        let opTok = advance()
        let atomicOp = tokenText(opTok)

        skipNewlines()

        // Parse pointer type and operand
        let ptrType = try parseType()
        let ptr = try parseOperand(type: ptrType)
        _ = try expect(.comma)

        // Parse value type and operand
        let valType = try parseType()
        let val = try parseOperand(type: valType)

        // Skip ordering (monotonic, acquire, release, etc.)
        if current.kind == .keyword {
            _ = advance()
        }

        // Parse optional alignment
        if consumeIf(.comma) {
            skipNewlines()
            if consumeIfKeyword("align") {
                if current.kind == .integer { _ = advance() }
            }
        }

        let inst = IRInstruction(
            opcode: .atomicRMW,
            type: valType,
            name: resultName,
            operands: [ptr, val]
        )
        inst.attributes.atomicOp = atomicOp

        if !resultName.isEmpty {
            localValues[resultName] = IRValue(type: valType, name: resultName)
        }

        return inst
    }

    private mutating func parseGEP(resultName: String) throws -> IRInstruction {
        _ = try expectKeyword("getelementptr")
        skipNewlines()

        var inBounds = false
        if consumeIfKeyword("inbounds") {
            inBounds = true
        }
        _ = consumeIfKeyword("nuw")

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
        let opcodeTok = advance()
        skipNewlines()

        // Parse optional flags (nuw, nsw, exact, fast, nnan, etc.)
        while current.kind == .keyword && tokenInSet(current, Self.binOpFlags) {
            _ = advance()
        }

        let type = try parseType()
        let lhs = try parseOperand(type: type)
        _ = try expect(.comma)
        let rhs = try parseOperand(type: type)

        let opcode = binOpFromToken(opcodeTok)
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

        // Skip optional fast-math flags (e.g., fcmp fast ogt ...)
        while current.kind == .keyword && tokenInSet(current, Self.binOpFlags) {
            _ = advance()
        }

        // Parse predicate
        let predTok = try expect(.keyword)
        let predicate = cmpPredicate(predTok, isFP: isFP)

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
            let bb = IRBasicBlock(name: tokenText(bbTok, dropFirst: 1))
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
            indices.append(.intLiteral(Int64(tokenInt(tok) ?? 0)))
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
            operands.append(.intLiteral(Int64(tokenInt(tok) ?? 0)))
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
            let tt = current
            let fb = source[tt.start]
            let tlen = tt.end - tt.start
            if fb == 0x76 && textEquals(tt, "void") { _ = advance(); baseType = .void }
            else if fb == 0x69 && tlen == 2 && source[tt.start+1] == 0x31 { _ = advance(); baseType = .i1 }  // i1
            else if fb == 0x69 && tlen == 2 && source[tt.start+1] == 0x38 { _ = advance(); baseType = .i8 }  // i8
            else if fb == 0x69 && tlen == 3 && textEquals(tt, "i16") { _ = advance(); baseType = .i16 }
            else if fb == 0x69 && tlen == 3 && textEquals(tt, "i32") { _ = advance(); baseType = .i32 }
            else if fb == 0x69 && tlen == 3 && textEquals(tt, "i64") { _ = advance(); baseType = .i64 }
            else if fb == 0x68 && textEquals(tt, "half") { _ = advance(); baseType = .float16 }
            else if fb == 0x62 && textEquals(tt, "bfloat") { _ = advance(); baseType = .bfloat16 }
            else if fb == 0x66 && tlen == 5 && textEquals(tt, "float") { _ = advance(); baseType = .float32 }
            else if fb == 0x64 && textEquals(tt, "double") { _ = advance(); baseType = .float64 }
            else if fb == 0x6C && textEquals(tt, "label") { _ = advance(); baseType = .label }
            else if fb == 0x6D && textEquals(tt, "metadata") { _ = advance(); baseType = .metadata }
            else if fb == 0x74 && textEquals(tt, "token") { _ = advance(); baseType = .token }
            else if fb == 0x70 && textEquals(tt, "ptr") {
                _ = advance()
                if current.kind == .keyword && textEquals(current, "addrspace") {
                    _ = advance()
                    _ = try expect(.leftParen)
                    let spaceTok = try expect(.integer)
                    let addrSpace = tokenInt(spaceTok) ?? 0
                    _ = try expect(.rightParen)
                    baseType = .opaquePointer(addressSpace: addrSpace)
                } else {
                    baseType = .opaquePointer(addressSpace: 0)
                }
            }
            else if fb == 0x6F && textEquals(tt, "opaque") { _ = advance(); baseType = .opaque(name: "opaque") }
            else if fb == 0x69 {
                // iN (arbitrary width integer)
                let sub = Token(kind: tt.kind, start: tt.start + 1, end: tt.end)
                if let bits = tokenInt(sub) {
                    _ = advance()
                    baseType = .int(bits: bits)
                } else {
                    throw ParseError.unexpected("type keyword '\(text(tt))'", line: 0, column: 0)
                }
            }
            else {
                throw ParseError.unexpected("type keyword '\(text(tt))'", line: 0, column: 0)
            }

        case .localIdent:
            // Struct type reference: %struct_name
            let name = tokenText(advance(), dropFirst: 1)
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
                let count = tokenInt(countTok) ?? 0
                _ = try expectKeyword("x")
                let elemType = try parseType()
                _ = try expect(.rightAngle)
                baseType = .vector(element: elemType, count: count)
            }

        case .leftBracket:
            // Array: [N x T]
            _ = advance()
            let countTok = try expect(.integer)
            let count = tokenInt(countTok) ?? 0
            _ = try expectKeyword("x")
            let elemType = try parseType()
            _ = try expect(.rightBracket)
            baseType = .array(element: elemType, count: count)

        default:
            throw ParseError.unexpected(
                "in type position: \(current.kind)(\(text(current)))",
                line: 0, column: 0
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
                while current.kind == .keyword && tokenInSet(current, Self.paramAttributeKeywords) {
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
                    while current.kind == .keyword && tokenInSet(current, Self.paramAttributeKeywords) {
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
            if current.kind == .keyword && textEquals(current, "addrspace") {
                _ = advance()
                _ = try expect(.leftParen)
                let spaceTok = try expect(.integer)
                let addrSpace = tokenInt(spaceTok) ?? 0
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

    /// Parse a hex literal (0x...) as a floating-point constant.
    /// LLVM IR encodes float hex constants as double-precision bits (16 hex digits).
    /// The lexer classifies these as `.integer` since they lack a typed prefix (0xH/0xK/etc.).
    private mutating func parseHexFloatOperand(type: IRType) throws -> IRInstruction.Operand {
        let tok = advance()
        guard let bits = tokenHexUInt64(tok, dropFirst: 2) else {
            return .constant(.float64(0))
        }
        let asDouble = Double(bitPattern: bits)
        switch type {
        case .float32:
            return .constant(.float32(Float(asDouble)))
        case .float16:
            return .constant(.float16(UInt16(Float(asDouble).bitPattern & 0xFFFF)))
        case .bfloat16:
            return .constant(.bfloat16(UInt16(Float(asDouble).bitPattern >> 16)))
        default:
            return .constant(.float64(asDouble))
        }
    }

    private mutating func parseOperand(type: IRType) throws -> IRInstruction.Operand {
        skipNewlines()

        switch current.kind {
        case .localIdent:
            let name = tokenText(advance(), dropFirst: 1)
            if let val = localValues[name] {
                return .value(val)
            }
            let val = IRValue(type: type, name: name)
            localValues[name] = val
            return .value(val)

        case .globalIdent:
            let name = tokenText(advance(), dropFirst: 1)
            let val = resolveGlobalValue(name, type: type)
            return .value(val)

        case .integer:
            // Check for hex prefix (0x/0X)
            let isHex = (current.end - current.start) > 2
                && source[current.start] == 0x30
                && (source[current.start + 1] == 0x78 || source[current.start + 1] == 0x58)
            if isHex && type.isFloatingPoint {
                return try parseHexFloatOperand(type: type)
            }
            let tok = advance()
            let value: Int64
            if isHex {
                let bits = tokenHexUInt64(tok, dropFirst: 2) ?? 0
                value = Int64(bitPattern: bits)
            } else {
                value = Int64(tokenInt(tok) ?? 0)
            }
            return .constant(.integer(type, value))

        case .float_:
            let tok = advance()
            let len = tok.end - tok.start
            if len > 3 && source[tok.start] == 0x30
                && (source[tok.start + 1] == 0x78 || source[tok.start + 1] == 0x58) {
                if source[tok.start + 2] == 0x48 { // 0xH - half precision
                    let bits = UInt16(tokenHexUInt64(tok, dropFirst: 3) ?? 0)
                    return .constant(.float16(bits))
                } else if source[tok.start + 2] == 0x52 { // 0xR - bfloat16
                    let bits = UInt16(tokenHexUInt64(tok, dropFirst: 3) ?? 0)
                    return .constant(.bfloat16(bits))
                } else { // 0x - double precision
                    if let bits = tokenHexUInt64(tok, dropFirst: 2) {
                        let asDouble = Double(bitPattern: bits)
                        if case .float32 = type {
                            return .constant(.float32(Float(asDouble)))
                        } else if case .float64 = type {
                            return .constant(.float64(asDouble))
                        }
                        if case .float16 = type {
                            return .constant(.float16(UInt16(bits & 0xFFFF)))
                        }
                    }
                }
            }
            if let d = Double(tokenText(tok)) {
                if case .float32 = type {
                    return .constant(.float32(Float(d)))
                }
                return .constant(.float64(d))
            }
            return .constant(.float64(0))

        case .keyword:
            if textEquals(current, "null") {
                _ = advance()
                return .constant(.null(type))
            } else if textEquals(current, "undef") {
                _ = advance()
                return .constant(.undef(type))
            } else if textEquals(current, "zeroinitializer") {
                _ = advance()
                return .constant(.zeroInitializer(type))
            } else if textEquals(current, "true") {
                _ = advance()
                return .constant(.integer(.i1, 1))
            } else if textEquals(current, "false") {
                _ = advance()
                return .constant(.integer(.i1, 0))
            } else if textEquals(current, "getelementptr") {
                return try parseConstantGEP(type: type)
            } else if textEquals(current, "bitcast") {
                return try parseConstantCast(type: type)
            } else if textEquals(current, "splat") {
                // splat (T val) — replicate scalar constant across all vector lanes
                _ = advance() // consume 'splat'
                _ = try expect(.leftParen)
                let (_, elemOp) = try parseTypedOperand()
                _ = try expect(.rightParen)
                guard case .vector(let elemType, let count) = type,
                      case .constant(let c) = elemOp else {
                    return .constant(.undef(type))
                }
                _ = elemType  // already encoded in c.type
                let elements = [IRConstant](repeating: c, count: count)
                return .constant(.vectorValue(type, elements))
            }
            throw ParseError.unexpected("operand '\(text(current))'", line: 0, column: 0)

        case .leftAngle:
            // Vector constant literal: <i32 0, i32 1, ...>
            _ = advance() // consume '<'
            skipNewlines()
            var elements: [IRConstant] = []
            while current.kind != .rightAngle {
                let elemType = try parseType()
                let elemOp = try parseOperand(type: elemType)
                switch elemOp {
                case .constant(let c):
                    elements.append(c)
                default:
                    // Non-constant in vector literal — treat as undef
                    elements.append(.undef(elemType))
                }
                if current.kind == .comma {
                    _ = advance()
                    skipNewlines()
                }
            }
            _ = try expect(.rightAngle)
            return .constant(.vectorValue(type, elements))

        default:
            throw ParseError.unexpected(
                "operand: \(current.kind)(\(text(current)))",
                line: 0, column: 0
            )
        }
    }

    private mutating func parseTypedOperand() throws -> (IRType, IRInstruction.Operand) {
        let type = try parseType()

        // Parse parameter attributes between type and value
        while current.kind == .keyword && tokenInSet(current, Self.paramAttributeKeywords) {
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
        _ = consumeIfKeyword("nuw")
        _ = try expect(.leftParen)
        let srcType = try parseType()
        _ = try expect(.comma)
        let ptrType = try parseType()
        let ptrOp = try parseOperand(type: ptrType)
        // Compute constant byte offset
        var byteOffset: Int64 = 0
        while consumeIf(.comma) {
            let idxType = try parseType()
            let idxOp = try parseOperand(type: idxType)
            if case .constant(let c) = idxOp, case .integer(_, let v) = c {
                // For i8 source type, each index is 1 byte
                let elemSize: Int64
                switch srcType {
                case .i8: elemSize = 1
                case .int(let bits): elemSize = Int64(bits / 8)
                case .float16, .bfloat16: elemSize = 2
                case .float32: elemSize = 4
                case .float64: elemSize = 8
                default: elemSize = 1
                }
                byteOffset += v * elemSize
            }
        }
        _ = try expect(.rightParen)
        if byteOffset == 0 {
            return ptrOp
        }
        // Create a NEW value with the byte-offset (don't mutate shared global reference)
        if case .value(let v) = ptrOp {
            let newVal = IRValue(type: v.type, name: v.name)
            newVal.constantGEPByteOffset = byteOffset
            return .value(newVal)
        }
        return ptrOp
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
            let tok = advance()
            return .integer(type, Int64(tokenInt(tok) ?? 0))
        case .float_:
            let tok = advance()
            // Check prefix bytes for 0xH vs 0x
            let len = tok.end - tok.start
            if len > 3 && source[tok.start] == 0x30
                && (source[tok.start + 1] == 0x78 || source[tok.start + 1] == 0x58) {
                if source[tok.start + 2] == 0x48 { // 0xH - half precision
                    let bits = UInt16(tokenHexUInt64(tok, dropFirst: 3) ?? 0)
                    return .float16(bits)
                } else if source[tok.start + 2] == 0x52 { // 0xR - bfloat16
                    let bits = UInt16(tokenHexUInt64(tok, dropFirst: 3) ?? 0)
                    return .bfloat16(bits)
                } else { // 0x - double precision
                    if let bits = tokenHexUInt64(tok, dropFirst: 2) {
                        let asDouble = Double(bitPattern: bits)
                        if case .float32 = type {
                            return .float32(Float(asDouble))
                        }
                        if case .float16 = type {
                            return .float16(UInt16(bits & 0xFFFF))
                        }
                        return .float64(asDouble)
                    }
                }
            }
            if let d = Double(tokenText(tok)) {
                if case .float32 = type { return .float32(Float(d)) }
                return .float64(d)
            }
            return .float64(0)
        case .keyword:
            if textEquals(current, "undef") { _ = advance(); return .undef(type) }
            if textEquals(current, "zeroinitializer") { _ = advance(); return .zeroInitializer(type) }
            if textEquals(current, "null") { _ = advance(); return .null(type) }
            // Skip unknown
            skipToNextLine()
            return .undef(type)
        case .leftBracket:
            // Array literal: [type val, type val, ...]
            _ = advance() // consume [
            var elements: [IRConstant] = []
            while current.kind != .rightBracket && current.kind != .eof {
                let elemType = try parseType()
                let elem = try parseConstantValue(type: elemType)
                elements.append(elem)
                if !consumeIf(.comma) { break }
                skipNewlines()
            }
            _ = try expect(.rightBracket)
            return .arrayValue(type, elements)
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
            let text = tokenText(ident, dropFirst: 1) // remove !

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
            let tok = advance()
            // Remove ! and quotes: !"foo" → foo
            let inner = tokenText(tok, dropFirst: 1) // remove !
            return .string(unquote(inner))
        }

        // Metadata reference: !N
        if current.kind == .metadataIdent {
            let tok = advance()
            if let idx = tokenInt(Token(kind: tok.kind, start: tok.start + 1, end: tok.end)) {
                return .metadata(idx)
            }
            return .string(tokenText(tok, dropFirst: 1))
        }

        // Null/empty
        if current.kind == .keyword && textEquals(current, "null") {
            _ = advance()
            return .null
        }

        // Typed value: type value (e.g., i32 42 or void (...)* @fn)
        let savedPos = pos
        if let type = try? parseType() {
            skipNewlines()
            // Check for a value
            if current.kind == .integer {
                let tok = advance()
                let value = Int64(tokenInt(tok) ?? 0)
                return .constant(type, .integer(type, value))
            } else if current.kind == .globalIdent {
                let name = tokenText(advance(), dropFirst: 1)
                return .value(type, name)
            } else if current.kind == .keyword && textEquals(current, "null") {
                _ = advance()
                return .constant(type, .null(type))
            } else if current.kind == .keyword && textEquals(current, "undef") {
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
            let tok = advance()
            let sub = Token(kind: tok.kind, start: tok.start + 1, end: tok.end)
            if let idx = tokenInt(sub) {
                return .index(idx)
            }
            return .string(tokenText(tok, dropFirst: 1))
        }
        _ = try expect(.exclamation)
        if current.kind == .integer {
            let idx = tokenInt(advance()) ?? 0
            return .index(idx)
        }
        throw ParseError.unexpected("metadata ref", line: 0, column: 0)
    }

    // MARK: - Attribute groups

    private mutating func parseAttributeGroup() throws -> IRAttributeGroup {
        _ = try expectKeyword("attributes")
        let refTok = try expect(.attrGroupRef)
        let index = tokenInt(Token(kind: refTok.kind, start: refTok.start + 1, end: refTok.end)) ?? 0
        _ = try expect(.equals)
        _ = try expect(.leftBrace)

        var attributes: [IRAttribute] = []
        skipNewlines()
        while current.kind != .rightBrace && current.kind != .eof {
            if current.kind == .string {
                // String attribute: "key" or "key"="value"
                let key = unquote(text(advance()))
                if consumeIf(.equals) {
                    let value = unquote(text(try expect(.string)))
                    attributes.append(.stringAttr(key: key, value: value))
                } else {
                    attributes.append(.stringAttr(key: key, value: nil))
                }
            } else if current.kind == .keyword {
                if let attr = attributeFromToken(current) {
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
        let t = current
        if textEquals(t, "private") { _ = advance(); return .private }
        if textEquals(t, "internal") { _ = advance(); return .internal }
        if textEquals(t, "linkonce") { _ = advance(); return .linkonce }
        if textEquals(t, "linkonce_odr") { _ = advance(); return .linkonceODR }
        if textEquals(t, "weak") { _ = advance(); return .weak }
        if textEquals(t, "weak_odr") { _ = advance(); return .weakODR }
        if textEquals(t, "common") { _ = advance(); return .common }
        if textEquals(t, "appending") { _ = advance(); return .appending }
        if textEquals(t, "extern_weak") { _ = advance(); return .externWeak }
        if textEquals(t, "available_externally") { _ = advance(); return .available_externally }
        return nil
    }

    private static let attributeKeywords: Set<String> = [
        "nounwind", "convergent", "mustprogress", "willreturn", "nofree",
        "nosync", "nocallback", "argmemonly", "readnone", "readonly",
        "writeonly", "noreturn", "noinline", "alwaysinline", "optnone",
        "optsize", "minsize", "local_unnamed_addr", "unnamed_addr"
    ]

    private static let paramAttributeKeywords: Set<String> = [
        "noundef", "nocapture", "nonnull", "readonly", "writeonly",
        "signext", "zeroext", "inreg", "noalias", "immarg",
        "returned", "nofree", "nosync"
    ]

    private static let binOpFlags: Set<String> = [
        "nuw", "nsw", "exact", "nnan", "ninf", "nsz",
        "arcp", "contract", "afn", "reassoc", "fast", "disjoint"
    ]

    private func isAttributeKeyword(_ text: String) -> Bool {
        Self.attributeKeywords.contains(text)
    }

    private func isParamAttributeKeyword(_ text: String) -> Bool {
        Self.paramAttributeKeywords.contains(text)
    }

    private func isBinOpFlag(_ text: String) -> Bool {
        Self.binOpFlags.contains(text)
    }

    private func isFastMathFlag(_ text: String) -> Bool {
        Self.binOpFlags.contains(text)
    }

    private func binOpFromToken(_ t: Token) -> IRInstruction.Opcode {
        let len = t.end - t.start
        let fb = source[t.start]
        switch fb {
        case 0x61: // a
            if len == 3 {
                if textEquals(t, "add") { return .add }
                if textEquals(t, "and") { return .and }
            }
            if textEquals(t, "ashr") { return .ashr }
            return .add
        case 0x66: // f
            if len == 4 {
                if textEquals(t, "fadd") { return .fadd }
                if textEquals(t, "fsub") { return .fsub }
                if textEquals(t, "fmul") { return .fmul }
                if textEquals(t, "fdiv") { return .fdiv }
                if textEquals(t, "frem") { return .frem }
            }
            return .add
        case 0x73: // s
            if textEquals(t, "sub") { return .sub }
            if textEquals(t, "sdiv") { return .sdiv }
            if textEquals(t, "srem") { return .srem }
            if textEquals(t, "shl") { return .shl }
            return .add
        case 0x6D: // m
            if textEquals(t, "mul") { return .mul }
            return .add
        case 0x75: // u
            if textEquals(t, "udiv") { return .udiv }
            if textEquals(t, "urem") { return .urem }
            return .add
        case 0x6C: // l
            if textEquals(t, "lshr") { return .lshr }
            return .add
        case 0x6F: // o
            if textEquals(t, "or") { return .or }
            return .add
        case 0x78: // x
            if textEquals(t, "xor") { return .xor }
            return .add
        default: return .add
        }
    }

    private func cmpPredicate(_ t: Token, isFP: Bool) -> Int {
        let len = t.end - t.start
        let fb = source[t.start]
        if isFP {
            if len == 3 {
                switch fb {
                case 0x6F: // o
                    if textEquals(t, "oeq") { return 1 }
                    if textEquals(t, "ogt") { return 2 }
                    if textEquals(t, "oge") { return 3 }
                    if textEquals(t, "olt") { return 4 }
                    if textEquals(t, "ole") { return 5 }
                    if textEquals(t, "one") { return 6 }
                    if textEquals(t, "ord") { return 7 }
                case 0x75: // u
                    if textEquals(t, "uno") { return 8 }
                    if textEquals(t, "ueq") { return 9 }
                    if textEquals(t, "ugt") { return 10 }
                    if textEquals(t, "uge") { return 11 }
                    if textEquals(t, "ult") { return 12 }
                    if textEquals(t, "ule") { return 13 }
                    if textEquals(t, "une") { return 14 }
                default: break
                }
            }
            if textEquals(t, "false") { return 0 }
            if textEquals(t, "true") { return 15 }
            return 0
        } else {
            if len == 2 {
                if textEquals(t, "eq") { return 32 }
                if textEquals(t, "ne") { return 33 }
            } else if len == 3 {
                switch fb {
                case 0x75: // u
                    if textEquals(t, "ugt") { return 34 }
                    if textEquals(t, "uge") { return 35 }
                    if textEquals(t, "ult") { return 36 }
                    if textEquals(t, "ule") { return 37 }
                case 0x73: // s
                    if textEquals(t, "sgt") { return 38 }
                    if textEquals(t, "sge") { return 39 }
                    if textEquals(t, "slt") { return 40 }
                    if textEquals(t, "sle") { return 41 }
                default: break
                }
            }
            return 32
        }
    }

    private func attributeFromToken(_ t: Token) -> IRAttribute? {
        let len = t.end - t.start
        let fb = source[t.start]
        switch fb {
        case 0x6E: // n
            if len == 8 && textEquals(t, "nounwind") { return .noUnwind }
            if len == 6 && textEquals(t, "nofree") { return .noFree }
            if len == 6 && textEquals(t, "nosync") { return .noSync }
            if len == 10 && textEquals(t, "nocallback") { return .noCallback }
            if len == 8 && textEquals(t, "noreturn") { return .noReturn }
            if len == 8 && textEquals(t, "noinline") { return .noInline }
            if len == 9 && textEquals(t, "nocapture") { return .noCapture }
            if len == 7 && textEquals(t, "noalias") { return .noAlias }
        case 0x63: // c
            if textEquals(t, "convergent") { return .convergent }
        case 0x6D: // m
            if textEquals(t, "mustprogress") { return .mustProgress }
        case 0x77: // w
            if textEquals(t, "willreturn") { return .willReturn }
        case 0x61: // a
            if textEquals(t, "argmemonly") { return .argMemOnly }
            if textEquals(t, "alwaysinline") { return .alwaysInline }
        case 0x72: // r
            if textEquals(t, "readnone") { return .readNone }
            if textEquals(t, "readonly") { return .readOnly }
        case 0x69: // i
            if textEquals(t, "immarg") { return .immArg }
        case 0x6F: // o
            if textEquals(t, "optnone") { return .optNone }
        case 0x73: // s
            if textEquals(t, "signext") { return .signExt }
        default: break
        }
        return nil
    }
}
