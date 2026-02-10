/// Writes FUNCTION_BLOCK entries in LLVM bitcode format.
///
/// Each defined function gets its own FUNCTION_BLOCK containing:
/// - DECLAREBLOCKS: number of basic blocks
/// - CONSTANTS_BLOCK (if needed)
/// - Instructions for each basic block
/// - VALUE_SYMTAB_BLOCK for named values
final class FunctionWriter {

    // FUNCTION_BLOCK record codes
    static let declareBlocksCode: UInt64 = 1
    static let instBinopCode: UInt64 = 2
    static let instCastCode: UInt64 = 3
    static let instGEPOldCode: UInt64 = 4
    static let instSelectCode: UInt64 = 5
    static let instExtractEltCode: UInt64 = 6
    static let instInsertEltCode: UInt64 = 7
    static let instShuffleVecCode: UInt64 = 8
    static let instCmpCode: UInt64 = 9
    static let instRetCode: UInt64 = 10
    static let instBrCode: UInt64 = 11
    static let instSwitchCode: UInt64 = 12
    static let instInvokeCode: UInt64 = 13
    static let instUnreachableCode: UInt64 = 15
    static let instPhiCode: UInt64 = 16
    static let instAllocaCode: UInt64 = 19
    static let instLoadCode: UInt64 = 20
    static let instStoreOldCode: UInt64 = 24
    static let instCallCode: UInt64 = 34
    static let instExtractValCode: UInt64 = 26
    static let instInsertValCode: UInt64 = 27
    static let instCmp2Code: UInt64 = 28
    static let instVSelectCode: UInt64 = 29
    static let instGEPCode: UInt64 = 43
    static let instStoreCode: UInt64 = 44
    static let instLoadCode2: UInt64 = 41  // new load with explicit type
    static let instCallCode2: UInt64 = 45  // INST_CALL with flags

    // CONSTANTS_BLOCK record codes
    static let cstSetTypeCode: UInt64 = 1
    static let cstNullCode: UInt64 = 2
    static let cstUndefCode: UInt64 = 3
    static let cstIntegerCode: UInt64 = 4
    static let cstWideIntegerCode: UInt64 = 5
    static let cstFloatCode: UInt64 = 6
    static let cstAggregateCode: UInt64 = 7
    static let cstStringCode: UInt64 = 8
    static let cstCStringCode: UInt64 = 9
    static let cstCEBinopCode: UInt64 = 10
    static let cstCECastCode: UInt64 = 11
    static let cstCEGEPCode: UInt64 = 12
    static let cstDataCode: UInt64 = 22

    // VALUE_SYMTAB record codes
    static let vstEntryCode: UInt64 = 1
    static let vstBBEntryCode: UInt64 = 2

    // Block IDs
    static let functionBlockID: UInt64 = 12
    static let constantsBlockID: UInt64 = 11
    static let valueSymtabBlockID: UInt64 = 14

    /// Write a FUNCTION_BLOCK for a defined function.
    static func write(
        to writer: BitstreamWriter,
        function: IRFunction,
        enumerator: ValueEnumerator
    ) {
        guard !function.isDeclaration else { return }

        writer.enterSubblock(blockID: functionBlockID, abbrevLen: 5)

        // DECLAREBLOCKS: number of basic blocks
        writer.emitUnabbrevRecord(code: declareBlocksCode, operands: [
            UInt64(function.basicBlocks.count)
        ])

        // Build the local value table for this function:
        // Global values first (inherited), then parameters, then instruction results
        var localValues: [String: Int] = [:]
        var nextValueID = enumerator.globalValueCount

        // Parameters get value IDs
        for (i, param) in function.parameters.enumerated() {
            let name = param.name.isEmpty ? "\(i)" : param.name
            localValues[name] = nextValueID
            nextValueID += 1
        }

        // First pass: assign value IDs to all instructions that produce values
        for bb in function.basicBlocks {
            for inst in bb.instructions {
                if !inst.type.isVoid && !inst.name.isEmpty {
                    localValues[inst.name] = nextValueID
                }
                if !inst.type.isVoid {
                    nextValueID += 1
                }
            }
        }

        // Collect constants used in this function
        var constants: [(IRType, IRConstant)] = []
        var constantValueMap: [String: Int] = [:]  // key → valueID

        func collectConstants(_ inst: IRInstruction) {
            for op in inst.operands {
                if case .constant(let c) = op {
                    let key = "\(c)"
                    if constantValueMap[key] == nil {
                        constantValueMap[key] = nextValueID
                        constants.append((c.type, c))
                        nextValueID += 1
                    }
                }
            }
        }

        for bb in function.basicBlocks {
            for inst in bb.instructions {
                collectConstants(inst)
            }
        }

        // Emit CONSTANTS_BLOCK if needed
        if !constants.isEmpty {
            writeConstantsBlock(to: writer, constants: constants, enumerator: enumerator)
        }

        // Emit instructions
        var currentValueID = enumerator.globalValueCount + function.parameters.count

        for bb in function.basicBlocks {
            for inst in bb.instructions {
                writeInstruction(
                    inst,
                    to: writer,
                    enumerator: enumerator,
                    localValues: localValues,
                    constantValueMap: constantValueMap,
                    currentValueID: currentValueID
                )
                if !inst.type.isVoid {
                    currentValueID += 1
                }
            }
        }

        // Emit VALUE_SYMTAB for named values
        writeValueSymtab(
            to: writer,
            function: function,
            localValues: localValues,
            enumerator: enumerator
        )

        writer.exitBlock()
    }

    // MARK: - Constants block

    private static func writeConstantsBlock(
        to writer: BitstreamWriter,
        constants: [(IRType, IRConstant)],
        enumerator: ValueEnumerator
    ) {
        writer.enterSubblock(blockID: constantsBlockID, abbrevLen: 5)

        var currentType: IRType? = nil

        for (type, constant) in constants {
            // Emit SETTYPE if type changed
            if type != currentType {
                writer.emitUnabbrevRecord(code: cstSetTypeCode, operands: [
                    UInt64(enumerator.typeIndex(type))
                ])
                currentType = type
            }

            switch constant {
            case .integer(_, let value):
                // Encode as signed VBR: positive → 2*v, negative → (-2*v)+1
                let encoded: UInt64
                if value >= 0 {
                    encoded = UInt64(value) << 1
                } else {
                    encoded = (UInt64(bitPattern: -value) << 1) | 1
                }
                writer.emitUnabbrevRecord(code: cstIntegerCode, operands: [encoded])

            case .float32(let value):
                writer.emitUnabbrevRecord(code: cstFloatCode, operands: [
                    UInt64(value.bitPattern)
                ])

            case .float64(let value):
                writer.emitUnabbrevRecord(code: cstFloatCode, operands: [
                    UInt64(value.bitPattern)
                ])

            case .float16(let bits):
                writer.emitUnabbrevRecord(code: cstFloatCode, operands: [
                    UInt64(bits)
                ])

            case .null:
                writer.emitUnabbrevRecord(code: cstNullCode, operands: [])

            case .undef:
                writer.emitUnabbrevRecord(code: cstUndefCode, operands: [])

            case .zeroInitializer:
                writer.emitUnabbrevRecord(code: cstNullCode, operands: [])

            default:
                // For complex constants, emit as null for now
                // TODO: handle struct/array/vector constants, constant expressions
                writer.emitUnabbrevRecord(code: cstNullCode, operands: [])
            }
        }

        writer.exitBlock()
    }

    // MARK: - Instructions

    private static func writeInstruction(
        _ inst: IRInstruction,
        to writer: BitstreamWriter,
        enumerator: ValueEnumerator,
        localValues: [String: Int],
        constantValueMap: [String: Int],
        currentValueID: Int
    ) {
        // Helper to resolve an operand to a value ID
        func resolveOperand(_ op: IRInstruction.Operand) -> UInt64 {
            switch op {
            case .value(let val):
                // Check local values first
                if !val.name.isEmpty, let id = localValues[val.name] {
                    return UInt64(id)
                }
                // Check global values
                if let id = enumerator.globalValueID(name: val.name) {
                    return UInt64(id)
                }
                return 0

            case .constant(let c):
                let key = "\(c)"
                if let id = constantValueMap[key] {
                    return UInt64(id)
                }
                return 0

            case .basicBlock:
                // BB reference - not used as value ID here
                return 0

            case .intLiteral(let val):
                return UInt64(bitPattern: val)

            case .type(let ty):
                return UInt64(enumerator.typeIndex(ty))

            case .metadata:
                return 0
            }
        }

        // Helper to compute relative value ID (for forward references)
        func relativeID(_ absoluteID: UInt64) -> UInt64 {
            return UInt64(currentValueID) &- absoluteID
        }

        switch inst.opcode {
        case .ret:
            if inst.type.isVoid && inst.operands.isEmpty {
                // void return
                writer.emitUnabbrevRecord(code: instRetCode, operands: [])
            } else if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitUnabbrevRecord(code: instRetCode, operands: [relativeID(valID)])
            }

        case .br:
            if inst.operands.count == 1 {
                // Unconditional branch
                // operand is a basic block index
                if case .basicBlock = inst.operands[0] {
                    // Find BB index in function
                    writer.emitUnabbrevRecord(code: instBrCode, operands: [0])
                } else {
                    writer.emitUnabbrevRecord(code: instBrCode, operands: [0])
                }
            }

        case .alloca:
            // INST_ALLOCA: [type, opty, op, align]
            let allocaType = inst.attributes.allocaType ?? inst.type
            let operands: [UInt64] = [
                UInt64(enumerator.typeIndex(allocaType)),
                UInt64(enumerator.typeIndex(.i32)), // count type
                0, // count value (1)
                UInt64(log2Align(inst.attributes.alignment ?? 1))
            ]
            writer.emitUnabbrevRecord(code: instAllocaCode, operands: operands)

        case .load:
            // INST_LOAD: [val, opty, align, vol]
            if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitUnabbrevRecord(code: instLoadCode2, operands: [
                    relativeID(valID),
                    UInt64(enumerator.typeIndex(inst.type)),
                    UInt64(log2Align(inst.attributes.alignment ?? 1)),
                    inst.attributes.isVolatile ? 1 : 0
                ])
            }

        case .store:
            // INST_STORE: [val, ptr, align, vol]
            if inst.operands.count >= 2 {
                let valID = resolveOperand(inst.operands[0])
                let ptrID = resolveOperand(inst.operands[1])
                writer.emitUnabbrevRecord(code: instStoreCode, operands: [
                    relativeID(ptrID),
                    relativeID(valID),
                    UInt64(log2Align(inst.attributes.alignment ?? 1)),
                    inst.attributes.isVolatile ? 1 : 0
                ])
            }

        case .getelementptr:
            // INST_GEP: [inbounds, type, base, ...indices]
            var operands: [UInt64] = [
                inst.attributes.inBounds ? 1 : 0,
                UInt64(enumerator.typeIndex(inst.attributes.gepSourceType ?? inst.type))
            ]
            for op in inst.operands {
                let valID = resolveOperand(op)
                operands.append(relativeID(valID))
            }
            writer.emitUnabbrevRecord(code: instGEPCode, operands: operands)

        case .bitcast, .zext, .sext, .trunc, .fpToUI, .fpToSI, .uiToFP, .siToFP,
             .fpTrunc, .fpExt, .ptrToInt, .intToPtr, .addrSpaceCast:
            // INST_CAST: [val, destty, castopc]
            if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitUnabbrevRecord(code: instCastCode, operands: [
                    relativeID(valID),
                    UInt64(enumerator.typeIndex(inst.type)),
                    UInt64(castOpcode(inst.opcode))
                ])
            }

        case .add, .fadd, .sub, .fsub, .mul, .fmul, .udiv, .sdiv, .fdiv,
             .urem, .srem, .frem, .shl, .lshr, .ashr, .and, .or, .xor:
            // INST_BINOP: [val1, val2, opcode]
            if inst.operands.count >= 2 {
                let lhs = resolveOperand(inst.operands[0])
                let rhs = resolveOperand(inst.operands[1])
                writer.emitUnabbrevRecord(code: instBinopCode, operands: [
                    relativeID(lhs),
                    relativeID(rhs),
                    UInt64(binopOpcode(inst.opcode))
                ])
            }

        case .icmp, .fcmp:
            // INST_CMP2: [val1, val2, predicate]
            if inst.operands.count >= 2 {
                let lhs = resolveOperand(inst.operands[0])
                let rhs = resolveOperand(inst.operands[1])
                writer.emitUnabbrevRecord(code: instCmp2Code, operands: [
                    relativeID(lhs),
                    relativeID(rhs),
                    UInt64(inst.attributes.predicate ?? 0)
                ])
            }

        case .call:
            // INST_CALL: [paramattr, cc, fmf, fnty, fnid, ...args]
            var operands: [UInt64] = []
            // Param attribute list ID (0 = no attributes)
            operands.append(0)
            // Calling convention flags:
            // bit 0: tail call
            // bit 1-3: calling convention
            // bit 15: explicit type
            var flags: UInt64 = 0
            if inst.attributes.tailCall == .tail { flags |= 1 }
            if inst.attributes.tailCall == .mustTail { flags |= 1 }
            flags |= (1 << 15) // explicit type
            operands.append(flags)

            // Function type
            if let fnOp = inst.operands.last {
                if case .value(let fnVal) = fnOp, case .pointer(let pointee, _) = fnVal.type {
                    if case .function = pointee {
                        let fnTypeIdx = enumerator.typeIndex(pointee)
                        operands.append(UInt64(fnTypeIdx))
                    } else {
                        operands.append(0)
                    }
                } else {
                    operands.append(0)
                }
            }

            // Function value ID
            if let fnOp = inst.operands.last {
                let fnID = resolveOperand(fnOp)
                operands.append(relativeID(fnID))
            }

            // Arguments (all operands except the last which is the function)
            for i in 0..<(inst.operands.count - 1) {
                let argID = resolveOperand(inst.operands[i])
                operands.append(relativeID(argID))
            }

            writer.emitUnabbrevRecord(code: instCallCode, operands: operands)

        case .phi:
            // INST_PHI: [ty, val0, bb0, val1, bb1, ...]
            var operands: [UInt64] = [UInt64(enumerator.typeIndex(inst.type))]
            // PHI operands come in (value, basicblock) pairs
            var i = 0
            while i + 1 < inst.operands.count {
                let valID = resolveOperand(inst.operands[i])
                operands.append(relativeID(valID))
                // BB index
                if case .basicBlock = inst.operands[i+1] {
                    operands.append(0) // simplified
                } else {
                    operands.append(0)
                }
                i += 2
            }
            writer.emitUnabbrevRecord(code: instPhiCode, operands: operands)

        case .select:
            if inst.operands.count >= 3 {
                let cond = resolveOperand(inst.operands[0])
                let trueVal = resolveOperand(inst.operands[1])
                let falseVal = resolveOperand(inst.operands[2])
                writer.emitUnabbrevRecord(code: instSelectCode, operands: [
                    relativeID(trueVal),
                    relativeID(falseVal),
                    relativeID(cond)
                ])
            }

        case .unreachable:
            writer.emitUnabbrevRecord(code: instUnreachableCode, operands: [])

        default:
            // Unsupported instruction - emit as unreachable
            writer.emitUnabbrevRecord(code: instUnreachableCode, operands: [])
        }
    }

    // MARK: - Value Symbol Table

    private static func writeValueSymtab(
        to writer: BitstreamWriter,
        function: IRFunction,
        localValues: [String: Int],
        enumerator: ValueEnumerator
    ) {
        // Collect named values
        var entries: [(Int, String)] = []

        // Parameters
        for param in function.parameters {
            if !param.name.isEmpty {
                if let id = localValues[param.name] {
                    entries.append((id, param.name))
                }
            }
        }

        // Instructions
        for bb in function.basicBlocks {
            for inst in bb.instructions {
                if !inst.name.isEmpty && !inst.type.isVoid {
                    if let id = localValues[inst.name] {
                        entries.append((id, inst.name))
                    }
                }
            }
        }

        // Basic blocks
        for (i, bb) in function.basicBlocks.enumerated() {
            if !bb.name.isEmpty && bb.name != "entry" && bb.name != "\(i)" {
                entries.append((i, bb.name))
            }
        }

        guard !entries.isEmpty else { return }

        writer.enterSubblock(blockID: valueSymtabBlockID, abbrevLen: 4)

        for (id, name) in entries {
            let nameBytes = Array(name.utf8)
            var operands: [UInt64] = [UInt64(id)]
            for b in nameBytes {
                operands.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: vstEntryCode, operands: operands)
        }

        // BB entries
        for (i, bb) in function.basicBlocks.enumerated() {
            if !bb.name.isEmpty {
                let nameBytes = Array(bb.name.utf8)
                var operands: [UInt64] = [UInt64(i)]
                for b in nameBytes {
                    operands.append(UInt64(b))
                }
                writer.emitUnabbrevRecord(code: vstBBEntryCode, operands: operands)
            }
        }

        writer.exitBlock()
    }

    // MARK: - Helpers

    /// Convert alignment to log2 encoding used in bitcode.
    private static func log2Align(_ align: Int) -> Int {
        if align <= 1 { return 0 }
        var n = 0
        var v = align
        while v > 1 {
            v >>= 1
            n += 1
        }
        return n + 1  // bitcode alignment is log2(align) + 1, 0 means unspecified
    }

    /// Map instruction opcode to LLVM bitcode binop opcode.
    private static func binopOpcode(_ opcode: IRInstruction.Opcode) -> Int {
        switch opcode {
        case .add: return 0
        case .sub: return 1
        case .mul: return 2
        case .udiv: return 3
        case .sdiv: return 4
        case .urem: return 5
        case .srem: return 6
        case .shl: return 7
        case .lshr: return 8
        case .ashr: return 9
        case .and: return 10
        case .or: return 11
        case .xor: return 12
        case .fadd: return 0  // FP versions use same codes
        case .fsub: return 1
        case .fmul: return 2
        case .fdiv: return 4
        case .frem: return 6
        default: return 0
        }
    }

    /// Map instruction opcode to LLVM bitcode cast opcode.
    private static func castOpcode(_ opcode: IRInstruction.Opcode) -> Int {
        switch opcode {
        case .trunc: return 0
        case .zext: return 1
        case .sext: return 2
        case .fpToUI: return 3
        case .fpToSI: return 4
        case .uiToFP: return 5
        case .siToFP: return 6
        case .fpTrunc: return 7
        case .fpExt: return 8
        case .ptrToInt: return 9
        case .intToPtr: return 10
        case .bitcast: return 11
        case .addrSpaceCast: return 12
        default: return 11
        }
    }
}
