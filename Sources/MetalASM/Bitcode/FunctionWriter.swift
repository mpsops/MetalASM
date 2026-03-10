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
    static let cstCEGEPOldCode: UInt64 = 12
    static let cstCEInboundsGEPCode: UInt64 = 20
    static let cstDataCode: UInt64 = 22

    // VALUE_SYMTAB record codes
    static let vstEntryCode: UInt64 = 1
    static let vstBBEntryCode: UInt64 = 2

    // Block IDs
    static let functionBlockID: UInt64 = 12
    static let constantsBlockID: UInt64 = 11
    static let valueSymtabBlockID: UInt64 = 14

    // Abbreviation IDs (assigned sequentially from 4)
    // These are defined per FUNCTION_BLOCK and provide compact encoding
    // for the most common instruction patterns.
    static let abbrevBinop: UInt64 = 4      // INST_BINOP: [relID, relID, opcode]
    static let abbrevCast: UInt64 = 5       // INST_CAST: [relID, typeID, castop]
    static let abbrevExtractElt: UInt64 = 6 // INST_EXTRACTELT: [relID, relID]
    static let abbrevInsertElt: UInt64 = 7  // INST_INSERTELT: [relID, relID, relID]
    static let abbrevLoad: UInt64 = 8       // INST_LOAD: [relID, typeID, align, volatile]
    static let abbrevStore: UInt64 = 9      // INST_STORE: [relID, relID, align, volatile]
    static let abbrevCmp: UInt64 = 10       // INST_CMP2: [relID, relID, predicate]
    static let abbrevSelect: UInt64 = 11    // INST_SELECT: [relID, relID, relID]
    static let abbrevBrUnc: UInt64 = 12     // INST_BR unconditional: [bbIdx]
    static let abbrevBrCond: UInt64 = 13    // INST_BR conditional: [bbIdx, bbIdx, relID]
    static let abbrevRet: UInt64 = 14       // INST_RET void: []
    static let abbrevPhi2: UInt64 = 15      // INST_PHI 2-entry: [typeID, val, bb, val, bb]
    static let abbrevAlloca: UInt64 = 16   // INST_ALLOCA: [typeID, cntTypeID, cntID, align]
    static let abbrevGEP: UInt64 = 17      // INST_GEP: [inbounds, typeID, ...relIDs] (variable via Array)
    static let abbrevShuffleVec: UInt64 = 18 // INST_SHUFFLEVEC: [relID, relID, relID]
    static let abbrevRetVal: UInt64 = 19   // INST_RET with value: [relID]
    static let abbrevUnreachable: UInt64 = 20 // INST_UNREACHABLE: []
    static let abbrevCall: UInt64 = 21     // INST_CALL: [paramattr, flags, fnty, fnid, ...args] via Array

    /// Define abbreviations for common instruction patterns.
    /// Must be called right after entering FUNCTION_BLOCK.
    private static func defineAbbreviations(to writer: BitstreamWriter) {
        // Abbrev 4: BINOP [code=Fixed6, relID=VBR8, relID=VBR8, opcode=Fixed4]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instBinopCode)),   // literal: code = 2
            (2, 8),                        // VBR8: lhs relID
            (2, 8),                        // VBR8: rhs relID
            (1, 4),                        // Fixed4: binop opcode (0-12)
        ])
        // Abbrev 5: CAST [code=Fixed6, relID=VBR8, typeID=VBR6, castop=Fixed4]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instCastCode)),
            (2, 8),                        // VBR8: operand relID
            (2, 6),                        // VBR6: dest type ID
            (1, 4),                        // Fixed4: cast opcode (0-12)
        ])
        // Abbrev 6: EXTRACTELT [code, relID, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instExtractEltCode)),
            (2, 8),
            (2, 8),
        ])
        // Abbrev 7: INSERTELT [code, relID, relID, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instInsertEltCode)),
            (2, 8),
            (2, 8),
            (2, 8),
        ])
        // Abbrev 8: LOAD [code, relID, typeID, align, volatile]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instLoadCode)),
            (2, 8),                        // VBR8: ptr relID
            (2, 6),                        // VBR6: type ID
            (1, 4),                        // Fixed4: align
            (1, 1),                        // Fixed1: volatile
        ])
        // Abbrev 9: STORE [code, relID, relID, align, volatile]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instStoreCode)),
            (2, 8),
            (2, 8),
            (1, 4),                        // Fixed4: align
            (1, 1),                        // Fixed1: volatile
        ])
        // Abbrev 10: CMP2 [code, relID, relID, predicate]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instCmp2Code)),
            (2, 8),
            (2, 8),
            (1, 6),                        // Fixed6: predicate
        ])
        // Abbrev 11: SELECT [code, relID, relID, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instSelectCode)),
            (2, 8),
            (2, 8),
            (2, 8),
        ])
        // Abbrev 12: BR unconditional [code, bbIdx]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instBrCode)),
            (2, 6),                        // VBR6: bb index
        ])
        // Abbrev 13: BR conditional [code, bbIdx, bbIdx, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instBrCode)),
            (2, 6),
            (2, 6),
            (2, 8),
        ])
        // Abbrev 14: RET void [code]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instRetCode)),
        ])
        // Abbrev 15: PHI 2-entry [code, typeID, sval, bb, sval, bb]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instPhiCode)),
            (2, 6),                        // VBR6: type ID
            (2, 8),                        // VBR8: val0 (signed encoded)
            (2, 6),                        // VBR6: bb0
            (2, 8),                        // VBR8: val1
            (2, 6),                        // VBR6: bb1
        ])
        // Abbrev 16: ALLOCA [code, typeID, cntTypeID, cntID, align]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instAllocaCode)),
            (2, 6),                        // VBR6: alloca type ID
            (2, 6),                        // VBR6: count type ID
            (2, 8),                        // VBR8: count value ID
            (2, 6),                        // VBR6: align encoded
        ])
        // Abbrev 17: GEP [code, inbounds, typeID, array of VBR8 relIDs]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instGEPCode)),
            (1, 1),                        // Fixed1: inbounds
            (2, 6),                        // VBR6: source type ID
            (3, nil),                      // Array
            (2, 8),                        // VBR8: each relID
        ])
        // Abbrev 18: SHUFFLEVEC [code, relID, relID, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instShuffleVecCode)),
            (2, 8),
            (2, 8),
            (2, 8),
        ])
        // Abbrev 19: RET with value [code, relID]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instRetCode)),
            (2, 8),                        // VBR8: value relID
        ])
        // Abbrev 20: UNREACHABLE [code]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instUnreachableCode)),
        ])
        // Abbrev 21: CALL [code, array of VBR8]
        writer.emitDefineAbbrev(operandEncodings: [
            (0, UInt64(instCallCode)),
            (3, nil),                      // Array
            (2, 8),                        // VBR8: each operand
        ])
    }

    /// Write a FUNCTION_BLOCK for a defined function.
    static func write(
        to writer: BitstreamWriter,
        function: IRFunction,
        enumerator: ValueEnumerator,
        moduleConstantCount: Int = 0
    ) {
        guard !function.isDeclaration else { return }

        writer.enterSubblock(blockID: functionBlockID, abbrevLen: 5)

        // Define abbreviations for compact encoding
        defineAbbreviations(to: writer)

        // DECLAREBLOCKS: number of basic blocks
        writer.emitUnabbrevRecord(code: declareBlocksCode, operands: [
            UInt64(function.basicBlocks.count)
        ])

        // Build the local value table for this function.
        // LLVM bitcode value ID order: globals → module_constants → params → fn_constants → instruction results
        var localValues: [String: Int] = [:]
        var nextValueID = enumerator.globalValueCount + moduleConstantCount

        // Parameters get value IDs first
        for (i, param) in function.parameters.enumerated() {
            let name = param.name.isEmpty ? "\(i)" : param.name
            localValues[name] = nextValueID
            nextValueID += 1
        }

        // Collect constants BEFORE assigning instruction IDs
        // (constants block is emitted before instructions, so constants get lower IDs)
        var constants: [(IRType, IRConstant)] = []
        var constantValueMap: [IRConstant: Int] = [:]

        func registerConstant(_ c: IRConstant) {
            if constantValueMap[c] != nil { return }

            // First, recursively register sub-constants of constant expressions
            switch c {
            case .bitcast(let inner, _), .inttoptr(let inner, _), .ptrtoint(let inner, _):
                registerConstant(inner)
            case .getelementptr(_, _, let base, let indices):
                registerConstant(base)
                for idx in indices { registerConstant(idx) }
            case .arrayValue(_, let elems), .structValue(_, let elems):
                for elem in elems { registerConstant(elem) }
            case .vectorValue:
                // Vector elements are encoded inline via DATA record; don't register them separately
                break
            default:
                break
            }

            // Then register this constant (sub-constants get lower IDs)
            if constantValueMap[c] == nil {
                constantValueMap[c] = nextValueID
                constants.append((c.type, c))
                nextValueID += 1
            }
        }

        for bb in function.basicBlocks {
            for inst in bb.instructions {
                for op in inst.operands {
                    if case .constant(let c) = op {
                        registerConstant(c)
                    }
                }
                // Alloca needs an implicit i32 1 count constant
                if inst.opcode == .alloca {
                    registerConstant(.integer(.i32, 1))
                }
            }
        }

        // Now assign instruction result IDs (after constants)
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

        // Emit CONSTANTS_BLOCK if needed
        if !constants.isEmpty {
            writeConstantsBlock(to: writer, constants: constants, constantValueMap: constantValueMap, enumerator: enumerator)
        }

        // Build BB name → index mapping
        var bbIndexMap: [String: Int] = [:]
        for (i, bb) in function.basicBlocks.enumerated() {
            if !bb.name.isEmpty {
                bbIndexMap[bb.name] = i
            }
        }

        // Emit instructions
        // currentValueID starts after globals + module_constants + params + fn_constants
        var currentValueID = enumerator.globalValueCount + moduleConstantCount + function.parameters.count + constants.count

        for bb in function.basicBlocks {
            for inst in bb.instructions {
                writeInstruction(
                    inst,
                    to: writer,
                    enumerator: enumerator,
                    localValues: localValues,
                    constantValueMap: constantValueMap,
                    currentValueID: currentValueID,
                    bbIndexMap: bbIndexMap
                )
                if !inst.type.isVoid {
                    currentValueID += 1
                }
            }
        }

        // Skip VALUE_SYMTAB — Metal doesn't need SSA names for pipeline creation.
        // This saves ~20-30% of bitcode size for large kernels.

        writer.exitBlock()
    }

    // MARK: - Constants block

    private static func writeConstantsBlock(
        to writer: BitstreamWriter,
        constants: [(IRType, IRConstant)],
        constantValueMap: [IRConstant: Int],
        enumerator: ValueEnumerator
    ) {
        writer.enterSubblock(blockID: constantsBlockID, abbrevLen: 5)

        var currentType: IRType? = nil

        func resolveConstant(_ c: IRConstant) -> UInt64 {
            if let id = constantValueMap[c] {
                return UInt64(id)
            }
            return 0
        }

        for (type, constant) in constants {
            if type != currentType {
                writer.emitUnabbrevRecord(code: cstSetTypeCode, UInt64(enumerator.typeIndex(type)))
                currentType = type
            }

            switch constant {
            case .integer(_, let value):
                let encoded: UInt64
                if value >= 0 {
                    encoded = UInt64(value) << 1
                } else {
                    encoded = (UInt64(bitPattern: -value) << 1) | 1
                }
                writer.emitUnabbrevRecord(code: cstIntegerCode, encoded)

            case .float32(let value):
                writer.emitUnabbrevRecord(code: cstFloatCode, UInt64(value.bitPattern))

            case .float64(let value):
                writer.emitUnabbrevRecord(code: cstFloatCode, UInt64(value.bitPattern))

            case .float16(let bits):
                writer.emitUnabbrevRecord(code: cstFloatCode, UInt64(bits))

            case .bfloat16(let bits):
                writer.emitUnabbrevRecord(code: cstFloatCode, UInt64(bits))

            case .null:
                writer.emitUnabbrevRecord(code: cstNullCode)

            case .undef:
                writer.emitUnabbrevRecord(code: cstUndefCode)

            case .zeroInitializer:
                writer.emitUnabbrevRecord(code: cstNullCode)

            case .bitcast(let inner, let destTy):
                // CE_CAST: [opcode, desttype, val]
                writer.emitUnabbrevRecord(code: cstCECastCode, operands: [
                    11,  // bitcast opcode
                    UInt64(enumerator.typeIndex(destTy)),
                    resolveConstant(inner)
                ])

            case .inttoptr(let inner, let destTy):
                // CE_CAST: [opcode, desttype, val]
                writer.emitUnabbrevRecord(code: cstCECastCode, operands: [
                    10,  // inttoptr opcode
                    UInt64(enumerator.typeIndex(destTy)),
                    resolveConstant(inner)
                ])

            case .ptrtoint(let inner, let destTy):
                // CE_CAST: [opcode, desttype, val]
                writer.emitUnabbrevRecord(code: cstCECastCode, operands: [
                    9,   // ptrtoint opcode
                    UInt64(enumerator.typeIndex(destTy)),
                    resolveConstant(inner)
                ])

            case .getelementptr(let inBounds, let srcTy, let base, let indices):
                // CE_INBOUNDS_GEP (code 20): [ty, (type, val), (type, val), ...]
                let code = inBounds ? cstCEInboundsGEPCode : cstCEGEPOldCode
                var operands: [UInt64] = [UInt64(enumerator.typeIndex(srcTy))]
                // Base: type + value
                operands.append(UInt64(enumerator.typeIndex(base.type)))
                operands.append(resolveConstant(base))
                // Indices: type + value each
                for idx in indices {
                    operands.append(UInt64(enumerator.typeIndex(idx.type)))
                    operands.append(resolveConstant(idx))
                }
                writer.emitUnabbrevRecord(code: code, operands: operands)

            case .vectorValue(_, let elems):
                // Use CST_CODE_DATA (code 22) for integer/float vectors: stores element values
                // directly without value ID references, matching metal-as output.
                var operands: [UInt64] = []
                for elem in elems {
                    switch elem {
                    case .integer(_, let v):
                        operands.append(v >= 0 ? UInt64(v) : UInt64(bitPattern: v))
                    case .float32(let f):
                        operands.append(UInt64(f.bitPattern))
                    case .float16(let bits):
                        operands.append(UInt64(bits))
                    case .bfloat16(let bits):
                        operands.append(UInt64(bits))
                    default:
                        operands.append(resolveConstant(elem))
                    }
                }
                writer.emitUnabbrevRecord(code: cstDataCode, operands: operands)

            case .arrayValue(_, let elems):
                var operands: [UInt64] = []
                for elem in elems {
                    operands.append(resolveConstant(elem))
                }
                writer.emitUnabbrevRecord(code: cstAggregateCode, operands: operands)

            case .structValue(_, let elems):
                var operands: [UInt64] = []
                for elem in elems {
                    operands.append(resolveConstant(elem))
                }
                writer.emitUnabbrevRecord(code: cstAggregateCode, operands: operands)

            default:
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
        constantValueMap: [IRConstant: Int],
        currentValueID: Int,
        bbIndexMap: [String: Int]
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
                if let id = constantValueMap[c] {
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
                writer.emitAbbreviatedRecord(abbrevID: abbrevRet, operands: [])
            } else if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitAbbreviatedRecord(abbrevID: abbrevRetVal, operands: [relativeID(valID)])
            }

        case .br:
            if inst.operands.count == 1 {
                let bbIdx: UInt64
                if case .basicBlock(let bb) = inst.operands[0] {
                    bbIdx = UInt64(bbIndexMap[bb.name] ?? 0)
                } else { bbIdx = 0 }
                writer.emitAbbreviatedRecord(abbrevID: abbrevBrUnc, operands: [bbIdx])
            } else if inst.operands.count == 3 {
                let trueBBIdx: UInt64
                let falseBBIdx: UInt64
                if case .basicBlock(let bb) = inst.operands[1] {
                    trueBBIdx = UInt64(bbIndexMap[bb.name] ?? 0)
                } else { trueBBIdx = 0 }
                if case .basicBlock(let bb) = inst.operands[2] {
                    falseBBIdx = UInt64(bbIndexMap[bb.name] ?? 0)
                } else { falseBBIdx = 0 }
                let condID = resolveOperand(inst.operands[0])
                writer.emitAbbreviatedRecord(abbrevID: abbrevBrCond, operands: [trueBBIdx, falseBBIdx, relativeID(condID)])
            }

        case .alloca:
            // INST_ALLOCA: [insttype, opty, op, align]
            // align encoding: (log2(align)+1) | (inalloca << 5) | (explicittype << 6) | (swifterror << 7)
            let allocaType = inst.attributes.allocaType ?? inst.type
            // The count operand is a constant i32 1 — find it in the constant map
            let countID: UInt64
            if let cid = constantValueMap[.integer(.i32, 1)] {
                countID = UInt64(cid)  // ALLOCA uses absolute value IDs, not relative
            } else {
                // Fallback: look for any i32 constant with value 1
                countID = 0
            }
            let alignEncoded = UInt64(log2Align(inst.attributes.alignment ?? 1)) | (1 << 6) // explicit type flag
            let operands: [UInt64] = [
                UInt64(enumerator.typeIndex(allocaType)),
                UInt64(enumerator.typeIndex(.i32)), // count type
                countID,
                alignEncoded
            ]
            writer.emitAbbreviatedRecord(abbrevID: abbrevAlloca, operands: operands)

        case .load:
            if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitAbbreviatedRecord(abbrevID: abbrevLoad, operands: [
                    relativeID(valID),
                    UInt64(enumerator.typeIndex(inst.type)),
                    UInt64(log2Align(inst.attributes.alignment ?? 1)),
                    inst.attributes.isVolatile ? 1 : 0])
            }

        case .store:
            if inst.operands.count >= 2 {
                let valID = resolveOperand(inst.operands[0])
                let ptrID = resolveOperand(inst.operands[1])
                writer.emitAbbreviatedRecord(abbrevID: abbrevStore, operands: [
                    relativeID(ptrID), relativeID(valID),
                    UInt64(log2Align(inst.attributes.alignment ?? 1)),
                    inst.attributes.isVolatile ? 1 : 0])
            }

        case .getelementptr:
            // INST_GEP: [inbounds, typeID, relBase, relIdx1, ...]
            var operands: [UInt64] = [
                inst.attributes.inBounds ? 1 : 0,
                UInt64(enumerator.typeIndex(inst.attributes.gepSourceType ?? inst.type))
            ]
            for op in inst.operands {
                let valID = resolveOperand(op)
                operands.append(relativeID(valID))
            }
            writer.emitAbbreviatedRecord(abbrevID: abbrevGEP, operands: operands)

        case .bitcast, .zext, .sext, .trunc, .fpToUI, .fpToSI, .uiToFP, .siToFP,
             .fpTrunc, .fpExt, .ptrToInt, .intToPtr, .addrSpaceCast:
            if let op = inst.operands.first {
                let valID = resolveOperand(op)
                writer.emitAbbreviatedRecord(abbrevID: abbrevCast, operands: [
                    relativeID(valID),
                    UInt64(enumerator.typeIndex(inst.type)),
                    UInt64(castOpcode(inst.opcode))])
            }

        case .add, .fadd, .sub, .fsub, .mul, .fmul, .udiv, .sdiv, .fdiv,
             .urem, .srem, .frem, .shl, .lshr, .ashr, .and, .or, .xor:
            if inst.operands.count >= 2 {
                let lhs = resolveOperand(inst.operands[0])
                let rhs = resolveOperand(inst.operands[1])
                let isFP = inst.opcode == .fadd || inst.opcode == .fsub ||
                           inst.opcode == .fmul || inst.opcode == .fdiv || inst.opcode == .frem
                // FP binops require a fast-math flags field (0 = no flags); integer binops do not.
                var binopOps: [UInt64] = [relativeID(lhs), relativeID(rhs), UInt64(binopOpcode(inst.opcode))]
                if isFP { binopOps.append(0) }
                writer.emitUnabbrevRecord(code: Self.instBinopCode, operands: binopOps)
            }

        case .icmp, .fcmp:
            if inst.operands.count >= 2 {
                let lhs = resolveOperand(inst.operands[0])
                let rhs = resolveOperand(inst.operands[1])
                writer.emitAbbreviatedRecord(abbrevID: abbrevCmp, operands: [
                    relativeID(lhs), relativeID(rhs), UInt64(inst.attributes.predicate ?? 0)])
            }

        case .call:
            // INST_CALL: [paramattr, cc_flags, fnty, fnid, ...args, [bundle_count]]
            var operands: [UInt64] = []
            // Param attribute list ID (0 = no attributes).
            if let groupIdx = inst.attributes.funcAttributes.first {
                operands.append(UInt64(groupIdx + 1))
            } else {
                operands.append(0)
            }

            // Detect AIR MMA intrinsic calls — they require operand bundle encoding
            var calleeName: String? = nil
            if let fnOp = inst.operands.last, case .value(let fnVal) = fnOp {
                calleeName = fnVal.name
            }
            // MMA load/multiply_accumulate use operand bundle encoding (bit17 + sentinel 254)
            // MMA store uses normal call encoding (no bit17, no sentinel)
            let isMMAWithBundles = calleeName.map {
                $0.hasPrefix("air.simdgroup_matrix_8x8_load") ||
                $0.hasPrefix("air.simdgroup_matrix_8x8_multiply_accumulate")
            } ?? false
            let isMMAStore = calleeName.map {
                $0.hasPrefix("air.simdgroup_matrix_8x8_store")
            } ?? false

            // Calling convention flags:
            // bit 0: tail call
            // bits 13:1: calling convention
            // bit 14: must-tail
            // bit 15: explicit type
            // bit 17: hasOperandBundles (required for AIR MMA load/mul, NOT store)
            var flags: UInt64 = 0
            if inst.attributes.tailCall == .tail || inst.attributes.tailCall == .mustTail { flags |= 1 }
            if inst.attributes.tailCall == .mustTail { flags |= (1 << 14) }
            flags |= (1 << 15) // explicit type
            if isMMAWithBundles {
                flags |= (1 << 17) | 1  // hasOperandBundles + tail
            }
            if isMMAStore {
                flags |= 1  // tail (reference Metal compiler always sets tail for MMA store)
            }
            operands.append(flags)

            // Function type: 254 sentinel for MMA load/mul, direct type for everything else
            if isMMAWithBundles {
                operands.append(254)
                // After sentinel, emit the actual function type index
                if let fnOp = inst.operands.last, case .value(let fnVal) = fnOp,
                   let fnDecl = enumerator.findFunction(named: fnVal.name) {
                    if case .pointer(let pointee, _) = fnDecl.type, case .function = pointee {
                        operands.append(UInt64(enumerator.typeIndex(pointee)))
                    } else if case .function = fnDecl.type {
                        operands.append(UInt64(enumerator.typeIndex(fnDecl.type)))
                    }
                }
            } else if let fnOp = inst.operands.last {
                if case .value(let fnVal) = fnOp {
                    if let fnDecl = enumerator.findFunction(named: fnVal.name) {
                        if case .pointer(let pointee, _) = fnDecl.type, case .function = pointee {
                            operands.append(UInt64(enumerator.typeIndex(pointee)))
                        } else if case .function = fnDecl.type {
                            operands.append(UInt64(enumerator.typeIndex(fnDecl.type)))
                        } else {
                            operands.append(0)
                        }
                    } else if case .pointer(let pointee, _) = fnVal.type, case .function = pointee {
                        operands.append(UInt64(enumerator.typeIndex(pointee)))
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

            // Note: ref bc does NOT have trailing bundle_count despite bit17 being set
            // Apple's format may not use it, or bundle_count=0 is implied by omission

            writer.emitAbbreviatedRecord(abbrevID: abbrevCall, operands: operands)

        case .phi:
            // INST_PHI: [ty, val0, bb0, val1, bb1, ...]
            // Values are encoded as signed VBR relative IDs
            let typeID = UInt64(enumerator.typeIndex(inst.type))

            // Fast path: 2-entry phi (most common) uses abbreviation
            if inst.operands.count == 4 {
                let v0 = resolveOperand(inst.operands[0])
                let r0 = Int64(currentValueID) - Int64(v0)
                let e0: UInt64 = r0 >= 0 ? UInt64(r0) << 1 : (UInt64(-r0) << 1) | 1
                let bb0: UInt64
                if case .basicBlock(let bb) = inst.operands[1] { bb0 = UInt64(bbIndexMap[bb.name] ?? 0) } else { bb0 = 0 }
                let v1 = resolveOperand(inst.operands[2])
                let r1 = Int64(currentValueID) - Int64(v1)
                let e1: UInt64 = r1 >= 0 ? UInt64(r1) << 1 : (UInt64(-r1) << 1) | 1
                let bb1: UInt64
                if case .basicBlock(let bb) = inst.operands[3] { bb1 = UInt64(bbIndexMap[bb.name] ?? 0) } else { bb1 = 0 }
                writer.emitAbbreviatedRecord(abbrevID: abbrevPhi2, operands: [typeID, e0, bb0, e1, bb1])
            } else {
                var operands: [UInt64] = [typeID]
                var i = 0
                while i + 1 < inst.operands.count {
                    let valID = resolveOperand(inst.operands[i])
                    let rel = Int64(currentValueID) - Int64(valID)
                    let encoded: UInt64 = rel >= 0 ? UInt64(rel) << 1 : (UInt64(-rel) << 1) | 1
                    operands.append(encoded)
                    if case .basicBlock(let bb) = inst.operands[i+1] {
                        operands.append(UInt64(bbIndexMap[bb.name] ?? 0))
                    } else {
                        operands.append(0)
                    }
                    i += 2
                }
                writer.emitUnabbrevRecord(code: instPhiCode, operands: operands)
            }

        case .select:
            if inst.operands.count >= 3 {
                let cond = resolveOperand(inst.operands[0])
                let trueVal = resolveOperand(inst.operands[1])
                let falseVal = resolveOperand(inst.operands[2])
                writer.emitAbbreviatedRecord(abbrevID: abbrevSelect, operands: [
                    relativeID(trueVal), relativeID(falseVal), relativeID(cond)])
            }

        case .extractElement:
            if inst.operands.count >= 2 {
                let vec = resolveOperand(inst.operands[0])
                let idx = resolveOperand(inst.operands[1])
                writer.emitAbbreviatedRecord(abbrevID: abbrevExtractElt, operands: [
                    relativeID(vec), relativeID(idx)])
            }

        case .insertElement:
            if inst.operands.count >= 3 {
                let vec = resolveOperand(inst.operands[0])
                let elt = resolveOperand(inst.operands[1])
                let idx = resolveOperand(inst.operands[2])
                writer.emitAbbreviatedRecord(abbrevID: abbrevInsertElt, operands: [
                    relativeID(vec), relativeID(elt), relativeID(idx)])
            }

        case .shuffleVector:
            if inst.operands.count >= 3 {
                let vec1 = resolveOperand(inst.operands[0])
                let vec2 = resolveOperand(inst.operands[1])
                let mask = resolveOperand(inst.operands[2])
                writer.emitAbbreviatedRecord(abbrevID: abbrevShuffleVec, operands: [
                    relativeID(vec1), relativeID(vec2), relativeID(mask)])
            }

        case .switchInst:
            // INST_SWITCH: [opty, cond, default_bb, ncases, [val, bb]*]
            if inst.operands.count >= 2 {
                var operands: [UInt64] = [UInt64(enumerator.typeIndex(inst.type == .void ? .i32 : inst.type))]
                let condID = resolveOperand(inst.operands[0])
                operands.append(relativeID(condID))
                // Default BB
                if case .basicBlock(let bb) = inst.operands[1] {
                    operands.append(UInt64(bbIndexMap[bb.name] ?? 0))
                } else {
                    operands.append(0)
                }
                // Case pairs: [val, bb]
                var i = 2
                while i + 1 < inst.operands.count {
                    let caseVal = resolveOperand(inst.operands[i])
                    operands.append(relativeID(caseVal))
                    if case .basicBlock(let bb) = inst.operands[i+1] {
                        operands.append(UInt64(bbIndexMap[bb.name] ?? 0))
                    } else {
                        operands.append(0)
                    }
                    i += 2
                }
                writer.emitUnabbrevRecord(code: instSwitchCode, operands: operands) // switch is rare, keep unabbrev
            }

        case .extractValue:
            // INST_EXTRACTVAL (26): [agg_relID, idx0, idx1, ...]
            if inst.operands.count >= 2 {
                let agg = resolveOperand(inst.operands[0])
                var operands: [UInt64] = [relativeID(agg)]
                for i in 1..<inst.operands.count {
                    if case .intLiteral(let idx) = inst.operands[i] {
                        operands.append(UInt64(idx))
                    }
                }
                writer.emitUnabbrevRecord(code: instExtractValCode, operands: operands)
            }

        case .insertValue:
            // INST_INSERTVAL (27): [agg_relID, val_relID, idx0, idx1, ...]
            if inst.operands.count >= 3 {
                let agg = resolveOperand(inst.operands[0])
                let val = resolveOperand(inst.operands[1])
                var operands: [UInt64] = [relativeID(agg), relativeID(val)]
                for i in 2..<inst.operands.count {
                    if case .intLiteral(let idx) = inst.operands[i] {
                        operands.append(UInt64(idx))
                    }
                }
                writer.emitUnabbrevRecord(code: instInsertValCode, operands: operands)
            }

        case .unreachable:
            writer.emitAbbreviatedRecord(abbrevID: abbrevUnreachable, operands: [])

        default:
            print("[FunctionWriter] WARNING: unhandled opcode \(inst.opcode) — emitting unreachable")
            writer.emitAbbreviatedRecord(abbrevID: abbrevUnreachable, operands: [])
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

        // Collect BB names for BBENTRY records (separate from value ENTRY records)
        var bbEntries: [(Int, String)] = []
        for (i, bb) in function.basicBlocks.enumerated() {
            if !bb.name.isEmpty {
                bbEntries.append((i, bb.name))
            }
        }

        guard !entries.isEmpty || !bbEntries.isEmpty else { return }

        writer.enterSubblock(blockID: valueSymtabBlockID, abbrevLen: 4)

        // Value ENTRY records (parameters, instructions)
        for (id, name) in entries {
            let nameBytes = Array(name.utf8)
            var operands: [UInt64] = [UInt64(id)]
            for b in nameBytes {
                operands.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: vstEntryCode, operands: operands)
        }

        // BB ENTRY records (basic block names)
        for (i, name) in bbEntries {
            let nameBytes = Array(name.utf8)
            var operands: [UInt64] = [UInt64(i)]
            for b in nameBytes {
                operands.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: vstBBEntryCode, operands: operands)
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
