import Foundation

/// Orchestrates serialization of an IRModule into LLVM bitcode bytes.
///
/// Produces bitcode starting with "BC\xC0\xDE" that can be wrapped in
/// a metallib container and loaded by Metal.
public final class BitcodeWriter {

    // MODULE_BLOCK record codes
    static let moduleVersionCode: UInt64 = 1
    static let tripleCode: UInt64 = 2
    static let datalayoutCode: UInt64 = 3
    static let asmCode: UInt64 = 4  // not used
    static let sectionNameCode: UInt64 = 5
    static let depLibCode: UInt64 = 6
    static let globalVarCode: UInt64 = 7
    static let functionCode: UInt64 = 8
    static let aliasCode: UInt64 = 9  // not used
    static let gcNameCode: UInt64 = 11
    static let sourceFilenameCode: UInt64 = 16

    // PARAMATTR_GROUP_BLOCK record codes
    static let paramAttrGroupEntryCode: UInt64 = 3

    // PARAMATTR_BLOCK record codes
    static let paramAttrEntryCode: UInt64 = 2

    // Block IDs
    static let moduleBlockID: UInt64 = 8
    static let paramAttrBlockID: UInt64 = 9
    static let paramAttrGroupBlockID: UInt64 = 10

    /// Timing breakdown for the last write() call.
    public static var _bcBreakdown: String = ""

    /// Write a complete LLVM bitcode from the given module.
    ///
    /// Returns the raw bitcode bytes starting with "BC\xC0\xDE".
    public static func write(module: IRModule) -> [UInt8] {
        // Ensure paramAttrLists is populated for all attribute groups.
        if module.paramAttrLists.isEmpty && !module.attributeGroups.isEmpty {
            for group in module.attributeGroups {
                module.paramAttrLists.append([group.index])
            }
        }

        // TypeTableWriter.emitOpaqueAsTyped is set by IRTransform (single source
        // of truth for the hasMMA || hasTGByteGlobal condition). Reset after use.
        defer {
            TypeTableWriter.emitOpaqueAsTyped = false
            TypeTableWriter.collapseDevicePtrsToFloat = false
        }

        let te = CFAbsoluteTimeGetCurrent()
        let enumerator = ValueEnumerator(module: module)
        let te1 = CFAbsoluteTimeGetCurrent()
        let instCount = module.functions.reduce(0) { $0 + $1.basicBlocks.reduce(0) { $0 + $1.instructions.count } }
        let writer = BitstreamWriter(capacity: instCount * 8 + 4096)

        // Emit magic
        writer.emitBitcodeMagic()

        // IDENTIFICATION_BLOCK
            writer.enterSubblock(blockID: 13, abbrevLen: 5)
            writer.emitUnabbrevStringRecord(code: 1, "MetalASM")
            writer.emitUnabbrevRecord(code: 2, operands: [0])
            writer.exitBlock()

            // MODULE_BLOCK
            writer.enterSubblock(blockID: moduleBlockID, abbrevLen: 4)
            writer.emitUnabbrevRecord(code: moduleVersionCode, operands: [1])

            writeParamAttrGroupBlock(to: writer, module: module)
            writeParamAttrBlock(to: writer, module: module)
            TypeTableWriter.write(to: writer, enumerator: enumerator)

            if !module.targetTriple.isEmpty {
                writer.emitUnabbrevStringRecord(code: tripleCode, module.targetTriple)
            }
            if !module.dataLayout.isEmpty {
                writer.emitUnabbrevStringRecord(code: datalayoutCode, module.dataLayout)
            }
            if !module.sourceFilename.isEmpty {
                writer.emitUnabbrevStringRecord(code: sourceFilenameCode, module.sourceFilename)
            }

            let moduleConstants = buildModuleConstantMap(module: module, enumerator: enumerator)

            writeGlobalVars(to: writer, module: module, enumerator: enumerator,
                            moduleConstants: moduleConstants)
            writeFunctionDecls(to: writer, module: module, enumerator: enumerator)

            emitModuleConstantsBlock(to: writer, enumerator: enumerator, moduleConstants: moduleConstants)
            MetadataWriter.writeMetadataKindBlock(to: writer)
            MetadataWriter.write(to: writer, module: module, enumerator: enumerator, moduleConstants: moduleConstants)
            MetadataWriter.writeOperandBundleTagsBlock(to: writer)
            MetadataWriter.writeSinglethreadBlock(to: writer)

            let tf = CFAbsoluteTimeGetCurrent()
            for fn in module.functions {
                if !fn.isDeclaration {
                    FunctionWriter.write(to: writer, function: fn, enumerator: enumerator, moduleConstantCount: moduleConstants.entries.count)
                }
            }
            let tf1 = CFAbsoluteTimeGetCurrent()

            writeModuleVST(to: writer, module: module, enumerator: enumerator)
            writer.exitBlock() // end MODULE_BLOCK

        _bcBreakdown = String(format: "enum=%.0fms fn=%.0fms", (te1-te)*1000, (tf1-tf)*1000)

        return writer.finalize()
    }

    // MARK: - Attribute blocks

    private static func writeParamAttrGroupBlock(to writer: BitstreamWriter, module: IRModule) {
        guard !module.attributeGroups.isEmpty else { return }

        writer.enterSubblock(blockID: paramAttrGroupBlockID, abbrevLen: 4)

        for group in module.attributeGroups {
            // PARAMATTR_GRP_CODE_ENTRY: [group_id, param_index, ...attrs]
            // group_id is 1-based in bitcode
            // param_index = 0xFFFFFFFF for function-level; 0..N for per-param
            let paramIndex: UInt64 = group.paramIndex.map { UInt64($0) } ?? 0xFFFFFFFF
            var operands: [UInt64] = [
                UInt64(group.index + 1),   // 1-based group ID
                paramIndex,
            ]

            for attr in group.attributes {
                switch attr {
                // Enum attribute IDs from LLVMBitCodes.h (ATTR_KIND_*)
                case .noUnwind:
                    appendEnumAttr(to: &operands, id: 18)  // ATTR_KIND_NO_UNWIND
                case .noReturn:
                    appendEnumAttr(to: &operands, id: 17)  // ATTR_KIND_NO_RETURN
                case .convergent:
                    appendEnumAttr(to: &operands, id: 43)  // ATTR_KIND_CONVERGENT
                case .mustProgress:
                    appendEnumAttr(to: &operands, id: 70)  // ATTR_KIND_MUSTPROGRESS
                case .willReturn:
                    appendEnumAttr(to: &operands, id: 61)  // ATTR_KIND_WILLRETURN
                case .noFree:
                    appendEnumAttr(to: &operands, id: 62)  // ATTR_KIND_NOFREE
                case .noSync:
                    appendEnumAttr(to: &operands, id: 63)  // ATTR_KIND_NOSYNC
                case .noCallback:
                    appendEnumAttr(to: &operands, id: 71)  // ATTR_KIND_NO_CALLBACK
                case .argMemOnly:
                    appendEnumAttr(to: &operands, id: 45)  // ATTR_KIND_ARGMEMONLY
                case .readNone:
                    appendEnumAttr(to: &operands, id: 20)  // ATTR_KIND_READ_NONE
                case .readOnly:
                    appendEnumAttr(to: &operands, id: 21)  // ATTR_KIND_READ_ONLY
                case .noCapture:
                    appendEnumAttr(to: &operands, id: 11)  // ATTR_KIND_NO_CAPTURE
                case .noAlias:
                    appendEnumAttr(to: &operands, id: 9)   // ATTR_KIND_NO_ALIAS
                case .nonNull:
                    appendEnumAttr(to: &operands, id: 39)  // ATTR_KIND_NON_NULL
                case .signExt:
                    appendEnumAttr(to: &operands, id: 24)  // ATTR_KIND_S_EXT
                case .zeroExt:
                    appendEnumAttr(to: &operands, id: 34)  // ATTR_KIND_Z_EXT
                case .inReg:
                    appendEnumAttr(to: &operands, id: 5)   // ATTR_KIND_IN_REG
                case .structRet:
                    appendEnumAttr(to: &operands, id: 29)  // ATTR_KIND_STRUCT_RET
                case .byVal:
                    appendEnumAttr(to: &operands, id: 3)   // ATTR_KIND_BY_VAL
                case .nest:
                    appendEnumAttr(to: &operands, id: 8)   // ATTR_KIND_NEST
                case .noRecurse:
                    appendEnumAttr(to: &operands, id: 48)  // ATTR_KIND_NO_RECURSE
                case .noInline:
                    appendEnumAttr(to: &operands, id: 14)  // ATTR_KIND_NO_INLINE
                case .alwaysInline:
                    appendEnumAttr(to: &operands, id: 2)   // ATTR_KIND_ALWAYS_INLINE
                case .optNone:
                    appendEnumAttr(to: &operands, id: 37)  // ATTR_KIND_OPTIMIZE_NONE
                case .optSize:
                    appendEnumAttr(to: &operands, id: 19)  // ATTR_KIND_OPTIMIZE_FOR_SIZE
                case .minSize:
                    appendEnumAttr(to: &operands, id: 6)   // ATTR_KIND_MIN_SIZE
                case .speculatable:
                    appendEnumAttr(to: &operands, id: 53)  // ATTR_KIND_SPECULATABLE
                case .strictFP:
                    appendEnumAttr(to: &operands, id: 54)  // ATTR_KIND_STRICT_FP
                case .immArg:
                    appendEnumAttr(to: &operands, id: 60)  // ATTR_KIND_IMMARG
                case .noundef:
                    appendEnumAttr(to: &operands, id: 68)  // ATTR_KIND_NOUNDEF
                case .returned:
                    appendEnumAttr(to: &operands, id: 22)  // ATTR_KIND_RETURNED
                case .writeOnly:
                    appendEnumAttr(to: &operands, id: 52)  // ATTR_KIND_WRITEONLY
                case .cold:
                    appendEnumAttr(to: &operands, id: 36)  // ATTR_KIND_COLD
                case .hot:
                    appendEnumAttr(to: &operands, id: 72)  // ATTR_KIND_HOT
                case .naked:
                    appendEnumAttr(to: &operands, id: 7)   // ATTR_KIND_NAKED
                case .noBuiltin:
                    appendEnumAttr(to: &operands, id: 10)  // ATTR_KIND_NO_BUILTIN
                case .noImplicitFloat:
                    appendEnumAttr(to: &operands, id: 13)  // ATTR_KIND_NO_IMPLICIT_FLOAT
                case .noProfile:
                    appendEnumAttr(to: &operands, id: 73)  // ATTR_KIND_NO_PROFILE
                case .noSanitizeCoverage:
                    appendEnumAttr(to: &operands, id: 76)  // ATTR_KIND_NO_SANITIZE_COVERAGE
                case .noRedZone:
                    appendEnumAttr(to: &operands, id: 16)  // ATTR_KIND_NO_RED_ZONE
                case .noMerge:
                    appendEnumAttr(to: &operands, id: 66)  // ATTR_KIND_NO_MERGE
                case .inaccessibleMemOnly:
                    appendEnumAttr(to: &operands, id: 49)  // ATTR_KIND_INACCESSIBLEMEM_ONLY
                case .inaccessibleMemOrArgMemOnly:
                    appendEnumAttr(to: &operands, id: 50)  // ATTR_KIND_INACCESSIBLEMEM_OR_ARGMEMONLY
                case .stringAttr(let key, let value):
                    if let value = value {
                        operands.append(4) // string attr with value
                        operands.append(contentsOf: key.utf8.lazy.map { UInt64($0) })
                        operands.append(0) // null terminator
                        operands.append(contentsOf: value.utf8.lazy.map { UInt64($0) })
                        operands.append(0) // null terminator
                    } else {
                        operands.append(3) // string attr without value
                        operands.append(contentsOf: key.utf8.lazy.map { UInt64($0) })
                        operands.append(0) // null terminator
                    }
                default:
                    break
                }
            }

            writer.emitUnabbrevRecord(code: paramAttrGroupEntryCode, operands: operands)
        }

        writer.exitBlock()
    }

    /// Append an enum attribute to the operands array (inline, no allocation).
    @inline(__always)
    private static func appendEnumAttr(to operands: inout [UInt64], id: UInt64) {
        operands.append(0)  // kind=0 (enum)
        operands.append(id)
    }

    private static func writeParamAttrBlock(to writer: BitstreamWriter, module: IRModule) {
        // Emit from paramAttrLists if present, otherwise fall back to one-entry-per-group
        let hasLists = !module.paramAttrLists.isEmpty
        guard hasLists || !module.attributeGroups.isEmpty else { return }

        writer.enterSubblock(blockID: paramAttrBlockID, abbrevLen: 4)

        if hasLists {
            // Each paramAttrList is a set of group indices that form a combined attr list.
            // PARAMATTR_CODE_ENTRY: [group_id_1, group_id_2, ...] (all 1-based)
            for list in module.paramAttrLists {
                let operands = list.map { UInt64($0 + 1) }
                writer.emitUnabbrevRecord(code: paramAttrEntryCode, operands: operands)
            }
        } else {
            // Fallback: one entry per attribute group (fn-level only)
            for group in module.attributeGroups {
                writer.emitUnabbrevRecord(code: paramAttrEntryCode, operands: [
                    UInt64(group.index + 1)
                ])
            }
        }

        writer.exitBlock()
    }

    // MARK: - Global variables

    private static func writeGlobalVars(
        to writer: BitstreamWriter,
        module: IRModule,
        enumerator: ValueEnumerator,
        moduleConstants: ModuleConstantMap = ModuleConstantMap()
    ) {
        for global in module.globals {
            let typeIdx = enumerator.typeIndex(global.type)
            let isConst: UInt64 = global.isConstant ? 1 : 0
            let initID: UInt64
            if let init_ = global.initializer {
                let key = constantKey(for: init_, type: global.valueType, map: moduleConstants)
                if let valID = moduleConstants.valueMap[key] {
                    initID = UInt64(valID) + 1
                } else {
                    initID = 0
                }
            } else {
                initID = 0
            }
            let linkage = encodeLinkage(global.linkage)
            let align = log2Align(global.alignment ?? 0)
            writer.emitUnabbrevRecord(code: globalVarCode, operands: [
                UInt64(typeIdx), isConst, initID, linkage, UInt64(align),
                0, 0, 0,
                global.unnamedAddr ? 1 : (global.localUnnamedAddr ? 2 : 0),
                global.externallyInitialized ? 1 : 0,
                0, 0, UInt64(global.addressSpace), 0,
            ])
        }
    }

    // MARK: - Function declarations

    private static func writeFunctionDecls(
        to writer: BitstreamWriter,
        module: IRModule,
        enumerator: ValueEnumerator
    ) {
        for fn in module.functions {
            let typeIdx = enumerator.typeIndex(fn.type)
            let isProto: UInt64 = fn.isDeclaration ? 1 : 0
            let linkage = encodeLinkage(fn.linkage)
            let paramAttr: UInt64 = fn.attributeGroupIndex.map { UInt64($0 + 1) } ?? 0
            writer.emitUnabbrevRecord(code: functionCode, operands: [
                UInt64(typeIdx),
                UInt64(fn.callingConvention.rawValue),
                isProto, linkage, paramAttr,
                UInt64(log2Align(fn.alignment ?? 0)),
                0, 0, 0,
                fn.unnamedAddr ? 1 : (fn.localUnnamedAddr ? 2 : 0),
                0, 0, 0, 0, 0, 0, UInt64(fn.addressSpace),
            ])
        }
    }

    // MARK: - Module-level Value Symbol Table

    private static func writeModuleVST(
        to writer: BitstreamWriter,
        module: IRModule,
        enumerator: ValueEnumerator
    ) {
        writer.enterSubblock(blockID: 14, abbrevLen: 4)
        for entry in enumerator.globalValues {
            writer.emitUnabbrevStringRecord(code: 1, leading: UInt64(entry.valueID), entry.name)
        }
        writer.exitBlock()
    }

    // MARK: - Module-level constants block

    /// Map from (type, encoded_constant) to its value ID.
    struct ModuleConstantMap {
        enum ConstantKind {
            case integer(UInt64)  // signed VBR encoded
            case float32(UInt32)  // raw bit pattern
            case float64(UInt64)  // raw bit pattern
            case float16(UInt16)  // raw bit pattern
            case bfloat16(UInt16) // raw bit pattern
            case aggregate([Int]) // value IDs of elements
            case null
            case undef
        }
        var entries: [(IRType, ConstantKind)] = []
        var valueMap: [String: Int] = [:]      // "type:value" → value ID
        var baseValueID: Int = 0               // first value ID for module constants

        /// Add a constant and return its value ID. Deduplicates by key.
        mutating func addConstant(key: String, type: IRType, kind: ConstantKind) -> Int {
            if let existing = valueMap[key] { return existing }
            let id = baseValueID + entries.count
            valueMap[key] = id
            entries.append((type, kind))
            return id
        }
    }

    /// Build the module constant map (metadata integers + global initializers)
    /// without emitting anything. Used to get value IDs for global var initIDs.
    private static func buildModuleConstantMap(
        module: IRModule,
        enumerator: ValueEnumerator
    ) -> ModuleConstantMap {
        var map = ModuleConstantMap()
        map.baseValueID = enumerator.globalValueCount

        // Collect unique constants from metadata operands
        let sortedNodes = module.metadataNodes.sorted { $0.index < $1.index }
        for node in sortedNodes {
            for op in node.operands {
                switch op {
                case .constant(let type, let constant):
                    let encoded: UInt64
                    switch constant {
                    case .integer(_, let val):
                        encoded = val >= 0 ? UInt64(val) << 1 : (UInt64(bitPattern: -val) << 1) | 1
                    default: encoded = 0
                    }
                    let key = "\(type):\(encoded)"
                    if map.valueMap[key] == nil {
                        let id = map.baseValueID + map.entries.count
                        map.valueMap[key] = id
                        map.entries.append((type, .integer(encoded)))
                    }
                default:
                    break
                }
            }
        }

        // Add global variable initializers
        for global in module.globals {
            if let init_ = global.initializer {
                _ = addConstantToMap(&map, constant: init_, type: global.valueType)
            }
        }

        return map
    }

    /// Recursively add a constant (and its sub-constants) to the module constant map.
    /// Returns the value ID assigned to this constant.
    @discardableResult
    private static func addConstantToMap(
        _ map: inout ModuleConstantMap,
        constant: IRConstant,
        type: IRType
    ) -> Int {
        switch constant {
        case .undef:
            let key = "\(type):undef"
            return map.addConstant(key: key, type: type, kind: .undef)

        case .zeroInitializer, .null:
            let key = "\(type):null"
            return map.addConstant(key: key, type: type, kind: .null)

        case .integer(_, let val):
            let encoded = val >= 0 ? UInt64(val) << 1 : (UInt64(bitPattern: -val) << 1) | 1
            let key = "\(type):\(encoded)"
            return map.addConstant(key: key, type: type, kind: .integer(encoded))

        case .float32(let v):
            let key = "f32:\(v.bitPattern)"
            return map.addConstant(key: key, type: .float32, kind: .float32(v.bitPattern))

        case .float64(let v):
            let key = "f64:\(v.bitPattern)"
            return map.addConstant(key: key, type: .float64, kind: .float64(v.bitPattern))

        case .float16(let bits):
            let key = "f16:\(bits)"
            return map.addConstant(key: key, type: .float16, kind: .float16(bits))

        case .bfloat16(let bits):
            let key = "bf16:\(bits)"
            return map.addConstant(key: key, type: .bfloat16, kind: .bfloat16(bits))

        case .arrayValue(_, let elems):
            // First, recursively add all element constants
            let elemType: IRType
            if case .array(let et, _) = type { elemType = et } else { elemType = .float32 }
            var elemIDs: [Int] = []
            for elem in elems {
                let id = addConstantToMap(&map, constant: elem, type: elemType)
                elemIDs.append(id)
            }
            // Then add the aggregate itself
            let key = "\(type):agg:\(elemIDs)"
            return map.addConstant(key: key, type: type, kind: .aggregate(elemIDs))

        default:
            // Unsupported constant type — treat as undef
            let key = "\(type):undef"
            return map.addConstant(key: key, type: type, kind: .undef)
        }
    }

    /// Generate the lookup key for a constant (must match the key used in addConstantToMap).
    private static func constantKey(
        for constant: IRConstant, type: IRType, map: ModuleConstantMap
    ) -> String {
        switch constant {
        case .undef: return "\(type):undef"
        case .zeroInitializer, .null: return "\(type):null"
        case .integer(_, let val):
            let encoded = val >= 0 ? UInt64(val) << 1 : (UInt64(bitPattern: -val) << 1) | 1
            return "\(type):\(encoded)"
        case .float32(let v): return "f32:\(v.bitPattern)"
        case .float64(let v): return "f64:\(v.bitPattern)"
        case .float16(let bits): return "f16:\(bits)"
        case .bfloat16(let bits): return "bf16:\(bits)"
        case .arrayValue(_, let elems):
            let elemType: IRType
            if case .array(let et, _) = type { elemType = et } else { elemType = .float32 }
            let elemIDs = elems.map { elem -> Int in
                let k = constantKey(for: elem, type: elemType, map: map)
                return map.valueMap[k] ?? 0
            }
            return "\(type):agg:\(elemIDs)"
        default: return "\(type):undef"
        }
    }

    /// Emit the CONSTANTS_BLOCK for the given module constant map.
    private static func emitModuleConstantsBlock(
        to writer: BitstreamWriter,
        enumerator: ValueEnumerator,
        moduleConstants: ModuleConstantMap
    ) {
        guard !moduleConstants.entries.isEmpty else { return }

        writer.enterSubblock(blockID: 11, abbrevLen: 5)

        var currentType: IRType? = nil
        for (type, kind) in moduleConstants.entries {
            if type != currentType {
                writer.emitUnabbrevRecord(code: 1, UInt64(enumerator.typeIndex(type)))
                currentType = type
            }
            switch kind {
            case .integer(let encoded):
                writer.emitUnabbrevRecord(code: 4, encoded)  // CST_CODE_INTEGER
            case .float32(let bits):
                writer.emitUnabbrevRecord(code: 6, UInt64(bits))  // CST_CODE_FLOAT
            case .float64(let bits):
                writer.emitUnabbrevRecord(code: 6, bits)  // CST_CODE_FLOAT
            case .float16(let bits):
                writer.emitUnabbrevRecord(code: 6, UInt64(bits))  // CST_CODE_FLOAT
            case .bfloat16(let bits):
                writer.emitUnabbrevRecord(code: 6, UInt64(bits))  // CST_CODE_FLOAT
            case .aggregate(let elemIDs):
                // CST_CODE_AGGREGATE: [val, val, ...]
                let operands = elemIDs.map { UInt64($0) }
                writer.emitUnabbrevRecord(code: 7, operands: operands)  // CST_CODE_AGGREGATE
            case .null:
                writer.emitUnabbrevRecord(code: 2)  // CST_CODE_NULL
            case .undef:
                writer.emitUnabbrevRecord(code: 3)  // CST_CODE_UNDEF
            }
        }

        writer.exitBlock()
    }

    // MARK: - Helpers

    private static func encodeLinkage(_ linkage: IRFunction.Linkage) -> UInt64 {
        switch linkage {
        case .external: return 0
        case .weak: return 1
        case .appending: return 2
        case .internal: return 3
        case .linkonce: return 4
        case .externWeak: return 7
        case .common: return 8
        case .private: return 9
        case .weakODR: return 10
        case .linkonceODR: return 11
        case .available_externally: return 12
        }
    }

    private static func log2Align(_ align: Int) -> Int {
        if align <= 0 { return 0 }
        if align == 1 { return 0 }
        var n = 0
        var v = align
        while v > 1 {
            v >>= 1
            n += 1
        }
        return n + 1
    }
}
