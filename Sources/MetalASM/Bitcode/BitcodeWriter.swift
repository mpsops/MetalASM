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
    static let vstOffsetCode: UInt64 = 13
    static let sourceFilenameCode: UInt64 = 16

    // PARAMATTR_GROUP_BLOCK record codes
    static let paramAttrGroupEntryCode: UInt64 = 3

    // PARAMATTR_BLOCK record codes
    static let paramAttrEntryCode: UInt64 = 2

    // Block IDs
    static let moduleBlockID: UInt64 = 8
    static let paramAttrBlockID: UInt64 = 9
    static let paramAttrGroupBlockID: UInt64 = 10

    // MODULE_CODE_GLOBALVAR fields
    // [strtab offset, strtab size, type, isconst, initid, linkage, align,
    //  section, visibility, tls, unnamed_addr, externally_initialized,
    //  dllstorageclass, comdat, addrspace, preemptionspecifier]

    /// Write a complete LLVM bitcode from the given module.
    ///
    /// Returns the raw bitcode bytes starting with "BC\xC0\xDE".
    public static func write(module: IRModule) -> [UInt8] {
        let enumerator = ValueEnumerator(module: module)
        let writer = BitstreamWriter()

        // Emit magic
        writer.emitBitcodeMagic()

        // IDENTIFICATION_BLOCK (block ID 13)
        writer.enterSubblock(blockID: 13, abbrevLen: 5)
        // STRING record (code 1): producer string (empty)
        writer.emitUnabbrevRecord(code: 1, operands: [])
        // EPOCH record (code 2): bitcode epoch = 0
        writer.emitUnabbrevRecord(code: 2, operands: [0])
        writer.exitBlock()

        // MODULE_BLOCK
        writer.enterSubblock(blockID: moduleBlockID, abbrevLen: 4)

        // Module version (1 = uses VST for names, 2 = uses STRTAB)
        // We use version 1 since we emit VALUE_SYMTAB, not STRTAB
        writer.emitUnabbrevRecord(code: moduleVersionCode, operands: [1])

        // Attribute groups
        writeParamAttrGroupBlock(to: writer, module: module)

        // Parameter attributes
        writeParamAttrBlock(to: writer, module: module)

        // Type table
        TypeTableWriter.write(to: writer, enumerator: enumerator)

        // Triple
        if !module.targetTriple.isEmpty {
            let tripleBytes = Array(module.targetTriple.utf8)
            var ops: [UInt64] = []
            for b in tripleBytes { ops.append(UInt64(b)) }
            writer.emitUnabbrevRecord(code: tripleCode, operands: ops)
        }

        // Data layout
        if !module.dataLayout.isEmpty {
            let dlBytes = Array(module.dataLayout.utf8)
            var ops: [UInt64] = []
            for b in dlBytes { ops.append(UInt64(b)) }
            writer.emitUnabbrevRecord(code: datalayoutCode, operands: ops)
        }

        // Source filename
        if !module.sourceFilename.isEmpty {
            let sfBytes = Array(module.sourceFilename.utf8)
            var ops: [UInt64] = []
            for b in sfBytes { ops.append(UInt64(b)) }
            writer.emitUnabbrevRecord(code: sourceFilenameCode, operands: ops)
        }

        // Build module constant map first (needed for global var initIDs)
        let moduleConstants = buildModuleConstantMap(module: module, enumerator: enumerator)

        // Global variables
        writeGlobalVars(to: writer, module: module, enumerator: enumerator, moduleConstants: moduleConstants)

        // Function declarations (MODULE_CODE_FUNCTION for each function)
        writeFunctionDecls(to: writer, module: module, enumerator: enumerator)

        // Module-level value symbol table
        writeModuleVST(to: writer, module: module, enumerator: enumerator)

        // Module-level constants block (for metadata references + global initializers)
        emitModuleConstantsBlock(to: writer, enumerator: enumerator, moduleConstants: moduleConstants)

        // Metadata kind block (standard LLVM metadata kind IDs)
        MetadataWriter.writeMetadataKindBlock(to: writer)

        // Metadata block (module-level, before function bodies)
        MetadataWriter.write(to: writer, module: module, enumerator: enumerator, moduleConstants: moduleConstants)

        // Operand bundle tags block
        MetadataWriter.writeOperandBundleTagsBlock(to: writer)

        // Block 26 (singlethread execution width info)
        MetadataWriter.writeSinglethreadBlock(to: writer)

        // Function bodies
        for fn in module.functions {
            if !fn.isDeclaration {
                FunctionWriter.write(to: writer, function: fn, enumerator: enumerator, moduleConstantCount: moduleConstants.entries.count)
            }
        }

        writer.exitBlock() // end MODULE_BLOCK

        return writer.finalize()
    }

    // MARK: - Attribute blocks

    private static func writeParamAttrGroupBlock(to writer: BitstreamWriter, module: IRModule) {
        guard !module.attributeGroups.isEmpty else { return }

        writer.enterSubblock(blockID: paramAttrGroupBlockID, abbrevLen: 4)

        for group in module.attributeGroups {
            // PARAMATTR_GRP_CODE_ENTRY: [group_id, param_index, ...attrs]
            // group_id is 1-based in bitcode
            // param_index = 0xFFFFFFFF for function-level attributes
            var operands: [UInt64] = [
                UInt64(group.index + 1),   // 1-based group ID
                UInt64(0xFFFFFFFF),        // function-level attributes
            ]

            for attr in group.attributes {
                switch attr {
                // Enum attribute IDs from LLVMBitCodes.h (ATTR_KIND_*)
                case .noUnwind:
                    operands.append(contentsOf: encodeEnumAttr(id: 18))  // ATTR_KIND_NO_UNWIND
                case .noReturn:
                    operands.append(contentsOf: encodeEnumAttr(id: 17))  // ATTR_KIND_NO_RETURN
                case .convergent:
                    operands.append(contentsOf: encodeEnumAttr(id: 43))  // ATTR_KIND_CONVERGENT
                case .mustProgress:
                    operands.append(contentsOf: encodeEnumAttr(id: 70))  // ATTR_KIND_MUSTPROGRESS
                case .willReturn:
                    operands.append(contentsOf: encodeEnumAttr(id: 61))  // ATTR_KIND_WILLRETURN
                case .noFree:
                    operands.append(contentsOf: encodeEnumAttr(id: 62))  // ATTR_KIND_NOFREE
                case .noSync:
                    operands.append(contentsOf: encodeEnumAttr(id: 63))  // ATTR_KIND_NOSYNC
                case .noCallback:
                    operands.append(contentsOf: encodeEnumAttr(id: 71))  // ATTR_KIND_NO_CALLBACK
                case .argMemOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 45))  // ATTR_KIND_ARGMEMONLY
                case .readNone:
                    operands.append(contentsOf: encodeEnumAttr(id: 20))  // ATTR_KIND_READ_NONE
                case .readOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 21))  // ATTR_KIND_READ_ONLY
                case .noCapture:
                    operands.append(contentsOf: encodeEnumAttr(id: 11))  // ATTR_KIND_NO_CAPTURE
                case .noAlias:
                    operands.append(contentsOf: encodeEnumAttr(id: 9))   // ATTR_KIND_NO_ALIAS
                case .nonNull:
                    operands.append(contentsOf: encodeEnumAttr(id: 39))  // ATTR_KIND_NON_NULL
                case .signExt:
                    operands.append(contentsOf: encodeEnumAttr(id: 24))  // ATTR_KIND_S_EXT
                case .zeroExt:
                    operands.append(contentsOf: encodeEnumAttr(id: 34))  // ATTR_KIND_Z_EXT
                case .inReg:
                    operands.append(contentsOf: encodeEnumAttr(id: 5))   // ATTR_KIND_IN_REG
                case .structRet:
                    operands.append(contentsOf: encodeEnumAttr(id: 29))  // ATTR_KIND_STRUCT_RET
                case .byVal:
                    operands.append(contentsOf: encodeEnumAttr(id: 3))   // ATTR_KIND_BY_VAL
                case .nest:
                    operands.append(contentsOf: encodeEnumAttr(id: 8))   // ATTR_KIND_NEST
                case .noRecurse:
                    operands.append(contentsOf: encodeEnumAttr(id: 48))  // ATTR_KIND_NO_RECURSE
                case .noInline:
                    operands.append(contentsOf: encodeEnumAttr(id: 14))  // ATTR_KIND_NO_INLINE
                case .alwaysInline:
                    operands.append(contentsOf: encodeEnumAttr(id: 2))   // ATTR_KIND_ALWAYS_INLINE
                case .optNone:
                    operands.append(contentsOf: encodeEnumAttr(id: 37))  // ATTR_KIND_OPTIMIZE_NONE
                case .optSize:
                    operands.append(contentsOf: encodeEnumAttr(id: 19))  // ATTR_KIND_OPTIMIZE_FOR_SIZE
                case .minSize:
                    operands.append(contentsOf: encodeEnumAttr(id: 6))   // ATTR_KIND_MIN_SIZE
                case .speculatable:
                    operands.append(contentsOf: encodeEnumAttr(id: 53))  // ATTR_KIND_SPECULATABLE
                case .strictFP:
                    operands.append(contentsOf: encodeEnumAttr(id: 54))  // ATTR_KIND_STRICT_FP
                case .immArg:
                    operands.append(contentsOf: encodeEnumAttr(id: 60))  // ATTR_KIND_IMMARG
                case .noundef:
                    operands.append(contentsOf: encodeEnumAttr(id: 68))  // ATTR_KIND_NOUNDEF
                case .returned:
                    operands.append(contentsOf: encodeEnumAttr(id: 22))  // ATTR_KIND_RETURNED
                case .writeOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 52))  // ATTR_KIND_WRITEONLY
                case .cold:
                    operands.append(contentsOf: encodeEnumAttr(id: 36))  // ATTR_KIND_COLD
                case .hot:
                    operands.append(contentsOf: encodeEnumAttr(id: 72))  // ATTR_KIND_HOT
                case .naked:
                    operands.append(contentsOf: encodeEnumAttr(id: 7))   // ATTR_KIND_NAKED
                case .noBuiltin:
                    operands.append(contentsOf: encodeEnumAttr(id: 10))  // ATTR_KIND_NO_BUILTIN
                case .noImplicitFloat:
                    operands.append(contentsOf: encodeEnumAttr(id: 13))  // ATTR_KIND_NO_IMPLICIT_FLOAT
                case .noProfile:
                    operands.append(contentsOf: encodeEnumAttr(id: 73))  // ATTR_KIND_NO_PROFILE
                case .noSanitizeCoverage:
                    operands.append(contentsOf: encodeEnumAttr(id: 76))  // ATTR_KIND_NO_SANITIZE_COVERAGE
                case .noRedZone:
                    operands.append(contentsOf: encodeEnumAttr(id: 16))  // ATTR_KIND_NO_RED_ZONE
                case .noMerge:
                    operands.append(contentsOf: encodeEnumAttr(id: 66))  // ATTR_KIND_NO_MERGE
                case .inaccessibleMemOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 49))  // ATTR_KIND_INACCESSIBLEMEM_ONLY
                case .inaccessibleMemOrArgMemOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 50))  // ATTR_KIND_INACCESSIBLEMEM_OR_ARGMEMONLY
                case .stringAttr(let key, let value):
                    if let value = value {
                        operands.append(4) // string attr with value
                        for b in key.utf8 { operands.append(UInt64(b)) }
                        operands.append(0) // null terminator
                        for b in value.utf8 { operands.append(UInt64(b)) }
                        operands.append(0) // null terminator
                    } else {
                        operands.append(3) // string attr without value
                        for b in key.utf8 { operands.append(UInt64(b)) }
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

    /// Encode an enum attribute ID for the PARAMATTR_GRP record.
    private static func encodeEnumAttr(id: UInt64) -> [UInt64] {
        // In PARAMATTR_GRP, enum attrs are: [kind=0, attr_id]
        return [0, id]
    }

    private static func writeParamAttrBlock(to writer: BitstreamWriter, module: IRModule) {
        guard !module.attributeGroups.isEmpty else { return }

        writer.enterSubblock(blockID: paramAttrBlockID, abbrevLen: 4)

        // For each function, emit a PARAMATTR_CODE_ENTRY that references attribute groups
        // For now, emit one entry per attribute group
        for group in module.attributeGroups {
            writer.emitUnabbrevRecord(code: paramAttrEntryCode, operands: [
                UInt64(group.index + 1) // 1-based index
            ])
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
            // MODULE_CODE_GLOBALVAR: [type, isconst, initid, linkage, alignment,
            //   section, visibility, tls, unnamed_addr, ext_init, dllstorageclass,
            //   comdat, addrspace, preemption]
            let typeIdx = enumerator.typeIndex(global.type)
            let isConst: UInt64 = global.isConstant ? 1 : 0

            // Init ID: 0 = no initializer, otherwise 1-based value ID
            let initID: UInt64
            if let init_ = global.initializer {
                // Look up the initializer in the module constants map
                let key: String
                switch init_ {
                case .undef: key = "\(global.valueType):undef"
                case .zeroInitializer, .null: key = "\(global.valueType):null"
                default: key = "\(global.valueType):unknown"
                }
                if let valID = moduleConstants.valueMap[key] {
                    initID = UInt64(valID) + 1 // 1-based
                } else {
                    initID = 0
                }
            } else {
                initID = 0
            }

            let linkage = encodeLinkage(global.linkage)
            let align = log2Align(global.alignment ?? 0)

            writer.emitUnabbrevRecord(code: globalVarCode, operands: [
                UInt64(typeIdx),       // pointer type
                isConst,               // isconst
                initID,                // initid
                linkage,               // linkage
                UInt64(align),         // alignment
                0,                     // section
                0,                     // visibility (default)
                0,                     // thread local mode
                global.unnamedAddr ? 1 : (global.localUnnamedAddr ? 2 : 0),  // unnamed_addr
                global.externallyInitialized ? 1 : 0,
                0,                     // dll storage class
                0,                     // comdat
                UInt64(global.addressSpace),
                0,                     // preemption specifier
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
            // MODULE_CODE_FUNCTION: [type, callingconv, isproto, linkage, paramattr,
            //   alignment, section, visibility, gc, unnamed_addr, prologuedata,
            //   dllstorageclass, comdat, prefixdata, personalityfn, preemption,
            //   addrspace, ...]
            let typeIdx = enumerator.typeIndex(fn.type)
            let isProto: UInt64 = fn.isDeclaration ? 1 : 0
            let linkage = encodeLinkage(fn.linkage)
            let paramAttr: UInt64 = fn.attributeGroupIndex.map { UInt64($0 + 1) } ?? 0

            writer.emitUnabbrevRecord(code: functionCode, operands: [
                UInt64(typeIdx),
                UInt64(fn.callingConvention.rawValue),
                isProto,
                linkage,
                paramAttr,
                UInt64(log2Align(fn.alignment ?? 0)),
                0,  // section
                0,  // visibility
                0,  // gc
                fn.unnamedAddr ? 1 : (fn.localUnnamedAddr ? 2 : 0),
                0,  // prologuedata
                0,  // dllstorageclass
                0,  // comdat
                0,  // prefixdata
                0,  // personalityfn
                0,  // preemption
                UInt64(fn.addressSpace),
            ])
        }
    }

    // MARK: - Module-level Value Symbol Table

    private static func writeModuleVST(
        to writer: BitstreamWriter,
        module: IRModule,
        enumerator: ValueEnumerator
    ) {
        writer.enterSubblock(blockID: 14, abbrevLen: 4) // VALUE_SYMTAB_BLOCK

        for entry in enumerator.globalValues {
            let nameBytes = Array(entry.name.utf8)
            var operands: [UInt64] = [UInt64(entry.valueID)]
            for b in nameBytes {
                operands.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: 1, operands: operands) // VST_CODE_ENTRY
        }

        writer.exitBlock()
    }

    // MARK: - Module-level constants block

    /// Map from (type, encoded_constant) to its value ID.
    struct ModuleConstantMap {
        enum ConstantKind {
            case integer(UInt64)  // signed VBR encoded
            case null
            case undef
        }
        var entries: [(IRType, ConstantKind)] = []
        var valueMap: [String: Int] = [:]      // "type:value" → value ID
        var baseValueID: Int = 0               // first value ID for module constants
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

        // Add global variable initializers (undef, zeroinit)
        for global in module.globals {
            if let init_ = global.initializer {
                let key: String
                let kind: ModuleConstantMap.ConstantKind
                switch init_ {
                case .undef:
                    key = "\(global.valueType):undef"
                    kind = .undef
                case .zeroInitializer, .null:
                    key = "\(global.valueType):null"
                    kind = .null
                default:
                    continue // skip complex initializers for now
                }
                if map.valueMap[key] == nil {
                    let id = map.baseValueID + map.entries.count
                    map.valueMap[key] = id
                    map.entries.append((global.valueType, kind))
                }
            }
        }

        return map
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
                writer.emitUnabbrevRecord(code: 1, operands: [
                    UInt64(enumerator.typeIndex(type))
                ]) // SETTYPE
                currentType = type
            }
            switch kind {
            case .integer(let encoded):
                writer.emitUnabbrevRecord(code: 4, operands: [encoded]) // CST_CODE_INTEGER
            case .null:
                writer.emitUnabbrevRecord(code: 2, operands: []) // CST_CODE_NULL
            case .undef:
                writer.emitUnabbrevRecord(code: 3, operands: []) // CST_CODE_UNDEF
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
