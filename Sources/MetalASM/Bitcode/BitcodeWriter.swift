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

        // MODULE_BLOCK
        writer.enterSubblock(blockID: moduleBlockID, abbrevLen: 4)

        // Module version
        writer.emitUnabbrevRecord(code: moduleVersionCode, operands: [2])

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

        // Global variables
        writeGlobalVars(to: writer, module: module, enumerator: enumerator)

        // Function declarations (MODULE_CODE_FUNCTION for each function)
        writeFunctionDecls(to: writer, module: module, enumerator: enumerator)

        // Module-level value symbol table
        writeModuleVST(to: writer, module: module, enumerator: enumerator)

        // Function bodies
        for fn in module.functions {
            if !fn.isDeclaration {
                FunctionWriter.write(to: writer, function: fn, enumerator: enumerator)
            }
        }

        // Metadata
        MetadataWriter.write(to: writer, module: module, enumerator: enumerator)

        writer.exitBlock() // end MODULE_BLOCK

        return writer.finalize()
    }

    // MARK: - Attribute blocks

    private static func writeParamAttrGroupBlock(to writer: BitstreamWriter, module: IRModule) {
        guard !module.attributeGroups.isEmpty else { return }

        writer.enterSubblock(blockID: paramAttrGroupBlockID, abbrevLen: 4)

        for group in module.attributeGroups {
            // PARAMATTR_GRP_CODE_ENTRY: [group_id, idx, ...attrs]
            // For simplicity, emit each attribute as a separate record
            var operands: [UInt64] = [UInt64(group.index), UInt64(group.index)]

            for attr in group.attributes {
                switch attr {
                case .noUnwind:
                    operands.append(contentsOf: encodeEnumAttr(id: 5))
                case .convergent:
                    operands.append(contentsOf: encodeEnumAttr(id: 46))
                case .mustProgress:
                    operands.append(contentsOf: encodeEnumAttr(id: 74))
                case .willReturn:
                    operands.append(contentsOf: encodeEnumAttr(id: 69))
                case .noFree:
                    operands.append(contentsOf: encodeEnumAttr(id: 68))
                case .noSync:
                    operands.append(contentsOf: encodeEnumAttr(id: 67))
                case .noCallback:
                    operands.append(contentsOf: encodeEnumAttr(id: 71))
                case .argMemOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 33))
                case .readNone:
                    operands.append(contentsOf: encodeEnumAttr(id: 11))
                case .readOnly:
                    operands.append(contentsOf: encodeEnumAttr(id: 12))
                case .noCapture:
                    operands.append(contentsOf: encodeEnumAttr(id: 10))
                case .noReturn:
                    operands.append(contentsOf: encodeEnumAttr(id: 4))
                case .stringAttr(let key, let value):
                    // String attributes: kind=3 for key-only, kind=4 for key=value
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
        enumerator: ValueEnumerator
    ) {
        for global in module.globals {
            // MODULE_CODE_GLOBALVAR: [type, isconst, initid, linkage, alignment,
            //   section, visibility, tls, unnamed_addr, ext_init, dllstorageclass,
            //   comdat, addrspace, preemption]
            let typeIdx = enumerator.typeIndex(global.type)
            let isConst: UInt64 = global.isConstant ? 1 : 0

            // Init ID: 0 = no initializer, otherwise 1-based index into value table
            // For undef/zeroinit, we still need a constant
            let initID: UInt64
            if global.initializer != nil {
                // Use a placeholder; the actual constant will be in CONSTANTS_BLOCK
                initID = 1 // simplified - real impl would enumerate the constant
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
