/// Writes METADATA_BLOCK, METADATA_KIND_BLOCK, OPERAND_BUNDLE_TAGS_BLOCK,
/// and Block 26 (singlethread) in LLVM bitcode format.
final class MetadataWriter {
    // METADATA_BLOCK record codes
    static let stringOldCode: UInt64 = 1   // METADATA_STRING (old, deprecated)
    static let valueCode: UInt64 = 2       // METADATA_VALUE
    static let nodeCode: UInt64 = 3        // METADATA_NODE
    static let nameCode: UInt64 = 4        // METADATA_NAME
    static let distinctNodeCode: UInt64 = 5 // METADATA_DISTINCT_NODE
    static let kindCode: UInt64 = 6        // METADATA_KIND
    static let namedNodeCode: UInt64 = 10  // METADATA_NAMED_NODE
    static let stringCode: UInt64 = 11     // METADATA_STRINGS (new, with count+offset)

    /// Block IDs.
    static let metadataBlockID: UInt64 = 15
    static let metadataKindBlockID: UInt64 = 22
    static let operandBundleTagsBlockID: UInt64 = 21
    static let singlethreadBlockID: UInt64 = 26

    /// Standard LLVM metadata kind names (must match metal-as output exactly).
    static let standardMetadataKinds: [String] = [
        "dbg", "tbaa", "prof", "fpmath", "range", "tbaa.struct",
        "invariant.load", "alias.scope", "noalias", "nontemporal",
        "llvm.mem.parallel_loop_access", "nonnull", "dereferenceable",
        "dereferenceable_or_null", "make.implicit", "unpredictable",
        "invariant.group", "align", "llvm.loop", "type",
        "section_prefix", "absolute_symbol", "associated", "callees",
        "irr_loop", "llvm.access.group", "callback",
        "llvm.preserve.access.index", "vcall_visibility", "noundef",
        "annotation", "heapallocsite", "air.function_groups",
    ]

    /// Standard operand bundle tag names.
    static let standardOperandBundleTags: [String] = [
        "deopt", "funclet", "gc-transition", "cfguardtarget",
        "preallocated", "gc-live", "clang.arc.attachedcall", "ptrauth",
    ]

    /// Write the METADATA_KIND_BLOCK (standard metadata kind definitions).
    static func writeMetadataKindBlock(to writer: BitstreamWriter) {
        writer.enterSubblock(blockID: metadataKindBlockID, abbrevLen: 3)

        for (idx, name) in standardMetadataKinds.enumerated() {
            // KIND record: [kind_id, ...name_chars]
            writer.emitUnabbrevStringRecord(code: kindCode, leading: UInt64(idx), name)
        }

        writer.exitBlock()
    }

    /// Write the OPERAND_BUNDLE_TAGS_BLOCK.
    static func writeOperandBundleTagsBlock(to writer: BitstreamWriter) {
        writer.enterSubblock(blockID: operandBundleTagsBlockID, abbrevLen: 3)

        for tag in standardOperandBundleTags {
            // OPERAND_BUNDLE_TAG record (code 1): [...name_chars]
            writer.emitUnabbrevStringRecord(code: 1, tag)
        }

        writer.exitBlock()
    }

    /// Write Block 26 (singlethread execution width info).
    static func writeSinglethreadBlock(to writer: BitstreamWriter) {
        writer.enterSubblock(blockID: singlethreadBlockID, abbrevLen: 2)

        // UnknownCode1 record: "singlethread"
        writer.emitUnabbrevStringRecord(code: 1, "singlethread")

        writer.exitBlock()
    }

    /// Write the METADATA_BLOCK for the module.
    ///
    /// LLVM bitcode metadata ID assignment:
    ///   0..stringCount-1  → METADATA_STRING records
    ///   stringCount..stringCount+valueCount-1  → METADATA_VALUE records
    ///   stringCount+valueCount..  → METADATA_NODE records
    static func write(to writer: BitstreamWriter, module: IRModule, enumerator: ValueEnumerator, moduleConstants: BitcodeWriter.ModuleConstantMap = BitcodeWriter.ModuleConstantMap()) {
        // Skip if no metadata
        guard !module.metadataNodes.isEmpty || !module.namedMetadata.isEmpty else {
            return
        }

        writer.enterSubblock(blockID: metadataBlockID, abbrevLen: 4)

        let sortedNodes = module.metadataNodes.sorted { $0.index < $1.index }

        // Map from node.index → sequential position in sortedNodes (used for cross-references)
        var nodeIndexToPosition: [Int: Int] = [:]
        for (pos, node) in sortedNodes.enumerated() {
            nodeIndexToPosition[node.index] = pos
        }

        // --- Phase 1: Collect metadata strings ---
        var metadataStrings: [String] = []
        var stringToIndex: [String: Int] = [:]

        for node in sortedNodes {
            for op in node.operands {
                if case .string(let s) = op {
                    if stringToIndex[s] == nil {
                        stringToIndex[s] = metadataStrings.count
                        metadataStrings.append(s)
                    }
                }
            }
        }

        // --- Phase 2: Collect METADATA_VALUE entries ---
        // Each .constant or .value operand needs its own METADATA_VALUE record.
        // METADATA_VALUE [type_index, value_id] where value_id refers to a constant
        // in the module-level CONSTANTS_BLOCK or a global/function.
        struct MetadataValueEntry: Hashable {
            let typeIdx: Int
            let valueID: UInt64  // absolute value ID in the module value table
        }
        var metadataValues: [MetadataValueEntry] = []
        var valueToIndex: [MetadataValueEntry: Int] = [:]

        func registerMetadataValue(_ entry: MetadataValueEntry) -> Int {
            if let existing = valueToIndex[entry] {
                return existing
            }
            let idx = metadataValues.count
            valueToIndex[entry] = idx
            metadataValues.append(entry)
            return idx
        }

        // Pre-scan all nodes to collect VALUE entries
        for node in sortedNodes {
            for op in node.operands {
                switch op {
                case .constant(let type, let constant):
                    let typeIdx = enumerator.typeIndex(type)
                    // Look up the value ID from the module constants map
                    let encoded: UInt64
                    switch constant {
                    case .integer(_, let val):
                        encoded = val >= 0 ? UInt64(val) << 1 : (UInt64(bitPattern: -val) << 1) | 1
                    default: encoded = 0
                    }
                    let key = "\(type):\(encoded)"
                    let valID = UInt64(moduleConstants.valueMap[key] ?? 0)
                    _ = registerMetadataValue(MetadataValueEntry(typeIdx: typeIdx, valueID: valID))

                case .value(let type, let name):
                    let typeIdx = enumerator.typeIndex(type)
                    // Function/global references use their global value ID directly
                    let valID = UInt64(enumerator.globalValueID(name: name) ?? 0)
                    _ = registerMetadataValue(MetadataValueEntry(typeIdx: typeIdx, valueID: valID))

                default:
                    break
                }
            }
        }

        let stringCount = metadataStrings.count
        let valueCount = metadataValues.count

        // --- Phase 3: Emit metadata strings ---
        for str in metadataStrings {
            writer.emitUnabbrevStringRecord(code: stringOldCode, str)
        }

        // --- Phase 4: Emit METADATA_VALUE records ---
        // Each gets a metadata ID = stringCount + valueIndex
        // METADATA_VALUE: [type_index, value_id]
        for entry in metadataValues {
            writer.emitUnabbrevRecord(code: valueCode, operands: [
                UInt64(entry.typeIdx),
                entry.valueID
            ])
        }

        // --- Phase 5: Emit METADATA_NODE records ---
        // Each node gets metadata ID = stringCount + valueCount + nodeIndex
        // IMPORTANT: LLVM METADATA_NODE operands use (metadata_id + 1), where 0 = null
        for node in sortedNodes {
            var operands: [UInt64] = []
            for op in node.operands {
                switch op {
                case .string(let s):
                    operands.append(UInt64(stringToIndex[s]!) + 1)

                case .metadata(let idx):
                    // Reference to another node: ID = stringCount + valueCount + position
                    let pos = nodeIndexToPosition[idx] ?? idx
                    operands.append(UInt64(stringCount + valueCount + pos) + 1)

                case .constant(let type, let constant):
                    // Reference to the METADATA_VALUE we emitted
                    let typeIdx = enumerator.typeIndex(type)
                    let encoded: UInt64
                    switch constant {
                    case .integer(_, let val):
                        encoded = val >= 0 ? UInt64(val) << 1 : (UInt64(bitPattern: -val) << 1) | 1
                    default: encoded = 0
                    }
                    let key = "\(type):\(encoded)"
                    let valID = UInt64(moduleConstants.valueMap[key] ?? 0)
                    let entry = MetadataValueEntry(typeIdx: typeIdx, valueID: valID)
                    let valueIdx = valueToIndex[entry]!
                    operands.append(UInt64(stringCount + valueIdx) + 1)

                case .value(let type, let name):
                    let typeIdx = enumerator.typeIndex(type)
                    let valID = UInt64(enumerator.globalValueID(name: name) ?? 0)
                    let entry = MetadataValueEntry(typeIdx: typeIdx, valueID: valID)
                    let valueIdx = valueToIndex[entry]!
                    operands.append(UInt64(stringCount + valueIdx) + 1)

                case .null:
                    operands.append(0)
                }
            }

            let code = node.isDistinct ? distinctNodeCode : nodeCode
            writer.emitUnabbrevRecord(code: code, operands: operands)
        }

        // --- Phase 6: Emit named metadata ---
        for named in module.namedMetadata {
            writer.emitUnabbrevStringRecord(code: nameCode, named.name)

            // NAMED_NODE operands also use metadata_id + 1 (but NOT +1 for named nodes per LLVM spec)
            var nodeOps: [UInt64] = []
            for idx in named.operands {
                let pos = nodeIndexToPosition[idx] ?? idx
                nodeOps.append(UInt64(stringCount + valueCount + pos))
            }
            writer.emitUnabbrevRecord(code: namedNodeCode, operands: nodeOps)
        }

        writer.exitBlock()
    }
}
