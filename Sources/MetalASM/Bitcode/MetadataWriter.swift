/// Writes METADATA_BLOCK and METADATA_KIND_BLOCK in LLVM bitcode format.
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

    /// METADATA_BLOCK ID.
    static let blockID: UInt64 = 15

    /// Write the METADATA_BLOCK for the module.
    static func write(to writer: BitstreamWriter, module: IRModule, enumerator: ValueEnumerator) {
        // Skip if no metadata
        guard !module.metadataNodes.isEmpty || !module.namedMetadata.isEmpty else {
            return
        }

        writer.enterSubblock(blockID: blockID, abbrevLen: 4)

        // Emit all metadata nodes first
        // We need to emit them in index order
        let sortedNodes = module.metadataNodes.sorted { $0.index < $1.index }

        // Build a metadata string table: collect all strings referenced by metadata
        var metadataStrings: [String] = []
        var stringToIndex: [String: Int] = [:]

        // Collect strings from metadata operands
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

        // Emit metadata strings using old-style METADATA_STRING records
        // (simpler than the new METADATA_STRINGS blob format)
        for str in metadataStrings {
            let bytes = Array(str.utf8)
            var operands: [UInt64] = []
            for b in bytes {
                operands.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: stringOldCode, operands: operands)
        }

        // Emit metadata nodes
        // Metadata IDs: strings come first (0..stringCount-1), then nodes
        let stringCount = metadataStrings.count

        for node in sortedNodes {
            var operands: [UInt64] = []
            for op in node.operands {
                switch op {
                case .string(let s):
                    // Reference to metadata string
                    let strIdx = stringToIndex[s]!
                    operands.append(UInt64(strIdx))

                case .metadata(let idx):
                    // Reference to another metadata node
                    // The ID is stringCount + nodeIndex
                    operands.append(UInt64(stringCount + idx))

                case .constant(let type, let constant):
                    // METADATA_VALUE: type_id, value
                    // For constants embedded in metadata, we need to emit them inline
                    // Using a simplified approach: emit type index and constant value
                    let typeIdx = enumerator.typeIndex(type)
                    operands.append(UInt64(typeIdx))
                    switch constant {
                    case .integer(_, let val):
                        operands.append(UInt64(bitPattern: val))
                    default:
                        operands.append(0)
                    }

                case .value(let type, let name):
                    // Reference to a function/global as metadata value
                    let typeIdx = enumerator.typeIndex(type)
                    operands.append(UInt64(typeIdx))
                    if let valID = enumerator.globalValueID(name: name) {
                        operands.append(UInt64(valID))
                    } else {
                        operands.append(0)
                    }

                case .null:
                    // Null metadata operand: use a special sentinel
                    // In LLVM bitcode, null metadata is represented by type=0 (void)
                    operands.append(0)
                }
            }

            let code = node.isDistinct ? distinctNodeCode : nodeCode
            writer.emitUnabbrevRecord(code: code, operands: operands)
        }

        // Emit named metadata
        for named in module.namedMetadata {
            // METADATA_NAME: emit the name as chars
            let nameBytes = Array(named.name.utf8)
            var nameOps: [UInt64] = []
            for b in nameBytes {
                nameOps.append(UInt64(b))
            }
            writer.emitUnabbrevRecord(code: nameCode, operands: nameOps)

            // METADATA_NAMED_NODE: emit references to metadata nodes
            var nodeOps: [UInt64] = []
            for idx in named.operands {
                nodeOps.append(UInt64(stringCount + idx))
            }
            writer.emitUnabbrevRecord(code: namedNodeCode, operands: nodeOps)
        }

        writer.exitBlock()
    }
}
