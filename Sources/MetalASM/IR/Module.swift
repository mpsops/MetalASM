/// The top-level LLVM IR module.
public final class IRModule {
    /// Module ID / source filename.
    public var moduleID: String

    /// Source filename.
    public var sourceFilename: String

    /// Target data layout string.
    public var dataLayout: String

    /// Target triple string.
    public var targetTriple: String

    /// Struct type definitions (%name = type { ... }).
    public var structTypes: [(name: String, type: IRType)]

    /// Global variables.
    public var globals: [IRGlobal]

    /// Functions (both definitions and declarations).
    public var functions: [IRFunction]

    /// Attribute groups (fn-level and per-param, indexed by group.index).
    public var attributeGroups: [IRAttributeGroup]

    /// PARAMATTR_BLOCK entries: each is a list of group indices that form a combined attr list.
    /// INST_CALL op0 = 1-based index into this array. Empty = no attributes.
    public var paramAttrLists: [[Int]]

    /// Metadata nodes.
    public var metadataNodes: [IRMetadataNode]

    /// Named metadata entries.
    public var namedMetadata: [IRNamedMetadata]

    /// Module flags (from !llvm.module.flags).
    public var moduleFlags: [Int]

    public init() {
        self.moduleID = ""
        self.sourceFilename = ""
        self.dataLayout = ""
        self.targetTriple = ""
        self.structTypes = []
        self.globals = []
        self.functions = []
        self.attributeGroups = []
        self.paramAttrLists = []
        self.metadataNodes = []
        self.namedMetadata = []
        self.moduleFlags = []
    }

    // MARK: - Metallib helpers

    /// AIR version extracted from metadata (major, minor).
    public var airVersion: (major: UInt16, minor: UInt16)? {
        // Look for !air.version = !{!N} → !N = !{i32 major, i32 minor, i32 patch}
        guard let airVersionMD = namedMetadata.first(where: { $0.name == "air.version" }),
              let nodeIdx = airVersionMD.operands.first,
              nodeIdx < metadataNodes.count else {
            return nil
        }
        let node = metadataNodes[nodeIdx]
        guard node.operands.count >= 2,
              case .constant(_, .integer(_, let major)) = node.operands[0],
              case .constant(_, .integer(_, let minor)) = node.operands[1] else {
            return nil
        }
        return (UInt16(major), UInt16(minor))
    }

    /// Metal language version extracted from metadata (major, minor).
    public var metalVersion: (major: UInt16, minor: UInt16)? {
        guard let metalLangMD = namedMetadata.first(where: { $0.name == "air.language_version" }),
              let nodeIdx = metalLangMD.operands.first,
              nodeIdx < metadataNodes.count else {
            return nil
        }
        let node = metadataNodes[nodeIdx]
        // !{!"Metal", i32 major, i32 minor, i32 patch}
        guard node.operands.count >= 3,
              case .constant(_, .integer(_, let major)) = node.operands[1],
              case .constant(_, .integer(_, let minor)) = node.operands[2] else {
            return nil
        }
        return (UInt16(major), UInt16(minor))
    }

    /// Kernel function names (from !air.kernel metadata).
    public var kernelEntries: [String] {
        guard let kernelMD = namedMetadata.first(where: { $0.name == "air.kernel" }) else {
            return []
        }
        return kernelMD.operands.compactMap { nodeIdx -> String? in
            guard nodeIdx < metadataNodes.count else { return nil }
            let node = metadataNodes[nodeIdx]
            // First operand should be a function pointer value
            guard let firstOp = node.operands.first,
                  case .value(_, let name) = firstOp else {
                return nil
            }
            return name
        }
    }

    /// Visible function names (from !air.vertex, !air.fragment, or explicit visible).
    public var visibleEntries: [String] {
        // For now, return empty. Will be populated as needed.
        return []
    }
}
