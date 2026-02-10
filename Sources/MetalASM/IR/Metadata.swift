/// A reference to a metadata node.
public enum IRMetadataRef: Hashable {
    /// Named metadata index (e.g., !0, !1).
    case index(Int)
    /// Inline string metadata (e.g., !"some string").
    case string(String)
}

/// A metadata node.
public final class IRMetadataNode {
    /// The index of this metadata node.
    public var index: Int

    /// The operands of this metadata node.
    public var operands: [IRMetadataOperand]

    /// Whether this is a distinct metadata node.
    public var isDistinct: Bool

    public init(index: Int, operands: [IRMetadataOperand] = [], isDistinct: Bool = false) {
        self.index = index
        self.operands = operands
        self.isDistinct = isDistinct
    }
}

/// An operand in a metadata node.
public enum IRMetadataOperand: Hashable {
    /// Reference to another metadata node.
    case metadata(Int)
    /// A metadata string.
    case string(String)
    /// A typed constant value.
    case constant(IRType, IRConstant)
    /// Null/empty operand.
    case null
    /// A value reference (function pointer etc).
    case value(IRType, String)  // type + name for functions

    // Hashable conformance for types containing non-Hashable elements
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .metadata(let idx):
            hasher.combine(0)
            hasher.combine(idx)
        case .string(let s):
            hasher.combine(1)
            hasher.combine(s)
        case .constant(let t, let c):
            hasher.combine(2)
            hasher.combine(t)
            hasher.combine(c)
        case .null:
            hasher.combine(3)
        case .value(let t, let n):
            hasher.combine(4)
            hasher.combine(t)
            hasher.combine(n)
        }
    }

    public static func == (lhs: IRMetadataOperand, rhs: IRMetadataOperand) -> Bool {
        switch (lhs, rhs) {
        case (.metadata(let a), .metadata(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.constant(let t1, let c1), .constant(let t2, let c2)): return t1 == t2 && c1 == c2
        case (.null, .null): return true
        case (.value(let t1, let n1), .value(let t2, let n2)): return t1 == t2 && n1 == n2
        default: return false
        }
    }
}

/// A named metadata entry (e.g., !air.kernel = !{!0}).
public struct IRNamedMetadata {
    /// The name (e.g., "air.kernel").
    public var name: String

    /// References to metadata nodes.
    public var operands: [Int]

    public init(name: String, operands: [Int]) {
        self.name = name
        self.operands = operands
    }
}
