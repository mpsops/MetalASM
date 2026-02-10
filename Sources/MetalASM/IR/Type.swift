/// LLVM IR type representation.
///
/// Supports the typed pointer model used by Metal AIR (LLVM ~4.0 era),
/// NOT the modern opaque pointer model.
public indirect enum IRType: Hashable {
    case void
    case int(bits: Int)          // i1, i8, i16, i32, i64
    case float16                 // half
    case bfloat16                // bfloat
    case float32                 // float
    case float64                 // double
    case pointer(pointee: IRType, addressSpace: Int)  // typed pointer: T addrspace(N)*
    case array(element: IRType, count: Int)            // [N x T]
    case vector(element: IRType, count: Int)           // <N x T>
    case structure(name: String?, elements: [IRType], isPacked: Bool)  // { T1, T2 } or <{ T1, T2 }>
    case function(ret: IRType, params: [IRType], isVarArg: Bool)       // T (T1, T2, ...)
    case label                   // label type (for basic blocks)
    case metadata                // metadata type
    case opaque(name: String)    // opaque struct (%name = type opaque)
    case token                   // token type

    // Convenience constructors
    static let i1 = IRType.int(bits: 1)
    static let i8 = IRType.int(bits: 8)
    static let i16 = IRType.int(bits: 16)
    static let i32 = IRType.int(bits: 32)
    static let i64 = IRType.int(bits: 64)

    /// Whether this is a void type.
    var isVoid: Bool {
        if case .void = self { return true }
        return false
    }

    /// Whether this is an integer type.
    var isInteger: Bool {
        if case .int = self { return true }
        return false
    }

    /// Whether this is a pointer type.
    var isPointer: Bool {
        if case .pointer = self { return true }
        return false
    }

    /// Get the pointee type if this is a pointer.
    var pointeeType: IRType? {
        if case .pointer(let pointee, _) = self { return pointee }
        return nil
    }

    /// Get the address space if this is a pointer.
    var addressSpace: Int? {
        if case .pointer(_, let addrSpace) = self { return addrSpace }
        return nil
    }

    /// Whether this is a first-class type (can be used as a value).
    var isFirstClass: Bool {
        switch self {
        case .void, .label, .metadata, .function:
            return false
        default:
            return true
        }
    }
}
