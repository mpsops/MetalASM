/// An SSA value in the IR.
///
/// Values are referenced by other instructions via their index in the
/// function's value table or in the module's global value table.
public final class IRValue {
    /// The type of this value.
    public var type: IRType

    /// The name of this value (may be empty for unnamed values).
    public var name: String

    /// Unique ID assigned during value enumeration.
    var valueID: Int = -1

    /// Non-zero if this value represents a constant GEP offset from a global.
    /// The IR transform will flatten this into a separate GEP instruction.
    public var constantGEPByteOffset: Int64 = 0

    public init(type: IRType, name: String = "") {
        self.type = type
        self.name = name
    }
}

/// A constant value.
///
/// Custom Hashable/Equatable uses bitPattern for floats so NaN == NaN
/// (IEEE 754 says NaN != NaN, which breaks dictionary lookup).
public indirect enum IRConstant: Equatable, Hashable {
    case integer(IRType, Int64)
    case float32(Float)
    case float64(Double)
    case float16(UInt16)  // stored as raw bits
    case bfloat16(UInt16) // stored as raw bits
    case null(IRType)
    case undef(IRType)
    case zeroInitializer(IRType)
    case string([UInt8])
    case structValue(IRType, [IRConstant])
    case arrayValue(IRType, [IRConstant])
    case vectorValue(IRType, [IRConstant])

    /// Constant expression operations.
    case bitcast(IRConstant, IRType)
    case getelementptr(inBounds: Bool, IRType, IRConstant, [IRConstant])
    case inttoptr(IRConstant, IRType)
    case ptrtoint(IRConstant, IRType)

    public static func == (lhs: IRConstant, rhs: IRConstant) -> Bool {
        switch (lhs, rhs) {
        case (.float32(let a), .float32(let b)):
            return a.bitPattern == b.bitPattern
        case (.float64(let a), .float64(let b)):
            return a.bitPattern == b.bitPattern
        case (.integer(let t1, let v1), .integer(let t2, let v2)):
            return t1 == t2 && v1 == v2
        case (.float16(let a), .float16(let b)):
            return a == b
        case (.bfloat16(let a), .bfloat16(let b)):
            return a == b
        case (.null(let a), .null(let b)):
            return a == b
        case (.undef(let a), .undef(let b)):
            return a == b
        case (.zeroInitializer(let a), .zeroInitializer(let b)):
            return a == b
        case (.string(let a), .string(let b)):
            return a == b
        case (.structValue(let t1, let e1), .structValue(let t2, let e2)):
            return t1 == t2 && e1 == e2
        case (.arrayValue(let t1, let e1), .arrayValue(let t2, let e2)):
            return t1 == t2 && e1 == e2
        case (.vectorValue(let t1, let e1), .vectorValue(let t2, let e2)):
            return t1 == t2 && e1 == e2
        case (.bitcast(let c1, let t1), .bitcast(let c2, let t2)):
            return c1 == c2 && t1 == t2
        case (.getelementptr(let ib1, let t1, let p1, let i1), .getelementptr(let ib2, let t2, let p2, let i2)):
            return ib1 == ib2 && t1 == t2 && p1 == p2 && i1 == i2
        case (.inttoptr(let c1, let t1), .inttoptr(let c2, let t2)):
            return c1 == c2 && t1 == t2
        case (.ptrtoint(let c1, let t1), .ptrtoint(let c2, let t2)):
            return c1 == c2 && t1 == t2
        default:
            return false
        }
    }

    public func hash(into hasher: inout Hasher) {
        switch self {
        case .float32(let v):
            hasher.combine(0)
            hasher.combine(v.bitPattern)
        case .float64(let v):
            hasher.combine(1)
            hasher.combine(v.bitPattern)
        case .integer(let t, let v):
            hasher.combine(2)
            hasher.combine(t)
            hasher.combine(v)
        case .float16(let v):
            hasher.combine(3)
            hasher.combine(v)
        case .bfloat16(let v):
            hasher.combine(4)
            hasher.combine(v)
        case .null(let t):
            hasher.combine(5)
            hasher.combine(t)
        case .undef(let t):
            hasher.combine(6)
            hasher.combine(t)
        case .zeroInitializer(let t):
            hasher.combine(7)
            hasher.combine(t)
        case .string(let b):
            hasher.combine(8)
            hasher.combine(b)
        case .structValue(let t, let e):
            hasher.combine(9)
            hasher.combine(t)
            hasher.combine(e)
        case .arrayValue(let t, let e):
            hasher.combine(10)
            hasher.combine(t)
            hasher.combine(e)
        case .vectorValue(let t, let e):
            hasher.combine(11)
            hasher.combine(t)
            hasher.combine(e)
        case .bitcast(let c, let t):
            hasher.combine(12)
            hasher.combine(c)
            hasher.combine(t)
        case .getelementptr(let ib, let t, let p, let i):
            hasher.combine(13)
            hasher.combine(ib)
            hasher.combine(t)
            hasher.combine(p)
            hasher.combine(i)
        case .inttoptr(let c, let t):
            hasher.combine(14)
            hasher.combine(c)
            hasher.combine(t)
        case .ptrtoint(let c, let t):
            hasher.combine(15)
            hasher.combine(c)
            hasher.combine(t)
        }
    }

    /// The type of this constant.
    public var type: IRType {
        switch self {
        case .integer(let t, _): return t
        case .float32: return .float32
        case .float64: return .float64
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .null(let t): return t
        case .undef(let t): return t
        case .zeroInitializer(let t): return t
        case .string(let bytes): return .array(element: .i8, count: bytes.count)
        case .structValue(let t, _): return t
        case .arrayValue(let t, _): return t
        case .vectorValue(let t, _): return t
        case .bitcast(_, let t): return t
        case .getelementptr(_, _, _, _): return .i64  // simplified
        case .inttoptr(_, let t): return t
        case .ptrtoint(_, let t): return t
        }
    }
}
