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

    public init(type: IRType, name: String = "") {
        self.type = type
        self.name = name
    }
}

/// A constant value.
public indirect enum IRConstant: Hashable {
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
