/// An LLVM IR instruction within a basic block.
public final class IRInstruction {
    /// The instruction opcode.
    public var opcode: Opcode

    /// The result type (void for instructions with no result).
    public var type: IRType

    /// The result name (empty if unnamed or void result).
    public var name: String

    /// Operands: values referenced by this instruction.
    public var operands: [Operand]

    /// Additional instruction-specific attributes.
    public var attributes: InstructionAttributes

    public init(
        opcode: Opcode,
        type: IRType = .void,
        name: String = "",
        operands: [Operand] = [],
        attributes: InstructionAttributes = InstructionAttributes()
    ) {
        self.opcode = opcode
        self.type = type
        self.name = name
        self.operands = operands
        self.attributes = attributes
    }

    /// Instruction opcodes.
    public enum Opcode: UInt32 {
        // Terminator instructions (1-11)
        case ret = 1
        case br = 2
        case switchInst = 3
        case indirectBr = 4
        case invoke = 5
        case unreachable = 7
        case cleanupRet = 8
        case catchRet = 9
        case catchSwitch = 10
        case callBr = 11

        // Unary operators (12-13)
        case fneg = 12

        // Binary operators (13-28)
        case add = 13
        case fadd = 14
        case sub = 15
        case fsub = 16
        case mul = 17
        case fmul = 18
        case udiv = 19
        case sdiv = 20
        case fdiv = 21
        case urem = 22
        case srem = 23
        case frem = 24

        // Logical operators (25-28)
        case shl = 25
        case lshr = 26
        case ashr = 27
        case and = 28
        case or = 29
        case xor = 30

        // Memory operators (31-40)
        case alloca = 31
        case load = 32
        case store = 33
        case getelementptr = 34
        case fence = 35
        case cmpxchg = 36
        case atomicRMW = 37

        // Cast operators (38-50)
        case trunc = 38
        case zext = 39
        case sext = 40
        case fpToUI = 41
        case fpToSI = 42
        case uiToFP = 43
        case siToFP = 44
        case fpTrunc = 45
        case fpExt = 46
        case ptrToInt = 47
        case intToPtr = 48
        case bitcast = 49
        case addrSpaceCast = 50

        // Other operators
        case icmp = 53
        case fcmp = 54
        case phi = 55
        case call = 56
        case select = 57
        case extractValue = 62
        case insertValue = 63
        case extractElement = 64
        case insertElement = 65
        case shuffleVector = 66
    }

    /// An operand to an instruction.
    public enum Operand {
        case value(IRValue)
        case constant(IRConstant)
        case basicBlock(IRBasicBlock)
        case type(IRType)
        case intLiteral(Int64)
        case metadata(IRMetadataRef)
    }

    /// Extra attributes for instructions.
    public struct InstructionAttributes {
        /// For GEP: is inbounds?
        public var inBounds: Bool = false

        /// For load/store: alignment
        public var alignment: Int? = nil

        /// For load: is volatile?
        public var isVolatile: Bool = false

        /// For call: calling convention
        public var callingConvention: UInt32? = nil

        /// For call: tail call kind
        public var tailCall: TailCallKind? = nil

        /// For call: function attributes group index
        public var funcAttributes: [Int] = []

        /// For call: parameter attributes
        public var paramAttributes: [[IRAttribute]] = []

        /// For call: return attributes
        public var returnAttributes: [IRAttribute] = []

        /// Fast math flags
        public var fastMathFlags: FastMathFlags = FastMathFlags()

        /// For icmp/fcmp: predicate
        public var predicate: Int? = nil

        /// For alloca: alignment, count type
        public var allocaNumElements: Operand? = nil
        public var allocaType: IRType? = nil

        /// For GEP: source element type
        public var gepSourceType: IRType? = nil

        public init() {}
    }

    public enum TailCallKind {
        case none
        case tail
        case mustTail
        case noTail
    }

    public struct FastMathFlags {
        public var nnan: Bool = false
        public var ninf: Bool = false
        public var nsz: Bool = false
        public var arcp: Bool = false
        public var contract: Bool = false
        public var afn: Bool = false
        public var reassoc: Bool = false
        public var fast: Bool = false
        public init() {}
    }
}
