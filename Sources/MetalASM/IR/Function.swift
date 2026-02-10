/// An LLVM IR function.
public final class IRFunction {
    /// The function name (without '@' prefix).
    public var name: String

    /// The function type.
    public var type: IRType

    /// The return type.
    public var returnType: IRType

    /// Parameter types.
    public var parameterTypes: [IRType]

    /// Parameter names (may be empty for unnamed params).
    public var parameterNames: [String]

    /// Parameter attributes (one array per parameter).
    public var parameterAttributes: [[IRAttribute]]

    /// Return attributes.
    public var returnAttributes: [IRAttribute]

    /// Function attributes (e.g., attribute group references).
    public var functionAttributes: [IRAttribute]

    /// Attribute group index (e.g., #0).
    public var attributeGroupIndex: Int?

    /// Linkage type.
    public var linkage: Linkage

    /// Whether this is a declaration (no body) or a definition (has body).
    public var isDeclaration: Bool

    /// Basic blocks (empty for declarations).
    public var basicBlocks: [IRBasicBlock]

    /// Parameters as IRValue references.
    public var parameters: [IRValue]

    /// Calling convention.
    public var callingConvention: CallingConvention

    /// Additional string attributes on parameters.
    public var parameterStringAttributes: [[String: String]]

    /// Section name if specified.
    public var section: String?

    /// Alignment if specified.
    public var alignment: Int?

    /// Address space for the function pointer.
    public var addressSpace: Int

    /// `local_unnamed_addr` flag.
    public var localUnnamedAddr: Bool

    /// `unnamed_addr` flag.
    public var unnamedAddr: Bool

    public init(
        name: String,
        returnType: IRType,
        parameterTypes: [IRType],
        isDeclaration: Bool = false
    ) {
        self.name = name
        self.returnType = returnType
        self.parameterTypes = parameterTypes
        self.type = .function(ret: returnType, params: parameterTypes, isVarArg: false)
        self.parameterNames = Array(repeating: "", count: parameterTypes.count)
        self.parameterAttributes = Array(repeating: [], count: parameterTypes.count)
        self.returnAttributes = []
        self.functionAttributes = []
        self.attributeGroupIndex = nil
        self.linkage = .external
        self.isDeclaration = isDeclaration
        self.basicBlocks = []
        self.parameters = parameterTypes.enumerated().map { i, ty in
            IRValue(type: ty, name: "")
        }
        self.callingConvention = .c
        self.parameterStringAttributes = Array(repeating: [:], count: parameterTypes.count)
        self.section = nil
        self.alignment = nil
        self.addressSpace = 0
        self.localUnnamedAddr = false
        self.unnamedAddr = false
    }

    public enum Linkage: String {
        case external
        case `private`
        case `internal`
        case linkonce
        case linkonceODR = "linkonce_odr"
        case weak
        case weakODR = "weak_odr"
        case common
        case appending
        case externWeak = "extern_weak"
        case available_externally
    }

    public enum CallingConvention: UInt32 {
        case c = 0
        case fast = 8
        case cold = 9
        case swiftcc = 16
        case cxxFastTLS = 17
    }
}
