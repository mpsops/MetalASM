/// A global variable in the LLVM IR module.
public final class IRGlobal {
    /// The global variable name (without '@' prefix).
    public var name: String

    /// The type of the global variable.
    public var type: IRType

    /// The pointee type (what the global points to).
    public var valueType: IRType

    /// The initializer constant (nil for declarations).
    public var initializer: IRConstant?

    /// Linkage type.
    public var linkage: IRFunction.Linkage

    /// Whether this is a constant.
    public var isConstant: Bool

    /// Address space.
    public var addressSpace: Int

    /// Alignment in bytes.
    public var alignment: Int?

    /// Section name.
    public var section: String?

    /// `local_unnamed_addr` flag.
    public var localUnnamedAddr: Bool

    /// `unnamed_addr` flag.
    public var unnamedAddr: Bool

    /// Thread local mode.
    public var threadLocal: ThreadLocalMode?

    /// Externally initialized.
    public var externallyInitialized: Bool

    public init(
        name: String,
        valueType: IRType,
        addressSpace: Int = 0,
        initializer: IRConstant? = nil
    ) {
        self.name = name
        self.valueType = valueType
        self.type = .pointer(pointee: valueType, addressSpace: addressSpace)
        self.initializer = initializer
        self.linkage = .external
        self.isConstant = false
        self.addressSpace = addressSpace
        self.alignment = nil
        self.section = nil
        self.localUnnamedAddr = false
        self.unnamedAddr = false
        self.threadLocal = nil
        self.externallyInitialized = false
    }

    public enum ThreadLocalMode: String {
        case generalDynamic = "generaldynamic"
        case localDynamic = "localdynamic"
        case initialExec = "initialexec"
        case localExec = "localexec"
    }
}
