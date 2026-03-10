/// An LLVM IR attribute that can appear on functions or parameters.
public enum IRAttribute: Hashable {
    // Enum attributes (no value)
    case noUnwind
    case noReturn
    case readNone
    case readOnly
    case writeOnly
    case noCapture
    case signExt
    case zeroExt
    case inReg
    case structRet
    case byVal
    case nest
    case noAlias
    case nonNull
    case convergent
    case mustProgress
    case willReturn
    case noFree
    case noSync
    case noCallback
    case argMemOnly
    case inaccessibleMemOnly
    case inaccessibleMemOrArgMemOnly
    case noRecurse
    case noInline
    case alwaysInline
    case optNone
    case optSize
    case minSize
    case speculatable
    case strictFP
    case immArg
    case noundef
    case returned
    case cold
    case hot
    case naked
    case noBuiltin
    case noImplicitFloat
    case noProfile
    case noSanitizeCoverage
    case noRedZone
    case noMerge

    // Valued enum attributes
    case alignment(Int)
    case dereferenceableAttr(Int)
    case dereferenceableOrNull(Int)
    case stackAlignment(Int)

    // String attributes
    case stringAttr(key: String, value: String?)

    // Attribute group reference (e.g., #0)
    case groupRef(Int)
}

/// A named attribute group (e.g., attributes #0 = { ... }).
public struct IRAttributeGroup {
    /// The group index.
    public var index: Int

    /// The attributes in this group.
    public var attributes: [IRAttribute]

    /// Parameter index for per-param groups (0 = arg 0, 1 = arg 1, etc.).
    /// nil = function-level attributes (encoded as 0xFFFFFFFF in bitcode).
    public var paramIndex: Int?

    public init(index: Int, attributes: [IRAttribute], paramIndex: Int? = nil) {
        self.index = index
        self.attributes = attributes
        self.paramIndex = paramIndex
    }
}
