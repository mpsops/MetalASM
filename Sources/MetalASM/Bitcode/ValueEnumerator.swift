/// Assigns numeric IDs to all types and values in a module.
///
/// LLVM bitcode references types and values by their integer index in
/// respective tables. This class builds those tables.
final class ValueEnumerator {
    /// All unique types, in the order they should appear in the TYPE_BLOCK.
    private(set) var types: [IRType] = []

    /// Map from type to its index in the type table.
    private var typeMap: [IRType: Int] = [:]

    /// All global values (globals + functions) in module order.
    private(set) var globalValues: [GlobalValueEntry] = []

    /// Map from global value name to its index.
    private var globalValueMap: [String: Int] = [:]

    /// Attribute groups indexed by their group number.
    private(set) var attributeGroups: [IRAttributeGroup] = []

    /// The module being enumerated.
    let module: IRModule

    struct GlobalValueEntry {
        enum Kind {
            case global(IRGlobal)
            case function(IRFunction)
        }
        let kind: Kind
        let name: String
        var valueID: Int
    }

    init(module: IRModule) {
        self.module = module
        enumerate()
    }

    // MARK: - Type table

    /// Get or assign an index for a type.
    @discardableResult
    func enumerateType(_ type: IRType) -> Int {
        if let existing = typeMap[type] {
            return existing
        }
        // Enumerate component types first (depth-first)
        switch type {
        case .pointer(let pointee, _):
            enumerateType(pointee)
        case .array(let elem, _):
            enumerateType(elem)
        case .vector(let elem, _):
            enumerateType(elem)
        case .structure(_, let elems, _):
            for elem in elems {
                enumerateType(elem)
            }
        case .function(let ret, let params, _):
            enumerateType(ret)
            for p in params {
                enumerateType(p)
            }
        default:
            break
        }

        let idx = types.count
        types.append(type)
        typeMap[type] = idx
        return idx
    }

    /// Look up the type index (must already be enumerated).
    func typeIndex(_ type: IRType) -> Int {
        guard let idx = typeMap[type] else {
            fatalError("Type not enumerated: \(type)")
        }
        return idx
    }

    // MARK: - Value enumeration

    /// Enumerate all values in the module.
    private func enumerate() {
        // 1. Enumerate struct types first
        for (_, type) in module.structTypes {
            enumerateType(type)
        }

        // 2. Enumerate all types used in globals
        for global in module.globals {
            enumerateType(global.type)
            enumerateType(global.valueType)
            if let init_ = global.initializer {
                enumerateConstantType(init_)
            }
        }

        // 3. Enumerate all types used in functions
        for fn in module.functions {
            enumerateType(fn.type)
            enumerateType(fn.returnType)
            for pt in fn.parameterTypes {
                enumerateType(pt)
            }
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    enumerateType(inst.type)
                    enumerateInstructionTypes(inst)
                }
            }
        }

        // 3b. Enumerate metadata and label types (always needed by LLVM)
        enumerateType(.metadata)

        // 4. Enumerate types used in metadata
        for node in module.metadataNodes {
            for op in node.operands {
                switch op {
                case .constant(let type, _):
                    enumerateType(type)
                case .value(let type, _):
                    enumerateType(type)
                default:
                    break
                }
            }
        }

        // 5. Build global value table: globals first, then functions
        var valueID = 0
        for global in module.globals {
            let entry = GlobalValueEntry(
                kind: .global(global),
                name: global.name,
                valueID: valueID
            )
            globalValues.append(entry)
            globalValueMap[global.name] = valueID
            valueID += 1
        }
        for fn in module.functions {
            let entry = GlobalValueEntry(
                kind: .function(fn),
                name: fn.name,
                valueID: valueID
            )
            globalValues.append(entry)
            globalValueMap[fn.name] = valueID
            valueID += 1
        }

        // 6. Copy attribute groups
        attributeGroups = module.attributeGroups
    }

    /// Enumerate types used by a constant.
    private func enumerateConstantType(_ constant: IRConstant) {
        enumerateType(constant.type)
        switch constant {
        case .structValue(_, let elems):
            for e in elems { enumerateConstantType(e) }
        case .arrayValue(_, let elems):
            for e in elems { enumerateConstantType(e) }
        case .vectorValue(_, let elems):
            for e in elems { enumerateConstantType(e) }
        case .bitcast(let inner, let ty):
            enumerateConstantType(inner)
            enumerateType(ty)
        case .getelementptr(_, let baseTy, let ptr, let indices):
            enumerateType(baseTy)
            enumerateConstantType(ptr)
            for idx in indices { enumerateConstantType(idx) }
        case .inttoptr(let inner, let ty):
            enumerateConstantType(inner)
            enumerateType(ty)
        case .ptrtoint(let inner, let ty):
            enumerateConstantType(inner)
            enumerateType(ty)
        default:
            break
        }
    }

    /// Enumerate types used by instruction operands.
    private func enumerateInstructionTypes(_ inst: IRInstruction) {
        for op in inst.operands {
            switch op {
            case .value(let val):
                enumerateType(val.type)
            case .constant(let c):
                enumerateConstantType(c)
            case .type(let t):
                enumerateType(t)
            default:
                break
            }
        }
        // GEP source element type
        if let gepTy = inst.attributes.gepSourceType {
            enumerateType(gepTy)
        }
        // Alloca type
        if let allocaTy = inst.attributes.allocaType {
            enumerateType(allocaTy)
        }
    }

    // MARK: - Global value lookup

    /// Get the value ID for a global value by name.
    func globalValueID(name: String) -> Int? {
        return globalValueMap[name]
    }

    /// Total number of global values.
    var globalValueCount: Int {
        return globalValues.count
    }

    /// Find a function by name.
    func findFunction(named name: String) -> IRFunction? {
        for fn in module.functions {
            if fn.name == name { return fn }
        }
        return nil
    }
}
