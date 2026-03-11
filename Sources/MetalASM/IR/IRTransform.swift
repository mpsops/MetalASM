/// IRTransform: Post-parse lowering passes for Metal AIR compatibility.
///
/// Applies 4 transforms to an IRModule in-memory:
///   1. air.threadgroup.barrier → air.wg.barrier rename + arg fixup
///   2. MMA intrinsic typed-pointer fixup (opaque ptr → float addrspace(3)*)
///   3. Threadgroup global GEP rewrite (opaque ptr → typed [N x float] addrspace(3)*)
///   4. Air system-value lowering: inject tid/pid/simdlane params, remove intrinsic
///      calls, emit !air.kernel metadata

// MARK: - Public entry point

/// Apply all Air compatibility transforms to a parsed module, mutating it in-place.
public func applyAirTransforms(module: IRModule) {
    let airDataLayout =
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32" +
        "-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32" +
        "-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256" +
        "-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
    let airTriple = "air64_v28-apple-macosx26.0.0"

    module.dataLayout = airDataLayout
    module.targetTriple = airTriple

    transformStructPhis(module: module)
    transformBarrierRename(module: module)
    transformLLVMIntrinsicRename(module: module)
    transformAtomicRMW(module: module)
    transformTGGlobalGEPs(module: module)
    transformMMATypedPtrs(module: module)
    transformBFloat16Casts(module: module)
    transformScalarStoreGuard(module: module)
    transformAirSystemValues(module: module)
    transformDeviceLoadsVolatile(module: module)
}

// MARK: - Transform: Decompose int-to-bfloat casts
//
// Metal GPU JIT treats `sitofp iN to bfloat` as `sitofp iN to half` (wrong).
// Decompose into: sitofp iN to float + fptrunc float to bfloat.
// Same for uitofp.

private func transformBFloat16Casts(module: IRModule) {
    for fn in module.functions {
        for bb in fn.basicBlocks {
            var i = 0
            while i < bb.instructions.count {
                let inst = bb.instructions[i]
                let isCastToBF16 = (inst.opcode == .siToFP || inst.opcode == .uiToFP) &&
                                   inst.type == .bfloat16
                guard isCastToBF16 else { i += 1; continue }

                // Metal GPU JIT doesn't support sitofp/uitofp to bfloat correctly.
                // Reference Metal compiler uses: air.convert.f.f32.s/u.iN(iN) + fptrunc float to bfloat.
                // We decompose: sitofp/uitofp iN to bfloat → call air.convert + fptrunc float to bfloat.

                // Determine source type
                let isSigned = inst.opcode == .siToFP
                let srcOperand = inst.operands[0]
                var srcType: IRType = .i32
                if case .value(let v) = srcOperand { srcType = v.type }

                let signChar = isSigned ? "s" : "u"
                let srcBits: String
                switch srcType {
                case .i8: srcBits = "i8"
                case .i16: srcBits = "i16"
                case .i32: srcBits = "i32"
                case .i64: srcBits = "i64"
                default: srcBits = "i32"
                }
                let intrinsicName = "air.convert.f.f32.\(signChar).\(srcBits)"

                // Declare the intrinsic if not already declared
                if !module.functions.contains(where: { $0.name == intrinsicName }) {
                    let decl = IRFunction(
                        name: intrinsicName,
                        returnType: .float32,
                        parameterTypes: [srcType],
                        isDeclaration: true
                    )
                    module.functions.append(decl)
                }

                // Create call to air.convert
                // Call operands: [arg0, ..., argN, function_value]
                let fnType = IRType.pointer(
                    pointee: .function(ret: .float32, params: [srcType], isVarArg: false),
                    addressSpace: 0
                )
                let fnVal = IRValue(type: fnType, name: intrinsicName)
                var callOps = inst.operands
                callOps.append(.value(fnVal))
                let f32Name = inst.name.isEmpty ? "__bf16_f32_\(i)" : "\(inst.name).f32"
                let callInst = IRInstruction(
                    opcode: .call,
                    type: .float32,
                    name: f32Name,
                    operands: callOps
                )
                let f32Val = IRValue(type: .float32, name: f32Name)

                // Replace original with: fptrunc float to bfloat
                inst.opcode = .fpTrunc
                inst.operands = [.value(f32Val)]

                bb.instructions.insert(callInst, at: i)
                i += 2 // skip both instructions
            }
        }
    }
}

/// Infer buffer element type from GEP instructions that use a given function parameter.
/// Returns (size, align, name) for AIR metadata.
private func inferBufferElementType(fn: IRFunction, paramIdx: Int) -> (Int, Int, String) {
    let paramName = paramIdx < fn.parameters.count ? fn.parameters[paramIdx].name : ""
    // Scan all GEPs in the function looking for ones that reference this parameter
    for bb in fn.basicBlocks {
        for inst in bb.instructions {
            guard inst.opcode == .getelementptr else { continue }
            // GEP operands[0] = base ptr, source type in attributes.gepSourceType
            if case .value(let val) = inst.operands[0], val.name == paramName {
                if let elemType = inst.attributes.gepSourceType {
                    return airTypeInfo(elemType)
                }
            }
        }
    }
    // Default: float
    return (4, 4, "float")
}

private func airTypeInfo(_ type: IRType) -> (Int, Int, String) {
    switch type {
    case .i1, .i8: return (1, 1, "char")
    case .i16: return (2, 2, "short")
    case .i32: return (4, 4, "int")
    case .i64: return (8, 8, "long")
    case .float16: return (2, 2, "half")
    case .bfloat16: return (2, 2, "bfloat")
    case .float32: return (4, 4, "float")
    case .float64: return (8, 8, "double")
    default: return (4, 4, "float")
    }
}

// MARK: - Transform 0: Split struct phis into scalar phis
//
// Metal GPU JIT crashes on phi nodes with struct types ({float, float}).
// Split them into individual scalar phis and rewrite extractvalue/insertvalue.

private func transformStructPhis(module: IRModule) {
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            var newInstructions: [IRInstruction] = []
            // Map: struct phi name → [scalar phi names]
            var scalarPhiNames: [String: [String]] = [:]
            // Map: struct phi name → [element types]
            var scalarPhiTypes: [String: [IRType]] = [:]

            for inst in bb.instructions {
                guard inst.opcode == .phi,
                      case .structure(_, let elemTypes, _) = inst.type else {
                    newInstructions.append(inst)
                    continue
                }

                // Create one scalar phi per struct element
                var names: [String] = []
                for (elemIdx, elemType) in elemTypes.enumerated() {
                    let scalarName = "\(inst.name)_e\(elemIdx)"
                    names.append(scalarName)

                    // Build operands for scalar phi: extract element from each incoming value
                    var scalarOps: [IRInstruction.Operand] = []
                    // Phi operands: [val0, bb0, val1, bb1, ...]
                    var i = 0
                    while i < inst.operands.count - 1 {
                        let valOp = inst.operands[i]
                        let bbOp = inst.operands[i + 1]

                        // Determine scalar value for this incoming edge
                        let scalarVal: IRInstruction.Operand
                        switch valOp {
                        case .constant(let c):
                            switch c {
                            case .zeroInitializer(_):
                                // zeroinitializer → 0.0 for float, 0 for int
                                if case .float32 = elemType {
                                    scalarVal = .constant(.float32(0.0))
                                } else if case .int(let bits) = elemType {
                                    scalarVal = .constant(.integer(.int(bits: bits), 0))
                                } else {
                                    scalarVal = .constant(.zeroInitializer(elemType))
                                }
                            default:
                                scalarVal = valOp // fallback
                            }
                        case .value(let v):
                            // This is an insertvalue result — we'll resolve it later
                            // For now, reference the scalar phi name that will be created
                            // from the insertvalue chain in the predecessor block
                            scalarVal = .value(IRValue(type: elemType, name: "\(v.name)_e\(elemIdx)"))
                        default:
                            scalarVal = valOp
                        }

                        scalarOps.append(scalarVal)
                        scalarOps.append(bbOp)
                        i += 2
                    }

                    let scalarPhi = IRInstruction(
                        opcode: .phi,
                        type: elemType,
                        name: scalarName,
                        operands: scalarOps
                    )
                    newInstructions.append(scalarPhi)
                }

                scalarPhiNames[inst.name] = names
                scalarPhiTypes[inst.name] = elemTypes
            }

            guard !scalarPhiNames.isEmpty else { continue }
            bb.instructions = newInstructions
        }

        // Now rewrite extractvalue/insertvalue across ALL basic blocks
        // Collect scalar phi info
        var allScalarPhis: [String: [String]] = [:]
        var allScalarTypes: [String: [IRType]] = [:]
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .phi {
                // Check if this was generated by our transform (name contains _e0, _e1, etc.)
                // Actually, track by the struct phi name
            }
        }

        // Re-scan: find struct phis that were split (they're gone now, replaced by _e0, _e1)
        // We need to track which names were struct phis
        // Let's re-collect from what we have
        for bb in fn.basicBlocks {
            var scalarGroups: [String: [String]] = [:]
            for inst in bb.instructions where inst.opcode == .phi {
                // Pattern: "X_eN" where X was the original struct phi
                if let range = inst.name.range(of: "_e", options: .backwards) {
                    let base = String(inst.name[..<range.lowerBound])
                    let idxStr = String(inst.name[range.upperBound...])
                    if let _ = Int(idxStr) {
                        scalarGroups[base, default: []].append(inst.name)
                    }
                }
            }
            for (base, names) in scalarGroups {
                allScalarPhis[base] = names.sorted()
                // Get types from the phi instructions
                allScalarTypes[base] = names.sorted().compactMap { name in
                    bb.instructions.first(where: { $0.name == name })?.type
                }
            }
        }

        guard !allScalarPhis.isEmpty else { continue }

        // Pass 1: Collect all insertvalue chains across all blocks
        // Map: insertvalue result name → {elemIdx: scalar operand}
        var insertValueScalars: [String: [Int: IRInstruction.Operand]] = [:]
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .insertValue && inst.operands.count >= 3 {
                if case .intLiteral(let idx) = inst.operands[2] {
                    insertValueScalars[inst.name, default: [:]][Int(idx)] = inst.operands[1]
                    // Inherit from aggregate base
                    if case .value(let agg) = inst.operands[0],
                       let existing = insertValueScalars[agg.name] {
                        for (k, v) in existing where insertValueScalars[inst.name]?[k] == nil {
                            insertValueScalars[inst.name, default: [:]][k] = v
                        }
                    }
                }
            }
        }

        // Pass 2: Fix scalar phi incoming values that reference insertvalue results
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .phi {
                // Check if this is a scalar phi from our split
                if let range = inst.name.range(of: "_e", options: .backwards),
                   let elemIdx = Int(String(inst.name[range.upperBound...])) {
                    // Fix incoming values: replace insertvalue_result_eN with actual scalar
                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j] {
                            // Strip _eN suffix to get the insertvalue name
                            if let r2 = v.name.range(of: "_e\(elemIdx)", options: .backwards),
                               r2.upperBound == v.name.endIndex {
                                let ivName = String(v.name[..<r2.lowerBound])
                                if let scalarOp = insertValueScalars[ivName]?[elemIdx] {
                                    inst.operands[j] = scalarOp
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pass 3: Replace extractvalue from struct phis OR insertvalue chains, remove insertvalues
        // Build rename map: extractvalue result name → replacement name
        var renameMap: [String: (String, IRType)] = [:]
        for bb in fn.basicBlocks {
            var newInsts: [IRInstruction] = []
            for inst in bb.instructions {
                if inst.opcode == .extractValue,
                   inst.operands.count >= 2,
                   case .value(let agg) = inst.operands[0],
                   case .intLiteral(let idx) = inst.operands[1] {
                    // Check if aggregate is a struct phi or insertvalue chain
                    if let _ = allScalarPhis[agg.name] {
                        let scalarName = "\(agg.name)_e\(idx)"
                        let elemType = allScalarTypes[agg.name]?[Int(idx)] ?? .float32
                        renameMap[inst.name] = (scalarName, elemType)
                        continue // Drop extractvalue
                    }
                    if let scalars = insertValueScalars[agg.name],
                       let scalarOp = scalars[Int(idx)],
                       case .value(let scalarVal) = scalarOp {
                        renameMap[inst.name] = (scalarVal.name, scalarVal.type)
                        continue // Drop extractvalue
                    }
                }

                if inst.opcode == .insertValue {
                    // Check if this is part of a struct phi chain
                    if case .value(let agg) = inst.operands[0] {
                        if agg.name == "undef" || insertValueScalars[agg.name] != nil ||
                           (inst.operands.count >= 1 && {
                               if case .constant(.undef(_)) = inst.operands[0] { return true }
                               return false
                           }()) {
                            continue // Drop insertvalue in struct phi chains
                        }
                    }
                    if case .constant(.undef(_)) = inst.operands[0] {
                        continue // Drop insertvalue with undef base
                    }
                }

                newInsts.append(inst)
            }
            bb.instructions = newInsts
        }

        // Pass 4: Apply renames throughout the function
        for (oldName, (newName, newType)) in renameMap {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j], v.name == oldName {
                            inst.operands[j] = .value(IRValue(type: newType, name: newName))
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Transform 1: barrier rename

private func transformBarrierRename(module: IRModule) {
    // Rename the declaration
    for fn in module.functions where fn.isDeclaration {
        if fn.name == "air.threadgroup.barrier" {
            fn.name = "air.wg.barrier"
        }
    }
    // Fix call sites: change callee name + update args from (i32 1, i32 4) → (i32 2, i32 1)
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .call {
                guard let calleeOp = inst.operands.last,
                      case .value(let calleeVal) = calleeOp,
                      calleeVal.name == "air.threadgroup.barrier" else { continue }
                calleeVal.name = "air.wg.barrier"
                // Replace args: first arg 1→2, second arg 4→1
                if inst.operands.count >= 3 {
                    if case .constant(let c) = inst.operands[0], case .integer(let t, 1) = c {
                        inst.operands[0] = .constant(.integer(t, 2))
                    }
                    if case .constant(let c) = inst.operands[1], case .integer(let t, 4) = c {
                        inst.operands[1] = .constant(.integer(t, 1))
                    }
                }
            }
        }
    }
}

// MARK: - Transform 1.5: LLVM intrinsic → AIR intrinsic renaming

/// Map LLVM math intrinsics to their AIR equivalents.
/// e.g. llvm.maxnum.f32 → air.fast_fmax.f32, llvm.minnum.f32 → air.fast_fmin.f32
private let llvmToAirIntrinsics: [(String, String)] = [
    // min/max
    ("llvm.maxnum.f32", "air.fmax.f32"),
    ("llvm.minnum.f32", "air.fmin.f32"),
    ("llvm.maxnum.f16", "air.fmax.f16"),
    ("llvm.minnum.f16", "air.fmin.f16"),
    // trig
    ("llvm.sin.f32",    "air.fast_sin.f32"),
    ("llvm.cos.f32",    "air.fast_cos.f32"),
    ("llvm.sin.f16",    "air.fast_sin.f16"),
    ("llvm.cos.f16",    "air.fast_cos.f16"),
    // exp/log
    ("llvm.exp.f32",    "air.fast_exp.f32"),
    ("llvm.log.f32",    "air.fast_log.f32"),
    ("llvm.exp2.f32",   "air.fast_exp2.f32"),
    ("llvm.log2.f32",   "air.fast_log2.f32"),
    ("llvm.exp.f16",    "air.fast_exp.f16"),
    ("llvm.log.f16",    "air.fast_log.f16"),
    ("llvm.exp2.f16",   "air.fast_exp2.f16"),
    ("llvm.log2.f16",   "air.fast_log2.f16"),
    // sqrt/abs/floor/ceil
    ("llvm.sqrt.f32",   "air.fast_sqrt.f32"),
    ("llvm.fabs.f32",   "air.fabs.f32"),
    ("llvm.floor.f32",  "air.fast_floor.f32"),
    ("llvm.ceil.f32",   "air.fast_ceil.f32"),
    ("llvm.sqrt.f16",   "air.fast_sqrt.f16"),
    ("llvm.fabs.f16",   "air.fabs.f16"),
    ("llvm.floor.f16",  "air.fast_floor.f16"),
    ("llvm.ceil.f16",   "air.fast_ceil.f16"),
    // fma
    ("llvm.fma.f32",    "air.fma.f32"),
    ("llvm.fma.f16",    "air.fma.f16"),
]

private func transformLLVMIntrinsicRename(module: IRModule) {
    let nameMap = Dictionary(llvmToAirIntrinsics, uniquingKeysWith: { _, b in b })
    // Rename declarations and strip LLVM intrinsic attributes
    for fn in module.functions where fn.isDeclaration {
        if let airName = nameMap[fn.name] {
            fn.name = airName
            fn.attributeGroupIndex = nil
        }
    }
    // Rename call sites
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .call {
                guard let calleeOp = inst.operands.last,
                      case .value(let calleeVal) = calleeOp,
                      let airName = nameMap[calleeVal.name] else { continue }
                calleeVal.name = airName
            }
        }
    }

    // Remove attribute groups that are no longer referenced by any function
    let usedGroupIndices = Set(module.functions.compactMap { $0.attributeGroupIndex })
    module.attributeGroups.removeAll { !usedGroupIndices.contains($0.index) }
}

// MARK: - Transform 2: MMA load/store elimination + multiply-accumulate fixup
//
// air.simdgroup_matrix_8x8_load/store crash the Metal GPU shader compiler XPC service.
// The upstream metal-flash-attention never uses these intrinsics — it uses async copy
// to move data into threadgroup memory and operates directly on <64 x float> registers.
//
// Replace:
//   %r = call <64 x float> @air.simdgroup_matrix_8x8_load(ptr addrspace(3) %p, ...)
// With:
//   %r = load <64 x float>, <64 x float> addrspace(3)* %p, align 4
//
// Replace:
//   call void @air.simdgroup_matrix_8x8_store(<64 x float> %v, ptr addrspace(3) %p, ...)
// With:
//   store <64 x float> %v, <64 x float> addrspace(3)* %p, align 4
//
// For multiply_accumulate, just add fast-math flag — no replacement needed.

private let mmaLoadName  = "air.simdgroup_matrix_8x8_load.v64f32.p3f32"
private let mmaStoreName = "air.simdgroup_matrix_8x8_store.v64f32.p3f32"
private let mmaMulName   = "air.simdgroup_matrix_8x8_multiply_accumulate"

private func transformMMATypedPtrs(module: IRModule) {
    let floatTGPtr = IRType.pointer(pointee: .float32, addressSpace: 3)
    let floatDevPtr = IRType.pointer(pointee: .float32, addressSpace: 1)

    // Fix declaration param types: opaquePointer(3) → float addrspace(3)*
    // Add nocapture+readonly paramattr on mmaLoad param 1 (matches ref bc)
    for fn in module.functions where fn.isDeclaration {
        if fn.name == mmaLoadName || fn.name == mmaStoreName {
            fn.parameterTypes = fn.parameterTypes.map { t in
                if case .opaquePointer(3) = t { return floatTGPtr }
                return t
            }
            fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)
        }
        if fn.name == mmaLoadName {
            let groupIdx = module.attributeGroups.count
            module.attributeGroups.append(IRAttributeGroup(
                index: groupIdx,
                attributes: [.noCapture, .readOnly],
                paramIndex: 1
            ))
            fn.attributeGroupIndex = groupIdx
            module.paramAttrLists.append([groupIdx])
        }
        if fn.name == mmaStoreName {
            let groupIdx = module.attributeGroups.count
            module.attributeGroups.append(IRAttributeGroup(
                index: groupIdx,
                attributes: [.noCapture, .writeOnly],
                paramIndex: 2  // param 2 in store: (value, ptr, stride, stride, stride)
            ))
            fn.attributeGroupIndex = groupIdx
            module.paramAttrLists.append([groupIdx])
        }
    }

    // Convert remaining opaque pointer params to typed pointers based on usage.
    // Metal AIR (LLVM 14) requires typed pointers — opaque ptrs crash the GPU JIT
    // when complex ops (typed GEPs) reference them. IRTransform converts param
    // types; emitOpaqueAsTyped catches remaining opaque ptr types (GEP results,
    // intermediates) at bitcode emission time. Both gates live here.
    let hasMMADecl = module.functions.contains { $0.name.hasPrefix("air.simdgroup_matrix_8x8_") }
    let hasTGByteGlobal = module.globals.contains { g in
        g.addressSpace == 3 && {
            if case .array(let e, _) = g.valueType, e == .i8 { return true }
            return false
        }()
    }
    let hasAtomicDecl = module.functions.contains { $0.name.hasPrefix("air.atomic.") }
    let hasTGGlobal = module.globals.contains { $0.addressSpace == 3 }
    // i1 loads from constant buffer need typed ptrs — opaque ptr + i1 load crashes GPU JIT
    let hasI1Load = module.functions.contains { fn in
        !fn.isDeclaration && fn.basicBlocks.contains { bb in
            bb.instructions.contains { $0.opcode == .load && $0.type == .int(bits: 1) }
        }
    }
    let needsTypedPtrs = hasMMADecl || hasTGByteGlobal || hasAtomicDecl || hasTGGlobal || hasI1Load
    TypeTableWriter.emitOpaqueAsTyped = needsTypedPtrs
    guard needsTypedPtrs else { return }

    // Fix cmpxchg: pointer types in cmpxchg calls must be i32*, not float*.
    // emitOpaqueAsTyped converts opaquePointer → float* by default, but cmpxchg operates on i32.
    // We collect SSA names that feed cmpxchg (alloca, inttoptr) and convert those + call operands.
    let hasAtomicCalls = module.functions.contains { $0.name.contains(".cmpxchg.") || $0.name.hasPrefix("air.atomic.") }
    if hasAtomicCalls {
        let i32Ptr = { (a: Int) in IRType.pointer(pointee: .int(bits: 32), addressSpace: a) }
        // Step 1: Fix atomic declarations — all opaque ptr params → i32*
        for fn in module.functions where fn.isDeclaration && (fn.name.contains(".cmpxchg.") || fn.name.hasPrefix("air.atomic.")) {
            for i in fn.parameterTypes.indices {
                if case .opaquePointer(let a) = fn.parameterTypes[i] {
                    fn.parameterTypes[i] = i32Ptr(a)
                }
            }
            fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)
        }
        // Step 2: Collect names of values feeding atomic ptr args
        // Also fix GEP instructions whose results feed atomics: change gepSourceType to i32
        for fn in module.functions where !fn.isDeclaration {
            var atomicPtrNames: Set<String> = []
            for bb in fn.basicBlocks {
                for inst in bb.instructions where inst.opcode == .alloca {
                    if let allocTy = inst.attributes.allocaType, allocTy == .int(bits: 32) {
                        atomicPtrNames.insert(inst.name)
                    }
                }
                for inst in bb.instructions where inst.opcode == .call {
                    guard let calleeOp = inst.operands.last,
                          case .value(let callee) = calleeOp,
                          (callee.name.contains(".cmpxchg.") || callee.name.hasPrefix("air.atomic.")) else { continue }
                    let numPtrArgs = callee.name.contains(".cmpxchg.") ? 2 : 1
                    for i in 0..<min(numPtrArgs, inst.operands.count - 1) {
                        if case .value(let v) = inst.operands[i] {
                            atomicPtrNames.insert(v.name)
                        }
                    }
                }
            }
            guard !atomicPtrNames.isEmpty else { continue }
            // Convert types: opaque→i32*, typed ptr→i32*, GEP source type→i32
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if atomicPtrNames.contains(inst.name) {
                        // Fix GEP source type so result becomes i32* consistently
                        if inst.opcode == .getelementptr {
                            inst.attributes.gepSourceType = .int(bits: 32)
                        }
                        switch inst.type {
                        case .opaquePointer(let a):
                            inst.type = i32Ptr(a)
                        case .pointer(_, let a):
                            inst.type = i32Ptr(a)
                        default: break
                        }
                    }
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i],
                           atomicPtrNames.contains(v.name) {
                            switch v.type {
                            case .opaquePointer(let a):
                                v.type = i32Ptr(a)
                            case .pointer(_, let a):
                                v.type = i32Ptr(a)
                            default: break
                            }
                        }
                    }
                }
            }
            for name in atomicPtrNames {
                if let idx = fn.parameters.firstIndex(where: { $0.name == name }) {
                    switch fn.parameterTypes[idx] {
                    case .opaquePointer(let a), .pointer(_, let a):
                        fn.parameterTypes[idx] = i32Ptr(a)
                        fn.parameters[idx].type = i32Ptr(a)
                    default: break
                    }
                }
            }
            fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)
        }
        // Step 3: Fix callee operand types in call instructions to match declarations
        var atomicDeclTypes: [String: IRType] = [:]
        for fn in module.functions where fn.isDeclaration && (fn.name.contains(".cmpxchg.") || fn.name.hasPrefix("air.atomic.")) {
            atomicDeclTypes[fn.name] = fn.type
        }
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                for inst in bb.instructions where inst.opcode == .call {
                    if let lastOp = inst.operands.last,
                       case .value(let callee) = lastOp,
                       let declType = atomicDeclTypes[callee.name] {
                        callee.type = .pointer(pointee: declType, addressSpace: 0)
                    }
                }
            }
        }
    }

    for fn in module.functions where !fn.isDeclaration {
        // Map param name → index for quick lookup
        var paramIndices: [String: Int] = [:]
        for (i, p) in fn.parameters.enumerated() {
            paramIndices[p.name] = i
        }

        // For each opaque pointer param, find what type is loaded from it
        var paramPointeeTypes: [Int: IRType] = [:]
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                if inst.opcode == .load, let op = inst.operands.first,
                   case .value(let v) = op, let idx = paramIndices[v.name],
                   case .opaquePointer(_) = fn.parameterTypes[idx] {
                    // Metal has no i1 memory type — booleans are i8 in memory
                    paramPointeeTypes[idx] = (inst.type == .int(bits: 1)) ? .int(bits: 8) : inst.type
                }
                // Also check GEPs that use a param as base
                if inst.opcode == .getelementptr, let op = inst.operands.first,
                   case .value(let v) = op, let idx = paramIndices[v.name],
                   case .opaquePointer(_) = fn.parameterTypes[idx],
                   let srcTy = inst.attributes.gepSourceType {
                    // Metal has no i1 memory type — booleans are i8 in memory
                    paramPointeeTypes[idx] = (srcTy == .int(bits: 1)) ? .int(bits: 8) : srcTy
                }
            }
        }

        // Convert opaque pointer params to typed pointers
        var changed = false
        for i in fn.parameterTypes.indices {
            if case .opaquePointer(let addrSpace) = fn.parameterTypes[i] {
                let pointee = paramPointeeTypes[i] ?? .float32
                fn.parameterTypes[i] = .pointer(pointee: pointee, addressSpace: addrSpace)
                fn.parameters[i].type = fn.parameterTypes[i]
                changed = true
            }
        }
        if changed {
            fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)

            // Update operand types throughout the function body
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j],
                           let idx = paramIndices[v.name],
                           v.type != fn.parameterTypes[idx] {
                            inst.operands[j] = .value(IRValue(type: fn.parameterTypes[idx], name: v.name))
                        }
                    }
                }
            }
        }

        // Fix i1 loads from i8* params: change `load i1, i8* %p` to `load i8, i8* %p` + `trunc i8 to i1`
        // Metal has no i1 memory type; typed pointer i8* with load i1 = type mismatch → GPU JIT crash
        for bb in fn.basicBlocks {
            var insertions: [(Int, IRInstruction)] = []
            for (idx, inst) in bb.instructions.enumerated() {
                guard inst.opcode == .load, inst.type == .int(bits: 1) else { continue }
                // Change load type from i1 to i8
                let origName = inst.name
                let tmpName = origName + "_i8"
                inst.type = .int(bits: 8)
                inst.name = tmpName
                // Insert trunc i8 → i1 right after
                let trunc = IRInstruction(opcode: .trunc, type: .int(bits: 1), name: origName,
                                          operands: [.value(IRValue(type: .int(bits: 8), name: tmpName))])
                insertions.append((idx + 1, trunc))
            }
            for (offset, (idx, trunc)) in insertions.enumerated() {
                bb.instructions.insert(trunc, at: idx + offset)
            }
        }

        // Update GEP result types: if the base pointer is now typed,
        // the GEP result should also be typed (using gepSourceType).
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .getelementptr {
                if case .opaquePointer(let addrSpace) = inst.type,
                   let srcTy = inst.attributes.gepSourceType {
                    // Metal has no i1 memory type — remap to i8
                    let pointee = (srcTy == .int(bits: 1)) ? IRType.int(bits: 8) : srcTy
                    if srcTy == .int(bits: 1) {
                        inst.attributes.gepSourceType = .int(bits: 8)
                    }
                    let typedPtr = IRType.pointer(pointee: pointee, addressSpace: addrSpace)
                    inst.type = typedPtr
                    // Propagate to uses
                    let gepName = inst.name
                    if !gepName.isEmpty {
                        for bb2 in fn.basicBlocks {
                            for inst2 in bb2.instructions {
                                for j in inst2.operands.indices {
                                    if case .value(let v) = inst2.operands[j],
                                       v.name == gepName,
                                       case .opaquePointer(_) = v.type {
                                        inst2.operands[j] = .value(IRValue(type: typedPtr, name: v.name))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Update metadata function pointer types to match actual function types
    // (paramPointeeTypes may have changed fn.type from opaque → typed pointers)
    let fnTypeMap: [String: IRType] = Dictionary(
        uniqueKeysWithValues: module.functions.filter { !$0.isDeclaration }.map { ($0.name, $0.type) }
    )
    for md in module.metadataNodes {
        for i in md.operands.indices {
            if case .value(let ty, let name) = md.operands[i],
               let actualType = fnTypeMap[name] {
                let expectedPtrType = IRType.pointer(pointee: actualType, addressSpace: 0)
                if ty != expectedPtrType {
                    md.operands[i] = .value(expectedPtrType, name)
                }
            }
        }
    }

    // Build new typed function types for load/store
    let mmaLoadType  = IRType.function(ret: .vector(element: .float32, count: 64),
                                       params: [floatTGPtr, .vector(element: .int(bits: 64), count: 2),
                                                .vector(element: .int(bits: 64), count: 2),
                                                .vector(element: .int(bits: 64), count: 2)],
                                       isVarArg: false)
    let mmaStoreType = IRType.function(ret: .void,
                                       params: [.vector(element: .float32, count: 64), floatTGPtr,
                                                .vector(element: .int(bits: 64), count: 2),
                                                .vector(element: .int(bits: 64), count: 2),
                                                .vector(element: .int(bits: 64), count: 2)],
                                       isVarArg: false)

    // No kernel attrs (matching reference — metal-as doesn't emit min-legal-vector-width for this kernel)

    // Update callee operand types + call attributes matching metal compiler output:
    // tail call fast @mmaLoad(float addrspace(3)* nocapture readonly ...)
    // tail call      @mmaStore(vec, float addrspace(3)* nocapture writeonly ...)
    for fn in module.functions where !fn.isDeclaration {
        var hasMMA = false
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .call {
                guard let calleeOp = inst.operands.last,
                      case .value(let callee) = calleeOp else { continue }
                if callee.name == mmaLoadName {
                    hasMMA = true
                    inst.operands[inst.operands.count - 1] = .value(IRValue(type: mmaLoadType, name: callee.name))
                    // Set call-site paramattr to match ref (nocapture+readonly on ptr param)
                    if let loadFn = module.functions.first(where: { $0.name == mmaLoadName }),
                       let groupIdx = loadFn.attributeGroupIndex {
                        inst.attributes.funcAttributes = [groupIdx]
                    } else {
                        inst.attributes.funcAttributes = []
                    }
                } else if callee.name == mmaStoreName {
                    hasMMA = true
                    inst.operands[inst.operands.count - 1] = .value(IRValue(type: mmaStoreType, name: callee.name))
                    if let storeFn = module.functions.first(where: { $0.name == mmaStoreName }),
                       let groupIdx = storeFn.attributeGroupIndex {
                        inst.attributes.funcAttributes = [groupIdx]
                    } else {
                        inst.attributes.funcAttributes = []
                    }
                } else if callee.name.hasPrefix(mmaMulName) {
                    hasMMA = true
                    inst.attributes.funcAttributes = []
                }
            }
        }
        // No kernel attrs — matching reference (metal-as doesn't emit kernel attrs for mma kernels)
    }
}

// MARK: - Transform 3: Threadgroup global GEP rewrite

private func transformTGGlobalGEPs(module: IRModule) {
    // Two kinds of addrspace(3) globals:
    //   1. [N x i8] byte-addressed scratch (e.g. @global_smem) → convert to kernel param
    //   2. [N x float] MMA buffers (e.g. @__tg_dot_*) → preamble GEP for typed pointer

    var byteGlobals: [IRGlobal] = []       // → preamble GEP (keep as globals)
    var mmaGlobals: [(name: String, arrayType: IRType, count: Int)] = []  // → preamble GEP

    for g in module.globals where g.addressSpace == 3 {
        if case .array(let elemTy, let n) = g.valueType {
            if elemTy == .i8 {
                byteGlobals.append(g)
            } else {
                mmaGlobals.append((name: g.name, arrayType: g.valueType, count: n))
            }
        }
    }

    // Flatten constant GEP offsets for ALL TG globals. The bitcode writer
    // ignores constantGEPByteOffset, so inline constant GEP expressions like
    // `getelementptr i8, @__tg_cvt_0, 4` must be expanded into explicit GEP
    // instructions to preserve the byte offset.
    // For MMA globals (non-i8 element type), convert byte offset to element offset.
    do {
        var tgGlobalInfo: [String: (elemType: IRType, elemBytes: Int64)] = [:]
        for g in module.globals where g.addressSpace == 3 {
            if case .array(let elemTy, _) = g.valueType {
                let bytes: Int64
                switch elemTy {
                case .i8: bytes = 1
                case .i16, .float16, .bfloat16: bytes = 2
                case .i32, .float32: bytes = 4
                case .i64, .float64: bytes = 8
                default: bytes = 1 // fallback to i8
                }
                tgGlobalInfo[g.name] = (elemType: elemTy, elemBytes: bytes)
            }
        }
        var ctr = 5000
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                var newInsts: [IRInstruction] = []
                for inst in bb.instructions {
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i], v.constantGEPByteOffset != 0,
                           let info = tgGlobalInfo[v.name] {
                            let ssaName = "__cgep_\(ctr)"
                            ctr += 1
                            // Convert byte offset to element offset for non-i8 globals
                            let gepSourceType: IRType
                            let gepOffset: Int64
                            if info.elemBytes > 1 && v.constantGEPByteOffset % info.elemBytes == 0 {
                                gepSourceType = info.elemType
                                gepOffset = v.constantGEPByteOffset / info.elemBytes
                            } else {
                                gepSourceType = .i8
                                gepOffset = v.constantGEPByteOffset
                            }
                            let gepInst = IRInstruction(
                                opcode: .getelementptr,
                                type: .opaquePointer(addressSpace: 3),
                                name: ssaName,
                                operands: [
                                    .value(IRValue(type: v.type, name: v.name)),
                                    .constant(.integer(.i64, gepOffset)),
                                ],
                                attributes: {
                                    var a = IRInstruction.InstructionAttributes()
                                    a.inBounds = true
                                    a.gepSourceType = gepSourceType
                                    return a
                                }()
                            )
                            newInsts.append(gepInst)
                            inst.operands[i] = .value(IRValue(type: .opaquePointer(addressSpace: 3), name: ssaName))
                        }
                    }
                    newInsts.append(inst)
                }
                bb.instructions = newInsts
            }
        }
    }

    // --- Part 1: Preamble GEP for [N x i8] globals (keep as globals, driver allocates TG memory) ---
    if !byteGlobals.isEmpty {
        let i8TGPtr = IRType.pointer(pointee: .i8, addressSpace: 3)

        // Note: constant GEP flattening already done above for all TG globals.
        let byteGlobalNames = Set(byteGlobals.map { $0.name })

        for fn in module.functions where !fn.isDeclaration {
            guard let entryBB = fn.basicBlocks.first else { continue }

            var usedByteGlobals: [IRGlobal] = []
            for g in byteGlobals {
                let used = fn.basicBlocks.contains { bb in
                    bb.instructions.contains { inst in
                        inst.operands.contains { op in
                            if case .value(let v) = op, v.name == g.name { return true }
                            return false
                        }
                    }
                }
                if used { usedByteGlobals.append(g) }
            }
            guard !usedByteGlobals.isEmpty else { continue }

            var preambleInstrs: [IRInstruction] = []
            var baseSSAs: [String: String] = [:]

            for g in usedByteGlobals {
                let ssaName = "__base_\(g.name)"
                baseSSAs[g.name] = ssaName
                let baseInst = IRInstruction(
                    opcode: .getelementptr,
                    type: i8TGPtr,
                    name: ssaName,
                    operands: [
                        .value(IRValue(type: .pointer(pointee: g.valueType, addressSpace: 3),
                                       name: g.name)),
                        .constant(.integer(.i64, 0)),
                        .constant(.integer(.i64, 0)),
                    ],
                    attributes: {
                        var a = IRInstruction.InstructionAttributes()
                        a.inBounds = true
                        a.gepSourceType = g.valueType
                        return a
                    }()
                )
                preambleInstrs.append(baseInst)
            }

            if !preambleInstrs.isEmpty {
                entryBB.instructions = preambleInstrs + entryBB.instructions
            }

            let preambleNames = Set(preambleInstrs.map { $0.name })

            // Replace uses of byte globals with preamble SSA (typed i8*)
            // For GEPs with non-i8 source type, convert index to byte offset
            var gepScaleCounter = 6000
            for bb in fn.basicBlocks {
                var extraInsts: [(Int, IRInstruction)] = []
                for (instIdx, inst) in bb.instructions.enumerated() where !preambleNames.contains(inst.name) {
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i],
                           let ssaName = baseSSAs[v.name] {
                            if inst.opcode == .getelementptr && i == 0 {
                                if case .array(_, _) = inst.attributes.gepSourceType {
                                    continue
                                }
                                // Non-i8 GEP source type: scale index by element size
                                if let srcTy = inst.attributes.gepSourceType, srcTy != .i8 {
                                    let elemSize: Int
                                    switch srcTy {
                                    case .int(let bits): elemSize = bits / 8
                                    case .float32: elemSize = 4
                                    case .float64: elemSize = 8
                                    case .float16, .bfloat16: elemSize = 2
                                    default: elemSize = 4
                                    }
                                    if elemSize > 1, inst.operands.count > 1 {
                                        // Multiply index by elemSize: %scaled = mul i32 %idx, elemSize
                                        let scaledName = "__gep_scale_\(gepScaleCounter)"
                                        gepScaleCounter += 1
                                        let mulInst = IRInstruction(
                                            opcode: .mul,
                                            type: .int(bits: 32),
                                            name: scaledName,
                                            operands: [inst.operands[1], .constant(.integer(.int(bits: 32), Int64(elemSize)))]
                                        )
                                        extraInsts.append((instIdx, mulInst))
                                        inst.operands[1] = .value(IRValue(type: .int(bits: 32), name: scaledName))
                                    }
                                    inst.attributes.gepSourceType = .i8
                                }
                            }
                            inst.operands[i] = .value(IRValue(type: i8TGPtr, name: ssaName))
                        }
                    }
                }
                // Insert scale instructions before their GEPs
                for (offset, (instIdx, mulInst)) in extraInsts.enumerated() {
                    bb.instructions.insert(mulInst, at: instIdx + offset)
                }
            }

            // Fix GEP result types from opaquePointer(3) to typed i8*(3)
            // Only for GEPs whose base is a byte-global preamble SSA or another byte-global GEP
            let byteSSANames = Set(baseSSAs.values)
            var gepResultNames: Set<String> = []
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if inst.opcode == .getelementptr,
                       case .opaquePointer(3) = inst.type,
                       case .value(let baseVal) = inst.operands.first,
                       byteSSANames.contains(baseVal.name) || gepResultNames.contains(baseVal.name) {
                        inst.type = i8TGPtr
                        gepResultNames.insert(inst.name)
                    }
                }
            }
            // Propagate typed pointer to downstream uses
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i],
                           gepResultNames.contains(v.name),
                           case .opaquePointer(3) = v.type {
                            inst.operands[i] = .value(IRValue(type: i8TGPtr, name: v.name))
                        }
                    }
                }
            }

            // Insert bitcasts: store/load of non-i8 types through i8*(3)
            for bb in fn.basicBlocks {
                var newInsts: [IRInstruction] = []
                var castMap: [String: (name: String, type: IRType)] = [:]
                for inst in bb.instructions {
                    if inst.opcode == .store,
                       inst.operands.count >= 2,
                       case .value(let ptrVal) = inst.operands[1],
                       gepResultNames.contains(ptrVal.name) || byteSSANames.contains(ptrVal.name) {
                        let storedType: IRType
                        switch inst.operands[0] {
                        case .value(let v): storedType = v.type
                        case .constant(let c): storedType = c.type
                        default: storedType = .i8
                        }
                        if storedType != .i8 {
                            let key = ptrVal.name
                            let cast: (name: String, type: IRType)
                            if let existing = castMap[key], existing.type == IRType.pointer(pointee: storedType, addressSpace: 3) {
                                cast = existing
                            } else {
                                let castName = "__bc_\(ptrVal.name)_\(newInsts.count)"
                                let castType = IRType.pointer(pointee: storedType, addressSpace: 3)
                                let bitcast = IRInstruction(
                                    opcode: .bitcast,
                                    type: castType,
                                    name: castName,
                                    operands: [.value(IRValue(type: i8TGPtr, name: ptrVal.name))],
                                    attributes: IRInstruction.InstructionAttributes()
                                )
                                newInsts.append(bitcast)
                                cast = (castName, castType)
                                castMap[key] = cast
                            }
                            inst.operands[1] = .value(IRValue(type: cast.type, name: cast.name))
                        }
                    }
                    if inst.opcode == .load,
                       inst.operands.count >= 1,
                       case .value(let ptrVal) = inst.operands[0],
                       gepResultNames.contains(ptrVal.name) || byteSSANames.contains(ptrVal.name) {
                        let loadedType = inst.type
                        if loadedType != .i8 {
                            let key = ptrVal.name
                            let cast: (name: String, type: IRType)
                            if let existing = castMap[key], existing.type == IRType.pointer(pointee: loadedType, addressSpace: 3) {
                                cast = existing
                            } else {
                                let castName = "__bc_\(ptrVal.name)_\(newInsts.count)"
                                let castType = IRType.pointer(pointee: loadedType, addressSpace: 3)
                                let bitcast = IRInstruction(
                                    opcode: .bitcast,
                                    type: castType,
                                    name: castName,
                                    operands: [.value(IRValue(type: i8TGPtr, name: ptrVal.name))],
                                    attributes: IRInstruction.InstructionAttributes()
                                )
                                newInsts.append(bitcast)
                                cast = (castName, castType)
                                castMap[key] = cast
                            }
                            inst.operands[0] = .value(IRValue(type: cast.type, name: cast.name))
                        }
                    }
                    // Insert bitcast for atomic calls on i8*(3) TG pointers
                    if inst.opcode == .call,
                       inst.operands.count >= 2,
                       case .value(let callee) = inst.operands.last,
                       callee.name.hasPrefix("air.atomic."),
                       case .value(let ptrVal) = inst.operands[0],
                       gepResultNames.contains(ptrVal.name) || byteSSANames.contains(ptrVal.name) {
                        let valType = inst.type // e.g. i32
                        let key = "\(ptrVal.name)_atomic"
                        let cast: (name: String, type: IRType)
                        if let existing = castMap[key], existing.type == IRType.pointer(pointee: valType, addressSpace: 3) {
                            cast = existing
                        } else {
                            let castName = "__bc_\(ptrVal.name)_\(newInsts.count)"
                            let castType = IRType.pointer(pointee: valType, addressSpace: 3)
                            let bitcast = IRInstruction(
                                opcode: .bitcast,
                                type: castType,
                                name: castName,
                                operands: [.value(IRValue(type: i8TGPtr, name: ptrVal.name))],
                                attributes: IRInstruction.InstructionAttributes()
                            )
                            newInsts.append(bitcast)
                            cast = (castName, castType)
                            castMap[key] = cast
                        }
                        inst.operands[0] = .value(IRValue(type: cast.type, name: cast.name))
                    }
                    newInsts.append(inst)
                }
                bb.instructions = newInsts
            }
        }
    }

    // --- Part 1b: Device pointer bitcasts ---
    // When emitOpaqueAsTyped is on, remaining opaquePointer(1) params become float*.
    // Stores/loads of non-float types through these need bitcast to the correct type.
    for fn in module.functions where !fn.isDeclaration {
        var nextSSA = fn.basicBlocks.flatMap(\.instructions).compactMap {
            if let n = Int($0.name) { return n } else { return nil }
        }.max().map { $0 + 1 } ?? 1000

        for bb in fn.basicBlocks {
            var newInsts: [IRInstruction] = []
            for inst in bb.instructions {
                if inst.opcode == .store, inst.operands.count >= 2 {
                    if case .value(let ptrVal) = inst.operands[1] {
                        let pointee: IRType
                        let addrSpace: Int
                        switch ptrVal.type {
                        case .pointer(let p, let a): pointee = p; addrSpace = a
                        case .opaquePointer(let a): pointee = .float32; addrSpace = a
                        default: pointee = .void; addrSpace = 0
                        }
                        // Only fix device pointers (addrspace 1) — TG (addrspace 3) handled above
                        guard addrSpace == 1 else { newInsts.append(inst); continue }
                        let valTy: IRType
                        switch inst.operands[0] {
                        case .value(let v): valTy = v.type
                        case .constant(let c): valTy = c.type
                        default: valTy = .void
                        }
                        if valTy != .void && pointee != .void && valTy != pointee {
                            let castName = "\(nextSSA)"
                            nextSSA += 1
                            let castTy = IRType.pointer(pointee: valTy, addressSpace: addrSpace)
                            let castInst = IRInstruction(
                                opcode: .bitcast, type: castTy, name: castName,
                                operands: [.value(ptrVal)])
                            newInsts.append(castInst)
                            inst.operands[1] = .value(IRValue(type: castTy, name: castName))
                        }
                    }
                } else if inst.opcode == .load, inst.operands.count >= 1 {
                    if case .value(let ptrVal) = inst.operands[0] {
                        let pointee: IRType
                        let addrSpace: Int
                        switch ptrVal.type {
                        case .pointer(let p, let a): pointee = p; addrSpace = a
                        case .opaquePointer(let a): pointee = .float32; addrSpace = a
                        default: pointee = .void; addrSpace = 0
                        }
                        guard addrSpace == 1 else { newInsts.append(inst); continue }
                        if inst.type != .void && pointee != .void && inst.type != pointee {
                            let castName = "\(nextSSA)"
                            nextSSA += 1
                            let castTy = IRType.pointer(pointee: inst.type, addressSpace: addrSpace)
                            let castInst = IRInstruction(
                                opcode: .bitcast, type: castTy, name: castName,
                                operands: [.value(ptrVal)])
                            newInsts.append(castInst)
                            inst.operands[0] = .value(IRValue(type: castTy, name: castName))
                        }
                    }
                }
                newInsts.append(inst)
            }
            bb.instructions = newInsts
        }
    }

    // --- Part 2: Preamble GEP for MMA globals (keep as globals, add typed pointer) ---
    guard !mmaGlobals.isEmpty else { return }

    let mmaGlobalSet = Set(mmaGlobals.map { $0.name })
    // Map global name → element type for typed pointer creation
    var mmaGlobalElemType: [String: IRType] = [:]
    for g in mmaGlobals {
        if case .array(let elemTy, _) = g.arrayType {
            mmaGlobalElemType[g.name] = elemTy
        } else {
            mmaGlobalElemType[g.name] = .float32 // fallback
        }
    }

    for fn in module.functions where !fn.isDeclaration {
        guard let entryBB = fn.basicBlocks.first else { continue }

        var needsPreamble: Set<String> = []
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                for op in inst.operands {
                    if case .value(let v) = op, mmaGlobalSet.contains(v.name) {
                        needsPreamble.insert(v.name)
                    }
                }
            }
        }

        var preambleInstrs: [IRInstruction] = []
        var baseSSAs: [String: String] = [:]
        var ssaTypeMap: [String: IRType] = [:]

        for g in mmaGlobals where needsPreamble.contains(g.name) {
            var elemTy = mmaGlobalElemType[g.name] ?? .float32
            // Metal GPU JIT doesn't support bfloat* typed pointers — use half* (same size)
            if elemTy == .bfloat16 { elemTy = .float16 }
            let typedTGPtr = IRType.pointer(pointee: elemTy, addressSpace: 3)
            let ssaName = "__base_\(g.name)"
            baseSSAs[g.name] = ssaName
            ssaTypeMap[ssaName] = typedTGPtr
            let baseInst = IRInstruction(
                opcode: .getelementptr,
                type: typedTGPtr,
                name: ssaName,
                operands: [
                    .value(IRValue(type: .pointer(pointee: g.arrayType, addressSpace: 3),
                                   name: g.name)),
                    .constant(.integer(.i64, 0)),
                    .constant(.integer(.i64, 0)),
                ],
                attributes: {
                    var a = IRInstruction.InstructionAttributes()
                    a.inBounds = true
                    a.gepSourceType = g.arrayType
                    return a
                }()
            )
            preambleInstrs.append(baseInst)
        }

        if !preambleInstrs.isEmpty {
            entryBB.instructions = preambleInstrs + entryBB.instructions
        }

        let preambleNames = Set(preambleInstrs.map { $0.name })

        // Replace uses of TG globals with preamble SSA (typed ptr)
        for bb in fn.basicBlocks {
            for inst in bb.instructions where !preambleNames.contains(inst.name) {
                for i in inst.operands.indices {
                    if case .value(let v) = inst.operands[i],
                       mmaGlobalSet.contains(v.name),
                       let ssaName = baseSSAs[v.name],
                       let typedPtr = ssaTypeMap[ssaName] {
                        if inst.opcode == .getelementptr && i == 0 {
                            if case .array(_, _) = inst.attributes.gepSourceType {
                                continue
                            }
                        }
                        inst.operands[i] = .value(IRValue(type: typedPtr, name: ssaName))
                    }
                }
            }
        }

        // Fix stale operand types
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                for i in inst.operands.indices {
                    if case .value(let v) = inst.operands[i],
                       let newType = ssaTypeMap[v.name], v.type != newType {
                        inst.operands[i] = .value(IRValue(type: newType, name: v.name))
                    }
                }
            }
        }

        // Fix GEP result types from opaquePointer(3) to typed ptr(3)
        // GEPs whose base operand is a typed preamble SSA should also produce typed pointers
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                if inst.opcode == .getelementptr,
                   case .opaquePointer(3) = inst.type {
                    // Find the element type from the base operand
                    if let baseOp = inst.operands.first,
                       case .value(let v) = baseOp,
                       let typedPtr = ssaTypeMap[v.name],
                       case .pointer(let pointee, 3) = typedPtr {
                        let resultType = IRType.pointer(pointee: pointee, addressSpace: 3)
                        inst.type = resultType
                        ssaTypeMap[inst.name] = resultType
                    }
                }
            }
        }

        // Fix select result types: if a select produces opaquePointer(3) but
        // its operands are typed TG pointers, set result to the same typed ptr
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                if inst.opcode == .select,
                   case .opaquePointer(3) = inst.type {
                    // Check true/false operands (indices 1 and 2) for typed TG ptr
                    for opIdx in 1...min(2, inst.operands.count - 1) {
                        if case .value(let v) = inst.operands[opIdx],
                           case .pointer(_, 3) = v.type {
                            inst.type = v.type
                            ssaTypeMap[inst.name] = v.type
                            break
                        }
                    }
                }
            }
        }

        // Propagate typed pointer to downstream uses of fixed GEPs/selects
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                for i in inst.operands.indices {
                    if case .value(let v) = inst.operands[i],
                       let newType = ssaTypeMap[v.name],
                       case .opaquePointer(3) = v.type {
                        inst.operands[i] = .value(IRValue(type: newType, name: v.name))
                    }
                }
            }
        }

    }
}

// MARK: - Transform: Scalar store thread-0 guard
//
// Metal dispatches multiple SIMD groups per threadgroup. In purely scalar kernels
// (no thread_position_in_threadgroup usage), all threads share the same program_id
// and would load/store the same address. SIMD groups execute out of order, so
// a load-compute-store pattern races: one SIMD may read the already-written value.
// Fix: if the kernel has no tid usage but stores to device memory, wrap the body
// in a `tid.x == 0` early-return guard so only one thread executes.

private func transformScalarStoreGuard(module: IRModule) {
    // Skip if IR is pre-lowered (already has !air.kernel metadata)
    if module.namedMetadata.contains(where: { $0.name == "air.kernel" }) {
        return
    }

    let tidTGName = "air.thread_position_in_threadgroup"

    for fn in module.functions where !fn.isDeclaration {
        // Check if function uses per-thread indexing or only per-threadgroup indexing
        var hasTidTG = false     // uses thread_position_in_threadgroup
        var hasTidGlobal = false // uses thread_position_in_grid (global per-thread)
        var hasDeviceStore = false  // non-atomic store to device memory
        var hasAtomicWrite = false  // atomic call (thread-safe, no guard needed)
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                if inst.opcode == .call,
                   let calleeOp = inst.operands.last,
                   case .value(let v) = calleeOp {
                    if v.name == tidTGName { hasTidTG = true }
                    if v.name == "air.thread_position_in_grid" { hasTidGlobal = true }
                    if v.name == "air.thread_index_in_simdgroup" { hasTidTG = true }
                    if v.name.hasPrefix("air.atomic.global.") { hasAtomicWrite = true }
                }
                if inst.opcode == .store, inst.operands.count >= 2,
                   case .value(let ptrVal) = inst.operands[1] {
                    // Check if storing to addrspace(1) — device memory
                    switch ptrVal.type {
                    case .pointer(_, let as1) where as1 == 1: hasDeviceStore = true
                    case .opaquePointer(let as1) where as1 == 1: hasDeviceStore = true
                    default: break
                    }
                }
            }
        }

        // Only guard if: uses only threadgroup_position (program_id), not any per-thread index,
        // and has device stores. Kernels using thread_position_in_grid or
        // thread_position_in_threadgroup already have per-thread addressing.
        // Only guard if: uses only threadgroup_position (program_id), not any per-thread index,
        // and has device writes (stores or atomics).
        guard !hasTidTG && !hasTidGlobal && (hasDeviceStore || hasAtomicWrite) else { continue }

        // Ensure air.thread_position_in_threadgroup is declared
        let tidTGDecl: IRFunction
        if let existing = module.functions.first(where: { $0.name == tidTGName && $0.isDeclaration }) {
            tidTGDecl = existing
        } else {
            let decl = IRFunction(
                name: tidTGName,
                returnType: .array(element: .int(bits: 32), count: 3),
                parameterTypes: [],
                isDeclaration: true
            )
            module.functions.insert(decl, at: 0)
            tidTGDecl = decl
        }
        _ = tidTGDecl // used via name reference below

        // The function should have a single entry block for simple scalar kernels.
        // We'll prepend guard instructions to the entry block and split.
        guard let entryBB = fn.basicBlocks.first else { continue }

        // Create new blocks: "guard" (entry) and "exit" (early return)
        let bodyBB = IRBasicBlock(name: "body", instructions: entryBB.instructions)
        let exitBB = IRBasicBlock(name: "exit", instructions: [
            IRInstruction(opcode: .ret)
        ])

        // Replace the entry block's instructions with the guard
        let tidCallResult = IRValue(type: .array(element: .int(bits: 32), count: 3), name: "guard_tid")
        let tidX = IRValue(type: .int(bits: 32), name: "guard_tid_x")
        let isThread0 = IRValue(type: .int(bits: 1), name: "guard_is_t0")

        let tidCallFnVal = IRValue(type: .void, name: tidTGName)

        let oldEntryName = entryBB.name  // capture before rename (e.g. "2" for 2-param kernel)
        entryBB.name = "entry"
        entryBB.instructions = [
            // %guard_tid = call [3 x i32] @air.thread_position_in_threadgroup()
            IRInstruction(opcode: .call,
                          type: .array(element: .int(bits: 32), count: 3),
                          name: "guard_tid",
                          operands: [.value(tidCallFnVal)]),
            // %guard_tid_x = extractvalue [3 x i32] %guard_tid, 0
            IRInstruction(opcode: .extractValue,
                          type: .int(bits: 32),
                          name: "guard_tid_x",
                          operands: [.value(tidCallResult), .intLiteral(0)]),
            // %guard_is_t0 = icmp eq i32 %guard_tid_x, 0
            {
                let inst = IRInstruction(opcode: .icmp,
                              type: .int(bits: 1),
                              name: "guard_is_t0",
                              operands: [.value(tidX), .constant(.integer(.int(bits: 32), 0))])
                inst.attributes.predicate = 32 // eq
                return inst
            }(),
            // br i1 %guard_is_t0, label %body, label %exit
            IRInstruction(opcode: .br,
                          operands: [.value(isThread0), .basicBlock(bodyBB), .basicBlock(exitBB)]),
        ]

        // Make sure body block's terminator is ret void → change to br %exit if it's ret
        // Actually, keep ret as-is — body already has ret void at the end.
        // No, we need to keep the body's ret. It returns normally.
        // The exit block also has ret. So body's ret is fine.

        // Update phi nodes: references to old entry block must point to bodyBB
        for bb in fn.basicBlocks[1...] {  // skip entry (now guard)
            for inst in bb.instructions where inst.opcode == .phi {
                for i in inst.operands.indices {
                    if case .basicBlock(let target) = inst.operands[i],
                       target.name == oldEntryName {
                        inst.operands[i] = .basicBlock(bodyBB)
                    }
                }
            }
        }

        // Replace function's basic blocks: entry guard, then ALL original blocks, then exit
        var allBlocks = [entryBB, bodyBB]
        // Append any additional basic blocks from the original function (branches, loops)
        for i in 1..<fn.basicBlocks.count {
            allBlocks.append(fn.basicBlocks[i])
        }
        allBlocks.append(exitBB)
        fn.basicBlocks = allBlocks
    }
}

// MARK: - Transform 4: Air system-value lowering

private let airTidName      = "air.thread_position_in_grid"
private let airTidTGName    = "air.thread_position_in_threadgroup"
private let airPidName      = "air.threadgroup_position_in_grid"
private let airSimdlaneName = "air.thread_index_in_simdgroup"
private let airNumProgramsName = "air.threadgroups_per_grid"

private func transformAirSystemValues(module: IRModule) {
    // If !air.kernel already exists the IR was pre-lowered — skip transform.
    if module.namedMetadata.contains(where: { $0.name == "air.kernel" }) {
        ensureVersionMetadata(module: module)
        return
    }

    // Strip intrinsic declarations; we'll re-declare nothing (Metal provides them as params)
    module.functions.removeAll { fn in
        fn.isDeclaration && (
            fn.name == airTidName ||
            fn.name == airTidTGName ||
            fn.name == airPidName ||
            fn.name == airSimdlaneName ||
            fn.name == airNumProgramsName
        )
    }

    var kernelNames: [String] = []

    // Build set of function names that are called by other functions (= device functions).
    // These should NOT get kernel transforms (Pass 5b scalar rewrite, Pass 6 air.kernel metadata).
    let declaredFnNames = Set(module.functions.filter { !$0.isDeclaration }.map { $0.name })
    var calledFnNames: Set<String> = []
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .call {
                if let calleeOp = inst.operands.last, case .value(let v) = calleeOp,
                   declaredFnNames.contains(v.name) {
                    calledFnNames.insert(v.name)
                }
            }
        }
    }

    for fn in module.functions where !fn.isDeclaration {
        // Scan for which system values are used
        var hasTid = false
        var hasTidTG = false
        var hasPid = false
        var hasSimdlane = false
        var hasNumPrograms = false
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .call {
                if let calleeOp = inst.operands.last, case .value(let v) = calleeOp {
                    if v.name == airTidName      { hasTid = true }
                    if v.name == airTidTGName    { hasTidTG = true }
                    if v.name == airPidName      { hasPid = true }
                    if v.name == airSimdlaneName { hasSimdlane = true }
                    if v.name == airNumProgramsName { hasNumPrograms = true }
                }
            }
        }
        // No system value calls — kernel already has explicit params, skip param rewrites
        let needsParamTransform = hasTid || hasTidTG || hasPid || hasSimdlane || hasNumPrograms

        let origParamCount = fn.parameterTypes.count

        if needsParamTransform {
            // Build replacement map: old SSA name → new param name
            // Pass 1: find struct SSAs for tid/pid calls and their extractvalue users
            var renameMap: [String: String] = [:]  // old SSA → new name (e.g. "tid_x")
            var callSSAs: Set<String> = []         // SSA names that are the call results

            for bb in fn.basicBlocks {
                for inst in bb.instructions where inst.opcode == .call {
                    guard let calleeOp = inst.operands.last,
                          case .value(let v) = calleeOp else { continue }
                    if v.name == airTidName || v.name == airTidTGName || v.name == airPidName || v.name == airNumProgramsName {
                        callSSAs.insert(inst.name)
                    }
                    if v.name == airSimdlaneName {
                        renameMap[inst.name] = "simdlane"
                    }
                }
            }

            // Pass 2: find extractvalue instructions referencing those struct SSAs
            var toRemove: Set<ObjectIdentifier> = []
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if inst.opcode == .extractValue {
                        if let baseOp = inst.operands.first,
                           case .value(let baseVal) = baseOp,
                           callSSAs.contains(baseVal.name) {
                            let axis = inst.operands.count >= 2 ? axisIndex(inst.operands[1]) : 0
                            var callKind = "tid" // default: thread_position_in_grid
                            outerLoop: for bb2 in fn.basicBlocks {
                                for inst2 in bb2.instructions where inst2.opcode == .call {
                                    if inst2.name == baseVal.name,
                                       let cop = inst2.operands.last,
                                       case .value(let cv) = cop {
                                        if cv.name == airPidName {
                                            callKind = "pid"
                                            break outerLoop
                                        } else if cv.name == airTidTGName {
                                            callKind = "tidtg"
                                            break outerLoop
                                        } else if cv.name == airNumProgramsName {
                                            callKind = "numprog"
                                            break outerLoop
                                        }
                                    }
                                }
                            }
                            let prefix = callKind
                            let dimName = ["x", "y", "z"][min(axis, 2)]
                            renameMap[inst.name] = "\(prefix)_\(dimName)"
                            toRemove.insert(ObjectIdentifier(inst))
                        }
                    }
                    if inst.opcode == .call,
                       let cop = inst.operands.last,
                       case .value(let cv) = cop,
                       cv.name == airTidName || cv.name == airTidTGName || cv.name == airPidName || cv.name == airSimdlaneName || cv.name == airNumProgramsName {
                        toRemove.insert(ObjectIdentifier(inst))
                    }
                }
            }

            // Pass 3: rename all uses of old SSAs to new param names
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i],
                           let newName = renameMap[v.name] {
                            inst.operands[i] = .value(IRValue(type: .i32, name: newName))
                        }
                    }
                }
            }

            // Pass 4: remove the call/extractvalue instructions
            for bb in fn.basicBlocks {
                bb.instructions.removeAll { toRemove.contains(ObjectIdentifier($0)) }
            }

            // Pass 5: append new parameters (uint3 vectors for pid/tid, i32 for simdlane)
            // and add extractelement preamble instructions to the entry block.
            var newParamTypes: [IRType] = []
            var newParamNames: [String] = []
            let uint3Type = IRType.vector(element: .i32, count: 3)

            if hasPid {
                newParamTypes.append(uint3Type)
                newParamNames.append("pid")
            }
            if hasTid {
                newParamTypes.append(uint3Type)
                newParamNames.append("tid")
            }
            if hasTidTG {
                newParamTypes.append(uint3Type)
                newParamNames.append("tidtg")
            }
            if hasSimdlane {
                newParamTypes.append(.i32)
                newParamNames.append("simdlane")
            }
            if hasNumPrograms {
                newParamTypes.append(uint3Type)
                newParamNames.append("numprog")
            }

            fn.parameterTypes.append(contentsOf: newParamTypes)
            fn.parameterNames.append(contentsOf: newParamNames)
            fn.parameterAttributes.append(contentsOf: newParamTypes.map { _ in [] })
            fn.parameterStringAttributes.append(contentsOf: newParamTypes.map { _ in [:] })
            fn.parameters.append(contentsOf: zip(newParamTypes, newParamNames).map { t, n in
                IRValue(type: t, name: n)
            })
            fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)

            // Add extractelement instructions at the start of the entry block
            var preamble: [IRInstruction] = []
            if hasPid {
                for (i, dim) in ["x", "y", "z"].enumerated() {
                    preamble.append(IRInstruction(
                        opcode: .extractElement,
                        type: .i32,
                        name: "pid_\(dim)",
                        operands: [
                            .value(IRValue(type: uint3Type, name: "pid")),
                            .constant(.integer(IRType.i32, Int64(i))),
                        ]
                    ))
                }
            }
            if hasTid {
                for (i, dim) in ["x", "y", "z"].enumerated() {
                    preamble.append(IRInstruction(
                        opcode: .extractElement,
                        type: .i32,
                        name: "tid_\(dim)",
                        operands: [
                            .value(IRValue(type: uint3Type, name: "tid")),
                            .constant(.integer(IRType.i32, Int64(i))),
                        ]
                    ))
                }
            }
            if hasTidTG {
                for (i, dim) in ["x", "y", "z"].enumerated() {
                    preamble.append(IRInstruction(
                        opcode: .extractElement,
                        type: .i32,
                        name: "tidtg_\(dim)",
                        operands: [
                            .value(IRValue(type: uint3Type, name: "tidtg")),
                            .constant(.integer(IRType.i32, Int64(i))),
                        ]
                    ))
                }
            }
            if hasNumPrograms {
                for (i, dim) in ["x", "y", "z"].enumerated() {
                    preamble.append(IRInstruction(
                        opcode: .extractElement,
                        type: .i32,
                        name: "numprog_\(dim)",
                        operands: [
                            .value(IRValue(type: uint3Type, name: "numprog")),
                            .constant(.integer(IRType.i32, Int64(i))),
                        ]
                    ))
                }
            }
            if !preamble.isEmpty, let entryBB = fn.basicBlocks.first {
                entryBB.instructions.insert(contentsOf: preamble, at: 0)
            }
        }

        // Device functions (called by other functions) don't get kernel transforms.
        // They keep their original parameter types and don't get !air.kernel metadata.
        let isDeviceFunction = calledFnNames.contains(fn.name)
        guard !isDeviceFunction else { continue }

        // Pass 5b: rewrite scalar (non-pointer) params to constant buffer pointers.
        // Metal requires all kernel params to be buffers or system values. Scalar
        // params (float, i32) from Triton are passed via setBytes which maps to
        // addrspace(2) constant buffers. Rewrite: float %x → ptr addrspace(2) %x,
        // and insert a load at the start of the entry block.
        do {
            var scalarPreamble: [IRInstruction] = []
            for i in 0..<origParamCount {
                let paramType = fn.parameterTypes[i]
                let scalarType: IRType
                switch paramType {
                case .float32: scalarType = .float32
                case .i32:     scalarType = .i32
                default: continue
                }
                let oldName = fn.parameterNames[i]
                let loadName = oldName.isEmpty ? "scalar_\(i)" : oldName
                let ptrName = "\(loadName)_ptr"
                let ptrType = IRType.pointer(pointee: scalarType, addressSpace: 2)
                fn.parameterTypes[i] = ptrType
                fn.parameterNames[i] = ptrName
                let loadInst = IRInstruction(
                    opcode: .load, type: scalarType, name: loadName,
                    operands: [.value(IRValue(type: ptrType, name: ptrName))]
                )
                loadInst.attributes.alignment = 4
                scalarPreamble.append(loadInst)
            }
            if !scalarPreamble.isEmpty, let entryBB = fn.basicBlocks.first {
                entryBB.instructions.insert(contentsOf: scalarPreamble, at: 0)
                // Update fn.type and fn.parameters to match modified parameterTypes
                fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)
                for i in 0..<fn.parameters.count {
                    fn.parameters[i] = IRValue(type: fn.parameterTypes[i], name: fn.parameterNames[i])
                }
            }
        }

        // Pass 6: emit !air.kernel metadata for this function
        let allTypes = fn.parameterTypes
        let allNames = fn.parameterNames

        // Compute param indices for pid/tid/tidtg/simdlane (1 param each)
        var sysIdx = origParamCount
        let pidIdx = hasPid ? { let i = sysIdx; sysIdx += 1; return i }() : -1
        let tidIdx = hasTid ? { let i = sysIdx; sysIdx += 1; return i }() : -1
        let tidtgIdx = hasTidTG ? { let i = sysIdx; sysIdx += 1; return i }() : -1
        let slIdx  = hasSimdlane ? { let i = sysIdx; sysIdx += 1; return i }() : -1
        let npIdx  = hasNumPrograms ? { let i = sysIdx; sysIdx += 1; return i }() : -1

        var argNodeIndices: [Int] = []
        let baseNodeIdx = module.metadataNodes.count

        for (idx, (argType, argName)) in zip(allTypes, allNames).enumerated() {
            let nodeIdx = baseNodeIdx + idx
            let operands: [IRMetadataOperand]

            let isPid = hasPid && idx == pidIdx
            let isTid = hasTid && idx == tidIdx
            let isTidTG = hasTidTG && idx == tidtgIdx
            let isSL  = hasSimdlane && idx == slIdx
            let isNP  = hasNumPrograms && idx == npIdx

            if isSL {
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.thread_index_in_simdgroup"),
                    .string("air.arg_type_name"), .string("uint"),
                    .string("air.arg_name"), .string(argName),
                ]
            } else if isTidTG {
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.thread_position_in_threadgroup"),
                    .string("air.arg_type_name"), .string("uint3"),
                    .string("air.arg_name"), .string("tidtg"),
                ]
            } else if isTid {
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.thread_position_in_grid"),
                    .string("air.arg_type_name"), .string("uint3"),
                    .string("air.arg_name"), .string("tid"),
                ]
            } else if isNP {
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.threadgroups_per_grid"),
                    .string("air.arg_type_name"), .string("uint3"),
                    .string("air.arg_name"), .string("numprog"),
                ]
            } else if isPid {
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.threadgroup_position_in_grid"),
                    .string("air.arg_type_name"), .string("uint3"),
                    .string("air.arg_name"), .string("pid"),
                ]
            } else if isTGBufferType(argType) {
                // Threadgroup buffer — Metal pattern: air.buffer with address_space 3
                // location_index 0 is shared with device buffers but distinguished by addrspace
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.buffer"),
                    .string("air.location_index"), .constant(.i32, .integer(.i32, 0)), .constant(.i32, .integer(.i32, 1)),
                    .string("air.read_write"),
                    .string("air.address_space"), .constant(.i32, .integer(.i32, 3)),
                    .string("air.arg_type_size"), .constant(.i32, .integer(.i32, 1)),
                    .string("air.arg_type_align_size"), .constant(.i32, .integer(.i32, 1)),
                    .string("air.arg_type_name"), .string("char"),
                    .string("air.arg_name"), .string(argName),
                ]
            } else if isDeviceBufferType(argType) {
                // Device buffer — infer element type from usage
                let (typeSize, typeAlign, typeName) = inferBufferElementType(fn: fn, paramIdx: idx)
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.buffer"),
                    .string("air.location_index"), .constant(.i32, .integer(.i32, Int64(idx))), .constant(.i32, .integer(.i32, 1)),
                    .string("air.read_write"),
                    .string("air.address_space"), .constant(.i32, .integer(.i32, 1)),
                    .string("air.arg_type_size"), .constant(.i32, .integer(.i32, Int64(typeSize))),
                    .string("air.arg_type_align_size"), .constant(.i32, .integer(.i32, Int64(typeAlign))),
                    .string("air.arg_type_name"), .string(typeName),
                    .string("air.arg_name"), .string(argName.isEmpty ? "arg\(idx)" : argName),
                ]
            } else {
                // Scalar constant buffer (addrspace(2))
                operands = [
                    .constant(.i32, .integer(.i32, Int64(idx))),
                    .string("air.buffer"),
                    .string("air.buffer_size"), .constant(.i32, .integer(.i32, 4)),
                    .string("air.location_index"), .constant(.i32, .integer(.i32, Int64(idx))), .constant(.i32, .integer(.i32, 1)),
                    .string("air.read"),
                    .string("air.address_space"), .constant(.i32, .integer(.i32, 2)),
                    .string("air.arg_type_size"), .constant(.i32, .integer(.i32, 4)),
                    .string("air.arg_type_align_size"), .constant(.i32, .integer(.i32, 4)),
                    .string("air.arg_type_name"), .string("uint"),
                    .string("air.arg_name"), .string(argName.isEmpty ? "arg\(idx)" : argName),
                ]
            }
            module.metadataNodes.append(IRMetadataNode(index: nodeIdx, operands: operands))
            argNodeIndices.append(nodeIdx)
        }

        // !air.kernel entry: !{void(...)* @fnname, !empty, !arglist}
        let emptyNodeIdx  = baseNodeIdx + allTypes.count
        let argListNodeIdx = emptyNodeIdx + 1
        let kernelNodeIdx  = argListNodeIdx + 1

        module.metadataNodes.append(IRMetadataNode(index: emptyNodeIdx, operands: []))
        module.metadataNodes.append(IRMetadataNode(
            index: argListNodeIdx,
            operands: argNodeIndices.map { .metadata($0) }
        ))
        // Use typed function pointer in metadata: void(...)* @fnname
        // (opaque ptr → float* via emitOpaqueAsTyped confuses GPU JIT)
        let fnPtrType = IRType.pointer(pointee: fn.type, addressSpace: 0)
        module.metadataNodes.append(IRMetadataNode(
            index: kernelNodeIdx,
            operands: [
                .value(fnPtrType, fn.name),
                .metadata(emptyNodeIdx),
                .metadata(argListNodeIdx),
            ]
        ))

        kernelNames.append(fn.name)

        // Append to !air.kernel named metadata
        if let existingIdx = module.namedMetadata.firstIndex(where: { $0.name == "air.kernel" }) {
            module.namedMetadata[existingIdx].operands.append(kernelNodeIdx)
        } else {
            module.namedMetadata.append(IRNamedMetadata(name: "air.kernel", operands: [kernelNodeIdx]))
        }
    }

    // Ensure !air.version and !air.language_version exist
    ensureVersionMetadata(module: module)
}

// MARK: - Helpers

private func isDeviceBufferType(_ t: IRType) -> Bool {
    switch t {
    case .opaquePointer(addressSpace: 1): return true
    case .pointer(_, addressSpace: 1): return true
    default: return false
    }
}

private func isTGBufferType(_ t: IRType) -> Bool {
    switch t {
    case .opaquePointer(addressSpace: 3): return true
    case .pointer(_, addressSpace: 3): return true
    default: return false
    }
}

private func axisIndex(_ operand: IRInstruction.Operand) -> Int {
    switch operand {
    case .intLiteral(let v): return Int(v)
    case .constant(let c):
        if case .integer(_, let v) = c { return Int(v) }
        return 0
    default: return 0
    }
}

private func ensureVersionMetadata(module: IRModule) {
    // Add resource limit module flags that Metal GPU compiler expects.
    // These must be in !llvm.module.flags (not as individual named metadata),
    // matching what metal-as produces: !{i32 7, !"air.max_device_buffers", i32 31}
    // Add resource limit module flags. Append to existing !llvm.module.flags if present,
    // or create it. Skip if air resource limits are already there.
    let resourceFlags: [(String, Int32)] = [
        ("air.max_device_buffers",      31),
        ("air.max_constant_buffers",    31),
        ("air.max_threadgroup_buffers", 31),
        ("air.max_textures",            128),
        ("air.max_read_write_textures", 8),
        ("air.max_samplers",            16),
    ]
    // Check if resource flags already present (by scanning existing module flags nodes)
    let hasResourceFlags = module.namedMetadata
        .first(where: { $0.name == "llvm.module.flags" })
        .map { named -> Bool in
            named.operands.contains { nodeIdx in
                guard let node = module.metadataNodes.first(where: { $0.index == nodeIdx }) else { return false }
                return node.operands.contains { if case .string(let s) = $0 { return s == "air.max_device_buffers" }; return false }
            }
        } ?? false

    if !hasResourceFlags {
        var flagNodeIndices: [Int] = []
        for (name, val) in resourceFlags {
            let idx = module.metadataNodes.count
            module.metadataNodes.append(IRMetadataNode(index: idx, operands: [
                .constant(.i32, .integer(.i32, 7)),   // flag type 7 = module requirement
                .string(name),
                .constant(.i32, .integer(.i32, Int64(val))),
            ]))
            flagNodeIndices.append(idx)
        }
        if let existingIdx = module.namedMetadata.firstIndex(where: { $0.name == "llvm.module.flags" }) {
            module.namedMetadata[existingIdx].operands.append(contentsOf: flagNodeIndices)
        } else {
            module.namedMetadata.append(IRNamedMetadata(name: "llvm.module.flags", operands: flagNodeIndices))
        }
    }

    if !module.namedMetadata.contains(where: { $0.name == "air.version" }) {
        let idx = module.metadataNodes.count
        module.metadataNodes.append(IRMetadataNode(index: idx, operands: [
            .constant(.i32, .integer(.i32, 2)),
            .constant(.i32, .integer(.i32, 8)),
            .constant(.i32, .integer(.i32, 0)),
        ]))
        module.namedMetadata.append(IRNamedMetadata(name: "air.version", operands: [idx]))
    }
    if !module.namedMetadata.contains(where: { $0.name == "air.language_version" }) {
        let idx = module.metadataNodes.count
        module.metadataNodes.append(IRMetadataNode(index: idx, operands: [
            .string("Metal"),
            .constant(.i32, .integer(.i32, 3)),
            .constant(.i32, .integer(.i32, 2)),
            .constant(.i32, .integer(.i32, 0)),
        ]))
        module.namedMetadata.append(IRNamedMetadata(name: "air.language_version", operands: [idx]))
    }
}

// MARK: - Transform 5: Opaque pointer → typed i8* lowering

/// Metal AIR is LLVM 14-based and requires typed pointers.
/// Replace all opaquePointer(N) with pointer(i8, N) throughout the IR.
private func transformOpaqueToTypedPtrs(module: IRModule) {
    func lowerType(_ t: IRType) -> IRType {
        switch t {
        case .opaquePointer(let as_):
            return .pointer(pointee: .int(bits: 8), addressSpace: as_)
        case .pointer(let pointee, let as_):
            return .pointer(pointee: lowerType(pointee), addressSpace: as_)
        case .array(let elem, let n):
            return .array(element: lowerType(elem), count: n)
        case .vector(let elem, let n):
            return .vector(element: lowerType(elem), count: n)
        case .function(let ret, let params, let va):
            return .function(ret: lowerType(ret), params: params.map(lowerType), isVarArg: va)
        case .structure(let name, let elems, let packed):
            return .structure(name: name, elements: elems.map(lowerType), isPacked: packed)
        default:
            return t
        }
    }
    func lowerValue(_ v: IRValue) -> IRValue {
        let newType = lowerType(v.type)
        return newType == v.type ? v : IRValue(type: newType, name: v.name)
    }
    func lowerOperand(_ op: IRInstruction.Operand) -> IRInstruction.Operand {
        if case .value(let v) = op { return .value(lowerValue(v)) }
        return op
    }

    // Rewrite function signatures
    for fn in module.functions {
        fn.type = lowerType(fn.type)
        fn.returnType = lowerType(fn.returnType)
        fn.parameterTypes = fn.parameterTypes.map(lowerType)
        // Rewrite parameter values
        for i in fn.parameters.indices {
            fn.parameters[i] = lowerValue(fn.parameters[i])
        }
        // Rewrite instructions
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                inst.type = lowerType(inst.type)
                inst.operands = inst.operands.map(lowerOperand)
            }
        }
    }

    // Rewrite global types
    for g in module.globals {
        g.type = lowerType(g.type)
    }
}

// MARK: - Transform: atomicrmw → air.atomic.* calls

private func transformAtomicRMW(module: IRModule) {
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            for (instIdx, inst) in bb.instructions.enumerated() where inst.opcode == .atomicRMW {
                guard let atomicOp = inst.attributes.atomicOp,
                      inst.operands.count >= 2 else { continue }

                let ptrOp = inst.operands[0]
                var addrSpace = 1
                if case .value(let v) = ptrOp {
                    switch v.type {
                    case .opaquePointer(let a): addrSpace = a
                    case .pointer(_, let a): addrSpace = a
                    default: break
                    }
                }

                let locality = addrSpace == 3 ? "local" : "global"

                let airOp: String
                let prefix: String
                switch atomicOp {
                case "xchg": airOp = "xchg"; prefix = ""
                case "add": airOp = "add"; prefix = ".s"
                case "sub": airOp = "sub"; prefix = ".s"
                case "max": airOp = "max"; prefix = ".s"
                case "min": airOp = "min"; prefix = ".s"
                case "umax": airOp = "umax"; prefix = ".u"
                case "umin": airOp = "umin"; prefix = ".u"
                case "and": airOp = "and"; prefix = ".s"
                case "or": airOp = "or"; prefix = ".s"
                case "xor": airOp = "xor"; prefix = ".s"
                default: continue
                }

                let valType = inst.type
                let typeSuffix: String
                switch valType {
                case .int(let bits): typeSuffix = "i\(bits)"
                case .float32: typeSuffix = "f32"
                default: typeSuffix = "i32"
                }

                let intrinsicName = "air.atomic.\(locality).\(airOp)\(prefix).\(typeSuffix)"

                // air.atomic.{local,global}.{op}.{s,u}.{type}(ptr, val, i32 order, i32 scope, i1 volatile)
                let i32Ty = IRType.int(bits: 32)
                let i1Ty = IRType.int(bits: 1)
                if module.functions.first(where: { $0.name == intrinsicName }) == nil {
                    let ptrTy: IRType = .pointer(pointee: valType, addressSpace: addrSpace)
                    let declFn = IRFunction(
                        name: intrinsicName,
                        returnType: valType,
                        parameterTypes: [ptrTy, valType, i32Ty, i32Ty, i1Ty],
                        isDeclaration: true
                    )
                    module.functions.append(declFn)
                }

                // Replace atomicrmw with call: (ptr, val, order=0, scope=1, volatile=true)
                let callee = IRValue(type: .function(ret: valType, params: [], isVarArg: false), name: intrinsicName)
                let callInst = IRInstruction(
                    opcode: .call,
                    type: valType,
                    name: inst.name,
                    operands: [
                        inst.operands[0],
                        inst.operands[1],
                        .constant(.integer(i32Ty, 0)),    // order = relaxed
                        .constant(.integer(i32Ty, 1)),    // scope = threadgroup
                        .constant(.integer(i1Ty, 1)),     // volatile = true
                        .value(callee),
                    ]
                )
                bb.instructions[instIdx] = callInst
            }
        }
    }
}

// MARK: - Transform: Mark device loads in loops as volatile
//
// Metal's GPU JIT performs LICM that hoists non-volatile loads out of loops,
// even when a store to the same pointer exists in the loop body. This causes
// loops with load+store patterns to effectively execute only 1 iteration.
//
// Fix: detect loops (via back-edges) and mark addrspace(1) loads as volatile
// when there's a store to the same base pointer in the loop body.

private func transformDeviceLoadsVolatile(module: IRModule) {
    for fn in module.functions where !fn.isDeclaration {
        let bbs = fn.basicBlocks
        guard bbs.count >= 2 else { continue }

        // Build BB name → index map
        var bbIndex: [String: Int] = [:]
        for (i, bb) in bbs.enumerated() {
            bbIndex[bb.name] = i
        }

        // Find loop bodies via back-edges (branch from bb[j] to bb[i] where i <= j)
        var loopBlocks: [Set<Int>] = []
        for (j, bb) in bbs.enumerated() {
            guard let lastInst = bb.instructions.last, lastInst.opcode == .br else { continue }
            for op in lastInst.operands {
                if case .basicBlock(let target) = op, let i = bbIndex[target.name], i <= j {
                    var loopSet = Set<Int>()
                    for k in i...j { loopSet.insert(k) }
                    loopBlocks.append(loopSet)
                }
            }
        }
        guard !loopBlocks.isEmpty else { continue }

        // For each loop, find stores to addrspace(1) and mark matching loads volatile
        for loopSet in loopBlocks {
            var storedPtrs: Set<String> = []
            for idx in loopSet {
                for inst in bbs[idx].instructions where inst.opcode == .store {
                    if inst.operands.count >= 2,
                       case .value(let ptrVal) = inst.operands[1],
                       ptrVal.type.addressSpace == 1 {
                        storedPtrs.insert(ptrVal.name)
                    }
                }
            }
            guard !storedPtrs.isEmpty else { continue }

            for idx in loopSet {
                for inst in bbs[idx].instructions where inst.opcode == .load {
                    if let op = inst.operands.first,
                       case .value(let ptrVal) = op,
                       storedPtrs.contains(ptrVal.name) {
                        inst.attributes.isVolatile = true
                    }
                }
            }
        }
    }
}
