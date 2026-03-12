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

    transformInlineNonKernelFunctions(module: module)
    transformStructPhis(module: module)
    transformPtrPhisToI64(module: module)
    transformBarrierRename(module: module)
    transformFNegToFSub(module: module)
    transformLLVMIntrinsicRename(module: module)
    transformI64ShuffleSplit(module: module)
    transformAtomicRMW(module: module)
    transformTGGlobalDeadElim(module: module)
    transformTGGlobalCoalesce(module: module)
    transformTGGlobalGEPs(module: module)
    transformMMATypedPtrs(module: module)
    transformBFloat16Casts(module: module)
    transformScalarStoreGuard(module: module)
    transformAirSystemValues(module: module)
    transformDeviceLoadsVolatile(module: module)
    transformWidenDeviceLoads(module: module)
}

// MARK: - Transform: Inline non-kernel functions
//
// Metal kernels cannot call other functions (no function pointers in AIR).
// Inline all calls to non-declaration, non-kernel functions into callers.
// Single-pass: handles simple cases (no recursion, single basic block callees).

private func transformInlineNonKernelFunctions(module: IRModule) {
    var inlineCounter = 0

    // Repeat until no more inlining is possible (handles nested call graphs)
    for _ in 0..<20 {  // safety limit
        // Rebuild callee map each iteration (functions get removed after inlining)
        var calledNames = Set<String>()
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                for inst in bb.instructions where inst.opcode == .call {
                    guard let calleeOp = inst.operands.last,
                          case .value(let calleeVal) = calleeOp else { continue }
                    calledNames.insert(calleeVal.name)
                }
            }
        }

        let inlineableFns = Dictionary(uniqueKeysWithValues:
            module.functions.filter { !$0.isDeclaration && calledNames.contains($0.name) }
                .map { ($0.name, $0) }
        )

        guard !inlineableFns.isEmpty else { return }

        var didInline = false
        let moduleFnNames = Set(module.functions.map { $0.name })
        let globalNames = Set(module.globals.map { $0.name })

        for fn in module.functions where !fn.isDeclaration {
            var bbIdx = 0
            while bbIdx < fn.basicBlocks.count {
                let bb = fn.basicBlocks[bbIdx]
                var i = 0
                while i < bb.instructions.count {
                    let inst = bb.instructions[i]
                    guard inst.opcode == .call,
                          let calleeOp = inst.operands.last,
                          case .value(let calleeVal) = calleeOp,
                          let callee = inlineableFns[calleeVal.name] else {
                        i += 1
                        continue
                    }

                    didInline = true
                    let prefix = "_inl\(inlineCounter)_"
                    inlineCounter += 1

                    // Build param → argument mapping
                    let args = Array(inst.operands.dropLast())
                    var renameMap: [String: IRInstruction.Operand] = [:]
                    for (idx, param) in callee.parameters.enumerated() {
                        if idx < args.count {
                            renameMap[param.name] = args[idx]
                        }
                    }

                    let renameOp = { (op: IRInstruction.Operand) -> IRInstruction.Operand in
                        switch op {
                        case .value(let v):
                            if let replacement = renameMap[v.name] { return replacement }
                            if moduleFnNames.contains(v.name) { return op }
                            if globalNames.contains(v.name) { return op }
                            return .value(IRValue(type: v.type, name: prefix + v.name))
                        default:
                            return op
                        }
                    }

                    if callee.basicBlocks.count == 1 {
                        // --- Single-BB inlining (simple path) ---
                        var newInsts: [IRInstruction] = []
                        var retOperand: IRInstruction.Operand? = nil

                        for srcInst in callee.basicBlocks[0].instructions {
                            if srcInst.opcode == .ret {
                                if let firstOp = srcInst.operands.first {
                                    retOperand = renameOp(firstOp)
                                }
                                continue
                            }

                            let cloned = IRInstruction(
                                opcode: srcInst.opcode,
                                type: srcInst.type,
                                name: srcInst.name.isEmpty ? "" : prefix + srcInst.name,
                                operands: srcInst.operands.map(renameOp),
                                attributes: srcInst.attributes
                            )
                            if !srcInst.name.isEmpty {
                                renameMap[srcInst.name] = .value(IRValue(type: srcInst.type, name: prefix + srcInst.name))
                            }
                            newInsts.append(cloned)
                        }

                        // Map call result to return value — rename across ALL BBs
                        if !inst.name.isEmpty, let retVal = retOperand {
                            if case .value(let v) = retVal, !v.name.isEmpty {
                                let oldName = inst.name
                                let newName = v.name
                                // Rename in rest of current BB
                                for j in (i + 1)..<bb.instructions.count {
                                    let later = bb.instructions[j]
                                    for (k, op) in later.operands.enumerated() {
                                        if case .value(let v2) = op, v2.name == oldName {
                                            later.operands[k] = .value(IRValue(type: v2.type, name: newName))
                                        }
                                    }
                                }
                                // Rename in all subsequent BBs of the function
                                let currentBBIdx = fn.basicBlocks.firstIndex(where: { $0 === bb })!
                                for laterBBIdx in (currentBBIdx + 1)..<fn.basicBlocks.count {
                                    for later in fn.basicBlocks[laterBBIdx].instructions {
                                        for (k, op) in later.operands.enumerated() {
                                            if case .value(let v2) = op, v2.name == oldName {
                                                later.operands[k] = .value(IRValue(type: v2.type, name: newName))
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        bb.instructions.replaceSubrange(i...i, with: newInsts)
                        // Don't advance i — re-process for nested calls

                    } else {
                        // --- Multi-BB inlining ---
                        // Split caller BB at call site:
                        //   bb: [before_call..., call, after_call...] →
                        //   bb: [before_call..., br callee_entry]
                        //   callee_bb0, callee_bb1, ... (ret → br continuation)
                        //   continuation: [after_call...]

                        // Create continuation BB with instructions after the call
                        let contBBName = "\(prefix)cont"
                        let afterCallInsts = Array(bb.instructions[(i + 1)...])
                        let contBB = IRBasicBlock(name: contBBName, instructions: afterCallInsts)

                        // Clone callee BBs with renamed labels and SSAs
                        var bbRenameMap: [String: IRBasicBlock] = [:]
                        var clonedBBs: [IRBasicBlock] = []

                        for srcBB in callee.basicBlocks {
                            let newName = prefix + srcBB.name
                            let clonedBB = IRBasicBlock(name: newName)
                            bbRenameMap[srcBB.name] = clonedBB
                            clonedBBs.append(clonedBB)
                        }

                        // Rename operand with BB awareness
                        let renameOpMultiBB = { (op: IRInstruction.Operand) -> IRInstruction.Operand in
                            switch op {
                            case .basicBlock(let refBB):
                                if let renamed = bbRenameMap[refBB.name] {
                                    return .basicBlock(renamed)
                                }
                                return op
                            default:
                                return renameOp(op)
                            }
                        }

                        // Clone instructions into each BB
                        for (srcBBIdx, srcBB) in callee.basicBlocks.enumerated() {
                            let clonedBB = clonedBBs[srcBBIdx]

                            for srcInst in srcBB.instructions {
                                if srcInst.opcode == .ret {
                                    // Replace ret with br to continuation
                                    // If call had a result, we need to propagate the return value
                                    if !inst.name.isEmpty, let firstOp = srcInst.operands.first {
                                        let retVal = renameOpMultiBB(firstOp)
                                        if case .value(let v) = retVal, !v.name.isEmpty {
                                            let oldName = inst.name
                                            // Rename uses in continuation BB
                                            for later in contBB.instructions {
                                                for (k, op) in later.operands.enumerated() {
                                                    if case .value(let v2) = op, v2.name == oldName {
                                                        later.operands[k] = .value(IRValue(type: v2.type, name: v.name))
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    clonedBB.instructions.append(
                                        IRInstruction(opcode: .br, operands: [.basicBlock(contBB)])
                                    )
                                    continue
                                }

                                let cloned = IRInstruction(
                                    opcode: srcInst.opcode,
                                    type: srcInst.type,
                                    name: srcInst.name.isEmpty ? "" : prefix + srcInst.name,
                                    operands: srcInst.operands.map(renameOpMultiBB),
                                    attributes: srcInst.attributes
                                )
                                if !srcInst.name.isEmpty {
                                    renameMap[srcInst.name] = .value(IRValue(type: srcInst.type, name: prefix + srcInst.name))
                                }
                                clonedBB.instructions.append(cloned)
                            }
                        }

                        // Truncate caller BB: remove call + after-call, add br to callee entry
                        bb.instructions.removeSubrange(i...)
                        bb.instructions.append(
                            IRInstruction(opcode: .br, operands: [.basicBlock(clonedBBs[0])])
                        )

                        // Insert cloned BBs + continuation after current BB
                        var insertBBs = clonedBBs
                        insertBBs.append(contBB)
                        fn.basicBlocks.insert(contentsOf: insertBBs, at: bbIdx + 1)

                        // Don't advance i — we've restructured BBs, break inner loop
                        break
                    }
                }
                bbIdx += 1
            }
        }

        // Remove functions that were fully inlined (no longer called)
        let stillCalledNames: Set<String> = {
            var names = Set<String>()
            for fn in module.functions where !fn.isDeclaration {
                for bb in fn.basicBlocks {
                    for inst in bb.instructions where inst.opcode == .call {
                        if let calleeOp = inst.operands.last,
                           case .value(let v) = calleeOp {
                            names.insert(v.name)
                        }
                    }
                }
            }
            return names
        }()
        module.functions.removeAll {
            !$0.isDeclaration && inlineableFns.keys.contains($0.name) && !stillCalledNames.contains($0.name)
        }

        if !didInline { return }
    }  // end repeat loop
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
        // allScalarPhis collected

        // Pass 1: Collect all insertvalue chains across all blocks
        // Map: insertvalue result name → {elemIdx: scalar operand}
        var insertValueScalars: [String: [Int: IRInstruction.Operand]] = [:]
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .insertValue && inst.operands.count >= 3 {
                if case .intLiteral(let idx) = inst.operands[2] {
                    insertValueScalars[inst.name, default: [:]][Int(idx)] = inst.operands[1]
                    // Inherit from aggregate base (another insertvalue chain)
                    if case .value(let agg) = inst.operands[0] {
                        if let existing = insertValueScalars[agg.name] {
                            for (k, v) in existing where insertValueScalars[inst.name]?[k] == nil {
                                insertValueScalars[inst.name, default: [:]][k] = v
                            }
                        }
                        // Inherit from split struct phi (agg was a struct phi → scalar phis)
                        if let phiNames = allScalarPhis[agg.name],
                           let phiTypes = allScalarTypes[agg.name] {
                            for (k, name) in phiNames.enumerated() where insertValueScalars[inst.name]?[k] == nil {
                                insertValueScalars[inst.name, default: [:]][k] = .value(IRValue(type: phiTypes[k], name: name))
                            }
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
                           allScalarPhis[agg.name] != nil ||
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

// MARK: - Transform: ptr phi → i64 phi (GPU JIT workaround)
//
// Metal GPU JIT has a limit of ~63 ptr-typed phi nodes in a single loop block.
// Pipelined matmul can produce 96+ ptr phis (32 per operand × 3 operands).
// Fix: convert ptr addrspace(1) phis to i64 phis using ptrtoint/inttoptr.
// i64 phis have no such limit (tested up to 96).

private func transformPtrPhisToI64(module: IRModule) {
    let i64Type = IRType.int(bits: 64)

    for fn in module.functions where !fn.isDeclaration {
        // Build block lookup for predecessor insertion
        let blocksByName = Dictionary(uniqueKeysWithValues: fn.basicBlocks.map { ($0.name, $0) })

        for bb in fn.basicBlocks {
            // Count ptr phis in this block
            let ptrPhis = bb.instructions.filter { inst in
                guard inst.opcode == .phi else { return false }
                switch inst.type {
                case .opaquePointer(let addrSpace) where addrSpace == 1: return true
                case .pointer(_, let addrSpace) where addrSpace == 1: return true
                default: return false
                }
            }

            guard ptrPhis.count > 32 else { continue }

            // Track: original phi name → inttoptr name (for use-site renaming)
            var ptrRestore: [(phiName: String, intToPtrName: String, ptrType: IRType)] = []
            // Track: predecessor block → [(ptrtoint name, source value operand)]
            var predInsertions: [String: [(name: String, srcOp: IRInstruction.Operand, ptrType: IRType)]] = [:]

            for phi in ptrPhis {
                let origPtrType = phi.type
                let origName = phi.name
                let restoreName = "\(origName)_ptr"

                // Change phi type to i64
                phi.type = i64Type

                // Process incoming values: [val0, bb0, val1, bb1, ...]
                var i = 0
                while i < phi.operands.count - 1 {
                    let valOp = phi.operands[i]
                    let bbOp = phi.operands[i + 1]

                    // Create ptrtoint instruction name
                    let p2iName: String
                    switch valOp {
                    case .value(let v):
                        p2iName = "\(v.name)_p2i"
                    case .constant(_):
                        p2iName = "\(origName)_const_p2i_\(i/2)"
                    default:
                        i += 2; continue
                    }

                    // Get predecessor block name
                    if case .basicBlock(let predBB) = bbOp {
                        predInsertions[predBB.name, default: []].append(
                            (name: p2iName, srcOp: valOp, ptrType: origPtrType)
                        )
                    }

                    // Update phi incoming to reference ptrtoint result
                    phi.operands[i] = .value(IRValue(type: i64Type, name: p2iName))

                    i += 2
                }

                ptrRestore.append((phiName: origName, intToPtrName: restoreName, ptrType: origPtrType))
            }

            // Infer typed pointer from GEP usage: if a GEP uses the phi's inttoptr result
            // with a non-float source type, the inttoptr must produce that typed ptr
            // (otherwise emitOpaqueAsTyped makes it float* → type mismatch with GEP source)
            var phiPointeeTypes: [String: IRType] = [:]
            for bb2 in fn.basicBlocks {
                for inst in bb2.instructions where inst.opcode == .getelementptr {
                    if let srcTy = inst.attributes.gepSourceType,
                       let baseOp = inst.operands.first,
                       case .value(let v) = baseOp {
                        for entry in ptrRestore where entry.phiName == v.name {
                            phiPointeeTypes[entry.phiName] = srcTy
                        }
                    }
                }
            }

            // Insert inttoptr instructions after ALL phis in the block
            let firstNonPhiIdx = bb.instructions.firstIndex(where: { $0.opcode != .phi }) ?? bb.instructions.count
            var intToPtrInsts: [IRInstruction] = []
            for (phiName, i2pName, ptrType) in ptrRestore {
                // Use typed pointer if we know the pointee from GEP usage
                let actualPtrType: IRType
                if let pointee = phiPointeeTypes[phiName],
                   case .opaquePointer(let as1) = ptrType {
                    actualPtrType = .pointer(pointee: pointee, addressSpace: as1)
                } else {
                    actualPtrType = ptrType
                }
                let i2p = IRInstruction(
                    opcode: .intToPtr,
                    type: actualPtrType,
                    name: i2pName,
                    operands: [.value(IRValue(type: i64Type, name: phiName))]
                )
                intToPtrInsts.append(i2p)
            }
            bb.instructions.insert(contentsOf: intToPtrInsts, at: firstNonPhiIdx)

            // Rename all uses of original phi name → inttoptr name (in ALL blocks)
            // But NOT in the phi itself (it stays as i64)
            let renameMap: [String: (String, IRType)] = Dictionary(uniqueKeysWithValues: ptrRestore.map {
                let actualType: IRType
                if let pointee = phiPointeeTypes[$0.phiName],
                   case .opaquePointer(let as1) = $0.ptrType {
                    actualType = .pointer(pointee: pointee, addressSpace: as1)
                } else {
                    actualType = $0.ptrType
                }
                return ($0.phiName, ($0.intToPtrName, actualType))
            })

            for renBB in fn.basicBlocks {
                for inst in renBB.instructions {
                    // Skip the phi instructions we just converted (they reference i64 values)
                    if inst.opcode == .phi && renameMap[inst.name] != nil { continue }
                    // Skip the inttoptr instructions we just created (they reference the phi by i64 name)
                    if inst.opcode == .intToPtr && ptrRestore.contains(where: { $0.intToPtrName == inst.name }) { continue }

                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j],
                           let (newName, newType) = renameMap[v.name] {
                            inst.operands[j] = .value(IRValue(type: newType, name: newName))
                        }
                    }
                }
            }

            // Insert ptrtoint instructions in predecessor blocks (before terminator)
            // Deduplicate: same source value in same block → same ptrtoint
            for (predName, insertions) in predInsertions {
                guard let predBB = blocksByName[predName] else { continue }
                // Find terminator (last instruction)
                let termIdx = predBB.instructions.count - 1
                guard termIdx >= 0 else { continue }

                // Deduplicate by name
                var seen = Set<String>()
                var p2iInsts: [IRInstruction] = []
                for (name, srcOp, ptrType) in insertions {
                    guard !seen.contains(name) else { continue }
                    seen.insert(name)

                    // If the source references a renamed phi (back-edge), use the inttoptr name
                    var actualSrcOp = srcOp
                    if case .value(let v) = srcOp, let (newName, newType) = renameMap[v.name] {
                        actualSrcOp = .value(IRValue(type: newType, name: newName))
                    }

                    let p2i = IRInstruction(
                        opcode: .ptrToInt,
                        type: i64Type,
                        name: name,
                        operands: [actualSrcOp]
                    )
                    _ = ptrType  // suppress unused warning
                    p2iInsts.append(p2i)
                }
                predBB.instructions.insert(contentsOf: p2iInsts, at: termIdx)
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
    // rint (round to nearest integer)
    ("llvm.rint.f32",   "air.fast_rint.f32"),
    ("llvm.rint.f16",   "air.fast_rint.f16"),
    // mulhi (unsigned multiply-high)
    ("__mulhi",         "air.mul_hi.u.i32"),
    ("__mul64hi",       "air.mul_hi.u.i64"),
]

// MARK: - Transform: fneg → fsub -0.0
//
// Metal GPU JIT (LLVM 14) doesn't support FUNC_CODE_INST_UNOP (fneg).
// Lower fneg %x → fsub float -0.0, %x.

private func transformFNegToFSub(module: IRModule) {
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            for i in bb.instructions.indices {
                let inst = bb.instructions[i]
                guard inst.opcode == .fneg, inst.operands.count >= 1 else { continue }
                let negZero: IRConstant
                switch inst.type {
                case .float16:  negZero = .float16(0x8000)  // -0.0 in f16 bits
                case .float32:  negZero = .float32(-0.0)
                case .float64:  negZero = .float64(-0.0)
                default:        negZero = .float32(-0.0)
                }
                bb.instructions[i] = IRInstruction(
                    opcode: .fsub, type: inst.type, name: inst.name,
                    operands: [.constant(negZero), inst.operands[0]])
            }
        }
    }
}

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

// MARK: - Transform: Split i64 SIMD shuffles into pairs of i32 shuffles
//
// Metal GPU JIT crashes (XPC_ERROR_CONNECTION_INTERRUPTED) on air.simd_shuffle_*.i64.
// Fix: split each i64 shuffle into two i32 shuffles on the low/high halves,
// then reassemble. Works for shuffle_up, shuffle_down, shuffle_xor, shuffle.

private func transformI64ShuffleSplit(module: IRModule) {
    let i32 = IRType.int(bits: 32)
    let i64 = IRType.int(bits: 64)
    let i16 = IRType.int(bits: 16)

    // Ensure i32 shuffle declarations exist
    var declaredShuffles = Set<String>()
    for fn in module.functions where fn.isDeclaration {
        declaredShuffles.insert(fn.name)
    }

    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            var i = 0
            while i < bb.instructions.count {
                let inst = bb.instructions[i]
                guard inst.opcode == .call,
                      let calleeOp = inst.operands.last,
                      case .value(let calleeVal) = calleeOp,
                      calleeVal.name.hasPrefix("air.simd_shuffle"),
                      calleeVal.name.hasSuffix(".i64") else {
                    i += 1
                    continue
                }

                let resultName = inst.name
                let i32Name = calleeVal.name.replacingOccurrences(of: ".i64", with: ".i32")

                // Declare i32 variant if needed
                if !declaredShuffles.contains(i32Name) {
                    let decl = IRFunction(
                        name: i32Name,
                        returnType: i32,
                        parameterTypes: [i32, i16],
                        isDeclaration: true
                    )
                    module.functions.append(decl)
                    declaredShuffles.insert(i32Name)
                }

                // operands: [value_i64, offset_i16, callee]
                let srcOp = inst.operands[0]
                let offsetOp = inst.operands[1]

                let loName = "\(resultName).lo"
                let hiName = "\(resultName).hi"
                let shufLoName = "\(resultName).shuf_lo"
                let shufHiName = "\(resultName).shuf_hi"
                let hiExtName = "\(resultName).hi_ext"
                let hiShlName = "\(resultName).hi_shl"

                // %lo = trunc i64 %src to i32
                let truncLo = IRInstruction(opcode: .trunc, type: i32, name: loName, operands: [srcOp])

                // %hi_pre = lshr i64 %src, 32
                let lshrInst = IRInstruction(opcode: .lshr, type: i64, name: "\(resultName).hi_pre", operands: [
                    srcOp, .constant(.integer(i64, 32))
                ])
                let hiPreVal = IRValue(type: i64, name: "\(resultName).hi_pre")

                // %hi = trunc i64 %hi_pre to i32
                let truncHi = IRInstruction(opcode: .trunc, type: i32, name: hiName, operands: [.value(hiPreVal)])

                // %shuf_lo = call i32 @air.simd_shuffle_*.i32(i32 %lo, i16 %offset)
                let loVal = IRValue(type: i32, name: loName)
                let hiVal = IRValue(type: i32, name: hiName)
                let fnType = IRType.pointer(
                    pointee: .function(ret: i32, params: [i32, i16], isVarArg: false),
                    addressSpace: 0
                )
                let fnVal = IRValue(type: fnType, name: i32Name)
                let shufLo = IRInstruction(opcode: .call, type: i32, name: shufLoName, operands: [
                    .value(loVal), offsetOp, .value(fnVal)
                ])

                // %shuf_hi = call i32 @air.simd_shuffle_*.i32(i32 %hi, i16 %offset)
                let fnVal2 = IRValue(type: fnType, name: i32Name)
                let shufHi = IRInstruction(opcode: .call, type: i32, name: shufHiName, operands: [
                    .value(hiVal), offsetOp, .value(fnVal2)
                ])

                // %hi_ext = zext i32 %shuf_hi to i64
                let shufLoVal = IRValue(type: i32, name: shufLoName)
                let shufHiVal = IRValue(type: i32, name: shufHiName)
                let hiExt = IRInstruction(opcode: .zext, type: i64, name: hiExtName, operands: [.value(shufHiVal)])

                // %hi_shl = shl i64 %hi_ext, 32
                let hiExtVal = IRValue(type: i64, name: hiExtName)
                let hiShl = IRInstruction(opcode: .shl, type: i64, name: hiShlName, operands: [
                    .value(hiExtVal), .constant(.integer(i64, 32))
                ])

                // %result = zext i32 %shuf_lo to i64  (lo_ext)
                // then or with hi_shl
                let loExtName = "\(resultName).lo_ext"
                let loExt = IRInstruction(opcode: .zext, type: i64, name: loExtName, operands: [.value(shufLoVal)])
                let loExtVal = IRValue(type: i64, name: loExtName)
                let hiShlVal = IRValue(type: i64, name: hiShlName)
                let orInst = IRInstruction(opcode: .or, type: i64, name: resultName, operands: [
                    .value(loExtVal), .value(hiShlVal)
                ])

                // Replace the original call with the sequence
                bb.instructions.replaceSubrange(i...i, with: [
                    truncLo, lshrInst, truncHi, shufLo, shufHi, hiExt, hiShl, loExt, orInst
                ])
                i += 9  // skip all new instructions
            }
        }
    }

    // Remove i64 shuffle declarations (no longer called)
    module.functions.removeAll { $0.isDeclaration && $0.name.hasPrefix("air.simd_shuffle") && $0.name.hasSuffix(".i64") }
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
    TypeTableWriter.collapseDevicePtrsToFloat = hasMMADecl
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
        // When MMA is present, GPU JIT requires all device ptrs to be float* —
        // half*, i8* etc. crash when MMA intrinsics are in the same module.
        var changed = false
        for i in fn.parameterTypes.indices {
            if case .opaquePointer(let addrSpace) = fn.parameterTypes[i] {
                var pointee = paramPointeeTypes[i] ?? .float32
                if hasMMADecl && addrSpace == 1 { pointee = .float32 }
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
                // Get addrspace from either opaque or typed pointer
                let addrSpace: Int
                let isOpaque: Bool
                switch inst.type {
                case .opaquePointer(let a): addrSpace = a; isOpaque = true
                case .pointer(_, let a): addrSpace = a; isOpaque = false
                default: continue
                }
                guard let srcTy = inst.attributes.gepSourceType else { continue }

                    // Metal has no i1 memory type — remap to i8
                    var pointee = (srcTy == .int(bits: 1)) ? IRType.int(bits: 8) : srcTy
                    if srcTy == .int(bits: 1) {
                        inst.attributes.gepSourceType = .int(bits: 8)
                    }

                    // For device ptrs with MMA: force float32 GEP source type.
                    // GPU JIT requires GEP source type to match pointer pointee type.
                    if addrSpace == 1 && pointee != .float32 && hasMMADecl {
                        let elemSize = scalarSizeAlign(srcTy).0
                        pointee = .float32
                        inst.attributes.gepSourceType = .float32
                        if elemSize != 4, inst.operands.count >= 2 {
                            if let newIdx = scaleGepIndexToFloat(inst.operands[1], elemSize: elemSize) {
                                inst.operands[1] = newIdx
                            } else {
                                // Variable index: insert scaling instruction
                                // half (2 bytes) → float (4 bytes): new_idx = old_idx * 2 / 4
                                // But 2/4 = 0.5 which loses precision for odd indices.
                                // Instead: byte_offset = old_idx * elemSize, new_idx = byte_offset / 4
                                // For half (2 bytes): new_idx = old_idx / 2 (truncating)
                                // This is only correct if the original indices are even, which they
                                // should be for half→float stride conversion in practice.
                                // Actually for half→float we need: new_byte_offset = old_idx * 2
                                // new_float_idx = new_byte_offset / 4 = old_idx / 2
                                // We insert: %scaled = lshr i32 %old_idx, 1 (divide by 2)
                                if elemSize == 2 {
                                    if case .value(let idxVal) = inst.operands[1] {
                                        let origIdxName = idxVal.name
                                        let scaledName = "\(inst.name)_hidx"
                                        let scaledInst = IRInstruction(
                                            opcode: .lshr, type: idxVal.type, name: scaledName,
                                            operands: [
                                                .value(idxVal),
                                                .constant(.integer(idxVal.type, 1))
                                            ])
                                        // Insert before this GEP
                                        let bb = fn.basicBlocks.first(where: { $0.instructions.contains(where: { $0 === inst }) })!
                                        if let idx = bb.instructions.firstIndex(where: { $0 === inst }) {
                                            bb.instructions.insert(scaledInst, at: idx)
                                        }
                                        inst.operands[1] = .value(IRValue(type: idxVal.type, name: scaledName))
                                        // Tag GEP so load widening can emit extractelement
                                        inst.attributes.gepHalfScaledOrigIdx = origIdxName
                                    }
                                }
                            }
                        }
                    }

                    let typedPtr = IRType.pointer(pointee: pointee, addressSpace: addrSpace)
                    if inst.type != typedPtr {
                        let oldType = inst.type
                        inst.type = typedPtr
                        // Propagate to uses
                        let gepName = inst.name
                        if !gepName.isEmpty {
                            for bb2 in fn.basicBlocks {
                                for inst2 in bb2.instructions {
                                    for j in inst2.operands.indices {
                                        if case .value(let v) = inst2.operands[j],
                                           v.name == gepName,
                                           (v.type == oldType || {
                                               if case .opaquePointer(_) = v.type { return true }
                                               return false
                                           }()) {
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

// MARK: - Transform: Coalesce threadgroup globals
//
// Merge __tg_cvt_* buffers (from ConvertLayoutOp) into a dot buffer they don't
// overlap with. The cvt op finishes before/after the dot — they're never live
// simultaneously. This reduces TG memory for large tile sizes.
// MARK: - Transform: Remove unreferenced TG globals
// Globals declared but never used in any instruction waste TG memory.
private func transformTGGlobalDeadElim(module: IRModule) {
    // Collect all global names referenced in instructions
    var referencedNames: Set<String> = []
    for fn in module.functions {
        for bb in fn.basicBlocks {
            for inst in bb.instructions {
                for op in inst.operands {
                    if case .value(let v) = op {
                        referencedNames.insert(v.name)
                    }
                }
            }
        }
    }
    // Remove unreferenced addrspace(3) globals
    module.globals.removeAll(where: { g in
        g.addressSpace == 3 && !referencedNames.contains(g.name)
    })
}

// Safe merges: __tg_cvt_* → __tg_dot_ab_* (cvt runs before dot scatter)
//              __tg_cvt_* → __tg_dot_c_*  (cvt of result runs after dot gather)
// Both the ab and c buffers are live during MMA, so they CANNOT be merged.
//
private func transformTGGlobalCoalesce(module: IRModule) {
    // Find cvt and dot globals
    var cvtGlobals: [(global: IRGlobal, elemType: IRType, count: Int)] = []
    var dotGlobals: [(global: IRGlobal, elemType: IRType, count: Int)] = []

    for g in module.globals where g.addressSpace == 3 {
        if case .array(let elemTy, let n) = g.valueType, elemTy != .i8, n > 64 {
            if g.name.hasPrefix("__tg_cvt_") {
                cvtGlobals.append((global: g, elemType: elemTy, count: n))
            } else if g.name.hasPrefix("__tg_dot_") {
                dotGlobals.append((global: g, elemType: elemTy, count: n))
            }
        }
    }

    guard !cvtGlobals.isEmpty, !dotGlobals.isEmpty else { return }

    // For each cvt global, merge into a dot_ab global (NOT dot_c — cvt reads from dot_c!)
    for cvt in cvtGlobals {
        guard let target = dotGlobals.first(where: {
            $0.elemType == cvt.elemType && $0.global.name.contains("_ab_")
        }) ?? dotGlobals.first(where: { $0.elemType == cvt.elemType }) else { continue }
        let oldName = cvt.global.name
        let newName = target.global.name
        // Rename all references
        for fn in module.functions {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    for (k, op) in inst.operands.enumerated() {
                        if case .value(let v) = op, v.name == oldName {
                            inst.operands[k] = .value(IRValue(type: v.type, name: newName))
                        }
                    }
                }
            }
        }

        // Resize target if cvt is larger
        if cvt.count > target.count {
            target.global.valueType = .array(element: cvt.elemType, count: cvt.count)
        }

        // Remove cvt global
        module.globals.removeAll(where: { $0.name == oldName })
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
    //
    // IMPORTANT: For [N x i8] byte globals, byte-offset GEPs into the middle of
    // the global crash Metal GPU JIT ("Failed to materializeAll"). Instead of
    // emitting GEP instructions with non-zero offsets, we create separate TG
    // globals for each distinct byte offset region and redirect references.
    var offsetGlobalMap: [String: [Int64: String]] = [:]
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

        // Phase A: Collect distinct byte offsets for each [N x i8] byte global.
        // Two sources of offsets:
        //   1. constantGEPByteOffset on operand values (from parser constant GEP expressions)
        //   2. GEP instructions with byte global base and constant index
        var byteGlobalOffsets: [String: Set<Int64>] = [:]
        let byteGlobalNames = Set(byteGlobals.map { $0.name })
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    // Source 1: constant GEP expression offsets
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i], v.constantGEPByteOffset != 0,
                           let info = tgGlobalInfo[v.name], info.elemBytes == 1 {
                            byteGlobalOffsets[v.name, default: []].insert(v.constantGEPByteOffset)
                        }
                    }
                    // Source 2: GEP instructions like `getelementptr i8, @global_smem, i64 512`
                    if inst.opcode == .getelementptr,
                       inst.operands.count >= 2,
                       case .value(let baseVal) = inst.operands[0],
                       byteGlobalNames.contains(baseVal.name),
                       baseVal.constantGEPByteOffset == 0,
                       case .constant(let c) = inst.operands[1],
                       case .integer(_, let offset) = c,
                       offset != 0 {
                        // Compute actual byte offset considering GEP source type
                        let elemSize: Int64
                        if let srcTy = inst.attributes.gepSourceType {
                            switch srcTy {
                            case .i8: elemSize = 1
                            case .i16, .float16, .bfloat16: elemSize = 2
                            case .i32, .float32: elemSize = 4
                            case .i64, .float64: elemSize = 8
                            default: elemSize = 1
                            }
                        } else {
                            elemSize = 1
                        }
                        byteGlobalOffsets[baseVal.name, default: []].insert(offset * elemSize)
                    }
                }
            }
        }

        // Phase B: Create new TG globals for each non-zero byte offset region
        // offsetGlobalMap: original name → [offset → new global name] (declared above do block)
        for (globalName, offsets) in byteGlobalOffsets {
            guard let g = module.globals.first(where: { $0.name == globalName }),
                  case .array(_, let totalBytes) = g.valueType else { continue }
            let sortedOffsets = offsets.sorted()
            var mapping: [Int64: String] = [:]
            for offset in sortedOffsets {
                let regionSize = totalBytes - Int(offset)
                guard regionSize > 0 else { continue }
                let newName = "\(globalName)__off\(offset)"
                mapping[offset] = newName
                let newGlobal = IRGlobal(
                    name: newName,
                    valueType: .array(element: .i8, count: regionSize),
                    addressSpace: 3,
                    initializer: .undef(.array(element: .i8, count: regionSize))
                )
                newGlobal.linkage = g.linkage
                newGlobal.alignment = g.alignment
                module.globals.append(newGlobal)
                // Also register in byteGlobals list so Part 1 picks it up
                byteGlobals.append(newGlobal)
            }
            offsetGlobalMap[globalName] = mapping
            // Shrink the original global to just the first region
            if let firstOffset = sortedOffsets.first {
                let newValueType = IRType.array(element: .i8, count: Int(firstOffset))
                g.valueType = newValueType
                g.type = .pointer(pointee: newValueType, addressSpace: 3)
            }
        }

        // Phase C: Flatten constant GEPs — for byte globals with offset regions,
        // redirect to the new split global instead of emitting a byte-offset GEP
        var ctr = 5000
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                var newInsts: [IRInstruction] = []
                for inst in bb.instructions {
                    for i in inst.operands.indices {
                        if case .value(let v) = inst.operands[i], v.constantGEPByteOffset != 0,
                           let info = tgGlobalInfo[v.name] {
                            // Check if this is a byte global with a split region
                            if let mapping = offsetGlobalMap[v.name],
                               let newGlobalName = mapping[v.constantGEPByteOffset] {
                                // Replace with direct reference to the new split global (offset 0)
                                inst.operands[i] = .value(IRValue(
                                    type: .opaquePointer(addressSpace: 3),
                                    name: newGlobalName))
                            } else {
                                // Non-byte global or no split: emit GEP instruction as before
                                let ssaName = "__cgep_\(ctr)"
                                ctr += 1
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
                    }
                    newInsts.append(inst)
                }
                bb.instructions = newInsts
            }
        }
    }

    // Phase D: Rewrite GEP instructions with byte-global base + constant offset
    // e.g. `%x = getelementptr i8, @global_smem, i64 512` → replace uses of %x
    // with direct reference to @global_smem__off512
    if !offsetGlobalMap.isEmpty {
        for fn in module.functions where !fn.isDeclaration {
            for bb in fn.basicBlocks {
                var replacements: [String: String] = [:]  // old SSA name → new global name
                bb.instructions = bb.instructions.compactMap { inst in
                    if inst.opcode == .getelementptr,
                       inst.operands.count >= 2,
                       case .value(let baseVal) = inst.operands[0],
                       let mapping = offsetGlobalMap[baseVal.name],
                       case .constant(let c) = inst.operands[1],
                       case .integer(_, let rawOffset) = c,
                       rawOffset != 0 {
                        // Compute actual byte offset
                        let elemSize: Int64
                        if let srcTy = inst.attributes.gepSourceType {
                            switch srcTy {
                            case .i8: elemSize = 1
                            case .i16, .float16, .bfloat16: elemSize = 2
                            case .i32, .float32: elemSize = 4
                            case .i64, .float64: elemSize = 8
                            default: elemSize = 1
                            }
                        } else {
                            elemSize = 1
                        }
                        let byteOffset = rawOffset * elemSize
                        if let newGlobalName = mapping[byteOffset] {
                            replacements[inst.name] = newGlobalName
                            return nil  // Remove this GEP instruction
                        }
                    }
                    return inst
                }
                // Replace all uses of removed GEPs with split global references
                if !replacements.isEmpty {
                    for inst in bb.instructions {
                        for i in inst.operands.indices {
                            if case .value(let v) = inst.operands[i],
                               let newGlobalName = replacements[v.name] {
                                inst.operands[i] = .value(IRValue(
                                    type: .opaquePointer(addressSpace: 3),
                                    name: newGlobalName))
                            }
                        }
                    }
                }
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
    // SKIP when MMA is present: type table collapses all device ptrs to float*,
    // so bitcasts would become identity (float* → float*) which is invalid.
    let hasMMAForBitcast = module.functions.contains { $0.name.hasPrefix("air.simdgroup_matrix_8x8_") }
    if !hasMMAForBitcast {
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
    } // end if !hasMMAForBitcast

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

        var origParamCount = fn.parameterTypes.count

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

        // Pass 5b: pack all scalar params into ONE device buffer.
        // Metal has a 31-buffer argument limit. The old approach gave each scalar
        // its own buffer slot, which exhausted the limit for kernels with many params.
        //
        // Scalar params come in two forms from the MLIR backend:
        //   a) ptr addrspace(2) — MLIR already converted i32/i64 to constant buffer ptrs
        //   b) raw scalars (float, i32, etc.) — MLIR didn't convert these
        // Both consume one Metal buffer slot each. We pack ALL of them into ONE
        // device buffer (ptr addrspace(1)). For form (a), we rewrite the existing
        // loads to use the packed buffer. For form (b), we insert new loads.
        // Dead params (from MLIR descriptor lowering) are included in the buffer
        // layout but get no GEP/load — they just occupy space so that offsets
        // match the Python driver's expand_signature-based packing.
        do {
            // Collect scalar param indices, their types, and whether they're already ptrs
            struct ScalarParam {
                let origIdx: Int
                let scalarType: IRType  // the underlying scalar type (e.g. .i32, .float32)
                let isConstPtr: Bool    // true if param is ptr addrspace(2)
                let isDead: Bool        // true if param has no loads (dead from MLIR DCE)
                let paramName: String
            }
            var scalarParams: [ScalarParam] = []

            // First pass: identify descriptor groups to infer types for dead params.
            // Between consecutive device pointers (addrspace(1)), scalar params form
            // a descriptor group. For no-metadata descriptors, _expand_descriptor
            // produces: [i64×(2N), i1, i1, i32×N, i64×N] where N = ndim.
            // N = (group_size - 2) / 4.
            struct DescriptorGroup {
                let startIdx: Int  // first scalar param index (after the device ptr)
                let count: Int     // number of scalar params in this group
            }
            var descriptorGroups: [DescriptorGroup] = []
            var i = 0
            while i < origParamCount {
                let pt = fn.parameterTypes[i]
                let isDevicePtr: Bool
                switch pt {
                case .pointer(_, addressSpace: 1), .opaquePointer(addressSpace: 1):
                    isDevicePtr = true
                default:
                    isDevicePtr = false
                }
                if isDevicePtr {
                    // Count consecutive addrspace(2) params after this device ptr
                    var count = 0
                    var j = i + 1
                    while j < origParamCount {
                        var matched = false
                        switch fn.parameterTypes[j] {
                        case .pointer(_, addressSpace: 2), .opaquePointer(addressSpace: 2):
                            matched = true
                            count += 1; j += 1
                        default:
                            break
                        }
                        if !matched { break }
                    }
                    if count > 0 {
                        descriptorGroups.append(DescriptorGroup(startIdx: i + 1, count: count))
                    }
                }
                i += 1
            }

            /// Given a descriptor group of `count` scalar params, return the type
            /// pattern matching _expand_descriptor(no metadata): [i64×(2N), i1, i1, i32×N, i64×N].
            func descriptorTypePattern(count: Int) -> [IRType] {
                // count = 4*N + 2 → N = (count - 2) / 4
                guard count >= 2, (count - 2) % 4 == 0 else {
                    // Not a recognized descriptor pattern — fall back to i64
                    return Array(repeating: .int(bits: 64), count: count)
                }
                let ndim = (count - 2) / 4
                var types: [IRType] = []
                for _ in 0..<(2 * ndim) { types.append(.int(bits: 64)) }  // shape + strides
                types.append(.int(bits: 1))   // padding
                types.append(.int(bits: 1))   // tf32
                for _ in 0..<ndim { types.append(.int(bits: 32)) }  // block_shape
                for _ in 0..<ndim { types.append(.int(bits: 64)) }  // block_strides
                return types
            }

            // Build lookup: param index → type from descriptor pattern (for dead params).
            // Only apply the descriptor pattern if live params' types actually match
            // the pattern — otherwise it's just regular scalars (e.g. M, N, K, strides)
            // that happen to follow a device pointer.
            var descriptorTypeForParam: [Int: IRType] = [:]
            for group in descriptorGroups {
                let pattern = descriptorTypePattern(count: group.count)
                // Validate: check that live params' load types match the pattern
                var patternValid = true
                for offset in 0..<group.count {
                    let paramIdx = group.startIdx + offset
                    let paramName = fn.parameterNames[paramIdx]
                    let liveType = inferLoadType(fn: fn, paramIdx: paramIdx, paramName: paramName)
                    if liveType != .void {
                        // Live param — its type must match the pattern
                        if !typeSizesMatch(liveType, pattern[offset]) {
                            patternValid = false
                            break
                        }
                    }
                }
                if patternValid {
                    for (offset, ty) in pattern.enumerated() {
                        descriptorTypeForParam[group.startIdx + offset] = ty
                    }
                }
            }

            for i in 0..<origParamCount {
                let paramType = fn.parameterTypes[i]
                let paramName = fn.parameterNames[i]

                switch paramType {
                // Form (b): raw scalar params
                case .float32, .float64, .float16, .bfloat16,
                     .int(bits: 1), .int(bits: 8), .int(bits: 16),
                     .int(bits: 32), .int(bits: 64):
                    scalarParams.append(ScalarParam(
                        origIdx: i, scalarType: paramType,
                        isConstPtr: false, isDead: false, paramName: paramName
                    ))

                // Form (a): ptr addrspace(2) — constant buffer pointer from MLIR
                case .pointer(_, addressSpace: 2), .opaquePointer(addressSpace: 2):
                    let loadType = inferLoadType(fn: fn, paramIdx: i, paramName: paramName)
                    if loadType == .void {
                        // Dead param — use descriptor pattern type for correct layout
                        let deadType = descriptorTypeForParam[i] ?? .int(bits: 32)
                        scalarParams.append(ScalarParam(
                            origIdx: i, scalarType: deadType,
                            isConstPtr: true, isDead: true, paramName: paramName
                        ))
                    } else {
                        scalarParams.append(ScalarParam(
                            origIdx: i, scalarType: loadType,
                            isConstPtr: true, isDead: false, paramName: paramName
                        ))
                    }

                default:
                    continue  // device/TG pointers stay as-is
                }
            }

            if !scalarParams.isEmpty, let entryBB = fn.basicBlocks.first {
                let scalarTypes = scalarParams.map { $0.scalarType }

                // Compute byte offset for each scalar field (natural alignment)
                var fieldOffsets: [Int] = []
                var currentOffset = 0
                for ty in scalarTypes {
                    let (size, align) = scalarSizeAlign(ty)
                    let padding = (align - (currentOffset % align)) % align
                    currentOffset += padding
                    fieldOffsets.append(currentOffset)
                    currentOffset += size
                }
                // When MMA is present, use float as the scalar buf element type
                // to avoid i8* typed pointers that crash GPU JIT with MMA.
                let hasMMAForScalar = module.functions.contains { $0.name.hasPrefix("air.simdgroup_matrix_8x8_") }
                let scalarElemType: IRType = hasMMAForScalar ? .float32 : .i8
                let scalarElemSize: Int = hasMMAForScalar ? 4 : 1
                let bufPtrType = IRType.pointer(pointee: scalarElemType, addressSpace: 1)
                let bufPtrName = "_scalar_buf"

                // Build preamble: GEP + load for each LIVE scalar (skip dead)
                var preamble: [IRInstruction] = []
                for (j, sp) in scalarParams.enumerated() {
                    if sp.isDead { continue }  // dead params just occupy space in the buffer

                    let offset = fieldOffsets[j]
                    let baseName = sp.paramName.isEmpty ? "scalar_\(sp.origIdx)" : sp.paramName
                    let loadName: String
                    if sp.isConstPtr {
                        loadName = baseName.hasSuffix("_ptr") ? String(baseName.dropLast(4)) : baseName
                    } else {
                        loadName = baseName
                    }

                    // GEP: index in scalarElemSize-byte units
                    let gepIndex = Int64(offset / scalarElemSize)
                    let gepName = "\(loadName)_gep"
                    let gepInst = IRInstruction(
                        opcode: .getelementptr,
                        type: bufPtrType,
                        name: gepName,
                        operands: [
                            .value(IRValue(type: bufPtrType, name: bufPtrName)),
                            .constant(.integer(.i64, gepIndex)),
                        ]
                    )
                    gepInst.attributes.inBounds = true
                    gepInst.attributes.gepSourceType = scalarElemType
                    preamble.append(gepInst)

                    // Bitcast to typed pointer for the load (skip when MMA — types collapse to float*)
                    let loadPtrName: String
                    let loadPtrType: IRType
                    if hasMMAForScalar {
                        // No bitcast needed — GEP already produces float*, load widening handles the rest
                        loadPtrName = gepName
                        loadPtrType = bufPtrType
                    } else {
                        loadPtrType = IRType.pointer(pointee: sp.scalarType, addressSpace: 1)
                        let castName = "\(loadName)_bcp"
                        let castInst = IRInstruction(
                            opcode: .bitcast,
                            type: loadPtrType,
                            name: castName,
                            operands: [.value(IRValue(type: bufPtrType, name: gepName))]
                        )
                        preamble.append(castInst)
                        loadPtrName = castName
                    }

                    if sp.isConstPtr {
                        // Form (a): rewrite existing loads from this param to use new pointer.
                        let oldParamName = sp.paramName
                        for bb in fn.basicBlocks {
                            for inst in bb.instructions where inst.opcode == .load {
                                if let op = inst.operands.first, case .value(let v) = op,
                                   v.name == oldParamName {
                                    inst.operands[0] = .value(IRValue(type: loadPtrType, name: loadPtrName))
                                }
                            }
                        }
                    } else {
                        // Form (b): insert new load instruction
                        let loadInst = IRInstruction(
                            opcode: .load, type: sp.scalarType, name: loadName,
                            operands: [.value(IRValue(type: loadPtrType, name: loadPtrName))]
                        )
                        let (_, align) = scalarSizeAlign(sp.scalarType)
                        loadInst.attributes.alignment = align
                        preamble.append(loadInst)
                    }
                }

                // Remove ALL scalar params (live + dead), add single packed buffer param
                let allScalarIndices = scalarParams.map { $0.origIdx }
                for i in allScalarIndices.sorted().reversed() {
                    fn.parameterTypes.remove(at: i)
                    fn.parameterNames.remove(at: i)
                    fn.parameters.remove(at: i)
                }
                let remainingOrigCount = origParamCount - allScalarIndices.count
                fn.parameterTypes.insert(bufPtrType, at: remainingOrigCount)
                fn.parameterNames.insert(bufPtrName, at: remainingOrigCount)
                fn.parameters.insert(IRValue(type: bufPtrType, name: bufPtrName), at: remainingOrigCount)
                origParamCount = remainingOrigCount + 1

                entryBB.instructions.insert(contentsOf: preamble, at: 0)
                fn.type = .function(ret: fn.returnType, params: fn.parameterTypes, isVarArg: false)
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

/// Infer the scalar type loaded from a ptr addrspace(2) parameter.
/// Scans the function for `load T, ptr addrspace(2) %paramName` and returns T.
/// Returns .void if no load is found (dead parameter).
private func inferLoadType(fn: IRFunction, paramIdx: Int, paramName: String) -> IRType {
    for bb in fn.basicBlocks {
        for inst in bb.instructions where inst.opcode == .load {
            if let op = inst.operands.first, case .value(let v) = op,
               v.name == paramName {
                return inst.type
            }
        }
    }
    return .void  // dead param — no loads found
}

/// Check if two scalar types have the same byte size (for descriptor pattern validation).
private func typeSizesMatch(_ a: IRType, _ b: IRType) -> Bool {
    return scalarSizeAlign(a).0 == scalarSizeAlign(b).0
}

/// Returns (size, alignment) in bytes for scalar types in packed buffer layout.
private func scalarSizeAlign(_ t: IRType) -> (Int, Int) {
    switch t {
    case .i1, .int(bits: 1):   return (1, 1)
    case .i8, .int(bits: 8):   return (1, 1)
    case .i16, .int(bits: 16): return (2, 2)
    case .i32, .int(bits: 32): return (4, 4)
    case .i64, .int(bits: 64): return (8, 8)
    case .float16:             return (2, 2)
    case .bfloat16:            return (2, 2)
    case .float32:             return (4, 4)
    case .float64:             return (8, 8)
    default:                   return (4, 4) // fallback
    }
}

// MARK: - Transform: Force float* for all device pointers when MMA is present
//
// The Metal GPU JIT crashes when non-float typed pointers (half*, i32*, i8*)
// coexist with MMA intrinsics in the same module. This pass runs last and
// forces all device pointer types to float*.

// MARK: - Transform: Widen device loads for MMA compatibility

/// When MMA intrinsics are present, ALL device loads must be `load float`.
/// GPU JIT crashes on `load half`/`load i32`/etc from device ptr with MMA.
/// This pass runs LAST so it catches loads created by earlier passes (e.g. scalar buffer packing).
private func transformWidenDeviceLoads(module: IRModule) {
    let hasMMADecl = module.functions.contains { $0.name.hasPrefix("air.simdgroup_matrix_8x8_") }
    guard hasMMADecl else { return }

    for fn in module.functions where !fn.isDeclaration {
        // Build map of GEP names that were half-scaled (lshr) → original index name + type.
        // We find the lshr instruction (%<gep>_hidx) and get its first operand as the
        // current name of the original index (may have been renamed by later passes).
        var halfScaledGEPs: [String: (origIdxName: String, idxType: IRType)] = [:]
        // First, build map of lshr name → first operand (the original index)
        var lshrOrigIdx: [String: (name: String, type: IRType)] = [:]
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .lshr {
                if inst.name.hasSuffix("_hidx"),
                   inst.operands.count >= 1,
                   case .value(let origVal) = inst.operands[0] {
                    lshrOrigIdx[inst.name] = (name: origVal.name, type: origVal.type)
                }
            }
        }
        for bb in fn.basicBlocks {
            for inst in bb.instructions where inst.opcode == .getelementptr {
                if inst.attributes.gepHalfScaledOrigIdx != nil {
                    // Find the lshr that feeds this GEP's index
                    let lshrName = "\(inst.name)_hidx"
                    if let origInfo = lshrOrigIdx[lshrName] {
                        halfScaledGEPs[inst.name] = (origIdxName: origInfo.name, idxType: origInfo.type)
                    }
                }
            }
        }
        // Propagate through phi, inttoptr, ptrtoint, bitcast — any instruction that
        // passes a half-scaled pointer through to a new name
        var changed = true
        while changed {
            changed = false
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if halfScaledGEPs[inst.name] != nil { continue }
                    switch inst.opcode {
                    case .phi, .intToPtr, .bitcast:
                        for op in inst.operands {
                            if case .value(let v) = op, let info = halfScaledGEPs[v.name] {
                                halfScaledGEPs[inst.name] = info
                                changed = true
                                break
                            }
                        }
                    case .ptrToInt:
                        if let op = inst.operands.first,
                           case .value(let v) = op,
                           let info = halfScaledGEPs[v.name] {
                            halfScaledGEPs[inst.name] = info
                            changed = true
                        }
                    default: break
                    }
                }
            }
        }



        var widenedLoads: [String: (castName: String, origType: IRType)] = [:]
        for bb in fn.basicBlocks {
            var newInsts: [IRInstruction] = []
            for inst in bb.instructions {
                if inst.opcode == .load,
                   inst.type != .float32,
                   inst.operands.count >= 1,
                   case .value(let ptrVal) = inst.operands[0],
                   ({
                       switch ptrVal.type {
                       case .pointer(_, 1), .opaquePointer(1): return true
                       default: return false
                       }
                   }()) {
                    let origType = inst.type
                    let castName = "\(inst.name)_wcast"
                    // Check if this load's pointer came from a half-scaled GEP
                    if (origType == .float16 || origType == .bfloat16),
                       let halfInfo = halfScaledGEPs[ptrVal.name] {
                        // Load-and-extract pattern:
                        // %f = load float, float* %gep        (load 4 bytes = 2 halves)
                        // %v2 = bitcast float %f to <2 x half>
                        // %bit = and i32 %origIdx, 1           (odd/even selector)
                        // %val = extractelement <2 x half> %v2, i32 %bit
                        inst.type = .float32
                        newInsts.append(inst)

                        let v2Type = IRType.vector(element: origType, count: 2)
                        let bcName = "\(inst.name)_v2h"
                        let bcInst = IRInstruction(
                            opcode: .bitcast, type: v2Type, name: bcName,
                            operands: [.value(IRValue(type: .float32, name: inst.name))])
                        newInsts.append(bcInst)

                        let andName = "\(inst.name)_lane"
                        let andInst = IRInstruction(
                            opcode: .and, type: halfInfo.idxType, name: andName,
                            operands: [
                                .value(IRValue(type: halfInfo.idxType, name: halfInfo.origIdxName)),
                                .constant(.integer(halfInfo.idxType, 1))
                            ])
                        newInsts.append(andInst)

                        let extractInst = IRInstruction(
                            opcode: .extractElement, type: origType, name: castName,
                            operands: [
                                .value(IRValue(type: v2Type, name: bcName)),
                                .value(IRValue(type: halfInfo.idxType, name: andName))
                            ])
                        newInsts.append(extractInst)
                        widenedLoads[inst.name] = (castName, origType)
                        continue
                    }

                    inst.type = .float32
                    newInsts.append(inst)
                    let castOpcode: IRInstruction.Opcode
                    switch origType {
                    case .float16, .bfloat16: castOpcode = .fpTrunc
                    case .int(let bits) where bits == 32: castOpcode = .bitcast
                    case .int(let bits) where bits < 32:
                        let bcName = "\(inst.name)_bc"
                        let bcInst = IRInstruction(
                            opcode: .bitcast, type: .int(bits: 32), name: bcName,
                            operands: [.value(IRValue(type: .float32, name: inst.name))])
                        newInsts.append(bcInst)
                        let truncInst = IRInstruction(
                            opcode: .trunc, type: origType, name: castName,
                            operands: [.value(IRValue(type: .int(bits: 32), name: bcName))])
                        newInsts.append(truncInst)
                        widenedLoads[inst.name] = (castName, origType)
                        continue
                    default:
                        inst.type = origType
                        newInsts.append(inst)
                        continue
                    }
                    let castInst = IRInstruction(
                        opcode: castOpcode, type: origType, name: castName,
                        operands: [.value(IRValue(type: .float32, name: inst.name))])
                    newInsts.append(castInst)
                    widenedLoads[inst.name] = (castName, origType)
                    continue
                }
                newInsts.append(inst)
            }
            bb.instructions = newInsts
        }
        if !widenedLoads.isEmpty {
            for bb in fn.basicBlocks {
                for inst in bb.instructions {
                    if widenedLoads[inst.name] != nil && inst.opcode == .load { continue }
                    if inst.name.hasSuffix("_wcast") || inst.name.hasSuffix("_bc")
                        || inst.name.hasSuffix("_v2h") || inst.name.hasSuffix("_lane") { continue }
                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j],
                           let entry = widenedLoads[v.name] {
                            inst.operands[j] = .value(IRValue(type: entry.origType, name: entry.castName))
                        }
                    }
                }
            }
        }
    }

    // Eliminate device pointer bitcasts that become identity after type table collapse.
    // e.g. bitcast i32* %x → i8* becomes bitcast float* → float* (identity) which is invalid.
    let floatDevPtr = IRType.pointer(pointee: .float32, addressSpace: 1)
    for fn in module.functions where !fn.isDeclaration {
        for bb in fn.basicBlocks {
            var renames: [String: String] = [:] // old name → replacement name
            bb.instructions.removeAll { inst in
                if inst.opcode == .bitcast,
                   case .pointer(_, 1) = inst.type,
                   inst.operands.count == 1,
                   case .value(let src) = inst.operands[0],
                   case .pointer(_, 1) = src.type {
                    // Both source and dest are device pointers — will collapse to same float*
                    renames[inst.name] = src.name
                    return true
                }
                if inst.opcode == .bitcast,
                   case .opaquePointer(1) = inst.type,
                   inst.operands.count == 1,
                   case .value(let src) = inst.operands[0],
                   case .pointer(_, 1) = src.type {
                    renames[inst.name] = src.name
                    return true
                }
                if inst.opcode == .bitcast,
                   case .pointer(_, 1) = inst.type,
                   inst.operands.count == 1,
                   case .value(let src) = inst.operands[0],
                   case .opaquePointer(1) = src.type {
                    renames[inst.name] = src.name
                    return true
                }
                return false
            }
            if !renames.isEmpty {
                for inst in bb.instructions {
                    for j in inst.operands.indices {
                        if case .value(let v) = inst.operands[j],
                           let replacement = renames[v.name] {
                            inst.operands[j] = .value(IRValue(type: floatDevPtr, name: replacement))
                        }
                    }
                }
            }
        }
    }
}

/// Try to scale a GEP index from `elemSize`-byte elements to float (4-byte) elements.
/// Returns the new operand if scaling is possible, nil otherwise.
private func scaleGepIndexToFloat(_ op: IRInstruction.Operand, elemSize: Int) -> IRInstruction.Operand? {
    if elemSize == 4 { return op }
    switch op {
    case .intLiteral(let val):
        let byteOffset = val * Int64(elemSize)
        guard byteOffset % 4 == 0 else { return nil }
        return .intLiteral(byteOffset / 4)
    case .constant(let c):
        if case .integer(let cTy, let val) = c {
            let byteOffset = val * Int64(elemSize)
            guard byteOffset % 4 == 0 else { return nil }
            return .constant(.integer(cTy, byteOffset / 4))
        }
        return nil
    default:
        return nil
    }
}

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
