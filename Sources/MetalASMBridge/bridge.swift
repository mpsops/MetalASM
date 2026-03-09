import Metal
import Foundation
import MetalASM

// ── Opaque handle ──────────────────────────────────────────────────────────
final class PSOHandle {
    let pso: MTLComputePipelineState
    let tew: Int
    init(_ pso: MTLComputePipelineState) { self.pso = pso; self.tew = pso.threadExecutionWidth }
}
private func retain<T: AnyObject>(_ obj: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(obj).toOpaque()
}
private func release(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<AnyObject>.fromOpaque(ptr).release()
}

// ── Shared device + queue ─────────────────────────────────────────────────
private let _device: MTLDevice = {
    guard let d = MTLCreateSystemDefaultDevice() else { fatalError("no MTLDevice") }
    return d
}()
private let _queue: MTLCommandQueue = {
    guard let q = _device.makeCommandQueue() else { fatalError("makeCommandQueue failed") }
    return q
}()

// ── compile ───────────────────────────────────────────────────────────────
@_cdecl("metalasm_compile")
public func metalasm_compile(
    _ llText: UnsafePointer<CChar>,
    _ outLen:  UnsafeMutablePointer<UInt64>,
    _ errbuf:  UnsafeMutablePointer<CChar>?,
    _ errlen:  Int
) -> UnsafeMutableRawPointer? {
    do {
        let data = try MetalASM.assemble(ir: String(cString: llText))
        let ptr = malloc(data.count)!
        data.copyBytes(to: ptr.assumingMemoryBound(to: UInt8.self), count: data.count)
        outLen.pointee = UInt64(data.count)
        return ptr
    } catch {
        let msg = error.localizedDescription
        if let buf = errbuf, errlen > 0 {
            msg.withCString { memcpy(buf, $0, min(strlen($0), errlen-1)); buf[min(strlen($0), errlen-1)] = 0 }
        }
        outLen.pointee = 0; return nil
    }
}

@_cdecl("metalasm_free")
public func metalasm_free(_ ptr: UnsafeMutableRawPointer?) { free(ptr) }

// ── load_pso ──────────────────────────────────────────────────────────────
@_cdecl("metalasm_load_pso")
public func metalasm_load_pso(
    _ metallib: UnsafeRawPointer, _ len: UInt64,
    _ fnName: UnsafePointer<CChar>,
    _ errbuf: UnsafeMutablePointer<CChar>?, _ errlen: Int
) -> UnsafeMutableRawPointer? {
    func fail(_ msg: String) -> UnsafeMutableRawPointer? {
        if let buf = errbuf, errlen > 0 {
            msg.withCString { memcpy(buf, $0, min(strlen($0), errlen-1)); buf[min(strlen($0), errlen-1)] = 0 }
        }
        return nil
    }
    var data = Data(bytes: metallib, count: Int(len))
    let lib: MTLLibrary
    do {
        lib = try data.withUnsafeMutableBytes { ptr in
            let dd = DispatchData(bytes: UnsafeRawBufferPointer(ptr))
            return try _device.makeLibrary(data: dd)
        }
    } catch { return fail("makeLibrary: \(error)") }
    guard let fn = lib.makeFunction(name: String(cString: fnName))
    else { return fail("no function '\(String(cString: fnName))'") }
    do {
        let pso = try _device.makeComputePipelineState(function: fn)
        return retain(PSOHandle(pso))
    } catch { return fail("makePSO: \(error)") }
}

@_cdecl("metalasm_release_pso")
public func metalasm_release_pso(_ ptr: UnsafeMutableRawPointer?) {
    guard let p = ptr else { return }; release(p)
}

@_cdecl("metalasm_tew")
public func metalasm_tew(_ ptr: UnsafeMutableRawPointer) -> Int {
    Unmanaged<PSOHandle>.fromOpaque(ptr).takeUnretainedValue().tew
}

// ── dispatch ──────────────────────────────────────────────────────────────
/// gpu_vas[i]    = tensor.data_ptr()   (GPU virtual address)
/// buf_sizes[i]  = tensor.nbytes       (so we can make a correctly-sized MTLBuffer)
/// offsets[i]    = storage_offset * element_size (byte offset within the buffer)
@_cdecl("metalasm_dispatch")
public func metalasm_dispatch(
    _ pso_ptr:      UnsafeMutableRawPointer,
    _ gpu_vas:      UnsafePointer<UInt64>,
    _ buf_sizes:    UnsafePointer<UInt64>,   // ← NEW: actual size of each buffer
    _ offsets:      UnsafePointer<UInt64>,
    _ n_buffers:    Int,
    _ scalars:      UnsafeRawPointer?,
    _ scalar_sizes: UnsafePointer<UInt32>?,
    _ scalar_idxs:  UnsafePointer<UInt32>?,
    _ n_scalars:    Int,
    _ gx: Int, _ gy: Int, _ gz: Int,
    _ lx: Int, _ ly: Int, _ lz: Int
) {
    let handle = Unmanaged<PSOHandle>.fromOpaque(pso_ptr).takeUnretainedValue()
    guard let cb  = _queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { return }
    enc.setComputePipelineState(handle.pso)

    let pageSize = UInt64(getpagesize())
    for i in 0 ..< n_buffers {
        let va      = gpu_vas[i]
        let sz      = Int(buf_sizes[i])
        let off     = Int(offsets[i])
        // Align VA down to page boundary; compensate in offset
        let alignedVA  = va & ~(pageSize - 1)
        let headPad    = Int(va - alignedVA)
        let totalBytes = headPad + off + sz  // must cover headPad + user offset + data

        guard let cpuPtr = UnsafeMutableRawPointer(bitPattern: UInt(alignedVA)),
              let buf = _device.makeBuffer(
                bytesNoCopy: cpuPtr,
                length:      totalBytes,
                options:     [.storageModeShared],
                deallocator: nil)
        else {
            // fallback: skip — shader will get nil buffer (will crash on access, but at least we don't hang)
            continue
        }
        enc.setBuffer(buf, offset: headPad + off, index: i)
    }

    // Scalars via setBytes
    if n_scalars > 0, let scalarBytes = scalars,
       let sizes = scalar_sizes, let idxs = scalar_idxs {
        var byteOff = 0
        for i in 0 ..< n_scalars {
            enc.setBytes(scalarBytes.advanced(by: byteOff), length: Int(sizes[i]), index: Int(idxs[i]))
            byteOff += Int(sizes[i])
        }
    }

    enc.dispatchThreads(MTLSize(width: gx, height: gy, depth: gz),
                        threadsPerThreadgroup: MTLSize(width: lx, height: ly, depth: lz))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

// Debug: print whether makeBuffer succeeds
@_cdecl("metalasm_debug_buffer")
public func metalasm_debug_buffer(_ gpu_va: UInt64, _ sz: UInt64) -> Int {
    let pageSize = UInt64(getpagesize())
    let alignedVA  = gpu_va & ~(pageSize - 1)
    let headPad    = Int(gpu_va - alignedVA)
    let totalBytes = headPad + Int(sz)
    guard let cpuPtr = UnsafeMutableRawPointer(bitPattern: UInt(alignedVA)) else {
        print("metalasm_debug_buffer: bitPattern failed"); return -1
    }
    if let buf = _device.makeBuffer(bytesNoCopy: cpuPtr, length: totalBytes,
                                    options: [.storageModeShared], deallocator: nil) {
        print("metalasm_debug_buffer: buf=\(buf) gpuAddr=\(String(format:"%llx", buf.gpuAddress)) headPad=\(headPad)")
        return 1
    } else {
        print("metalasm_debug_buffer: makeBuffer FAILED — va=\(String(format:"%llx",gpu_va)) aligned=\(String(format:"%llx",alignedVA)) sz=\(sz)")
        return 0
    }
}
