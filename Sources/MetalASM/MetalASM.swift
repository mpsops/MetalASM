import Foundation

/// MetalASM: In-process LLVM IR → metallib serializer.
///
/// Converts LLVM IR text into a metallib that can be loaded via
/// `device.makeLibrary(data:)` — entirely in-process, no external tools.
public enum MetalASM {

    /// Last assembly timing breakdown (set after each `assemble` call).
    public static var _lastTiming: String = ""

    /// Assemble LLVM IR text into metallib data.
    ///
    /// - Parameters:
    ///   - ir: LLVM IR text (the contents of a .ll file).
    ///   - platform: Target platform (default: macOS 26).
    /// - Returns: Metallib data suitable for `MTLDevice.makeLibrary(data:)`.
    public static func assemble(
        ir: String,
        platform: MetallibWriter.Platform = .macOS(version: 26)
    ) throws -> Data {
        // Phase 1: Parse IR text into in-memory representation
        let t0 = CFAbsoluteTimeGetCurrent()
        let lexer = Lexer(source: ir)
        let tokens = lexer.tokenize()
        let t1 = CFAbsoluteTimeGetCurrent()
        var parser = Parser(tokens: tokens, source: lexer.source)
        let module = try parser.parse()
        let t2 = CFAbsoluteTimeGetCurrent()

        // Phase 1.5: Apply Air compatibility transforms
        applyAirTransforms(module: module)

        // Debug: check for opaque pointers after transforms
        let ve_dbg = ValueEnumerator(module: module)
        let opqTypes = ve_dbg.types.filter { if case .opaquePointer(_) = $0 { return true }; return false }
        if !opqTypes.isEmpty { print("[assemble] WARNING: opaque ptr types remain: \(opqTypes)") }

        // Debug: dump metadata
        if ProcessInfo.processInfo.environment["METALASM_DUMP_IR"] != nil {
            var dump = ""
            for fn in module.functions {
                dump += "fn: \(fn.name)(\(fn.parameterTypes.map { "\($0)" }.joined(separator: ", ")))\n"
            }
            for (i, md) in module.metadataNodes.enumerated() {
                dump += "!\(i) = !{\(md.operands.map { "\($0)" }.joined(separator: ", "))}\n"
            }
            try? dump.write(toFile: "/tmp/metalasm_debug.txt", atomically: true, encoding: .utf8)
            print("[assemble] dumped debug info to /tmp/metalasm_debug.txt")
        }

        // Phase 2: Serialize IR to LLVM bitcode
        let bitcodeBytes = BitcodeWriter.write(module: module)
        let t3 = CFAbsoluteTimeGetCurrent()

        // Store timing in thread-local so caller can read it
        MetalASM._lastTiming = String(format: "lex=%.0fms parse=%.0fms bc=%.0fms (%dKB→%dKB) %@",
                     (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000,
                     ir.utf8.count/1024, bitcodeBytes.count/1024,
                     BitcodeWriter._bcBreakdown)

        // Phase 3: Wrap in metallib container
        let entries = module.kernelEntries.map { name in
            MetallibWriter.Entry(
                name: name,
                type: .kernel,
                airVersionMajor: module.airVersion?.major ?? 2,
                airVersionMinor: module.airVersion?.minor ?? 8,
                metalVersionMajor: module.metalVersion?.major ?? 4,
                metalVersionMinor: module.metalVersion?.minor ?? 0
            )
        } + module.visibleEntries.map { name in
            MetallibWriter.Entry(
                name: name,
                type: .visible,
                airVersionMajor: module.airVersion?.major ?? 2,
                airVersionMinor: module.airVersion?.minor ?? 8,
                metalVersionMajor: module.metalVersion?.major ?? 4,
                metalVersionMinor: module.metalVersion?.minor ?? 0
            )
        }

        let wrappedBitcode = MetallibWriter.wrapBitcode(bitcodeBytes)
        var metallib = MetallibWriter.buildFromWrappedBitcode(
            wrappedBitcode: wrappedBitcode,
            entries: entries,
            platform: platform
        )
        MetallibWriter.patchHashes(in: &metallib, wrappedBitcode: wrappedBitcode)
        return metallib
    }

    /// Wrap already-compiled LLVM bitcode bytes (e.g. from `metal-as`) into a metallib.
    ///
    /// This is useful when you have pre-compiled bitcode and just need the metallib container.
    ///
    /// - Parameters:
    ///   - bitcode: Raw LLVM bitcode bytes (starting with "BC\xC0\xDE").
    ///   - kernelNames: Names of kernel functions in the bitcode.
    ///   - platform: Target platform.
    /// - Returns: Metallib data.
    public static func wrapBitcode(
        _ bitcode: [UInt8],
        kernelNames: [String],
        platform: MetallibWriter.Platform = .macOS(version: 26)
    ) -> Data {
        let entries = kernelNames.map {
            MetallibWriter.Entry(name: $0, type: .kernel)
        }
        let wrappedBitcode = MetallibWriter.wrapBitcode(bitcode)
        var metallib = MetallibWriter.buildFromWrappedBitcode(
            wrappedBitcode: wrappedBitcode,
            entries: entries,
            platform: platform
        )
        MetallibWriter.patchHashes(in: &metallib, wrappedBitcode: wrappedBitcode)
        return metallib
    }

    /// Wrap an already-wrapped bitcode section (wrapper + bitcode) into a metallib.
    ///
    /// Use this when you have a .air file (which is already a wrapped bitcode).
    public static func wrapAIR(
        _ airData: [UInt8],
        kernelNames: [String],
        platform: MetallibWriter.Platform = .macOS(version: 26)
    ) -> Data {
        let entries = kernelNames.map {
            MetallibWriter.Entry(name: $0, type: .kernel)
        }
        var metallib = MetallibWriter.buildFromWrappedBitcode(
            wrappedBitcode: airData,
            entries: entries,
            platform: platform
        )
        MetallibWriter.patchHashes(in: &metallib, wrappedBitcode: airData)
        return metallib
    }
}
