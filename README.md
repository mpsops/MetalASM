# MetalASM

In-process LLVM IR to Metal library (`.metallib`) assembler, written in pure Swift.

Parses LLVM IR strings into an IR model, serializes to AIR bitcode, and wraps in the metallib container format — all without shelling out to `metal-as` or `metallib` CLI tools. This means it works on **both macOS and iOS**.

## Usage

```swift
import MetalASM

let ir = "..."  // LLVM IR string (generated in-memory, e.g. by a JIT kernel compiler)
let metallib = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))

// Load directly on GPU
let dispatchData = metallib.withUnsafeBytes { DispatchData(bytes: $0) }
let library = try device.makeLibrary(data: dispatchData)
```

## Supported platforms

- macOS 14+
- iOS 17+

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/imperatormk/MetalASM.git", from: "0.1.0"),
]
```

## Tests

```
swift test
```

## License

MIT
