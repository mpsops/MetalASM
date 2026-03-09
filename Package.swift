// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalASM",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "MetalASM", targets: ["MetalASM"]),
        .library(name: "MetalASMBridge", type: .dynamic, targets: ["MetalASMBridge"]),
    ],
    targets: [
        .target(
            name: "MetalASM",
            path: "Sources/MetalASM"
        ),
        .target(
            name: "MetalASMBridge",
            dependencies: ["MetalASM"],
            path: "Sources/MetalASMBridge",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .testTarget(
            name: "MetalASMTests",
            dependencies: ["MetalASM"],
            path: "Tests/MetalASMTests"
        ),
    ]
)
