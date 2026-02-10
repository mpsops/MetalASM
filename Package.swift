// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalASM",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "MetalASM", targets: ["MetalASM"]),
    ],
    targets: [
        .target(
            name: "MetalASM",
            path: "Sources/MetalASM"
        ),
        .testTarget(
            name: "MetalASMTests",
            dependencies: ["MetalASM"],
            path: "Tests/MetalASMTests"
        ),
    ]
)
