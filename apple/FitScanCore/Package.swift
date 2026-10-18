// swift-tools-version:5.9
// FitScanCore: Annys Körper, Maße wie mit dem Maßband, Anpassung an Maße und GLB-Export.
// Ohne Oberfläche und ohne Netz, damit es auf iPhone und Mac gleich rechnet.
import PackageDescription

let package = Package(
    name: "FitScanCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "FitScanCore", targets: ["FitScanCore"]),
    ],
    targets: [
        .target(
            name: "FitScanCore",
            resources: [.copy("Resources/anny_v1.fsan")]
        ),
        .testTarget(
            name: "FitScanCoreTests",
            dependencies: ["FitScanCore"],
            resources: [.copy("Resources/referenz.json")]
        ),
    ]
)
