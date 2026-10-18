import Foundation

public enum FitScanError: Error, LocalizedError, Equatable {
    case data(String)
    case input([String])

    public var errorDescription: String? {
        switch self {
        case .data(let text): return text
        case .input(let problems): return problems.joined(separator: " ")
        }
    }
}

/// Annys Daten aus `anny_v1.fsan` (geschrieben von `fitscan.swiftdata` auf dem PC).
///
/// Aufbau: "FSAN" | u32 Version | u32 Länge JSON | u32 Länge Daten | u32 gepackte Länge | JSON | Daten.
/// Der Datenteil ist rohes Deflate, wie es `NSData.decompressed(using: .zlib)` liest.
public final class AnnyData {
    public let numVertices: Int
    public let numBones: Int
    public let numShapes: Int
    public let numMacroShapes: Int
    public let phenotypeLabels: [String]
    public let localLabels: [String]
    public let boneLabels: [String]
    public let boneParents: [Int]
    let variations: [(feature: String, keys: [String])]
    let anchors: [String: [Double]]
    let fixedPhenotypes: [String: Double]

    // Vorlage
    public let templateVertices: [Double]  // 3 · V
    public let faces: [UInt32]  // 3 · F
    public let vertexBoneIndices: [UInt8]  // 9 · V
    public let vertexBoneWeights: [Float]  // 9 · V
    public let bonesPerVertex: Int
    let phenotypeMask: [UInt8]  // P · 26
    let maskColumns: Int
    // Knochen
    let templateBoneHeads: [Double]  // 3 · J
    let boneHeadShapes: [Float]  // S · J · 3
    let boneTemplateOrientation: [Double]  // J · 9
    let boneOrientationShapes: [Float]  // S · J · 9
    public let referenceBoneOrientations: [Float]  // J · 9
    // Formen, dünn besetzt: Punkte shapeStart[s] ..< shapeStart[s + 1]
    let shapeStart: [Int]
    let shapeScale: [Double]
    let shapeVertex: [UInt16]
    let shapeDelta: [Int16]  // 3 je Eintrag
    // Hautbild-Koordinaten (für spätere Builds)
    public let textureCoordinates: [Float]
    public let faceTextureIndices: [UInt32]

    public var numFaces: Int { faces.count / 3 }

    public static func bundled() throws -> AnnyData {
        guard let url = Bundle.module.url(forResource: "anny_v1", withExtension: "fsan") else {
            throw FitScanError.data("Annys Daten fehlen in der App (anny_v1.fsan).")
        }
        return try AnnyData(url: url)
    }

    public convenience init(url: URL) throws {
        try self.init(data: Data(contentsOf: url, options: .mappedIfSafe))
    }

    public init(data raw: Data) throws {
        func u32(_ at: Int) -> Int {
            var value: UInt32 = 0
            withUnsafeMutableBytes(of: &value) { dst in
                _ = raw.copyBytes(to: dst, from: (raw.startIndex + at)..<(raw.startIndex + at + 4))
            }
            return Int(UInt32(littleEndian: value))
        }
        guard raw.count > 20, raw.prefix(4) == Data("FSAN".utf8) else {
            throw FitScanError.data("Keine FitScan-Anny-Datei.")
        }
        let version = u32(4), jsonLength = u32(8), payloadLength = u32(12), packedLength = u32(16)
        guard version == 1 else { throw FitScanError.data("Unbekannte Version \(version) der Anny-Datei.") }
        guard 20 + jsonLength <= raw.count else {
            throw FitScanError.data("Anny-Datei ist beschädigt (Kopf zu lang).")
        }
        let jsonStart = raw.startIndex + 20
        let jsonData = raw.subdata(in: jsonStart..<(jsonStart + jsonLength))
        let body = raw.subdata(in: (jsonStart + jsonLength)..<raw.endIndex)
        let payload: Data
        if packedLength > 0 {
            payload = try (body as NSData).decompressed(using: .zlib) as Data
        } else {
            payload = body
        }
        guard payload.count == payloadLength else {
            throw FitScanError.data("Anny-Datei ist beschädigt (Datenteil hat die falsche Länge).")
        }
        guard let meta = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let arrays = meta["arrays"] as? [String: [String: Any]] else {
            throw FitScanError.data("Anny-Datei: Kopf nicht lesbar.")
        }

        func spec(_ name: String) throws -> (offset: Int, count: Int) {
            guard let s = arrays[name], let offset = s["offset"] as? Int, let shape = s["shape"] as? [Int] else {
                throw FitScanError.data("Anny-Datei: Feld \(name) fehlt.")
            }
            return (offset, shape.reduce(1, *))
        }
        func read<T>(_ name: String, _ zero: T) throws -> [T] {
            let (offset, count) = try spec(name)
            var out = [T](repeating: zero, count: count)
            let start = payload.startIndex + offset
            let end = start + count * MemoryLayout<T>.stride
            guard end <= payload.endIndex else { throw FitScanError.data("Anny-Datei: Feld \(name) zu kurz.") }
            out.withUnsafeMutableBytes { dst in _ = payload.copyBytes(to: dst, from: start..<end) }
            return out
        }

        let nv = meta["num_vertices"] as? Int ?? 0
        let nm = meta["num_macro_shapes"] as? Int ?? 0
        numVertices = nv
        numBones = meta["num_bones"] as? Int ?? 0
        numShapes = meta["num_shapes"] as? Int ?? 0
        numMacroShapes = nm
        phenotypeLabels = meta["phenotype_labels"] as? [String] ?? []
        localLabels = meta["local_labels"] as? [String] ?? []
        boneLabels = meta["bone_labels"] as? [String] ?? []
        boneParents = meta["bone_parents"] as? [Int] ?? []
        variations = (meta["phenotype_variations"] as? [[Any]] ?? []).compactMap { pair in
            guard pair.count == 2, let f = pair[0] as? String, let k = pair[1] as? [String] else { return nil }
            return (feature: f, keys: k)
        }
        anchors = (meta["anchors"] as? [String: [Double]]) ?? [:]
        fixedPhenotypes = (meta["fixed_phenotypes"] as? [String: Double]) ?? [:]

        templateVertices = try read("template_vertices", Float(0)).map(Double.init)
        faces = try read("faces", UInt32(0))
        vertexBoneIndices = try read("vertex_bone_indices", UInt8(0))
        let weights = try read("vertex_bone_weights", Float(0))
        vertexBoneWeights = weights
        bonesPerVertex = nv > 0 ? weights.count / nv : 0
        let mask = try read("phenotype_mask", UInt8(0))
        phenotypeMask = mask
        maskColumns = nm > 0 ? mask.count / nm : 0
        templateBoneHeads = try read("template_bone_heads", Float(0)).map(Double.init)
        boneHeadShapes = try read("bone_heads_blendshapes", Float(0))
        boneTemplateOrientation = try read("bone_template_orientation", Float(0)).map(Double.init)
        boneOrientationShapes = try read("bone_orientation_blendshapes", Float(0))
        referenceBoneOrientations = try read("reference_bone_orientations", Float(0))
        shapeStart = try read("shape_start", UInt32(0)).map(Int.init)
        shapeScale = try read("shape_scale", Float(0)).map(Double.init)
        shapeVertex = try read("shape_vertex", UInt16(0))
        shapeDelta = try read("shape_delta", Int16(0))
        textureCoordinates = try read("texture_coordinates", Float(0))
        faceTextureIndices = try read("face_texture_indices", UInt32(0))

        guard templateVertices.count == 3 * numVertices, shapeStart.count == numShapes + 1,
              boneHeadShapes.count == numShapes * numBones * 3,
              variations.reduce(0, { $0 + $1.keys.count }) == maskColumns else {
            throw FitScanError.data("Anny-Datei: Größen passen nicht zusammen.")
        }
    }

    public func bone(_ name: String) -> Int? { boneLabels.firstIndex(of: name) }

    /// Punkte, deren Skinning-Gewicht überwiegend auf Knochen liegt, für die `predicate` gilt
    /// (wie `BodyModel.dominant_bone_mask`).
    public func dominantBoneMask(_ predicate: (String) -> Bool) -> [Bool] {
        let selected = boneLabels.map(predicate)
        var mask = [Bool](repeating: false, count: numVertices)
        for v in 0..<numVertices {
            var w: Float = 0
            for k in 0..<bonesPerVertex where selected[Int(vertexBoneIndices[v * bonesPerVertex + k])] {
                w += vertexBoneWeights[v * bonesPerVertex + k]
            }
            mask[v] = w > 0.5
        }
        return mask
    }

    static let armBonePrefixes = ["shoulder01", "upperarm", "lowerarm", "wrist", "finger", "metacarpal"]

    public func armVertexMask() -> [Bool] {
        dominantBoneMask { name in Self.armBonePrefixes.contains { name.hasPrefix($0) } }
    }
}
