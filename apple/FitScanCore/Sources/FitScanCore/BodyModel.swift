import Foundation

/// Annys Ruhepose für eine Person: Punkte, Knochenköpfe und Knochenlagen, in Metern, Z oben,
/// Blick nach −Y, linke Körperseite bei +X (wie am PC).
public struct RestBody {
    public var vertices: [Vec3]
    public var boneHeads: [Vec3]
    /// Lage jedes Knochens in der Ruhepose (Drehung + Kopf), wie `rest_numpy()["bone_poses"]`
    public var bonePoses: [Rigid]
    /// Punkte vor dem Maßstab (für die Anpassung)
    var unscaledVertices: [Vec3]
}

/// Rechnet Annys Körper aus den Reglern, wie `fitscan.body.BodyModel` mit PyTorch.
public final class BodyModel {
    public let data: AnnyData

    public init(data: AnnyData) { self.data = data }

    public static func bundled() throws -> BodyModel { BodyModel(data: try AnnyData.bundled()) }

    public var numVertices: Int { data.numVertices }
    public var faces: [UInt32] { data.faces }

    // MARK: Koeffizienten

    /// Gewichte je Anker eines Merkmals, wie `anny.utils.interpolation.linear_interpolation_coefficients`.
    static func interpolation(_ value: Double, _ anchors: [Double]) -> (index: Int, weights: [Double], slope: [Double]) {
        let n = anchors.count
        let v = min(max(value, anchors[0]), anchors[n - 1])
        var idx = anchors.firstIndex { $0 >= v } ?? n  // searchsorted(side: left)
        idx = min(max(idx, 1), n - 1)
        let lo = anchors[idx - 1], hi = anchors[idx]
        let alpha = (v - lo) / (hi - lo)
        var w = [Double](repeating: 0, count: n)
        var s = [Double](repeating: 0, count: n)
        w[idx - 1] = 1 - alpha
        w[idx] = alpha
        let clamped = value < anchors[0] || value > anchors[n - 1]
        if !clamped {
            s[idx - 1] = -1 / (hi - lo)
            s[idx] = 1 / (hi - lo)
        }
        return (idx, w, s)
    }

    /// Werte der 26 Phänotyp-Spalten und ihre Ableitung nach jedem Phänotyp-Regler.
    func phenotypeColumns(_ phenotype: [String: Double]) -> (values: [Double], slopes: [String: [Double]]) {
        var values: [Double] = []
        var slopes: [String: [Double]] = [:]
        func value(_ key: String) -> Double { phenotype[key] ?? data.fixedPhenotypes[key] ?? 0.5 }
        for (feature, keys) in data.variations {
            if feature == "race" {
                let raw = keys.map(value)
                let total = raw.reduce(0, +)
                values += raw.map { total > 0 ? $0 / total : 1.0 / 3.0 }
                continue
            }
            let start = values.count
            let anchors = data.anchors[feature] ?? []
            let (_, w, s) = Self.interpolation(value(feature), anchors)
            values += w
            if data.phenotypeLabels.contains(feature) {
                var col = [Double](repeating: 0, count: data.maskColumns)
                for i in 0..<s.count { col[start + i] = s[i] }
                slopes[feature] = col
            }
        }
        return (values, slopes)
    }

    /// Formgewichte (Makroformen aus dem Phänotyp, dann je lokalem Regler + und −),
    /// wie `Anny._get_phenotype_blendshape_coefficients`.
    public func coefficients(_ params: BodyParams) -> [Double] {
        let (cols, _) = phenotypeColumns(params.phenotype)
        var c = [Double](repeating: 0, count: data.numShapes)
        let mc = data.maskColumns
        for j in 0..<data.numMacroShapes {
            var w = 1.0
            for i in 0..<mc where data.phenotypeMask[j * mc + i] != 0 {
                w *= cols[i]
            }
            c[j] = w
        }
        for (i, name) in data.localLabels.enumerated() {
            let v = params.local[name] ?? 0
            c[data.numMacroShapes + 2 * i] = max(v, 0)
            c[data.numMacroShapes + 2 * i + 1] = max(-v, 0)
        }
        return c
    }

    /// Ableitung der Makro-Formgewichte nach einem Phänotyp-Regler (nur Formen ≠ 0).
    func macroDerivatives(_ phenotype: [String: Double], feature: String) -> [(shape: Int, value: Double)] {
        let (cols, slopes) = phenotypeColumns(phenotype)
        guard let slope = slopes[feature] else { return [] }
        let mc = data.maskColumns
        var out: [(Int, Double)] = []
        for j in 0..<data.numMacroShapes {
            var d = 0.0
            for i in 0..<mc where data.phenotypeMask[j * mc + i] != 0 && slope[i] != 0 {
                var prod = slope[i]
                for k in 0..<mc where k != i && data.phenotypeMask[j * mc + k] != 0 {
                    prod *= cols[k]
                }
                d += prod
            }
            if d != 0 { out.append((j, d)) }
        }
        return out
    }

    // MARK: Ruhepose

    /// Punkte vor dem Maßstab: Vorlage + Σ Gewicht · Form (+ Feinform).
    func unscaledVertices(_ c: [Double], offsets: [Double]?) -> [Vec3] {
        var flat = data.templateVertices
        flat.withUnsafeMutableBufferPointer { out in
            data.shapeDelta.withUnsafeBufferPointer { delta in
                data.shapeVertex.withUnsafeBufferPointer { vert in
                    for s in 0..<data.numShapes where c[s] != 0 {
                        let f = c[s] * data.shapeScale[s]
                        for e in data.shapeStart[s]..<data.shapeStart[s + 1] {
                            let v = Int(vert[e]) * 3
                            out[v] += f * Double(delta[3 * e])
                            out[v + 1] += f * Double(delta[3 * e + 1])
                            out[v + 2] += f * Double(delta[3 * e + 2])
                        }
                    }
                }
            }
        }
        if let offsets, offsets.count == flat.count {
            for i in 0..<flat.count { flat[i] += offsets[i] }
        }
        return (0..<data.numVertices).map { Vec3(flat[3 * $0], flat[3 * $0 + 1], flat[3 * $0 + 2]) }
    }

    /// Σ Gewicht · Form für eine Knochengröße mit `width` Werten je Knochen.
    private func boneBlend(_ c: [Double], template: [Double], shapes: [Float], width: Int) -> [Double] {
        var out = template
        let stride = data.numBones * width
        for s in 0..<data.numShapes where c[s] != 0 {
            let base = s * stride
            for i in 0..<stride { out[i] += c[s] * Double(shapes[base + i]) }
        }
        return out
    }

    public func rest(_ params: BodyParams) -> RestBody {
        rest(coefficients: coefficients(params), offsets: params.offsets, scale: params.scale)
    }

    func rest(coefficients c: [Double], offsets: [Double]?, scale: Double) -> RestBody {
        let unscaled = unscaledVertices(c, offsets: offsets)
        let headsFlat = boneBlend(c, template: data.templateBoneHeads, shapes: data.boneHeadShapes, width: 3)
        let orientFlat = boneBlend(c, template: data.boneTemplateOrientation, shapes: data.boneOrientationShapes, width: 9)
        var heads: [Vec3] = []
        var poses: [Rigid] = []
        for j in 0..<data.numBones {
            let h = Vec3(headsFlat[3 * j], headsFlat[3 * j + 1], headsFlat[3 * j + 2]) * scale
            heads.append(h)
            let r = specialProcrustes(Mat3(Array(orientFlat[(9 * j)..<(9 * j + 9)])))
            poses.append(Rigid(rotation: r, translation: h))
        }
        return RestBody(vertices: unscaled.map { $0 * scale }, boneHeads: heads, bonePoses: poses,
                        unscaledVertices: unscaled)
    }

    /// Σ über die Formeinträge von `shape`: Form · g (für den Gradienten der Anpassung).
    func shapeDot(_ shape: Int, _ g: [Vec3]) -> Double {
        var sum = 0.0
        for e in data.shapeStart[shape]..<data.shapeStart[shape + 1] {
            let gv = g[Int(data.shapeVertex[e])]
            sum += gv.x * Double(data.shapeDelta[3 * e]) + gv.y * Double(data.shapeDelta[3 * e + 1])
                + gv.z * Double(data.shapeDelta[3 * e + 2])
        }
        return sum * data.shapeScale[shape]
    }
}
