import Foundation

/// Netz für die 3D-Vorschau in glTF-Koordinaten (Y oben, Blick +Z, Füße auf y = 0), ohne Skelett.
/// Schnell genug, um es bei jeder Reglerbewegung neu zu rechnen.
public struct PreviewMesh {
    public var vertices: [Vec3]
    public var normals: [Vec3]
    public var faces: [UInt32]
    public var height: Double
}

/// Ein Formregler für die Nachbearbeitung: Phänotyp (0…1) oder lokaler Anny-Regler (−1…1).
public struct ShapeRegulator: Identifiable, Hashable {
    public enum Kind: Hashable { case phenotype, local }

    public var id: String { key }
    public let key: String
    public let kind: Kind
    public let label: String
    public let lowLabel: String
    public let highLabel: String
    /// Maß, das sich mit dem Regler vor allem ändert (für die Anzeige daneben)
    public let measure: MeasureKey?

    public var range: ClosedRange<Double> { kind == .phenotype ? 0...1 : -1...1 }
    public var neutral: Double { kind == .phenotype ? 0.5 : 0 }

    public func value(in params: BodyParams) -> Double {
        switch kind {
        case .phenotype: return params.phenotype[key] ?? 0.5
        case .local: return params.local[key] ?? 0
        }
    }

    /// Ganzer Körper: Muskeln, Körperfett, Proportionen
    public static let body: [ShapeRegulator] = [
        ShapeRegulator(key: "muscle", kind: .phenotype, label: "Muskeln", lowLabel: "weniger", highLabel: "mehr",
                       measure: .upperarm),
        ShapeRegulator(key: "weight", kind: .phenotype, label: "Körperfett", lowLabel: "schlank", highLabel: "kräftig",
                       measure: .weight),
        ShapeRegulator(key: "proportions", kind: .phenotype, label: "Proportionen", lowLabel: "kompakt",
                       highLabel: "gestreckt", measure: .inseam),
    ]

    /// Einzelne Körperstellen, dieselben Regler wie bei der Anpassung an das Maßband
    public static let regions: [ShapeRegulator] = [
        ShapeRegulator(key: "measure-bust-circ-incr", kind: .local, label: "Brust", lowLabel: "schmaler",
                       highLabel: "breiter", measure: .chest),
        ShapeRegulator(key: "measure-waist-circ-incr", kind: .local, label: "Taille", lowLabel: "schmaler",
                       highLabel: "breiter", measure: .waist),
        ShapeRegulator(key: "measure-hips-circ-incr", kind: .local, label: "Hüfte", lowLabel: "schmaler",
                       highLabel: "breiter", measure: .hips),
        ShapeRegulator(key: "measure-upperarm-circ-incr", kind: .local, label: "Oberarme", lowLabel: "dünner",
                       highLabel: "dicker", measure: .upperarm),
        ShapeRegulator(key: "measure-thigh-circ-incr", kind: .local, label: "Oberschenkel", lowLabel: "dünner",
                       highLabel: "dicker", measure: .thigh),
        ShapeRegulator(key: "measure-calf-circ-incr", kind: .local, label: "Waden", lowLabel: "dünner",
                       highLabel: "dicker", measure: .calf),
        ShapeRegulator(key: "measure-neck-circ-incr", kind: .local, label: "Hals", lowLabel: "dünner",
                       highLabel: "dicker", measure: .neck),
    ]
}

public extension BodyModel {
    /// Rechnet Anny (Z oben) in den glTF-Raum um, Füße auf y = 0, wie `buildRig`.
    private func glTFTransform(_ r: RestBody) -> Rigid {
        let minZ = r.vertices.map(\.z).min() ?? 0
        return Rigid(rotation: toYUp, translation: Vec3(0, -minZ, 0))
    }

    func previewMesh(_ params: BodyParams) -> PreviewMesh {
        let r = rest(params)
        let toY = glTFTransform(r)
        let vy = r.vertices.map { toY.apply($0) }
        let height = vy.map(\.y).max() ?? 1.8
        return PreviewMesh(vertices: vy, normals: vertexNormals(vy, data.faces), faces: data.faces, height: height)
    }

    /// Wo das Maßband liegt: je Umfang die geschlossene Linie im glTF-Raum (wie `previewMesh`).
    func tapeLines(_ params: BodyParams) -> [MeasureKey: [Vec3]] {
        let r = rest(params)
        let toY = glTFTransform(r)
        let probe = MeasureProbe(model: self, rest: r)
        var out: [MeasureKey: [Vec3]] = [:]
        for (key, ring) in probe.rings { out[key] = ring.points(r.vertices).map { toY.apply($0) } }
        return out
    }

    /// Nachbearbeitung von Hand: setzt Formregler auf neue Werte. Mit `heightCM` bleibt die
    /// Körpergröße genau erhalten, der Maßstab gleicht die Regler aus.
    func adjusted(_ params: BodyParams, _ values: [ShapeRegulator: Double], heightCM: Double?) -> BodyParams {
        var p = params
        for (regulator, value) in values {
            let v = min(max(value, regulator.range.lowerBound), regulator.range.upperBound)
            switch regulator.kind {
            case .phenotype: p.phenotype[regulator.key] = v
            case .local: p.local[regulator.key] = v
            }
        }
        if let heightCM { p = exactHeight(p, heightCM: heightCM) }
        return p
    }
}

/// Dünner Schlauch um eine geschlossene Linie, z. B. ein Maßband in der Vorschau. Er liegt außen
/// auf der Linie (um `radius` nach außen versetzt), damit er nicht in der Haut verschwindet.
public struct TubeMesh {
    public var vertices: [Vec3] = []
    public var normals: [Vec3] = []
    public var indices: [UInt32] = []

    public init(closedLine p: [Vec3], radius: Double = 0.004, sides: Int = 8) {
        let n = p.count
        guard n >= 3, sides >= 3 else { return }
        let center = p.reduce(Vec3.zero, +) / Double(n)
        var axis = Vec3.zero
        for i in 0..<n { axis += cross(p[i] - center, p[(i + 1) % n] - center) }
        guard length(axis) > 1e-12 else { return }
        axis /= length(axis)
        for i in 0..<n {
            var out = p[i] - center
            out -= dot(out, axis) * axis
            out /= max(length(out), 1e-12)
            let middle = p[i] + radius * out
            for s in 0..<sides {
                let a = 2 * Double.pi * Double(s) / Double(sides)
                let dir = cos(a) * out + sin(a) * axis
                vertices.append(middle + radius * dir)
                normals.append(dir)
            }
        }
        for i in 0..<n {
            let j = (i + 1) % n
            for s in 0..<sides {
                let s2 = (s + 1) % sides
                let a = UInt32(i * sides + s), b = UInt32(j * sides + s)
                let c = UInt32(j * sides + s2), d = UInt32(i * sides + s2)
                indices += [a, b, c, a, c, d]
            }
        }
    }
}
