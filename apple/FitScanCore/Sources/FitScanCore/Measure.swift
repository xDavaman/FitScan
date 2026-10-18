import Foundation

/// Maße wie mit dem Maßband, wie `fitscan.measure`.
public enum MeasureKey: String, CaseIterable, Codable, Identifiable {
    case height, weight, chest, waist, hips, thigh, calf, upperarm, neck, inseam

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .height: return "Körpergröße"
        case .weight: return "Gewicht"
        case .chest: return "Brustumfang"
        case .waist: return "Taillenumfang"
        case .hips: return "Hüftumfang"
        case .thigh: return "Oberschenkelumfang"
        case .calf: return "Wadenumfang"
        case .upperarm: return "Oberarmumfang"
        case .neck: return "Halsumfang"
        case .inseam: return "Schrittlänge"
        }
    }

    public var unit: String { self == .weight ? "kg" : "cm" }

    public static let circumferences: [MeasureKey] = [.chest, .waist, .hips, .thigh, .calf, .upperarm, .neck]
}

/// Damit `[MeasureKey: Double]` als JSON-Objekt gespeichert wird ({"height": 181, …})
extension MeasureKey: CodingKeyRepresentable {}

/// kg/m³, wie in Annys Anthropometrie
public let bodyDensity = 980.0

/// Schnitt durch das Netz, auf die konvexe Hülle verkürzt (das Maßband überspannt Dellen).
/// Jeder Punkt liegt auf einer Kante: a + t · (b − a).
struct Ring {
    var a: [Int]
    var b: [Int]
    var t: [Double]

    func points(_ verts: [Vec3]) -> [Vec3] {
        (0..<a.count).map { verts[a[$0]] + t[$0] * (verts[b[$0]] - verts[a[$0]]) }
    }

    func length(_ verts: [Vec3]) -> Double {
        let p = points(verts)
        var sum = 0.0
        for i in 0..<p.count { sum += FitScanCore.length(p[i] - p[(i + p.count - 1) % p.count]) }
        return sum
    }
}

typealias Edge = (Int, Int)

func uniqueEdges(_ faces: [UInt32], mask: [Bool]? = nil) -> [Edge] {
    var seen = Set<Int>()
    var edges: [Edge] = []
    let nf = faces.count / 3
    for f in 0..<nf {
        let tri = [Int(faces[3 * f]), Int(faces[3 * f + 1]), Int(faces[3 * f + 2])]
        if let mask, !(mask[tri[0]] && mask[tri[1]] && mask[tri[2]]) { continue }
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let lo = min(tri[i], tri[j]), hi = max(tri[i], tri[j])
            if seen.insert(lo &* 1_000_003 &+ hi).inserted { edges.append((lo, hi)) }
        }
    }
    edges.sort { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }
    return edges
}

func planeBasis(_ normal: Vec3) -> (Vec3, Vec3) {
    let helper = abs(normal.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0)
    var u = cross(normal, helper)
    u /= length(u)
    return (u, cross(normal, u))
}

/// Konvexe Hülle in 2D (Andrews Verfahren), Indizes gegen den Uhrzeigersinn, ohne Punkte auf Kanten.
func convexHull(_ p: [(Double, Double)]) -> [Int] {
    let order = p.indices.sorted { p[$0].0 != p[$1].0 ? p[$0].0 < p[$1].0 : p[$0].1 < p[$1].1 }
    if order.count < 3 { return [] }
    func turn(_ o: Int, _ a: Int, _ b: Int) -> Double {
        (p[a].0 - p[o].0) * (p[b].1 - p[o].1) - (p[a].1 - p[o].1) * (p[b].0 - p[o].0)
    }
    var lower: [Int] = []
    for i in order {
        while lower.count >= 2 && turn(lower[lower.count - 2], lower[lower.count - 1], i) <= 0 { lower.removeLast() }
        lower.append(i)
    }
    var upper: [Int] = []
    for i in order.reversed() {
        while upper.count >= 2 && turn(upper[upper.count - 2], upper[upper.count - 1], i) <= 0 { upper.removeLast() }
        upper.append(i)
    }
    let hull = Array(lower.dropLast()) + Array(upper.dropLast())
    return hull.count >= 3 ? hull : []
}

func sectionRing(_ verts: [Vec3], _ edges: [Edge], origin: Vec3, normal rawNormal: Vec3,
                 near: Vec3? = nil, maxDist: Double? = nil) -> Ring? {
    let normal = rawNormal / length(rawNormal)
    var a: [Int] = [], b: [Int] = [], t: [Double] = [], pts: [Vec3] = []
    for (i, j) in edges {
        let da = dot(verts[i] - origin, normal)
        let db = dot(verts[j] - origin, normal)
        guard da * db < 0 else { continue }
        let tt = da / (da - db)
        let p = verts[i] + tt * (verts[j] - verts[i])
        if let near, let maxDist, length(p - near) >= maxDist { continue }
        a.append(i)
        b.append(j)
        t.append(tt)
        pts.append(p)
    }
    if pts.count < 3 { return nil }
    let (u, v) = planeBasis(normal)
    let flat = pts.map { (dot($0 - origin, u), dot($0 - origin, v)) }
    let hull = convexHull(flat)
    if hull.isEmpty { return nil }
    return Ring(a: hull.map { a[$0] }, b: hull.map { b[$0] }, t: hull.map { t[$0] })
}

func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
    n == 1 ? [a] : (0..<n).map { a + (b - a) * Double($0) / Double(n - 1) }
}

/// Legt fest, wo gemessen wird (wie `MeasureProbe`), und misst dann an beliebigen Punkten.
public struct MeasureProbe {
    let top: Int
    let bottom: Int
    let crotch: Int
    var rings: [MeasureKey: Ring] = [:]
    let faces: [UInt32]

    public init(model: BodyModel, rest: RestBody) {
        let data = model.data
        let verts = rest.vertices
        let z = verts.map(\.z)
        let arm = data.armVertexMask()
        faces = data.faces
        func head(_ name: String) -> Vec3 { rest.boneHeads[data.bone(name)!] }
        func region(_ predicate: @escaping (String) -> Bool) -> [Edge] {
            uniqueEdges(data.faces, mask: data.dominantBoneMask(predicate))
        }

        let torsoEdges = uniqueEdges(data.faces, mask: arm.map { !$0 })
        let thighEdges = region { $0.hasPrefix("upperleg") && $0.hasSuffix(".L") }
        let calfEdges = region { $0.hasPrefix("lowerleg") && $0.hasSuffix(".L") }
        let uarmEdges = region { $0.hasPrefix("upperarm") && $0.hasSuffix(".L") }

        var top = 0, bottom = 0
        for i in 1..<z.count {
            if z[i] > z[top] { top = i }
            if z[i] < z[bottom] { bottom = i }
        }
        self.top = top
        self.bottom = bottom
        let height = z[top] - z[bottom]

        let kneeZ = head("lowerleg01.L").z
        let spineZ = head("spine05").z
        var crotch = -1
        for i in 0..<verts.count where abs(verts[i].x) < 0.004 * height && z[i] > kneeZ && z[i] < spineZ && !arm[i] {
            if crotch < 0 || z[i] < z[crotch] { crotch = i }
        }
        if crotch < 0 {
            crotch = 0
            for i in 1..<z.count where abs(z[i] - kneeZ) < abs(z[crotch] - kneeZ) { crotch = i }
        }
        self.crotch = crotch
        let crotchZ = z[crotch]
        let armpitZ = head("upperarm01.L").z - 0.05 * height
        let span = armpitZ - crotchZ
        let up = Vec3(0, 0, 1)

        func band(_ edges: [Edge], _ z0: Double, _ z1: Double, max pickMax: Bool,
                  near: Vec3? = nil, maxDist: Double? = nil, steps: Int = 14) -> Ring? {
            var best: Ring?
            var bestLength = 0.0
            for zz in linspace(z0, z1, steps) {
                let origin = near.map { Vec3($0.x, $0.y, zz) } ?? Vec3(0, 0, zz)
                guard let ring = sectionRing(verts, edges, origin: origin, normal: up,
                                             near: near == nil ? nil : origin, maxDist: maxDist) else { continue }
                let l = ring.length(verts)
                if best == nil || (pickMax ? l > bestLength : l < bestLength) {
                    best = ring
                    bestLength = l
                }
            }
            return best
        }

        func alongBone(_ edges: [Edge], _ start: Vec3, _ end: Vec3, _ t0: Double, _ t1: Double,
                       max pickMax: Bool, steps: Int = 10, maxDist: Double? = nil) -> Ring? {
            let direction = end - start
            var best: Ring?
            var bestLength = 0.0
            for tt in linspace(t0, t1, steps) {
                let origin = start + tt * direction
                guard let ring = sectionRing(verts, edges, origin: origin, normal: direction,
                                             near: origin, maxDist: maxDist ?? 0.15 * height) else { continue }
                let l = ring.length(verts)
                if best == nil || (pickMax ? l > bestLength : l < bestLength) {
                    best = ring
                    bestLength = l
                }
            }
            return best
        }

        let legLength = crotchZ - z[bottom]
        let thighXY = head("upperleg02.L")
        let calfXY = head("lowerleg02.L")
        let ankleZ = head("foot.L").z
        let found: [MeasureKey: Ring?] = [
            .hips: band(torsoEdges, crotchZ + 0.03 * span, crotchZ + 0.35 * span, max: true),
            .waist: band(torsoEdges, crotchZ + 0.35 * span, crotchZ + 0.75 * span, max: false),
            .chest: band(torsoEdges, crotchZ + 0.70 * span, armpitZ, max: true),
            .thigh: band(thighEdges, crotchZ - 0.15 * legLength, crotchZ - 0.02 * legLength, max: true,
                         near: thighXY, maxDist: 0.2 * height),
            .calf: band(calfEdges, ankleZ + 0.3 * (kneeZ - ankleZ), kneeZ - 0.05 * (kneeZ - ankleZ), max: true,
                        near: calfXY, maxDist: 0.15 * height),
            .upperarm: alongBone(uarmEdges, head("upperarm01.L"), head("lowerarm01.L"), 0.3, 0.7, max: true),
            .neck: alongBone(torsoEdges, head("neck01"), head("head"), 0.3, 0.8, max: false,
                             maxDist: 0.08 * height),
        ]
        for (key, ring) in found {
            if let ring { rings[key] = ring }
        }
    }

    static func volume(_ verts: [Vec3], _ faces: [UInt32]) -> Double {
        var sum = 0.0
        for f in 0..<(faces.count / 3) {
            let v0 = verts[Int(faces[3 * f])], v1 = verts[Int(faces[3 * f + 1])], v2 = verts[Int(faces[3 * f + 2])]
            sum += dot(cross(v0, v1), v2)
        }
        return sum / 6
    }

    /// Maße in cm bzw. kg an den Punkten `verts` (Meter).
    public func evaluate(_ verts: [Vec3]) -> [MeasureKey: Double] {
        var out: [MeasureKey: Double] = [:]
        out[.height] = (verts[top].z - verts[bottom].z) * 100
        out[.inseam] = (verts[crotch].z - verts[bottom].z) * 100
        out[.weight] = abs(Self.volume(verts, faces)) * bodyDensity
        for (key, ring) in rings { out[key] = ring.length(verts) * 100 }
        return out
    }

    /// Ableitung eines Maßes nach den Punkten, aufaddiert mit Faktor `scale` in `g`.
    func accumulateGradient(_ key: MeasureKey, _ verts: [Vec3], scale: Double, into g: inout [Vec3]) {
        switch key {
        case .height:
            g[top].z += 100 * scale
            g[bottom].z -= 100 * scale
        case .inseam:
            g[crotch].z += 100 * scale
            g[bottom].z -= 100 * scale
        case .weight:
            let sign: Double = Self.volume(verts, faces) >= 0 ? 1 : -1
            let k = sign * bodyDensity / 6 * scale
            for f in 0..<(faces.count / 3) {
                let i0 = Int(faces[3 * f]), i1 = Int(faces[3 * f + 1]), i2 = Int(faces[3 * f + 2])
                let v0 = verts[i0], v1 = verts[i1], v2 = verts[i2]
                g[i0] += k * cross(v1, v2)
                g[i1] += k * cross(v2, v0)
                g[i2] += k * cross(v0, v1)
            }
        default:
            guard let ring = rings[key] else { return }
            let p = ring.points(verts)
            let n = p.count
            var dp = [Vec3](repeating: .zero, count: n)
            for i in 0..<n {
                let prev = (i + n - 1) % n
                let e = p[i] - p[prev]
                let l = length(e)
                guard l > 0 else { continue }
                let u = e / l
                dp[i] += u
                dp[prev] -= u
            }
            for i in 0..<n {
                let w = 100 * scale * dp[i]
                g[ring.a[i]] += (1 - ring.t[i]) * w
                g[ring.b[i]] += ring.t[i] * w
            }
        }
    }
}

public extension BodyModel {
    /// Maße einer Person in cm bzw. kg, auf 0,1 gerundet (wie `measure_params`).
    func measure(_ params: BodyParams) -> [MeasureKey: Double] {
        let r = rest(params)
        let probe = MeasureProbe(model: self, rest: r)
        return probe.evaluate(r.vertices).mapValues { ($0 * 10).rounded() / 10 }
    }
}
