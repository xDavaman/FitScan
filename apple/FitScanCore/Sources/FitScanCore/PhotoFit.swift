import Foundation

// Foto- und Videoscan: Körperform aus Umrissen und Körperpunkten weniger Bilder.
// Rechnet wie `fitscan.photofit` am PC (dort mit PyTorch geprüft), Gradienten von Hand abgeleitet.

// MARK: - Einstellungen

/// Einstellungen des Foto- und Videoscans aus `fotoscan.json` (geschrieben von
/// `scripts/export_swift_data.py`, gleiche Werte wie `fitscan.photofit`).
public struct PhotoScanSettings: Decodable {
    public struct Keypoint: Decodable {
        public let bone: String
        public let offset: [Double]
    }

    public let workSide: Int
    public let contourPoints: Int
    public let poseBones: [String]
    public let videoLocals: [String]
    public let keypoints: [String: Keypoint]
    public let silhouetteWeight: Double
    public let localReg: Double
    public let poseReg: Double
    public let keypointWeightShape: Double
    public let muscleStarts: [Double]
    public let learningRate: Double
    public let probeEvery: Int
    public let steps: [Int]
    /// Bildwinkel über die lange Seite, wenn das Bild keine Brennweite mitbringt
    public let fovLongSideDeg: Double

    enum CodingKeys: String, CodingKey {
        case workSide = "work_side", contourPoints = "contour_points", poseBones = "pose_bones"
        case videoLocals = "video_locals", keypoints, silhouetteWeight = "silhouette_weight"
        case localReg = "local_reg", poseReg = "pose_reg", keypointWeightShape = "keypoint_weight_shape"
        case muscleStarts = "muscle_starts", learningRate = "learning_rate", probeEvery = "probe_every"
        case steps, fovLongSideDeg = "fov_long_side_deg"
    }

    public static func bundled() throws -> PhotoScanSettings {
        guard let url = Bundle.module.url(forResource: "fotoscan", withExtension: "json") else {
            throw FitScanError.data("Die Einstellungen des Fotoscans fehlen in der App (fotoscan.json).")
        }
        return try JSONDecoder().decode(PhotoScanSettings.self, from: Data(contentsOf: url))
    }
}

// MARK: - Eingabe

/// Ein Bild für den Scan (wie `fit.FitFrame`): Personenmaske, Kamera und Körperpunkte, alles in
/// den Pixeln der Maske.
public struct PhotoFrame {
    public var width: Int
    public var height: Int
    /// Maske zeilenweise, 0…1 (1 = Person)
    public var mask: [Float]
    /// Brennweite und Bildmitte in Pixeln
    public var focal: Double
    public var cx: Double
    public var cy: Double
    /// Körperpunkte nach MediaPipe-Nummer (11/12 Schultern, 13/14 Ellbogen, 15/16 Handgelenke,
    /// 23/24 Hüften, 25/26 Knie, 27/28 Knöchel; ungerade = links): x, y in Pixeln, Sicherheit 0…1
    public var keypoints: [Int: SIMD3<Double>]
    /// Ungefähre Drehung der Person im Bogenmaß, falls bekannt (0 = schaut in die Kamera)
    public var yawHint: Double?
    /// false: die Punkte liegen genau auf Annys Gelenken (künstliche Testbilder)
    public var calibrated: Bool

    public init(width: Int, height: Int, mask: [Float], focal: Double, cx: Double? = nil, cy: Double? = nil,
                keypoints: [Int: SIMD3<Double>] = [:], yawHint: Double? = nil, calibrated: Bool = true) {
        self.width = width
        self.height = height
        self.mask = mask
        self.focal = focal
        self.cx = cx ?? Double(width) / 2
        self.cy = cy ?? Double(height) / 2
        self.keypoints = keypoints
        self.yawHint = yawHint
        self.calibrated = calibrated
    }

    /// Brennweite in Pixeln wie `Intrinsics.phone_default`: fester Bildwinkel über die lange Seite.
    public static func defaultFocal(width: Int, height: Int, fovLongSideDeg: Double) -> Double {
        0.5 * Double(max(width, height)) / tan(fovLongSideDeg * .pi / 360)
    }
}

/// Wie gut das Modell auf einem Bild liegt.
public struct PhotoViewResult {
    /// Drehung der Person auf dem Bild (0° = schaut in die Kamera, 90° = rechte Seite sichtbar)
    public let yawDegrees: Double
    /// Überdeckung von Maske und Modell (Schnitt durch Vereinigung)
    public let iou: Double
    /// Kontrollbild in Arbeitsauflösung, RGB zeilenweise: grün = passt, rot = nur im Bild
    /// (Modell zu schmal), blau = nur im Modell
    public let overlay: [UInt8]
    public let width: Int
    public let height: Int
}

public struct PhotoFitResult {
    public var params: BodyParams
    public var measures: [MeasureKey: Double]
    public var views: [PhotoViewResult]
}

// MARK: - Bild vorbereiten

/// Ein vorbereitetes Bild in Arbeitsauflösung (öffentlich für die Tests).
public struct PreparedView {
    public let width: Int
    public let height: Int
    public let focal: Double
    public let cx: Double
    public let cy: Double
    public let mask: [Bool]
    /// Abstand außerhalb der Maske in Pixeln, innen 0
    public let dt: [Double]
    public let contour: [SIMD2<Double>]
    public let pxHeight: Double
    public let bboxCenter: SIMD2<Double>
    public let kpBones: [Int]
    public let kpXY: [SIMD2<Double>]
    public let kpW: [Double]
    /// Versatz Punkt → Gelenk als Anteil der Körpergröße
    public let kpOff: [Vec3]
    public let yawHint: Double?
}

/// Maskenwerkzeuge wie `photofit.resize_mask`, `largest_component`, `distance_outside`, `boundary_points`.
public enum MaskTools {
    /// Flächenmittel über ganzzahlige Quellbereiche, dann Schwelle 0,5.
    public static func resize(_ mask: [Float], width w: Int, height h: Int, toWidth ow: Int, height oh: Int) -> [Bool] {
        let stride = w + 1
        var ii = [Double](repeating: 0, count: (h + 1) * stride)
        for y in 0..<h {
            var row = 0.0
            for x in 0..<w {
                row += Double(mask[y * w + x])
                ii[(y + 1) * stride + x + 1] = ii[y * stride + x + 1] + row
            }
        }
        var out = [Bool](repeating: false, count: ow * oh)
        for oy in 0..<oh {
            let y0 = oy * h / oh
            let y1 = max(y0 + 1, (oy + 1) * h / oh)
            for ox in 0..<ow {
                let x0 = ox * w / ow
                let x1 = max(x0 + 1, (ox + 1) * w / ow)
                let s = ii[y1 * stride + x1] - ii[y0 * stride + x1] - ii[y1 * stride + x0] + ii[y0 * stride + x0]
                out[oy * ow + ox] = s / Double((y1 - y0) * (x1 - x0)) > 0.5
            }
        }
        return out
    }

    /// Größte zusammenhängende Fläche (8er-Nachbarschaft); bei Gleichstand die zuerst gefundene.
    public static func largestComponent(_ mask: [Bool], width w: Int, height h: Int) -> [Bool] {
        var label = [Int32](repeating: 0, count: w * h)
        var sizes = [0]
        var stack: [Int] = []
        for first in 0..<(w * h) where mask[first] && label[first] == 0 {
            let id = Int32(sizes.count)
            var count = 0
            label[first] = id
            stack.append(first)
            while let p = stack.popLast() {
                count += 1
                let x = p % w, y = p / w
                for ny in max(y - 1, 0)...min(y + 1, h - 1) {
                    for nx in max(x - 1, 0)...min(x + 1, w - 1) {
                        let q = ny * w + nx
                        if mask[q] && label[q] == 0 {
                            label[q] = id
                            stack.append(q)
                        }
                    }
                }
            }
            sizes.append(count)
        }
        guard sizes.count > 1 else { return [Bool](repeating: false, count: w * h) }
        var best = 1
        for i in 1..<sizes.count where sizes[i] > sizes[best] { best = i }
        let id = Int32(best)
        return label.map { $0 == id }
    }

    /// Exakter euklidischer Abstand zum nächsten Maskenpixel (Felzenszwalb und Huttenlocher), innen 0.
    public static func distanceOutside(_ mask: [Bool], width w: Int, height h: Int) -> [Double] {
        let far = 1e20
        var f = mask.map { $0 ? 0.0 : far }
        var column = [Double](repeating: 0, count: h)
        for x in 0..<w {
            for y in 0..<h { column[y] = f[y * w + x] }
            let d = squaredDistance1D(column)
            for y in 0..<h { f[y * w + x] = d[y] }
        }
        var row = [Double](repeating: 0, count: w)
        for y in 0..<h {
            for x in 0..<w { row[x] = f[y * w + x] }
            let d = squaredDistance1D(row)
            for x in 0..<w { f[y * w + x] = d[x].squareRoot() }
        }
        return f
    }

    /// Untere Hüllkurve der Parabeln: quadrierter Abstand in einer Zeile oder Spalte.
    public static func squaredDistance1D(_ f: [Double]) -> [Double] {
        let n = f.count
        guard n > 0 else { return [] }
        var d = [Double](repeating: 0, count: n)
        var v = [Int](repeating: 0, count: n)
        var z = [Double](repeating: 0, count: n + 1)
        var k = 0
        z[0] = -.infinity
        z[1] = .infinity
        func meet(_ q: Int, _ p: Int) -> Double {
            ((f[q] + Double(q * q)) - (f[p] + Double(p * p))) / Double(2 * q - 2 * p)
        }
        for q in stride(from: 1, to: n, by: 1) {
            var s = meet(q, v[k])
            while s <= z[k] {
                k -= 1
                s = meet(q, v[k])
            }
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = .infinity
        }
        k = 0
        for q in 0..<n {
            while z[k + 1] < Double(q) { k += 1 }
            let dq = Double(q - v[k])
            d[q] = dq * dq + f[v[k]]
        }
        return d
    }

    /// Randpixel (ein 4er-Nachbar liegt außen oder am Bildrand), gleichmäßig ausgedünnt.
    public static func boundaryPoints(_ mask: [Bool], width w: Int, height h: Int, count: Int) -> [SIMD2<Double>] {
        func at(_ x: Int, _ y: Int) -> Bool { x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x] }
        var points: [SIMD2<Double>] = []
        for y in 0..<h {
            for x in 0..<w where mask[y * w + x] {
                if !(at(x - 1, y) && at(x + 1, y) && at(x, y - 1) && at(x, y + 1)) {
                    points.append(SIMD2(Double(x), Double(y)))
                }
            }
        }
        let n = points.count, m = min(count, n)
        guard m > 1 else { return Array(points.prefix(m)) }
        let step = Double(n - 1) / Double(m - 1)  // wie np.linspace(0, n − 1, m).astype(int)
        return (0..<m).map { points[$0 == m - 1 ? n - 1 : Int(Double($0) * step)] }
    }
}

public extension BodyModel {
    func prepareView(_ frame: PhotoFrame, settings: PhotoScanSettings) throws -> PreparedView {
        let factor = Double(settings.workSide) / Double(max(frame.width, frame.height))
        let w = Int((Double(frame.width) * factor).rounded(.toNearestOrEven))
        let h = Int((Double(frame.height) * factor).rounded(.toNearestOrEven))
        let resized = MaskTools.resize(frame.mask, width: frame.width, height: frame.height, toWidth: w, height: h)
        let mask = MaskTools.largestComponent(resized, width: w, height: h)
        var minX = Int.max, maxX = Int.min, minY = Int.max, maxY = Int.min, count = 0
        for y in 0..<h {
            for x in 0..<w where mask[y * w + x] {
                count += 1
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }
        guard count >= 50, maxY - minY >= 20 else {
            throw FitScanError.input(["Auf einem Bild ist keine Person zu erkennen."])
        }

        var bones: [Int] = [], xy: [SIMD2<Double>] = [], weights: [Double] = [], offsets: [Vec3] = []
        for index in frame.keypoints.keys.sorted() {
            guard let point = frame.keypoints[index], point.z > 0.5,
                  let spec = settings.keypoints[String(index)], let bone = data.bone(spec.bone) else { continue }
            bones.append(bone)
            xy.append(SIMD2(point.x * factor, point.y * factor))
            weights.append(point.z)
            let o = spec.offset
            offsets.append(frame.calibrated && o.count == 3 ? Vec3(o[0], o[1], o[2]) : .zero)
        }
        return PreparedView(
            width: w, height: h, focal: frame.focal * factor, cx: frame.cx * factor, cy: frame.cy * factor,
            mask: mask, dt: MaskTools.distanceOutside(mask, width: w, height: h),
            contour: MaskTools.boundaryPoints(mask, width: w, height: h, count: settings.contourPoints),
            pxHeight: Double(maxY - minY), bboxCenter: SIMD2(Double(minX + maxX) / 2, Double(minY + maxY) / 2),
            kpBones: bones, kpXY: xy, kpW: weights, kpOff: offsets, yawHint: frame.yawHint)
    }
}

// MARK: - Kinematik

/// Drehung um die Achse a (Länge = Winkel), Rodrigues.
func rotationMatrix(_ a: Vec3) -> Mat3 {
    let theta = length(a)
    if theta < 1e-8 { return Mat3([1, -a.z, a.y, a.z, 1, -a.x, -a.y, a.x, 1]) }
    let k = a / theta
    let s = sin(theta), c = 1 - cos(theta)
    return Mat3([
        1 + c * (k.x * k.x - 1), -s * k.z + c * k.x * k.y, s * k.y + c * k.x * k.z,
        s * k.z + c * k.x * k.y, 1 + c * (k.y * k.y - 1), -s * k.x + c * k.y * k.z,
        -s * k.y + c * k.x * k.z, s * k.x + c * k.y * k.z, 1 + c * (k.z * k.z - 1),
    ])
}

/// J mit exp(a + δ) ≈ exp(J δ) · exp(a), wie `photofit.left_jacobian_np`.
func leftJacobian(_ a: Vec3) -> Mat3 {
    let theta = length(a)
    let k = [0, -a.z, a.y, a.z, 0, -a.x, -a.y, a.x, 0]
    let kk = (Mat3(k) * Mat3(k)).m
    let c1: Double, c2: Double
    if theta < 1e-6 {
        c1 = 0.5
        c2 = 0
    } else {
        c1 = (1 - cos(theta)) / (theta * theta)
        c2 = (theta - sin(theta)) / (theta * theta * theta)
    }
    let eye: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return Mat3((0..<9).map { eye[$0] + c1 * k[$0] + c2 * kk[$0] })
}

/// Haltung für die Anpassung: einige Gelenke drehen um ihren Kopf (im Raum der Ruhepose) und
/// nehmen ihre Kinder mit, wie `photofit.Rig`.
public struct PoseRig {
    public let parents: [Int]
    public let poseBones: [Int]
    /// je Knochen: Nummer seines Haltungsgelenks oder −1
    public let slot: [Int]
    /// je Knochen: die Haltungsgelenke, die ihn mitdrehen (er selbst und seine Vorfahren)
    public let movedBy: [[Int]]
    public let indices: [Int]
    public let weights: [Double]
    public let perVertex: Int

    public init(data: AnnyData, bones: [String]) throws {
        parents = data.boneParents
        var pose: [Int] = []
        for name in bones {
            guard let b = data.bone(name) else { throw FitScanError.data("Knochen \(name) fehlt in Annys Daten.") }
            pose.append(b)
        }
        poseBones = pose
        var slot = [Int](repeating: -1, count: data.numBones)
        for (j, b) in pose.enumerated() { slot[b] = j }
        self.slot = slot
        var moved = [[Int]](repeating: [], count: data.numBones)
        for b in 0..<data.numBones {
            var a = b
            while a >= 0 {
                if slot[a] >= 0 { moved[b].append(slot[a]) }
                a = data.boneParents[a]
            }
        }
        movedBy = moved
        indices = data.vertexBoneIndices.map { Int($0) }
        weights = data.vertexBoneWeights.map { Double($0) }
        perVertex = data.bonesPerVertex
    }

    /// Je Knochen: Ruhepunkt → gestellter Punkt.
    public func transforms(heads: [Vec3], pose: [Vec3]) -> [Rigid] {
        var out = [Rigid](repeating: .identity, count: parents.count)
        for b in 0..<parents.count {
            var m = parents[b] < 0 ? Rigid.identity : out[parents[b]]
            let j = slot[b]
            if j >= 0 {
                let ra = rotationMatrix(pose[j])
                let h = heads[b]
                m = Rigid(rotation: m.rotation * ra, translation: m.rotation * (h - ra * h) + m.translation)
            }
            out[b] = m
        }
        return out
    }

    /// Gestellte Punkte und je Punkt die gemischte Drehung (9 Werte, für den Gradienten).
    public func skin(_ verts: [Vec3], _ m: [Rigid]) -> (posed: [Vec3], blend: [Double]) {
        let n = verts.count
        let rot = m.flatMap { $0.rotation.m }
        let trans = m.map(\.translation)
        var posed = [Vec3](repeating: .zero, count: n)
        var blend = [Double](repeating: 0, count: 9 * n)
        for v in 0..<n {
            let base = 9 * v
            var c = Vec3.zero
            for k in 0..<perVertex {
                let w = weights[v * perVertex + k]
                if w == 0 { continue }
                let b = indices[v * perVertex + k]
                for i in 0..<9 { blend[base + i] += w * rot[9 * b + i] }
                c += w * trans[b]
            }
            let p = verts[v]
            posed[v] = Vec3(blend[base] * p.x + blend[base + 1] * p.y + blend[base + 2] * p.z,
                            blend[base + 3] * p.x + blend[base + 4] * p.y + blend[base + 5] * p.z,
                            blend[base + 6] * p.x + blend[base + 7] * p.y + blend[base + 8] * p.z) + c
        }
        return (posed, blend)
    }

    /// Gradient nach den Ruhepunkten und nach der Haltung aus dem Gradienten nach den gestellten
    /// Punkten und den gestellten Körperpunkt-Gelenken, wie `photofit.skin_gradient_np`.
    public func gradient(rest: [Vec3], heads: [Vec3], pose: [Vec3], transforms m: [Rigid], blend: [Double],
                  points g: [Vec3], keypointBones kpBones: [Int], keypoints gk: [Vec3]) -> (rest: [Vec3], pose: [Vec3]) {
        let n = rest.count
        let np = poseBones.count
        let rot = m.flatMap { $0.rotation.m }
        let trans = m.map(\.translation)
        var gRest = [Vec3](repeating: .zero, count: n)
        var torque = [Vec3](repeating: .zero, count: np)  // Σ w (R_b v + t_b) × g
        var pull = [Vec3](repeating: .zero, count: np)  // Σ w g
        for v in 0..<n {
            let gv = g[v]
            if gv == .zero { continue }
            let base = 9 * v
            gRest[v] = Vec3(blend[base] * gv.x + blend[base + 3] * gv.y + blend[base + 6] * gv.z,
                            blend[base + 1] * gv.x + blend[base + 4] * gv.y + blend[base + 7] * gv.z,
                            blend[base + 2] * gv.x + blend[base + 5] * gv.y + blend[base + 8] * gv.z)
            let p = rest[v]
            for k in 0..<perVertex {
                let w = weights[v * perVertex + k]
                if w == 0 { continue }
                let b = indices[v * perVertex + k]
                if movedBy[b].isEmpty { continue }
                let r = 9 * b
                let x = Vec3(rot[r] * p.x + rot[r + 1] * p.y + rot[r + 2] * p.z,
                             rot[r + 3] * p.x + rot[r + 4] * p.y + rot[r + 5] * p.z,
                             rot[r + 6] * p.x + rot[r + 7] * p.y + rot[r + 8] * p.z) + trans[b]
                let t = w * cross(x, gv)
                let f = w * gv
                for j in movedBy[b] {
                    torque[j] += t
                    pull[j] += f
                }
            }
        }
        var gPose = [Vec3](repeating: .zero, count: np)
        for (j, b) in poseBones.enumerated() {
            let parent = parents[b] < 0 ? Rigid.identity : m[parents[b]]
            let head = parent.apply(heads[b])  // gestellter Gelenkkopf
            var tq = torque[j] - cross(head, pull[j])
            for (i, kb) in kpBones.enumerated() where movedBy[kb].contains(j) {
                tq += cross(m[kb].apply(heads[kb]) - head, gk[i])
            }
            gPose[j] = leftJacobian(pose[j]).transposed * (parent.rotation.transposed * tq)
        }
        return (gRest, gPose)
    }
}

// MARK: - Bildterme

/// Nächster Nachbar unter den projizierten Modellpunkten (Raster, exakt).
struct PointGrid {
    let xs: [Double]
    let ys: [Double]
    let minX: Double
    let minY: Double
    let cell: Double
    let nx: Int
    let ny: Int
    let start: [Int]
    let items: [Int]

    init(xs: [Double], ys: [Double], count: Int, cell: Double = 8) {
        self.xs = xs
        self.ys = ys
        var lx = Double.infinity, hx = -Double.infinity, ly = Double.infinity, hy = -Double.infinity
        for i in 0..<count {
            lx = min(lx, xs[i])
            hx = max(hx, xs[i])
            ly = min(ly, ys[i])
            hy = max(hy, ys[i])
        }
        // höchstens etwa 512 × 512 Zellen, auch wenn Punkte weit aus dem Bild fallen
        let cell = max(cell, max(hx - lx, hy - ly) / 512)
        self.cell = cell
        minX = lx
        minY = ly
        let gx = max(1, Int(((hx - lx) / cell).rounded(.down)) + 1)
        let gy = max(1, Int(((hy - ly) / cell).rounded(.down)) + 1)
        nx = gx
        ny = gy
        var cellOf = [Int](repeating: 0, count: count)
        var counts = [Int](repeating: 0, count: gx * gy + 1)
        for i in 0..<count {
            let cx = min(gx - 1, Int(((xs[i] - lx) / cell).rounded(.down)))
            let cy = min(gy - 1, Int(((ys[i] - ly) / cell).rounded(.down)))
            cellOf[i] = cy * gx + cx
            counts[cellOf[i] + 1] += 1
        }
        for c in 0..<(gx * gy) { counts[c + 1] += counts[c] }
        start = counts
        var fill = counts
        var items = [Int](repeating: 0, count: count)
        for i in 0..<count {
            items[fill[cellOf[i]]] = i
            fill[cellOf[i]] += 1
        }
        self.items = items
    }

    func nearest(_ qx: Double, _ qy: Double) -> (index: Int, distance2: Double) {
        let cx = Int(((qx - minX) / cell).rounded(.down))
        let cy = Int(((qy - minY) / cell).rounded(.down))
        var best = -1
        var bestD = Double.infinity
        func visit(_ gx: Int, _ gy: Int) {
            let c = gy * nx + gx
            for e in start[c]..<start[c + 1] {
                let i = items[e]
                let dx = xs[i] - qx, dy = ys[i] - qy
                let d = dx * dx + dy * dy
                if d < bestD || (d == bestD && i < best) {
                    bestD = d
                    best = i
                }
            }
        }
        let rMax = max(abs(cx), abs(cx - (nx - 1)), abs(cy), abs(cy - (ny - 1)))
        var r = 0
        while r <= rMax {
            for gy in stride(from: max(cy - r, 0), through: min(cy + r, ny - 1), by: 1) {
                if gy == cy - r || gy == cy + r {
                    for gx in stride(from: max(cx - r, 0), through: min(cx + r, nx - 1), by: 1) { visit(gx, gy) }
                } else {
                    if cx - r >= 0 && cx - r < nx { visit(cx - r, gy) }
                    if r > 0 && cx + r >= 0 && cx + r < nx { visit(cx + r, gy) }
                }
            }
            // Punkte in weiteren Ringen sind mindestens r Zellen entfernt
            if best >= 0 && bestD <= (Double(r) * cell) * (Double(r) * cell) { break }
            r += 1
        }
        return (best, bestD)
    }
}

public struct ViewTerms {
    public var loss: Double
    public var points: [Vec3]
    public var keypoints: [Vec3]
    public var yaw: Double
    public var trans: Vec3
}

/// Fehler eines Bildes und seine Gradienten, wie `photofit.view_loss_np`.
public func viewTerms(_ verts: [Vec3], _ kp: [Vec3], yaw: Double, trans: Vec3, view: PreparedView,
               keypointWeight: Double) -> ViewTerms {
    let nv = verts.count, nk = kp.count, n = nv + nk
    let c = cos(yaw), s = sin(yaw)
    let f = view.focal
    func point(_ i: Int) -> Vec3 { i < nv ? verts[i] : kp[i - nv] }
    var cam = [Vec3](repeating: .zero, count: n)
    var u = [Double](repeating: 0, count: n)
    var v = [Double](repeating: 0, count: n)
    for i in 0..<n {
        let p = point(i)
        let q = Vec3(c * p.x - s * p.y, -p.z, s * p.x + c * p.y) + trans
        cam[i] = q
        let z = max(q.z, 1e-3)
        u[i] = f * q.x / z + view.cx
        v[i] = f * q.y / z + view.cy
        if !u[i].isFinite || !v[i].isFinite {
            return ViewTerms(loss: .infinity, points: [Vec3](repeating: .zero, count: nv),
                             keypoints: [Vec3](repeating: .zero, count: nk), yaw: 0, trans: .zero)
        }
    }
    let h = view.pxHeight
    var gu = [Double](repeating: 0, count: n)
    var gv = [Double](repeating: 0, count: n)

    // Modellpunkte außerhalb der Maske
    let w = view.width, hh = view.height
    var sumIn = 0.0
    for i in 0..<nv {
        let x = min(max(u[i], 0), Double(w - 1))
        let y = min(max(v[i], 0), Double(hh - 1))
        let x0 = min(Int(x.rounded(.down)), w - 2)
        let y0 = min(Int(y.rounded(.down)), hh - 2)
        let fx = x - Double(x0), fy = y - Double(y0)
        let d00 = view.dt[y0 * w + x0], d01 = view.dt[y0 * w + x0 + 1]
        let d10 = view.dt[(y0 + 1) * w + x0], d11 = view.dt[(y0 + 1) * w + x0 + 1]
        let d = (1 - fx) * (1 - fy) * d00 + fx * (1 - fy) * d01 + (1 - fx) * fy * d10 + fx * fy * d11
        sumIn += (d / h) * (d / h)
        if d != 0 {
            let dd = 8 * d / (h * h * Double(nv))
            if u[i] >= 0 && u[i] <= Double(w - 1) { gu[i] += dd * ((1 - fy) * (d01 - d00) + fy * (d11 - d10)) }
            if v[i] >= 0 && v[i] <= Double(hh - 1) { gv[i] += dd * ((1 - fx) * (d10 - d00) + fx * (d11 - d01)) }
        }
    }
    var loss = 4 * sumIn / Double(nv)

    // Randpunkte der Maske zum nächsten Modellpunkt
    if !view.contour.isEmpty {
        let grid = PointGrid(xs: u, ys: v, count: nv)
        let gc = 8 / (h * h * Double(view.contour.count))
        var sumCov = 0.0
        for p in view.contour {
            let (j, d2) = grid.nearest(p.x, p.y)
            sumCov += d2 / (h * h)
            gu[j] += gc * (u[j] - p.x)
            gv[j] += gc * (v[j] - p.y)
        }
        loss += 4 * sumCov / Double(view.contour.count)
    }

    // Körperpunkte
    if nk > 0 {
        var sumK = 0.0
        for i in 0..<nk {
            let du = u[nv + i] - view.kpXY[i].x, dv = v[nv + i] - view.kpXY[i].y
            sumK += view.kpW[i] * (du * du + dv * dv)
            let g = keypointWeight * 2 * view.kpW[i] / (h * h * Double(nk))
            gu[nv + i] += g * du
            gv[nv + i] += g * dv
        }
        loss += keypointWeight * sumK / Double(nk) / (h * h)
    }

    // Projektion → Kamera → Welt
    var gPoints = [Vec3](repeating: .zero, count: n)
    var gYaw = 0.0, gTrans = Vec3.zero
    for i in 0..<n where gu[i] != 0 || gv[i] != 0 {
        let q = cam[i]
        let z = max(q.z, 1e-3)
        let gz = q.z < 1e-3 ? 0 : -(gu[i] * f * q.x + gv[i] * f * q.y) / (z * z)
        let gq = Vec3(gu[i] * f / z, gv[i] * f / z, gz)
        gTrans += gq
        let p = point(i)
        gYaw += gq.x * (-s * p.x - c * p.y) + gq.z * (c * p.x - s * p.y)
        let gr = Vec3(gq.x, gq.z, -gq.y)  // Pᵀ g mit P(x, y, z) = (x, −z, y)
        gPoints[i] = Vec3(c * gr.x + s * gr.y, -s * gr.x + c * gr.y, gr.z)
    }
    return ViewTerms(loss: loss, points: Array(gPoints[0..<nv]), keypoints: Array(gPoints[nv..<n]),
                     yaw: gYaw, trans: gTrans)
}

/// Verschiebung, bei der das Modell bei dieser Drehung so groß ist und so liegt wie im Bild.
public func alignTranslation(_ verts: [Vec3], yaw: Double, view: PreparedView) -> Vec3 {
    let c = cos(yaw), s = sin(yaw)
    let cam = verts.map { p in Vec3(c * p.x - s * p.y, -p.z, s * p.x + c * p.y) }
    let zMean = cam.reduce(0) { $0 + $1.z } / Double(cam.count)
    var lowY = Double.infinity, highY = -Double.infinity
    for q in cam {
        lowY = min(lowY, q.y)
        highY = max(highY, q.y)
    }
    let f = view.focal
    var t = Vec3(0, 0, f * (highY - lowY) / view.pxHeight - zMean)
    for _ in 0..<3 {
        var lu = Double.infinity, hu = -Double.infinity, lv = Double.infinity, hv = -Double.infinity
        for q0 in cam {
            let q = q0 + t
            let z = max(q.z, 1e-3)
            let u = f * q.x / z + view.cx, v = f * q.y / z + view.cy
            lu = min(lu, u)
            hu = max(hu, u)
            lv = min(lv, v)
            hv = max(hv, v)
        }
        let z = zMean + t.z
        t.z = (t.z + zMean) * (hv - lv) / view.pxHeight - zMean
        t.x += (view.bboxCenter.x - (hu + lu) / 2) * z / f
        t.y += (view.bboxCenter.y - (hv + lv) / 2) * z / f
    }
    return t
}

/// Drehung und Lage eines Bildes: 12 Kandidaten (plus Hinweis) prüfen, die besten 4 nachschärfen.
func initialView(_ verts: [Vec3], _ kp: [Vec3], view: PreparedView, learningRate: Double,
                 refineSteps: Int = 25) -> (yaw: Double, trans: Vec3) {
    var candidates = (0..<12).map { Double($0) * .pi / 6 }
    if let y = view.yawHint { candidates += [y, -y, y + .pi, .pi - y] }
    var scored = candidates.map { yaw -> (loss: Double, yaw: Double, trans: Vec3) in
        let t = alignTranslation(verts, yaw: yaw, view: view)
        return (viewTerms(verts, kp, yaw: yaw, trans: t, view: view, keypointWeight: 1).loss, yaw, t)
    }
    scored.sort { $0.loss < $1.loss }
    var best: (loss: Double, yaw: Double, trans: Vec3)?
    for candidate in scored.prefix(4) {
        var x = [candidate.yaw, candidate.trans.x, candidate.trans.y, candidate.trans.z]
        var adam = Adam(count: 4, lr: learningRate)
        for _ in 0..<refineSteps {
            let g = viewTerms(verts, kp, yaw: x[0], trans: Vec3(x[1], x[2], x[3]), view: view, keypointWeight: 1)
            adam.step(&x, [g.yaw, g.trans.x, g.trans.y, g.trans.z])
        }
        let final = viewTerms(verts, kp, yaw: x[0], trans: Vec3(x[1], x[2], x[3]), view: view, keypointWeight: 1).loss
        if best == nil || final < best!.loss { best = (final, x[0], Vec3(x[1], x[2], x[3])) }
    }
    return (best?.yaw ?? 0, best?.trans ?? .zero)
}

/// Modell als Maske (Dreiecke, Pixelmitten), für Überdeckung und Kontrollbild.
func renderMask(_ verts: [Vec3], faces: [UInt32], yaw: Double, trans: Vec3, view: PreparedView) -> [Bool] {
    let w = view.width, h = view.height
    let c = cos(yaw), s = sin(yaw)
    let uv: [SIMD2<Double>] = verts.map { p in
        let q = Vec3(c * p.x - s * p.y, -p.z, s * p.x + c * p.y) + trans
        let z = max(q.z, 1e-3)
        return SIMD2(view.focal * q.x / z + view.cx, view.focal * q.y / z + view.cy)
    }
    var out = [Bool](repeating: false, count: w * h)
    for f in 0..<(faces.count / 3) {
        let a = uv[Int(faces[3 * f])], b = uv[Int(faces[3 * f + 1])], d = uv[Int(faces[3 * f + 2])]
        let area = (b.x - a.x) * (d.y - a.y) - (b.y - a.y) * (d.x - a.x)
        if area == 0 { continue }
        let x0 = max(0, Int(min(a.x, b.x, d.x).rounded(.up))), x1 = min(w - 1, Int(max(a.x, b.x, d.x).rounded(.down)))
        let y0 = max(0, Int(min(a.y, b.y, d.y).rounded(.up))), y1 = min(h - 1, Int(max(a.y, b.y, d.y).rounded(.down)))
        if x0 > x1 || y0 > y1 { continue }
        for y in y0...y1 {
            let py = Double(y)
            for x in x0...x1 {
                let px = Double(x)
                let e0 = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
                let e1 = (d.x - b.x) * (py - b.y) - (d.y - b.y) * (px - b.x)
                let e2 = (a.x - d.x) * (py - d.y) - (a.y - d.y) * (px - d.x)
                if area > 0 ? (e0 >= 0 && e1 >= 0 && e2 >= 0) : (e0 <= 0 && e1 <= 0 && e2 <= 0) {
                    out[y * w + x] = true
                }
            }
        }
    }
    return out
}

// MARK: - Anpassung

public extension BodyModel {
    /// Passt Form und Haltung an Umrisse und Körperpunkte aus einem oder mehreren Bildern an
    /// (wie `photofit.fit_photos`). Größe und Gewicht sind Pflicht, Maßband-Werte machen es genauer.
    /// `progress` bekommt einen Text und den Anteil 0…1.
    func fitPhotos(_ person: Person, frames: [PhotoFrame], settings: PhotoScanSettings,
                   initial: BodyParams? = nil, progress: ((String, Double) -> Void)? = nil) throws -> PhotoFitResult {
        guard !frames.isEmpty else { throw FitScanError.input(["Keine brauchbaren Bilder für die Anpassung."]) }
        let problems = person.validate()
        if !problems.isEmpty { throw FitScanError.input(problems) }
        let targets = person.measures.filter { $0.value != 0 }
        progress?("Startform aus deinen Maßen", 0)
        var start = try initial ?? fitToMeasurements(person).params
        start.offsets = nil
        let space = ShapeSpace(model: self, person: person, initial: start, extraLocals: settings.videoLocals)
        var x = space.start(start)
        let views = try frames.map { try prepareView($0, settings: settings) }
        let rig = try PoseRig(data: data, bones: settings.poseBones)
        let k = views.count, np = rig.poseBones.count
        let lr = settings.learningRate
        let s0 = max(1, settings.steps.first ?? 80), s1 = max(1, settings.steps.count > 1 ? settings.steps[1] : 160)
        let trial = max(1, s1 / 3)
        let probeEvery = max(1, settings.probeEvery)
        let starts = uniqueMuscleStarts(sigmoid(x[space.free.firstIndex(of: "muscle") ?? 0]), settings.muscleStarts)
        let totalSteps = Double(k * 4 + s0 + trial * starts.count + (s1 - trial))
        var done = 0.0

        struct State {
            let params: BodyParams
            let rest: RestBody
            let transforms: [Rigid]
            let posed: [Vec3]
            let blend: [Double]
            let height: Double
        }
        func body(_ x: [Double], _ pose: [Vec3]) -> State {
            let p = space.params(x)
            let r = rest(p)
            let m = rig.transforms(heads: r.boneHeads, pose: pose)
            let (posed, blend) = rig.skin(r.vertices, m)
            var low = Double.infinity, high = -Double.infinity
            for q in r.vertices {
                low = min(low, q.z)
                high = max(high, q.z)
            }
            return State(params: p, rest: r, transforms: m, posed: posed, blend: blend, height: high - low)
        }
        func keypoints(_ view: PreparedView, _ st: State) -> [Vec3] {
            view.kpBones.enumerated().map { i, b in
                st.transforms[b].apply(st.rest.boneHeads[b]) + view.kpOff[i] * st.height
            }
        }

        // Startwerte für Drehung und Verschiebung je Bild
        var pose = [Vec3](repeating: .zero, count: np)
        let first = body(x, pose)
        var yaws: [Double] = [], trans: [Vec3] = []
        for (i, view) in views.enumerated() {
            let (y, t) = initialView(first.posed, keypoints(view, first), view: view, learningRate: lr)
            yaws.append(y)
            trans.append(t)
            done += 4
            progress?("Bild \(i + 1) von \(k) ausgerichtet", done / totalSteps)
        }

        // Ansichtsgrößen als ein Vektor: [Drehungen, Verschiebungen, Haltung]
        func packView() -> [Double] {
            yaws + trans.flatMap { [$0.x, $0.y, $0.z] } + pose.flatMap { [$0.x, $0.y, $0.z] }
        }
        func unpackView(_ a: [Double]) {
            for i in 0..<k {
                yaws[i] = a[i]
                trans[i] = Vec3(a[k + 3 * i], a[k + 3 * i + 1], a[k + 3 * i + 2])
            }
            for j in 0..<np { pose[j] = Vec3(a[4 * k + 3 * j], a[4 * k + 3 * j + 1], a[4 * k + 3 * j + 2]) }
        }

        func runStage(_ name: String, steps: Int, shape: Bool, keypointWeight: Double) throws -> Double {
            var viewAdam = Adam(count: 4 * k + 3 * np, lr: lr)
            var shapeAdam = Adam(count: x.count, lr: lr)
            var probe: MeasureProbe?
            var last = Double.infinity
            let silhouette = settings.silhouetteWeight / Double(k)
            for step in 0..<steps {
                let st = body(x, pose)
                if step % probeEvery == 0 || probe == nil { probe = MeasureProbe(model: self, rest: st.rest) }
                var loss = 0.0
                var gPosed = [Vec3](repeating: .zero, count: numVertices)
                var gView = [Double](repeating: 0, count: 4 * k + 3 * np)
                var kpBones: [Int] = [], gKp: [Vec3] = []
                for (i, view) in views.enumerated() {
                    let terms = viewTerms(st.posed, keypoints(view, st), yaw: yaws[i], trans: trans[i], view: view,
                                          keypointWeight: keypointWeight)
                    loss += silhouette * terms.loss
                    for vi in 0..<numVertices where terms.points[vi] != .zero { gPosed[vi] += silhouette * terms.points[vi] }
                    kpBones += view.kpBones
                    gKp += terms.keypoints.map { silhouette * $0 }
                    gView[i] = silhouette * terms.yaw
                    gView[k + 3 * i] = silhouette * terms.trans.x
                    gView[k + 3 * i + 1] = silhouette * terms.trans.y
                    gView[k + 3 * i + 2] = silhouette * terms.trans.z
                }
                let (gRest, gPose) = rig.gradient(rest: st.rest.vertices, heads: st.rest.boneHeads, pose: pose,
                                                  transforms: st.transforms, blend: st.blend, points: gPosed,
                                                  keypointBones: kpBones, keypoints: gKp)
                for j in 0..<np {
                    let g = gPose[j] + 2 * settings.poseReg * pose[j]
                    gView[4 * k + 3 * j] = g.x
                    gView[4 * k + 3 * j + 1] = g.y
                    gView[4 * k + 3 * j + 2] = g.z
                    loss += settings.poseReg * dot(pose[j], pose[j])
                }

                // Maße und Regler
                var g = gRest
                if let current = probe {
                    let values = current.evaluate(st.rest.vertices)
                    for (key, target) in targets {
                        guard let value = values[key] else { continue }
                        let rel = (value - target) / target
                        loss += 0.5 * measureWeight(key) * rel * rel
                        if shape {
                            current.accumulateGradient(key, st.rest.vertices,
                                                       scale: measureWeight(key) * (value - target) / (target * target),
                                                       into: &g)
                        }
                    }
                }
                for i in 0..<space.nl {
                    let l = tanh(x[space.nf + i])
                    loss += settings.localReg * l * l
                }
                if shape {
                    var gx = space.gradient(x, params: st.params, rest: st.rest, pointGradient: g)
                    for i in 0..<space.nl {
                        let l = tanh(x[space.nf + i])
                        gx[space.nf + i] += 2 * settings.localReg * l * (1 - l * l)
                    }
                    shapeAdam.step(&x, gx)
                }
                guard loss.isFinite else {
                    throw FitScanError.input(["Die Anpassung an die Bilder ist fehlgeschlagen. Bitte andere Bilder wählen."])
                }
                var a = packView()
                viewAdam.step(&a, gView)
                unpackView(a)
                last = loss
                done += 1
                if step % 10 == 0 { progress?(name, min(done / totalSteps, 1)) }
            }
            return last
        }

        _ = try runStage("Lage und Haltung", steps: s0, shape: false, keypointWeight: 1)

        // Muskeln und Fett sehen bei gleichem Gewicht ähnlich aus: mehrere Startwerte kurz anrechnen
        let muscle = space.free.firstIndex(of: "muscle") ?? 0
        let snapshot = (x, yaws, trans, pose)
        var best: (score: Double, x: [Double], yaws: [Double], trans: [Vec3], pose: [Vec3])?
        for m0 in starts {
            (x, yaws, trans, pose) = snapshot
            x[muscle] = logit(m0)
            let score = try runStage("Körperform", steps: trial, shape: true, keypointWeight: settings.keypointWeightShape)
            if best == nil || score < best!.score { best = (score, x, yaws, trans, pose) }
        }
        if let best { (x, yaws, trans, pose) = (best.x, best.yaws, best.trans, best.pose) }
        _ = try runStage("Körperform", steps: max(0, s1 - trial), shape: true, keypointWeight: settings.keypointWeightShape)

        var params = space.params(x)
        if let h = targets[.height] { params = exactHeight(params, heightCM: h) }

        // Passgenauigkeit je Bild
        progress?("Passgenauigkeit prüfen", 1)
        let final = body(x, pose)
        var results: [PhotoViewResult] = []
        for (i, view) in views.enumerated() {
            let rendered = renderMask(final.posed, faces: faces, yaw: yaws[i], trans: trans[i], view: view)
            var inter = 0, union = 0
            var overlay = [UInt8](repeating: 245, count: 3 * view.width * view.height)
            for p in 0..<(view.width * view.height) {
                let a = rendered[p], b = view.mask[p]
                if a && b { inter += 1 }
                if a || b { union += 1 }
                let color: [UInt8]? = a && b ? [110, 180, 110] : (b ? [220, 60, 60] : (a ? [60, 100, 230] : nil))
                if let color {
                    overlay[3 * p] = color[0]
                    overlay[3 * p + 1] = color[1]
                    overlay[3 * p + 2] = color[2]
                }
            }
            let degrees = (yaws[i] * 180 / .pi).truncatingRemainder(dividingBy: 360)
            results.append(PhotoViewResult(yawDegrees: degrees < 0 ? degrees + 360 : degrees,
                                           iou: Double(inter) / Double(max(union, 1)),
                                           overlay: overlay, width: view.width, height: view.height))
        }
        return PhotoFitResult(params: params, measures: measure(params), views: results)
    }
}

/// Startwerte für den Muskelanteil ohne Doppelte, in dieser Reihenfolge (wie `dict.fromkeys`).
func uniqueMuscleStarts(_ current: Double, _ extra: [Double]) -> [Double] {
    var out: [Double] = []
    for m in [current] + extra where !out.contains(m) { out.append(m) }
    return out
}
