import Foundation

// MARK: - Bewegungen und Mimik (vom PC mitgeliefert)

/// Eine Bewegung der Bibliothek, wie `fitscan.motion.Motion`. Sie passt auf jeden Körper: Die
/// Drehungen stehen in Annys Form „local-ref“ (Weltachsen, gegenüber Annys Bezugshaltung), der
/// Hüftpunkt in Hüfthöhen über dem Boden (Anny-Achsen: Z oben, Blick −Y).
public struct MotionClip: Decodable, Identifiable {
    /// Dateiname am PC, z. B. „winken“
    public let key: String
    /// Name in der App, z. B. „Winken“
    public let label: String
    public let fps: Double
    /// Schleife: Das letzte Bild gleicht dem ersten (beim Gehen einen Doppelschritt weiter vorn)
    public let loop: Bool
    public let bones: [String]
    /// Je Bild: Quaternion x, y, z, w für jeden Knochen in `bones`, hintereinander
    public let rotations: [[Double]]
    /// Je Bild: Hüftpunkt geteilt durch die Hüfthöhe
    public let root: [[Double]]
    /// Mimik: ARKit-Regler (z. B. „jawOpen“) → Wert 0…1 je Bild
    public let face: [String: [Double]]

    public var id: String { key }
    public var frames: Int { root.count }
    public var duration: Double { Double(max(frames - 1, 0)) / fps }
}

/// Die Clips der App (`bewegungen.json`, geschrieben von `swiftdata.clip_library`).
public struct MotionLibrary: Decodable {
    public let clips: [MotionClip]

    public static func bundled() throws -> MotionLibrary {
        guard let url = Bundle.module.url(forResource: "bewegungen", withExtension: "json") else {
            throw FitScanError.data("Die Bewegungen fehlen in der App (bewegungen.json).")
        }
        return try JSONDecoder().decode(MotionLibrary.self, from: Data(contentsOf: url))
    }

    public func clip(_ key: String) -> MotionClip? { clips.first { $0.key == key } }
}

/// Annys 52 Gesichtsregler mit den Namen von Apple ARKit (`mimik.json`, wie `fitscan.face_rig`):
/// je Regler die Verschiebung der Punkte, die er bewegt, in Mikrometern für Annys Grundkopf.
public struct FaceRig: Decodable {
    public struct Action: Decodable {
        public let name: String
        let index: [Int]
        /// 3 Werte je Punkt in `index`, Mikrometer, Anny-Achsen
        let delta: [Int]
    }

    /// Breite (äußere Augenwinkel) und Höhe (Nasenwurzel bis Kinn) von Annys Grundgesicht
    let baseFaceSize: [Double]
    /// Augenwinkel links, rechts, Nasenwurzel, Kinn: Dreieck und drei baryzentrische Gewichte
    let landmarks: [[Double]]
    public let actions: [Action]

    enum CodingKeys: String, CodingKey {
        case baseFaceSize = "base_face_size", landmarks, actions
    }

    public static func bundled() throws -> FaceRig {
        guard let url = Bundle.module.url(forResource: "mimik", withExtension: "json") else {
            throw FitScanError.data("Die Gesichtsregler fehlen in der App (mimik.json).")
        }
        return try JSONDecoder().decode(FaceRig.self, from: Data(contentsOf: url))
    }

    public func action(_ name: String) -> Action? { actions.first { $0.name == name } }

    /// Die Regler sind für Annys Grundkopf gebaut; für die Person werden sie mit der Größe ihres
    /// Gesichts skaliert, wie `face_rig.face_morphs`. `vertices`: Ruhepose der Person.
    public func scale(for vertices: [Vec3], faces: [UInt32]) -> Double {
        guard landmarks.count == 4, baseFaceSize.count == 2 else { return 1 }
        let p = landmarks.map { l -> Vec3 in
            let t = Int(l[0])
            var s = Vec3.zero
            for c in 0..<3 { s += l[1 + c] * vertices[Int(faces[3 * t + c])] }
            return s
        }
        let width = length(p[1] - p[0]), height = length(p[2] - p[3])
        return (width / baseFaceSize[0] * height / baseFaceSize[1]).squareRoot()
    }

    /// Verschiebung je bewegtem Punkt in Metern (Anny-Achsen).
    public func sparse(_ action: Action, scale: Double) -> [(index: Int, delta: Vec3)] {
        let f = 1e-6 * scale
        return action.index.enumerated().map { i, v in
            (v, Vec3(Double(action.delta[3 * i]), Double(action.delta[3 * i + 1]), Double(action.delta[3 * i + 2])) * f)
        }
    }

    /// Verschiebung aller `count` Punkte in Metern (Anny-Achsen), z. B. für glTF-Formregler.
    public func dense(_ action: Action, count: Int, scale: Double) -> [Vec3] {
        var out = [Vec3](repeating: .zero, count: count)
        for (v, d) in sparse(action, scale: scale) where v < count { out[v] = d }
        return out
    }
}

// MARK: - Abspielen

/// Bewegungen auf einem Körper abspielen, wie `swiftdata.MotionRig` (am PC gegen `motion.play`
/// geprüft). Annys Haltung „local-ref“: Jede Drehung wirkt in Weltachsen um den Gelenkkopf der
/// Bezugshaltung und nimmt die Kinder mit. Gegenüber der Ruhepose bewegt sich die Haut eines
/// Knochens daher erst von der Ruhepose in die Bezugshaltung (`ref`), dann mit den Drehungen. Die
/// Hüfte steht im Ursprung und wird um den Hüftpunkt der Bewegung (mal Hüfthöhe) verschoben.
public struct MotionRig {
    public let parents: [Int]
    /// Ruhepose der Person (Anny-Achsen, Meter)
    public let vertices: [Vec3]
    let restPoses: [Rigid]
    /// je Knochen: Ruhepose → Bezugshaltung
    let ref: [Rigid]
    /// Gelenkköpfe in der Bezugshaltung
    let heads: [Vec3]
    let indices: [Int]
    let weights: [Double]
    let perVertex: Int
    let boneIndex: [String: Int]
    /// Höhe des Hüftpunkts über dem Boden in der Bezugshaltung
    public private(set) var hipHeight: Double = 0

    public init(model: BodyModel, params: BodyParams) {
        let data = model.data
        let rest = model.rest(params)
        parents = data.boneParents
        vertices = rest.vertices
        restPoses = rest.bonePoses
        let n = parents.count
        var ref = [Rigid](repeating: .identity, count: n)
        var heads = [Vec3](repeating: .zero, count: n)
        for j in 0..<n {
            let p = parents[j]
            let head = p < 0 ? rest.bonePoses[j].translation : ref[p].apply(rest.bonePoses[j].translation)
            heads[j] = head
            if p >= 0 {
                let o = Array(data.referenceBoneOrientations[(9 * j)..<(9 * j + 9)]).map(Double.init)
                ref[j] = Rigid(rotation: Mat3(o), translation: head) * rest.bonePoses[j].inverse
            }
        }
        self.ref = ref
        self.heads = heads
        perVertex = data.bonesPerVertex
        indices = data.vertexBoneIndices.map(Int.init)
        weights = data.vertexBoneWeights.map(Double.init)
        var lookup: [String: Int] = [:]
        for (j, name) in data.boneLabels.enumerated() { lookup[name] = j }
        boneIndex = lookup
        let identity = frame(rotations: [Mat3](repeating: .identity, count: n), offset: .zero)
        hipHeight = -(skin(identity.moves).map(\.z).min() ?? 0)
    }

    /// Knochen in der Welt und Hautbewegung je Knochen (gegenüber der Ruhepose), beide um `offset`
    /// verschoben. `rotations`: je Knochen, Einheitsmatrix = Bezugshaltung.
    public func frame(rotations: [Mat3], offset: Vec3) -> (bones: [Rigid], moves: [Rigid]) {
        let n = parents.count
        var inner = [Rigid](repeating: .identity, count: n)
        var bones = [Rigid](repeating: .identity, count: n)
        var moves = [Rigid](repeating: .identity, count: n)
        for j in 0..<n {
            let r = rotations[j], h = heads[j], p = parents[j]
            inner[j] = p < 0 ? Rigid(rotation: r, translation: -(r * h))
                : inner[p] * Rigid(rotation: r, translation: h - r * h)
            var m = inner[j] * ref[j]
            m.translation += offset
            moves[j] = m
            bones[j] = m * restPoses[j]
        }
        return (bones, moves)
    }

    /// Haut nach `moves` (lineares Skinning wie Anny); `points` statt der Ruhepose, z. B. mit Mimik.
    public func skin(_ moves: [Rigid], points: [Vec3]? = nil) -> [Vec3] {
        let src = points ?? vertices
        let rot = moves.flatMap { $0.rotation.m }
        let trans = moves.map(\.translation)
        var out = [Vec3](repeating: .zero, count: src.count)
        for v in 0..<src.count {
            // gemischte Drehung zeilenweise und Verschiebung, ohne Zwischenspeicher je Punkt
            var r0 = Vec3.zero, r1 = Vec3.zero, r2 = Vec3.zero, t = Vec3.zero
            for k in 0..<perVertex {
                let w = weights[v * perVertex + k]
                if w == 0 { continue }
                let b = 9 * indices[v * perVertex + k]
                r0 += w * Vec3(rot[b], rot[b + 1], rot[b + 2])
                r1 += w * Vec3(rot[b + 3], rot[b + 4], rot[b + 5])
                r2 += w * Vec3(rot[b + 6], rot[b + 7], rot[b + 8])
                t += w * trans[b / 9]
            }
            let p = src[v]
            out[v] = Vec3(dot(r0, p), dot(r1, p), dot(r2, p)) + t
        }
        return out
    }

    /// Bild `f` einer Bewegung. `inPlace`: gleichmäßiges Vorwärtskommen abziehen (Gehen auf der Stelle).
    public func clipFrame(_ clip: MotionClip, _ f: Int, inPlace: Bool = false) -> (bones: [Rigid], moves: [Rigid]) {
        var rotations = [Mat3](repeating: .identity, count: parents.count)
        let q = clip.rotations[f]
        for (i, name) in clip.bones.enumerated() {
            guard let j = boneIndex[name], 4 * i + 3 < q.count else { continue }
            rotations[j] = Mat3(quaternion: Array(q[(4 * i)..<(4 * i + 4)]))
        }
        let r = clip.root[f]
        var root = Vec3(r[0], r[1], r[2])
        if inPlace, let first = clip.root.first, let last = clip.root.last, clip.frames > 1 {
            let share = Double(f) / Double(clip.frames - 1)
            root.x -= (last[0] - first[0]) * share
            root.y -= (last[1] - first[1]) * share
        }
        return frame(rotations: rotations, offset: root * hipHeight)
    }

    /// Ruhepose mit Mimik: Ruhepunkte plus die Gesichtsregler mit ihren Werten in Bild `f`.
    func points(_ clip: MotionClip, _ f: Int, face: [String: [(index: Int, delta: Vec3)]]) -> [Vec3] {
        var p = vertices
        for (name, curve) in clip.face where f < curve.count && curve[f] != 0 {
            for (v, d) in face[name] ?? [] { p[v] += curve[f] * d }
        }
        return p
    }

    /// Bewegung als glTF-Animation: Drehung jedes Gelenks (Bodengelenk zuerst) und Lage der Hüfte
    /// je Bild in glTF-Achsen, wie `motion.gltf_channels`; dazu die Mimik als Formregler-Werte.
    public func animation(_ clip: MotionClip) -> GLBAnimation {
        let toY = Rigid(rotation: toYUp, translation: .zero)
        let gparents = [-1] + parents.map { $0 >= 0 ? $0 + 1 : 0 }
        var rotations: [[Double]] = []
        var root: [Vec3] = []
        var previous: [Double] = []
        for f in 0..<clip.frames {
            let world = [Rigid.identity] + clipFrame(clip, f).bones.map { toY * $0 }
            var q: [Double] = []
            for (k, p) in gparents.enumerated() {
                let local = p < 0 ? world[k] : world[p].inverse * world[k]
                var qk = quaternion(local.rotation)
                // Vorzeichen wie im Bild davor, sonst dreht der Übergang den langen Weg
                if previous.count >= q.count + 4 {
                    let d = (0..<4).reduce(0.0) { $0 + qk[$1] * previous[q.count + $1] }
                    if d < 0 { qk = qk.map { -$0 } }
                }
                q += qk
                if k == 1 { root.append(local.translation) }
            }
            rotations.append(q)
            previous = q
        }
        return GLBAnimation(name: clip.label, times: (0..<clip.frames).map { Float(Double($0) / clip.fps) },
                            rotations: rotations, rootTranslation: root, weights: clip.face)
    }
}

/// Eine Bewegung für den GLB-Export, wie `export.GlbAnimation`.
public struct GLBAnimation {
    public var name: String
    public var times: [Float]
    /// Je Bild: Quaternion x, y, z, w je Gelenk, in der Reihenfolge von `RiggedMesh.jointNames`
    public var rotations: [[Double]]
    /// Je Bild: Lage von Gelenk 1 (Annys „root“) unter dem Bodengelenk
    public var rootTranslation: [Vec3]
    /// Formregler → Wert je Bild (nicht genannte bleiben 0)
    public var weights: [String: [Double]]
}

public extension BodyModel {
    /// Bilder einer Bewegung für die Vorschau in glTF-Achsen (Y oben, Boden y = 0), mit Mimik,
    /// Gehen auf der Stelle. Bei Schleifen fehlt das letzte Bild (es gleicht dem ersten).
    func previewFrames(_ params: BodyParams, clip: MotionClip, face: FaceRig?,
                       progress: ((Double) -> Void)? = nil) -> [PreviewMesh] {
        let rig = MotionRig(model: self, params: params)
        var sparse: [String: [(index: Int, delta: Vec3)]] = [:]
        if let face, !clip.face.isEmpty {
            let s = face.scale(for: rig.vertices, faces: faces)
            for name in clip.face.keys {
                if let action = face.action(name) { sparse[name] = face.sparse(action, scale: s) }
            }
        }
        let count = clip.loop && clip.frames > 1 ? clip.frames - 1 : clip.frames
        let height = (rig.vertices.map(\.z).max() ?? 1.8) - (rig.vertices.map(\.z).min() ?? 0)
        var out: [PreviewMesh] = []
        for f in 0..<count {
            let moves = rig.clipFrame(clip, f, inPlace: true).moves
            let posed = rig.skin(moves, points: sparse.isEmpty ? nil : rig.points(clip, f, face: sparse))
            let vy = posed.map { toYUp * $0 }
            out.append(PreviewMesh(vertices: vy, normals: vertexNormals(vy, data.faces), faces: data.faces,
                                   height: height))
            progress?(Double(f + 1) / Double(count))
        }
        return out
    }

    /// Modell mit Bewegungen als GLB: Skelett, Gewichte, Formregler, dazu je Bewegung eine
    /// glTF-Animation und, wenn eine Bewegung Mimik hat, die 52 Gesichtsregler (wie `motion.export_glb`).
    func writeGLB(_ params: BodyParams, to url: URL, naming: BoneNaming = .anny, name: String = "FitScan",
                  clips: [MotionClip], face: FaceRig?) throws {
        var rig = buildRig(params, naming: naming)
        let motion = MotionRig(model: self, params: params)
        if let face, clips.contains(where: { !$0.face.isEmpty }) {
            let s = face.scale(for: motion.vertices, faces: faces)
            for action in face.actions {
                rig.morphs.append((name: action.name,
                                   deltas: face.dense(action, count: numVertices, scale: s).map { toYUp * $0 }))
            }
        }
        try rig.glb(name: name, animations: clips.map { motion.animation($0) }).write(to: url)
    }
}
