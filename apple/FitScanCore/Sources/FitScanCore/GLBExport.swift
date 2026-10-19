import Foundation

/// Namensschemata der Knochen: `anny` (z. B. `upperarm01.L`) und `unreal` (z. B. `upperarm_l`,
/// für Cascadeur und Unreal), wie `fitscan.export`.
public enum BoneNaming: String, CaseIterable, Identifiable {
    case anny, unreal

    public var id: String { rawValue }

    var groundJoint: String { self == .anny ? "ground" : "root" }

    static let unrealBase: [String: String] = [
        "root": "pelvis", "pelvis": "hip", "upperleg01": "thigh", "upperleg02": "thigh_twist_01",
        "lowerleg01": "calf", "lowerleg02": "calf_twist_01", "foot": "foot",
        "toe1-1": "bigtoe_01", "toe1-2": "bigtoe_02",
        "toe2-1": "indextoe_01", "toe2-2": "indextoe_02", "toe2-3": "indextoe_03",
        "toe3-1": "middletoe_01", "toe3-2": "middletoe_02", "toe3-3": "middletoe_03",
        "toe4-1": "ringtoe_01", "toe4-2": "ringtoe_02", "toe4-3": "ringtoe_03",
        "toe5-1": "littletoe_01", "toe5-2": "littletoe_02", "toe5-3": "littletoe_03",
        "spine05": "spine_01", "spine04": "spine_02", "spine03": "spine_03", "spine02": "spine_04",
        "spine01": "spine_05", "clavicle": "clavicle", "shoulder01": "shoulder",
        "upperarm01": "upperarm", "upperarm02": "upperarm_twist_01",
        "lowerarm01": "lowerarm", "lowerarm02": "lowerarm_twist_01", "wrist": "hand",
        "finger1-1": "thumb_01", "finger1-2": "thumb_02", "finger1-3": "thumb_03",
        "metacarpal1": "index_metacarpal", "finger2-1": "index_01", "finger2-2": "index_02",
        "finger2-3": "index_03",
        "metacarpal2": "middle_metacarpal", "finger3-1": "middle_01", "finger3-2": "middle_02",
        "finger3-3": "middle_03",
        "metacarpal3": "ring_metacarpal", "finger4-1": "ring_01", "finger4-2": "ring_02",
        "finger4-3": "ring_03",
        "metacarpal4": "pinky_metacarpal", "finger5-1": "pinky_01", "finger5-2": "pinky_02",
        "finger5-3": "pinky_03",
        "neck01": "neck_01", "neck02": "neck_02", "neck03": "neck_03", "head": "head", "eye": "eye",
    ]

    public func name(_ bone: String) -> String {
        guard self == .unreal else { return bone }
        let parts = bone.split(separator: ".", maxSplits: 1).map(String.init)
        let base = parts[0]
        let mapped = Self.unrealBase[base] ?? base.replacingOccurrences(of: "-", with: "_")
        return parts.count > 1 ? "\(mapped)_\(parts[1].lowercased())" : mapped
    }
}

/// Alles, was ein Exporter braucht, in Y-oben-Koordinaten mit Füßen auf y = 0.
public struct RiggedMesh {
    public var vertices: [Vec3]
    public var normals: [Vec3]
    public var faces: [UInt32]
    public var jointNames: [String]
    public var jointParents: [Int]  // −1 = keine Eltern
    public var jointGlobal: [Rigid]
    public var joints4: [UInt16]  // 4 je Punkt
    public var weights4: [Float]  // 4 je Punkt, Summe 1
    public var morphs: [(name: String, deltas: [Vec3])]
}

/// Wie weit ein Formregler auf 1.0 den Phänotyp verschiebt
let morphStep = 0.25
let morphRegulators: [(name: String, key: String, sign: Double)] = [
    ("weight_up", "weight", 1), ("weight_down", "weight", -1),
    ("muscle_up", "muscle", 1), ("muscle_down", "muscle", -1),
]

/// Anny (Z oben, Blick −Y) → glTF (Y oben, Blick +Z): (x, y, z) → (x, z, −y)
let toYUp = Mat3([1, 0, 0, 0, 0, 1, 0, -1, 0])

func vertexNormals(_ verts: [Vec3], _ faces: [UInt32]) -> [Vec3] {
    var n = [Vec3](repeating: .zero, count: verts.count)
    for f in 0..<(faces.count / 3) {
        let i0 = Int(faces[3 * f]), i1 = Int(faces[3 * f + 1]), i2 = Int(faces[3 * f + 2])
        let fn = cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
        n[i0] += fn
        n[i1] += fn
        n[i2] += fn
    }
    return n.map { $0 / max(length($0), 1e-12) }
}

public extension BodyModel {
    func buildRig(_ params: BodyParams, naming: BoneNaming = .anny, withMorphs: Bool = true) -> RiggedMesh {
        let r = rest(params)
        let minZ = r.vertices.map(\.z).min() ?? 0
        let toY = Rigid(rotation: toYUp, translation: Vec3(0, -minZ, 0))
        let vy = r.vertices.map { toY.apply($0) }

        let names = [naming.groundJoint] + data.boneLabels.map { naming.name($0) }
        let parents = [-1] + data.boneParents.map { $0 >= 0 ? $0 + 1 : 0 }
        let globals = [Rigid.identity] + r.bonePoses.map { toY * $0 }

        // Vier stärkste Gelenkeinflüsse je Punkt (das lesen alle Programme)
        let k = data.bonesPerVertex
        var j4 = [UInt16](repeating: 0, count: 4 * numVertices)
        var w4 = [Float](repeating: 0, count: 4 * numVertices)
        for v in 0..<numVertices {
            let order = (0..<k).sorted { data.vertexBoneWeights[v * k + $0] > data.vertexBoneWeights[v * k + $1] }
            var w = order.prefix(4).map { Double(data.vertexBoneWeights[v * k + $0]) }.map { $0 < 1e-4 ? 0 : $0 }
            let total = max(w.reduce(0, +), 1e-12)
            w = w.map { $0 / total }
            for (slot, b) in order.prefix(4).enumerated() {
                j4[4 * v + slot] = w[slot] > 0 ? UInt16(data.vertexBoneIndices[v * k + b]) + 1 : 0
                w4[4 * v + slot] = Float(w[slot])
            }
        }

        var morphDeltas: [(name: String, deltas: [Vec3])] = []
        if withMorphs {
            for (name, key, sign) in morphRegulators {
                var p = params
                let current = p.phenotype[key] ?? 0.5
                let value = min(max(current + sign * morphStep, 0), 1)
                if abs(value - current) < 1e-3 { continue }
                p.phenotype[key] = value
                let other = rest(p).vertices.map { toY.apply($0) }
                morphDeltas.append((name: name, deltas: zip(other, vy).map { $0 - $1 }))
            }
        }
        return RiggedMesh(vertices: vy, normals: vertexNormals(vy, data.faces), faces: data.faces,
                          jointNames: names, jointParents: parents, jointGlobal: globals,
                          joints4: j4, weights4: w4, morphs: morphDeltas)
    }

    /// Schreibt das Modell als GLB (Skelett, Gewichte, Formregler), wie `export.write_glb` ohne Hautbild.
    func writeGLB(_ params: BodyParams, to url: URL, naming: BoneNaming = .anny, name: String = "FitScan") throws {
        try buildRig(params, naming: naming).glb(name: name).write(to: url)
    }
}

/// Binärteil und Zugriffe einer glTF-Datei.
struct GLBBuffer {
    var data = Data()
    var views: [[String: Any]] = []
    var accessors: [[String: Any]] = []

    static let float = 5126, uint16 = 5123, uint32 = 5125
    static let arrayBuffer = 34962, elementArrayBuffer = 34963

    mutating func align() { while data.count % 4 != 0 { data.append(0) } }

    mutating func add<T>(_ values: [T], component: Int, type: String, count: Int, target: Int? = nil,
                         minMax: (min: [Double], max: [Double])? = nil) -> Int {
        align()
        var view: [String: Any] = ["buffer": 0, "byteOffset": data.count,
                                   "byteLength": values.count * MemoryLayout<T>.stride]
        if let target { view["target"] = target }
        values.withUnsafeBytes { data.append(contentsOf: $0) }
        views.append(view)
        var accessor: [String: Any] = ["bufferView": views.count - 1, "componentType": component,
                                       "count": count, "type": type]
        if let minMax {
            accessor["min"] = minMax.min
            accessor["max"] = minMax.max
        }
        accessors.append(accessor)
        return accessors.count - 1
    }

    mutating func addVec3(_ v: [Vec3], target: Int? = GLBBuffer.arrayBuffer, minMax: Bool = false) -> Int {
        let flat = v.flatMap { [Float($0.x), Float($0.y), Float($0.z)] }
        var bounds: (min: [Double], max: [Double])?
        if minMax {
            var lo = [Double](repeating: .infinity, count: 3), hi = [Double](repeating: -.infinity, count: 3)
            for i in 0..<v.count {
                for c in 0..<3 {
                    let x = Double(flat[3 * i + c])
                    lo[c] = min(lo[c], x)
                    hi[c] = max(hi[c], x)
                }
            }
            bounds = (lo, hi)
        }
        return add(flat, component: Self.float, type: "VEC3", count: v.count, target: target, minMax: bounds)
    }
}

extension RiggedMesh {
    func glb(name: String) throws -> Data {
        var buf = GLBBuffer()
        let pos = buf.addVec3(vertices, minMax: true)
        let nor = buf.addVec3(normals)
        let jnt = buf.add(joints4, component: GLBBuffer.uint16, type: "VEC4", count: vertices.count,
                          target: GLBBuffer.arrayBuffer)
        let wgt = buf.add(weights4, component: GLBBuffer.float, type: "VEC4", count: vertices.count,
                          target: GLBBuffer.arrayBuffer)
        let idx = buf.add(faces, component: GLBBuffer.uint32, type: "SCALAR", count: faces.count,
                          target: GLBBuffer.elementArrayBuffer)
        var targets: [[String: Any]] = []
        for morph in morphs { targets.append(["POSITION": buf.addVec3(morph.deltas, minMax: true)]) }
        let ibm = jointGlobal.flatMap { $0.inverse.columnMajor }
        let ibmAccessor = buf.add(ibm, component: GLBBuffer.float, type: "MAT4", count: jointGlobal.count)

        // 0: Hülle für das Skelett, 1: Netz (laut glTF nicht unterhalb von Knoten mit Transform)
        var nodes: [[String: Any]] = [["name": name, "children": [Int]()], ["name": "\(name)_Body", "mesh": 0, "skin": 0]]
        let firstJoint = nodes.count
        for (j, jointName) in jointNames.enumerated() {
            let parent = jointParents[j]
            let local = parent < 0 ? jointGlobal[j] : jointGlobal[parent].inverse * jointGlobal[j]
            nodes.append(["name": jointName,
                          "translation": [local.translation.x, local.translation.y, local.translation.z],
                          "rotation": quaternion(local.rotation)])
        }
        for (j, parent) in jointParents.enumerated() {
            let key = parent < 0 ? 0 : firstJoint + parent
            var children = nodes[key]["children"] as? [Int] ?? []
            children.append(firstJoint + j)
            nodes[key]["children"] = children
        }

        var primitive: [String: Any] = [
            "attributes": ["POSITION": pos, "NORMAL": nor, "JOINTS_0": jnt, "WEIGHTS_0": wgt],
            "indices": idx, "material": 0, "mode": 4,
        ]
        var mesh: [String: Any] = ["name": "\(name)_Body"]
        if !targets.isEmpty {
            primitive["targets"] = targets
            mesh["weights"] = [Double](repeating: 0, count: targets.count)
            mesh["extras"] = ["targetNames": morphs.map(\.name)]
        }
        mesh["primitives"] = [primitive]

        let gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "FitScan iOS"],
            "scene": 0,
            "scenes": [["name": name, "nodes": [0, 1]]],
            "nodes": nodes,
            "meshes": [mesh],
            "skins": [["name": "\(name)_Rig", "joints": Array(firstJoint..<(firstJoint + jointNames.count)),
                       "inverseBindMatrices": ibmAccessor, "skeleton": firstJoint]],
            "materials": [["name": "Haut", "pbrMetallicRoughness": [
                "baseColorFactor": [0.80, 0.62, 0.52, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.6]]],
            "accessors": buf.accessors,
            "bufferViews": buf.views,
            "buffers": [["byteLength": buf.data.count]],
        ]
        var json = try JSONSerialization.data(withJSONObject: gltf, options: [.sortedKeys])
        while json.count % 4 != 0 { json.append(0x20) }
        buf.align()

        var out = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { out.append(contentsOf: $0) } }
        out.append(contentsOf: Array("glTF".utf8))
        u32(2)
        u32(12 + 8 + json.count + 8 + buf.data.count)
        u32(json.count)
        out.append(contentsOf: Array("JSON".utf8))
        out.append(json)
        u32(buf.data.count)
        out.append(contentsOf: [0x42, 0x49, 0x4E, 0x00])  // "BIN\0"
        out.append(buf.data)
        return out
    }
}
