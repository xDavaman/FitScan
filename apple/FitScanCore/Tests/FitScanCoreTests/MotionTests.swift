import XCTest
import FitScanCore

/// Bewegungen und Mimik (Etappe 2) gegen den PC (`motion.play`, `motion.gltf_channels`,
/// `face_rig.face_morphs`; Prüfwerte in `referenz.json` unter „bewegungen“).
final class MotionTests: XCTestCase {
    var model: BodyModel { ReferenceTests.model }

    static var library: MotionLibrary = {
        do { return try MotionLibrary.bundled() } catch { fatalError("bewegungen.json fehlt: \(error)") }
    }()

    static var face: FaceRig = {
        do { return try FaceRig.bundled() } catch { fatalError("mimik.json fehlt: \(error)") }
    }()

    var reference: [String: Any] { ReferenceTests.reference["bewegungen"] as? [String: Any] ?? [:] }

    func params(_ raw: [String: Any]) -> BodyParams {
        BodyParams(phenotype: raw["phenotype"] as? [String: Double] ?? [:],
                   local: raw["local"] as? [String: Double] ?? [:],
                   scale: raw["scale"] as? Double ?? 1)
    }

    func testLibraryLoads() throws {
        let keys = Self.library.clips.map(\.key)
        XCTAssertEqual(keys, reference["clips"] as? [String])
        for clip in Self.library.clips {
            XCTAssertGreaterThan(clip.frames, 1, clip.key)
            XCTAssertEqual(clip.rotations.count, clip.frames, clip.key)
            for q in clip.rotations { XCTAssertEqual(q.count, 4 * clip.bones.count, clip.key) }
            for name in clip.bones { XCTAssertNotNil(model.data.bone(name), "\(clip.key): \(name)") }
            for name in clip.face.keys { XCTAssertNotNil(Self.face.action(name), "\(clip.key): \(name)") }
        }
        XCTAssertEqual(Self.face.actions.count, reference["face_actions"] as? Int)
    }

    /// Knochen, Haut, glTF-Spuren und Hüfthöhe wie am PC.
    func testMotionMatchesPC() throws {
        let sample = try XCTUnwrap(reference["sample"] as? [Int])
        let bodies = try XCTUnwrap(reference["bodies"] as? [String: [String: Any]])
        XCTAssertFalse(bodies.isEmpty)
        for (body, ref) in bodies {
            let rig = MotionRig(model: model, params: params(ref["params"] as! [String: Any]))
            for (key, raw) in ref["clips"] as! [String: [String: Any]] {
                let clip = try XCTUnwrap(Self.library.clip(key))
                XCTAssertEqual(rig.hipHeight, raw["hip_height"] as! Double, accuracy: 1e-5, "\(body) \(key)")
                let animation = rig.animation(clip)
                for frame in raw["frames"] as! [[String: Any]] {
                    let f = frame["frame"] as! Int
                    let (bones, moves) = rig.clipFrame(clip, f)
                    var worst = 0.0
                    for (j, m) in (frame["bones"] as! [[[Double]]]).enumerated() {
                        for r in 0..<3 {
                            for c in 0..<3 { worst = max(worst, abs(bones[j].rotation[r, c] - m[r][c])) }
                            worst = max(worst, abs(bones[j].translation[r] - m[r][3]))
                        }
                    }
                    XCTAssertLessThan(worst, 1e-4, "\(body) \(key) Bild \(f): Knochen")
                    let skin = rig.skin(moves)
                    worst = 0
                    for (i, v) in (frame["vertices"] as! [[Double]]).enumerated() {
                        let p = skin[sample[i]]
                        worst = max(worst, abs(p.x - v[0]), abs(p.y - v[1]), abs(p.z - v[2]))
                    }
                    XCTAssertLessThan(worst, 1e-4, "\(body) \(key) Bild \(f): Haut")
                    // Quaternionen mit beliebigem Vorzeichen vergleichen
                    worst = 0
                    for (k, q) in (frame["gltf_rotations"] as! [[Double]]).enumerated() {
                        let mine = Array(animation.rotations[f][(4 * k)..<(4 * k + 4)])
                        let plus = (0..<4).map { abs(mine[$0] - q[$0]) }.max()!
                        let minus = (0..<4).map { abs(mine[$0] + q[$0]) }.max()!
                        worst = max(worst, min(plus, minus))
                    }
                    XCTAssertLessThan(worst, 1e-4, "\(body) \(key) Bild \(f): glTF-Drehungen")
                    let root = frame["gltf_root"] as! [Double]
                    let mine = animation.rootTranslation[f]
                    XCTAssertLessThan(max(abs(mine.x - root[0]), abs(mine.y - root[1]), abs(mine.z - root[2])), 1e-4,
                                      "\(body) \(key) Bild \(f): Hüfte")
                }
            }
        }
    }

    func testFaceMorphsMatchPC() throws {
        let sample = try XCTUnwrap(reference["sample"] as? [Int])
        let bodies = try XCTUnwrap(reference["bodies"] as? [String: [String: Any]])
        for (body, ref) in bodies {
            let rest = model.rest(params(ref["params"] as! [String: Any]))
            let scale = Self.face.scale(for: rest.vertices, faces: model.faces)
            for (name, values) in ref["face"] as! [String: [[Double]]] {
                let action = try XCTUnwrap(Self.face.action(name))
                let dense = Self.face.dense(action, count: model.numVertices, scale: scale)
                var worst = 0.0
                for (i, v) in values.enumerated() {
                    let d = dense[sample[i]]
                    worst = max(worst, abs(d.x - v[0]), abs(d.y - v[1]), abs(d.z - v[2]))
                }
                XCTAssertLessThan(worst, 1e-6, "\(body) \(name)")
            }
        }
    }

    /// GLB mit allen Bewegungen: gültiger Aufbau, je Bewegung eine Animation, Mimik als Formregler.
    func testGLBWithMotions() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bewegungen-test.glb")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.writeGLB(BodyParams(), to: url, naming: .unreal, clips: Self.library.clips, face: Self.face)
        let data = try Data(contentsOf: url)
        func u32(_ at: Int) -> Int { Int(data[at]) | Int(data[at + 1]) << 8 | Int(data[at + 2]) << 16 | Int(data[at + 3]) << 24 }
        XCTAssertEqual(u32(8), data.count)
        let jsonLength = u32(12)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data.subdata(in: 20..<(20 + jsonLength))) as? [String: Any])
        let binLength = u32(20 + jsonLength)
        let animations = try XCTUnwrap(json["animations"] as? [[String: Any]])
        XCTAssertEqual(animations.map { $0["name"] as? String ?? "" }, Self.library.clips.map(\.label))
        let meshes = try XCTUnwrap(json["meshes"] as? [[String: Any]])
        let targets = (meshes[0]["extras"] as? [String: Any])?["targetNames"] as? [String] ?? []
        XCTAssertEqual(targets.count, 4 + Self.face.actions.count)
        let accessors = try XCTUnwrap(json["accessors"] as? [[String: Any]])
        let views = try XCTUnwrap(json["bufferViews"] as? [[String: Any]])
        for view in views {
            XCTAssertLessThanOrEqual((view["byteOffset"] as! Int) + (view["byteLength"] as! Int), binLength)
        }
        for anim in animations {
            let samplers = anim["samplers"] as! [[String: Any]]
            let channels = anim["channels"] as! [[String: Any]]
            XCTAssertEqual(samplers.count, channels.count)
            XCTAssertGreaterThan(channels.count, 10)
            for s in samplers {
                let input = accessors[s["input"] as! Int], output = accessors[s["output"] as! Int]
                XCTAssertNotNil(input["max"])
                let frames = input["count"] as! Int
                let type = output["type"] as! String
                if type != "SCALAR" { XCTAssertEqual(output["count"] as? Int, frames) }
            }
        }
        let paths = Set(animations.flatMap { ($0["channels"] as! [[String: Any]]).map { ($0["target"] as! [String: Any])["path"] as! String } })
        XCTAssertEqual(paths, ["rotation", "translation", "weights"])
    }

    /// Vorschau: ein Bild je Schleifenschritt, Füße auf dem Boden, Gehen bleibt auf der Stelle.
    func testPreviewFrames() throws {
        let walk = try XCTUnwrap(Self.library.clip("gehen"))
        let frames = model.previewFrames(BodyParams(), clip: walk, face: Self.face)
        XCTAssertEqual(frames.count, walk.frames - 1)
        for mesh in frames {
            let low = mesh.vertices.map(\.y).min()!
            XCTAssertEqual(low, 0, accuracy: 0.05)
            let z = mesh.vertices.map(\.z).reduce(0, +) / Double(mesh.vertices.count)
            XCTAssertEqual(z, 0, accuracy: 0.25, "Gehen auf der Stelle")
        }
        let talk = try XCTUnwrap(Self.library.clip("sprechen"))
        let talking = model.previewFrames(BodyParams(), clip: talk, face: Self.face)
        let silent = model.previewFrames(BodyParams(), clip: talk, face: nil)
        let moved = zip(talking, silent).map { a, b in
            zip(a.vertices, b.vertices).map { abs($0.y - $1.y) + abs($0.z - $1.z) }.max()!
        }.max()!
        XCTAssertGreaterThan(moved, 0.002, "Mimik bewegt den Mund")
    }
}
