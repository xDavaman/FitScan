import XCTest
import FitScanCore

/// Vergleicht FitScanCore mit den Prüfwerten vom PC (`referenz.json`, geschrieben von
/// `scripts/export_swift_data.py`). Schneller mit `swift test -c release`.
final class ReferenceTests: XCTestCase {
    static var model: BodyModel = {
        do { return try BodyModel.bundled() } catch { fatalError("Anny-Daten fehlen: \(error)") }
    }()

    static var reference: [String: Any] = {
        guard let url = Bundle.module.url(forResource: "referenz", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            fatalError("referenz.json fehlt")
        }
        return json
    }()

    var model: BodyModel { Self.model }

    func params(_ raw: [String: Any]) -> BodyParams {
        BodyParams(phenotype: raw["phenotype"] as? [String: Double] ?? [:],
                   local: raw["local"] as? [String: Double] ?? [:],
                   scale: raw["scale"] as? Double ?? 1)
    }

    func bodies() -> [(String, [String: Any])] {
        let b = Self.reference["bodies"] as? [String: [String: Any]] ?? [:]
        return b.keys.sorted().map { ($0, b[$0]!) }
    }

    func testDataLoads() {
        XCTAssertEqual(model.data.numVertices, 13718)
        XCTAssertEqual(model.data.numBones, 104)
        XCTAssertEqual(model.data.numShapes, model.data.numMacroShapes + 2 * model.data.localLabels.count)
        XCTAssertEqual(model.data.boneParents.first, -1)
    }

    /// Punkte und Knochen wie am PC, jeder Punkt weniger als 1 mm daneben.
    func testBodiesMatchPC() throws {
        XCTAssertFalse(bodies().isEmpty)
        for (name, ref) in bodies() {
            let rest = model.rest(params(ref["params"] as! [String: Any]))
            let sample = ref["sample_vertices"] as! [Int]
            let verts = ref["vertices"] as! [[Double]]
            var worst = 0.0
            for (i, v) in sample.enumerated() {
                let p = rest.vertices[v]
                worst = max(worst, abs(p.x - verts[i][0]), abs(p.y - verts[i][1]), abs(p.z - verts[i][2]))
            }
            XCTAssertLessThan(worst, 0.001, "\(name): Punkt \(worst * 1000) mm daneben")

            let heads = ref["bone_heads"] as! [[Double]]
            var worstHead = 0.0
            for (j, h) in heads.enumerated() {
                let p = rest.boneHeads[j]
                worstHead = max(worstHead, abs(p.x - h[0]), abs(p.y - h[1]), abs(p.z - h[2]))
            }
            XCTAssertLessThan(worstHead, 0.001, "\(name): Knochenkopf \(worstHead * 1000) mm daneben")

            let rots = ref["bone_rotations"] as! [[[Double]]]
            var worstRot = 0.0
            for (j, r) in rots.enumerated() {
                let mine = rest.bonePoses[j].rotation
                for a in 0..<3 { for b in 0..<3 { worstRot = max(worstRot, abs(mine[a, b] - r[a][b])) } }
            }
            XCTAssertLessThan(worstRot, 1e-3, "\(name): Knochendrehung weicht um \(worstRot) ab")
        }
    }

    /// Maße wie mit dem Maßband, auf 0,1 cm wie am PC.
    func testMeasuresMatchPC() {
        for (name, ref) in bodies() {
            let mine = model.measure(params(ref["params"] as! [String: Any]))
            let pc = ref["measures"] as! [String: Double]
            for (key, value) in pc {
                guard let k = MeasureKey(rawValue: key) else { continue }
                XCTAssertEqual(mine[k] ?? -1, value, accuracy: 0.11, "\(name): \(k.label)")
            }
        }
    }

    /// Anpassung an Maße: jedes eingegebene Maß höchstens 0,5 cm (Gewicht 0,5 kg) weiter vom Ziel
    /// als am PC. Der PC trifft nicht jedes Maß genau (z. B. Gewicht neben vielen Umfängen).
    func testFitReachesTargets() throws {
        let fits = Self.reference["fits"] as! [String: [String: Any]]
        for name in fits.keys.sorted() {
            let spec = fits[name]!["person"] as! [String: Any]
            var measures: [MeasureKey: Double] = [:]
            for (k, v) in spec["measures"] as! [String: Double] { measures[MeasureKey(rawValue: k)!] = v }
            let person = Person(name: name, gender: Gender(rawValue: spec["gender"] as! String)!,
                                ageYears: spec["age_years"] as! Double, measures: measures)
            let start = Date()
            let result = try model.fitToMeasurements(person)
            let pc = fits[name]!["pc_measures"] as! [String: Double]
            let seconds = Date().timeIntervalSince(start)
            print("Anpassung \(name): \(String(format: "%.1f", seconds)) s, \(result.measures), PC: \(pc)")
            for (key, target) in measures {
                let mine = abs((result.measures[key] ?? -100) - target)
                let theirs = abs((pc[key.rawValue] ?? target) - target)
                XCTAssertLessThanOrEqual(mine, theirs + 0.5, "\(name): \(key.label) \(result.measures[key] ?? -1), PC \(pc[key.rawValue] ?? -1), Ziel \(target)")
            }
        }
    }

    func testInvalidPersonIsRejected() {
        let person = Person(measures: [.height: 50, .weight: 80])
        XCTAssertThrowsError(try model.fitToMeasurements(person)) { error in
            XCTAssertTrue((error as? FitScanError)?.errorDescription?.contains("Körpergröße") ?? false)
        }
    }

    /// GLB: Skelett mit Bodengelenk, Gewichte, vier Formregler, Füße auf y = 0.
    func testGLBStructure() throws {
        let glbRef = Self.reference["glb"] as! [String: Any]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fitscan_test.glb")
        try model.writeGLB(BodyParams(), to: url, naming: .unreal)
        let data = try Data(contentsOf: url)
        func u32(_ at: Int) -> Int {
            Int(data[at]) | Int(data[at + 1]) << 8 | Int(data[at + 2]) << 16 | Int(data[at + 3]) << 24
        }
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "glTF")
        XCTAssertEqual(u32(4), 2)
        XCTAssertEqual(u32(8), data.count)
        let jsonLength = u32(12)
        XCTAssertEqual(String(decoding: data[16..<20], as: UTF8.self), "JSON")
        let json = try JSONSerialization.jsonObject(with: data[20..<(20 + jsonLength)]) as! [String: Any]
        let binLength = u32(20 + jsonLength)
        XCTAssertEqual(20 + jsonLength + 8 + binLength, data.count)

        let skin = (json["skins"] as! [[String: Any]])[0]
        XCTAssertEqual((skin["joints"] as! [Int]).count, glbRef["joints"] as! Int)
        let nodes = json["nodes"] as! [[String: Any]]
        XCTAssertEqual(nodes[2]["name"] as? String, "root")  // Bodengelenk im Unreal-Schema
        XCTAssertTrue(nodes.contains { ($0["name"] as? String) == "upperarm_l" })
        let mesh = (json["meshes"] as! [[String: Any]])[0]
        let extras = mesh["extras"] as! [String: Any]
        XCTAssertEqual(extras["targetNames"] as! [String], glbRef["morphs"] as! [String])
        let accessors = json["accessors"] as! [[String: Any]]
        let attributes = ((mesh["primitives"] as! [[String: Any]])[0]["attributes"]) as! [String: Int]
        let position = accessors[attributes["POSITION"]!]
        XCTAssertEqual(position["count"] as? Int, glbRef["vertices"] as? Int)
        let minY = (position["min"] as! [Double])[1]
        XCTAssertEqual(minY, 0, accuracy: 1e-5, "Füße stehen auf dem Boden")
        let maxY = (position["max"] as! [Double])[1]
        XCTAssertGreaterThan(maxY, 1.4)

        let rig = model.buildRig(BodyParams())
        for v in 0..<(rig.weights4.count / 4) {
            let sum = rig.weights4[(4 * v)..<(4 * v + 4)].reduce(0, +)
            XCTAssertEqual(sum, 1, accuracy: 1e-4)
        }
    }

    func testParamsRoundTrip() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("fitscan_params_\(UUID().uuidString)")
        var p = BodyParams(phenotype: ["gender": 0, "age": 0.6, "muscle": 0.7, "weight": 0.4, "height": 0.5,
                                       "proportions": 0.5],
                           local: ["measure-waist-circ-incr": 0.25], scale: 1.03)
        p.offsets = (0..<(3 * model.numVertices)).map { Double($0 % 7) * 1e-4 }
        try p.save(to: folder)
        let back = try BodyParams.load(from: folder)
        XCTAssertEqual(back.phenotype, p.phenotype)
        XCTAssertEqual(back.local, p.local)
        XCTAssertEqual(back.scale, p.scale, accuracy: 1e-12)
        XCTAssertEqual(back.offsets?.count, p.offsets?.count)
        XCTAssertEqual(back.offsets?[5] ?? 0, p.offsets![5], accuracy: 1e-7)
    }

    func testProcrustesGivesRotation() {
        let r = Mat3([0.36, 0.48, -0.8, -0.8, 0.6, 0, 0.48, 0.64, 0.6])  // echte Drehung
        let noisy = Mat3(zip(r.m, [0.01, -0.02, 0.0, 0.03, 0.0, -0.01, 0.0, 0.02, 0.01]).map { 2.5 * ($0 + $1) })
        let q = specialProcrustes(noisy)
        XCTAssertEqual(q.determinant, 1, accuracy: 1e-9)
        let i = q.transposed * q
        for a in 0..<3 { for b in 0..<3 { XCTAssertEqual(i[a, b], a == b ? 1 : 0, accuracy: 1e-9) } }
        for k in 0..<9 { XCTAssertEqual(q.m[k], r.m[k], accuracy: 0.05) }
    }

    func testStoreSavesAndLists() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("fitscan_store_\(UUID().uuidString)")
        let store = PersonStore(root: root)
        let person = Person(name: "Test Person", measures: [.height: 175, .weight: 70])
        let result = FitResult(params: BodyParams(), measures: model.measure(BodyParams()))
        let record = try store.save(person: person, result: result, model: model)
        XCTAssertEqual(record.personID, "test-person")
        XCTAssertEqual(store.allScans().map(\.id), [record.id])
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.folder(of: record).appendingPathComponent("koerper.glb").path))
    }
}
