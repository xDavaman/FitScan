import XCTest
import FitScanCore

/// Vorschau, Maßbänder und Nachbearbeitung (Etappe 1 der iPhone-App).
final class EditTests: XCTestCase {
    var model: BodyModel { ReferenceTests.model }

    /// Die Vorschau steht wie das GLB: Füße auf y = 0, gleiche Größe wie gemessen.
    func testPreviewStandsOnFloor() {
        let p = BodyParams()
        let mesh = model.previewMesh(p)
        XCTAssertEqual(mesh.vertices.count, model.numVertices)
        XCTAssertEqual(mesh.normals.count, model.numVertices)
        XCTAssertEqual(mesh.vertices.map(\.y).min() ?? -1, 0, accuracy: 1e-9)
        XCTAssertEqual(mesh.height * 100, model.measure(p)[.height] ?? 0, accuracy: 0.11)
    }

    /// Die Maßbänder liegen am Körper und sind so lang wie das gemessene Maß.
    func testTapeLinesMatchMeasures() {
        let p = BodyParams(phenotype: ["gender": 0, "age": 0.5, "muscle": 0.6, "weight": 0.55, "height": 0.5,
                                       "proportions": 0.5])
        let lines = model.tapeLines(p)
        let measures = model.measure(p)
        let mesh = model.previewMesh(p)
        for key in MeasureKey.circumferences {
            guard let line = lines[key] else { return XCTFail("\(key.label): kein Maßband") }
            var length = 0.0
            for i in 0..<line.count {
                let d = line[i] - line[(i + line.count - 1) % line.count]
                length += (d * d).sum().squareRoot()
            }
            XCTAssertEqual(length * 100, measures[key] ?? 0, accuracy: 0.11, key.label)
            let y = line.map(\.y).reduce(0, +) / Double(line.count)
            XCTAssertGreaterThan(y, 0.05 * mesh.height, key.label)
            XCTAssertLessThan(y, mesh.height, key.label)
        }
        // Brust über Taille über Hüfte
        let mean = { (k: MeasureKey) in lines[k]!.map(\.y).reduce(0, +) / Double(lines[k]!.count) }
        XCTAssertGreaterThan(mean(.chest), mean(.waist))
        XCTAssertGreaterThan(mean(.waist), mean(.hips))
    }

    /// Regler ändern die Form, die Körpergröße bleibt genau.
    func testAdjustedKeepsHeight() throws {
        let p = BodyParams()
        let height = model.measure(p)[.height]!
        let more = model.adjusted(p, [ShapeRegulator.body[0]: 0.9, ShapeRegulator.regions[1]: 0.6], heightCM: height)
        XCTAssertEqual(more.phenotype["muscle"], 0.9)
        XCTAssertEqual(more.local["measure-waist-circ-incr"], 0.6)
        let after = model.measure(more)
        XCTAssertEqual(after[.height] ?? 0, height, accuracy: 0.11)
        XCTAssertGreaterThan(after[.waist]!, model.measure(p)[.waist]! + 1)

        // Werte außerhalb des Bereichs werden begrenzt
        let clamped = model.adjusted(p, [ShapeRegulator.body[1]: 3], heightCM: nil)
        XCTAssertEqual(clamped.phenotype["weight"], 1)
    }

    /// Scans aus dem ersten Build (ohne `source`) lassen sich weiter lesen; neue merken sich die Herkunft.
    func testScanSourceIsStored() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("fitscan_source_\(UUID().uuidString)")
        let store = PersonStore(root: root)
        let person = Person(name: "Quelle", measures: [.height: 175, .weight: 70])
        let result = FitResult(params: BodyParams(), measures: model.measure(BodyParams()))
        let first = try store.save(person: person, result: result, model: model)
        let second = try store.save(person: person, result: result, model: model, date: first.date, source: "bearbeitet")
        XCTAssertNotEqual(first.id, second.id, "gleiche Sekunde, eigener Ordner")
        let scans = store.allScans()
        XCTAssertEqual(Set(scans.map(\.id)), [first.id, second.id])
        XCTAssertEqual(scans.first { $0.id == second.id }?.sourceLabel, "nachbearbeitet")

        // scan.json wie aus dem ersten Build, ohne „source“
        let url = store.folder(of: first).appendingPathComponent("scan.json")
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        json["source"] = nil
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        let old = store.allScans().first { $0.id == first.id }
        XCTAssertNotNil(old)
        XCTAssertNil(old?.source)
        XCTAssertEqual(old?.sourceLabel, "aus Maßen")
    }
}

extension EditTests {
    /// Der Schlauch um ein Maßband: richtige Größe und nah an der Linie.
    func testTubeAroundLine() {
        let line = (0..<20).map { i -> Vec3 in
            let a = 2 * Double.pi * Double(i) / 20
            return Vec3(0.15 * cos(a), 1.0, 0.1 * sin(a))
        }
        let tube = TubeMesh(closedLine: line, radius: 0.004, sides: 8)
        XCTAssertEqual(tube.vertices.count, 160)
        XCTAssertEqual(tube.normals.count, 160)
        XCTAssertEqual(tube.indices.count, 20 * 8 * 6)
        XCTAssertEqual(Int(tube.indices.max() ?? 0), 159)
        for (k, v) in tube.vertices.enumerated() {
            let d = v - line[k / 8]
            XCTAssertLessThanOrEqual((d * d).sum().squareRoot(), 0.0081)
            XCTAssertEqual(v.y, 1.0, accuracy: 0.0041)
        }
        XCTAssertTrue(TubeMesh(closedLine: Array(line.prefix(2))).vertices.isEmpty)
    }
}
