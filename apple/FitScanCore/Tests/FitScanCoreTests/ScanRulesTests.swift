import CoreGraphics
import ImageIO
import XCTest
import FitScanCore

/// Scan-Assistent (Etappe 4): dieselben Hinweise wie `quality.check_frame` am PC für künstliche
/// Kamerabilder (Prüfwerte in `referenz.json` unter „scan“), dazu Blickrichtung und Drehung.
final class ScanRulesTests: XCTestCase {
    static var rules: ScanRules = {
        do { return try ScanRules.bundled() } catch { fatalError("scanregeln.json fehlt: \(error)") }
    }()

    var reference: [String: Any] { ReferenceTests.reference["scan"] as? [String: Any] ?? [:] }

    func keypoints(_ raw: Any?) -> [Int: SIMD3<Double>]? {
        guard let raw = raw as? [String: [Double]] else { return nil }
        var out: [Int: SIMD3<Double>] = [:]
        for (key, v) in raw { out[Int(key)!] = SIMD3(v[0], v[1], v[2]) }
        return out
    }

    /// Graubild aus dem PNG vom PC, Byte für Byte (ohne Farbumrechnung).
    func luma(_ base64: String, width: Int, height: Int) throws -> [UInt8] {
        let data = try XCTUnwrap(Data(base64Encoded: base64))
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(image.bitsPerPixel, 8)
        let bytes = try XCTUnwrap(image.dataProvider?.data as Data?)
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width { out[y * width + x] = bytes[y * image.bytesPerRow + x] }
        }
        return out
    }

    func testRulesLoad() {
        let rules = Self.rules
        XCTAssertEqual(rules.limit("min_person_luma"), 70)
        XCTAssertEqual(rules.hint("dunkel").level, .rot)
        XCTAssertEqual(rules.hint("zu_weit").level, .gelb)
        XCTAssertFalse(rules.setting("side_shoulder_ratio").isNaN)
        XCTAssertGreaterThan(rules.hints.count, 15)
    }

    /// Jeder künstliche Fall ergibt dieselben Hinweise wie am PC (ohne die live weggelassenen Regeln).
    func testChecksMatchPC() throws {
        let cases = reference.filter { !$0.key.hasPrefix("_") }
        XCTAssertGreaterThanOrEqual(cases.count, 8)
        for (name, raw) in cases {
            let c = try XCTUnwrap(raw as? [String: Any])
            let w = c["width"] as! Int, h = c["height"] as! Int
            var mask = [Float](repeating: 0, count: w * h)
            var pos = 0
            for (i, n) in (c["mask_runs"] as! [Int]).enumerated() {
                if i % 2 == 1 { for p in pos..<(pos + n) { mask[p] = 1 } }
                pos += n
            }
            let frame = LiveFrame(width: w, height: h, luma: try luma(c["luma_png"] as! String, width: w, height: h),
                                  mask: mask, keypoints: try XCTUnwrap(keypoints(c["keypoints"])), persons: 1,
                                  sharpness: c["sharpness"] as? Double)
            let result = Self.rules.check(frame, previous: keypoints(c["previous"]))
            XCTAssertEqual(result.hints.map(\.code).sorted(), c["hints"] as? [String], name)
            XCTAssertEqual(result.view, .vorn, name)
            let expected = c["metrics"] as! [String: Double]
            for key in ["person_luma", "background_luma", "height_fraction", "center_offset", "arm_left_deg",
                        "feet_ratio", "highlight_share"] {
                guard let want = expected[key] else { continue }
                XCTAssertEqual(try XCTUnwrap(result.metrics[key], "\(name) \(key)"), want, accuracy: 0.02, "\(name) \(key)")
            }
        }
    }

    func testNoPerson() {
        let frame = LiveFrame(width: 4, height: 4, luma: [UInt8](repeating: 100, count: 16),
                              mask: [Float](repeating: 0, count: 16), keypoints: [:], persons: 0, sharpness: nil)
        let result = Self.rules.check(frame)
        XCTAssertEqual(result.level, .rot)
        XCTAssertEqual(result.hints.map(\.code), ["keine_person"])
    }

    func testLaplaceMatchesOpenCV() throws {
        let c = try XCTUnwrap(reference["_laplace"] as? [String: Any])
        let gray = (c["gray"] as! [Int]).map(Float.init)
        let mine = ScanRules.laplaceVariance(gray, width: c["width"] as! Int, height: c["height"] as! Int)
        let want = c["variance"] as! Double
        XCTAssertEqual(mine, want, accuracy: want * 1e-9)
        XCTAssertEqual(ScanRules.laplaceVariance([Float](repeating: 7, count: 30), width: 6, height: 5), 0, accuracy: 1e-12)
        // Verkleinern ändert eine glatte Fläche nicht
        XCTAssertEqual(ScanRules.sharpness([Float](repeating: 90, count: 300 * 900), width: 300, height: 900), 0,
                       accuracy: 1e-9)
    }

    func testBodyView() {
        func person(_ shoulderDX: Double, confidence: Double = 1, face: Double = 1) -> [Int: SIMD3<Double>] {
            [0: SIMD3(100, 70, face), 2: SIMD3(104, 66, face), 5: SIMD3(96, 66, face),
             11: SIMD3(100 + shoulderDX / 2, 100, confidence), 12: SIMD3(100 - shoulderDX / 2, 100, confidence),
             23: SIMD3(110, 180, 1), 24: SIMD3(90, 180, 1)]
        }
        let ratio = Self.rules.setting("side_shoulder_ratio")
        XCTAssertEqual(ScanRules.bodyView(person(56), sideRatio: ratio), .vorn)
        XCTAssertEqual(ScanRules.bodyView(person(-60), sideRatio: ratio), .ruecken)
        XCTAssertEqual(ScanRules.bodyView(person(12), sideRatio: ratio), .seite)
        XCTAssertNil(ScanRules.bodyView(person(56, confidence: 0.1), sideRatio: ratio))
        // von hinten ohne Gesicht, Seiten vertauscht erkannt
        XCTAssertEqual(ScanRules.bodyView(person(56, face: 0.05), sideRatio: ratio), .ruecken)
    }

    func testTurnTracker() {
        var turn = TurnTracker(confirm: 2)
        XCTAssertEqual(turn.progress, 1)
        // einzelne Ausreißer zählen nicht
        for view: BodyView? in [.vorn, .seite, .vorn, nil, .seite, .seite, .ruecken, .ruecken, .ruecken, .vorn] {
            turn.add(view)
        }
        XCTAssertEqual(turn.seen, [.vorn, .seite, .ruecken])
        XCTAssertEqual(turn.progress, 3)
        XCTAssertFalse(turn.complete)
        for view: BodyView? in [.seite, .seite, .vorn, .vorn] { turn.add(view) }
        XCTAssertTrue(turn.complete)
        // Halbe Drehung und zurück ist keine Runde
        var half = TurnTracker(confirm: 1)
        for view: BodyView? in [.seite, .vorn] { half.add(view) }
        XCTAssertEqual(half.progress, 2)
        XCTAssertFalse(half.complete)
    }
}
