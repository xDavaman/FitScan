import XCTest
import FitScanCore

/// Foto- und Videoscan (Etappe 3): Maskenwerkzeuge, Zwischenwerte und Ergebnis wie am PC
/// (`fitscan.photofit`, Prüfwerte in `referenz.json` unter „fotoscan“).
final class PhotoFitTests: XCTestCase {
    var model: BodyModel { ReferenceTests.model }

    static var settings: PhotoScanSettings = {
        do { return try PhotoScanSettings.bundled() } catch { fatalError("fotoscan.json fehlt: \(error)") }
    }()

    var reference: [String: Any] { ReferenceTests.reference["fotoscan"] as? [String: Any] ?? [:] }

    func frames() -> [PhotoFrame] {
        let raw = reference["frames"] as? [[String: Any]] ?? []
        let calibrated = reference["calibrated"] as? Bool ?? true
        return raw.map { f in
            let w = f["width"] as! Int, h = f["height"] as! Int
            var mask = [Float](repeating: 0, count: w * h)
            var pos = 0
            for (i, n) in (f["runs"] as! [Int]).enumerated() {
                if i % 2 == 1 { for p in pos..<(pos + n) { mask[p] = 1 } }
                pos += n
            }
            var kp: [Int: SIMD3<Double>] = [:]
            for (key, value) in f["keypoints"] as! [String: [Double]] {
                kp[Int(key)!] = SIMD3(value[0], value[1], value[2])
            }
            return PhotoFrame(width: w, height: h, mask: mask, focal: f["focal"] as! Double,
                              cx: f["cx"] as? Double, cy: f["cy"] as? Double, keypoints: kp,
                              yawHint: f["yaw_hint"] as? Double, calibrated: calibrated)
        }
    }

    func norm(_ v: Vec3) -> Double { (v * v).sum().squareRoot() }

    func vec(_ a: Any?) -> Vec3 {
        let v = a as! [Double]
        return Vec3(v[0], v[1], v[2])
    }

    // MARK: Maskenwerkzeuge (ohne PC)

    func randomMask(width: Int, height: Int, seed: UInt64) -> [Bool] {
        var state = seed
        return (0..<(width * height)).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (state >> 33) % 7 == 0
        }
    }

    /// Abstand wie mit roher Gewalt gerechnet.
    func testDistanceOutsideIsExact() {
        let w = 23, h = 17
        let mask = randomMask(width: w, height: h, seed: 3)
        let dt = MaskTools.distanceOutside(mask, width: w, height: h)
        for y in 0..<h {
            for x in 0..<w {
                var best = Double.infinity
                for yy in 0..<h {
                    for xx in 0..<w where mask[yy * w + xx] {
                        best = min(best, Double((x - xx) * (x - xx) + (y - yy) * (y - yy)).squareRoot())
                    }
                }
                XCTAssertEqual(dt[y * w + x], best, accuracy: 1e-9, "Pixel \(x), \(y)")
            }
        }
    }

    func testLargestComponentAndBoundary() {
        let w = 12, h = 8
        var mask = [Bool](repeating: false, count: w * h)
        for y in 1...5 { for x in 1...4 { mask[y * w + x] = true } }  // 20 Pixel
        for y in 2...3 { for x in 8...10 { mask[y * w + x] = true } }  // 6 Pixel
        mask[6 * w + 5] = true  // berührt die große Fläche diagonal
        let big = MaskTools.largestComponent(mask, width: w, height: h)
        XCTAssertEqual(big.filter { $0 }.count, 21)
        XCTAssertFalse(big[2 * w + 8])
        let edge = MaskTools.boundaryPoints(big, width: w, height: h, count: 400)
        XCTAssertEqual(edge.count, 21 - 6)  // innen liegen 2 × 3 Pixel
        let few = MaskTools.boundaryPoints(big, width: w, height: h, count: 4)
        XCTAssertEqual(few.count, 4)
        XCTAssertEqual(few.first, edge.first)
        XCTAssertEqual(few.last, edge.last)
    }

    func testResizeAveragesAreas() {
        let mask: [Float] = [1, 1, 0, 0,
                             1, 0, 0, 0,
                             1, 1, 1, 0,
                             1, 1, 0, 1]
        XCTAssertEqual(MaskTools.resize(mask, width: 4, height: 4, toWidth: 2, height: 2),
                       [true, false, true, false])
        XCTAssertEqual(MaskTools.resize(mask, width: 4, height: 4, toWidth: 4, height: 4), mask.map { $0 > 0.5 })
    }

    // MARK: Gegen den PC

    func testSettingsLoad() {
        let s = Self.settings
        XCTAssertEqual(s.keypoints.count, 12)
        XCTAssertFalse(s.poseBones.isEmpty)
        for name in s.poseBones + s.keypoints.values.map(\.bone) {
            XCTAssertNotNil(model.data.bone(name), name)
        }
        for name in s.videoLocals { XCTAssertTrue(model.data.localLabels.contains(name), name) }
    }

    func testPreparedViewMatchesPC() throws {
        let ref = try XCTUnwrap(reference["prepared"] as? [String: Any])
        let view = try model.prepareView(frames()[0], settings: Self.settings)
        XCTAssertEqual(view.width, ref["width"] as? Int)
        XCTAssertEqual(view.height, ref["height"] as? Int)
        XCTAssertEqual(view.focal, ref["focal"] as! Double, accuracy: 1e-9)
        XCTAssertEqual(view.pxHeight, ref["px_height"] as! Double)
        let center = ref["bbox_center"] as! [Double]
        XCTAssertEqual(view.bboxCenter.x, center[0])
        XCTAssertEqual(view.bboxCenter.y, center[1])
        XCTAssertEqual(view.mask.filter { $0 }.count, ref["pixels"] as? Int)
        XCTAssertEqual(view.contour.count, ref["contour_count"] as? Int)
        for (i, p) in (ref["contour"] as! [[Double]]).enumerated() {
            XCTAssertEqual(view.contour[20 * i], SIMD2(p[0], p[1]), "Randpunkt \(20 * i)")
        }
        for s in ref["dt"] as! [[Double]] {
            let x = Int(s[0]), y = Int(s[1])
            XCTAssertEqual(view.dt[y * view.width + x], s[2], accuracy: 1e-3, "Abstand bei \(x), \(y)")
        }
        XCTAssertEqual(view.kpBones.map { model.data.boneLabels[$0] }, ref["kp_bones"] as! [String])
    }

    /// Haltung, Bildfehler und alle Gradienten an einem festen Zustand wie `photofit` (numpy).
    func testTermsAndGradientsMatchPC() throws {
        let ref = try XCTUnwrap(reference["terms"] as? [String: Any])
        let view = try model.prepareView(frames()[0], settings: Self.settings)
        let rig = try PoseRig(data: model.data, bones: Self.settings.poseBones)
        let pose = (ref["pose"] as! [[Double]]).map { Vec3($0[0], $0[1], $0[2]) }
        let rest = model.rest(BodyParams())
        let m = rig.transforms(heads: rest.boneHeads, pose: pose)
        let (posed, blend) = rig.skin(rest.vertices, m)
        let sample = ref["sample"] as! [Int]
        for (i, p) in (ref["posed"] as! [[Double]]).enumerated() {
            XCTAssertLessThan(norm(posed[sample[i]] - Vec3(p[0], p[1], p[2])), 1e-7, "Punkt \(sample[i])")
        }
        let kp = view.kpBones.map { m[$0].apply(rest.boneHeads[$0]) }
        let yaw = ref["yaw"] as! Double
        let trans = alignTranslation(posed, yaw: yaw, view: view)
        XCTAssertLessThan(norm(trans - vec(ref["align_trans"])), 1e-6)

        let terms = viewTerms(posed, kp, yaw: yaw, trans: vec(ref["align_trans"]), view: view,
                              keypointWeight: ref["keypoint_weight"] as! Double)
        let loss = ref["loss"] as! Double
        XCTAssertEqual(terms.loss, loss, accuracy: 1e-5 * loss)
        let gYaw = ref["g_yaw"] as! Double
        XCTAssertEqual(terms.yaw, gYaw, accuracy: 1e-4 * abs(gYaw) + 1e-12)
        let gTrans = vec(ref["g_trans"])
        XCTAssertLessThan(norm(terms.trans - gTrans), 1e-4 * norm(gTrans) + 1e-12)
        func close(_ mine: [Vec3], _ theirs: [[Double]], _ index: (Int) -> Int, _ what: String) {
            let scale = theirs.map { norm(Vec3($0[0], $0[1], $0[2])) }.max() ?? 1
            var worst = 0.0
            for (i, t) in theirs.enumerated() { worst = max(worst, norm(mine[index(i)] - Vec3(t[0], t[1], t[2]))) }
            XCTAssertLessThan(worst, 1e-3 * scale + 1e-12, what)
        }
        close(terms.points, ref["g_points"] as! [[Double]], { sample[$0] }, "Gradient nach Punkten")
        close(terms.keypoints, ref["g_keypoints"] as! [[Double]], { $0 }, "Gradient nach Körperpunkten")

        let (gRest, gPose) = rig.gradient(rest: rest.vertices, heads: rest.boneHeads, pose: pose, transforms: m,
                                          blend: blend, points: terms.points, keypointBones: view.kpBones,
                                          keypoints: terms.keypoints)
        close(gRest, ref["g_rest"] as! [[Double]], { sample[$0] }, "Gradient nach Ruhepunkten")
        close(gPose, ref["g_pose"] as! [[Double]], { $0 }, "Gradient nach der Haltung")
    }

    /// Ganzer Scan aus vier künstlichen Bildern: dieselben Maße und Drehungen wie am PC.
    func testFitMatchesPC() throws {
        let ref = try XCTUnwrap(reference["fit"] as? [String: Any])
        let p = try XCTUnwrap(reference["person"] as? [String: Any])
        var measures: [MeasureKey: Double] = [:]
        for (k, v) in p["measures"] as! [String: Double] { measures[MeasureKey(rawValue: k)!] = v }
        let person = Person(gender: Gender(rawValue: p["gender"] as! String)!, ageYears: p["age_years"] as! Double,
                            measures: measures)
        let result = try model.fitPhotos(person, frames: frames(), settings: Self.settings)
        let pcMeasures = ref["measures"] as! [String: Double]
        for (k, pc) in pcMeasures {
            guard let key = MeasureKey(rawValue: k), let mine = result.measures[key] else { continue }
            XCTAssertEqual(mine, pc, accuracy: 1.0, "\(key.label): App \(mine), PC \(pc)")
        }
        let truth = reference["truth"] as! [String: Double]
        for key in MeasureKey.circumferences {
            guard let mine = result.measures[key], let t = truth[key.rawValue] else { continue }
            XCTAssertEqual(mine, t, accuracy: 3.5, "\(key.label) gegen den wahren Körper")
        }
        let pcYaws = ref["yaw_deg"] as! [Double]
        for (i, view) in result.views.enumerated() {
            var d = abs(view.yawDegrees - pcYaws[i]).truncatingRemainder(dividingBy: 360)
            d = min(d, 360 - d)
            XCTAssertLessThan(d, 5, "Bild \(i + 1): Drehung \(view.yawDegrees)° statt \(pcYaws[i])°")
            XCTAssertGreaterThan(view.iou, 0.9, "Bild \(i + 1)")
            XCTAssertEqual(view.overlay.count, 3 * view.width * view.height)
        }
    }
}
