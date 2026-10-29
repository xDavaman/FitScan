import Foundation

// MARK: - Scan-Assistent (Kamera-Scan mit Ampel)

/// Ampelstufe eines Kamerabilds, wie in `quality.py`.
public enum ScanLevel: String, Decodable, Comparable {
    case gruen, gelb, rot

    private var rank: Int {
        switch self {
        case .gruen: return 0
        case .gelb: return 1
        case .rot: return 2
        }
    }

    public static func < (a: ScanLevel, b: ScanLevel) -> Bool { a.rank < b.rank }
}

/// Ein Hinweis an die Person vor der Kamera.
public struct ScanHint: Equatable, Identifiable {
    public let code: String
    public let text: String
    public let level: ScanLevel
    public var id: String { code }
}

/// Grobe Blickrichtung der Person zur Kamera.
public enum BodyView: String {
    case vorn, seite, ruecken
}

/// Was die App aus einem Kamerabild misst, auf einem Raster (x nach rechts, y nach unten).
public struct LiveFrame {
    public var width: Int
    public var height: Int
    /// Helligkeit 0…255 je Rasterpunkt
    public var luma: [UInt8]
    /// Personenmaske 0…1 je Rasterpunkt
    public var mask: [Float]
    /// Körperpunkte in Rasterpixeln mit MediaPipe-Nummern (wie `PhotoFrame`), z = Sicherheit 0…1
    public var keypoints: [Int: SIMD3<Double>]
    /// Erkannte Personen
    public var persons: Int
    /// Schärfe: Varianz des Laplace-Filters auf 400 px Personenhöhe, aus dem vollen Bild (`ScanRules.sharpness`)
    public var sharpness: Double?

    public init(width: Int, height: Int, luma: [UInt8], mask: [Float], keypoints: [Int: SIMD3<Double>],
                persons: Int, sharpness: Double?) {
        self.width = width
        self.height = height
        self.luma = luma
        self.mask = mask
        self.keypoints = keypoints
        self.persons = persons
        self.sharpness = sharpness
    }
}

/// Ergebnis der Prüfung eines Kamerabilds.
public struct LiveCheck {
    public var level: ScanLevel
    public var hints: [ScanHint]
    public var metrics: [String: Double]
    public var view: BodyView?
}

/// Grenzwerte und Hinweistexte des Scan-Assistenten (`scanregeln.json`, aus `quality.py`).
/// Live fehlen die Regeln, die Farbe oder feine Kanten im vollen Bild brauchen (Farbstich,
/// Socken, Kontrast, Hintergrund); sie prüft der PC beim Scan aus einem Video.
public struct ScanRules: Decodable {
    public struct HintText: Decodable {
        public let text: String
        public let level: ScanLevel
    }

    public let limits: [String: Double]
    public let hints: [String: HintText]
    /// Nur in der App: Blickrichtung und Ablauf der Aufnahme (`swiftdata.LIVE_RULES`)
    public let live: [String: Double]

    public static func bundled() throws -> ScanRules {
        guard let url = Bundle.module.url(forResource: "scanregeln", withExtension: "json") else {
            throw FitScanError.data("Die Regeln des Scan-Assistenten fehlen in der App (scanregeln.json).")
        }
        return try JSONDecoder().decode(ScanRules.self, from: Data(contentsOf: url))
    }

    public func limit(_ key: String) -> Double { limits[key] ?? .nan }

    public func setting(_ key: String) -> Double { live[key] ?? .nan }

    public func hint(_ code: String) -> ScanHint {
        let known = hints[code]
        return ScanHint(code: code, text: known?.text ?? code, level: known?.level ?? .gelb)
    }

    /// Prüft ein Kamerabild wie `quality.check_frame`. `steady`: auch Stillhalten prüfen (vor dem
    /// Drehen); beim Drehen ändern sich die Armwinkel im Bild von selbst.
    public func check(_ frame: LiveFrame, previous: [Int: SIMD3<Double>]? = nil, steady: Bool = true) -> LiveCheck {
        var hints: [ScanHint] = []
        var metrics: [String: Double] = [:]
        func add(_ code: String) { hints.append(hint(code)) }
        let w = frame.width, h = frame.height
        let kp = frame.keypoints
        guard !kp.isEmpty, w > 0, h > 0, frame.luma.count == w * h, frame.mask.count == w * h else {
            add("keine_person")
            return LiveCheck(level: .rot, hints: hints, metrics: metrics, view: nil)
        }
        if frame.persons > 1 { add("mehrere_personen") }

        // Person: Fläche, Rahmen, Helligkeit
        var count = 0, sum = 0.0
        var x0 = w, y0 = h, x1 = -1, y1 = -1
        for y in 0..<h {
            for x in 0..<w where frame.mask[y * w + x] > 0.5 {
                count += 1
                sum += Double(frame.luma[y * w + x])
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
            }
        }

        // Licht
        if count > 500 {
            let personLuma = sum / Double(count)
            let near = Self.dilated(frame.mask, width: w, height: h, radius: max(1, Int((12 * Double(h) / 1920).rounded())))
            var bgSum = 0.0, bgCount = 0
            for i in 0..<(w * h) where !near[i] {
                bgSum += Double(frame.luma[i])
                bgCount += 1
            }
            let backgroundLuma = bgCount > 0 ? bgSum / Double(bgCount) : personLuma
            metrics["person_luma"] = personLuma
            metrics["background_luma"] = backgroundLuma
            if personLuma < limit("min_person_luma") {
                add("dunkel")
            } else if personLuma > limit("max_person_luma") {
                add("hell")
            }
            if backgroundLuma - personLuma > limit("max_backlight") && personLuma < limit("backlight_person_luma") {
                add("gegenlicht")
            }
            // Glanzlichter nur über den Knien (helle Socken zählen nicht)
            var upperEnd = h
            if let lk = kp[25], let rk = kp[26], min(lk.z, rk.z) > 0.5 {
                upperEnd = min(h, max(0, Int(max(0, lk.y, rk.y))))
            }
            var upper = 0, shiny = 0
            for y in 0..<upperEnd {
                for x in 0..<w where frame.mask[y * w + x] > 0.5 {
                    upper += 1
                    if Double(frame.luma[y * w + x]) > limit("highlight_luma") { shiny += 1 }
                }
            }
            if upper > 500 {
                let share = Double(shiny) / Double(upper)
                metrics["highlight_share"] = share
                if share > limit("max_highlight_share") { add("glanzlicht") }
            }
        }

        // Schärfe, Position und Abstand
        if count > 500 {
            if let sharp = frame.sharpness {
                metrics["sharpness"] = sharp
                if sharp < limit("min_sharpness") { add("unscharf") }
            }
            let fraction = Double(y1 - y0) / Double(h)
            metrics["height_fraction"] = fraction
            let border = max(1, Int((2 * Double(h) / 1920).rounded()))
            let cut = y0 < border || y1 >= h - border || (kp[0].map { $0.z > 0.5 && $0.y < 0 } ?? false)
            if cut {
                add("abgeschnitten")
            } else if fraction > limit("max_height_fraction") {
                add("zu_nah")
            } else if fraction < limit("min_height_fraction") {
                add("zu_weit")
            }
            let offset = abs(Double(x0 + x1) / 2 - Double(w) / 2) / Double(w)
            metrics["center_offset"] = offset
            if offset > limit("max_center_offset") { add("mitte") }
        }

        // Haltung, nur von vorn oder hinten sinnvoll
        let view = Self.bodyView(kp, sideRatio: setting("side_shoulder_ratio"))
        func point(_ points: [Int: SIMD3<Double>], _ i: Int) -> SIMD2<Double>? {
            guard let p = points[i], p.z > 0.5 else { return nil }
            return SIMD2(p.x, p.y)
        }
        if view != .seite, let ls = point(kp, 11), let rs = point(kp, 12), let le = point(kp, 13),
           let re = point(kp, 14) {
            let down = SIMD2<Double>(0, 1)
            let la = Self.angleDeg(le - ls, down), ra = Self.angleDeg(re - rs, down)
            metrics["arm_left_deg"] = la
            metrics["arm_right_deg"] = ra
            if min(la, ra) < limit("arm_angle_min_deg") {
                add("arme_anliegend")
            } else if max(la, ra) > limit("arm_angle_max_deg") {
                add("arme_hoch")
            }
            if let lw = point(kp, 15), let rw = point(kp, 16) {
                let lElbow = Self.angleDeg(ls - le, lw - le), rElbow = Self.angleDeg(rs - re, rw - re)
                metrics["elbow_left_deg"] = lElbow
                metrics["elbow_right_deg"] = rElbow
                if min(lElbow, rElbow) < limit("min_elbow_deg") { add("arme_gebeugt") }
            }
            if let lAnkle = point(kp, 27), let rAnkle = point(kp, 28), let lh = point(kp, 23), let rh = point(kp, 24) {
                let ratio = abs(lAnkle.x - rAnkle.x) / max(abs(lh.x - rh.x), 1)
                metrics["feet_ratio"] = ratio
                if ratio < limit("feet_ratio_min") {
                    add("fuesse_eng")
                } else if ratio > limit("feet_ratio_max") {
                    add("fuesse_weit")
                }
            }
            if steady, let previous, let pls = previous[11], let prs = previous[12], let ple = previous[13],
               let pre = previous[14] {
                let pla = Self.angleDeg(SIMD2(ple.x - pls.x, ple.y - pls.y), down)
                let pra = Self.angleDeg(SIMD2(pre.x - prs.x, pre.y - prs.y), down)
                let change = max(abs(la - pla), abs(ra - pra))
                metrics["arm_change_deg"] = change
                if change > limit("max_arm_change_deg") { add("stillhalten") }
            }
        }

        let level = hints.map(\.level).max() ?? .gruen
        return LiveCheck(level: level, hints: hints, metrics: metrics, view: view)
    }

    /// Blickrichtung aus 2D-Punkten: Schulterbreite geteilt durch Rumpflänge, mit Vorzeichen. Schaut
    /// die Person zur Kamera, liegt ihre linke Schulter rechts im Bild (Bild nicht gespiegelt).
    /// Ohne Nase und Augen gilt die Ansicht als Rücken, auch wenn links und rechts vertauscht erkannt
    /// wurden (MediaPipe meldet das Gesicht auch von hinten, dort ändert das nichts).
    public static func bodyView(_ kp: [Int: SIMD3<Double>], sideRatio: Double) -> BodyView? {
        guard let ls = kp[11], let rs = kp[12], let lh = kp[23], let rh = kp[24],
              min(ls.z, rs.z, lh.z, rh.z) > 0.3 else { return nil }
        let dx = (ls.x + rs.x - lh.x - rh.x) / 2, dy = (ls.y + rs.y - lh.y - rh.y) / 2
        let torso = (dx * dx + dy * dy).squareRoot()
        guard torso > 1 else { return nil }
        let signed = (ls.x - rs.x) / torso
        if abs(signed) < sideRatio { return .seite }
        let face = [0, 2, 5].contains { (kp[$0]?.z ?? 0) > 0.3 }
        return signed > 0 && face ? .vorn : .ruecken
    }

    /// Rahmen der Person (Maske > 0,5) im Raster, falls vorhanden.
    public static func personBox(_ mask: [Float], width w: Int, height h: Int) -> (x0: Int, y0: Int, x1: Int, y1: Int)? {
        var x0 = w, y0 = h, x1 = -1, y1 = -1
        for y in 0..<h {
            for x in 0..<w where mask[y * w + x] > 0.5 {
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
            }
        }
        return x1 >= 0 ? (x0, y0, x1, y1) : nil
    }

    /// Schärfe eines Ausschnitts wie am PC: auf `targetHeight` Zeilen verkleinert (Flächenmittel),
    /// dann Varianz des Laplace-Filters. Kleinere Ausschnitte bleiben, wie sie sind.
    public static func sharpness(_ gray: [Float], width w: Int, height h: Int, targetHeight: Int = 400) -> Double {
        guard w > 1, h > 1, gray.count == w * h else { return 0 }
        guard h > targetHeight else { return laplaceVariance(gray, width: w, height: h) }
        let f = Double(targetHeight) / Double(h)
        let ow = max(2, Int((Double(w) * f).rounded())), oh = targetHeight
        var out = [Float](repeating: 0, count: ow * oh)
        for oy in 0..<oh {
            let sy0 = oy * h / oh, sy1 = max(sy0 + 1, (oy + 1) * h / oh)
            for ox in 0..<ow {
                let sx0 = ox * w / ow, sx1 = max(sx0 + 1, (ox + 1) * w / ow)
                var s: Float = 0
                for y in sy0..<sy1 {
                    for x in sx0..<sx1 { s += gray[y * w + x] }
                }
                out[oy * ow + ox] = s / Float((sy1 - sy0) * (sx1 - sx0))
            }
        }
        return laplaceVariance(out, width: ow, height: oh)
    }

    /// Varianz des Laplace-Filters (3×3, Rand gespiegelt wie OpenCV `BORDER_REFLECT_101`).
    public static func laplaceVariance(_ gray: [Float], width w: Int, height h: Int) -> Double {
        guard w > 1, h > 1, gray.count == w * h else { return 0 }
        func reflect(_ i: Int, _ n: Int) -> Int { i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i) }
        var sum = 0.0, squares = 0.0
        for y in 0..<h {
            let up = reflect(y - 1, h), down = reflect(y + 1, h)
            for x in 0..<w {
                let left = reflect(x - 1, w), right = reflect(x + 1, w)
                let v = Double(gray[up * w + x]) + Double(gray[down * w + x]) + Double(gray[y * w + left])
                    + Double(gray[y * w + right]) - 4 * Double(gray[y * w + x])
                sum += v
                squares += v * v
            }
        }
        let n = Double(w * h), mean = sum / n
        return squares / n - mean * mean
    }

    static func angleDeg(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let c = (a * b).sum() / (((a * a).sum()).squareRoot() * ((b * b).sum()).squareRoot() + 1e-9)
        return acos(max(-1, min(1, c))) * 180 / .pi
    }

    /// Maske (> 0,5) um `radius` Rasterpunkte in jede Richtung erweitert (Quadrat, wie `cv2.dilate`).
    static func dilated(_ mask: [Float], width w: Int, height h: Int, radius r: Int) -> [Bool] {
        var rows = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            var last = -r - 1  // letzte Person links von x
            var next = [Int](repeating: Int.max, count: w)  // nächste Person ab x nach rechts
            var upcoming = Int.max
            for x in stride(from: w - 1, through: 0, by: -1) {
                if mask[y * w + x] > 0.5 { upcoming = x }
                next[x] = upcoming
            }
            for x in 0..<w {
                if mask[y * w + x] > 0.5 { last = x }
                rows[y * w + x] = x - last <= r || next[x] - x <= r
            }
        }
        var out = [Bool](repeating: false, count: w * h)
        for x in 0..<w {
            var last = -r - 1
            var next = [Int](repeating: Int.max, count: h)
            var upcoming = Int.max
            for y in stride(from: h - 1, through: 0, by: -1) {
                if rows[y * w + x] { upcoming = y }
                next[y] = upcoming
            }
            for y in 0..<h {
                if rows[y * w + x] { last = y }
                out[y * w + x] = y - last <= r || next[y] - y <= r
            }
        }
        return out
    }
}

/// Verfolgt eine Drehung um die eigene Achse: vorn, Seite, Rücken, Seite und wieder vorn.
public struct TurnTracker {
    public static let steps: [BodyView] = [.vorn, .seite, .ruecken, .seite, .vorn]

    /// Ansichten in der Reihenfolge, in der sie sicher erkannt wurden (ohne direkte Wiederholung)
    public private(set) var seen: [BodyView] = []
    private var candidate: BodyView?
    private var streak = 0
    let confirm: Int

    /// `confirm`: so viele Prüfungen in Folge, bis eine Ansicht zählt. Die Runde beginnt von vorn.
    public init(confirm: Int = 2, start: BodyView? = .vorn) {
        self.confirm = max(1, confirm)
        if let start { seen = [start] }
    }

    public mutating func add(_ view: BodyView?) {
        guard let view else { return }
        if view == candidate {
            streak += 1
        } else {
            candidate = view
            streak = 1
        }
        if streak >= confirm && seen.last != view { seen.append(view) }
    }

    /// Geschaffte Schritte der Runde (0 bis 5)
    public var progress: Int {
        var k = 0
        for v in seen where k < Self.steps.count && v == Self.steps[k] { k += 1 }
        return k
    }

    public var complete: Bool { progress == Self.steps.count }
}
