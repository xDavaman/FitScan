import AVFoundation
import CoreImage
import UIKit
import Vision
import FitScanCore

/// Kamera für den Scan: liefert das Bild für die Vorschau, misst alle `interval` Sekunden Person,
/// Körperpunkte, Helligkeit und Schärfe (Apple Vision, auf dem iPhone) und nimmt auf Anfrage einzelne
/// Bilder als JPEG auf. Bilder und Messung laufen auf der eigenen Warteschlange `queue`.
final class CameraFeed: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    /// Blickwinkel über die lange Bildseite (hochkant die Höhe), für die Brennweite der Anpassung
    private(set) var fovLongSideDeg = 69.0
    /// Neues Messergebnis; wird auf `queue` aufgerufen. Vor `start` setzen.
    var onFrame: ((LiveFrame) -> Void)?
    var interval = 0.2

    /// Vision-Gelenke für die Live-Prüfung, dazu Gesicht (MediaPipe-Nummern)
    static let joints: [(Int, VNHumanBodyPoseObservation.JointName)] =
        [(0, .nose), (2, .leftEye), (5, .rightEye), (7, .leftEar), (8, .rightEar)] + PersonDetector.joints
    /// Zeilen des Messrasters (Breite nach Seitenverhältnis, hochkant 270)
    static let gridHeight = 480

    private let sessionQueue = DispatchQueue(label: "fitscan.kamera.sitzung")
    private let queue = DispatchQueue(label: "fitscan.kamera.bilder", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private let segmentation: VNGeneratePersonSegmentationRequest = {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }()
    private let pose = VNDetectHumanBodyPoseRequest()
    private let context = CIContext()
    private var lastMeasure = 0.0
    private var wantCapture = false
    private var captured: [Data] = []

    /// Kamera einrichten, vorn (Selfie) oder hinten.
    func configure(front: Bool) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.setUp(front: front)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func setUp(front: Bool) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                   position: front ? .front : .back) else {
            throw FitScanError.data("Keine Kamera gefunden.")
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw FitScanError.data("Die Kamera lässt sich nicht verwenden.") }
        session.addInput(input)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw FitScanError.data("Die Kamera lässt sich nicht verwenden.") }
        session.addOutput(output)
        // Bilder hochkant wie die App; nicht gespiegelt, damit links und rechts der Person stimmen
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
        fovLongSideDeg = Double(device.activeFormat.videoFieldOfView)
    }

    func start() {
        sessionQueue.async {
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Das nächste Kamerabild als JPEG aufheben.
    func captureNext() {
        queue.async { self.wantCapture = true }
    }

    /// Aufgenommene Bilder abholen (danach ist die Liste leer).
    func takeCaptured() async -> [Data] {
        await withCheckedContinuation { continuation in
            queue.async {
                let frames = self.captured
                self.captured = []
                self.wantCapture = false
                continuation.resume(returning: frames)
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if wantCapture {
            wantCapture = false
            if let space = CGColorSpace(name: CGColorSpace.sRGB),
               let jpeg = context.jpegRepresentation(of: CIImage(cvPixelBuffer: buffer), colorSpace: space, options: [:]) {
                captured.append(jpeg)
            }
        }
        let now = CACurrentMediaTime()
        guard now - lastMeasure >= interval else { return }
        lastMeasure = now
        if let frame = measure(buffer) { onFrame?(frame) }
    }

    /// Raster mit Helligkeit und Personenmaske, Körperpunkte und Schärfe eines Kamerabilds.
    private func measure(_ buffer: CVPixelBuffer) -> LiveFrame? {
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up, options: [:])
        do {
            try handler.perform([segmentation, pose])
        } catch {
            return nil
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let fw = CVPixelBufferGetWidthOfPlane(buffer, 0), fh = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        guard fw > 1, fh > 1, let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
        let plane = base.assumingMemoryBound(to: UInt8.self)  // Helligkeit (Y), volle Auflösung
        let gh = Self.gridHeight, gw = max(1, Int((Double(gh) * Double(fw) / Double(fh)).rounded()))

        var luma = [UInt8](repeating: 0, count: gw * gh)
        for gy in 0..<gh {
            let sy = min(fh - 1, Int((Double(gy) + 0.5) * Double(fh) / Double(gh)))
            for gx in 0..<gw {
                let sx = min(fw - 1, Int((Double(gx) + 0.5) * Double(fw) / Double(gw)))
                luma[gy * gw + gx] = plane[sy * rowBytes + sx]
            }
        }
        // Vision liefert die Maske in eigener Größe über das ganze Bild
        var mask = [Float](repeating: 0, count: gw * gh)
        if let maskBuffer = segmentation.results?.first?.pixelBuffer {
            let (values, mw, mh) = PersonDetector.maskValues(maskBuffer)
            if mw > 0, mh > 0 {
                for gy in 0..<gh {
                    let my = min(mh - 1, Int((Double(gy) + 0.5) * Double(mh) / Double(gh)))
                    for gx in 0..<gw {
                        let mx = min(mw - 1, Int((Double(gx) + 0.5) * Double(mw) / Double(gw)))
                        mask[gy * gw + gx] = values[my * mw + mx]
                    }
                }
            }
        }

        let observations = pose.results ?? []
        let persons = observations.filter { observation in
            let points = (try? observation.recognizedPoints(.all)) ?? [:]
            return points.values.filter { $0.confidence > 0.3 }.count >= 4
        }.count
        var keypoints: [Int: SIMD3<Double>] = [:]
        if let best = PersonDetector.bestObservation(observations), let points = try? best.recognizedPoints(.all) {
            for (index, joint) in Self.joints {
                guard let p = points[joint], p.confidence > 0 else { continue }
                keypoints[index] = SIMD3(Double(p.location.x) * Double(gw), (1 - Double(p.location.y)) * Double(gh),
                                         Double(p.confidence))
            }
        }

        // Schärfe im Personenbereich aus dem vollen Bild
        var sharpness: Double?
        if let box = ScanRules.personBox(mask, width: gw, height: gh) {
            let x0 = box.x0 * fw / gw, x1 = min(fw, (box.x1 + 1) * fw / gw)
            let y0 = box.y0 * fh / gh, y1 = min(fh, (box.y1 + 1) * fh / gh)
            let cw = x1 - x0, ch = y1 - y0
            if cw > 1, ch > 1 {
                var crop = [Float](repeating: 0, count: cw * ch)
                for r in 0..<ch {
                    for c in 0..<cw { crop[r * cw + c] = Float(plane[(y0 + r) * rowBytes + x0 + c]) }
                }
                sharpness = ScanRules.sharpness(crop, width: cw, height: ch)
            }
        }
        return LiveFrame(width: gw, height: gh, luma: luma, mask: mask, keypoints: keypoints, persons: persons,
                         sharpness: sharpness)
    }
}

/// Ablauf des Kamera-Scans: Ampel bis alles passt, Countdown, eine Drehung mit Aufnahme, fertig.
/// Hinweise kommen auch als Ansage, weil man das Display beim Drehen nicht sieht.
@MainActor
final class CameraScanModel: ObservableObject {
    enum Phase: Equatable {
        case starting
        case denied
        case failed(String)
        /// Ampel: warten, bis Licht, Abstand und Haltung passen
        case setup
        case countdown(Int)
        /// Drehung läuft, Bilder werden aufgenommen
        case turning
        case done
    }

    @Published private(set) var phase: Phase = .starting
    @Published private(set) var check: LiveCheck?
    /// Geschaffte Schritte der Drehung (0 bis 5: vorn, Seite, Rücken, Seite, vorn)
    @Published private(set) var progress = 0
    @Published private(set) var captured = 0
    /// Wird mit den aufgenommenen Bildern und dem Blickwinkel der Kamera (lange Seite, Grad) aufgerufen,
    /// wenn die Drehung fertig ist
    var onFinished: (@MainActor ([Data], Double) -> Void)?

    let feed = CameraFeed()
    private let rules = try? ScanRules.bundled()
    private let voice = AVSpeechSynthesizer()
    private var speaking = true
    private var tracker = TurnTracker()
    private var previous: [Int: SIMD3<Double>]?
    private var greenSince: Date?
    private var usableSince: Date?
    private var turnStart: Date?
    private var lastCapture: Date?
    private var lastSpoken: (text: String, at: Date)?
    private var countdown: Task<Void, Never>?
    private var stopped = false

    var fovLongSideDeg: Double { feed.fovLongSideDeg }

    private func setting(_ key: String, _ fallback: Double) -> Double {
        guard let value = rules?.setting(key), !value.isNaN else { return fallback }
        return value
    }

    func start(front: Bool, speak: Bool) async {
        speaking = speak
        stopped = false
        reset()
        phase = .starting
        guard rules != nil else {
            phase = .failed("Die Regeln des Scan-Assistenten fehlen in der App.")
            return
        }
        var allowed = AVCaptureDevice.authorizationStatus(for: .video) == .authorized
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            allowed = await AVCaptureDevice.requestAccess(for: .video)
        }
        guard !stopped else { return }
        guard allowed else {
            phase = .denied
            return
        }
        do {
            try await feed.configure(front: front)
        } catch {
            phase = .failed(error.localizedDescription)
            return
        }
        guard !stopped else { return }  // inzwischen abgebrochen
        feed.interval = setting("check_interval_s", 0.2)
        feed.onFrame = Self.forward(to: self)
        _ = await feed.takeCaptured()  // Reste einer früheren Aufnahme
        guard !stopped else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        feed.start()
        phase = .setup
        say("Stell dich etwa zwei Meter vor die Kamera, der ganze Körper im Bild, Arme leicht abgespreizt.", force: true)
    }

    /// Kamera und Ansage beenden, z. B. beim Abbrechen.
    func stop() {
        stopped = true
        countdown?.cancel()
        countdown = nil
        feed.onFrame = nil
        feed.stop()
        voice.stopSpeaking(at: .immediate)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Drehung von Hand beenden.
    func finishNow() {
        guard phase == .turning else { return }
        finish()
    }

    private func reset() {
        check = nil
        progress = 0
        captured = 0
        previous = nil
        greenSince = nil
        usableSince = nil
        turnStart = nil
        lastCapture = nil
        lastSpoken = nil
    }

    /// Ergebnisse vom Kamera-Thread auf den Main Actor holen. Außerhalb des Main Actors gebaut,
    /// damit der Abschluss nicht an ihn gebunden ist.
    nonisolated private static func forward(to model: CameraScanModel) -> (LiveFrame) -> Void {
        { [weak model] frame in
            Task { @MainActor in model?.apply(frame) }
        }
    }

    private func apply(_ frame: LiveFrame) {
        guard let rules else { return }
        let result = rules.check(frame, previous: previous, steady: phase == .setup)
        if !frame.keypoints.isEmpty { previous = frame.keypoints }
        check = result
        let now = Date()
        switch phase {
        case .setup:
            greenSince = result.level == .gruen ? (greenSince ?? now) : nil
            usableSince = result.level != .rot ? (usableSince ?? now) : nil
            let green = greenSince.map { now.timeIntervalSince($0) >= setting("green_hold_s", 1.5) } ?? false
            let usable = usableSince.map { now.timeIntervalSince($0) >= setting("yellow_start_s", 8) } ?? false
            if green || usable {
                beginCountdown()
            } else {
                sayHint(result)
            }
        case .countdown:
            if result.level == .rot {
                countdown?.cancel()
                countdown = nil
                greenSince = nil
                usableSince = nil
                phase = .setup
                sayHint(result, force: true)
            }
        case .turning:
            tracker.add(result.view)
            progress = tracker.progress
            if result.level != .rot,
               lastCapture.map({ now.timeIntervalSince($0) >= setting("capture_interval_s", 0.75) }) ?? true {
                lastCapture = now
                feed.captureNext()
                captured += 1
            }
            let elapsed = turnStart.map { now.timeIntervalSince($0) } ?? 0
            if tracker.complete || elapsed > setting("max_turn_s", 45) {
                finish()
            } else if result.level == .rot {
                sayHint(result)
            }
        default:
            break
        }
    }

    private func beginCountdown() {
        phase = .countdown(3)
        say("Gut so. Bleib so stehen.", force: true)
        countdown = Task { [weak self] in
            for (n, word) in [(3, "Drei"), (2, "Zwei"), (1, "Eins")] {
                guard let self, !Task.isCancelled else { return }
                self.phase = .countdown(n)
                self.say(word, force: true)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard let self, !Task.isCancelled else { return }
            self.startTurning()
        }
    }

    private func startTurning() {
        tracker = TurnTracker(confirm: Int(setting("confirm_checks", 2)), start: .vorn)
        progress = tracker.progress
        turnStart = Date()
        lastCapture = nil
        phase = .turning
        say("Jetzt langsam einmal ganz um dich selbst drehen.", force: true)
    }

    private func finish() {
        phase = .done
        say(tracker.complete ? "Fertig, danke!" : "Aufnahme beendet.", force: true)
        let maxFrames = Int(setting("max_frames", 24))
        Task { [weak self] in
            guard let self else { return }
            let frames = await self.feed.takeCaptured()
            self.feed.onFrame = nil
            self.feed.stop()
            // gleichmäßig über die Drehung verteilt
            let kept = frames.count > maxFrames
                ? (0..<maxFrames).map { frames[$0 * frames.count / maxFrames] } : frames
            // die Ansage ausreden lassen, dann weiter zur Prüfung der Bilder
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !self.stopped else { return }
            self.onFinished?(kept, self.fovLongSideDeg)
        }
    }

    private func sayHint(_ result: LiveCheck, force: Bool = false) {
        guard let hint = result.hints.first(where: { $0.level == result.level }) else { return }
        say(hint.text, force: force)
    }

    /// Ansage, höchstens alle 3 Sekunden und denselben Satz frühestens nach 8 Sekunden wieder.
    private func say(_ text: String, force: Bool = false) {
        guard speaking else { return }
        let now = Date()
        if !force, let last = lastSpoken {
            let since = now.timeIntervalSince(last.at)
            if since < 3 || (last.text == text && since < 8) { return }
        }
        lastSpoken = (text, now)
        if force { voice.stopSpeaking(at: .word) }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "de-DE")
        voice.speak(utterance)
    }
}
