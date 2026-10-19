import AVFoundation
import CoreTransferable
import ImageIO
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
import Vision
import FitScanCore

/// Ein Video aus der Mediathek, in den temporären Ordner der App kopiert.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let target = FileManager.default.temporaryDirectory
                .appendingPathComponent("fitscan-\(UUID().uuidString)")
                .appendingPathExtension(received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: target)
            return PickedMovie(url: target)
        }
    }
}

/// Ein ausgewähltes Bild (Foto oder Videobild) nach der Prüfung.
struct ScanImage: Identifiable {
    let id = UUID()
    let thumbnail: UIImage
    /// Für die Anpassung, wenn das Bild brauchbar ist
    let frame: PhotoFrame?
    /// Warum das Bild nicht verwendet wird
    let problem: String?
    let fromVideo: Bool
}

/// Person und Körperpunkte mit Apple Vision, alles auf dem iPhone.
enum PersonDetector {
    /// Vision-Gelenke in der Nummerierung, die FitScanCore erwartet (wie MediaPipe am PC; links = links der Person)
    static let joints: [(Int, VNHumanBodyPoseObservation.JointName)] = [
        (11, .leftShoulder), (12, .rightShoulder), (13, .leftElbow), (14, .rightElbow),
        (15, .leftWrist), (16, .rightWrist), (23, .leftHip), (24, .rightHip),
        (25, .leftKnee), (26, .rightKnee), (27, .leftAnkle), (28, .rightAnkle),
    ]

    /// Prüft ein aufrechtes Bild. `focal`: Brennweite in Pixeln dieses Bildes.
    static func analyze(_ image: CGImage, focal: Double, fromVideo: Bool) -> ScanImage {
        let thumbnail = UIImage(cgImage: image).preparingThumbnail(of: thumbnailSize(image)) ?? UIImage(cgImage: image)
        func reject(_ text: String) -> ScanImage {
            ScanImage(thumbnail: thumbnail, frame: nil, problem: text, fromVideo: fromVideo)
        }
        let segmentation = VNGeneratePersonSegmentationRequest()
        segmentation.qualityLevel = .accurate
        segmentation.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let pose = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        do {
            try handler.perform([segmentation, pose])
        } catch {
            return reject("Bild ließ sich nicht auswerten")
        }
        guard let buffer = segmentation.results?.first?.pixelBuffer else { return reject("Keine Person erkannt") }
        let (mask, w, h) = maskValues(buffer)
        guard w > 0, h > 0 else { return reject("Keine Person erkannt") }

        // Person groß genug und ganz im Bild?
        var person = 0, top = 0, bottom = 0
        let band = max(1, h / 100)
        for y in 0..<h {
            for x in 0..<w where mask[y * w + x] > 0.5 {
                person += 1
                if y < band { top += 1 }
                if y >= h - band { bottom += 1 }
            }
        }
        if Double(person) < 0.01 * Double(w * h) { return reject("Keine Person erkannt") }
        if top > 3 { return reject("Kopf abgeschnitten") }
        if bottom > 3 { return reject("Füße abgeschnitten") }

        // Körperpunkte in Maskenpixeln (Vision: 0…1, Ursprung unten links)
        var keypoints: [Int: SIMD3<Double>] = [:]
        if let observation = bestObservation(pose.results ?? []),
           let points = try? observation.recognizedPoints(.all) {
            for (index, joint) in joints {
                guard let p = points[joint], p.confidence > 0 else { continue }
                keypoints[index] = SIMD3(Double(p.location.x) * Double(w), (1 - Double(p.location.y)) * Double(h),
                                         Double(p.confidence))
            }
        }
        let scale = Double(w) / Double(image.width)
        let frame = PhotoFrame(width: w, height: h, mask: mask, focal: focal * scale, keypoints: keypoints)
        return ScanImage(thumbnail: thumbnail, frame: frame, problem: nil, fromVideo: fromVideo)
    }

    static func thumbnailSize(_ image: CGImage) -> CGSize {
        let s = 180 / Double(max(image.width, image.height))
        return CGSize(width: Double(image.width) * s, height: Double(image.height) * s)
    }

    /// Die Person mit den sichersten Punkten, falls mehrere im Bild sind.
    static func bestObservation(_ all: [VNHumanBodyPoseObservation]) -> VNHumanBodyPoseObservation? {
        all.max { a, b in
            let sa = (try? a.recognizedPoints(.all).values.reduce(0) { $0 + $1.confidence }) ?? 0
            let sb = (try? b.recognizedPoints(.all).values.reduce(0) { $0 + $1.confidence }) ?? 0
            return sa < sb
        }
    }

    static func maskValues(_ buffer: CVPixelBuffer) -> ([Float], Int, Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return ([], 0, 0) }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w { out[y * w + x] = Float(bytes[y * rowBytes + x]) / 255 }
        }
        return (out, w, h)
    }
}

/// Bilder aus Fotos und Videos holen.
enum MediaLoader {
    static let maxSide = 1920

    /// Foto aufrecht und verkleinert, dazu die Brennweite aus den EXIF-Daten, falls vorhanden.
    static func photo(_ data: Data, fovLongSideDeg: Double) -> (image: CGImage, focal: Double)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        let w = Double(image.width), h = Double(image.height)
        var focal = PhotoFrame.defaultFocal(width: image.width, height: image.height, fovLongSideDeg: fovLongSideDeg)
        if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
           let f35 = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double, f35 > 0 {
            // Kleinbild-Brennweite bezieht sich auf die Bilddiagonale (43,27 mm)
            focal = f35 * (w * w + h * h).squareRoot() / 43.27
        }
        return (image, focal)
    }

    /// `count` Bilder gleichmäßig über das Video verteilt, aufrecht.
    static func videoFrames(_ url: URL, count: Int) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else { return [] }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: maxSide, height: maxSide)
        let tolerance = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        var out: [CGImage] = []
        for i in 0..<count {
            let t = duration * (0.03 + 0.94 * Double(i) / Double(max(count - 1, 1)))
            if let frame = try? await generator.image(at: CMTime(seconds: t, preferredTimescale: 600)) {
                out.append(frame.image)
            }
        }
        return out
    }
}

/// Auswahl prüfen und für die Anpassung bereitlegen.
@MainActor
final class PhotoScanner: ObservableObject {
    /// Bilder, die aus einem Video geprüft werden (davon höchstens `maxFrames` verwendet)
    static let videoSamples = 24
    static let maxFrames = 12

    @Published private(set) var images: [ScanImage] = []
    @Published private(set) var analyzing: Double?
    @Published var fitting: (text: String, value: Double)?
    @Published var error: String?

    private var run = 0

    var usable: [ScanImage] { images.filter { $0.frame != nil } }
    var busy: Bool { analyzing != nil || fitting != nil }

    /// Aus einem Video gleichmäßig verteilt, damit alle Blickwinkel dabei sind.
    var framesForFit: [PhotoFrame] {
        let all = usable.compactMap(\.frame)
        guard all.count > Self.maxFrames else { return all }
        let step = Double(all.count) / Double(Self.maxFrames)
        return (0..<Self.maxFrames).map { all[Int(Double($0) * step)] }
    }

    var source: String { usable.contains { $0.fromVideo } ? "video" : "foto" }

    func load(_ items: [PhotosPickerItem], fovLongSideDeg: Double) async {
        run += 1
        let current = run
        images = []
        error = nil
        guard !items.isEmpty else {
            analyzing = nil
            return
        }
        analyzing = 0
        var result: [ScanImage] = []
        for (i, item) in items.enumerated() {
            let isMovie = item.supportedContentTypes.contains { $0.conforms(to: .movie) }
            do {
                if isMovie, let movie = try await item.loadTransferable(type: PickedMovie.self) {
                    let frames = try await MediaLoader.videoFrames(movie.url, count: Self.videoSamples)
                    try? FileManager.default.removeItem(at: movie.url)
                    for (j, image) in frames.enumerated() {
                        let focal = PhotoFrame.defaultFocal(width: image.width, height: image.height,
                                                            fovLongSideDeg: fovLongSideDeg)
                        let checked = await Task.detached(priority: .userInitiated) {
                            PersonDetector.analyze(image, focal: focal, fromVideo: true)
                        }.value
                        guard current == run else { return }
                        result.append(checked)
                        analyzing = (Double(i) + Double(j + 1) / Double(frames.count)) / Double(items.count)
                    }
                } else if let data = try await item.loadTransferable(type: Data.self),
                          let picked = MediaLoader.photo(data, fovLongSideDeg: fovLongSideDeg) {
                    let checked = await Task.detached(priority: .userInitiated) {
                        PersonDetector.analyze(picked.image, focal: picked.focal, fromVideo: false)
                    }.value
                    guard current == run else { return }
                    result.append(checked)
                }
            } catch {
                guard current == run else { return }
                self.error = "Eine Datei ließ sich nicht laden: \(error.localizedDescription)"
            }
            guard current == run else { return }
            analyzing = Double(i + 1) / Double(items.count)
            images = result
        }
        analyzing = nil
        if result.isEmpty && error == nil { error = "Keine Bilder gefunden." }
    }
}
