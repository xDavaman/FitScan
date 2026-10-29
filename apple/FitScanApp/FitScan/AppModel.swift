import Foundation
import ImageIO
import UniformTypeIdentifiers
import FitScanCore

extension Locale {
    /// Die Oberfläche ist deutsch, Zahlen daher immer mit Dezimalkomma, unabhängig von der Region des Geräts.
    static let deutsch = Locale(identifier: "de_DE")
}

/// Seiten, zu denen die App von überall springen kann.
enum Route: Hashable {
    case model(ScanRecord)
    case edit(ScanRecord)
    case history
    case guide
}

/// Blätter, die über allem aufgehen. Ein Wechsel schließt das alte und öffnet das neue.
enum AppSheet: String, Identifiable {
    case newScan
    case measures

    var id: String { rawValue }
}

/// Zustand der App: Annys Daten, gespeicherte Scans und die laufende Berechnung.
@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable {
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .loading
    @Published private(set) var scans: [ScanRecord] = []
    @Published var person = Person(name: "", gender: .maennlich, ageYears: 30, measures: [:])
    @Published private(set) var progress: Double?
    @Published var problems: [String] = []
    @Published var lastRecord: ScanRecord?
    @Published var path: [Route] = []
    @Published var sheet: AppSheet?

    private(set) var model: BodyModel?
    /// Einstellungen des Foto- und Videoscans (vom PC mitgeliefert)
    private(set) var photoSettings: PhotoScanSettings?
    /// Bewegungen (Winken, Rufen, Sprechen, Gehen) und Gesichtsregler, vom PC mitgeliefert
    private(set) var motions: [MotionClip] = []
    private(set) var faceRig: FaceRig?
    private var store: PersonStore?

    func start() async {
        guard model == nil else { return }
        do {
            let (loaded, library, face) = try await Task.detached(priority: .userInitiated) {
                (try BodyModel.bundled(), try? MotionLibrary.bundled(), try? FaceRig.bundled())
            }.value
            model = loaded
            motions = library?.clips ?? []
            faceRig = face
            photoSettings = try? PhotoScanSettings.bundled()
            store = try PersonStore.appDefault()
            reloadScans()
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func reloadScans() {
        scans = store?.allScans() ?? []
    }

    /// Neuestes Modell, das auf dem Startbildschirm steht.
    var latest: ScanRecord? { scans.first }

    /// Vorname aus dem neuesten Scan, für die Begrüßung.
    var firstName: String? {
        guard let name = latest?.person.name.trimmingCharacters(in: .whitespaces), !name.isEmpty else { return nil }
        return name.split(separator: " ").first.map(String.init)
    }

    /// Eingaben der letzten Person (oder von `record`) ins Formular übernehmen.
    func prefillPerson(from record: ScanRecord? = nil) {
        if let record = record ?? latest { person = record.person }
        problems = []
    }

    /// Maße ändern: Formular mit den Eingaben von `record` öffnen.
    func openMeasureForm(from record: ScanRecord) {
        prefillPerson(from: record)
        sheet = .measures
    }

    /// Nach dem Berechnen: Blatt schließen und das neue Modell zeigen.
    func finishedMeasureForm(_ record: ScanRecord) {
        sheet = nil
        path = [.model(record)]
    }

    /// Passt das Modell an die Eingaben an und speichert es. Rechnet im Hintergrund.
    func compute() async {
        guard let model, let store else { return }
        problems = person.validate()
        guard problems.isEmpty else { return }
        progress = 0
        let input = person
        do {
            let record = try await Task.detached(priority: .userInitiated) {
                let result = try model.fitToMeasurements(input) { step, total in
                    let value = Double(step) / Double(total)
                    if step % 10 == 0 {
                        // Späte Meldungen nach dem Ende verwerfen, sonst bleibt das Formular im Rechnen-Zustand.
                        Task { @MainActor in
                            if self.progress != nil { self.progress = value }
                        }
                    }
                }
                return try store.save(person: input, result: result, model: model)
            }.value
            progress = nil
            reloadScans()
            lastRecord = record
        } catch {
            progress = nil
            problems = [error.localizedDescription]
        }
    }

    func params(of record: ScanRecord) -> BodyParams? {
        guard let store else { return nil }
        return try? BodyParams.load(from: store.folder(of: record))
    }

    func file(_ name: String, of record: ScanRecord) -> URL? {
        store?.folder(of: record).appendingPathComponent(name)
    }

    /// Speichert ein von Hand nachbearbeitetes Modell als neuen Eintrag im Verlauf (der alte bleibt).
    func saveEdited(_ record: ScanRecord, params: BodyParams) async throws -> ScanRecord {
        guard let model, let store else { throw FitScanError.data("Die App ist noch nicht bereit.") }
        let saved = try await Task.detached(priority: .userInitiated) {
            let result = FitResult(params: params, measures: model.measure(params))
            return try store.save(person: record.person, result: result, model: model, source: "bearbeitet")
        }.value
        reloadScans()
        return saved
    }

    /// Scan aus Fotos oder Video: Form an Umrisse und Körperpunkte anpassen, speichern und je Bild
    /// ein Kontrollbild dazulegen. `progress` bekommt einen Text und den Anteil 0…1.
    func computePhotoScan(frames: [PhotoFrame], source: String,
                          progress: @escaping @MainActor (String, Double) -> Void) async -> ScanRecord? {
        guard let model, let store else { return nil }
        guard let settings = photoSettings else {
            problems = ["Die Einstellungen des Fotoscans fehlen in der App."]
            return nil
        }
        problems = person.validate()
        guard problems.isEmpty else { return nil }
        let input = person
        do {
            let record = try await Task.detached(priority: .userInitiated) {
                let result = try model.fitPhotos(input, frames: frames, settings: settings) { text, value in
                    Task { @MainActor in progress(text, value) }
                }
                let saved = try store.save(person: input, result: FitResult(params: result.params, measures: result.measures),
                                           model: model, source: source)
                try? FitView.save(result.views, to: store.folder(of: saved))
                return saved
            }.value
            reloadScans()
            return record
        } catch {
            problems = [error.localizedDescription]
            return nil
        }
    }

    /// Bilder einer Bewegung mit dieser Körperform für die Vorschau, mit weichen Übergängen aus der
    /// Ruhepose und zurück (je 12 Bilder, 0,4 s). Im Hintergrund gerechnet.
    func motionFrames(_ clip: MotionClip, params: BodyParams) async -> MotionFrames {
        guard let model else { return MotionFrames() }
        let face = faceRig
        return await Task.detached(priority: .userInitiated) {
            let end = clip.loop ? 0 : clip.frames - 1  // eine Schleife endet wieder bei Bild 0
            return MotionFrames(
                intro: model.previewTransition(params, clip: clip, face: face, steps: 12).map { BodyMesh($0) },
                main: model.previewFrames(params, clip: clip, face: face).map { BodyMesh($0) },
                outro: model.previewTransition(params, clip: clip, face: face, frame: end, steps: 12, toRest: true)
                    .map { BodyMesh($0) })
        }.value
    }

    static func motionFileName(_ naming: BoneNaming) -> String {
        naming == .unreal ? "koerper_bewegungen_unreal.glb" : "koerper_bewegungen.glb"
    }

    /// GLB mit allen Bewegungen und den Gesichtsreglern. Wird einmal geschrieben und liegt dann im
    /// Ordner des Scans; ein Scan ändert sich nicht mehr (Bearbeiten legt einen neuen an).
    func motionGLB(_ record: ScanRecord, naming: BoneNaming) async throws -> URL {
        guard let model, let store, let params = params(of: record) else {
            throw FitScanError.data("Die App ist noch nicht bereit.")
        }
        guard !motions.isEmpty else { throw FitScanError.data("Die Bewegungen fehlen in der App.") }
        let url = store.folder(of: record).appendingPathComponent(Self.motionFileName(naming))
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let clips = motions, face = faceRig
        try await Task.detached(priority: .userInitiated) {
            try model.writeGLB(params, to: url, naming: naming, clips: clips, face: face)
        }.value
        return url
    }

    /// Schon geschriebene GLB-Dateien mit Bewegungen.
    func motionFiles(of record: ScanRecord) -> [BoneNaming: URL] {
        var out: [BoneNaming: URL] = [:]
        for naming in BoneNaming.allCases {
            if let url = file(Self.motionFileName(naming), of: record), FileManager.default.fileExists(atPath: url.path) {
                out[naming] = url
            }
        }
        return out
    }

    /// Kontrollbilder eines Scans aus Bildern, falls vorhanden.
    func fitViews(of record: ScanRecord) -> [FitView] {
        guard let store else { return [] }
        return FitView.load(from: store.folder(of: record))
    }

    func delete(_ record: ScanRecord) {
        try? store?.delete(record)
        reloadScans()
    }

    /// Eingaben eines alten Scans übernehmen, z. B. für einen neuen Scan derselben Person.
    func reuse(_ record: ScanRecord) {
        person = record.person
    }
}

/// Wie das Modell auf einem Bild liegt: Kontrollbild `passform_N.png` und Werte in `passform.json`.
struct FitView: Identifiable, Codable {
    var id: Int { image }
    let image: Int
    let yawDegrees: Double
    let iou: Double
    var file: URL?

    enum CodingKeys: String, CodingKey {
        case image = "bild", yawDegrees = "drehung_grad", iou = "ueberdeckung"
    }

    static func save(_ views: [PhotoViewResult], to folder: URL) throws {
        var list: [FitView] = []
        for (i, view) in views.enumerated() {
            let url = folder.appendingPathComponent("passform_\(i + 1).png")
            try writePNG(view, to: url)
            list.append(FitView(image: i + 1, yawDegrees: view.yawDegrees, iou: view.iou))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(list).write(to: folder.appendingPathComponent("passform.json"))
    }

    static func load(from folder: URL) -> [FitView] {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("passform.json")),
              let list = try? JSONDecoder().decode([FitView].self, from: data) else { return [] }
        return list.map { view in
            var v = view
            v.file = folder.appendingPathComponent("passform_\(view.image).png")
            return v
        }
    }

    private static func writePNG(_ view: PhotoViewResult, to url: URL) throws {
        guard let provider = CGDataProvider(data: Data(view.overlay) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: view.width, height: view.height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: 3 * view.width, space: space,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw FitScanError.data("Kontrollbild ließ sich nicht schreiben.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw FitScanError.data("Kontrollbild ließ sich nicht schreiben.") }
    }
}
