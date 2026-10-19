import Foundation
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
    private var store: PersonStore?

    func start() async {
        guard model == nil else { return }
        do {
            let loaded = try await Task.detached(priority: .userInitiated) { try BodyModel.bundled() }.value
            model = loaded
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

    func delete(_ record: ScanRecord) {
        try? store?.delete(record)
        reloadScans()
    }

    /// Eingaben eines alten Scans übernehmen, z. B. für einen neuen Scan derselben Person.
    func reuse(_ record: ScanRecord) {
        person = record.person
    }
}
