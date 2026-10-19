import Foundation
import FitScanCore

extension Locale {
    /// Die Oberfläche ist deutsch, Zahlen daher immer mit Dezimalkomma, unabhängig von der Region des Geräts.
    static let deutsch = Locale(identifier: "de_DE")
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

    func delete(_ record: ScanRecord) {
        try? store?.delete(record)
        reloadScans()
    }

    /// Eingaben eines alten Scans übernehmen, z. B. für einen neuen Scan derselben Person.
    func reuse(_ record: ScanRecord) {
        person = record.person
    }
}
