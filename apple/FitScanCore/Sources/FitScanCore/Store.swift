import Foundation

/// Ein gespeicherter Scan: Eingaben, erreichte Maße und die Dateien im Ordner.
public struct ScanRecord: Codable, Identifiable, Hashable {
    public var id: String  // Ordnername, z. B. 2026-10-18_14-05-12
    public var personID: String
    public var date: Date
    public var person: Person
    public var measures: [MeasureKey: Double]
    /// Woher das Modell stammt: `massband` (aus den Maßen), `bearbeitet` (von Hand nachbearbeitet).
    /// Fehlt in Scans aus dem ersten Build.
    public var source: String?

    /// Kurz für die Liste, z. B. „aus Maßen“
    public var sourceLabel: String {
        switch source {
        case "bearbeitet": return "nachbearbeitet"
        default: return "aus Maßen"
        }
    }

    public var title: String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "de_DE")
        f.dateStyle = .medium
        f.timeStyle = .short
        return f.string(from: date)
    }
}

/// Personen und Scan-Verlauf auf dem Gerät, wie `fitscan.store` am PC:
/// `personen/<id>/scans/<Zeit>/` mit `params.json`, `scan.json` und den GLB-Dateien.
public final class PersonStore {
    public let root: URL

    public init(root: URL) { self.root = root }

    /// Dokumente-Ordner der App. Er bleibt auf dem Gerät und erscheint in der Dateien-App unter
    /// „Auf meinem iPhone › FitScan“, so kommen die GLB-Dateien auch ohne Teilen an den PC.
    public static func appDefault() throws -> PersonStore {
        let base = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        return PersonStore(root: base)
    }

    public static func slug(_ name: String) -> String {
        let map: [Character: String] = ["ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"]
        var out = ""
        for ch in name.lowercased() {
            if let r = map[ch] { out += r } else if ch.isLetter || ch.isNumber { out.append(ch) } else { out += "-" }
        }
        let trimmed = out.split(separator: "-").joined(separator: "-")
        return trimmed.isEmpty ? "person" : trimmed
    }

    func scansFolder(_ personID: String) -> URL {
        root.appendingPathComponent("personen/\(personID)/scans", isDirectory: true)
    }

    /// Speichert einen Scan mit beiden Namensschemata als GLB. Gibt den Eintrag zurück.
    @discardableResult
    public func save(person: Person, result: FitResult, model: BodyModel, date: Date = Date(),
                     source: String = "massband") throws -> ScanRecord {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let personID = Self.slug(person.name)
        // Zwei Scans in derselben Sekunde (z. B. schnell nachbearbeitet) bekommen eigene Ordner
        var id = f.string(from: date)
        var n = 2
        while FileManager.default.fileExists(atPath: scansFolder(personID).appendingPathComponent(id).path) {
            id = f.string(from: date) + "-\(n)"
            n += 1
        }
        let record = ScanRecord(id: id, personID: personID, date: date, person: person,
                                measures: result.measures, source: source)
        let folder = scansFolder(personID).appendingPathComponent(record.id, isDirectory: true)
        try result.params.save(to: folder)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: folder.appendingPathComponent("scan.json"))
        try model.writeGLB(result.params, to: folder.appendingPathComponent("koerper.glb"), naming: .anny)
        try model.writeGLB(result.params, to: folder.appendingPathComponent("koerper_unreal.glb"), naming: .unreal)
        return record
    }

    public func folder(of record: ScanRecord) -> URL {
        scansFolder(record.personID).appendingPathComponent(record.id, isDirectory: true)
    }

    /// Alle Scans, neueste zuerst.
    public func allScans() -> [ScanRecord] {
        let fm = FileManager.default
        let personsRoot = root.appendingPathComponent("personen", isDirectory: true)
        guard let persons = try? fm.contentsOfDirectory(atPath: personsRoot.path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var out: [ScanRecord] = []
        for p in persons {
            let scans = scansFolder(p)
            for s in (try? fm.contentsOfDirectory(atPath: scans.path)) ?? [] {
                let url = scans.appendingPathComponent(s).appendingPathComponent("scan.json")
                if let data = try? Data(contentsOf: url), let r = try? decoder.decode(ScanRecord.self, from: data) {
                    out.append(r)
                }
            }
        }
        return out.sorted { $0.date > $1.date }
    }

    public func delete(_ record: ScanRecord) throws {
        try FileManager.default.removeItem(at: folder(of: record))
    }
}
