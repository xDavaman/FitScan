import Foundation

/// Eine Person als Annys Regler, wie `fitscan.body.BodyParams` (gleiche `params.json`).
public struct BodyParams: Equatable {
    public static let phenotypes = ["gender", "age", "muscle", "weight", "height", "proportions"]

    /// Phänotyp-Regler 0…1 (Geschlecht, Alter, Muskeln, Gewicht, Größe, Proportionen)
    public var phenotype: [String: Double]
    /// Lokale Formregler −1…1, z. B. `measure-waist-circ-incr`
    public var local: [String: Double]
    /// Gesamtmaßstab, mit dem die eingegebene Körpergröße genau getroffen wird
    public var scale: Double
    /// Feinform je Punkt (3 · V Werte in Metern, Ruhepose, vor dem Maßstab)
    public var offsets: [Double]?

    public init(phenotype: [String: Double] = Dictionary(uniqueKeysWithValues: BodyParams.phenotypes.map { ($0, 0.5) }),
                local: [String: Double] = [:], scale: Double = 1.0, offsets: [Double]? = nil) {
        self.phenotype = phenotype
        self.local = local
        self.scale = scale
        self.offsets = offsets
    }
}

extension BodyParams: Codable {
    enum CodingKeys: String, CodingKey { case phenotype, local, scale, offsets }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        phenotype = try c.decode([String: Double].self, forKey: .phenotype)
        local = try c.decodeIfPresent([String: Double].self, forKey: .local) ?? [:]
        scale = try c.decodeIfPresent(Double.self, forKey: .scale) ?? 1.0
        offsets = nil  // liegen in offsets.npy, siehe load(from:)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(phenotype, forKey: .phenotype)
        try c.encode(local.filter { abs($0.value) > 1e-6 }, forKey: .local)
        try c.encode(scale, forKey: .scale)
        if offsets != nil {
            try c.encode("offsets.npy", forKey: .offsets)
        } else {
            try c.encodeNil(forKey: .offsets)
        }
    }

    /// Schreibt `params.json` (und `offsets.npy`) wie der PC.
    public func save(to folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: folder.appendingPathComponent("params.json"))
        if let offsets {
            try Npy.write(offsets, rows: offsets.count / 3, columns: 3,
                          to: folder.appendingPathComponent("offsets.npy"))
        }
    }

    /// Liest `params.json` (und `offsets.npy`) aus einem Scan-Ordner vom PC oder aus der App.
    public static func load(from folder: URL) throws -> BodyParams {
        let data = try Data(contentsOf: folder.appendingPathComponent("params.json"))
        var params = try JSONDecoder().decode(BodyParams.self, from: data)
        if let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let name = raw["offsets"] as? String {
            params.offsets = try Npy.read(folder.appendingPathComponent(name))
        }
        return params
    }
}

// MARK: - Geschlecht und Alter

public enum Gender: String, CaseIterable, Codable, Identifiable {
    case maennlich, weiblich, divers

    public var id: String { rawValue }

    /// Anny: 0 = männlich, 1 = weiblich
    public var value: Double {
        switch self {
        case .maennlich: return 0.0
        case .weiblich: return 1.0
        case .divers: return 0.5
        }
    }

    public var label: String {
        switch self {
        case .maennlich: return "männlich"
        case .weiblich: return "weiblich"
        case .divers: return "divers"
        }
    }
}

/// Altersanker von Anny/MakeHuman: Regler → Jahre
private let ageAnchors: [(Double, Double)] = [(0.0, 1.0), (1.0 / 3.0, 11.0), (2.0 / 3.0, 25.0), (1.0, 90.0)]

public func ageToParam(_ years: Double) -> Double {
    let y = min(max(years, 1.0), 90.0)
    for i in 0..<(ageAnchors.count - 1) {
        let (p0, y0) = ageAnchors[i]
        let (p1, y1) = ageAnchors[i + 1]
        if y <= y1 { return p0 + (y - y0) / (y1 - y0) * (p1 - p0) }
    }
    return 1.0
}

public func paramToAge(_ p: Double) -> Double {
    for i in 0..<(ageAnchors.count - 1) {
        let (p0, y0) = ageAnchors[i]
        let (p1, y1) = ageAnchors[i + 1]
        if p <= p1 { return y0 + (p - p0) / (p1 - p0) * (y1 - y0) }
    }
    return 90.0
}

// MARK: - .npy (numpy) für die Feinform

enum Npy {
    static func write(_ values: [Double], rows: Int, columns: Int, to url: URL) throws {
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(rows), \(columns)), }"
        while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93])
        data.append(contentsOf: Array("NUMPY".utf8))
        data.append(contentsOf: [1, 0])
        let length = UInt16(header.utf8.count)
        data.append(contentsOf: [UInt8(length & 0xFF), UInt8(length >> 8)])
        data.append(contentsOf: Array(header.utf8))
        var floats = values.map { Float($0) }
        floats.withUnsafeMutableBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }

    static func read(_ url: URL) throws -> [Double] {
        let data = try Data(contentsOf: url)
        let bytes = [UInt8](data)
        guard bytes.count > 12, bytes[0] == 0x93, Array(bytes[1..<6]) == Array("NUMPY".utf8) else {
            throw FitScanError.data("\(url.lastPathComponent) ist keine .npy-Datei.")
        }
        let major = bytes[6]
        let headerLength: Int
        let start: Int
        if major == 1 {
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8
            start = 10
        } else {
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            start = 12
        }
        guard start + headerLength <= bytes.count else {
            throw FitScanError.data("\(url.lastPathComponent) ist beschädigt.")
        }
        let header = String(decoding: bytes[start..<(start + headerLength)], as: UTF8.self)
        guard !header.contains("'fortran_order': True") else {
            throw FitScanError.data("\(url.lastPathComponent): Fortran-Reihenfolge wird nicht unterstützt.")
        }
        let body = data.subdata(in: (data.startIndex + start + headerLength)..<data.endIndex)
        if header.contains("'<f4'") {
            var out = [Float](repeating: 0, count: body.count / 4)
            out.withUnsafeMutableBytes { _ = body.copyBytes(to: $0) }
            return out.map(Double.init)
        }
        if header.contains("'<f8'") {
            var out = [Double](repeating: 0, count: body.count / 8)
            out.withUnsafeMutableBytes { _ = body.copyBytes(to: $0) }
            return out
        }
        throw FitScanError.data("\(url.lastPathComponent): nur float32 oder float64.")
    }
}
