import Foundation

/// Was der Mensch eingibt (wie `fitscan.fit.Person`). Maße in cm, Gewicht in kg.
public struct Person: Codable, Hashable {
    public var name: String
    public var gender: Gender
    public var ageYears: Double
    public var measures: [MeasureKey: Double]

    public init(name: String = "", gender: Gender = .maennlich, ageYears: Double = 30,
                measures: [MeasureKey: Double] = [:]) {
        self.name = name
        self.gender = gender
        self.ageYears = ageYears
        self.measures = measures
    }

    /// Prüft die Eingaben, Texte für die Oberfläche.
    public func validate() -> [String] {
        var problems: [String] = []
        let h = measures[.height] ?? 0
        let w = measures[.weight] ?? 0
        if !(100...230).contains(h) { problems.append("Bitte die Körpergröße in cm angeben (100 bis 230).") }
        if !(30...250).contains(w) { problems.append("Bitte das Gewicht in kg angeben (30 bis 250).") }
        for key in MeasureKey.circumferences {
            if let v = measures[key], !(15...220).contains(v) {
                problems.append("Der Wert für \(key.label) (\(String(format: "%.0f", v)) cm) wirkt unplausibel.")
            }
        }
        if h > 0, let inseam = measures[.inseam], !((0.3 * h)...(0.6 * h)).contains(inseam) {
            problems.append("Die Schrittlänge passt nicht zur Körpergröße.")
        }
        if !(1...90).contains(ageYears) { problems.append("Bitte ein Alter zwischen 1 und 90 Jahren angeben.") }
        return problems
    }
}

/// Welcher lokale Anny-Regler zu welchem Maß gehört (wie `LOCAL_FOR_MEASURE`)
let localForMeasure: [MeasureKey: [String]] = [
    .chest: ["measure-bust-circ-incr"],
    .waist: ["measure-waist-circ-incr"],
    .hips: ["measure-hips-circ-incr"],
    .thigh: ["measure-thigh-circ-incr"],
    .calf: ["measure-calf-circ-incr"],
    .upperarm: ["measure-upperarm-circ-incr"],
    .neck: ["measure-neck-circ-incr"],
    .inseam: ["measure-upperleg-height-incr", "measure-lowerleg-height-incr"],
]

/// Gewichtung der Maße im Fehler (relativer Fehler), wie `MEASURE_WEIGHTS`
func measureWeight(_ key: MeasureKey) -> Double { key == .height ? 40 : 6 }

let freePhenotypes = ["muscle", "weight", "height", "proportions"]

public struct FitResult {
    public var params: BodyParams
    public var measures: [MeasureKey: Double]

    public init(params: BodyParams, measures: [MeasureKey: Double]) {
        self.params = params
        self.measures = measures
    }
}

@inline(__always) func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-x)) }
@inline(__always) func logit(_ p: Double) -> Double { log(p / (1 - p)) }

/// Adam wie in PyTorch (lr, β = 0.9 / 0.999, ε = 1e-8).
struct Adam {
    var lr: Double
    var m: [Double]
    var v: [Double]
    var t = 0

    init(count: Int, lr: Double) {
        self.lr = lr
        m = [Double](repeating: 0, count: count)
        v = [Double](repeating: 0, count: count)
    }

    mutating func step(_ x: inout [Double], _ g: [Double]) {
        t += 1
        let b1 = 0.9, b2 = 0.999, eps = 1e-8
        let c1 = 1 - pow(b1, Double(t)), c2 = 1 - pow(b2, Double(t))
        for i in 0..<x.count {
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
            let denom = (v[i] / c2).squareRoot() + eps
            x[i] -= lr / c1 * m[i] / denom
        }
    }
}

/// Die frei veränderlichen Größen einer Anpassung (wie `fit.ShapeVariables`), in unbeschränkter Form:
/// x = [Phänotyp (logit) …, lokal (atanh) …, log(Maßstab)].
struct ShapeSpace {
    let model: BodyModel
    let fixed: [String: Double]
    let free: [String]
    let localNames: [String]
    let localIndex: [Int]
    let baseLocal: [String: Double]
    let offsets: [Double]?

    init(model: BodyModel, person: Person, initial: BodyParams, extraLocals: [String] = []) {
        self.model = model
        var fixed: [String: Double] = ["gender": person.gender.value, "age": ageToParam(person.ageYears)]
        var free = freePhenotypes
        if person.gender == .divers { free.append("gender"); fixed["gender"] = nil }
        self.fixed = fixed
        self.free = free
        var wanted = Set(extraLocals)
        for key in person.measures.keys { wanted.formUnion(localForMeasure[key] ?? []) }
        for (k, v) in initial.local where abs(v) > 1e-6 { wanted.insert(k) }
        let labels = model.data.localLabels
        let names = labels.filter { wanted.contains($0) }
        localNames = names
        localIndex = names.map { labels.firstIndex(of: $0)! }
        baseLocal = initial.local.filter { !names.contains($0.key) }
        offsets = initial.offsets
    }

    var nf: Int { free.count }
    var nl: Int { localNames.count }
    var count: Int { nf + nl + 1 }
    var scaleIndex: Int { nf + nl }

    func start(_ initial: BodyParams) -> [Double] {
        var x = free.map { logit(min(max(initial.phenotype[$0] ?? 0.5, 0.02), 0.98)) }
        x += localNames.map { atanh(min(max(initial.local[$0] ?? 0, -0.95), 0.95)) }
        x.append(log(initial.scale))
        return x
    }

    func params(_ x: [Double]) -> BodyParams {
        var phenotype = fixed
        for (i, k) in free.enumerated() { phenotype[k] = sigmoid(x[i]) }
        for k in BodyParams.phenotypes where phenotype[k] == nil { phenotype[k] = 0.5 }
        var local = baseLocal
        for (i, k) in localNames.enumerated() { local[k] = tanh(x[nf + i]) }
        return BodyParams(phenotype: phenotype, local: local, scale: exp(x[scaleIndex]), offsets: offsets)
    }

    /// dL/dx aus dL/dPunkte der Ruhepose (Meter, mit Maßstab), ohne Strafterme für die Regler.
    func gradient(_ x: [Double], params p: BodyParams, rest r: RestBody, pointGradient g: [Vec3]) -> [Double] {
        var grad = [Double](repeating: 0, count: count)
        let s = p.scale
        // Maßstab: V = s · V₀
        var dS = 0.0
        for i in 0..<g.count { dS += dot(g[i], r.unscaledVertices[i]) }
        grad[scaleIndex] = dS * s
        // Formgewichte: dL/dc_j = s · Σ g · Form_j
        var shapeGrad: [Int: Double] = [:]
        func dc(_ shape: Int) -> Double {
            if let v = shapeGrad[shape] { return v }
            let v = s * model.shapeDot(shape, g)
            shapeGrad[shape] = v
            return v
        }
        for (i, k) in free.enumerated() {
            let ph = sigmoid(x[i])
            var d = 0.0
            for (shape, value) in model.macroDerivatives(p.phenotype, feature: k) { d += value * dc(shape) }
            grad[i] = d * ph * (1 - ph)
        }
        for (i, li) in localIndex.enumerated() {
            let l = tanh(x[nf + i])
            let plus = model.data.numMacroShapes + 2 * li
            let d = (l >= 0 ? dc(plus) : 0) - (-l >= 0 ? dc(plus + 1) : 0)
            grad[nf + i] = d * (1 - l * l)
        }
        return grad
    }
}

public extension BodyModel {
    /// Passt Anny an die eingegebenen Maße an (wie `fit_to_measurements`): Phänotyp-Regler,
    /// die zu den Maßen gehörenden lokalen Regler und der Maßstab, mit Adam über 240 Schritte.
    /// Am Ende trifft der Maßstab die Körpergröße genau.
    func fitToMeasurements(_ person: Person, initial: BodyParams? = nil, steps: Int = 240,
                           progress: ((Int, Int) -> Void)? = nil) throws -> FitResult {
        let problems = person.validate()
        if !problems.isEmpty { throw FitScanError.input(problems) }
        let targets = person.measures.filter { $0.value != 0 }
        let space = ShapeSpace(model: self, person: person, initial: initial ?? BodyParams())
        var x = space.start(initial ?? BodyParams())
        let nf = space.nf

        // Startmaßstab grob auf die Größe
        if let h = targets[.height] {
            let zs = rest(space.params(x)).vertices.map(\.z)
            let current = (zs.max()! - zs.min()!) * 100
            x[space.scaleIndex] += log(h / current)
        }

        var adam = Adam(count: x.count, lr: 0.05)
        var probe: MeasureProbe?
        for step in 0..<steps {
            let p = space.params(x)
            let r = rest(p)
            if step % 40 == 0 || probe == nil { probe = MeasureProbe(model: self, rest: r) }
            guard let current = probe else { break }
            let values = current.evaluate(r.vertices)

            // dL/dPunkte
            var g = [Vec3](repeating: .zero, count: numVertices)
            for (key, target) in targets {
                guard let value = values[key], target != 0 else { continue }
                let dL = 2 * measureWeight(key) * (value - target) / (target * target)
                current.accumulateGradient(key, r.vertices, scale: dL, into: &g)
            }
            var grad = space.gradient(x, params: p, rest: r, pointGradient: g)
            // 0.004 · Σ (Phänotyp − 0.5)² und 0.004 · Σ lokal²
            for i in 0..<nf {
                let ph = sigmoid(x[i])
                grad[i] += 0.008 * (ph - 0.5) * ph * (1 - ph)
            }
            for i in 0..<space.nl {
                let l = tanh(x[nf + i])
                grad[nf + i] += 0.008 * l * (1 - l * l)
            }
            adam.step(&x, grad)
            progress?(step + 1, steps)
        }

        var result = space.params(x)
        if let h = targets[.height] { result = exactHeight(result, heightCM: h) }
        return FitResult(params: result, measures: measure(result))
    }

    /// Die eingegebene Körpergröße ist verbindlich: kleiner Maßstabsausgleich am Ende.
    func exactHeight(_ params: BodyParams, heightCM: Double) -> BodyParams {
        let zs = rest(params).vertices.map(\.z)
        var out = params
        out.scale *= heightCM / ((zs.max()! - zs.min()!) * 100)
        return out
    }
}
