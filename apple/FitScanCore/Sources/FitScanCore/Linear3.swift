import Foundation

/// Punkt oder Richtung in Metern.
public typealias Vec3 = SIMD3<Double>

@inline(__always) func dot(_ a: Vec3, _ b: Vec3) -> Double { (a * b).sum() }

@inline(__always) func cross(_ a: Vec3, _ b: Vec3) -> Vec3 {
    Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

@inline(__always) func length(_ a: Vec3) -> Double { (a * a).sum().squareRoot() }

/// 3×3-Matrix, zeilenweise gespeichert (`m[r * 3 + c]`).
public struct Mat3: Equatable {
    public var m: [Double]

    public init(_ values: [Double]) {
        precondition(values.count == 9)
        m = values
    }

    public static let identity = Mat3([1, 0, 0, 0, 1, 0, 0, 0, 1])

    public subscript(r: Int, c: Int) -> Double {
        get { m[r * 3 + c] }
        set { m[r * 3 + c] = newValue }
    }

    public var transposed: Mat3 {
        Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])
    }

    public var determinant: Double {
        m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    public func column(_ c: Int) -> Vec3 { Vec3(m[c], m[3 + c], m[6 + c]) }

    public static func * (a: Mat3, b: Mat3) -> Mat3 {
        var out = [Double](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                out[r * 3 + c] = a.m[r * 3] * b.m[c] + a.m[r * 3 + 1] * b.m[3 + c] + a.m[r * 3 + 2] * b.m[6 + c]
            }
        }
        return Mat3(out)
    }

    public static func * (a: Mat3, v: Vec3) -> Vec3 {
        Vec3(a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
             a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
             a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z)
    }

    static func fromColumns(_ c0: Vec3, _ c1: Vec3, _ c2: Vec3) -> Mat3 {
        Mat3([c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z])
    }
}

/// Eigenwerte und -vektoren einer symmetrischen 3×3-Matrix (zyklisches Jacobi-Verfahren).
/// Rückgabe absteigend sortiert; die Eigenvektoren stehen in den Spalten.
func symmetricEigen(_ a: Mat3) -> (values: [Double], vectors: Mat3) {
    var s = a
    var v = Mat3.identity
    for _ in 0..<50 {
        let off = s[0, 1] * s[0, 1] + s[0, 2] * s[0, 2] + s[1, 2] * s[1, 2]
        if off < 1e-30 { break }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = s[p, q]
            if abs(apq) < 1e-300 { continue }
            let theta = (s[q, q] - s[p, p]) / (2 * apq)
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot()
            let sn = t * c
            var j = Mat3.identity
            j[p, p] = c
            j[q, q] = c
            j[p, q] = sn
            j[q, p] = -sn
            s = j.transposed * s * j
            v = v * j
        }
    }
    let values = [s[0, 0], s[1, 1], s[2, 2]]
    let order = [0, 1, 2].sorted { values[$0] > values[$1] }
    let sorted = Mat3.fromColumns(v.column(order[0]), v.column(order[1]), v.column(order[2]))
    return (order.map { values[$0] }, sorted)
}

/// Drehung R, die ‖M − R‖ minimiert (wie `roma.special_procrustes`): M = U S Vᵀ,
/// R = U · diag(1, 1, det(U Vᵀ)) · Vᵀ.
public func specialProcrustes(_ m: Mat3) -> Mat3 {
    let (values, v) = symmetricEigen(m.transposed * m)
    let sigma = values.map { max($0, 0).squareRoot() }
    var u = [Vec3](repeating: .zero, count: 3)
    for i in 0..<2 {
        let col = m * v.column(i)
        u[i] = sigma[i] > 1e-12 ? col / sigma[i] : col
    }
    // Erste Spalte normieren, zweite orthogonal dazu, dritte als Kreuzprodukt
    u[0] = u[0] / max(length(u[0]), 1e-300)
    u[1] = u[1] - dot(u[1], u[0]) * u[0]
    u[1] = u[1] / max(length(u[1]), 1e-300)
    u[2] = cross(u[0], u[1])
    // Vorzeichen der dritten Spalte so, dass U · S · Vᵀ wieder M ergibt
    let col2 = m * v.column(2)
    if dot(col2, u[2]) < 0 && sigma[2] > 1e-12 { u[2] = -u[2] }
    let uMat = Mat3.fromColumns(u[0], u[1], u[2])
    let d: Double = (uMat.determinant * v.determinant) < 0 ? -1 : 1
    let uFixed = Mat3.fromColumns(u[0], u[1], d * u[2])
    return uFixed * v.transposed
}

/// Starre 4×4-Transformation (Drehung + Verschiebung), zeilenweise wie in numpy.
public struct Rigid: Equatable {
    public var rotation: Mat3
    public var translation: Vec3

    public init(rotation: Mat3, translation: Vec3) {
        self.rotation = rotation
        self.translation = translation
    }

    public static let identity = Rigid(rotation: .identity, translation: .zero)

    public static func * (a: Rigid, b: Rigid) -> Rigid {
        Rigid(rotation: a.rotation * b.rotation, translation: a.rotation * b.translation + a.translation)
    }

    public func apply(_ p: Vec3) -> Vec3 { rotation * p + translation }

    public var inverse: Rigid {
        let rt = rotation.transposed
        return Rigid(rotation: rt, translation: -(rt * translation))
    }

    /// Spaltenweise 16 Werte, wie glTF sie erwartet.
    public var columnMajor: [Float] {
        let r = rotation
        return [r[0, 0], r[1, 0], r[2, 0], 0,
                r[0, 1], r[1, 1], r[2, 1], 0,
                r[0, 2], r[1, 2], r[2, 2], 0,
                translation.x, translation.y, translation.z, 1].map { Float($0) }
    }
}

/// Drehmatrix → Quaternion (x, y, z, w) mit w ≥ 0, wie `export._matrix_to_trs`.
public func quaternion(_ r: Mat3) -> [Double] {
    let tr = r[0, 0] + r[1, 1] + r[2, 2]
    var q: [Double]
    if tr > 0 {
        let s = (tr + 1).squareRoot() * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    } else if r[0, 0] > r[1, 1] && r[0, 0] > r[2, 2] {
        let s = (1 + r[0, 0] - r[1, 1] - r[2, 2]).squareRoot() * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    } else if r[1, 1] > r[2, 2] {
        let s = (1 + r[1, 1] - r[0, 0] - r[2, 2]).squareRoot() * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    } else {
        let s = (1 + r[2, 2] - r[0, 0] - r[1, 1]).squareRoot() * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    }
    let n = q.reduce(0) { $0 + $1 * $1 }.squareRoot()
    q = q.map { $0 / n }
    if q[3] < 0 { q = q.map { -$0 } }
    return q
}
