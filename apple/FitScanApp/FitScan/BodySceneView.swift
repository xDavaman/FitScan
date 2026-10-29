import SceneKit
import SwiftUI
import UIKit
import FitScanCore

/// Netz für die Vorschau in Y-oben-Koordinaten, Füße auf dem Boden (wie im GLB).
struct BodyMesh {
    let positions: [SCNVector3]
    let normals: [SCNVector3]
    let indices: [UInt32]
    let height: Float

    init(_ mesh: PreviewMesh) {
        positions = mesh.vertices.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) }
        normals = mesh.normals.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) }
        indices = mesh.faces
        height = Float(mesh.height)
    }

    init(model: BodyModel, params: BodyParams) {
        self.init(model.previewMesh(params))
    }
}

/// Bilder einer Bewegung für die Vorschau: weicher Übergang aus der Ruhepose, die Bewegung selbst
/// und der Übergang zurück.
struct MotionFrames {
    var intro: [BodyMesh] = []
    var main: [BodyMesh] = []
    var outro: [BodyMesh] = []
}

/// Maßbänder als dünne Schläuche, je Umfang ein Netz.
struct TapeMeshes {
    struct Tube {
        let key: MeasureKey
        let positions: [SCNVector3]
        let normals: [SCNVector3]
        let indices: [UInt32]
    }

    let tubes: [Tube]

    init(lines: [MeasureKey: [Vec3]]) {
        var out: [Tube] = []
        for key in MeasureKey.circumferences {
            guard let line = lines[key] else { continue }
            let tube = TubeMesh(closedLine: line, radius: 0.0035)
            guard !tube.vertices.isEmpty else { continue }
            out.append(Tube(key: key,
                            positions: tube.vertices.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) },
                            normals: tube.normals.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) },
                            indices: tube.indices))
        }
        tubes = out
    }

    init(model: BodyModel, params: BodyParams) {
        self.init(lines: model.tapeLines(params))
    }
}

/// Die 3D-Szene der Vorschau: Körper, Licht, Boden, Kamera und Maßbänder. Wird von außen
/// aktualisiert (neues Netz beim Bearbeiten), ohne dass die Ansicht neu entsteht.
@MainActor
final class BodyScene: ObservableObject {
    let scene = SCNScene()
    let camera = SCNNode()
    private let turntable = SCNNode()
    private let body = SCNNode()
    private let tapes = SCNNode()
    private let skin = SCNMaterial()
    private(set) var height: Float = 1.8
    private var framed = false
    /// Bewegung, die gerade läuft (Schlüssel des Clips), sonst `nil`
    @Published private(set) var playing: String?
    private var restGeometry: SCNGeometry?
    /// Zuletzt gerechnete Bewegung, damit ein zweiter Tipp nicht neu rechnet
    private var lastMotion: Flipbook?
    /// Zählt Start und Stopp, damit eine verspätete Fertig-Meldung keine neuere Bewegung beendet
    private var motionToken = 0
    private var tapesWanted = false

    static let tapeColor = UIColor(red: 1.0, green: 0.55, blue: 0.12, alpha: 1)

    init(rotating: Bool = false) {
        skin.lightingModel = .physicallyBased
        skin.diffuse.contents = UIColor(red: 0.80, green: 0.62, blue: 0.52, alpha: 1)
        skin.roughness.contents = 0.65
        skin.metalness.contents = 0.0

        body.name = "Koerper"
        tapes.name = "Massbaender"
        tapes.isHidden = true
        turntable.addChildNode(body)
        turntable.addChildNode(tapes)
        scene.rootNode.addChildNode(turntable)

        let floor = SCNNode(geometry: SCNCylinder(radius: 0.55, height: 0.004))
        floor.geometry?.firstMaterial?.diffuse.contents = UIColor.label.withAlphaComponent(0.08)
        floor.geometry?.firstMaterial?.lightingModel = .constant
        floor.position = SCNVector3(0, -0.002, 0)
        scene.rootNode.addChildNode(floor)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 350
        scene.rootNode.addChildNode(ambient)

        let key = SCNNode()
        key.light = SCNLight()
        key.light?.type = .directional
        key.light?.intensity = 900
        key.eulerAngles = SCNVector3(-0.5, 0.45, 0)
        scene.rootNode.addChildNode(key)

        let rim = SCNNode()
        rim.light = SCNLight()
        rim.light?.type = .directional
        rim.light?.intensity = 450
        rim.eulerAngles = SCNVector3(-0.3, .pi * 0.85, 0)
        scene.rootNode.addChildNode(rim)

        camera.camera = SCNCamera()
        camera.camera?.fieldOfView = 30
        camera.camera?.zNear = 0.05
        scene.rootNode.addChildNode(camera)
        frameCamera()

        if rotating { setRotating(true) }
    }

    /// Neues Körpernetz zeigen. Die Kamera bleibt, wo der Nutzer sie hingedreht hat.
    func show(_ mesh: BodyMesh) {
        let geometry = makeGeometry(mesh, element: SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles))
        restGeometry = geometry
        lastMotion = nil  // gerechnete Bewegungen gehören zur alten Form
        if playing != nil { stopMotion() }
        body.geometry = geometry
        // Neu einrahmen beim ersten Netz oder wenn eine andere Person gezeigt wird; beim Bearbeiten
        // bleibt die Größe gleich und die Kamera da, wo der Nutzer sie hingedreht hat.
        let reframe = !framed || abs(mesh.height - height) > 0.01
        height = mesh.height
        if reframe {
            framed = true
            frameCamera()
        }
    }

    func show(_ tapeMeshes: TapeMeshes) {
        tapes.childNodes.forEach { $0.removeFromParentNode() }
        for tube in tapeMeshes.tubes {
            let geometry = SCNGeometry(
                sources: [SCNGeometrySource(vertices: tube.positions), SCNGeometrySource(normals: tube.normals)],
                elements: [SCNGeometryElement(indices: tube.indices, primitiveType: .triangles)]
            )
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = Self.tapeColor
            material.isDoubleSided = true
            geometry.materials = [material]
            let node = SCNNode(geometry: geometry)
            node.name = tube.key.rawValue
            tapes.addChildNode(node)
        }
    }

    /// Maßbänder zeigen. Während einer Bewegung bleiben sie aus, sie passen nur zur Ruhepose.
    var tapesVisible: Bool {
        get { tapesWanted }
        set {
            tapesWanted = newValue
            tapes.isHidden = !newValue || playing != nil
        }
    }

    private func makeGeometry(_ mesh: BodyMesh, element: SCNGeometryElement) -> SCNGeometry {
        let geometry = SCNGeometry(
            sources: [SCNGeometrySource(vertices: mesh.positions), SCNGeometrySource(normals: mesh.normals)],
            elements: [element]
        )
        geometry.materials = [skin]
        return geometry
    }

    private struct Flipbook {
        let key: String
        let intro: [SCNGeometry]
        let main: [SCNGeometry]
        let outro: [SCNGeometry]
        let fps: Double
        let loop: Bool
    }

    /// Bewegung abspielen: die fertig gerechneten Bilder nacheinander zeigen, mit weichem Übergang
    /// aus der Ruhepose. Eine Schleife läuft, bis `stopMotion` kommt; sonst einmal, mit Übergang zurück
    /// in die Ruhepose, danach läuft `then`.
    func play(_ key: String, frames: MotionFrames, fps: Double, loop: Bool, then: (@MainActor () -> Void)? = nil) {
        guard let first = frames.main.first, fps > 0 else { return }
        // Alle Bilder teilen sich die Dreiecke, nur Punkte und Normalen ändern sich
        let element = SCNGeometryElement(indices: first.indices, primitiveType: .triangles)
        lastMotion = Flipbook(key: key,
                              intro: frames.intro.map { makeGeometry($0, element: element) },
                              main: frames.main.map { makeGeometry($0, element: element) },
                              outro: frames.outro.map { makeGeometry($0, element: element) },
                              fps: fps, loop: loop)
        replay(key, then: then)
    }

    /// Die zuletzt gerechnete Bewegung noch einmal abspielen. `false`, wenn sie neu gerechnet werden muss.
    @discardableResult
    func replay(_ key: String, then: (@MainActor () -> Void)? = nil) -> Bool {
        guard let motion = lastMotion, motion.key == key else { return false }
        body.removeAction(forKey: "bewegung")
        motionToken += 1
        let token = motionToken
        playing = key
        tapes.isHidden = true
        if motion.loop {
            body.runAction(Self.flipbook(motion.intro + motion.main, fps: motion.fps, repeatFrom: motion.intro.count),
                           forKey: "bewegung")
        } else {
            let done = Self.onMain { [weak self] in
                guard let self, self.motionToken == token else { return }
                self.stopMotion()
                then?()
            }
            let all = motion.intro + motion.main + motion.outro
            body.runAction(.sequence([Self.flipbook(all, fps: motion.fps, repeatFrom: nil), done]), forKey: "bewegung")
        }
        return true
    }

    /// Bewegung anhalten und die Ruhepose zeigen.
    func stopMotion() {
        motionToken += 1
        body.removeAction(forKey: "bewegung")
        if let restGeometry { body.geometry = restGeometry }
        playing = nil
        tapes.isHidden = !tapesWanted
    }

    /// Tauscht das Netz des Knotens Bild für Bild. Mit `repeatFrom` laufen die Bilder ab dort in
    /// Schleife (bis `stopMotion`, längstens einen Tag), sonst einmal. Außerhalb des Main Actors gebaut,
    /// weil SceneKit Aktionen auf seinem eigenen Render-Thread ausführt.
    nonisolated private static func flipbook(_ geometries: [SCNGeometry], fps: Double, repeatFrom: Int?) -> SCNAction {
        let count = geometries.count
        let duration = repeatFrom == nil ? Double(count) / fps : 24 * 3600
        return .customAction(duration: duration) { node, elapsed in
            var i = max(Int(Double(elapsed) * fps), 0)
            if let start = repeatFrom, i >= count, start < count { i = start + (i - start) % (count - start) }
            let geometry = geometries[min(i, count - 1)]
            if node.geometry !== geometry { node.geometry = geometry }
        }
    }

    nonisolated private static func onMain(_ work: @escaping @MainActor () -> Void) -> SCNAction {
        .run { _ in Task { @MainActor in work() } }
    }

    /// Ein Maßband hervorheben, die anderen blasser. `nil` zeigt alle gleich.
    func highlight(_ key: MeasureKey?) {
        for node in tapes.childNodes {
            node.opacity = key == nil || node.name == key?.rawValue ? 1.0 : 0.2
        }
    }

    func setRotating(_ on: Bool) {
        if on {
            guard turntable.action(forKey: "drehen") == nil else { return }
            turntable.runAction(.repeatForever(.rotateBy(x: 0, y: 2 * .pi, z: 0, duration: 24)), forKey: "drehen")
        } else {
            turntable.removeAction(forKey: "drehen")
        }
    }

    /// Ganzer Körper von vorn, leicht von oben.
    func frameCamera() {
        let h = height
        camera.position = SCNVector3(0, h * 0.58, h * 2.25)
        camera.look(at: SCNVector3(0, h * 0.5, 0))
    }

    func resetView() {
        turntable.removeAction(forKey: "drehen")
        turntable.eulerAngles = SCNVector3(0, 0, 0)
        frameCamera()
    }

    var target: SCNVector3 { SCNVector3(0, height * 0.5, 0) }
}

/// SceneKit-Ansicht mit durchsichtigem Hintergrund. Ein Finger dreht um den Körper, zwei zoomen.
struct BodySceneView: UIViewRepresentable {
    @ObservedObject var scene: BodyScene
    var allowsControl = true

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene.scene
        view.pointOfView = scene.camera
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = false
        view.isPlaying = true
        view.allowsCameraControl = allowsControl
        // Sonst rendert SceneKit nach der ersten Geste durch eine eigene Kamerakopie, und
        // Zurücksetzen oder Neu-Einrahmen wirken nicht mehr
        view.cameraControlConfiguration.autoSwitchToFreeCamera = false
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.target = scene.target
        view.defaultCameraController.maximumVerticalAngle = 60
        view.defaultCameraController.minimumVerticalAngle = -20
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.allowsCameraControl = allowsControl
        view.defaultCameraController.target = scene.target
        if view.pointOfView !== scene.camera { view.pointOfView = scene.camera }
    }
}
