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
        let geometry = SCNGeometry(
            sources: [SCNGeometrySource(vertices: mesh.positions), SCNGeometrySource(normals: mesh.normals)],
            elements: [SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles)]
        )
        geometry.materials = [skin]
        body.geometry = geometry
        height = mesh.height
        if !framed {
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

    var tapesVisible: Bool {
        get { !tapes.isHidden }
        set { tapes.isHidden = !newValue }
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
