import SceneKit
import SwiftUI
import FitScanCore

/// Netz für die Vorschau in Y-oben-Koordinaten, Füße auf dem Boden (wie im GLB).
struct BodyMesh {
    let positions: [SCNVector3]
    let normals: [SCNVector3]
    let indices: [UInt32]
    let height: Float

    init(model: BodyModel, params: BodyParams) {
        let rig = model.buildRig(params, withMorphs: false)
        positions = rig.vertices.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) }
        normals = rig.normals.map { SCNVector3(Float($0.x), Float($0.y), Float($0.z)) }
        indices = rig.faces
        height = rig.vertices.map { Float($0.y) }.max() ?? 1.8
    }
}

struct BodySceneView: View {
    let mesh: BodyMesh
    @State private var scene: SCNScene
    @State private var pointOfView: SCNNode

    init(mesh: BodyMesh) {
        self.mesh = mesh
        _scene = State(initialValue: Self.makeScene(mesh))
        _pointOfView = State(initialValue: Self.camera(mesh))
    }

    var body: some View {
        SceneView(scene: scene, pointOfView: pointOfView,
                  options: [.allowsCameraControl, .autoenablesDefaultLighting])
    }

    private static func makeScene(_ mesh: BodyMesh) -> SCNScene {
        let scene = SCNScene()
        scene.background.contents = UIColor.systemGroupedBackground
        let geometry = SCNGeometry(
            sources: [SCNGeometrySource(vertices: mesh.positions), SCNGeometrySource(normals: mesh.normals)],
            elements: [SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles)]
        )
        let skin = SCNMaterial()
        skin.lightingModel = .physicallyBased
        skin.diffuse.contents = UIColor(red: 0.80, green: 0.62, blue: 0.52, alpha: 1)
        skin.roughness.contents = 0.6
        geometry.materials = [skin]
        let body = SCNNode(geometry: geometry)
        body.name = "Koerper"
        scene.rootNode.addChildNode(body)

        let floor = SCNNode(geometry: SCNCylinder(radius: 0.6, height: 0.002))
        floor.geometry?.firstMaterial?.diffuse.contents = UIColor.secondarySystemFill
        floor.position = SCNVector3(0, -0.001, 0)
        scene.rootNode.addChildNode(floor)
        return scene
    }

    private static func camera(_ mesh: BodyMesh) -> SCNNode {
        let node = SCNNode()
        node.camera = SCNCamera()
        node.camera?.fieldOfView = 35
        node.position = SCNVector3(0, mesh.height * 0.55, mesh.height * 1.9)
        node.look(at: SCNVector3(0, mesh.height * 0.5, 0))
        return node
    }
}
