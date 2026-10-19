import SwiftUI
import FitScanCore

/// Ein gespeichertes Modell: 3D-Vorschau, Maße gegen die Eingabe und Weitergabe als GLB.
struct ModelView: View {
    @EnvironmentObject var app: AppModel
    let record: ScanRecord
    @State private var mesh: BodyMesh?

    var body: some View {
        List {
            Section {
                ZStack {
                    if let mesh {
                        BodySceneView(mesh: mesh)
                    } else {
                        ProgressView()
                    }
                }
                .frame(height: 420)
                .listRowInsets(EdgeInsets())
            } footer: {
                Text("Mit einem Finger drehen, mit zwei Fingern zoomen.")
            }

            Section("Maße des Modells") {
                ForEach(MeasureKey.allCases) { key in
                    if let value = record.measures[key] {
                        LabeledContent(key.label) {
                            HStack(spacing: 6) {
                                Text(String(format: "%.1f %@", locale: .deutsch, value, key.unit))
                                if let target = record.person.measures[key], target > 0 {
                                    Text(String(format: "(Eingabe %.1f)", locale: .deutsch, target))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }

            Section {
                if let url = app.file("koerper.glb", of: record) {
                    ShareLink(item: url) { Label("Für Blender teilen (koerper.glb)", systemImage: "square.and.arrow.up") }
                }
                if let url = app.file("koerper_unreal.glb", of: record) {
                    ShareLink(item: url) {
                        Label("Für Cascadeur teilen (koerper_unreal.glb)", systemImage: "square.and.arrow.up")
                    }
                }
                Button {
                    app.reuse(record)
                } label: {
                    Label("Eingaben für ein neues Modell übernehmen", systemImage: "arrow.uturn.backward")
                }
            } header: {
                Text("Weitergeben")
            } footer: {
                Text("Die Dateien liegen auch in der Dateien-App unter „Auf meinem iPhone › FitScan“. Am PC in Blender mit Datei › Importieren › glTF 2.0 öffnen.")
            }
        }
        .navigationTitle(record.person.name.isEmpty ? "Modell" : record.person.name)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            guard mesh == nil, let model = app.model, let params = app.params(of: record) else { return }
            mesh = await Task.detached(priority: .userInitiated) { BodyMesh(model: model, params: params) }.value
        }
    }
}
