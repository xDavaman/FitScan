import SwiftUI
import FitScanCore

/// Ein gespeichertes Modell: 3D-Vorschau mit Maßbändern, Maße gegen die Eingabe, Nachbearbeiten
/// und Weitergabe als GLB.
struct ModelView: View {
    @EnvironmentObject var app: AppModel
    let record: ScanRecord
    @StateObject private var scene = BodyScene()
    @State private var loaded = false
    @State private var showTapes = true
    @State private var selected: MeasureKey?
    @State private var fitViews: [FitPicture] = []

    var body: some View {
        List {
            Section {
                ZStack(alignment: .topTrailing) {
                    BodySceneView(scene: scene)
                    if !loaded { ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity) }
                    Button {
                        scene.resetView()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .padding(8)
                            .background(.thinMaterial, in: Circle())
                    }
                    .buttonStyle(.borderless)  // sonst löst jeder Tipp in die Zeile den Knopf aus
                    .padding(10)
                    .accessibilityLabel("Ansicht zurücksetzen")
                }
                .frame(height: 440)
                .listRowInsets(EdgeInsets())

                Toggle("Maßbänder zeigen", isOn: $showTapes)
                    .onChange(of: showTapes) { _, on in scene.tapesVisible = on }
            } footer: {
                Text("Mit einem Finger drehen, mit zwei Fingern zoomen. Tippe auf ein Maß, um sein Maßband hervorzuheben.")
            }

            Section("Maße des Modells") {
                ForEach(MeasureKey.allCases) { key in
                    if let value = record.measures[key] {
                        Button {
                            selected = selected == key ? nil : key
                            let tape = selected.flatMap { MeasureKey.circumferences.contains($0) ? $0 : nil }
                            if tape != nil { showTapes = true }
                            scene.highlight(tape)
                        } label: {
                            measureRow(key, value)
                        }
                        .foregroundStyle(.primary)
                    }
                }
            }

            if !fitViews.isEmpty {
                Section {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(fitViews) { view in
                                VStack(spacing: 4) {
                                    if let image = view.image {
                                        Image(uiImage: image)
                                            .resizable()
                                            .scaledToFit()
                                            .frame(height: 180)
                                            .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                    Text(String(format: "%.0f %% Deckung", locale: .deutsch, view.iou))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                } header: {
                    Text("Passform auf deinen Bildern")
                } footer: {
                    Text("Grün: Modell und Bild decken sich. Rot: nur im Bild, dort ist das Modell zu schmal. Blau: nur im Modell. Etwas Rot und Blau an Händen, Füßen und Haaren ist normal.")
                }
            }

            Section("Nachbearbeiten") {
                NavigationLink(value: Route.edit(record)) {
                    Label("Form anpassen (Muskeln, Fett, Körperstellen)", systemImage: "slider.horizontal.3")
                }
                Button {
                    app.openMeasureForm(from: record)
                } label: {
                    Label("Maße ändern und neu berechnen", systemImage: "ruler")
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
            } header: {
                Text("Weitergeben")
            } footer: {
                Text("Die Dateien liegen auch in der Dateien-App unter „Auf meinem iPhone › FitScan“. Am PC in Blender mit Datei › Importieren › glTF 2.0 öffnen.")
            }
        }
        .navigationTitle(record.person.name.isEmpty ? "Modell" : record.person.name)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            if fitViews.isEmpty {
                let views = app.fitViews(of: record)
                fitViews = await Task.detached(priority: .userInitiated) {
                    views.map { FitPicture(id: $0.id, iou: $0.iou * 100,
                                           image: $0.file.flatMap { UIImage(contentsOfFile: $0.path) }) }
                }.value
            }
            guard !loaded, let model = app.model, let params = app.params(of: record) else { return }
            let (mesh, tapes) = await Task.detached(priority: .userInitiated) {
                (BodyMesh(model: model, params: params), TapeMeshes(model: model, params: params))
            }.value
            scene.show(mesh)
            scene.show(tapes)
            scene.tapesVisible = showTapes
            loaded = true
        }
    }

    private func measureRow(_ key: MeasureKey, _ value: Double) -> some View {
        LabeledContent {
            HStack(spacing: 6) {
                Text(String(format: "%.1f %@", locale: .deutsch, value, key.unit))
                if let target = record.person.measures[key], target > 0 {
                    Text(String(format: "(Eingabe %.1f)", locale: .deutsch, target))
                        .foregroundStyle(.secondary)
                }
            }
        } label: {
            HStack(spacing: 8) {
                if MeasureKey.circumferences.contains(key) {
                    Circle()
                        .fill(Color(BodyScene.tapeColor).opacity(selected == nil || selected == key ? 1 : 0.3))
                        .frame(width: 8, height: 8)
                }
                Text(key.label)
                    .fontWeight(selected == key ? .semibold : .regular)
            }
        }
    }
}

/// Kontrollbild der Passform, einmal geladen.
struct FitPicture: Identifiable {
    let id: Int
    /// Überdeckung in Prozent
    let iou: Double
    let image: UIImage?
}
