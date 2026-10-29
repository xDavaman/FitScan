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
    /// Bewegung, deren Bilder gerade gerechnet werden
    @State private var computingMotion: String?
    @State private var motionFiles: [BoneNaming: URL] = [:]
    @State private var writingFiles = false
    @State private var fileError: String?

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

            if !app.motions.isEmpty {
                Section {
                    HStack(spacing: 8) {
                        ForEach(app.motions) { clip in motionButton(clip) }
                    }
                } header: {
                    Text("Bewegungen")
                } footer: {
                    Text("Ein zweiter Tipp hält die Bewegung an. Gehen läuft hier auf der Stelle, in Blender und Cascadeur geht die Figur vorwärts. Sprechen und Rufen bewegen Mund und Gesicht, Ton gibt es keinen.")
                }
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
                if let url = motionFiles[.anny] {
                    ShareLink(item: url) {
                        Label("Mit Bewegungen für Blender teilen", systemImage: "square.and.arrow.up")
                    }
                }
                if let url = motionFiles[.unreal] {
                    ShareLink(item: url) {
                        Label("Mit Bewegungen für Cascadeur teilen", systemImage: "square.and.arrow.up")
                    }
                }
                if motionFiles.count < BoneNaming.allCases.count && !app.motions.isEmpty {
                    Button {
                        Task { await writeMotionFiles() }
                    } label: {
                        HStack {
                            Label("Dateien mit Bewegungen erstellen", systemImage: "figure.walk.motion")
                            if writingFiles {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }
                    .disabled(writingFiles)
                }
                if let fileError {
                    Text(fileError)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            } header: {
                Text("Weitergeben")
            } footer: {
                Text("Mit Bewegungen: Winken, Rufen, Sprechen und Gehen als Animationen, dazu 52 Gesichtsregler mit den Namen von Apple ARKit. Die Dateien liegen auch in der Dateien-App unter „Auf meinem iPhone › FitScan“. Am PC in Blender mit Datei › Importieren › glTF 2.0 öffnen.")
            }
        }
        .navigationTitle(record.person.name.isEmpty ? "Modell" : record.person.name)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            motionFiles = app.motionFiles(of: record)
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

    private func motionButton(_ clip: MotionClip) -> some View {
        let active = scene.playing == clip.key
        return Button {
            Task { await toggleMotion(clip) }
        } label: {
            VStack(spacing: 4) {
                if computingMotion == clip.key {
                    ProgressView().frame(height: 24)
                } else {
                    Image(systemName: active ? "stop.fill" : Self.motionIcon(clip.key))
                        .font(.title3)
                        .frame(height: 24)
                }
                Text(clip.label).font(.caption)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)
        }
        .buttonStyle(.bordered)
        .tint(active ? Color.red : Color.accentColor)
        .disabled(!loaded || (computingMotion != nil && computingMotion != clip.key))
        .accessibilityLabel(active ? "\(clip.label) anhalten" : clip.label)
    }

    private static func motionIcon(_ key: String) -> String {
        switch key {
        case "winken": return "hand.wave"
        case "rufen": return "megaphone"
        case "sprechen": return "bubble.left"
        case "gehen": return "figure.walk"
        default: return "play"
        }
    }

    /// Bewegung starten oder, wenn sie schon läuft, anhalten. Die Bilder werden beim ersten Mal gerechnet.
    private func toggleMotion(_ clip: MotionClip) async {
        guard computingMotion == nil else { return }
        if scene.playing == clip.key {
            scene.stopMotion()
            return
        }
        if scene.replay(clip.key) { return }
        guard let params = app.params(of: record) else { return }
        computingMotion = clip.key
        let frames = await app.motionFrames(clip, params: params)
        computingMotion = nil
        scene.play(clip.key, frames: frames, fps: clip.fps, loop: clip.loop)
    }

    private func writeMotionFiles() async {
        writingFiles = true
        fileError = nil
        do {
            for naming in BoneNaming.allCases {
                motionFiles[naming] = try await app.motionGLB(record, naming: naming)
            }
        } catch {
            fileError = "Datei ließ sich nicht schreiben: \(error.localizedDescription)"
        }
        writingFiles = false
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
