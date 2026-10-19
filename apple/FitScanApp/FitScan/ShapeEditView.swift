import SwiftUI
import FitScanCore

/// Rechnet beim Ziehen der Regler die Vorschau neu: immer nur eine Rechnung zugleich, danach
/// gleich die nächste mit dem neuesten Stand. Maße erst, wenn der Finger losgelassen hat.
@MainActor
final class ShapeEditor: ObservableObject {
    let model: BodyModel
    let original: BodyParams
    let heightCM: Double?
    let scene = BodyScene()
    @Published var values: [ShapeRegulator: Double]
    @Published private(set) var measures: [MeasureKey: Double]
    @Published private(set) var measuring = false

    private var running = false
    private var dirty = false
    private var measureRun = 0

    init(model: BodyModel, record: ScanRecord, params: BodyParams) {
        self.model = model
        original = params
        heightCM = record.measures[.height]
        measures = record.measures
        var start: [ShapeRegulator: Double] = [:]
        for r in ShapeRegulator.body + ShapeRegulator.regions { start[r] = r.value(in: params) }
        values = start
    }

    var changed: Bool {
        values.contains { abs($0.value - $0.key.value(in: original)) > 1e-3 }
    }

    /// Aktuelle Regler als Annys Werte, mit der ursprünglichen Körpergröße.
    func currentParams() -> BodyParams {
        model.adjusted(original, values, heightCM: heightCM)
    }

    func binding(_ regulator: ShapeRegulator) -> Binding<Double> {
        Binding(
            get: { self.values[regulator] ?? regulator.neutral },
            set: { value in
                self.values[regulator] = value
                self.refreshMesh()
            }
        )
    }

    func reset() {
        for r in values.keys { values[r] = r.value(in: original) }
        refreshMesh()
        refreshMeasures()
    }

    func refreshMesh() {
        dirty = true
        guard !running else { return }
        running = true
        dirty = false
        let model = self.model, original = self.original, values = self.values, heightCM = self.heightCM
        Task {
            let mesh = await Task.detached(priority: .userInitiated) {
                BodyMesh(model: model, params: model.adjusted(original, values, heightCM: heightCM))
            }.value
            scene.show(mesh)
            running = false
            if dirty { refreshMesh() }
        }
    }

    func refreshMeasures() {
        measureRun += 1
        let run = measureRun
        measuring = true
        let model = self.model, original = self.original, values = self.values, heightCM = self.heightCM
        Task {
            let (result, tapes) = await Task.detached(priority: .utility) {
                let p = model.adjusted(original, values, heightCM: heightCM)
                return (model.measure(p), TapeMeshes(model: model, params: p))
            }.value
            guard run == measureRun else { return }  // inzwischen weitergezogen
            measures = result
            scene.show(tapes)
            measuring = false
        }
    }
}

/// Nachbearbeitung: Regler für den ganzen Körper und für einzelne Körperstellen, mit Live-Vorschau.
/// Gespeichert wird als neues Modell im Verlauf, das alte bleibt erhalten.
struct ShapeEditView: View {
    @EnvironmentObject var app: AppModel
    let record: ScanRecord
    @State private var editor: ShapeEditor?
    @State private var saving = false
    @State private var error: String?

    var body: some View {
        Group {
            if let editor {
                ShapeEditContent(editor: editor, record: record, saving: saving, error: error, save: save)
            } else {
                ProgressView("Modell wird geladen …")
            }
        }
        .navigationTitle("Form anpassen")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            guard editor == nil, let model = app.model, let params = app.params(of: record) else { return }
            let e = ShapeEditor(model: model, record: record, params: params)
            editor = e
            e.scene.tapesVisible = true
            e.refreshMesh()
            e.refreshMeasures()
        }
    }

    private func save() {
        guard let editor, !saving else { return }
        saving = true
        error = nil
        let params = editor.currentParams()
        Task {
            do {
                let saved = try await app.saveEdited(record, params: params)
                app.path = [.model(saved)]
            } catch {
                self.error = error.localizedDescription
            }
            saving = false
        }
    }
}

private struct ShapeEditContent: View {
    @ObservedObject var editor: ShapeEditor
    let record: ScanRecord
    let saving: Bool
    let error: String?
    let save: @MainActor () -> Void

    var body: some View {
        VStack(spacing: 0) {
            BodySceneView(scene: editor.scene)
                .frame(height: 300)
                .background(Color(.secondarySystemBackground))

            List {
                Section {
                    ForEach(ShapeRegulator.body) { slider($0) }
                } header: {
                    Text("Ganzer Körper")
                } footer: {
                    Text("Die Körpergröße bleibt gleich, die Regler ändern nur die Form.")
                }

                Section("Körperstellen") {
                    ForEach(ShapeRegulator.regions) { slider($0) }
                }

                if let error {
                    Section { Text(error).foregroundStyle(.red) }
                }

                Section {
                    Button(action: save) {
                        HStack {
                            Spacer()
                            if saving { ProgressView() } else { Text("Als neues Modell speichern").font(.headline) }
                            Spacer()
                        }
                    }
                    .disabled(!editor.changed || saving)
                    Button("Alle Regler zurücksetzen", role: .destructive) { editor.reset() }
                        .disabled(!editor.changed || saving)
                } footer: {
                    Text("Das bisherige Modell bleibt im Verlauf erhalten.")
                }
            }
        }
    }

    private func slider(_ r: ShapeRegulator) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(r.label).font(.body.weight(.medium))
                Spacer()
                if let key = r.measure, let value = editor.measures[key] {
                    Text(measureText(key, value))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(editor.measuring ? HierarchicalShapeStyle.tertiary : HierarchicalShapeStyle.secondary)
                }
            }
            Slider(value: editor.binding(r), in: r.range,
                   label: { Text(r.label) },
                   minimumValueLabel: { Text(r.lowLabel).font(.caption2).foregroundStyle(.secondary) },
                   maximumValueLabel: { Text(r.highLabel).font(.caption2).foregroundStyle(.secondary) },
                   onEditingChanged: { editing in
                       // Maße und Maßbänder erst nach dem Loslassen, das dauert länger als die Form
                       if !editing { editor.refreshMeasures() }
                   })
        }
        .padding(.vertical, 2)
    }

    private func measureText(_ key: MeasureKey, _ value: Double) -> String {
        let now = String(format: "%.1f %@", locale: .deutsch, value, key.unit)
        guard let before = record.measures[key], abs(value - before) >= 0.1 else { return now }
        return now + String(format: " (%+.1f)", locale: .deutsch, value - before)
    }
}
