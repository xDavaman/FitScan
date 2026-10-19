import SwiftUI
import FitScanCore

/// Wurzel der App: Startbildschirm, darüber Modell, Bearbeiten, Verlauf und Anleitung.
struct ContentView: View {
    @EnvironmentObject var app: AppModel
    @AppStorage("begruessungGesehen") private var welcomeSeen = false

    var body: some View {
        NavigationStack(path: $app.path) {
            Group {
                switch app.phase {
                case .loading:
                    ProgressView("Körpervorlage wird geladen …")
                case .failed(let text):
                    ContentUnavailableView("FitScan konnte nicht starten", systemImage: "exclamationmark.triangle",
                                           description: Text(text))
                case .ready:
                    HomeView()
                }
            }
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .model(let record): ModelView(record: record)
                case .edit(let record): ShapeEditView(record: record)
                case .history: HistoryView()
                case .guide: ScanGuideView()
                }
            }
        }
        .sheet(item: $app.sheet) { sheet in
            NavigationStack {
                switch sheet {
                case .newScan: NewScanView()
                case .measures: MeasureFormView(onDone: { app.finishedMeasureForm($0) })
                }
            }
        }
        .fullScreenCover(isPresented: Binding(get: { !welcomeSeen }, set: { welcomeSeen = !$0 })) {
            WelcomeView { welcomeSeen = true }
        }
    }
}

/// Startbildschirm: Begrüßung, das eigene Modell groß im Vordergrund, darunter die nächsten Schritte.
struct HomeView: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var scene = BodyScene(rotating: true)
    @State private var shownID: String?

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                Text(greeting)
                    .font(.largeTitle.weight(.bold))
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.top, 8)

            ZStack(alignment: .bottom) {
                BodySceneView(scene: scene)
                if app.latest == nil {
                    Text("Beispielmodell. Dein eigenes erscheint hier nach dem ersten Scan.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                        .padding(.bottom, 8)
                }
            }
            .frame(maxHeight: .infinity)

            VStack(spacing: 12) {
                Button {
                    app.sheet = .newScan
                } label: {
                    Label("Neuer Scan", systemImage: "figure.stand")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)

                HStack(spacing: 12) {
                    if let latest = app.latest {
                        shortcut("Mein Modell", "cube.transparent") { app.path = [.model(latest)] }
                    }
                    shortcut("Verlauf", "clock.arrow.circlepath") { app.path = [.history] }
                        .disabled(app.scans.isEmpty)
                    shortcut("Anleitung", "list.bullet.clipboard") { app.path = [.guide] }
                }
            }
            .padding()
        }
        .background(
            LinearGradient(colors: [Color.accentColor.opacity(0.18), Color(.systemBackground)],
                           startPoint: .top, endPoint: .center)
                .ignoresSafeArea()
        )
        .toolbar(.hidden, for: .navigationBar)
        .task(id: app.latest?.id ?? "beispiel") { await showLatest() }
    }

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        let base = hour < 11 ? "Guten Morgen" : hour < 18 ? "Hallo" : "Guten Abend"
        if let name = app.firstName { return "\(base), \(name)" }
        return "Willkommen bei FitScan"
    }

    private var subtitle: String {
        if let latest = app.latest {
            return "Dein Modell vom \(latest.title), \(latest.sourceLabel)."
        }
        return "Dein Körper als 3D-Modell, offline auf deinem iPhone."
    }

    private func shortcut(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon).font(.title3)
                Text(title).font(.caption)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
        }
        .buttonStyle(.bordered)
    }

    private func showLatest() async {
        guard let model = app.model else { return }
        let record = app.latest
        guard shownID != (record?.id ?? "beispiel") else { return }
        let params = record.flatMap { app.params(of: $0) } ?? BodyParams()
        let mesh = await Task.detached(priority: .userInitiated) { BodyMesh(model: model, params: params) }.value
        scene.show(mesh)
        shownID = record?.id ?? "beispiel"
    }
}

/// Wie das neue Modell entstehen soll. Foto/Video und Kamera folgen in den nächsten Etappen.
struct NewScanView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var prefilled = false

    var body: some View {
        List {
            Section {
                NavigationLink {
                    MeasureFormView(showsCancel: false, onDone: { app.finishedMeasureForm($0) })
                } label: {
                    method("Mit deinen Maßen", "Größe und Gewicht, dazu nach Wunsch Maßband-Werte.",
                           "ruler", available: true)
                }
                method("Aus Fotos und Videos", "Kommt in einem der nächsten Builds.", "photo.on.rectangle",
                       available: false)
                method("Mit der Kamera", "Geführter Scan mit Ampel, kommt in einem der nächsten Builds.",
                       "camera.viewfinder", available: false)
            }
            Section {
                Button {
                    dismiss()
                    app.path = [.guide]
                } label: {
                    Label("Scan-Anleitung lesen", systemImage: "list.bullet.clipboard")
                }
            }
        }
        .navigationTitle("Neuer Scan")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Abbrechen") { dismiss() } }
        }
        .task {
            // Nur beim Öffnen, sonst gingen Eingaben beim Zurückblättern verloren
            guard !prefilled else { return }
            prefilled = true
            app.prefillPerson()
        }
    }

    private func method(_ title: String, _ detail: String, _ icon: String, available: Bool) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.title2)
                .frame(width: 36)
                .foregroundStyle(available ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline).foregroundStyle(available ? .primary : .secondary)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

/// Alle gespeicherten Modelle, neueste zuerst. Nachbearbeitete Fassungen stehen als eigene Einträge da.
struct HistoryView: View {
    @EnvironmentObject var app: AppModel

    var body: some View {
        List {
            ForEach(app.scans) { record in
                NavigationLink(value: Route.model(record)) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(record.person.name.isEmpty ? "Ohne Namen" : record.person.name)
                            .font(.body.weight(.semibold))
                        Text("\(record.title) · \(record.sourceLabel) · \(summary(record))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { offsets in
                offsets.map { app.scans[$0] }.forEach { app.delete($0) }
            }
        }
        .overlay {
            if app.scans.isEmpty {
                ContentUnavailableView("Noch keine Modelle", systemImage: "figure.stand",
                                       description: Text("Starte auf dem Startbildschirm einen neuen Scan."))
            }
        }
        .navigationTitle("Verlauf")
    }

    private func summary(_ r: ScanRecord) -> String {
        let h = r.measures[.height].map { String(format: "%.0f cm", locale: .deutsch, $0) } ?? "–"
        let w = r.measures[.weight].map { String(format: "%.1f kg", locale: .deutsch, $0) } ?? "–"
        return "\(h), \(w)"
    }
}
