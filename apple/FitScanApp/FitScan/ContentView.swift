import SwiftUI
import FitScanCore

struct ContentView: View {
    @EnvironmentObject var app: AppModel
    @State private var path: [ScanRecord] = []
    @State private var showForm = false

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                switch app.phase {
                case .loading:
                    ProgressView("Körpervorlage wird geladen …")
                case .failed(let text):
                    ContentUnavailableView("FitScan konnte nicht starten", systemImage: "exclamationmark.triangle",
                                           description: Text(text))
                case .ready:
                    list
                }
            }
            .navigationTitle("FitScan")
            .navigationDestination(for: ScanRecord.self) { record in
                ModelView(record: record)
            }
            .sheet(isPresented: $showForm) {
                NavigationStack { MeasureFormView(onDone: { record in
                    showForm = false
                    path = [record]
                }) }
            }
        }
    }

    private var list: some View {
        List {
            Section {
                Button {
                    showForm = true
                } label: {
                    Label("Neues Modell aus deinen Maßen", systemImage: "figure.stand")
                        .font(.headline)
                }
            } footer: {
                Text("In diesem Build entsteht das Modell aus Größe, Gewicht und den Maßband-Werten. Der Video-Scan kommt in einem der nächsten Builds.")
            }

            if !app.scans.isEmpty {
                Section("Gespeicherte Modelle") {
                    ForEach(app.scans) { record in
                        NavigationLink(value: record) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(record.person.name.isEmpty ? "Ohne Namen" : record.person.name)
                                    .font(.body.weight(.semibold))
                                Text("\(record.title) · \(summary(record))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { offsets in
                        for i in offsets { app.delete(app.scans[i]) }
                    }
                }
            }
        }
    }

    private func summary(_ r: ScanRecord) -> String {
        let h = r.measures[.height].map { String(format: "%.0f cm", $0) } ?? "–"
        let w = r.measures[.weight].map { String(format: "%.1f kg", $0) } ?? "–"
        return "\(h), \(w)"
    }
}
