import SwiftUI
import FitScanCore

/// Eingabe der Person: Pflicht sind Größe und Gewicht, die Maßband-Werte machen das Modell genauer.
struct MeasureFormView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    var onDone: (ScanRecord) -> Void

    private static let optional: [MeasureKey] = [.chest, .waist, .hips, .thigh, .upperarm, .calf, .neck, .inseam]

    var body: some View {
        Form {
            Section("Person") {
                TextField("Name", text: $app.person.name)
                    .textContentType(.name)
                Picker("Geschlecht", selection: $app.person.gender) {
                    ForEach(Gender.allCases) { Text($0.label).tag($0) }
                }
                Stepper(value: $app.person.ageYears, in: 1...90, step: 1) {
                    LabeledContent("Alter", value: "\(Int(app.person.ageYears)) Jahre")
                }
            }

            Section {
                measureField(.height)
                measureField(.weight)
            } header: {
                Text("Pflicht")
            }

            Section {
                ForEach(Self.optional) { measureField($0) }
            } header: {
                Text("Maßband (freiwillig)")
            } footer: {
                Text("Brust unter den Achseln, Taille an der schmalsten Stelle, Hüfte an der breitesten Stelle des Gesäßes. Oberschenkel, Oberarm und Wade links an der dicksten Stelle, entspannt.")
            }

            if !app.problems.isEmpty {
                Section {
                    ForEach(app.problems, id: \.self) { Text($0).foregroundStyle(.red) }
                }
            }

            Section {
                if let progress = app.progress {
                    ProgressView(value: progress) { Text("Modell wird berechnet …") }
                } else {
                    Button("Modell berechnen") {
                        Task {
                            await app.compute()
                            if app.problems.isEmpty, let record = app.lastRecord { onDone(record) }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .font(.headline)
                }
            }
        }
        .navigationTitle("Deine Maße")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Abbrechen") { dismiss() }.disabled(app.progress != nil)
            }
        }
        .interactiveDismissDisabled(app.progress != nil)
    }

    private func measureField(_ key: MeasureKey) -> some View {
        let binding = Binding<Double?>(
            get: { app.person.measures[key] },
            set: { app.person.measures[key] = $0 }
        )
        return LabeledContent(key.label) {
            HStack(spacing: 4) {
                TextField("", value: binding, format: .number.precision(.fractionLength(0...1)))
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 90)
                Text(key.unit).foregroundStyle(.secondary)
            }
        }
    }
}
