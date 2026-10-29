import PhotosUI
import SwiftUI
import FitScanCore

/// Scan aus Fotos und Videos der Mediathek: Bilder wählen und prüfen, Maße eingeben, Modell berechnen.
struct PhotoScanView: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var scanner = PhotoScanner()
    @State private var selection: [PhotosPickerItem] = []
    var onDone: @MainActor (ScanRecord) -> Void

    var body: some View {
        Form {
            Section {
                PhotosPicker(selection: $selection, maxSelectionCount: 12, matching: .any(of: [.videos, .images])) {
                    Label(scanner.images.isEmpty ? "Video oder Fotos auswählen" : "Andere Auswahl",
                          systemImage: "photo.on.rectangle.angled")
                }
                .disabled(scanner.busy)
                if let value = scanner.analyzing {
                    ProgressView(value: value) { Text("Bilder werden geprüft …") }
                }
                if let error = scanner.error {
                    Text(error).foregroundStyle(.red)
                }
            } header: {
                Text("1. Video oder Fotos")
            } footer: {
                Text("Am besten ein Video, in dem du dich einmal langsam um dich selbst drehst, ganzer Körper im Bild. Oder 2 bis 8 Fotos von vorn, von der Seite und von hinten. Die Bilder bleiben auf deinem iPhone.")
            }

            ScanResultSections(scanner: scanner, onDone: onDone)
        }
        .navigationTitle("Scan aus Fotos und Video")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(scanner.fitting != nil)
        .interactiveDismissDisabled(scanner.busy)
        .onChange(of: selection) { _, items in
            Task { await scanner.load(items, fovLongSideDeg: app.photoSettings?.fovLongSideDeg ?? 69) }
        }
    }
}

/// Geprüfte Bilder, Angaben zur Person und Berechnen: gemeinsam für den Scan aus der Mediathek
/// und den Kamera-Scan.
struct ScanResultSections: View {
    @EnvironmentObject var app: AppModel
    @ObservedObject var scanner: PhotoScanner
    var onDone: @MainActor (ScanRecord) -> Void

    var body: some View {
        if !scanner.images.isEmpty {
            Section {
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: 10) {
                        ForEach(scanner.images) { thumb($0) }
                    }
                    .padding(.vertical, 4)
                }
            } header: {
                Text("\(scanner.usable.count) von \(scanner.images.count) Bildern brauchbar")
            } footer: {
                if scanner.usable.count > PhotoScanner.maxFrames {
                    Text("Verwendet werden \(PhotoScanner.maxFrames) Bilder, gleichmäßig über die Drehung verteilt.")
                }
            }
        }

        PersonSections()

        if !app.problems.isEmpty {
            Section {
                ForEach(app.problems, id: \.self) { Text($0).foregroundStyle(.red) }
            }
        }

        Section {
            if let fitting = scanner.fitting {
                ProgressView(value: fitting.value) { Text(fitting.text) }
            } else {
                Button("Modell aus den Bildern berechnen", action: compute)
                    .frame(maxWidth: .infinity)
                    .font(.headline)
                    .disabled(scanner.usable.isEmpty || scanner.busy)
            }
        } footer: {
            Text("Das dauert je nach Anzahl der Bilder etwa eine halbe bis zwei Minuten. Größe und Gewicht sind Pflicht, Maßband-Werte machen das Modell genauer.")
        }
    }

    private func thumb(_ image: ScanImage) -> some View {
        VStack(spacing: 4) {
            Image(uiImage: image.thumbnail)
                .resizable()
                .scaledToFill()
                .frame(width: 72, height: 110)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(alignment: .topTrailing) {
                    Image(systemName: image.problem == nil ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(Color.white, image.problem == nil ? Color.green : Color.red)
                        .padding(4)
                }
                .opacity(image.problem == nil ? 1 : 0.5)
            Text(image.problem ?? "brauchbar")
                .font(.caption2)
                .foregroundStyle(image.problem == nil ? Color.secondary : Color.red)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(width: 72)
        }
    }

    private func compute() {
        let frames = scanner.framesForFit
        let source = scanner.source
        scanner.fitting = ("Wird vorbereitet …", 0)
        Task {
            let record = await app.computePhotoScan(frames: frames, source: source) { text, value in
                if scanner.fitting != nil { scanner.fitting = (text, value) }
            }
            scanner.fitting = nil
            if let record { onDone(record) }
        }
    }
}
