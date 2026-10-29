import AVFoundation
import SwiftUI
import FitScanCore

/// Kamera-Scan: Ampel und Ansage, bis alles passt, dann eine Drehung vor der Kamera. Die Bilder werden
/// danach geprüft und berechnet wie beim Scan aus Fotos und Videos.
struct CameraScanView: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var scanner = PhotoScanner()
    @AppStorage("kameraVorn") private var front = true
    @AppStorage("kameraAnsage") private var speak = true
    @State private var capturing = false
    var onDone: @MainActor (ScanRecord) -> Void

    var body: some View {
        Form {
            Section {
                Picker("Kamera", selection: $front) {
                    Text("Vorn (Selfie)").tag(true)
                    Text("Hinten").tag(false)
                }
                .pickerStyle(.segmented)
                Toggle("Ansagen", isOn: $speak)
                Button {
                    capturing = true
                } label: {
                    Label(scanner.images.isEmpty ? "Kamera starten" : "Neu aufnehmen", systemImage: "camera.viewfinder")
                }
                .disabled(scanner.busy)
                if let value = scanner.analyzing {
                    ProgressView(value: value) { Text("Bilder werden geprüft …") }
                }
                if let error = scanner.error {
                    Text(error).foregroundStyle(.red)
                }
            } header: {
                Text("1. Aufnahme")
            } footer: {
                Text("Handy hochkant auf etwa Hüfthöhe hinstellen, gut zwei Meter Abstand. Die Ampel zeigt, ob Licht, Abstand und Haltung passen. Bei Grün zählt die App herunter, dann drehst du dich einmal langsam um dich selbst. Die Ansagen helfen, wenn du das Display nicht siehst (hintere Kamera, Rücken zur Kamera). Die Bilder bleiben auf deinem iPhone.")
            }

            ScanResultSections(scanner: scanner, onDone: onDone)
        }
        .navigationTitle("Kamera-Scan")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(scanner.fitting != nil)
        .interactiveDismissDisabled(scanner.busy)
        .fullScreenCover(isPresented: $capturing) {
            CameraCaptureView(front: front, speak: speak) { frames, fov in
                capturing = false
                Task { await scanner.load(captured: frames, fovLongSideDeg: fov) }
            }
        }
    }
}

/// Vollbild während der Aufnahme: Kamerabild, Ampel mit dem wichtigsten Hinweis, Countdown und
/// Fortschritt der Drehung.
struct CameraCaptureView: View {
    let front: Bool
    let speak: Bool
    var onFinished: @MainActor ([Data], Double) -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    @StateObject private var camera = CameraScanModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: camera.feed.session, ready: camera.phase != .starting)
                .ignoresSafeArea()
            Rectangle()
                .strokeBorder(levelColor.opacity(showsLevel ? 0.9 : 0), lineWidth: 8)
                .ignoresSafeArea()
                .animation(.easeInOut(duration: 0.25), value: camera.check?.level)
            VStack(spacing: 12) {
                HStack(alignment: .top) {
                    Button("Abbrechen") { dismiss() }
                        .buttonStyle(.borderedProminent)
                        .tint(Color.black.opacity(0.6))
                    Spacer()
                    if showsLevel { trafficLight }
                }
                if showsLevel, let hint = topHint {
                    Text(hint.text)
                        .font(.headline)
                        .multilineTextAlignment(.center)
                        .padding(12)
                        .frame(maxWidth: .infinity)
                        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.white)
                }
                Spacer()
                if case .countdown(let n) = camera.phase {
                    Text("\(n)")
                        .font(.system(size: 120, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .shadow(radius: 8)
                        .transition(.scale)
                        .id(n)
                }
                Spacer()
                panel
                    .padding(14)
                    .frame(maxWidth: .infinity)
                    .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 16))
                    .foregroundStyle(.white)
            }
            .padding()
        }
        .statusBarHidden()
        .task {
            camera.onFinished = onFinished
            await camera.start(front: front, speak: speak)
        }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            camera.stop()
        }
    }

    private var showsLevel: Bool {
        switch camera.phase {
        case .setup, .countdown, .turning: return camera.check != nil
        default: return false
        }
    }

    private var levelColor: Color {
        switch camera.check?.level {
        case .gruen?: return .green
        case .gelb?: return .yellow
        case .rot?: return .red
        case nil: return .gray
        }
    }

    /// Wichtigster Hinweis: der erste der höchsten Stufe
    private var topHint: ScanHint? {
        guard let check = camera.check else { return nil }
        return check.hints.first { $0.level == check.level } ?? check.hints.first
    }

    private var trafficLight: some View {
        VStack(spacing: 6) {
            ForEach([ScanLevel.rot, .gelb, .gruen], id: \.self) { level in
                Circle()
                    .fill(lampColor(level).opacity(camera.check?.level == level ? 1 : 0.2))
                    .frame(width: 26, height: 26)
            }
        }
        .padding(8)
        .background(.black.opacity(0.6), in: Capsule())
    }

    private func lampColor(_ level: ScanLevel) -> Color {
        switch level {
        case .gruen: return .green
        case .gelb: return .yellow
        case .rot: return .red
        }
    }

    @ViewBuilder private var panel: some View {
        switch camera.phase {
        case .starting:
            ProgressView("Kamera startet …")
                .tint(.white)
        case .denied:
            VStack(spacing: 10) {
                Text("FitScan darf die Kamera nicht verwenden. Du kannst das in den Einstellungen unter FitScan erlauben.")
                    .multilineTextAlignment(.center)
                Button("Einstellungen öffnen") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                }
                .buttonStyle(.borderedProminent)
            }
        case .failed(let message):
            Text(message).multilineTextAlignment(.center)
        case .setup:
            VStack(spacing: 6) {
                Text(camera.check?.level == .gruen ? "Gut so, gleich geht es los." : "Stell dich so hin, dass die Ampel grün wird.")
                    .font(.headline)
                Text("Ganzer Körper im Bild, Arme leicht abgespreizt, Füße etwa schulterbreit.")
                    .font(.callout)
                    .multilineTextAlignment(.center)
            }
        case .countdown:
            Text("Stillhalten, gleich langsam einmal um dich selbst drehen.")
                .font(.headline)
                .multilineTextAlignment(.center)
        case .turning:
            VStack(spacing: 10) {
                Text("Langsam drehen")
                    .font(.headline)
                HStack(spacing: 8) {
                    ForEach(Array(TurnTracker.steps.enumerated()), id: \.offset) { index, view in
                        VStack(spacing: 4) {
                            Image(systemName: index < camera.progress ? "checkmark.circle.fill" : "circle")
                                .font(.title3)
                                .foregroundStyle(index < camera.progress ? Color.green : Color.white.opacity(0.6))
                            Text(Self.label(view))
                                .font(.caption2)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                HStack {
                    Text("\(camera.captured) Bilder")
                        .font(.callout)
                        .monospacedDigit()
                    Spacer()
                    Button("Fertig") { camera.finishNow() }
                        .buttonStyle(.borderedProminent)
                        .disabled(camera.captured == 0)
                }
            }
        case .done:
            Label("Fertig, die Bilder werden geprüft.", systemImage: "checkmark.circle.fill")
                .font(.headline)
        }
    }

    private static func label(_ view: BodyView) -> String {
        switch view {
        case .vorn: return "vorn"
        case .seite: return "Seite"
        case .ruecken: return "Rücken"
        }
    }
}

/// Kamerabild über `AVCaptureVideoPreviewLayer`, ganzes Bild ohne Zuschnitt, damit man sieht,
/// was die Prüfung sieht.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Kamera eingerichtet; danach gibt es die Verbindung zur Vorschau
    var ready: Bool

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        func upright() {
            if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            upright()
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.videoGravity = .resizeAspect
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        if ready { view.upright() }
    }
}
