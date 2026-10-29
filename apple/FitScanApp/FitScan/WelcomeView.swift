import SwiftUI

/// Begrüßung beim ersten Start: was die App macht, dass alles auf dem iPhone bleibt, wie es losgeht.
struct WelcomeView: View {
    var onDone: () -> Void
    @State private var page = 0

    private struct Page {
        let icon: String
        let title: String
        let text: String
    }

    private let pages = [
        Page(icon: "figure.stand", title: "Willkommen bei FitScan",
             text: "Aus deinen Maßen oder aus Fotos und Videos entsteht ein 3D-Modell deines Körpers im Maßstab 1:1. Mit Skelett, bereit für Blender und Cascadeur."),
        Page(icon: "lock.iphone", title: "Alles bleibt auf deinem iPhone",
             text: "FitScan rechnet ohne Internet. Deine Maße, Bilder und Modelle verlassen das Gerät nur, wenn du sie selbst teilst."),
        Page(icon: "ruler", title: "So geht es los",
             text: "Tippe auf „Neuer Scan“. Wähle ein Drehvideo oder einige Fotos von dir oder gib nur deine Maße ein. Größe und Gewicht sind Pflicht, Maßband-Werte machen das Modell genauer. Danach kannst du die Form mit Reglern nachbearbeiten."),
    ]

    var body: some View {
        VStack(spacing: 24) {
            TabView(selection: $page) {
                ForEach(pages.indices, id: \.self) { i in
                    VStack(spacing: 20) {
                        Spacer()
                        Image(systemName: pages[i].icon)
                            .font(.system(size: 72, weight: .light))
                            .foregroundStyle(Color.accentColor)
                        Text(pages[i].title)
                            .font(.title.weight(.bold))
                            .multilineTextAlignment(.center)
                        Text(pages[i].text)
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 28)
                        Spacer()
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .indexViewStyle(.page(backgroundDisplayMode: .always))

            Button {
                if page < pages.count - 1 {
                    withAnimation { page += 1 }
                } else {
                    onDone()
                }
            } label: {
                Text(page < pages.count - 1 ? "Weiter" : "Los geht's")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)
            .padding(.bottom)
        }
        .background(
            LinearGradient(colors: [Color.accentColor.opacity(0.2), Color(.systemBackground)],
                           startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
        )
    }
}
