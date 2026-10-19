import SwiftUI

/// Scan-Anleitung: dieselben Regeln, die der Scan-Assistent am PC prüft (`quality.CHECKLIST`).
struct ScanGuideView: View {
    struct Tip: Identifiable {
        let id = UUID()
        let icon: String
        let title: String
        let text: String
    }

    struct Step: Identifiable {
        let id = UUID()
        let title: String
        let tips: [Tip]
    }

    static let steps: [Step] = [
        Step(title: "Vorbereitung", tips: [
            Tip(icon: "tshirt", title: "Eng anliegende Kleidung",
                text: "Am besten Shorts und ein enges Oberteil oder oben ohne. Weite Kleidung macht das Modell dicker."),
            Tip(icon: "shoeprints.fill", title: "Barfuß",
                text: "Socken oder Schuhe verdecken die Zehen und verfälschen die Füße."),
            Tip(icon: "applewatch", title: "Ohne Uhr und Schmuck",
                text: "Lange Haare zusammenbinden, damit Hals und Schultern frei sind."),
        ]),
        Step(title: "Ort und Licht", tips: [
            Tip(icon: "sun.max", title: "Helles, weiches Licht von vorn",
                text: "Tageslicht oder eine weiße Lampe hinter dem Handy. Keine Lampe direkt über oder hinter dir, kein farbiges Licht."),
            Tip(icon: "rectangle.portrait", title: "Ruhiger Hintergrund",
                text: "Eine einfarbige Wand, die sich von Haut und Kleidung abhebt. Nur du bist im Bild."),
        ]),
        Step(title: "Handy", tips: [
            Tip(icon: "iphone", title: "Hochkant auf Hüfthöhe",
                text: "Am besten auf einem Stativ oder angelehnt, etwa 2 m entfernt."),
            Tip(icon: "viewfinder", title: "Ganzer Körper im Bild",
                text: "Mit etwas Rand über dem Kopf und unter den Füßen."),
        ]),
        Step(title: "Haltung und Drehung", tips: [
            Tip(icon: "figure.arms.open", title: "A-Pose",
                text: "Arme gestreckt, etwa 30° vom Körper, Füße schulterbreit, Blick geradeaus."),
            Tip(icon: "arrow.triangle.2.circlepath", title: "Einmal langsam drehen",
                text: "Auf der Stelle einmal ganz herum, etwa 20 Sekunden für eine Runde. Arme dabei ruhig halten."),
        ]),
        Step(title: "Maßband (macht das Modell genauer)", tips: [
            Tip(icon: "ruler", title: "Größe und Gewicht",
                text: "Ohne Schuhe messen, morgens wiegen."),
            Tip(icon: "circle.dashed", title: "Umfänge",
                text: "Brust unter den Achseln, Taille an der schmalsten Stelle, Hüfte an der breitesten Stelle des Gesäßes. Oberschenkel, Oberarm und Wade links an der dicksten Stelle, entspannt."),
        ]),
    ]

    var body: some View {
        List {
            Section {
                Text("So wird dein Modell möglichst genau. Der Kamera-Scan in einem der nächsten Builds prüft diese Punkte live und zeigt dir eine Ampel.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            ForEach(Array(Self.steps.enumerated()), id: \.element.id) { index, step in
                Section("\(index + 1). \(step.title)") {
                    ForEach(step.tips) { tip in
                        HStack(alignment: .top, spacing: 14) {
                            Image(systemName: tip.icon)
                                .font(.title3)
                                .foregroundStyle(Color.accentColor)
                                .frame(width: 30)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(tip.title).font(.headline)
                                Text(tip.text).font(.callout).foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .navigationTitle("Scan-Anleitung")
    }
}
