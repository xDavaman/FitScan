import SwiftUI

@main
struct FitScanApp: App {
    @StateObject private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(app)
                .task { await app.start() }
        }
    }
}
