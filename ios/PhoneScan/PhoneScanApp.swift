import SwiftUI

@main
struct PhoneScanApp: App {
    @StateObject private var capture = CaptureManager()
    @StateObject private var library = CaptureLibrary()

    init() {
        CaptureSettings.registerDefaults()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(capture)
                .environmentObject(library)
                .preferredColorScheme(.dark)
                .tint(Theme.primaryBright)
        }
    }
}
