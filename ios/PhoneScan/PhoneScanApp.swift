import SwiftUI

@main
struct PhoneScanApp: App {
    @StateObject private var capture = CaptureManager()
    @StateObject private var library = CaptureLibrary()
    @StateObject private var robot = RobotLink()

    init() {
        CaptureSettings.registerDefaults()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(capture)
                .environmentObject(library)
                .environmentObject(robot)
                .preferredColorScheme(.light)
                .tint(Theme.ink)
        }
    }
}

/// Home · Scans · Robot · Settings, with a plain hairline tab bar.
struct RootView: View {
    enum Tab: String, CaseIterable {
        case home = "Home", scans = "Scans", robot = "Robot", settings = "Settings"
        var symbol: String {
            switch self {
            case .home: return "house"
            case .scans: return "map"
            case .robot: return "figure.stand"
            case .settings: return "gearshape"
            }
        }
        var selectedSymbol: String {
            switch self {
            case .home: return "house.fill"
            case .scans: return "map.fill"
            case .robot: return "figure.stand"
            case .settings: return "gearshape.fill"
            }
        }
    }

    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.didOnboard) private var didOnboard = false
    @State private var tab: Tab = .home
    @State private var showOnboarding = false

    var body: some View {
        VStack(spacing: 0) {
            Group {
                switch tab {
                case .home: HomeView(openRobotTab: { tab = .robot })
                case .scans: LibraryView()
                case .robot: RobotView()
                case .settings: SettingsView(showOnboarding: $showOnboarding)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            tabBar
        }
        .background(Theme.paper.ignoresSafeArea())
        .onAppear { if !didOnboard { showOnboarding = true } }
        .fullScreenCover(isPresented: $showOnboarding) {
            OnboardingView(hasLiDAR: capture.hasLiDAR) {
                didOnboard = true
                showOnboarding = false
            }
        }
        .alert("Robot", isPresented: Binding(get: { robot.error != nil }, set: { if !$0 { robot.error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(robot.error ?? "") }
    }

    /// Icons only; the selected tab gets a filled icon and a short bar above it.
    private var tabBar: some View {
        HStack {
            ForEach(Tab.allCases, id: \.self) { t in
                Button { tab = t } label: {
                    VStack(spacing: 8) {
                        Rectangle().fill(tab == t ? Theme.ink : .clear).frame(width: 44, height: 2)
                        Image(systemName: tab == t ? t.selectedSymbol : t.symbol)
                            .font(.system(size: 19, weight: .regular))
                            .frame(height: 24)
                    }
                    .foregroundStyle(tab == t ? Theme.ink : Theme.ink2)
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(t.rawValue)
                .accessibilityAddTraits(tab == t ? .isSelected : [])
            }
        }
        .padding(.bottom, 4)
        .background(Theme.card.ignoresSafeArea(edges: .bottom))
        .overlay(alignment: .top) { Rectangle().fill(Theme.line).frame(height: 1) }
    }
}
