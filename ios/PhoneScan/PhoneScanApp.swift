import SwiftUI

@main
struct PhoneScanApp: App {
    @StateObject private var capture = CaptureManager()
    @StateObject private var library = CaptureLibrary()
    @StateObject private var robot = RobotLink()
    @AppStorage(CaptureSettings.Key.appearance) private var appearance = Appearance.system.rawValue

    init() {
        CaptureSettings.registerDefaults()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(capture)
                .environmentObject(library)
                .environmentObject(robot)
                .preferredColorScheme(Appearance(rawValue: appearance)?.colorScheme)
                .tint(Theme.ink)
        }
    }
}

/// Home · Scans · Robot · Settings, with a plain tab bar (icon + label).
struct RootView: View {
    enum Tab: String, CaseIterable {
        case home = "Home", scans = "Scans", robot = "Robot", settings = "Settings"
        var symbol: String {
            switch self {
            case .home: return "house"
            case .scans: return "viewfinder"
            case .robot: return "figure.stand"
            case .settings: return "gearshape"
            }
        }
        var selectedSymbol: String {
            switch self {
            case .home: return "house.fill"
            case .scans: return "viewfinder"
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
                case .home: HomeView(openRobotTab: { tab = .robot }, openScansTab: { tab = .scans })
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

    /// Icon and label; the selected tab is ink, the rest are quiet.
    private var tabBar: some View {
        HStack {
            ForEach(Tab.allCases, id: \.self) { t in
                Button { tab = t } label: {
                    VStack(spacing: 4) {
                        Image(systemName: tab == t ? t.selectedSymbol : t.symbol)
                            .font(.system(size: 19, weight: .regular))
                            .frame(height: 24)
                        Text(t.rawValue).font(.system(size: 10.5, weight: .medium))
                    }
                    .foregroundStyle(tab == t ? Theme.ink : Theme.ink3)
                    .frame(maxWidth: .infinity)
                    .padding(.top, 8)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(t.rawValue)
                .accessibilityAddTraits(tab == t ? .isSelected : [])
            }
        }
        .padding(.bottom, 2)
        .background(Theme.paper.ignoresSafeArea(edges: .bottom))
        .overlay(alignment: .top) { Rectangle().fill(Theme.line).frame(height: 1) }
    }
}
