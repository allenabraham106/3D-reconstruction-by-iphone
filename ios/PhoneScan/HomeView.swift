import SwiftUI
import ARKit

/// What Pioneer is doing, shared by Home and the Map screen.
enum PioneerState { case offline, noMap, ready, localizing, localized, building, failed }

extension RobotLink {
    var pioneerState: PioneerState {
        guard isConnected, let st = status else { return .offline }
        if ["unzipping", "reconstructing", "building relocalization map"].contains(st.job.state) { return .building }
        if busy?.hasPrefix("Localizing") == true { return .localizing }
        if st.map == nil { return .noMap }
        if let r = st.last["robot"], r.ok { return .localized }
        if let p = lastResult, !p.ok { return .failed }
        return .ready
    }

    var pioneerName: String { status?.name ?? "Pioneer" }

    /// Short status under the robot's name, e.g. "Localized · living-room".
    var statusText: String {
        let map = status?.map
        switch pioneerState {
        case .offline: return "Offline"
        case .noMap: return "Connected · no map yet"
        case .ready: return "Map ready" + (map.map { " · \($0)" } ?? "")
        case .localizing: return "Localizing…"
        case .localized: return "Localized" + (map.map { " · \($0)" } ?? "")
        case .building: return "Building the map…"
        case .failed: return "Couldn't find itself"
        }
    }

    var statusColor: Color {
        switch pioneerState {
        case .offline: return Theme.ink3
        case .failed: return Theme.warning
        case .building, .localizing: return Theme.accent
        default: return Theme.good
        }
    }

    var headline: String {
        let name = pioneerName
        switch pioneerState {
        case .offline: return "\(name) is offline"
        case .noMap: return "\(name) needs a map"
        case .ready: return "\(name) is ready"
        case .localizing: return "Finding \(name)…"
        case .localized: return "\(name) knows where it is"
        case .building: return "\(name) is building the map"
        case .failed: return "\(name) couldn't find itself"
        }
    }

    var subline: String {
        switch pioneerState {
        case .offline: return "Start robot_link on the robot, then connect in the Robot tab."
        case .noMap: return "Scan a room with this phone, then send the scan to Pioneer."
        case .ready: return "Localize to find where Pioneer is in the map."
        case .localizing: return "Matching what its camera sees against the map."
        case .localized:
            guard let r = status?.last["robot"] else { return "" }
            return String(format: "x %.2f m · y %.2f m · facing %.0f° · %d matches", r.x, r.y, r.yaw_deg, r.inliers)
        case .building: return (status?.job.state ?? "working").capitalized + "… This takes a few minutes."
        case .failed: return lastResult?.reason ?? "Try again facing a part of the room that was scanned."
        }
    }

    var canLocalize: Bool { isConnected && status?.map != nil && busy == nil }
}

/// Home tab: the robot as the hero, its status, four shortcuts and a short list.
struct HomeView: View {
    var openRobotTab: () -> Void = {}
    var openScansTab: () -> Void = {}

    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"

    @State private var showCapture = false
    @State private var showSendPicker = false
    @State private var showLive = false
    @State private var showMap = false

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 24)
                .padding(.top, 12)
            Image("PioneerHero")
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.vertical, 8)
                .accessibilityLabel("Pioneer")
            VStack(spacing: 4) {
                Text(robot.headline).font(.system(size: 17, weight: .semibold)).foregroundStyle(Theme.ink)
                Text(robot.subline).font(.footnote).foregroundStyle(Theme.ink2)
                    .multilineTextAlignment(.center).lineLimit(2)
            }
            .padding(.horizontal, 28)
            .padding(.bottom, 18)
            actions.padding(.horizontal, 18)
            list.padding(.horizontal, 16).padding(.top, 20).padding(.bottom, 12)
        }
        .background(Theme.paper)
        .fullScreenCover(isPresented: $showCapture) { CaptureView() }
        .fullScreenCover(isPresented: $showLive) { LiveFeedView() }
        .fullScreenCover(isPresented: $showMap) { MapScreen() }
        .sendScanPicker(isPresented: $showSendPicker)
        .task {
            library.refresh()
            await robot.refresh(robotURL, quiet: true)
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                await robot.refresh(robotURL, quiet: true)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                Text(robot.pioneerName).font(Theme.display(32)).foregroundStyle(Theme.ink)
                StatusLine(text: robot.statusText, color: robot.statusColor)
            }
            Spacer()
            if !capture.hasLiDAR { Chip(text: "Photo mode", symbol: "camera").padding(.top, 8) }
        }
    }

    private var actions: some View {
        let state = robot.pioneerState
        return HStack(spacing: 0) {
            if state == .offline {
                QuickAction(title: "Connect", symbol: "wifi", highlighted: true) { openRobotTab() }
            } else {
                QuickAction(title: state == .localized ? "Localize again" : "Localize", symbol: "scope",
                            highlighted: robot.canLocalize, busy: state == .localizing) {
                    Task { await robot.localizeRobot(robotURL) }
                }
                .disabled(!robot.canLocalize)
            }
            QuickAction(title: "Scan room", symbol: "viewfinder") { showCapture = true }
            QuickAction(title: "Send scan", symbol: "square.and.arrow.up",
                        busy: robot.busy?.hasPrefix("Sending") == true || state == .building) { showSendPicker = true }
                .disabled(library.items.isEmpty || !robot.isConnected || state == .building)
            QuickAction(title: "Map", symbol: "map") { showMap = true }
        }
    }

    private var list: some View {
        VStack(spacing: 0) {
            ListRow(symbol: "map", title: "Map", subtitle: mapSubtitle) { showMap = true }
            Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 56)
            ListRow(symbol: "video", title: "Live camera", subtitle: liveSubtitle) { showLive = true }
                .disabled(!liveAvailable)
                .opacity(liveAvailable ? 1 : 0.5)
            Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 56)
            ListRow(symbol: "viewfinder", title: "Scans", subtitle: scansSubtitle) { openScansTab() }
        }
        .paperCard(18)
    }

    private var mapSubtitle: String {
        if let m = robot.status?.map { return "\(m) · on \(robot.pioneerName)" }
        return robot.isConnected ? "No map on \(robot.pioneerName) yet" : "Connect to see Pioneer's map"
    }

    private var liveAvailable: Bool { robot.status?.live?.available == true }

    private var liveSubtitle: String {
        if liveAvailable { return "What \(robot.pioneerName)'s camera sees, live" }
        return robot.isConnected ? "No camera running on \(robot.pioneerName)" : "Connect to watch \(robot.pioneerName)'s camera"
    }

    private var scansSubtitle: String {
        let n = library.items.count
        return n == 0 ? "None yet" : "\(n) on this iPhone"
    }
}

/// Full-screen 3D map: the scan, the route, and where Pioneer and this phone are.
struct MapScreen: View {
    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @EnvironmentObject private var robot: RobotLink
    @Environment(\.dismiss) private var dismiss
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"

    enum ViewMode: Hashable { case threeD, top }
    @State private var mode: ViewMode = .threeD
    @State private var command: MapCommand?
    @State private var showCapture = false
    @State private var showSendPicker = false
    @State private var showLive = false

    var body: some View {
        ZStack(alignment: .bottom) {
            map.ignoresSafeArea(edges: .top)
            VStack(spacing: 0) {
                topBar.padding(.horizontal, 16).padding(.top, 8)
                HStack { Spacer(); buttons }.padding(.horizontal, 16).padding(.top, 12)
                Spacer(minLength: 0)
                card
            }
        }
        .background(Theme.paper)
        .onChange(of: mode) { _, m in command = MapCommand(kind: m == .top ? .topDown : .recenter) }
        .fullScreenCover(isPresented: $showCapture) { CaptureView() }
        .fullScreenCover(isPresented: $showLive) { LiveFeedView() }
        .sendScanPicker(isPresented: $showSendPicker)
    }

    private var topBar: some View {
        HStack {
            Button { dismiss() } label: {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.down").font(.system(size: 15, weight: .semibold))
                    Text("Map").font(.system(size: 17, weight: .semibold))
                }
                .foregroundStyle(Theme.ink)
                .padding(.vertical, 8).padding(.trailing, 8)
            }
            .buttonStyle(.plain)
            Spacer()
            Segmented(options: [(ViewMode.threeD, "3D"), (ViewMode.top, "Top")], selection: $mode)
                .frame(width: 130)
                .opacity(robot.cloud == nil ? 0 : 1)
        }
    }

    @ViewBuilder
    private var map: some View {
        ZStack {
            MapBackdrop()
            if let cloud = robot.cloud, !cloud.points.isEmpty {
                Map3DView(cloud: cloud, path: robot.scanPath,
                          robot: robot.status?.last["robot"], phone: robot.status?.last["phone"],
                          command: command, robotModel: robot.robotModel)
                    .padding(.bottom, 260)
            } else if let grid = robot.status?.grid, let img = robot.topdownImage ?? robot.mapImage {
                FloorMap(image: img, grid: grid, robot: robot.status?.last["robot"], phone: robot.status?.last["phone"])
                    .padding(.horizontal, 28)
                    .padding(.bottom, 300)
                    .padding(.top, 100)
            } else if robot.loadingMap {
                ProgressView().padding(.bottom, 260)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "cube.transparent").font(.system(size: 32, weight: .ultraLight)).foregroundStyle(Theme.ink3)
                    Text(robot.isConnected ? "No map yet" : "\(robot.pioneerName) is offline")
                        .font(.subheadline).foregroundStyle(Theme.ink2)
                }
                .padding(.bottom, 280)
            }
        }
    }

    private var buttons: some View {
        VStack(spacing: 10) {
            RoundButton(symbol: "location", label: "Find Pioneer") { command = MapCommand(kind: .locateRobot) }
            RoundButton(symbol: "arrow.up.and.down.and.arrow.left.and.right", label: "Recenter") {
                mode = .threeD
                command = MapCommand(kind: .recenter)
            }
            RoundButton(symbol: "airplane", label: "Fly through the scan") { command = MapCommand(kind: .flyThrough) }
                .disabled(robot.scanPath.count < 3)
            RoundButton(symbol: "video", label: "Pioneer's live camera") { showLive = true }
                .disabled(robot.status?.live?.available != true)
                .opacity(robot.status?.live?.available == true ? 1 : 0.4)
            RoundButton(symbol: "iphone", label: "Localize this phone") { localizePhone() }
                .disabled(!robot.canLocalize)
        }
        .opacity(robot.cloud == nil ? 0 : 1)
        .animation(.easeInOut(duration: 0.2), value: robot.cloud == nil)
    }

    private var card: some View {
        let state = robot.pioneerState
        return VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text(robot.status?.map ?? "No map").font(Theme.display(24)).foregroundStyle(Theme.ink)
                Spacer()
                if let r = robot.status?.last["robot"], r.ok {
                    Text("\(r.inliers) matches").font(.caption.weight(.medium)).foregroundStyle(Theme.ink3)
                }
            }
            StatusLine(text: robot.headline, color: robot.statusColor).padding(.top, 6)
            if state == .localized, let r = robot.status?.last["robot"] {
                HStack(spacing: 26) {
                    StatView(value: String(format: "%.2f", r.x), unit: "m", label: "X")
                    StatView(value: String(format: "%.2f", r.y), unit: "m", label: "Y")
                    StatView(value: String(format: "%.0f", r.yaw_deg), unit: "°", label: "Heading")
                }
                .padding(.top, 16)
            } else {
                Text(robot.subline).font(.subheadline).foregroundStyle(Theme.ink2)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 10)
            }
            HStack(spacing: 10) { buttonsRow(state) }.padding(.top, 22)
        }
        .padding(.horizontal, 24)
        .padding(.top, 24)
        .padding(.bottom, 14)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous)
                .fill(Theme.card)
                .shadow(color: .black.opacity(0.08), radius: 20, y: -4)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    @ViewBuilder
    private func buttonsRow(_ state: PioneerState) -> some View {
        switch state {
        case .offline, .noMap, .building:
            Tile(title: "Scan room", symbol: "viewfinder", prominent: true) { showCapture = true }
            Tile(title: "Send scan", symbol: "square.and.arrow.up",
                 busy: robot.busy?.hasPrefix("Sending") == true || state == .building) { showSendPicker = true }
                .disabled(library.items.isEmpty || !robot.isConnected || state == .building)
        case .ready, .localizing, .localized, .failed:
            Tile(title: state == .localized ? "Localize again" : "Localize", symbol: "scope",
                 prominent: true, busy: state == .localizing) {
                Task { await robot.localizeRobot(robotURL) }
            }
            .disabled(!robot.canLocalize)
            Tile(title: "Send scan", symbol: "square.and.arrow.up",
                 busy: robot.busy?.hasPrefix("Sending") == true) { showSendPicker = true }
                .disabled(library.items.isEmpty)
        }
    }

    /// The camera only runs while scanning, so start it briefly, take one frame, and stop it again.
    private func localizePhone() {
        Task {
            let started = ProcessInfo.processInfo.systemUptime
            capture.startSession()
            // Wait for a fresh, tracked frame (not one left over from before the last pause);
            // the first run also waits out the camera-permission prompt.
            var snap: (jpeg: Data, fx: Float, fy: Float, cx: Float, cy: Float)?
            for _ in 0..<50 {
                try? await Task.sleep(for: .milliseconds(100))
                if let f = capture.session.currentFrame, f.timestamp > started, case .normal = f.camera.trackingState {
                    snap = capture.snapshotForLocalization()
                    break
                }
            }
            capture.pauseSession()
            guard let snap else {
                robot.error = "Couldn't get a camera frame. Check camera access in Settings."
                return
            }
            await robot.localizePhone(robotURL, snapshot: snap)
        }
    }
}

/// "Send a scan to Pioneer": pick one of the recent complete scans and upload it.
private struct SendScanPicker: ViewModifier {
    @Binding var isPresented: Bool
    @EnvironmentObject private var library: CaptureLibrary
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"

    func body(content: Content) -> some View {
        content.confirmationDialog("Send a scan to Pioneer", isPresented: $isPresented, titleVisibility: .visible) {
            ForEach(Array(library.items.filter(\.isComplete).prefix(8))) { item in
                Button("\(item.date.formatted(date: .abbreviated, time: .shortened)) · \(item.frameCount) frames") {
                    Task {
                        if let zip = try? await library.makeZip(item) {
                            await robot.sendScan(robotURL, zip: zip, name: item.name)
                        }
                    }
                }
            }
        } message: {
            Text("Pioneer rebuilds it in 3D and makes a map it can find itself in. Takes a few minutes.")
        }
    }
}

extension View {
    func sendScanPicker(isPresented: Binding<Bool>) -> some View { modifier(SendScanPicker(isPresented: isPresented)) }
}
