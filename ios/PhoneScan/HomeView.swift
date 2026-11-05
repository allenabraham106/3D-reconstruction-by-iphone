import SwiftUI

/// Home tab: the robot's 3D map as the hero, round map buttons on the right, and a status sheet
/// whose headline and two big tiles change with what Pioneer is doing.
struct HomeView: View {
    var openRobotTab: () -> Void = {}

    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"

    @State private var showCapture = false
    @State private var showSendPicker = false
    @State private var command: MapCommand?
    @State private var topView = false

    var body: some View {
        ZStack(alignment: .bottom) {
            map.ignoresSafeArea(edges: .top)
            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    Spacer()
                    buttons
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                Spacer(minLength: 0)
                sheet
            }
        }
        .background(Theme.paper)
        .fullScreenCover(isPresented: $showCapture) { CaptureView() }
        .confirmationDialog("Send a scan to Pioneer", isPresented: $showSendPicker, titleVisibility: .visible) {
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
        .task {
            library.refresh()
            await robot.refresh(robotURL, quiet: true)
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                await robot.refresh(robotURL, quiet: true)
            }
        }
    }

    // MARK: Map

    @ViewBuilder
    private var map: some View {
        ZStack {
            MapBackdrop()
            if let cloud = robot.cloud, !cloud.points.isEmpty {
                Map3DView(cloud: cloud, path: robot.scanPath,
                          robot: robot.status?.last["robot"], phone: robot.status?.last["phone"],
                          command: command, robotModel: robot.robotModel)
                    .padding(.bottom, 200)
            } else if let grid = robot.status?.grid, let img = robot.topdownImage ?? robot.mapImage {
                FloorMap(image: img, grid: grid, robot: robot.status?.last["robot"], phone: robot.status?.last["phone"])
                    .padding(.horizontal, 28)
                    .padding(.bottom, 260)
                    .padding(.top, 60)
            } else if robot.loadingMap {
                ProgressView().padding(.bottom, 220)
            } else {
                emptyMap.padding(.bottom, 240)
            }
        }
    }

    private var emptyMap: some View {
        VStack(spacing: 8) {
            Image(systemName: "cube.transparent").font(.system(size: 30, weight: .ultraLight)).foregroundStyle(Theme.ink2)
            Text(robot.isConnected ? "No map yet" : "Pioneer is offline").monoLabel(11, color: Theme.ink2)
        }
    }

    private var buttons: some View {
        VStack(spacing: 10) {
            RoundButton(symbol: "location", label: "Find Pioneer") { command = MapCommand(kind: .locateRobot) }
            RoundButton(symbol: "square.3.layers.3d", label: topView ? "3D view" : "Top view", active: topView) {
                topView.toggle()
                command = MapCommand(kind: topView ? .topDown : .recenter)
            }
            RoundButton(symbol: "arrow.up.and.down.and.arrow.left.and.right", label: "Recenter") {
                topView = false
                command = MapCommand(kind: .recenter)
            }
            RoundButton(symbol: "airplane", label: "Fly through the scan") { command = MapCommand(kind: .flyThrough) }
                .disabled(robot.scanPath.count < 3)
            RoundButton(symbol: "video", label: "Localize this phone") { localizePhone() }
                .disabled(!canLocalize)
        }
        .opacity(robot.cloud == nil ? 0.0 : 1)
        .animation(.easeInOut(duration: 0.2), value: robot.cloud == nil)
    }

    // MARK: Sheet

    private enum HomeState { case offline, noMap, ready, localizing, localized, building, failed }

    private var state: HomeState {
        guard robot.isConnected, let st = robot.status else { return .offline }
        if ["unzipping", "reconstructing", "building relocalization map"].contains(st.job.state) { return .building }
        if robot.busy?.hasPrefix("Localizing") == true { return .localizing }
        if st.map == nil { return .noMap }
        if let r = st.last["robot"], r.ok { return .localized }
        if let p = robot.lastResult, !p.ok { return .failed }
        return .ready
    }

    private var sheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Theme.line).frame(width: 36, height: 4).frame(maxWidth: .infinity).padding(.bottom, 14)
            HStack(spacing: 6) { chips }
            Text(headline).font(Theme.display(22)).foregroundStyle(Theme.ink).padding(.top, 10)
            Text(subline).font(.subheadline).foregroundStyle(Theme.ink2).padding(.top, 2)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) { tiles }.padding(.top, 16)
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 14)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 22, topTrailingRadius: 22, style: .continuous)
                .fill(Theme.card)
                .shadow(color: .black.opacity(0.06), radius: 12, y: -2)
        )
    }

    @ViewBuilder
    private var chips: some View {
        switch state {
        case .offline:
            Chip(text: "Offline", symbol: "wifi.slash")
        default:
            Chip(text: "Connected", symbol: "wifi", style: .solid(Theme.chipGreen))
            switch state {
            case .building: Chip(text: "Building map", symbol: "cube.transparent", style: .solid(Theme.accent))
            case .localizing: Chip(text: "Localizing", symbol: "scope", style: .solid(Theme.accent))
            case .localized: Chip(text: "Localized", symbol: "location.fill", style: .solid(Theme.accent))
            case .ready: Chip(text: "Map ready", symbol: "map", style: .solid(Theme.accent))
            case .failed: Chip(text: "Not found", symbol: "location.slash", style: .solid(Theme.warning))
            case .noMap: Chip(text: "No map", symbol: "map")
            case .offline: EmptyView()
            }
            if robot.status?.ros == true { Chip(text: "ROS") }
        }
        Spacer(minLength: 0)
        if !capture.hasLiDAR { Chip(text: "Photo mode", symbol: "camera") }
    }

    private var headline: String {
        let name = robot.status?.name ?? "Pioneer"
        switch state {
        case .offline: return "\(name) is offline"
        case .noMap: return "\(name) needs a map"
        case .ready: return "\(name) is ready"
        case .localizing: return "Finding \(name)…"
        case .localized: return "\(name) knows where it is"
        case .building: return "\(name) is building the map"
        case .failed: return "\(name) couldn't find itself"
        }
    }

    private var subline: String {
        switch state {
        case .offline: return "Start robot_link on the robot, then connect in the Robot tab."
        case .noMap: return "Scan a room with this phone, then send the scan to Pioneer."
        case .ready: return "Localize to find where Pioneer is in the map."
        case .localizing: return "Matching what its camera sees against the map."
        case .localized:
            guard let r = robot.status?.last["robot"] else { return "" }
            return String(format: "x %.2f m · y %.2f m · facing %.0f° · %d matches", r.x, r.y, r.yaw_deg, r.inliers)
        case .building: return (robot.status?.job.state ?? "working").capitalized + "… This takes a few minutes."
        case .failed: return robot.lastResult?.reason ?? "Try again facing a part of the room that was scanned."
        }
    }

    @ViewBuilder
    private var tiles: some View {
        switch state {
        case .offline:
            Tile(title: "Scan room", symbol: "viewfinder") { showCapture = true }
            Tile(title: "Connect", symbol: "wifi") { openRobotTab() }
        case .noMap, .building:
            Tile(title: "Scan room", symbol: "viewfinder") { showCapture = true }
            Tile(title: "Send scan", symbol: "square.and.arrow.up", busy: robot.busy?.hasPrefix("Sending") == true || state == .building) {
                showSendPicker = true
            }
            .disabled(library.items.isEmpty || state == .building)
        case .ready, .localizing, .localized, .failed:
            Tile(title: state == .localized ? "Localize again" : "Localize", symbol: "scope", busy: state == .localizing) {
                Task { await robot.localizeRobot(robotURL) }
            }
            .disabled(!canLocalize)
            Tile(title: "Scan room", symbol: "viewfinder") { showCapture = true }
        }
    }

    private var canLocalize: Bool { robot.isConnected && robot.status?.map != nil && robot.busy == nil }

    /// The camera only runs while scanning, so start it briefly, take one frame, and stop it again.
    private func localizePhone() {
        Task {
            capture.startSession()
            try? await Task.sleep(for: .seconds(1.2))
            let snap = capture.snapshotForLocalization()
            capture.pauseSession()
            guard let snap else {
                robot.error = "Couldn't get a camera frame. Check camera access in Settings."
                return
            }
            await robot.localizePhone(robotURL, snapshot: snap)
        }
    }
}
