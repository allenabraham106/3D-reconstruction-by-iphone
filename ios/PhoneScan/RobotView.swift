import SwiftUI

/// Robot tab: connection, maps on the robot, camera and ROS status, localization.
struct RobotView: View {
    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"
    @FocusState private var addressFocused: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                connection
                if robot.isConnected {
                    maps
                    details
                    if let pose = robot.lastResult { ResultRow(pose: pose) }
                } else {
                    setup
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 12)
            .padding(.bottom, 24)
        }
        .background(Theme.paper)
        .scrollDismissesKeyboard(.interactively)
        .task {
            await robot.refresh(robotURL)
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                if robot.isConnected { await robot.refresh(robotURL, quiet: true) }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(robot.status?.name ?? "Pioneer").monoLabel(11, color: Theme.ink2)
            Text("Robot").font(Theme.display(30)).foregroundStyle(Theme.ink)
        }
    }

    private var connection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(text: "Connection")
            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    Circle().fill(dot).frame(width: 7, height: 7)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title).font(.subheadline).foregroundStyle(Theme.ink)
                        if case .failed(let why) = robot.connection {
                            Text(why).font(.caption).foregroundStyle(Theme.ink2).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Spacer()
                    if robot.connection == .connecting { ProgressView() }
                }
                .padding(14)
                Rectangle().fill(Theme.line).frame(height: 1)
                HStack(spacing: 8) {
                    TextField("http://pioneer.local:8770", text: $robotURL)
                        .font(Theme.mono(13))
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .focused($addressFocused)
                        .submitLabel(.go)
                        .onSubmit { Task { await robot.refresh(robotURL) } }
                    Button {
                        addressFocused = false
                        Task { await robot.refresh(robotURL) }
                    } label: {
                        Text(robot.isConnected ? "Reconnect" : "Connect").monoLabel(10, color: Theme.card)
                            .padding(.horizontal, 10).padding(.vertical, 7)
                            .background(Theme.ink, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
                .padding(14)
            }
            .paperCard()
        }
    }

    private var maps: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(text: "Maps on the robot")
            VStack(spacing: 0) {
                let list = robot.status?.maps ?? []
                if list.isEmpty {
                    Text("None yet. Scan a room, then tap Send scan on Home.")
                        .font(.footnote).foregroundStyle(Theme.ink2)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(14)
                }
                ForEach(Array(list.enumerated()), id: \.element) { i, m in
                    Button { Task { await robot.activate(robotURL, map: m) } } label: {
                        HStack {
                            Image(systemName: "map").font(.system(size: 14, weight: .light))
                            Text(m).font(.subheadline)
                            Spacer()
                            if m == robot.status?.map { Chip(text: "Active", style: .dark) }
                        }
                        .foregroundStyle(Theme.ink)
                        .padding(14)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if i < list.count - 1 { Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 14) }
                }
            }
            .paperCard()
        }
    }

    private var details: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(text: "Robot")
            VStack(spacing: 0) {
                row("Model", robot.status?.robot.map { "\($0.name ?? "Custom") · \($0.joints ?? 0) joints" } ?? "Pioneer (built in)")
                Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 14)
                row("Camera", robot.status?.camera ?? "—")
                Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 14)
                row("ROS", robot.status?.ros == true ? "Publishing /initialpose" : "Off")
                Rectangle().fill(Theme.line).frame(height: 1).padding(.leading, 14)
                row("Map frame", "metres · Z up · floor at 0")
            }
            .paperCard()
            Text(robot.status?.ros == true
                 ? "Each robot fix is sent to ROS as /initialpose, so the robot's own SLAM and nav2 start in this map."
                 : "Start the robot link with --ros to hand each fix to the robot's SLAM and nav2 as /initialpose.")
                .font(.caption).foregroundStyle(Theme.ink2)
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).font(.subheadline).foregroundStyle(Theme.ink)
            Spacer()
            Text(v).font(Theme.mono(12)).foregroundStyle(Theme.ink2).lineLimit(1)
        }
        .padding(14)
    }

    private var setup: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(text: "Set up")
            VStack(alignment: .leading, spacing: 10) {
                Text("On the robot's computer, in this repo:").font(.footnote).foregroundStyle(Theme.ink2)
                Text("python -m robot_link.server --camera realsense --ros")
                    .font(Theme.mono(12))
                    .foregroundStyle(Theme.ink)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .paperCard(8, fill: Theme.paper)
                    .textSelection(.enabled)
                Text("Then enter its address above. The phone and robot need the same Wi-Fi or Tailscale.")
                    .font(.footnote).foregroundStyle(Theme.ink2)
            }
            .padding(14)
            .paperCard()
        }
    }

    private var dot: Color {
        switch robot.connection {
        case .connected: return Theme.good
        case .failed: return Theme.warning
        default: return Theme.ink2
        }
    }

    private var title: String {
        switch robot.connection {
        case .connected: return "Connected to \(robot.status?.name ?? "robot")"
        case .connecting: return "Connecting…"
        case .unknown: return "Not connected"
        case .failed: return "Can't reach the robot"
        }
    }
}

/// The robot's map image with robot and phone poses drawn on top (grid-aligned).
struct FloorMap: View {
    let image: UIImage
    let grid: RobotLink.Grid
    let robot: RobotLink.Pose?
    let phone: RobotLink.Pose?

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .interpolation(.medium)
            .aspectRatio(CGFloat(grid.width) / CGFloat(grid.height), contentMode: .fit)
            .overlay {
                GeometryReader { geo in
                    ZStack {
                        if let phone { marker(phone, Theme.phone, label: "Phone", in: geo.size) }
                        if let robot { marker(robot, Theme.accent, label: "Pioneer", in: geo.size) }
                    }
                }
            }
            .accessibilityLabel("Map with robot and phone positions")
    }

    @ViewBuilder
    private func marker(_ pose: RobotLink.Pose, _ color: Color, label: String, in size: CGSize) -> some View {
        if let px = pose.grid_px, px.count == 2 {
            let x = (px[0] + 0.5) / Double(grid.width) * size.width
            let y = (px[1] + 0.5) / Double(grid.height) * size.height
            ZStack {
                // heading wedge: map yaw is CCW from +x; screen y points down
                Wedge(angle: .degrees(-pose.yaw_deg)).fill(color.opacity(0.28)).frame(width: 56, height: 56)
                Circle().fill(Theme.card).frame(width: 15, height: 15)
                Circle().fill(color).frame(width: 9, height: 9)
                Text(label).monoLabel(8, color: Theme.ink)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(Theme.card, in: RoundedRectangle(cornerRadius: 3))
                    .offset(y: 17)
            }
            .position(x: x, y: y)
            .animation(.spring(duration: 0.4), value: pose)
        }
    }
}

struct Wedge: Shape {
    let angle: Angle
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        p.move(to: c)
        p.addArc(center: c, radius: rect.width / 2, startAngle: angle - .degrees(28), endAngle: angle + .degrees(28), clockwise: false)
        p.closeSubpath()
        return p
    }
}

struct ResultRow: View {
    let pose: RobotLink.Pose
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(text: "Last fix")
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(pose.source == "phone" ? "Phone" : "Pioneer").font(.subheadline).foregroundStyle(Theme.ink)
                    Spacer()
                    Chip(text: pose.ok ? "Found" : "Not found", style: pose.ok ? .dark : .tint(Theme.warning))
                }
                if pose.ok {
                    Text(String(format: "x %.2f m  y %.2f m  %.0f°", pose.x, pose.y, pose.yaw_deg))
                        .font(Theme.mono(13)).foregroundStyle(Theme.ink)
                    Text("\(pose.inliers) of \(pose.matches) matches agree").font(.caption).foregroundStyle(Theme.ink2)
                } else {
                    Text(pose.reason).font(.footnote).foregroundStyle(Theme.ink2)
                }
            }
            .padding(14)
            .paperCard()
        }
    }
}
