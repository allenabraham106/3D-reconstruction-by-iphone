import SwiftUI
import simd

struct ContentView: View {
    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage(CaptureSettings.Key.showMesh) private var showMesh = true
    @AppStorage(CaptureSettings.Key.showFeaturePoints) private var showFeaturePoints = true
    @AppStorage(CaptureSettings.Key.didOnboard) private var didOnboard = false
    @AppStorage(CaptureSettings.Key.hidePhotoModeCard) private var hidePhotoModeCard = false

    @State private var showLibrary = false
    @State private var showSettings = false
    @State private var showOnboarding = false
    @State private var flash = false

    var body: some View {
        ZStack {
            ARViewContainer(session: capture.session,
                            showMesh: showMesh && capture.supportsMesh,
                            showFeaturePoints: showFeaturePoints && !(showMesh && capture.supportsMesh))
                .ignoresSafeArea()

            // Shutter flash on every saved keyframe.
            Rectangle()
                .strokeBorder(Color.white.opacity(flash ? 0.9 : 0), lineWidth: 6)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: 10) {
                topBar
                if let message = capture.message {
                    MessageCard(text: message) { capture.message = nil }
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                if !capture.hasLiDAR && !hidePhotoModeCard && !capture.isRecording {
                    PhotoModeCard { withAnimation { hidePhotoModeCard = true } }
                        .transition(.opacity)
                }
                Spacer()
                if capture.isRecording {
                    GuidanceCard(guidance: capture.guidance)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                    StatsHUD()
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                controls
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
            .animation(.spring(duration: 0.35), value: capture.isRecording)
            .animation(.spring(duration: 0.35), value: capture.message)
        }
        .onAppear {
            capture.startSession()
            if !didOnboard { showOnboarding = true }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: capture.startSession()
            case .background: capture.pauseSession()
            default: break
            }
        }
        .onChange(of: capture.keyframeTick) { _, _ in
            flash = true
            withAnimation(.easeOut(duration: 0.25)) { flash = false }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: capture.isRecording)
        .sensoryFeedback(.warning, trigger: capture.warningTick)
        .sheet(isPresented: $showLibrary) { LibraryView() }
        .sheet(isPresented: $showSettings, onDismiss: {
            if !capture.isRecording { capture.startSession() }
        }) {
            SettingsView(showOnboarding: $showOnboarding)
        }
        .fullScreenCover(isPresented: $showOnboarding) {
            OnboardingView(hasLiDAR: capture.hasLiDAR) {
                didOnboard = true
                showOnboarding = false
            }
        }
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(alignment: .top, spacing: 10) {
            HStack(spacing: 8) {
                Circle()
                    .fill(capture.trackingOK ? Theme.good : Theme.warning)
                    .frame(width: 9, height: 9)
                    .shadow(color: capture.trackingOK ? Theme.good : Theme.warning, radius: 4)
                Text(capture.trackingText)
                    .font(.footnote.weight(.bold))
                    .watoLabel()
                    .lineLimit(1)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .watoGlass(Capsule())

            Spacer(minLength: 0)

            if capture.isRecording {
                MiniMap(path: capture.path, current: capture.current, heading: capture.heading)
                    .frame(width: 112, height: 112)
                    .transition(.scale.combined(with: .opacity))
            } else {
                Label(capture.hasLiDAR ? "LiDAR" : "Photo mode",
                      systemImage: capture.hasLiDAR ? "dot.radiowaves.left.and.right" : "camera.aperture")
                    .font(.caption.weight(.bold))
                    .watoLabel()
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .watoGlass(Capsule())
            }
        }
    }

    // MARK: Controls

    private var controls: some View {
        HStack {
            CircleButton(symbol: "square.stack.3d.up", label: "Captures") { showLibrary = true }
                .opacity(capture.isRecording ? 0 : 1)
                .disabled(capture.isRecording)

            Spacer()

            RecordButton(isRecording: capture.isRecording, isFinishing: capture.isFinishing,
                         ready: capture.trackingOK, action: toggleRecording)

            Spacer()

            CircleButton(symbol: "slider.horizontal.3", label: "Settings") { showSettings = true }
                .opacity(capture.isRecording ? 0 : 1)
                .disabled(capture.isRecording)
        }
        .padding(.horizontal, 8)
    }

    private func toggleRecording() {
        if capture.isRecording {
            capture.stopRecording { _ in library.refresh() }
        } else {
            capture.startRecording()
        }
    }
}

// MARK: - Pieces

struct CircleButton: View {
    let symbol: String
    let label: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.title3.weight(.semibold))
                .frame(width: 54, height: 54)
                .watoGlass(Circle())
        }
        .foregroundStyle(.white)
        .accessibilityLabel(label)
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let isFinishing: Bool
    let ready: Bool
    let action: () -> Void
    @State private var pulse = false

    var body: some View {
        Button(action: action) {
            ZStack {
                if isRecording {
                    Circle()
                        .stroke(Theme.teal.opacity(0.6), lineWidth: 3)
                        .frame(width: 86, height: 86)
                        .scaleEffect(pulse ? 1.22 : 1)
                        .opacity(pulse ? 0 : 1)
                }
                Circle().stroke(Color.white, lineWidth: 4).frame(width: 80, height: 80)
                    .shadow(color: Theme.teal.opacity(isRecording ? 0 : 0.6), radius: 10)
                if isFinishing {
                    ProgressView().tint(.white).controlSize(.large)
                } else {
                    RoundedRectangle(cornerRadius: isRecording ? 8 : 33)
                        .fill(isRecording ? Theme.recording : Theme.primary)
                        .frame(width: isRecording ? 32 : 66, height: isRecording ? 32 : 66)
                }
            }
            .frame(width: 96, height: 96)
            .animation(.spring(duration: 0.3), value: isRecording)
        }
        .disabled(isFinishing)
        .opacity(!isRecording && !ready ? 0.6 : 1)
        .accessibilityLabel(isRecording ? "Stop scan" : "Start scan")
        .onChange(of: isRecording) { _, rec in
            pulse = false
            if rec {
                withAnimation(.easeOut(duration: 1.4).repeatForever(autoreverses: false)) { pulse = true }
            }
        }
    }
}

struct GuidanceCard: View {
    let guidance: CaptureManager.Guidance
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: guidance.symbol)
                .font(.headline)
                .foregroundStyle(guidance.isWarning ? Theme.warning : Theme.teal)
                .frame(width: 26)
            Text(guidance.text)
                .font(.subheadline.weight(.medium))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .watoGlass(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(guidance.isWarning ? Theme.warning.opacity(0.8) : .clear, lineWidth: 1.5)
        )
        .animation(.easeInOut(duration: 0.25), value: guidance)
        .id(guidance)
        .transition(.opacity)
    }
}

struct StatsHUD: View {
    @EnvironmentObject private var capture: CaptureManager

    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                stat(value: timeString(capture.elapsedSeconds), label: "time")
                Spacer()
                stat(value: "\(capture.frameCount)", label: "frames")
                Spacer()
                stat(value: String(format: "%.1f m", capture.distanceMeters), label: "walked")
            }
            HStack(spacing: 14) {
                Meter(title: "Pace", value: capture.pace == .good ? 0.35 : 1,
                      caption: capture.pace == .good ? "Good" : "Too fast",
                      tint: capture.pace == .good ? Theme.good : Theme.warning)
                Meter(title: "Detail", value: Double(capture.detail),
                      caption: capture.detail > 0.5 ? "High" : capture.detail > 0.15 ? "OK" : "Low",
                      tint: capture.detail > 0.15 ? Theme.good : Theme.warning)
            }
        }
        .padding(14)
        .watoGlass(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func stat(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.title2, design: .rounded).weight(.bold))
                .monospacedDigit()
                .contentTransition(.numericText())
                .animation(.snappy, value: value)
            Text(label)
                .font(.caption2.weight(.bold))
                .watoLabel()
                .foregroundStyle(.secondary)
        }
    }

    private func timeString(_ s: Int) -> String { String(format: "%d:%02d", s / 60, s % 60) }
}

struct Meter: View {
    let title: String
    let value: Double
    let caption: String
    let tint: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Text(caption).font(.caption.weight(.semibold)).foregroundStyle(tint)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.15))
                    Capsule().fill(tint).frame(width: max(6, geo.size.width * min(1, max(0, value))))
                }
            }
            .frame(height: 6)
            .animation(.easeOut(duration: 0.3), value: value)
        }
        .frame(maxWidth: .infinity)
    }
}

/// Top-down map of the walk so far: keyframes, start point, current position and heading.
struct MiniMap: View {
    let path: [SIMD2<Float>]
    let current: SIMD2<Float>
    let heading: SIMD2<Float>

    var body: some View {
        Canvas { ctx, size in
            let all = path + [current]
            let minP = all.reduce(SIMD2<Float>(repeating: .infinity)) { simd_min($0, $1) }
            let maxP = all.reduce(SIMD2<Float>(repeating: -.infinity)) { simd_max($0, $1) }
            let mid = (minP + maxP) / 2
            let span: Float = max(2.0, max(maxP.x - minP.x, maxP.y - minP.y))   // at least 2 m across
            let lo = mid - SIMD2<Float>(repeating: span / 2)
            let inset: CGFloat = 14
            let scale = (min(size.width, size.height) - inset * 2) / CGFloat(span)
            func pt(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: inset + CGFloat(p.x - lo.x) * scale, y: inset + CGFloat(p.y - lo.y) * scale)
            }

            // 1 m grid
            var grid = Path()
            let step = CGFloat(1) * scale
            var gx = inset - CGFloat(lo.x.truncatingRemainder(dividingBy: 1)) * scale
            while gx < size.width { grid.move(to: CGPoint(x: gx, y: 0)); grid.addLine(to: CGPoint(x: gx, y: size.height)); gx += step }
            var gy = inset - CGFloat(lo.y.truncatingRemainder(dividingBy: 1)) * scale
            while gy < size.height { grid.move(to: CGPoint(x: 0, y: gy)); grid.addLine(to: CGPoint(x: size.width, y: gy)); gy += step }
            ctx.stroke(grid, with: .color(.white.opacity(0.08)), lineWidth: 1)

            if path.count > 1 {
                var line = Path()
                line.move(to: pt(path[0]))
                for p in path.dropFirst() { line.addLine(to: pt(p)) }
                line.addLine(to: pt(current))
                ctx.stroke(line, with: .color(Theme.secondary.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
            }
            for p in path {
                let c = pt(p)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 1.5, y: c.y - 1.5, width: 3, height: 3)), with: .color(.white.opacity(0.7)))
            }
            if let start = path.first {
                let c = pt(start)
                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 6, y: c.y - 6, width: 12, height: 12)), with: .color(Theme.amber), lineWidth: 2)
            }
            // current position + heading wedge
            let c = pt(current)
            let h = CGPoint(x: CGFloat(heading.x), y: CGFloat(heading.y))
            var wedge = Path()
            wedge.move(to: c)
            let ang = atan2(h.y, h.x)
            wedge.addArc(center: c, radius: 22, startAngle: .radians(ang - 0.5), endAngle: .radians(ang + 0.5), clockwise: false)
            wedge.closeSubpath()
            ctx.fill(wedge, with: .color(Theme.secondary.opacity(0.28)))
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10)), with: .color(.white))
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(Theme.secondary))
        }
        .watoGlass(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(alignment: .bottomTrailing) {
            Text("1 m").font(.system(size: 9, weight: .bold)).fontWidth(.expanded).foregroundStyle(.secondary).padding(6)
        }
        .accessibilityLabel("Map of your path")
    }
}

struct MessageCard: View {
    let text: String
    let dismiss: () -> Void
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.teal)
            Text(text).font(.footnote.weight(.medium)).frame(maxWidth: .infinity, alignment: .leading)
            Button(action: dismiss) { Image(systemName: "xmark").font(.caption.weight(.bold)) }
                .foregroundStyle(.secondary)
                .accessibilityLabel("Dismiss")
        }
        .padding(12)
        .watoGlass(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct PhotoModeCard: View {
    let dismiss: () -> Void
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "camera.aperture").font(.headline).foregroundStyle(Theme.secondary)
            VStack(alignment: .leading, spacing: 3) {
                Text("Photo mode").font(.subheadline.weight(.semibold))
                Text("This iPhone has no LiDAR, so Pioneer Home saves photos, camera positions and tracked points. The yellow dots show what ARKit is tracking.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Button(action: dismiss) { Image(systemName: "xmark").font(.caption.weight(.bold)) }
                .foregroundStyle(.secondary)
                .accessibilityLabel("Dismiss")
        }
        .padding(12)
        .watoGlass(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
