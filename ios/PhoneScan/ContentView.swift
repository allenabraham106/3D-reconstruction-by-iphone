import SwiftUI
import simd

/// Full-screen scanning mode, opened from the Home tab's SCAN ROOM tile.
struct CaptureView: View {
    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage(CaptureSettings.Key.showMesh) private var showMesh = true
    @AppStorage(CaptureSettings.Key.showFeaturePoints) private var showFeaturePoints = true

    @State private var flash = false

    var body: some View {
        ZStack {
            ARViewContainer(session: capture.session,
                            showMesh: showMesh && capture.supportsMesh,
                            showFeaturePoints: showFeaturePoints && !(showMesh && capture.supportsMesh))
                .ignoresSafeArea()

            // Shutter flash on every saved keyframe.
            Rectangle()
                .strokeBorder(Color.white.opacity(flash ? 0.85 : 0), lineWidth: 5)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack(spacing: 0) {
                topBar
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                Spacer()
                sheet
            }
        }
        .onAppear { capture.startSession() }
        .onDisappear { if !capture.isRecording { capture.pauseSession() } }
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
        .statusBarHidden(capture.isRecording)
    }

    // MARK: Top

    private var topBar: some View {
        HStack(alignment: .top, spacing: 8) {
            if !capture.isRecording {
                RoundButton(symbol: "xmark", label: "Close") { dismiss() }
            }
            Spacer(minLength: 0)
            HStack(spacing: 7) {
                Circle()
                    .fill(capture.isRecording ? Theme.recording : (capture.trackingOK ? Theme.good : Theme.warning))
                    .frame(width: 8, height: 8)
                Text(capture.isRecording ? timeString(capture.elapsedSeconds) : capture.trackingText)
                    .font(.system(size: 15, weight: .semibold)).monospacedDigit().lineLimit(1)
                    .contentTransition(.numericText())
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.45), in: Capsule())
            Spacer(minLength: 0)
            if capture.isRecording {
                MiniMap(path: capture.path, current: capture.current, heading: capture.heading)
                    .frame(width: 96, height: 96)
                    .transition(.opacity)
            } else {
                Color.clear.frame(width: 42, height: 42)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: capture.isRecording)
    }

    // MARK: Bottom sheet

    private var sheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text(headline).font(Theme.display(22)).foregroundStyle(Theme.ink)
                Spacer(minLength: 8)
                Chip(text: capture.hasLiDAR ? "LiDAR" : "Photo mode",
                     symbol: capture.hasLiDAR ? "dot.radiowaves.left.and.right" : "camera")
            }
            Text(subline)
                .font(.subheadline)
                .foregroundStyle(capture.isRecording && capture.guidance.isWarning ? Theme.warning : Theme.ink2)
                .padding(.top, 4)
                .fixedSize(horizontal: false, vertical: true)
                .animation(.easeInOut(duration: 0.2), value: capture.guidance)
            if capture.isRecording { recordingStats.padding(.top, 18) }
            if let message = capture.message, !capture.isRecording {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.good)
                    Text(message).font(.footnote).foregroundStyle(Theme.ink)
                    Spacer(minLength: 0)
                    Button { capture.message = nil } label: { Image(systemName: "xmark").font(.caption2.weight(.bold)) }
                        .foregroundStyle(Theme.ink2)
                }
                .padding(12)
                .background(Theme.soft, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .padding(.top, 14)
            }
            controls.padding(.top, 22)
        }
        .padding(.horizontal, 24)
        .padding(.top, 22)
        .padding(.bottom, 8)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous)
                .fill(Theme.card)
                .shadow(color: .black.opacity(0.10), radius: 16, y: -2)
                .ignoresSafeArea(edges: .bottom)
        )
        .animation(.easeInOut(duration: 0.25), value: capture.isRecording)
    }

    /// Pause on the left, the record / stop button in the middle, the live-mesh toggle on the right.
    private var controls: some View {
        HStack {
            CircleButton(symbol: capture.isPaused ? "play.fill" : "pause.fill",
                         label: capture.isPaused ? "Resume" : "Pause") { capture.isPaused.toggle() }
                .opacity(capture.isRecording ? 1 : 0)
                .disabled(!capture.isRecording)
            Spacer()
            RecordButton(isRecording: capture.isRecording, isFinishing: capture.isFinishing,
                         ready: capture.trackingOK) { toggleRecording() }
            Spacer()
            CircleButton(symbol: showMesh ? "square.grid.3x3.fill" : "square.grid.3x3",
                         label: showMesh ? "Hide live mesh" : "Show live mesh") { showMesh.toggle() }
                .opacity(capture.supportsMesh ? 1 : 0)
                .disabled(!capture.supportsMesh)
        }
        .padding(.horizontal, 20)
    }

    private var headline: String {
        if capture.isRecording { return capture.isPaused ? "Mapping paused" : "Mapping your home" }
        return capture.isFinishing ? "Saving your scan…" : "Ready to map"
    }

    private var subline: String {
        if capture.isRecording {
            return capture.isPaused ? "Nothing is saved while paused. Resume from the same spot." : capture.guidance.text
        }
        return capture.hasLiDAR
            ? "Walk slowly and sweep up and down. Open the doors to every room you want mapped."
            : "No LiDAR on this iPhone, so the map is built from photos. Walk slowly, sweep up and down, and finish where you started."
    }

    private var recordingStats: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                StatView(value: "\(capture.frameCount)", label: "Frames")
                Spacer()
                StatView(value: String(format: "%.1f", capture.distanceMeters), unit: "m", label: "Walked")
                Spacer()
                StatView(value: capture.detail > 0.5 ? "High" : capture.detail > 0.15 ? "OK" : "Low", label: "Detail")
            }
            HStack(spacing: 16) {
                Meter(title: "Pace", value: capture.pace == .good ? 0.35 : 1,
                      caption: capture.pace == .good ? "Good" : "Too fast",
                      tint: capture.pace == .good ? Theme.good : Theme.warning)
                Meter(title: "Detail", value: Double(capture.detail),
                      caption: capture.detail > 0.5 ? "High" : capture.detail > 0.15 ? "OK" : "Low",
                      tint: capture.detail > 0.15 ? Theme.accent : Theme.warning)
            }
        }
    }

    private func toggleRecording() {
        if capture.isRecording {
            capture.stopRecording { _ in library.refresh() }
        } else {
            capture.startRecording()
        }
    }

    private func timeString(_ s: Int) -> String { String(format: "%02d:%02d", s / 60, s % 60) }
}

// MARK: - Pieces

/// Quiet round button beside the record button (pause, live mesh).
struct CircleButton: View {
    let symbol: String
    let label: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 18, weight: .regular))
                .foregroundStyle(Theme.ink)
                .frame(width: 54, height: 54)
                .background(Theme.soft, in: Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let isFinishing: Bool
    let ready: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(Theme.ink, lineWidth: 4).frame(width: 74, height: 74)
                if isFinishing {
                    ProgressView().tint(Theme.ink)
                } else {
                    RoundedRectangle(cornerRadius: isRecording ? 8 : 29, style: .continuous)
                        .fill(Theme.recording)
                        .frame(width: isRecording ? 30 : 58, height: isRecording ? 30 : 58)
                }
            }
            .frame(width: 82, height: 82)
            .animation(.spring(duration: 0.3), value: isRecording)
        }
        .buttonStyle(.plain)
        .disabled(isFinishing)
        .opacity(!isRecording && !ready ? 0.5 : 1)
        .accessibilityLabel(isRecording ? "Stop scan" : "Start scan")
    }
}

struct GuidanceLine: View {
    let guidance: CaptureManager.Guidance
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: guidance.symbol)
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(guidance.isWarning ? Theme.warning : Theme.ink)
                .frame(width: 22)
            Text(guidance.text)
                .font(.subheadline)
                .foregroundStyle(guidance.isWarning ? Theme.warning : Theme.ink)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .paperCard(10, fill: guidance.isWarning ? Theme.warning.opacity(0.08) : Theme.paper)
        .animation(.easeInOut(duration: 0.2), value: guidance)
    }
}

struct Meter: View {
    let title: String
    let value: Double
    let caption: String
    let tint: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title).monoLabel(9, color: Theme.ink2)
                Spacer()
                Text(caption).monoLabel(9, color: tint)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.soft)
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
            let inset: CGFloat = 12
            let scale = (min(size.width, size.height) - inset * 2) / CGFloat(span)
            func pt(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: inset + CGFloat(p.x - lo.x) * scale, y: inset + CGFloat(p.y - lo.y) * scale)
            }
            // 1 m dot grid
            let step = scale
            var gx = inset - CGFloat(lo.x.truncatingRemainder(dividingBy: 1)) * scale
            while gx < size.width {
                var gy = inset - CGFloat(lo.y.truncatingRemainder(dividingBy: 1)) * scale
                while gy < size.height {
                    ctx.fill(Path(ellipseIn: CGRect(x: gx - 1, y: gy - 1, width: 2, height: 2)), with: .color(Theme.line))
                    gy += step
                }
                gx += step
            }
            if path.count > 1 {
                var line = Path()
                line.move(to: pt(path[0]))
                for p in path.dropFirst() { line.addLine(to: pt(p)) }
                line.addLine(to: pt(current))
                ctx.stroke(line, with: .color(Theme.accent), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }
            if let start = path.first {
                let c = pt(start)
                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10)), with: .color(Theme.ink2), lineWidth: 1.5)
            }
            let c = pt(current)
            let ang = atan2(CGFloat(heading.y), CGFloat(heading.x))
            var wedge = Path()
            wedge.move(to: c)
            wedge.addArc(center: c, radius: 18, startAngle: .radians(ang - 0.5), endAngle: .radians(ang + 0.5), clockwise: false)
            wedge.closeSubpath()
            ctx.fill(wedge, with: .color(Theme.accent.opacity(0.22)))
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 4.5, y: c.y - 4.5, width: 9, height: 9)), with: .color(Theme.accent))
        }
        .paperCard(16)
        .overlay(alignment: .bottomTrailing) { Text("1 m").monoLabel(8, color: Theme.ink2).padding(7) }
        .accessibilityLabel("Map of your path")
    }
}
