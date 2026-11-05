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
            HStack(spacing: 6) {
                Circle().fill(capture.trackingOK ? Theme.good : Theme.warning).frame(width: 7, height: 7)
                Text(capture.trackingText).lineLimit(1)
            }
            .monoLabel(10, color: Theme.card)
            .padding(.horizontal, 9)
            .padding(.vertical, 7)
            .background(Theme.chip.opacity(0.92), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            Spacer(minLength: 0)
            if capture.isRecording {
                MiniMap(path: capture.path, current: capture.current, heading: capture.heading)
                    .frame(width: 104, height: 104)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: capture.isRecording)
    }

    // MARK: Bottom sheet

    private var sheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Theme.line).frame(width: 36, height: 4).frame(maxWidth: .infinity).padding(.bottom, 14)
            HStack(spacing: 6) {
                if capture.isRecording {
                    Chip(text: capture.isPaused ? "Paused" : "Mapping", symbol: capture.isPaused ? "pause.fill" : "map",
                         style: .solid(capture.isPaused ? Theme.ink2 : Theme.accent))
                    if capture.guidance.isWarning { Chip(text: "Slow down", style: .solid(Theme.warning)) }
                } else if capture.trackingOK {
                    Chip(text: "Ready", symbol: "checkmark", style: .solid(Theme.chipGreen))
                } else {
                    Chip(text: "Starting", symbol: "hourglass")
                }
                Spacer(minLength: 0)
                Chip(text: capture.hasLiDAR ? "LiDAR" : "Photo mode", symbol: capture.hasLiDAR ? "dot.radiowaves.left.and.right" : "camera")
            }
            Text(headline).font(Theme.display(22)).foregroundStyle(Theme.ink).padding(.top, 10)
            Text(subline)
                .font(.subheadline)
                .foregroundStyle(capture.isRecording && capture.guidance.isWarning ? Theme.warning : Theme.ink2)
                .padding(.top, 2)
                .fixedSize(horizontal: false, vertical: true)
                .animation(.easeInOut(duration: 0.2), value: capture.guidance)
            if capture.isRecording { recordingStats.padding(.top, 14) }
            if let message = capture.message, !capture.isRecording {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle").foregroundStyle(Theme.chipGreen)
                    Text(message).font(.footnote).foregroundStyle(Theme.ink)
                    Spacer(minLength: 0)
                    Button { capture.message = nil } label: { Image(systemName: "xmark").font(.caption2.weight(.bold)) }
                        .foregroundStyle(Theme.ink2)
                }
                .padding(10)
                .paperCard(10, fill: Theme.paper)
                .padding(.top, 12)
            }
            HStack(spacing: 10) {
                if capture.isRecording {
                    Tile(title: capture.isPaused ? "Resume" : "Pause", symbol: capture.isPaused ? "play" : "pause") {
                        capture.isPaused.toggle()
                    }
                    Tile(title: "Stop", symbol: "stop", busy: capture.isFinishing) {
                        capture.stopRecording { _ in library.refresh() }
                    }
                } else {
                    Tile(title: capture.isFinishing ? "Saving…" : "Start mapping", symbol: "record.circle",
                         prominent: true, busy: capture.isFinishing) {
                        capture.startRecording()
                    }
                    .disabled(capture.isFinishing)
                }
            }
            .padding(.top, 16)
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 22, topTrailingRadius: 22, style: .continuous)
                .fill(Theme.card)
                .shadow(color: .black.opacity(0.08), radius: 12, y: -2)
                .ignoresSafeArea(edges: .bottom)
        )
        .animation(.easeInOut(duration: 0.25), value: capture.isRecording)
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
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                stat(timeString(capture.elapsedSeconds), "time")
                Spacer()
                stat("\(capture.frameCount)", "frames")
                Spacer()
                stat(String(format: "%.1f m", capture.distanceMeters), "walked")
            }
            HStack(spacing: 16) {
                Meter(title: "Pace", value: capture.pace == .good ? 0.35 : 1,
                      caption: capture.pace == .good ? "Good" : "Too fast",
                      tint: capture.pace == .good ? Theme.chipGreen : Theme.warning)
                Meter(title: "Detail", value: Double(capture.detail),
                      caption: capture.detail > 0.5 ? "High" : capture.detail > 0.15 ? "OK" : "Low",
                      tint: capture.detail > 0.15 ? Theme.chipGreen : Theme.warning)
            }
        }
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 20, weight: .regular, design: .monospaced))
                .foregroundStyle(Theme.ink)
                .contentTransition(.numericText())
                .animation(.snappy, value: value)
            Text(label).monoLabel(9, color: Theme.ink2)
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

struct RecordButton: View {
    let isRecording: Bool
    let isFinishing: Bool
    let ready: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(Theme.ink, lineWidth: 2).frame(width: 70, height: 70)
                if isFinishing {
                    ProgressView().tint(Theme.ink)
                } else {
                    RoundedRectangle(cornerRadius: isRecording ? 6 : 28, style: .continuous)
                        .fill(Theme.recording)
                        .frame(width: isRecording ? 26 : 56, height: isRecording ? 26 : 56)
                }
            }
            .frame(width: 78, height: 78)
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
                    Rectangle().fill(Theme.line)
                    Rectangle().fill(tint).frame(width: max(3, geo.size.width * min(1, max(0, value))))
                }
            }
            .frame(height: 3)
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
                ctx.stroke(line, with: .color(Theme.ink), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
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
        .paperCard(12)
        .overlay(alignment: .bottomTrailing) { Text("1 m").monoLabel(8, color: Theme.ink2).padding(6) }
        .accessibilityLabel("Map of your path")
    }
}
