import SwiftUI
import UIKit

/// Plays an MJPEG stream (multipart/x-mixed-replace, as robot_link's /camera/stream sends).
/// iOS has no built-in MJPEG player, so this reads the bytes and cuts out each JPEG
/// (FF D8 ... FF D9). That works whatever the part boundary is called.
@MainActor
final class MJPEGStream: NSObject, ObservableObject, URLSessionDataDelegate {
    @Published private(set) var frame: UIImage?
    @Published private(set) var fps: Double = 0
    @Published private(set) var state: State = .idle

    enum State: Equatable { case idle, connecting, live, failed(String) }

    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var buffer = Data()
    private var lastFrame = Date()

    func start(_ url: URL) {
        stop()
        state = .connecting
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = .infinity
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        let session = URLSession(configuration: config, delegate: self, delegateQueue: .main)
        self.session = session
        let task = session.dataTask(with: url)
        self.task = task
        task.resume()
    }

    func stop() {
        task?.cancel()
        session?.invalidateAndCancel()
        task = nil
        session = nil
        buffer.removeAll()
        if state != .idle { state = .idle }
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                                completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        MainActor.assumeIsolated {
            if code != 200 { self.state = .failed(code == 503 ? "Pioneer has no camera running" : "Camera returned \(code)") }
        }
        completionHandler(code == 200 ? .allow : .cancel)
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        MainActor.assumeIsolated { self.consume(data) }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        MainActor.assumeIsolated {
            guard self.task === task else { return }
            if let error, (error as NSError).code != NSURLErrorCancelled {
                self.state = .failed("Lost the camera: \(error.localizedDescription)")
            }
        }
    }

    private func consume(_ data: Data) {
        buffer.append(data)
        let soi = Data([0xFF, 0xD8]), eoi = Data([0xFF, 0xD9])
        // keep only the newest complete frame; drop older ones if Wi-Fi bunches them up
        var newest: Data?
        while let s = buffer.range(of: soi), let e = buffer.range(of: eoi, in: s.upperBound..<buffer.endIndex) {
            newest = buffer.subdata(in: s.lowerBound..<e.upperBound)
            buffer.removeSubrange(buffer.startIndex..<e.upperBound)
        }
        if buffer.count > 4_000_000 { buffer.removeAll() }   // garbage guard
        if let newest, let img = UIImage(data: newest) {
            let now = Date()
            let dt = now.timeIntervalSince(lastFrame)
            lastFrame = now
            if dt > 0 { fps = fps == 0 ? 1 / dt : 0.85 * fps + 0.15 / dt }
            frame = img
            if state != .live { state = .live }
        }
    }
}

/// "Pioneer's view": the robot's live camera, full screen, with localization from what it sees.
struct LiveFeedView: View {
    @EnvironmentObject private var robot: RobotLink
    @Environment(\.dismiss) private var dismiss
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"
    @StateObject private var stream = MJPEGStream()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let img = stream.frame {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea(edges: .horizontal)
            } else {
                VStack(spacing: 10) {
                    if case .failed(let why) = stream.state {
                        Image(systemName: "video.slash").font(.system(size: 30, weight: .light))
                        Text(why).font(.footnote).multilineTextAlignment(.center)
                    } else {
                        ProgressView().tint(.white)
                        Text("Connecting to \(name)'s camera…").font(.footnote)
                    }
                }
                .foregroundStyle(.white.opacity(0.8))
                .padding(40)
            }

            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    RoundButton(symbol: "xmark", label: "Close") { dismiss() }
                    Spacer()
                    if stream.state == .live {
                        HStack(spacing: 6) {
                            Chip(text: "Live", symbol: "circle.fill", style: .solid(Theme.recording))
                            Text(String(format: "%.0f fps", stream.fps)).monoLabel(10, color: .white)
                        }
                        .padding(.top, 10)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                Spacer()
                sheet
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { if let url = streamURL { stream.start(url) } }
        .onDisappear { stream.stop() }
        .statusBarHidden()
    }

    private var sheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Theme.line).frame(width: 36, height: 4).frame(maxWidth: .infinity).padding(.bottom, 14)
            HStack(spacing: 6) {
                Chip(text: "Camera", symbol: "video", style: .solid(Theme.accent))
                if let cam = robot.status?.camera { Chip(text: cam) }
            }
            Text("\(name)'s view").font(Theme.display(22)).foregroundStyle(Theme.ink).padding(.top, 10)
            Text(subline).font(.subheadline).foregroundStyle(Theme.ink2).padding(.top, 2)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) {
                Tile(title: "Localize from this view", symbol: "scope", busy: robot.busy == "Localizing robot…") {
                    Task { await robot.localizeRobot(robotURL) }
                }
                .disabled(robot.status?.map == nil || robot.busy != nil)
                Tile(title: "Back to map", symbol: "map") { dismiss() }
            }
            .padding(.top, 16)
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 22, topTrailingRadius: 22, style: .continuous)
                .fill(Theme.card)
                .ignoresSafeArea(edges: .bottom)
        )
        .environment(\.colorScheme, .light)
    }

    private var name: String { robot.status?.name ?? "Pioneer" }

    private var subline: String {
        if let r = robot.lastResult, r.source == "robot" {
            return r.ok ? String(format: "Found at x %.2f · y %.2f m · %.0f° (%d matches)", r.x, r.y, r.yaw_deg, r.inliers)
                        : r.reason
        }
        return "What \(name)'s camera sees right now. Localize to place it on the map from this view."
    }

    private var streamURL: URL? {
        URL(string: robotURL.trimmingCharacters(in: .whitespacesAndNewlines))?.appendingPathComponent("camera/stream")
    }
}
