import ARKit
import CoreImage
import ImageIO
import Combine
import UIKit
import simd

/// Owns the ARSession, decides when to keep a keyframe, and hands keyframes to a CaptureWriter.
///
/// All ARSessionDelegate callbacks arrive on the main queue (delegateQueue is nil), so the
/// @Published state below is only ever touched on the main thread.
final class CaptureManager: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published private(set) var isRecording = false
    /// Recording, but not saving keyframes (the user tapped PAUSE).
    @Published var isPaused = false
    @Published private(set) var isFinishing = false
    @Published private(set) var frameCount = 0
    @Published private(set) var elapsedSeconds = 0
    @Published private(set) var distanceMeters: Float = 0
    @Published private(set) var trackingText = "Starting…"
    @Published private(set) var trackingOK = false
    @Published private(set) var droppedFrames = 0
    @Published var message: String?

    // Live feedback for the capture HUD.
    /// Keyframe positions seen from above (world x, z), for the mini-map.
    @Published private(set) var path: [SIMD2<Float>] = []
    /// Current position and viewing direction seen from above.
    @Published private(set) var current = SIMD2<Float>(0, 0)
    @Published private(set) var heading = SIMD2<Float>(0, -1)
    @Published private(set) var pace: Pace = .good
    /// How much visual detail ARKit is tracking right now, 0...1 (feature points in view).
    @Published private(set) var detail: Float = 0
    @Published private(set) var guidance: Guidance = .start
    /// Increments on every saved keyframe (drives the shutter flash).
    @Published private(set) var keyframeTick = 0
    /// Increments when tracking degrades while recording (drives a warning haptic).
    @Published private(set) var warningTick = 0

    enum Pace { case good, fast }

    enum Guidance: Equatable {
        case start, keepGoing, sweep, lowDetail, tooFast, relocalizing, headBack, loopClosed
        var text: String {
            switch self {
            case .start: return "Pan slowly across the room to begin"
            case .keepGoing: return "Looking good: keep walking slowly"
            case .sweep: return "Tilt up and down to catch floor and ceiling edges"
            case .lowDetail: return "Low detail here: aim at furniture or textured areas"
            case .tooFast: return "Too fast: slow down"
            case .relocalizing: return "Lost track: go back to where you just were"
            case .headBack: return "Head back to your start point to close the loop"
            case .loopClosed: return "Loop closed: you can stop, or keep scanning"
            }
        }
        var symbol: String {
            switch self {
            case .start: return "hand.point.up.left"
            case .keepGoing: return "checkmark.circle"
            case .sweep: return "arrow.up.and.down"
            case .lowDetail: return "eye.trianglebadge.exclamationmark"
            case .tooFast: return "tortoise"
            case .relocalizing: return "arrow.uturn.backward"
            case .headBack: return "arrow.triangle.turn.up.right.circle"
            case .loopClosed: return "checkmark.seal"
            }
        }
        var isWarning: Bool { [.lowDetail, .tooFast, .relocalizing].contains(self) }
    }

    let hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    let supportsMesh = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)

    private var writer: CaptureWriter?
    private var settings = CaptureSettings.load()
    private var lastKeyPose: simd_float4x4?
    private var lastKeyTime: TimeInterval = 0
    private var lastPosition: SIMD3<Float>?
    private var distanceAccum: Float = 0
    private var startDate = Date()
    private var sessionRunning = false
    private var paceSamples: [(t: TimeInterval, p: SIMD3<Float>, q: simd_quatf)] = []
    private var pitchMin: Float = 0
    private var pitchMax: Float = 0
    private var maxDistanceFromStart: Float = 0
    private var lastHUDUpdate: TimeInterval = 0
    private var stopReason: String?

    override init() {
        super.init()
        session.delegate = self
    }

    // MARK: Session lifecycle

    func startSession() {
        guard !isRecording, !isFinishing else { return }
        guard ARWorldTrackingConfiguration.isSupported else {
            message = "ARKit world tracking is not supported on this device."
            return
        }
        settings = CaptureSettings.load()
        session.run(makeConfiguration())
        sessionRunning = true
    }

    func pauseSession() {
        if isRecording { stopRecording() }
        session.pause()
        sessionRunning = false
    }

    private func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.environmentTexturing = .none
        config.isAutoFocusEnabled = true
        if settings.use4K, let format = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution {
            config.videoFormat = format
        }
        if hasLiDAR {
            config.frameSemantics.insert(.sceneDepth)
        }
        // The live-mesh overlay needs reconstruction running even when the mesh isn't saved.
        if supportsMesh && (settings.saveMesh || UserDefaults.standard.bool(forKey: CaptureSettings.Key.showMesh)) {
            config.sceneReconstruction = .mesh
        }
        return config
    }

    // MARK: Recording

    func startRecording() {
        guard !isRecording, !isFinishing else { return }
        settings = CaptureSettings.load()
        // Restart tracking so the capture has its own world origin and a fresh mesh.
        session.run(makeConfiguration(), options: [.resetTracking, .removeExistingAnchors])
        sessionRunning = true

        let videoSize = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution.flatMap {
            settings.use4K ? $0.imageResolution : nil
        }
        do {
            writer = try CaptureWriter(
                root: CaptureLibrary.capturesDirectory,
                settings: settings,
                hasDepth: hasLiDAR && settings.saveDepth,
                deviceInfo: DeviceInfo.current(hasLiDAR: hasLiDAR, videoSize: videoSize)
            )
        } catch {
            message = "Could not create capture folder: \(error.localizedDescription)"
            return
        }
        lastKeyPose = nil
        lastKeyTime = 0
        lastPosition = nil
        distanceAccum = 0
        distanceMeters = 0
        frameCount = 0
        droppedFrames = 0
        elapsedSeconds = 0
        path = []
        paceSamples = []
        pitchMin = 0; pitchMax = 0
        maxDistanceFromStart = 0
        pace = .good
        guidance = .start
        startDate = Date()
        isPaused = false
        isRecording = true
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func stopRecording(completion: ((URL?) -> Void)? = nil) {
        guard isRecording, let writer else { return }
        isRecording = false
        isFinishing = true
        UIApplication.shared.isIdleTimerDisabled = false

        // iOS suspends the app seconds after Home/lock; finish writing before that.
        let bgTask = UIApplication.shared.beginBackgroundTask(withName: "Finish scan")
        let meshAnchors = settings.saveMesh
            ? (session.currentFrame?.anchors.compactMap { $0 as? ARMeshAnchor } ?? []) : []

        writer.finish(meshAnchors: meshAnchors, distance: distanceAccum) { [weak self] result in
            DispatchQueue.main.async {
                defer { UIApplication.shared.endBackgroundTask(bgTask) }
                guard let self else { return }
                self.isFinishing = false
                self.writer = nil
                let reason = self.stopReason.map { "Stopped early (\($0)). " } ?? ""
                self.stopReason = nil
                switch result {
                case .success(let url):
                    let saved = writer.writtenCount
                    if let err = writer.writeError {
                        self.message = reason + "Saved \(saved) of \(self.frameCount) frames to \(url.lastPathComponent). Some failed: \(err.localizedDescription)"
                    } else {
                        self.message = reason + "Saved \(saved) frames to \(url.lastPathComponent)"
                    }
                    completion?(url)
                case .failure(let error):
                    self.message = "Saving failed: \(error.localizedDescription)"
                    completion?(nil)
                }
            }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateTrackingStatus(frame.camera.trackingState)

        guard isRecording, let writer else { return }

        let seconds = Int(Date().timeIntervalSince(startDate))
        if seconds != elapsedSeconds { elapsedSeconds = seconds }

        let pose = frame.camera.transform
        let position = SIMD3<Float>(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)

        guard case .normal = frame.camera.trackingState else {
            // Don't integrate distance across relocalization jumps either.
            lastPosition = nil
            return
        }
        if let last = lastPosition { distanceAccum += simd_distance(last, position) }
        lastPosition = position
        updateHUD(frame: frame, pose: pose, position: position)

        guard !isPaused, isKeyframe(pose: pose, time: frame.timestamp) else { return }

        if writer.writeError != nil || !Self.hasFreeSpace() {
            stopReason = writer.writeError == nil ? "iPhone storage almost full" : "frames stopped saving"
            stopRecording()
            return
        }

        // Back-pressure: if encoding falls behind, skip rather than holding ARKit's buffers.
        guard writer.pendingCount < 3 else {
            droppedFrames += 1
            return
        }

        lastKeyPose = pose
        lastKeyTime = frame.timestamp
        writer.add(frame: frame)
        frameCount = writer.frameCount
        distanceMeters = distanceAccum
        path.append(SIMD2(position.x, position.z))
        keyframeTick &+= 1
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        message = "AR session failed: \(error.localizedDescription)"
        sessionRunning = false
    }

    func sessionWasInterrupted(_ session: ARSession) {
        trackingText = "Interrupted"
        trackingOK = false
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        trackingText = "Relocalizing…"
    }

    func sessionShouldAttemptRelocalization(_ session: ARSession) -> Bool { true }

    // MARK: Robot localization

    /// The current camera frame as a JPEG plus its intrinsics, for "Localize this phone".
    /// The image stays in the sensor's landscape orientation; the intrinsics match it.
    func snapshotForLocalization() -> (jpeg: Data, fx: Float, fy: Float, cx: Float, cy: Float)? {
        guard let frame = session.currentFrame else { return nil }
        let K = frame.camera.intrinsics
        let image = CIImage(cvPixelBuffer: frame.capturedImage)
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let data = snapshotContext.jpegRepresentation(of: image, colorSpace: space, options: [key: 0.85])
        else { return nil }
        return (data, K[0][0], K[1][1], K[2][0], K[2][1])
    }

    private lazy var snapshotContext = CIContext()

    // MARK: HUD

    /// Pace, detail, mini-map and guidance. Runs every frame but publishes at most 10 times a second.
    private func updateHUD(frame: ARFrame, pose: simd_float4x4, position: SIMD3<Float>) {
        let t = frame.timestamp
        let q = simd_quatf(pose)
        paceSamples.append((t, position, q))
        paceSamples.removeAll { t - $0.t > 1.0 }

        let forward = -SIMD3<Float>(pose.columns.2.x, pose.columns.2.y, pose.columns.2.z)
        let pitch = asin(max(-1, min(1, forward.y)))
        pitchMin = min(pitchMin, pitch); pitchMax = max(pitchMax, pitch)
        if let start = path.first {
            maxDistanceFromStart = max(maxDistanceFromStart, simd_distance(start, SIMD2(position.x, position.z)))
        }

        guard t - lastHUDUpdate >= 0.1 else { return }
        lastHUDUpdate = t

        current = SIMD2(position.x, position.z)
        let flat = SIMD2(forward.x, forward.z)
        if simd_length(flat) > 0.1 { heading = simd_normalize(flat) }

        // Speed over the last second: walking > 0.6 m/s or turning > 70°/s blurs frames.
        var newPace = Pace.good
        if let first = paceSamples.first, t - first.t > 0.3 {
            let dt = Float(t - first.t)
            let speed = simd_distance(first.p, position) / dt
            let rel = first.q.inverse * q
            let turn = 2 * acos(min(1, abs(rel.real))) / dt
            if speed > 0.6 || turn > 70 * .pi / 180 { newPace = .fast }
        }
        if newPace != pace { pace = newPace }

        // ~400 tracked feature points in view is plenty for ARKit and SfM.
        let points = Float(frame.rawFeaturePoints?.points.count ?? 0)
        let newDetail = min(1, points / 400)
        if abs(newDetail - detail) > 0.03 { detail = newDetail }

        let next = nextGuidance(state: frame.camera.trackingState, position: position)
        if next != guidance {
            if next.isWarning && !guidance.isWarning { warningTick &+= 1 }
            guidance = next
        }
    }

    private func nextGuidance(state: ARCamera.TrackingState, position: SIMD3<Float>) -> Guidance {
        if case .limited(let reason) = state {
            switch reason {
            case .excessiveMotion: return .tooFast
            case .insufficientFeatures: return .lowDetail
            case .relocalizing: return .relocalizing
            default: return .start
            }
        }
        if pace == .fast { return .tooFast }
        if detail < 0.15 && frameCount > 3 { return .lowDetail }
        if frameCount < 8 { return .start }
        if pitchMax - pitchMin < 0.6 && frameCount > 25 { return .sweep }
        if let start = path.first, maxDistanceFromStart > 1.5, elapsedSeconds > 45 {
            let here = simd_distance(start, SIMD2(position.x, position.z))
            return here < 0.6 ? .loopClosed : .headBack
        }
        return .keepGoing
    }

    // MARK: Helpers

    /// Checked every keyframe; the lookup is a cheap statfs. 4K frames are several MB each.
    private static func hasFreeSpace() -> Bool {
        let values = try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        guard let free = values?.volumeAvailableCapacityForImportantUsage else { return true }
        return free > 500_000_000
    }

    private func isKeyframe(pose: simd_float4x4, time: TimeInterval) -> Bool {
        guard let last = lastKeyPose else { return true }
        guard time - lastKeyTime >= settings.minInterval else { return false }

        let p0 = SIMD3<Float>(last.columns.3.x, last.columns.3.y, last.columns.3.z)
        let p1 = SIMD3<Float>(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
        if simd_distance(p0, p1) >= settings.minTranslation { return true }

        let q0 = simd_quatf(last)
        let q1 = simd_quatf(pose)
        let relative = q0.inverse * q1
        let angle = 2 * acos(min(1, abs(relative.real)))
        return angle >= settings.minRotation
    }

    private func updateTrackingStatus(_ state: ARCamera.TrackingState) {
        let text: String
        let ok: Bool
        switch state {
        case .normal:
            text = "Tracking"; ok = true
        case .notAvailable:
            text = "Tracking unavailable"; ok = false
        case .limited(let reason):
            ok = false
            switch reason {
            case .initializing: text = "Initializing: move slowly"
            case .excessiveMotion: text = "Too fast: slow down"
            case .insufficientFeatures: text = "Low detail: aim at textured areas"
            case .relocalizing: text = "Relocalizing…"
            @unknown default: text = "Limited tracking"
            }
        }
        if text != trackingText { trackingText = text }
        if ok != trackingOK { trackingOK = ok }
    }
}

/// Static facts about the phone, written into metadata.json.
struct DeviceInfo: Codable {
    var model: String
    var systemVersion: String
    var appVersion: String
    var hasLiDAR: Bool
    var videoWidth: Int?
    var videoHeight: Int?

    static func current(hasLiDAR: Bool, videoSize: CGSize?) -> DeviceInfo {
        var sys = utsname()
        uname(&sys)
        let machine = withUnsafeBytes(of: sys.machine) { raw -> String in
            let bytes = raw.prefix { $0 != 0 }
            return String(decoding: bytes, as: UTF8.self)
        }
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        return DeviceInfo(
            model: machine,
            systemVersion: UIDevice.current.systemVersion,
            appVersion: version,
            hasLiDAR: hasLiDAR,
            videoWidth: videoSize.map { Int($0.width) },
            videoHeight: videoSize.map { Int($0.height) }
        )
    }
}
