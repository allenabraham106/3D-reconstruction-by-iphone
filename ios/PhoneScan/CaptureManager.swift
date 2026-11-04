import ARKit
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
    @Published private(set) var isFinishing = false
    @Published private(set) var frameCount = 0
    @Published private(set) var elapsedSeconds = 0
    @Published private(set) var distanceMeters: Float = 0
    @Published private(set) var trackingText = "Starting…"
    @Published private(set) var trackingOK = false
    @Published private(set) var droppedFrames = 0
    @Published var message: String?

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
        if settings.saveMesh && supportsMesh {
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
        startDate = Date()
        isRecording = true
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func stopRecording(completion: ((URL?) -> Void)? = nil) {
        guard isRecording, let writer else { return }
        isRecording = false
        isFinishing = true
        UIApplication.shared.isIdleTimerDisabled = false

        var mesh: MeshSnapshot?
        if settings.saveMesh, let anchors = session.currentFrame?.anchors {
            let meshAnchors = anchors.compactMap { $0 as? ARMeshAnchor }
            if !meshAnchors.isEmpty {
                mesh = MeshSnapshot(anchors: meshAnchors)
            }
        }

        writer.finish(mesh: mesh) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isFinishing = false
                self.writer = nil
                switch result {
                case .success(let url):
                    self.message = "Saved \(self.frameCount) frames to \(url.lastPathComponent)"
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

        guard isKeyframe(pose: pose, time: frame.timestamp) else { return }

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

    // MARK: Helpers

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
