import ARKit
import CoreImage
import Foundation
import UIKit

/// Records an ARKit session to disk in the format `recon/ingest_arkit.py` reads:
///
///   <Documents>/scan_<date>/
///       frames.jsonl          one JSON object per saved frame (pose, intrinsics, file names)
///       rgb/000000.jpg        captured camera image (1920x1440 on current iPhones)
///       depth/000000.bin      LiDAR depth, float32 metres, row-major (256x192)  [Pro models only]
///       conf/000000.bin       LiDAR confidence, uint8 0/1/2                       [Pro models only]
///       meta.json             device, LiDAR availability, counts
///
/// Frames are saved only when the phone has moved >= 5 cm or turned >= 5 degrees (and at
/// most ~6 per second), so a slow walk produces a compact, non-redundant capture.
final class CaptureSession: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isRecording = false
    @Published var savedFrames = 0
    @Published var trackingText = "Initialising"
    @Published var hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    @Published var lastScanURL: URL?
    @Published var distanceWalked: Float = 0

    let session = ARSession()
    private let writeQueue = DispatchQueue(label: "capture.write", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var scanDir: URL?
    private var jsonl: FileHandle?
    private var lastSavedTransform: simd_float4x4?
    private var lastSavedTime: TimeInterval = 0
    private var inFlight = 0  // frames queued for writing (drop if the disk can't keep up)

    // Thresholds
    private let minTranslation: Float = 0.05
    private let minRotationDeg: Float = 5
    private let minInterval: TimeInterval = 1.0 / 6.0

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity  // +Y is up: the robot map uses this
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)  // raw per-frame LiDAR depth
        }
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh  // live mesh preview in the AR view
        }
        // Highest-resolution 4:3 video format available (1920x1440 on iPhone 12+ / 17).
        if let best = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ abs(Double($0.imageResolution.width) / Double($0.imageResolution.height) - 4.0 / 3.0) < 0.01 })
            .max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = best
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func pause() { session.pause() }

    // MARK: recording

    func toggleRecording() { isRecording ? stopRecording() : startRecording() }

    private func startRecording() {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("scan_\(fmt.string(from: Date()))")
        for sub in ["rgb", "depth", "conf"] {
            try? FileManager.default.createDirectory(at: dir.appendingPathComponent(sub),
                                                     withIntermediateDirectories: true)
        }
        FileManager.default.createFile(atPath: dir.appendingPathComponent("frames.jsonl").path, contents: nil)
        jsonl = try? FileHandle(forWritingTo: dir.appendingPathComponent("frames.jsonl"))
        scanDir = dir
        savedFrames = 0
        distanceWalked = 0
        lastSavedTransform = nil
        isRecording = true
    }

    private func stopRecording() {
        isRecording = false
        writeQueue.async { [weak self] in
            guard let self, let dir = self.scanDir else { return }
            try? self.jsonl?.close()
            self.jsonl = nil
            let meta: [String: Any] = [
                "device": UIDevice.current.modelIdentifier,
                "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
                "lidar": self.hasLiDAR,
                "frames": self.savedFrames,
                "format": "scan-capture-v1",
                "pose_convention": "world_from_camera (column-major 4x4); camera x right, y up, z backward; world +y up (gravity), metres",
            ]
            if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted]) {
                try? data.write(to: dir.appendingPathComponent("meta.json"))
            }
            DispatchQueue.main.async { self.lastScanURL = dir }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateTrackingText(frame.camera.trackingState)
        guard isRecording, case .normal = frame.camera.trackingState else { return }
        let T = frame.camera.transform
        let now = frame.timestamp
        if now - lastSavedTime < minInterval { return }
        if let last = lastSavedTransform {
            let dt = simd_distance(T.columns.3, last.columns.3)
            let rel = simd_mul(simd_inverse(last), T)
            let angle = acos(max(-1, min(1, (rel.columns.0.x + rel.columns.1.y + rel.columns.2.z - 1) / 2))) * 180 / .pi
            if dt < minTranslation && angle < minRotationDeg { return }
            distanceWalked += dt
        }
        if inFlight > 4 { return }  // storage is behind; skip rather than stall tracking
        lastSavedTransform = T
        lastSavedTime = now
        save(frame)
    }

    private func save(_ frame: ARFrame) {
        guard let dir = scanDir else { return }
        let index = savedFrames
        savedFrames += 1
        inFlight += 1
        let name = String(format: "%06d", index)
        // Copy what we need off the ARFrame now (ARKit recycles its buffers).
        let image = CIImage(cvPixelBuffer: frame.capturedImage)
        let width = CVPixelBufferGetWidth(frame.capturedImage)
        let height = CVPixelBufferGetHeight(frame.capturedImage)
        let K = frame.camera.intrinsics
        let T = frame.camera.transform
        let depthCopy = frame.sceneDepth.map { copyFloatBuffer($0.depthMap) }
        let confCopy = frame.sceneDepth?.confidenceMap.map { copyByteBuffer($0) }
        let timestamp = frame.timestamp

        writeQueue.async { [weak self] in
            guard let self else { return }
            defer { DispatchQueue.main.async { self.inFlight -= 1 } }
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            if let jpg = self.ciContext.jpegRepresentation(of: image, colorSpace: cs,
                                                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9]) {
                try? jpg.write(to: dir.appendingPathComponent("rgb/\(name).jpg"))
            }
            var entry: [String: Any] = [
                "index": index, "timestamp": timestamp,
                "rgb": "rgb/\(name).jpg", "width": width, "height": height,
                // column-major 3x3 / 4x4, exactly as simd stores them
                "intrinsics": [K.columns.0.x, K.columns.0.y, K.columns.0.z,
                               K.columns.1.x, K.columns.1.y, K.columns.1.z,
                               K.columns.2.x, K.columns.2.y, K.columns.2.z],
                "world_from_camera": [T.columns.0.x, T.columns.0.y, T.columns.0.z, T.columns.0.w,
                                      T.columns.1.x, T.columns.1.y, T.columns.1.z, T.columns.1.w,
                                      T.columns.2.x, T.columns.2.y, T.columns.2.z, T.columns.2.w,
                                      T.columns.3.x, T.columns.3.y, T.columns.3.z, T.columns.3.w],
            ]
            if let (d, dw, dh) = depthCopy {
                try? d.write(to: dir.appendingPathComponent("depth/\(name).bin"))
                entry["depth"] = "depth/\(name).bin"
                entry["depth_width"] = dw
                entry["depth_height"] = dh
            }
            if let (c, _, _) = confCopy {
                try? c.write(to: dir.appendingPathComponent("conf/\(name).bin"))
                entry["confidence"] = "conf/\(name).bin"
            }
            if var line = try? JSONSerialization.data(withJSONObject: entry) {
                line.append(0x0A)
                self.jsonl?.write(line)
            }
        }
    }

    private func updateTrackingText(_ state: ARCamera.TrackingState) {
        let text: String
        switch state {
        case .normal: text = "Tracking OK"
        case .notAvailable: text = "Tracking unavailable"
        case .limited(.excessiveMotion): text = "Slow down"
        case .limited(.insufficientFeatures): text = "Point at more detail"
        case .limited(.initializing): text = "Initialising - move slowly"
        case .limited(.relocalizing): text = "Relocalising"
        case .limited: text = "Tracking limited"
        }
        if text != trackingText { DispatchQueue.main.async { self.trackingText = text } }
    }
}

// MARK: buffer helpers

private func copyFloatBuffer(_ pb: CVPixelBuffer) -> (Data, Int, Int) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!
    var out = Data(capacity: w * h * 4)
    for y in 0..<h { out.append(base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self), count: w * 4) }
    return (out, w, h)
}

private func copyByteBuffer(_ pb: CVPixelBuffer) -> (Data, Int, Int) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!
    var out = Data(capacity: w * h)
    for y in 0..<h { out.append(base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self), count: w) }
    return (out, w, h)
}

extension UIDevice {
    var modelIdentifier: String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
