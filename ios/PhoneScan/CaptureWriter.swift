import ARKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers

/// One keyframe, in the nerfstudio `transforms.json` frame format (plus a few extras).
///
/// `transform_matrix` is ARKit's camera-to-world transform, row-major. ARKit cameras use the
/// OpenGL convention (+x right, +y up, looking down -z) in a gravity-aligned, y-up world, which is
/// exactly what nerfstudio expects. Intrinsics refer to the full-resolution JPEG.
struct FrameRecord: Codable {
    var file_path: String
    var depth_file_path: String?
    var confidence_file_path: String?
    var transform_matrix: [[Float]]
    var fl_x: Float
    var fl_y: Float
    var cx: Float
    var cy: Float
    var w: Int
    var h: Int
    var timestamp: Double
    var exposure_duration: Double
}

/// Writes keyframes to disk on a background queue.
///
/// Folder layout of one capture:
///
///     2026-11-04_03-55-12/
///       images/frame_00000.jpg        RGB, sensor orientation (landscape), sRGB JPEG
///       depth/frame_00000.png         LiDAR depth, 16-bit PNG, millimetres (0 = invalid)
///       confidence/frame_00000.png    LiDAR confidence, 8-bit PNG, 0 = low, 1 = medium, 2 = high
///       transforms.json               nerfstudio format: poses + intrinsics for every frame
///       frames.jsonl                  same records, appended live (survives a crash)
///       mesh.ply                      ARKit scene mesh in world coordinates (LiDAR phones)
///       arkit_points.ply              ARKit feature points, sparse metric cloud (every iPhone)
///       metadata.json                 device, app version, counts
final class CaptureWriter {
    let directory: URL

    /// Number of keyframes handed to `add`. Main thread only.
    private(set) var frameCount = 0

    var pendingCount: Int {
        lock.lock(); defer { lock.unlock() }
        return pending
    }

    private let settings: CaptureSettings
    private let hasDepth: Bool
    private let deviceInfo: DeviceInfo
    private let queue = DispatchQueue(label: "PhoneScan.CaptureWriter", qos: .userInitiated)
    private let lock = NSLock()
    private var pending = 0
    private let ciContext = CIContext()
    private let jsonl: FileHandle
    private var records: [FrameRecord] = []   // queue only
    /// ARKit's tracked visual feature points, keyed by ARKit's persistent identifier, latest
    /// estimate wins. Works on every iPhone, so non-LiDAR phones (iPhone 17, 17 Air) still get a
    /// metric sparse point cloud to seed COLMAP / Gaussian splatting.
    private var featurePoints: [UInt64: SIMD3<Float>] = [:]   // queue only
    private let createdAt = Date()

    init(root: URL, settings: CaptureSettings, hasDepth: Bool, deviceInfo: DeviceInfo) throws {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        directory = root.appendingPathComponent(formatter.string(from: Date()), isDirectory: true)

        self.settings = settings
        self.hasDepth = hasDepth
        self.deviceInfo = deviceInfo

        let fm = FileManager.default
        try fm.createDirectory(at: directory.appendingPathComponent("images"), withIntermediateDirectories: true)
        if hasDepth {
            try fm.createDirectory(at: directory.appendingPathComponent("depth"), withIntermediateDirectories: true)
            try fm.createDirectory(at: directory.appendingPathComponent("confidence"), withIntermediateDirectories: true)
        }
        let jsonlURL = directory.appendingPathComponent("frames.jsonl")
        fm.createFile(atPath: jsonlURL.path, contents: nil)
        jsonl = try FileHandle(forWritingTo: jsonlURL)
    }

    /// Copies what it needs out of the frame synchronously (so ARKit's buffers are released
    /// immediately), then encodes and writes on the background queue. Main thread only.
    func add(frame: ARFrame) {
        let index = frameCount
        let name = String(format: "frame_%05d", index)

        guard let rgb = PixelBufferCopy.copy(frame.capturedImage) else { return }
        frameCount += 1

        var depth: DepthCopy?
        if hasDepth, let sceneDepth = frame.sceneDepth {
            depth = DepthCopy(depth: sceneDepth.depthMap, confidence: sceneDepth.confidenceMap)
        }
        let featureIDs = frame.rawFeaturePoints?.identifiers ?? []
        let featureXYZ = frame.rawFeaturePoints?.points ?? []

        let camera = frame.camera
        let K = camera.intrinsics
        let T = camera.transform
        var record = FrameRecord(
            file_path: "images/\(name).jpg",
            depth_file_path: nil,
            confidence_file_path: nil,
            transform_matrix: (0..<4).map { row in (0..<4).map { col in T[col][row] } },
            fl_x: K[0][0], fl_y: K[1][1], cx: K[2][0], cy: K[2][1],
            w: Int(camera.imageResolution.width), h: Int(camera.imageResolution.height),
            timestamp: frame.timestamp,
            exposure_duration: camera.exposureDuration
        )
        if depth != nil {
            record.depth_file_path = "depth/\(name).png"
            if depth?.confidence != nil { record.confidence_file_path = "confidence/\(name).png" }
        }

        lock.lock(); pending += 1; lock.unlock()
        let quality = settings.jpegQuality
        let dir = directory

        queue.async { [self] in
            autoreleasepool {
                do {
                    try writeJPEG(rgb, quality: quality, to: dir.appendingPathComponent(record.file_path))
                    if let depth, let path = record.depth_file_path {
                        try ImageFiles.writeGray16PNG(depth.millimetres, width: depth.width, height: depth.height,
                                                      to: dir.appendingPathComponent(path))
                        if let conf = depth.confidence, let cpath = record.confidence_file_path {
                            try ImageFiles.writeGray8PNG(conf, width: depth.width, height: depth.height,
                                                         to: dir.appendingPathComponent(cpath))
                        }
                    }
                    records.append(record)
                    for (id, p) in zip(featureIDs, featureXYZ) { featurePoints[id] = p }
                    var line = try JSONEncoder().encode(record)
                    line.append(0x0A)
                    jsonl.write(line)
                } catch {
                    NSLog("PhoneScan: failed to write \(name): \(error)")
                }
            }
            lock.lock(); pending -= 1; lock.unlock()
        }
    }

    /// Waits for queued frames, then writes transforms.json, mesh.ply and metadata.json.
    func finish(mesh: MeshSnapshot?, distance: Float = 0, completion: @escaping (Result<URL, Error>) -> Void) {
        queue.async { [self] in
            do {
                try jsonl.close()
                let sorted = records.sorted { $0.file_path < $1.file_path }
                try writeTransforms(sorted)
                if let mesh, !mesh.isEmpty {
                    try mesh.writePLY(to: directory.appendingPathComponent("mesh.ply"))
                }
                if !featurePoints.isEmpty {
                    try writeFeaturePoints(to: directory.appendingPathComponent("arkit_points.ply"))
                }
                try writeMetadata(frames: sorted.count, meshFaces: mesh?.faceCount ?? 0, distance: distance)
                completion(.success(directory))
            } catch {
                completion(.failure(error))
            }
        }
    }

    // MARK: Writers

    private func writeJPEG(_ buffer: CVPixelBuffer, quality: Double, to url: URL) throws {
        let image = CIImage(cvPixelBuffer: buffer)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let qualityKey = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        let options: [CIImageRepresentationOption: Any] = [qualityKey: quality]
        guard let data = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: options) else {
            throw CaptureError.encodeFailed("JPEG")
        }
        try data.write(to: url)
    }

    private func writeTransforms(_ frames: [FrameRecord]) throws {
        var root: [String: Any] = [
            "camera_model": "OPENCV",
            "k1": 0, "k2": 0, "p1": 0, "p2": 0,
            "orientation_override": "none",
            "source": "Pioneer Home (ARKit)",
            "pose_convention": "camera-to-world, OpenGL camera axes (x right, y up, -z forward), world y-up gravity aligned, metres",
        ]
        if hasDepth {
            root["depth_unit_scale_factor"] = 0.001
        }
        if !featurePoints.isEmpty {
            root["ply_file_path"] = "arkit_points.ply"   // nerfstudio splatfacto uses this as its initial point cloud
        }
        if let first = frames.first {
            root["fl_x"] = first.fl_x
            root["fl_y"] = first.fl_y
            root["cx"] = first.cx
            root["cy"] = first.cy
            root["w"] = first.w
            root["h"] = first.h
        }
        let frameDicts = try frames.map { record -> Any in
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(record))
        }
        root["frames"] = frameDicts
        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("transforms.json"))
    }

    /// Binary PLY of ARKit feature points (world frame, metres).
    private func writeFeaturePoints(to url: URL) throws {
        let points = Array(featurePoints.values)
        var data = Data(("ply\nformat binary_little_endian 1.0\ncomment PhoneScan ARKit feature points, world frame, metres\n"
                         + "element vertex \(points.count)\nproperty float x\nproperty float y\nproperty float z\nend_header\n").utf8)
        data.reserveCapacity(data.count + points.count * 12)
        for p in points {
            for f in [p.x, p.y, p.z] {
                withUnsafeBytes(of: f.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            }
        }
        try data.write(to: url)
    }

    private func writeMetadata(frames: Int, meshFaces: Int, distance: Float) throws {
        struct Metadata: Codable {
            var format_version = 1
            var created_at: String
            var duration_seconds: Double
            var frame_count: Int
            var has_depth: Bool
            var mesh_faces: Int
            var distance_m: Float
            var arkit_feature_points: Int
            var device: DeviceInfo
            var keyframe_min_translation_m: Float
            var keyframe_min_rotation_deg: Float
        }
        let iso = ISO8601DateFormatter()
        let meta = Metadata(
            created_at: iso.string(from: createdAt),
            duration_seconds: Date().timeIntervalSince(createdAt),
            frame_count: frames,
            has_depth: hasDepth,
            mesh_faces: meshFaces,
            distance_m: distance,
            arkit_feature_points: featurePoints.count,
            device: deviceInfo,
            keyframe_min_translation_m: settings.minTranslation,
            keyframe_min_rotation_deg: settings.minRotation * 180 / .pi
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(meta).write(to: directory.appendingPathComponent("metadata.json"))
    }
}

enum CaptureError: LocalizedError {
    case encodeFailed(String)
    var errorDescription: String? {
        switch self {
        case .encodeFailed(let what): return "Could not encode \(what)"
        }
    }
}

// MARK: - Buffer copies

enum PixelBufferCopy {
    /// Deep-copies a (possibly planar) pixel buffer so the ARKit-owned original can be released.
    static func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary

        var dstOut: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &dstOut) == kCVReturnSuccess,
              let dst = dstOut else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        let planes = CVPixelBufferGetPlaneCount(src)
        if planes == 0 {
            guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            copyRows(s, CVPixelBufferGetBytesPerRow(src), d, CVPixelBufferGetBytesPerRow(dst), height)
        } else {
            for plane in 0..<planes {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, plane) else { return nil }
                copyRows(s, CVPixelBufferGetBytesPerRowOfPlane(src, plane),
                         d, CVPixelBufferGetBytesPerRowOfPlane(dst, plane),
                         CVPixelBufferGetHeightOfPlane(src, plane))
            }
        }
        return dst
    }

    private static func copyRows(_ src: UnsafeMutableRawPointer, _ srcStride: Int,
                                 _ dst: UnsafeMutableRawPointer, _ dstStride: Int, _ rows: Int) {
        if srcStride == dstStride {
            memcpy(dst, src, srcStride * rows)
        } else {
            let rowBytes = min(srcStride, dstStride)
            for row in 0..<rows {
                memcpy(dst + row * dstStride, src + row * srcStride, rowBytes)
            }
        }
    }
}

/// LiDAR depth converted to millimetres (UInt16, 0 = invalid) plus the raw confidence map.
struct DepthCopy {
    let width: Int
    let height: Int
    let millimetres: [UInt16]
    let confidence: [UInt8]?

    init?(depth: CVPixelBuffer, confidence conf: CVPixelBuffer?) {
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else { return nil }
        width = CVPixelBufferGetWidth(depth)
        height = CVPixelBufferGetHeight(depth)

        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(depth)

        var mm = [UInt16](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = (base + y * stride).assumingMemoryBound(to: Float32.self)
            for x in 0..<width {
                let metres = row[x]
                if metres.isFinite && metres > 0 {
                    mm[y * width + x] = UInt16(min(65535, (metres * 1000).rounded()))
                }
            }
        }
        millimetres = mm

        if let conf, CVPixelBufferGetWidth(conf) == width, CVPixelBufferGetHeight(conf) == height {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
            if let cbase = CVPixelBufferGetBaseAddress(conf) {
                let cstride = CVPixelBufferGetBytesPerRow(conf)
                var c = [UInt8](repeating: 0, count: width * height)
                for y in 0..<height {
                    let row = (cbase + y * cstride).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<width { c[y * width + x] = row[x] }
                }
                confidence = c
            } else {
                confidence = nil
            }
        } else {
            confidence = nil
        }
    }
}

// MARK: - PNG output

enum ImageFiles {
    static func writeGray16PNG(_ pixels: [UInt16], width: Int, height: Int, to url: URL) throws {
        let data = pixels.withUnsafeBufferPointer { Data(buffer: $0) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).union(.byteOrder16Little)
        try writeGrayPNG(data, width: width, height: height, bits: 16, info: info, to: url)
    }

    static func writeGray8PNG(_ pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        try writeGrayPNG(Data(pixels), width: width, height: height, bits: 8, info: info, to: url)
    }

    private static func writeGrayPNG(_ data: Data, width: Int, height: Int, bits: Int,
                                     info: CGBitmapInfo, to url: URL) throws {
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height,
                                  bitsPerComponent: bits, bitsPerPixel: bits,
                                  bytesPerRow: width * bits / 8,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: info,
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CaptureError.encodeFailed("PNG") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CaptureError.encodeFailed("PNG") }
    }
}
