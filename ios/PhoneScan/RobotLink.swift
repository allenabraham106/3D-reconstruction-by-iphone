import Foundation
import simd
import UIKit

/// Talks to robot_link/server.py on the robot (default http://pioneer.local:8770).
@MainActor
final class RobotLink: ObservableObject {
    struct Grid: Codable, Equatable {
        let resolution: Double
        let origin: [Double]
        let width: Int
        let height: Int
    }

    struct Job: Codable, Equatable {
        let state: String
        let name: String?
        let error: String?
    }

    struct Pose: Codable, Equatable {
        let ok: Bool
        let reason: String
        let x: Double
        let y: Double
        let z: Double
        let yaw_deg: Double
        let inliers: Int
        let matches: Int
        let grid_px: [Double]?
        let source: String?
        let seconds: Double?
    }

    /// The robot's own 3D model, if its robot link was started with --robot-urdf.
    struct RobotInfo: Codable, Equatable {
        let name: String?
        let height: Double?
        let joints: Int?
        let id: String
    }

    /// Whether the robot's camera can be watched live (/camera/stream).
    struct Live: Codable, Equatable {
        let available: Bool
        let viewers: Int?
        let fps: Double?
    }

    struct Status: Codable, Equatable {
        let name: String
        let robot: RobotInfo?
        let live: Live?
        let map: String?
        let maps: [String]
        let camera: String
        let ros: Bool
        let grid: Grid?
        let last: [String: Pose]
        let job: Job
    }

    enum Connection: Equatable { case unknown, connecting, connected, failed(String) }

    @Published private(set) var connection: Connection = .unknown
    @Published private(set) var status: Status?
    @Published private(set) var mapImage: UIImage?
    /// Photo-like greyscale top-down view of the scan (same extent as the grid).
    @Published private(set) var topdownImage: UIImage?
    /// The coloured 3D scan (ceiling removed) and the route it was scanned along, for the 3D map.
    @Published private(set) var cloud: PointCloud?
    @Published private(set) var scanPath: [SIMD3<Float>] = []
    @Published private(set) var loadingMap = false
    /// The connected robot's model (.robot bytes) and its id; nil means "draw the built-in Pioneer".
    @Published private(set) var robotModel: (id: String, data: Data)?
    @Published private(set) var lastResult: Pose?
    @Published private(set) var busy: String?
    @Published var error: String?

    private var mapImageFor: String?

    var isConnected: Bool { connection == .connected }

    private func base(_ urlString: String) -> URL? {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme != nil else { return nil }
        return url
    }

    /// Fetches /status (and the floor map when the active map changes).
    func refresh(_ urlString: String, quiet: Bool = false) async {
        guard let url = base(urlString) else {
            connection = .failed("Invalid address: \(urlString)")
            return
        }
        if !quiet { connection = .connecting }
        do {
            var req = URLRequest(url: url.appendingPathComponent("status"))
            req.timeoutInterval = 4
            let (data, _) = try await URLSession.shared.data(for: req)
            let st = try JSONDecoder().decode(Status.self, from: data)
            status = st
            connection = .connected
            if st.map != mapImageFor { await loadMapImage(url, map: st.map) }
            if st.robot?.id != robotModel?.id { await loadRobotModel(url, info: st.robot) }
        } catch {
            connection = .failed(quiet ? "Lost connection" : "Can't reach \(url.host ?? urlString). Is the robot link running and on the same Wi-Fi?")
        }
    }

    private func loadMapImage(_ url: URL, map: String?) async {
        mapImageFor = map
        guard map != nil else { mapImage = nil; topdownImage = nil; cloud = nil; scanPath = []; return }
        loadingMap = true
        defer { loadingMap = false }
        var cloudReq = URLRequest(url: url.appendingPathComponent("map/cloud.ply"))
        cloudReq.timeoutInterval = 60
        if let (data, resp) = try? await URLSession.shared.data(for: cloudReq),
           (resp as? HTTPURLResponse)?.statusCode == 200 {
            cloud = await Task.detached(priority: .userInitiated) { PointCloud.parsePLY(data) }.value
        } else {
            cloud = nil
        }
        struct PathResponse: Decodable { let points: [[Float]] }
        if let (data, _) = try? await URLSession.shared.data(from: url.appendingPathComponent("map/path.json")),
           let r = try? JSONDecoder().decode(PathResponse.self, from: data) {
            scanPath = r.points.compactMap { $0.count == 3 ? SIMD3($0[0], $0[1], $0[2]) : nil }
        }
        if var comps = URLComponents(url: url.appendingPathComponent("map/occupancy.png"), resolvingAgainstBaseURL: false) {
            comps.queryItems = [URLQueryItem(name: "style", value: "mono")]
            if let u = comps.url, let (data, _) = try? await URLSession.shared.data(from: u) {
                mapImage = UIImage(data: data)
            }
        }
        if let (data, _) = try? await URLSession.shared.data(from: url.appendingPathComponent("map/topdown.png")) {
            topdownImage = UIImage(data: data)
        }
    }

    /// Downloads the robot's model once per id and keeps it in Caches, so reconnecting is instant.
    private func loadRobotModel(_ url: URL, info: RobotInfo?) async {
        guard let info else { robotModel = nil; return }
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RobotModels", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("\(info.id).robot")
        if let data = try? Data(contentsOf: file) {
            robotModel = (info.id, data)
            return
        }
        var req = URLRequest(url: url.appendingPathComponent("robot/model"))
        req.timeoutInterval = 60
        if let (data, resp) = try? await URLSession.shared.data(for: req),
           (resp as? HTTPURLResponse)?.statusCode == 200, data.prefix(4) == Data("PRBT".utf8) {
            try? data.write(to: file)
            robotModel = (info.id, data)
        }
    }

    func activate(_ urlString: String, map: String) async {
        guard let url = base(urlString),
              var comps = URLComponents(url: url.appendingPathComponent("maps/activate"), resolvingAgainstBaseURL: false)
        else { return }
        comps.queryItems = [URLQueryItem(name: "name", value: map)]
        guard let target = comps.url else { return }
        var req = URLRequest(url: target)
        req.httpMethod = "POST"
        _ = try? await URLSession.shared.data(for: req)
        await refresh(urlString, quiet: true)
    }

    /// Asks the robot to localize itself with its own camera.
    func localizeRobot(_ urlString: String) async {
        await localize(urlString, source: "robot", body: nil, headers: [:])
    }

    /// Sends the phone's current camera view to the robot link and localizes the phone.
    func localizePhone(_ urlString: String, snapshot: (jpeg: Data, fx: Float, fy: Float, cx: Float, cy: Float)) async {
        let headers = ["Content-Type": "image/jpeg",
                       "X-Fx": "\(snapshot.fx)", "X-Fy": "\(snapshot.fy)",
                       "X-Cx": "\(snapshot.cx)", "X-Cy": "\(snapshot.cy)"]
        await localize(urlString, source: "phone", body: snapshot.jpeg, headers: headers)
    }

    private func localize(_ urlString: String, source: String, body: Data?, headers: [String: String]) async {
        guard let url = base(urlString),
              var comps = URLComponents(url: url.appendingPathComponent("localize"), resolvingAgainstBaseURL: false)
        else { return }
        comps.queryItems = [URLQueryItem(name: "source", value: source)]
        guard let target = comps.url else { return }
        busy = source == "robot" ? "Localizing robot…" : "Localizing phone…"
        defer { busy = nil }
        var req = URLRequest(url: target)
        req.httpMethod = "POST"
        req.timeoutInterval = 30
        headers.forEach { req.setValue($1, forHTTPHeaderField: $0) }
        do {
            let (data, _) = try await URLSession.shared.upload(for: req, from: body ?? Data())
            if let pose = try? JSONDecoder().decode(Pose.self, from: data) {
                lastResult = pose
            } else if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let reason = obj["reason"] as? String {
                error = reason
            }
            await refresh(urlString, quiet: true)
        } catch {
            self.error = "Localize failed: \(error.localizedDescription)"
        }
    }

    /// Sends a scan to the robot, which reconstructs it and builds a relocalization map.
    func sendScan(_ urlString: String, zip: URL, name: String) async {
        guard let url = base(urlString) else { return }
        busy = "Sending \(name)…"
        defer { busy = nil }
        var req = URLRequest(url: url.appendingPathComponent("upload"))
        req.httpMethod = "POST"
        req.timeoutInterval = 900
        req.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        req.setValue(name, forHTTPHeaderField: "X-Capture-Name")
        do {
            _ = try await URLSession.shared.upload(for: req, fromFile: zip)
            await refresh(urlString, quiet: true)
        } catch {
            self.error = "Upload failed: \(error.localizedDescription)"
        }
    }
}
