import Foundation

struct CaptureItem: Identifiable, Hashable {
    var id: URL { url }
    let url: URL
    let name: String
    let date: Date
    let frameCount: Int
    let sizeBytes: Int64
    let hasDepth: Bool
    let hasMesh: Bool
    let isComplete: Bool   // transforms.json written
    let durationSeconds: Double?
    let distanceMeters: Double?
    /// First keyframe, for the thumbnail.
    let thumbnailURL: URL?
}

/// Lists captures in Documents/Captures (also visible in the Files app under "On My iPhone → PhoneScan"),
/// zips them for sharing, and uploads them to a reconstruction server.
@MainActor
final class CaptureLibrary: ObservableObject {
    @Published private(set) var items: [CaptureItem] = []
    @Published var busyItem: URL?
    @Published var status: String?

    nonisolated static var capturesDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Captures", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func refresh() {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: Self.capturesDirectory,
                                                includingPropertiesForKeys: [.creationDateKey, .isDirectoryKey],
                                                options: [.skipsHiddenFiles])) ?? []
        items = dirs.compactMap { url -> CaptureItem? in
            let values = try? url.resourceValues(forKeys: [.creationDateKey, .isDirectoryKey])
            guard values?.isDirectory == true else { return nil }
            let images = ((try? fm.contentsOfDirectory(atPath: url.appendingPathComponent("images").path)) ?? [])
                .filter { $0.hasSuffix(".jpg") }.sorted()
            let meta = (try? Data(contentsOf: url.appendingPathComponent("metadata.json")))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
            return CaptureItem(
                url: url,
                name: url.lastPathComponent,
                date: values?.creationDate ?? .distantPast,
                frameCount: images.count,
                sizeBytes: Self.directorySize(url),
                hasDepth: fm.fileExists(atPath: url.appendingPathComponent("depth").path),
                hasMesh: fm.fileExists(atPath: url.appendingPathComponent("mesh.ply").path),
                isComplete: fm.fileExists(atPath: url.appendingPathComponent("transforms.json").path),
                durationSeconds: meta?["duration_seconds"] as? Double,
                distanceMeters: meta?["distance_m"] as? Double,
                thumbnailURL: images.first.map { url.appendingPathComponent("images").appendingPathComponent($0) }
            )
        }
        .sorted { $0.date > $1.date }
    }

    func delete(_ item: CaptureItem) {
        try? FileManager.default.removeItem(at: item.url)
        refresh()
    }

    /// Zips a capture folder using the system's built-in archiver (NSFileCoordinator .forUploading).
    func makeZip(_ item: CaptureItem) async throws -> URL {
        busyItem = item.url
        status = "Zipping \(item.name)…"
        defer { busyItem = nil }
        let source = item.url
        let zip = try await Task.detached(priority: .userInitiated) {
            try Self.zipDirectory(source)
        }.value
        status = nil
        return zip
    }

    /// POSTs the zipped capture to `<server>/upload` (see tools/receive_server.py).
    func upload(_ item: CaptureItem, server: String) async {
        let trimmed = server.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let base = URL(string: trimmed), base.scheme != nil else {
            status = "Invalid server URL: \(server)"
            return
        }
        do {
            let zip = try await makeZip(item)
            busyItem = item.url
            defer { busyItem = nil }
            let size = (try? FileManager.default.attributesOfItem(atPath: zip.path)[.size] as? Int64) ?? 0
            status = "Uploading \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))…"

            var request = URLRequest(url: base.appendingPathComponent("upload"))
            request.httpMethod = "POST"
            request.timeoutInterval = 900
            request.setValue("application/zip", forHTTPHeaderField: "Content-Type")
            request.setValue(item.name, forHTTPHeaderField: "X-Capture-Name")

            let (data, response) = try await URLSession.shared.upload(for: request, fromFile: zip)
            try? FileManager.default.removeItem(at: zip)
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? ""
            status = code == 200 ? "Uploaded \(item.name) ✓" : "Server returned \(code): \(body.prefix(200))"
        } catch {
            status = "Upload failed: \(error.localizedDescription)"
        }
    }

    nonisolated private static func zipDirectory(_ dir: URL) throws -> URL {
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent(dir.lastPathComponent + ".zip")
        try? FileManager.default.removeItem(at: dest)

        var coordinatorError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading], error: &coordinatorError) { zipped in
            do { try FileManager.default.copyItem(at: zipped, to: dest) } catch { copyError = error }
        }
        if let coordinatorError { throw coordinatorError }
        if let copyError { throw copyError }
        return dest
    }

    nonisolated private static func directorySize(_ url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let file as URL in e {
            total += Int64((try? file.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize) ?? 0)
        }
        return total
    }
}
