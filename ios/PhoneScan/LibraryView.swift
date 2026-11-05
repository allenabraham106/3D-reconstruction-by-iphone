import ImageIO
import SwiftUI
import UIKit

struct LibraryView: View {
    @EnvironmentObject private var library: CaptureLibrary
    @EnvironmentObject private var robot: RobotLink
    @AppStorage(CaptureSettings.Key.serverURL) private var serverURL = "http://192.168.1.100:8765"
    @AppStorage(CaptureSettings.Key.robotURL) private var robotURL = "http://pioneer.local:8770"

    enum Where: Hashable { case phone, robot }
    @State private var showing: Where = .phone
    @State private var shareItem: ShareItem?
    @State private var pendingDelete: CaptureItem?
    @State private var showCapture = false

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("My Scans").font(Theme.display(34)).foregroundStyle(Theme.ink)
                        .padding(.top, 24)
                    Segmented(options: [(Where.phone, "On phone"), (Where.robot, "On Pioneer")], selection: $showing)
                    if let status = library.status ?? robot.busy {
                        HStack(spacing: 8) {
                            if library.busyItem != nil || robot.busy != nil { ProgressView() }
                            Text(status).font(Theme.mono(11)).foregroundStyle(Theme.ink)
                            Spacer(minLength: 0)
                        }
                        .padding(12)
                        .paperCard(10)
                    }
                    if showing == .phone { phoneList } else { robotList }
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 100)
            }
            PillButton(title: "New scan", symbol: "plus") { showCapture = true }
                .padding(.trailing, 18)
                .padding(.bottom, 18)
        }
        .background(Theme.card)
        .fullScreenCover(isPresented: $showCapture) { CaptureView() }
        .onAppear { library.refresh() }
        .refreshable { library.refresh() }
        .sheet(item: $shareItem) { item in ActivityView(items: [item.url]) }
        .confirmationDialog("Delete this scan?", isPresented: Binding(
            get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
        ), titleVisibility: .visible, presenting: pendingDelete) { item in
            Button("Delete \(item.frameCount) frames", role: .destructive) { library.delete(item) }
        } message: { item in
            Text("\(item.name) will be removed from this iPhone. Copies you've shared or sent are kept.")
        }
    }

    @ViewBuilder
    private var phoneList: some View {
        if library.items.isEmpty {
            empty("No scans yet", "Tap NEW SCAN and walk slowly around a room.")
        } else {
            VStack(spacing: 10) {
                ForEach(library.items) { item in
                    ScanRow(item: item, busy: library.busyItem == item.url,
                            canSendToRobot: robot.isConnected,
                            share: { share(item) },
                            sendToRobot: { send(item) },
                            upload: { Task { await library.upload(item, server: serverURL) } },
                            delete: { pendingDelete = item })
                        .paperCard(12)
                        .disabled(library.busyItem != nil)
                }
            }
            Text("Also in Files → On My iPhone → Pioneer Home → Captures").font(.caption).foregroundStyle(Theme.ink2)
        }
    }

    @ViewBuilder
    private var robotList: some View {
        let maps = robot.status?.maps ?? []
        if !robot.isConnected {
            empty("Pioneer is offline", "Connect in the Robot tab to see the maps on Pioneer.")
        } else if maps.isEmpty {
            empty("No maps on Pioneer", "Send a scan from On phone. Pioneer turns it into a map it can find itself in.")
        } else {
            VStack(spacing: 10) {
                ForEach(maps, id: \.self) { m in
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(m).font(.body).foregroundStyle(Theme.ink)
                            Text(m == robot.status?.map ? "Active map · localizes here" : "Saved map").monoLabel(10, color: Theme.ink2)
                        }
                        Spacer()
                        Toggle("", isOn: Binding(
                            get: { robot.status?.map == m },
                            set: { on in if on { Task { await robot.activate(robotURL, map: m) } } }
                        ))
                        .labelsHidden()
                        .tint(Theme.chipGreen)
                    }
                    .padding(14)
                    .paperCard(12)
                }
            }
        }
    }

    private func empty(_ title: String, _ body: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "cube.transparent").font(.system(size: 28, weight: .ultraLight))
            Text(title).monoLabel(11)
            Text(body).font(.footnote).foregroundStyle(Theme.ink2).multilineTextAlignment(.center)
        }
        .foregroundStyle(Theme.ink)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
        .padding(.horizontal, 20)
        .paperCard(12)
    }

    private func share(_ item: CaptureItem) {
        Task {
            if let zip = try? await library.makeZip(item) { shareItem = ShareItem(url: zip) }
        }
    }

    private func send(_ item: CaptureItem) {
        Task {
            if let zip = try? await library.makeZip(item) {
                await robot.sendScan(robotURL, zip: zip, name: item.name)
            }
        }
    }
}

struct ScanRow: View {
    let item: CaptureItem
    let busy: Bool
    let canSendToRobot: Bool
    let share: () -> Void
    let sendToRobot: () -> Void
    let upload: () -> Void
    let delete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Thumbnail(url: item.thumbnailURL)
                .frame(width: 54, height: 72)
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).stroke(Theme.line, lineWidth: 1))
            VStack(alignment: .leading, spacing: 5) {
                Text(item.date, format: .dateTime.weekday(.abbreviated).month().day().hour().minute())
                    .font(.subheadline).foregroundStyle(Theme.ink)
                Text(meta).monoLabel(10, color: Theme.ink2).lineLimit(1)
                HStack(spacing: 5) {
                    if item.hasDepth { Chip(text: "Depth") }
                    if item.hasMesh { Chip(text: "Mesh") }
                    if !item.isComplete { Chip(text: "Interrupted", style: .tint(Theme.warning)) }
                }
            }
            Spacer(minLength: 0)
            if busy {
                ProgressView()
            } else {
                Menu {
                    Button { sendToRobot() } label: { Label("Send to robot", systemImage: "figure.stand") }
                        .disabled(!canSendToRobot)
                    Button { share() } label: { Label("Share / AirDrop", systemImage: "square.and.arrow.up") }
                    Button { upload() } label: { Label("Upload to server", systemImage: "arrow.up.to.line") }
                    Button(role: .destructive) { delete() } label: { Label("Delete", systemImage: "trash") }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(Theme.ink)
                        .frame(width: 34, height: 34)
                        .paperCard(8, fill: Theme.paper)
                }
            }
        }
        .padding(14)
        .contentShape(Rectangle())
        .contextMenu {
            Button { share() } label: { Label("Share / AirDrop", systemImage: "square.and.arrow.up") }
            Button(role: .destructive) { delete() } label: { Label("Delete", systemImage: "trash") }
        }
    }

    private var meta: String {
        var parts = ["\(item.frameCount) frames"]
        if let d = item.durationSeconds { parts.append(String(format: "%d:%02d", Int(d) / 60, Int(d) % 60)) }
        if let m = item.distanceMeters { parts.append(String(format: "%.1f m", m)) }
        parts.append(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
        return parts.joined(separator: " · ")
    }
}

/// Small, downsampled thumbnail of the first keyframe, rotated to how the phone was held (portrait).
struct Thumbnail: View {
    let url: URL?
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Theme.paper
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "photo").foregroundStyle(Theme.ink2)
            }
        }
        .task(id: url) { image = await Self.load(url) }
    }

    static func load(_ url: URL?) async -> UIImage? {
        guard let url else { return nil }
        return await Task.detached(priority: .utility) { () -> UIImage? in
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let opts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 300,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ]
            guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
            // Keyframes are stored in the sensor's landscape orientation; the phone was held upright.
            return UIImage(cgImage: cg, scale: 1, orientation: .right)
        }.value
    }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: URL { url }
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
