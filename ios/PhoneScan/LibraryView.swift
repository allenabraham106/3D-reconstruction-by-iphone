import ImageIO
import SwiftUI
import UIKit

struct LibraryView: View {
    @EnvironmentObject private var library: CaptureLibrary
    @Environment(\.dismiss) private var dismiss
    @AppStorage(CaptureSettings.Key.serverURL) private var serverURL = "http://192.168.1.100:8765"

    @State private var shareItem: ShareItem?
    @State private var pendingDelete: CaptureItem?

    var body: some View {
        NavigationStack {
            Group {
                if library.items.isEmpty {
                    ContentUnavailableView {
                        Label("No scans yet", systemImage: "cube.transparent")
                    } description: {
                        Text("Tap the red button and walk slowly around a room. Your scans show up here.")
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            if let status = library.status {
                                StatusBanner(text: status, busy: library.busyItem != nil)
                            }
                            ForEach(library.items) { item in
                                CaptureCard(item: item,
                                            busy: library.busyItem == item.url,
                                            disabled: library.busyItem != nil,
                                            share: { share(item) },
                                            upload: { Task { await library.upload(item, server: serverURL) } },
                                            delete: { pendingDelete = item })
                            }
                            Text("Also in Files → On My iPhone → Pioneer Home → Captures")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .padding(.top, 4)
                        }
                        .padding(16)
                    }
                }
            }
            .background(Theme.navy.ignoresSafeArea())
            .navigationTitle("Captures")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { library.refresh() }
            .refreshable { library.refresh() }
            .sheet(item: $shareItem) { item in ActivityView(items: [item.url]) }
            .confirmationDialog("Delete this scan?", isPresented: Binding(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
            ), titleVisibility: .visible, presenting: pendingDelete) { item in
                Button("Delete \(item.frameCount) frames", role: .destructive) { library.delete(item) }
            } message: { item in
                Text("\(item.name) will be removed from this iPhone. Copies you've shared or uploaded are kept.")
            }
        }
    }

    private func share(_ item: CaptureItem) {
        Task {
            if let zip = try? await library.makeZip(item) { shareItem = ShareItem(url: zip) }
        }
    }
}

struct CaptureCard: View {
    let item: CaptureItem
    let busy: Bool
    let disabled: Bool
    let share: () -> Void
    let upload: () -> Void
    let delete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Thumbnail(url: item.thumbnailURL)
                    .frame(width: 72, height: 96)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(item.date, format: .dateTime.weekday(.abbreviated).month().day().hour().minute())
                            .font(.headline)
                        Spacer()
                        if busy { ProgressView() }
                        Menu {
                            Button { share() } label: { Label("Share / AirDrop", systemImage: "square.and.arrow.up") }
                            Button { upload() } label: { Label("Upload to server", systemImage: "icloud.and.arrow.up") }
                            Button(role: .destructive) { delete() } label: { Label("Delete", systemImage: "trash") }
                        } label: {
                            Image(systemName: "ellipsis.circle").font(.title3)
                        }
                        .disabled(disabled)
                    }
                    Text(item.name).font(.caption).foregroundStyle(.secondary)
                    FlowChips(item: item)
                    if !item.isComplete {
                        Label("Interrupted: frames.jsonl is still usable", systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(Theme.warning)
                    }
                }
            }
            HStack(spacing: 10) {
                Button(action: share) {
                    Label("Share", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                Button(action: upload) {
                    Label("Upload", systemImage: "icloud.and.arrow.up").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.primary)
            }
            .disabled(disabled)
        }
        .padding(12)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.teal.opacity(0.15), lineWidth: 1))
        .contextMenu {
            Button { share() } label: { Label("Share / AirDrop", systemImage: "square.and.arrow.up") }
            Button(role: .destructive) { delete() } label: { Label("Delete", systemImage: "trash") }
        }
    }
}

struct FlowChips: View {
    let item: CaptureItem
    var body: some View {
        HStack(spacing: 6) {
            chip("\(item.frameCount)", "photo.stack")
            if let d = item.durationSeconds { chip(String(format: "%d:%02d", Int(d) / 60, Int(d) % 60), "clock") }
            if let m = item.distanceMeters { chip(String(format: "%.1f m", m), "figure.walk") }
            chip(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file), "internaldrive")
        }
        .padding(.top, 2)
        if item.hasDepth || item.hasMesh {
            HStack(spacing: 6) {
                if item.hasDepth { chip("depth", "cube.transparent") }
                if item.hasMesh { chip("mesh", "square.grid.3x3") }
            }
        }
    }

    private func chip(_ text: String, _ symbol: String) -> some View {
        Label(text, systemImage: symbol)
            .font(.caption2.weight(.semibold))
            .labelStyle(.titleAndIcon)
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Theme.navy.opacity(0.8), in: Capsule())
    }
}

struct StatusBanner: View {
    let text: String
    let busy: Bool
    var body: some View {
        HStack(spacing: 10) {
            if busy { ProgressView() } else { Image(systemName: "info.circle").foregroundStyle(.secondary) }
            Text(text).font(.footnote.weight(.medium)).frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

/// Small, downsampled thumbnail of the first keyframe, rotated to how the phone was held (portrait).
struct Thumbnail: View {
    let url: URL?
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color(.tertiarySystemFill)
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "photo").foregroundStyle(.secondary)
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
