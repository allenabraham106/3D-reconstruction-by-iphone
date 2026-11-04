import SwiftUI
import UIKit

struct LibraryView: View {
    @EnvironmentObject private var library: CaptureLibrary
    @Environment(\.dismiss) private var dismiss
    @AppStorage(CaptureSettings.Key.serverURL) private var serverURL = "http://192.168.1.100:8765"

    @State private var shareItem: ShareItem?

    var body: some View {
        NavigationStack {
            List {
                if let status = library.status {
                    Section { Text(status).font(.footnote) }
                }
                Section {
                    if library.items.isEmpty {
                        Text("No captures yet. Tap the red button to record a scan.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(library.items) { item in
                        row(item)
                    }
                    .onDelete { offsets in
                        for i in offsets { library.delete(library.items[i]) }
                    }
                } footer: {
                    Text("Captures are also in the Files app → On My iPhone → PhoneScan → Captures. Upload server: \(serverURL)")
                }
            }
            .navigationTitle("Captures")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { library.refresh() }
            .refreshable { library.refresh() }
            .sheet(item: $shareItem) { item in
                ActivityView(items: [item.url])
            }
        }
    }

    private func row(_ item: CaptureItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(item.name).font(.headline)
                if !item.isComplete {
                    Text("incomplete").font(.caption).foregroundStyle(.orange)
                }
                Spacer()
                if library.busyItem == item.url { ProgressView() }
            }
            HStack(spacing: 12) {
                Label("\(item.frameCount)", systemImage: "photo.stack")
                if item.hasDepth { Label("depth", systemImage: "cube.transparent") }
                if item.hasMesh { Label("mesh", systemImage: "square.grid.3x3") }
                Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            HStack {
                Button("Share / AirDrop") {
                    Task {
                        if let zip = try? await library.makeZip(item) { shareItem = ShareItem(url: zip) }
                    }
                }
                Spacer()
                Button("Upload to server") {
                    Task { await library.upload(item, server: serverURL) }
                }
            }
            .buttonStyle(.bordered)
            .disabled(library.busyItem != nil)
        }
        .padding(.vertical, 4)
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
