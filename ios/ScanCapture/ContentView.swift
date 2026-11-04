import ARKit
import RealityKit
import SwiftUI

@main
struct ScanCaptureApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct ContentView: View {
    @StateObject private var capture = CaptureSession()
    @State private var showShare = false

    var body: some View {
        ZStack {
            ARPreview(session: capture.session).ignoresSafeArea()
            VStack(spacing: 12) {
                HStack {
                    Label(capture.trackingText, systemImage: "location.viewfinder")
                    Spacer()
                    Label(capture.hasLiDAR ? "LiDAR" : "No LiDAR",
                          systemImage: capture.hasLiDAR ? "sensor.tag.radiowaves.forward" : "camera")
                }
                .font(.footnote.weight(.semibold))
                .padding(10)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                Spacer()
                if capture.isRecording || capture.savedFrames > 0 {
                    Text("\(capture.savedFrames) frames · \(String(format: "%.1f", capture.distanceWalked)) m walked")
                        .font(.callout.monospacedDigit())
                        .padding(8)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                HStack(spacing: 24) {
                    if let url = capture.lastScanURL, !capture.isRecording {
                        ShareLink(item: url) {
                            Image(systemName: "square.and.arrow.up").font(.title2)
                                .frame(width: 56, height: 56)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                    }
                    Button(action: capture.toggleRecording) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 76, height: 76)
                            RoundedRectangle(cornerRadius: capture.isRecording ? 6 : 30)
                                .fill(.red)
                                .frame(width: capture.isRecording ? 30 : 60, height: capture.isRecording ? 30 : 60)
                        }
                    }
                    .accessibilityLabel(capture.isRecording ? "Stop recording" : "Start recording")
                }
                Text("Walk slowly. Keep the phone moving sideways, not just turning. Cover the floor and walls.")
                    .font(.caption).multilineTextAlignment(.center)
                    .padding(8).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
            .padding()
        }
        .onAppear { capture.start() }
        .onDisappear { capture.pause() }
    }
}

/// RealityKit view sharing our ARSession; shows the live LiDAR mesh while scanning.
struct ARPreview: UIViewRepresentable {
    let session: ARSession
    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.session = session
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            view.debugOptions.insert(.showSceneUnderstanding)
        } else {
            view.debugOptions.insert(.showFeaturePoints)
        }
        return view
    }
    func updateUIView(_ uiView: ARView, context: Context) {}
}
