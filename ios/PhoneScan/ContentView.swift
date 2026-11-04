import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var capture: CaptureManager
    @EnvironmentObject private var library: CaptureLibrary
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage(CaptureSettings.Key.showMesh) private var showMesh = true

    @State private var showLibrary = false
    @State private var showSettings = false

    var body: some View {
        ZStack {
            ARViewContainer(session: capture.session, showMesh: showMesh && capture.supportsMesh)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                statusBar
                if !capture.hasLiDAR {
                    Banner(text: "No LiDAR on this iPhone (e.g. iPhone 17): saving photos, ARKit poses and feature points. Depth and mesh need a Pro model.")
                }
                if let message = capture.message {
                    Banner(text: message)
                        .onTapGesture { capture.message = nil }
                }
                Spacer()
                if capture.isRecording { tips }
                controls
            }
            .padding()
        }
        .onAppear { capture.startSession() }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: capture.startSession()
            case .background: capture.pauseSession()
            default: break
            }
        }
        .sheet(isPresented: $showLibrary) {
            LibraryView()
        }
        .sheet(isPresented: $showSettings, onDismiss: {
            if !capture.isRecording { capture.startSession() }
        }) {
            SettingsView()
        }
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(capture.trackingOK ? Color.green : Color.orange)
                .frame(width: 10, height: 10)
            Text(capture.trackingText)
                .font(.subheadline.weight(.medium))
            Spacer()
            if capture.isRecording {
                Label("\(capture.frameCount)", systemImage: "photo.stack")
                Text(timeString(capture.elapsedSeconds)).monospacedDigit()
                Text(String(format: "%.1f m", capture.distanceMeters)).monospacedDigit()
            } else if capture.hasLiDAR {
                Label("LiDAR", systemImage: "dot.radiowaves.left.and.right")
            }
        }
        .font(.subheadline)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var tips: some View {
        Text("Walk slowly · keep 0.5–3 m from surfaces · sweep up and down · revisit your start point to close the loop")
            .font(.footnote)
            .multilineTextAlignment(.center)
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var controls: some View {
        HStack {
            Button { showLibrary = true } label: {
                Image(systemName: "folder").font(.title2).frame(width: 56, height: 56)
            }
            .background(.ultraThinMaterial, in: Circle())
            .disabled(capture.isRecording)

            Spacer()

            Button(action: toggleRecording) {
                ZStack {
                    Circle().stroke(Color.white, lineWidth: 4).frame(width: 78, height: 78)
                    if capture.isFinishing {
                        ProgressView().tint(.white)
                    } else if capture.isRecording {
                        RoundedRectangle(cornerRadius: 6).fill(Color.red).frame(width: 32, height: 32)
                    } else {
                        Circle().fill(Color.red).frame(width: 64, height: 64)
                    }
                }
            }
            .disabled(capture.isFinishing)
            .accessibilityLabel(capture.isRecording ? "Stop recording" : "Start recording")

            Spacer()

            Button { showSettings = true } label: {
                Image(systemName: "gearshape").font(.title2).frame(width: 56, height: 56)
            }
            .background(.ultraThinMaterial, in: Circle())
            .disabled(capture.isRecording)
        }
        .foregroundStyle(.white)
    }

    private func toggleRecording() {
        if capture.isRecording {
            capture.stopRecording { _ in library.refresh() }
        } else {
            capture.startRecording()
        }
    }

    private func timeString(_ s: Int) -> String {
        String(format: "%d:%02d", s / 60, s % 60)
    }
}

struct Banner: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.footnote)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}
