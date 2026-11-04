import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var capture: CaptureManager

    @AppStorage(CaptureSettings.Key.minTranslationCm) private var minTranslationCm = 5.0
    @AppStorage(CaptureSettings.Key.minRotationDeg) private var minRotationDeg = 5.0
    @AppStorage(CaptureSettings.Key.minIntervalSec) private var minIntervalSec = 0.2
    @AppStorage(CaptureSettings.Key.jpegQuality) private var jpegQuality = 0.9
    @AppStorage(CaptureSettings.Key.saveDepth) private var saveDepth = true
    @AppStorage(CaptureSettings.Key.saveMesh) private var saveMesh = true
    @AppStorage(CaptureSettings.Key.showMesh) private var showMesh = true
    @AppStorage(CaptureSettings.Key.use4K) private var use4K = false
    @AppStorage(CaptureSettings.Key.serverURL) private var serverURL = "http://192.168.1.100:8765"

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    sliderRow("Move at least", value: $minTranslationCm, range: 1...30, step: 1, unit: "cm")
                    sliderRow("…or turn at least", value: $minRotationDeg, range: 1...30, step: 1, unit: "°")
                    sliderRow("Min time between frames", value: $minIntervalSec, range: 0.05...1.0, step: 0.05, unit: "s")
                } header: {
                    Text("Keyframes")
                } footer: {
                    Text("A frame is saved whenever the phone has moved or turned enough. Smaller values give more overlap, which helps SfM and Gaussian splatting.")
                }

                Section("Image") {
                    sliderRow("JPEG quality", value: $jpegQuality, range: 0.6...1.0, step: 0.05, unit: "")
                    Toggle("4K video format (3840×2880)", isOn: $use4K)
                }

                Section {
                    Toggle("Save LiDAR depth + confidence", isOn: $saveDepth)
                    Toggle("Save ARKit mesh (mesh.ply)", isOn: $saveMesh)
                    Toggle("Show live mesh overlay", isOn: $showMesh)
                } header: {
                    Text("LiDAR")
                } footer: {
                    Text(capture.hasLiDAR
                         ? "This iPhone has LiDAR."
                         : "This iPhone has no LiDAR (only Pro models do). These options are ignored.")
                }

                Section {
                    TextField("http://192.168.1.100:8765", text: $serverURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Upload server")
                } footer: {
                    Text("Run `python3 tools/receive_server.py` on your laptop or the robot's computer, then enter its address. Same Wi-Fi or Tailscale.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }

    private func sliderRow(_ title: String, value: Binding<Double>, range: ClosedRange<Double>,
                           step: Double, unit: String) -> some View {
        VStack(alignment: .leading) {
            HStack {
                Text(title)
                Spacer()
                Text(step < 1 ? String(format: "%.2f %@", value.wrappedValue, unit)
                              : String(format: "%.0f %@", value.wrappedValue, unit))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
        }
    }
}
