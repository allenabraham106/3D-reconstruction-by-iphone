import SwiftUI

struct SettingsView: View {
    @Binding var showOnboarding: Bool
    @EnvironmentObject private var capture: CaptureManager

    @AppStorage(CaptureSettings.Key.minTranslationCm) private var minTranslationCm = 5.0
    @AppStorage(CaptureSettings.Key.minRotationDeg) private var minRotationDeg = 5.0
    @AppStorage(CaptureSettings.Key.minIntervalSec) private var minIntervalSec = 0.2
    @AppStorage(CaptureSettings.Key.jpegQuality) private var jpegQuality = 0.9
    @AppStorage(CaptureSettings.Key.saveDepth) private var saveDepth = true
    @AppStorage(CaptureSettings.Key.saveMesh) private var saveMesh = true
    @AppStorage(CaptureSettings.Key.showMesh) private var showMesh = true
    @AppStorage(CaptureSettings.Key.showFeaturePoints) private var showFeaturePoints = true
    @AppStorage(CaptureSettings.Key.use4K) private var use4K = false
    @AppStorage(CaptureSettings.Key.serverURL) private var serverURL = "http://192.168.1.100:8765"

    @State private var showAdvanced = false

    private var currentPreset: CaptureSettings.Preset? {
        CaptureSettings.Preset.allCases.first {
            let v = $0.values
            return v.0 == minTranslationCm && v.1 == minRotationDeg && abs(v.2 - minIntervalSec) < 0.001
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Detail", selection: Binding(
                        get: { currentPreset },
                        set: { if let p = $0 { apply(p) } }
                    )) {
                        ForEach(CaptureSettings.Preset.allCases) { Text($0.rawValue).tag(Optional($0)) }
                        if currentPreset == nil { Text("Custom").tag(Optional<CaptureSettings.Preset>.none) }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Scan detail")
                } footer: {
                    Text(currentPreset?.blurb ?? "Custom keyframe spacing: \(Int(minTranslationCm)) cm / \(Int(minRotationDeg))°.")
                }

                Section {
                    Toggle(isOn: $use4K) {
                        Label("4K photos", systemImage: "4k.tv")
                    }
                } footer: {
                    Text("3840×2880 instead of 1920×1440. Sharper models, about 4× the storage.")
                }

                Section {
                    if capture.supportsMesh {
                        Toggle(isOn: $showMesh) { Label("Live mesh overlay", systemImage: "square.grid.3x3") }
                    }
                    Toggle(isOn: $showFeaturePoints) { Label("Tracking dots", systemImage: "circle.dotted") }
                } header: {
                    Text("While scanning")
                } footer: {
                    Text(capture.hasLiDAR
                         ? "The mesh shows what's covered. Tracking dots show the points ARKit locks onto."
                         : "Tracking dots show the points ARKit locks onto. More dots means better tracking.")
                }

                Section {
                    TextField("http://192.168.1.100:8765", text: $serverURL)
                        .font(Theme.mono(13))
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Upload server")
                } footer: {
                    Text("On your computer run `python3 tools/receive_server.py`, then enter its address. Same Wi-Fi or Tailscale.")
                }

                Section {
                    Button {
                        showOnboarding = true
                    } label: {
                        Label("Show scanning tips", systemImage: "questionmark.circle")
                    }
                }

                Section {
                    DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                        sliderRow("Move at least", value: $minTranslationCm, range: 1...30, step: 1, unit: "cm")
                        sliderRow("…or turn at least", value: $minRotationDeg, range: 1...30, step: 1, unit: "°")
                        sliderRow("Min time between frames", value: $minIntervalSec, range: 0.05...1.0, step: 0.05, unit: "s")
                        sliderRow("JPEG quality", value: $jpegQuality, range: 0.6...1.0, step: 0.05, unit: "")
                        if capture.hasLiDAR {
                            Toggle("Save LiDAR depth + confidence", isOn: $saveDepth)
                            Toggle("Save ARKit mesh (mesh.ply)", isOn: $saveMesh)
                        }
                    }
                } footer: {
                    Text(capture.hasLiDAR ? "This iPhone has LiDAR." : "This iPhone has no LiDAR (only Pro models do).")
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.paper.ignoresSafeArea())
            .navigationTitle("Settings")
        }
    }

    private func apply(_ preset: CaptureSettings.Preset) {
        let v = preset.values
        minTranslationCm = v.0
        minRotationDeg = v.1
        minIntervalSec = v.2
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
