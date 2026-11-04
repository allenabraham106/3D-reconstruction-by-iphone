import ARKit
import RealityKit
import SwiftUI

/// Camera preview that renders the shared ARSession, optionally overlaying the live LiDAR mesh.
struct ARViewContainer: UIViewRepresentable {
    let session: ARSession
    var showMesh: Bool

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.session = session
        view.renderOptions.insert(.disableMotionBlur)
        view.renderOptions.insert(.disableDepthOfField)
        applyDebugOptions(view)
        return view
    }

    func updateUIView(_ view: ARView, context: Context) {
        applyDebugOptions(view)
    }

    private func applyDebugOptions(_ view: ARView) {
        if showMesh {
            view.debugOptions.insert(.showSceneUnderstanding)
        } else {
            view.debugOptions.remove(.showSceneUnderstanding)
        }
    }
}
