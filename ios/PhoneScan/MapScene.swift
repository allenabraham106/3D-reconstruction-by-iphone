import SceneKit
import SwiftUI
import UIKit
import simd

/// A coloured point cloud in the robot map frame (metres, Z up, floor at Z = 0).
struct PointCloud {
    var points: [SIMD3<Float>]
    var colors: [SIMD3<Float>]   // 0...1

    /// Parses the binary PLY served by robot_link (`float x y z, uchar r g b` per vertex).
    static func parsePLY(_ data: Data) -> PointCloud? {
        guard let marker = "end_header\n".data(using: .ascii),
              let range = data.range(of: marker),
              let header = String(data: data[..<range.lowerBound], encoding: .ascii),
              let line = header.split(separator: "\n").first(where: { $0.hasPrefix("element vertex") }),
              let n = Int(line.split(separator: " ").last ?? "")
        else { return nil }
        let start = range.upperBound
        let stride = 15
        guard data.count - start >= n * stride else { return nil }
        var pts = [SIMD3<Float>](); pts.reserveCapacity(n)
        var cols = [SIMD3<Float>](); cols.reserveCapacity(n)
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: start)
            for i in 0..<n {
                let p = base.advanced(by: i * stride)
                let x = p.loadUnaligned(as: Float.self)
                let y = p.advanced(by: 4).loadUnaligned(as: Float.self)
                let z = p.advanced(by: 8).loadUnaligned(as: Float.self)
                let r = p.advanced(by: 12).load(as: UInt8.self)
                let g = p.advanced(by: 13).load(as: UInt8.self)
                let b = p.advanced(by: 14).load(as: UInt8.self)
                pts.append(SIMD3(x, y, z))
                cols.append(SIMD3(Float(r), Float(g), Float(b)) / 255)
            }
        }
        return PointCloud(points: pts, colors: cols)
    }
}

/// One-shot camera moves, triggered from the round buttons.
struct MapCommand: Equatable {
    enum Kind { case recenter, topDown, locateRobot, flyThrough }
    let kind: Kind
    let id = UUID()
}

/// Map frame (x, y, z-up) -> SceneKit (x, y-up, z toward viewer).
@inline(__always) func scenePoint(_ p: SIMD3<Float>) -> SCNVector3 { SCNVector3(p.x, p.z, -p.y) }

struct Map3DView: UIViewRepresentable {
    let cloud: PointCloud
    var path: [SIMD3<Float>]
    var robot: RobotLink.Pose?
    var phone: RobotLink.Pose?
    var command: MapCommand?
    /// The connected robot's own model (id, .robot bytes); nil draws the built-in Pioneer.
    var robotModel: (id: String, data: Data)?

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.inertiaEnabled = true
        view.defaultCameraController.maximumVerticalAngle = 89
        view.defaultCameraController.minimumVerticalAngle = 5
        view.scene = context.coordinator.scene
        view.pointOfView = context.coordinator.cameraNode
        context.coordinator.view = view
        context.coordinator.setCloud(cloud)
        context.coordinator.setPath(path)
        context.coordinator.recenter(animated: false)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        let c = context.coordinator
        if c.cloudCount != cloud.points.count { c.setCloud(cloud); c.recenter(animated: false) }
        if c.pathCount != path.count { c.setPath(path) }
        if c.modelID != (robotModel?.id ?? "builtin") { c.setRobot(robotModel) }
        c.place(robot: robot, phone: phone)
        if let command, command != c.lastCommand {
            c.lastCommand = command
            c.run(command.kind)
        }
    }

    final class Coordinator {
        let scene = SCNScene()
        let cameraNode = SCNNode()
        let cloudNode = SCNNode()
        let pathNode = SCNNode()
        var pioneer = RobotModel.load()
        lazy var robotNode: SCNNode = pioneer?.node ?? Coordinator.makePioneer()
        var modelID = "builtin"
        let phoneNode = Coordinator.makePhone()
        weak var view: SCNView?
        var cloudCount = -1
        var pathCount = -1
        var lastCommand: MapCommand?
        var centre = SCNVector3Zero
        var radius: Float = 3
        var flyPath: [SCNVector3] = []

        init() {
            let cam = SCNCamera()
            cam.fieldOfView = 38
            cam.zNear = 0.05
            cam.zFar = 200
            cameraNode.camera = cam
            scene.rootNode.addChildNode(cameraNode)
            scene.rootNode.addChildNode(cloudNode)
            scene.rootNode.addChildNode(pathNode)
            robotNode.isHidden = true
            phoneNode.isHidden = true
            scene.rootNode.addChildNode(robotNode)
            scene.rootNode.addChildNode(phoneNode)
            let light = SCNNode()
            light.light = SCNLight()
            light.light!.type = .ambient
            light.light!.intensity = 900
            scene.rootNode.addChildNode(light)
            let sun = SCNNode()
            sun.light = SCNLight()
            sun.light!.type = .directional
            sun.light!.intensity = 500
            sun.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 5, 0)
            scene.rootNode.addChildNode(sun)
        }

        // MARK: Content

        func setCloud(_ cloud: PointCloud) {
            cloudCount = cloud.points.count
            cloudNode.geometry = nil
            guard !cloud.points.isEmpty else { return }
            let verts = cloud.points.map(scenePoint)
            let vsrc = SCNGeometrySource(vertices: verts)
            let colorData = cloud.colors.withUnsafeBufferPointer { buf -> Data in
                var flat = [Float](); flat.reserveCapacity(buf.count * 3)
                for c in buf { flat.append(c.x); flat.append(c.y); flat.append(c.z) }
                return flat.withUnsafeBufferPointer { Data(buffer: $0) }
            }
            let csrc = SCNGeometrySource(data: colorData, semantic: .color, vectorCount: verts.count,
                                         usesFloatComponents: true, componentsPerVector: 3,
                                         bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                                         dataStride: MemoryLayout<Float>.size * 3)
            let indices = (0..<UInt32(verts.count)).map { $0 }
            let element = SCNGeometryElement(indices: indices, primitiveType: .point)
            element.pointSize = 0.03
            element.minimumPointScreenSpaceRadius = 1.5
            element.maximumPointScreenSpaceRadius = 9
            let geo = SCNGeometry(sources: [vsrc, csrc], elements: [element])
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = UIColor.white
            geo.materials = [m]
            cloudNode.geometry = geo

            // frame the floor: centre and size from the points' spread
            let xs = cloud.points.map(\.x).sorted(), ys = cloud.points.map(\.y).sorted()
            let lo = SIMD2(xs[xs.count / 50], ys[ys.count / 50])
            let hi = SIMD2(xs[xs.count * 49 / 50], ys[ys.count * 49 / 50])
            let mid = (lo + hi) / 2
            centre = scenePoint(SIMD3(mid.x, mid.y, 0.3))
            radius = max(1.5, simd_length(hi - lo) / 2)
        }

        func setPath(_ path: [SIMD3<Float>]) {
            pathCount = path.count
            pathNode.geometry = Coordinator.ribbon(path.map { SIMD2($0.x, $0.y) }, width: 0.11)
            flyPath = path.map { scenePoint($0) }
        }

        func place(robot: RobotLink.Pose?, phone: RobotLink.Pose?) {
            func put(_ node: SCNNode, _ pose: RobotLink.Pose?, height: Float?, walk: Bool = false) {
                guard let pose, pose.ok else { node.isHidden = true; return }
                let target = SCNVector3(Float(pose.x), height ?? Float(pose.z), Float(-pose.y))
                let wasHidden = node.isHidden
                node.isHidden = false
                let dx = target.x - node.position.x, dz = target.z - node.position.z
                let dist = (dx * dx + dz * dz).squareRoot()
                guard wasHidden || dist > 0.01 || abs(node.eulerAngles.y - Float(pose.yaw_deg) * .pi / 180) > 0.01 else { return }
                // walk to a new fix instead of teleporting (about 0.6 m/s, like a careful humanoid)
                let duration = wasHidden ? 0 : min(4.0, max(0.5, Double(dist) / 0.6))
                if walk && !wasHidden && dist > 0.05 { pioneer?.walk(duration: duration) }
                SCNTransaction.begin()
                SCNTransaction.animationDuration = duration
                node.position = target
                node.eulerAngles = SCNVector3(0, Float(pose.yaw_deg) * .pi / 180, 0)
                SCNTransaction.commit()
            }
            put(robotNode, robot, height: 0, walk: true)
            put(phoneNode, phone, height: nil)
        }

        /// Swaps in another team's humanoid (or back to the built-in Pioneer), keeping its place.
        func setRobot(_ model: (id: String, data: Data)?) {
            let next = model.flatMap { RobotModel.load(data: $0.data) } ?? RobotModel.load()
            modelID = model?.id ?? "builtin"
            guard let next else { return }
            let old = robotNode
            next.node.position = old.position
            next.node.eulerAngles = old.eulerAngles
            next.node.isHidden = old.isHidden
            old.removeFromParentNode()
            pioneer = next
            robotNode = next.node
            scene.rootNode.addChildNode(robotNode)
        }

        // MARK: Camera moves

        func run(_ kind: MapCommand.Kind) {
            switch kind {
            case .recenter: recenter(animated: true)
            case .topDown: topDown()
            case .locateRobot: locateRobot()
            case .flyThrough: flyThrough()
            }
        }

        func recenter(animated: Bool) {
            let d = radius * 2.3
            move(to: SCNVector3(centre.x + d * 0.55, centre.y + d * 0.85, centre.z + d * 0.75),
                 lookAt: centre, animated: animated)
        }

        func topDown() {
            move(to: SCNVector3(centre.x, centre.y + radius * 3.2, centre.z + 0.001), lookAt: centre, animated: true)
        }

        func locateRobot() {
            guard !robotNode.isHidden else { recenter(animated: true); return }
            let p = robotNode.position
            move(to: SCNVector3(p.x + 1.6, p.y + 2.4, p.z + 2.0), lookAt: SCNVector3(p.x, 0.5, p.z), animated: true)
        }

        func flyThrough() {
            guard flyPath.count > 2 else { return }
            cameraNode.removeAllActions()
            let pts = stride(from: 0, to: flyPath.count, by: max(1, flyPath.count / 60)).map { flyPath[$0] }
            var actions: [SCNAction] = []
            for (i, p) in pts.enumerated() {
                let next = pts[min(i + 1, pts.count - 1)]
                let eye = SCNVector3(p.x, p.y + 0.15, p.z)
                actions.append(.group([
                    .move(to: eye, duration: 0.18),
                    .run { node in node.look(at: SCNVector3(next.x, next.y - 0.05, next.z)) },
                ]))
            }
            actions.append(.run { [weak self] _ in DispatchQueue.main.async { self?.recenter(animated: true) } })
            view?.pointOfView = cameraNode
            cameraNode.runAction(.sequence(actions))
        }

        private func move(to position: SCNVector3, lookAt target: SCNVector3, animated: Bool) {
            cameraNode.removeAllActions()
            view?.pointOfView = cameraNode
            SCNTransaction.begin()
            SCNTransaction.animationDuration = animated ? 0.7 : 0
            SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            cameraNode.position = position
            cameraNode.look(at: target)
            SCNTransaction.commit()
            view?.defaultCameraController.target = target
        }

        // MARK: Models

        /// A small Pioneer: white body, round white helmet with a dark visor facing +x.
        static func makePioneer() -> SCNNode {
            let root = SCNNode()
            let white = SCNMaterial(); white.diffuse.contents = UIColor(white: 0.97, alpha: 1); white.lightingModel = .physicallyBased; white.roughness.contents = 0.8
            let grey = SCNMaterial(); grey.diffuse.contents = UIColor(white: 0.35, alpha: 1)
            let black = SCNMaterial(); black.diffuse.contents = UIColor(white: 0.07, alpha: 1); black.lightingModel = .physicallyBased; black.roughness.contents = 0.25
            let body = SCNNode(geometry: SCNBox(width: 0.26, height: 0.46, length: 0.34, chamferRadius: 0.08))
            body.geometry?.materials = [white]; body.position = SCNVector3(0, 0.78, 0)
            let legs = SCNNode(geometry: SCNBox(width: 0.22, height: 0.52, length: 0.28, chamferRadius: 0.07))
            legs.geometry?.materials = [grey]; legs.position = SCNVector3(0, 0.27, 0)
            let head = SCNNode(geometry: SCNSphere(radius: 0.15))
            head.geometry?.materials = [white]; head.position = SCNVector3(0, 1.16, 0)
            let visor = SCNNode(geometry: SCNSphere(radius: 0.115))
            visor.geometry?.materials = [black]; visor.scale = SCNVector3(0.55, 0.8, 0.95); visor.position = SCNVector3(0.085, 1.14, 0)
            let shadow = SCNNode(geometry: SCNCylinder(radius: 0.24, height: 0.005))
            let sm = SCNMaterial(); sm.diffuse.contents = UIColor(white: 0, alpha: 0.18); sm.lightingModel = .constant
            shadow.geometry?.materials = [sm]; shadow.position = SCNVector3(0, 0.004, 0)
            [shadow, legs, body, head, visor].forEach(root.addChildNode)
            return root
        }

        /// The phone: a thin dark slab at camera height, screen facing back.
        static func makePhone() -> SCNNode {
            let node = SCNNode(geometry: SCNBox(width: 0.02, height: 0.16, length: 0.08, chamferRadius: 0.01))
            let m = SCNMaterial(); m.diffuse.contents = UIColor(white: 0.15, alpha: 1)
            node.geometry?.materials = [m]
            let ring = SCNNode(geometry: SCNTorus(ringRadius: 0.16, pipeRadius: 0.008))
            let rm = SCNMaterial(); rm.diffuse.contents = UIColor(Theme.ink2); rm.lightingModel = .constant
            ring.geometry?.materials = [rm]; ring.eulerAngles = SCNVector3(0, 0, Float.pi / 2)
            node.addChildNode(ring)
            return node
        }

        /// Flat ribbon on the floor along a 2D path (map x, y).
        static func ribbon(_ pts: [SIMD2<Float>], width: Float) -> SCNGeometry? {
            guard pts.count > 1 else { return nil }
            var verts: [SCNVector3] = []
            var idx: [UInt32] = []
            for i in 0..<pts.count {
                let a = pts[max(0, i - 1)], b = pts[min(pts.count - 1, i + 1)]
                var d = b - a
                if simd_length(d) < 1e-5 { d = SIMD2(1, 0) }
                let n = simd_normalize(SIMD2(-d.y, d.x)) * width / 2
                let l = pts[i] + n, r = pts[i] - n
                verts.append(SCNVector3(l.x, 0.012, -l.y))
                verts.append(SCNVector3(r.x, 0.012, -r.y))
                if i > 0 {
                    let k = UInt32(i * 2)
                    idx += [k - 2, k - 1, k, k - 1, k + 1, k]
                }
            }
            let geo = SCNGeometry(sources: [SCNGeometrySource(vertices: verts)],
                                  elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles)])
            let m = SCNMaterial()
            m.diffuse.contents = UIColor(Theme.trail)
            m.lightingModel = .constant
            m.isDoubleSided = true
            m.transparency = 0.92
            geo.materials = [m]
            return geo
        }
    }
}

/// The dotted "+" backdrop behind the 3D map.
struct MapBackdrop: View {
    var body: some View {
        Canvas { ctx, size in
            var y: CGFloat = 20
            while y < size.height {
                var x: CGFloat = 20
                while x < size.width {
                    var p = Path()
                    p.move(to: CGPoint(x: x - 3, y: y)); p.addLine(to: CGPoint(x: x + 3, y: y))
                    p.move(to: CGPoint(x: x, y: y - 3)); p.addLine(to: CGPoint(x: x, y: y + 3))
                    ctx.stroke(p, with: .color(Theme.line), lineWidth: 1)
                    x += 64
                }
                y += 64
            }
        }
        .background(Theme.paper)
    }
}
