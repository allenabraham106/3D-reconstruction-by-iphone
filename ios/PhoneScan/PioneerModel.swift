import Foundation
import SceneKit
import UIKit
import simd

/// An articulated humanoid built from a URDF by tools/urdf_to_app_model.py.
///
/// The app ships the Pioneer (WATonomous/pioneer_humanoid, pioneer_whole_body_humanoid.urdf,
/// Apache-2.0); a robot link started with --robot-urdf sends its own robot instead, so any
/// team's humanoid shows up here. Every link keeps its joint, so the legs, arms and fingers can
/// move. `node` is in SceneKit axes (y up), feet at y = 0, facing +x at yaw 0: the robot map's
/// heading convention, so `node.eulerAngles.y = yaw` points it the right way.
typealias PioneerModel = RobotModel

final class RobotModel {
    let node = SCNNode()
    let name: String
    private let gait: Gait
    private var joints: [String: Joint] = [:]
    private var walking = false

    private struct Joint {
        let node: SCNNode
        let axis: SIMD3<Float>
        let lower: Float
        let upper: Float
    }

    // MARK: Loading

    /// Which joints play which walking role (detected by the converter, or set with --gait).
    struct Gait: Decodable {
        var hip_pitch: [String: String]?
        var knee: [String: String]?
        var ankle_pitch: [String: String]?
        var shoulder_pitch: [String: String]?
        var elbow: [String: String]?
        var curl: [String]?

        /// Pioneer's joint names, for models converted before roles were stored.
        static let pioneer = Gait(
            hip_pitch: ["left": "Hip_F_L", "right": "Hip_F_R"], knee: ["left": "Knee_L", "right": "Knee_R"],
            ankle_pitch: ["left": "Ankle_P_L", "right": "Ankle_P_R"],
            shoulder_pitch: ["left": "l_shoulder_pitch", "right": "r_shoulder_pitch"],
            elbow: ["left": "l_elbow", "right": "r_elbow"],
            curl: ["mcp", "pip", "dip", "i_mcp", "i_pip", "i_dip", "r_mcp", "r_pip", "r_dip", "l_mcp", "l_pip", "l_dip"]
                .flatMap { ["r_\($0)", "l_\($0)"] })
    }

    private struct Manifest: Decodable {
        struct Part: Decodable { let color: [Float]; let v: [Int]; let i: [Int] }
        struct JointSpec: Decodable {
            let name: String; let type: String
            let xyz: [Float]; let rpy: [Float]; let axis: [Float]
            let lower: Float; let upper: Float
        }
        struct Link: Decodable { let name: String; let parent: String?; let joint: JointSpec?; let parts: [Part] }
        let links: [Link]
        let index_offset: Int
        let name: String?
        let forward: [Float]?
        let gait: Gait?
        /// Lowest point of the zero pose in the URDF frame; lifted so every robot stands on the floor.
        let foot_z: Float?
    }

    /// The built-in Pioneer (`Pioneer.robot` in the app bundle).
    static func load() -> RobotModel? {
        guard let url = Bundle.main.url(forResource: "Pioneer", withExtension: "robot"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        return load(data: data)
    }

    /// A model sent by a robot link (`GET /robot/model`). Returns nil if the data isn't a robot model.
    static func load(data: Data) -> RobotModel? {
        guard data.count > 8, data.prefix(4) == Data("PRBT".utf8) else { return nil }
        let jsonLen = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) })
        guard data.count > 8 + jsonLen,
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data.subdata(in: 8..<(8 + jsonLen)))
        else { return nil }
        let buffer = data.subdata(in: (8 + jsonLen)..<data.count)
        return RobotModel(manifest: manifest, buffer: buffer)
    }

    private init(manifest: Manifest, buffer: Data) {
        name = manifest.name ?? "Pioneer"
        gait = manifest.gait ?? .pioneer
        // URDF is z-up; turn the robot's forward axis to +x, then to SceneKit's y-up.
        let fwd = manifest.forward ?? [0, -1]   // models without it are the Pioneer, which faces -y
        let toYUp = SCNNode()
        toYUp.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)
        toYUp.position = SCNVector3(0, -(manifest.foot_z ?? 0), 0)   // e.g. hip-origin URDFs: feet at z = -0.84
        let facing = SCNNode()
        facing.eulerAngles = SCNVector3(0, 0, -atan2(fwd.count > 1 ? fwd[1] : 0, fwd.first ?? 1))
        node.addChildNode(toYUp)
        toYUp.addChildNode(facing)

        var linkNodes: [String: SCNNode] = [:]
        for link in manifest.links {
            let linkNode = SCNNode()
            linkNode.name = link.name
            for part in link.parts {
                if let g = Self.geometry(part, buffer: buffer, indexOffset: manifest.index_offset) {
                    linkNode.addChildNode(SCNNode(geometry: g))
                }
            }
            if let j = link.joint, let parentName = link.parent, let parent = linkNodes[parentName] {
                // origin (fixed) -> motion (the joint's rotation) -> link
                let origin = SCNNode()
                origin.simdPosition = SIMD3(j.xyz[0], j.xyz[1], j.xyz[2])
                origin.simdOrientation = Self.rpy(j.rpy)
                let motion = SCNNode()
                origin.addChildNode(motion)
                motion.addChildNode(linkNode)
                parent.addChildNode(origin)
                if j.type == "revolute" || j.type == "continuous" {
                    let axis = simd_normalize(SIMD3(j.axis[0], j.axis[1], j.axis[2]))
                    joints[j.name] = Joint(node: motion, axis: axis, lower: j.lower, upper: j.upper)
                }
            } else {
                facing.addChildNode(linkNode)   // base_link
            }
            linkNodes[link.name] = linkNode
        }
        stand()
    }

    private static func geometry(_ part: Manifest.Part, buffer: Data, indexOffset: Int) -> SCNGeometry? {
        let stride = 24
        let vStart = part.v[0] * stride, vCount = part.v[1]
        let iStart = indexOffset + part.i[0] * 4, iCount = part.i[1]
        guard vCount > 0, iCount >= 3, vStart + vCount * stride <= indexOffset,
              iStart + iCount * 4 <= buffer.count else { return nil }
        let vdata = buffer.subdata(in: vStart..<(vStart + vCount * stride))
        let positions = SCNGeometrySource(data: vdata, semantic: .vertex, vectorCount: vCount,
                                          usesFloatComponents: true, componentsPerVector: 3,
                                          bytesPerComponent: 4, dataOffset: 0, dataStride: stride)
        let normals = SCNGeometrySource(data: vdata, semantic: .normal, vectorCount: vCount,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 12, dataStride: stride)
        let element = SCNGeometryElement(data: buffer.subdata(in: iStart..<(iStart + iCount * 4)),
                                         primitiveType: .triangles, primitiveCount: iCount / 3, bytesPerIndex: 4)
        let g = SCNGeometry(sources: [positions, normals], elements: [element])
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        let c = part.color
        m.diffuse.contents = UIColor(red: CGFloat(c[0]), green: CGFloat(c[1]), blue: CGFloat(c[2]), alpha: CGFloat(c.count > 3 ? c[3] : 1))
        m.roughness.contents = 0.55
        m.metalness.contents = 0.05
        g.materials = [m]
        return g
    }

    /// URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    private static func rpy(_ v: [Float]) -> simd_quatf {
        simd_quatf(angle: v[2], axis: SIMD3(0, 0, 1)) *
        simd_quatf(angle: v[1], axis: SIMD3(0, 1, 0)) *
        simd_quatf(angle: v[0], axis: SIMD3(1, 0, 0))
    }

    // MARK: Posing

    /// Sets a joint angle (radians), clamped to its URDF limits.
    func set(_ name: String, _ angle: Float) {
        guard let j = joints[name] else { return }
        let a = j.upper > j.lower ? min(j.upper, max(j.lower, angle)) : angle
        j.node.simdOrientation = simd_quatf(angle: a, axis: j.axis)
    }

    /// An angle on the joint's larger-range side (so "forward" works whatever its axis sign).
    private func swing(_ name: String, _ amount: Float) {
        guard let j = joints[name] else { return }
        set(name, (abs(j.upper) >= abs(j.lower) ? amount : -amount))
    }

    /// A fraction of a joint's range on the side it bends most, e.g. fingers curling closed.
    private func bend(_ name: String, _ fraction: Float) {
        guard let j = joints[name] else { return }
        set(name, (abs(j.upper) >= abs(j.lower) ? j.upper : j.lower) * fraction)
    }

    private func role(_ r: [String: String]?, _ side: String) -> String? { r?[side] }

    /// Relaxed standing pose: soft knees, elbows a little bent, hands loosely curled.
    func stand() {
        for side in ["left", "right"] {
            if let hip = role(gait.hip_pitch, side) { set(hip, 0) }
            if let knee = role(gait.knee, side) { bend(knee, 0.07) }
            if let ankle = role(gait.ankle_pitch, side) { set(ankle, 0) }
            if let elbow = role(gait.elbow, side) { bend(elbow, 0.12) }
        }
        for j in gait.curl ?? [] { bend(j, 0.25) }
    }

    /// Plays a walking cycle for `duration` seconds (legs alternate, arms swing opposite).
    func walk(duration: TimeInterval) {
        guard !walking, duration > 0.1 else { return }
        walking = true
        let steps = max(1, Float(duration / 0.55))  // ~0.55 s per step
        let action = SCNAction.customAction(duration: duration) { [weak self] _, t in
            guard let self else { return }
            let phase = Float(t / CGFloat(duration)) * steps * .pi
            for (side, offset) in [("left", Float(0)), ("right", Float.pi)] {
                let s = sin(phase + offset)
                let lift = max(0, sin(phase + offset + .pi / 2))
                // swing sign follows each joint's own limits, so any URDF's axis direction works
                if let hip = self.role(self.gait.hip_pitch, side) { self.swing(hip, 0.32 * s) }
                if let knee = self.role(self.gait.knee, side) { self.bend(knee, 0.07 + 0.32 * lift) }
                if let ankle = self.role(self.gait.ankle_pitch, side) { self.swing(ankle, 0.18 * s) }
                if let sh = self.role(self.gait.shoulder_pitch, side) { self.swing(sh, -0.28 * s) }
            }
        }
        node.runAction(.sequence([action, .run { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                for side in ["left", "right"] {
                    if let sh = self.role(self.gait.shoulder_pitch, side) { self.set(sh, 0) }
                }
                self.stand()
                self.walking = false
            }
        }]))
    }
}
