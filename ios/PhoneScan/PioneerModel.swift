import Foundation
import SceneKit
import UIKit
import simd

/// The Pioneer humanoid, built from its URDF (WATonomous/pioneer_humanoid,
/// pioneer_whole_body_humanoid.urdf, Apache-2.0) by tools/urdf_to_app_model.py.
///
/// Every link keeps its joint, so the legs, arms and fingers can move. `node` is in SceneKit
/// axes (y up), with the feet at y = 0, facing +x at yaw 0: the same heading convention as the
/// robot map (yaw CCW from +x), so `node.eulerAngles.y = yaw` points it the right way.
final class PioneerModel {
    let node = SCNNode()
    private var joints: [String: Joint] = [:]
    private var walking = false

    private struct Joint {
        let node: SCNNode
        let axis: SIMD3<Float>
        let lower: Float
        let upper: Float
    }

    // MARK: Loading

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
    }

    /// Loads `Pioneer.robot` from the app bundle. Returns nil if it's missing or unreadable.
    static func load() -> PioneerModel? {
        guard let url = Bundle.main.url(forResource: "Pioneer", withExtension: "robot"),
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              data.count > 8, data.prefix(4) == Data("PRBT".utf8)
        else { return nil }
        let jsonLen = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) })
        guard data.count > 8 + jsonLen,
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data.subdata(in: 8..<(8 + jsonLen)))
        else { return nil }
        let buffer = data.subdata(in: (8 + jsonLen)..<data.count)
        return PioneerModel(manifest: manifest, buffer: buffer)
    }

    private init(manifest: Manifest, buffer: Data) {
        // URDF is z-up and Pioneer faces -y. Turn it to face +x, then to SceneKit's y-up.
        let toYUp = SCNNode()
        toYUp.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)
        let facing = SCNNode()
        facing.eulerAngles = SCNVector3(0, 0, Float.pi / 2)
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

    /// A fraction of a joint's range on the side it bends most, e.g. fingers curling closed.
    private func bend(_ name: String, _ fraction: Float) {
        guard let j = joints[name] else { return }
        set(name, (abs(j.upper) >= abs(j.lower) ? j.upper : j.lower) * fraction)
    }

    /// Relaxed standing pose: soft knees, hands loosely curled.
    func stand() {
        for side in ["L", "R"] {
            set("Hip_F_\(side)", 0.08)
            set("Knee_\(side)", -0.16)
            set("Ankle_P_\(side)", 0.08)
        }
        for side in ["r", "l"] {
            set("\(side)_elbow", -0.25)
            for f in ["", "i_", "r_", "l_"] {
                bend("\(side)_\(f)mcp", 0.25)
                bend("\(side)_\(f)pip", 0.3)
                bend("\(side)_\(f)dip", 0.25)
            }
        }
    }

    /// Plays a walking cycle for `duration` seconds (legs alternate, arms swing opposite).
    func walk(duration: TimeInterval) {
        guard !walking, duration > 0.1 else { return }
        walking = true
        let steps = max(1, Float(duration / 0.55))  // ~0.55 s per step
        let action = SCNAction.customAction(duration: duration) { [weak self] _, t in
            guard let self else { return }
            let phase = Float(t / CGFloat(duration)) * steps * .pi
            for (side, offset) in [("L", Float(0)), ("R", Float.pi)] {
                let s = sin(phase + offset)
                let lift = max(0, sin(phase + offset + .pi / 2))
                self.set("Hip_F_\(side)", 0.08 + 0.32 * s)
                self.set("Knee_\(side)", -0.16 - 0.75 * lift)
                self.set("Ankle_P_\(side)", 0.08 + 0.18 * s)
            }
            self.set("r_shoulder_pitch", 0.28 * sin(phase))
            self.set("l_shoulder_pitch", 0.28 * sin(phase))
        }
        node.runAction(.sequence([action, .run { [weak self] _ in
            DispatchQueue.main.async {
                self?.set("r_shoulder_pitch", 0)
                self?.set("l_shoulder_pitch", 0)
                self?.stand()
                self?.walking = false
            }
        }]))
    }
}
