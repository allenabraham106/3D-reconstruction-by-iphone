import ARKit
import simd

/// A CPU copy of ARKit's scene-reconstruction mesh, merged across anchors, in world coordinates.
///
/// Taken on the main thread when recording stops (ARMeshAnchor geometry lives in Metal buffers
/// that ARKit may recycle), then written as a binary PLY on the writer queue.
struct MeshSnapshot {
    private(set) var vertices: [SIMD3<Float>] = []
    private(set) var normals: [SIMD3<Float>] = []
    private(set) var faces: [UInt32] = []   // 3 indices per triangle

    var isEmpty: Bool { faces.isEmpty }
    var faceCount: Int { faces.count / 3 }

    init(anchors: [ARMeshAnchor]) {
        for anchor in anchors {
            let geometry = anchor.geometry
            let transform = anchor.transform
            let normalMatrix = simd_float3x3(
                SIMD3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
                SIMD3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
                SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
            )
            let base = UInt32(vertices.count)

            let vsrc = geometry.vertices
            let vptr = vsrc.buffer.contents()
            for i in 0..<vsrc.count {
                let p = (vptr + vsrc.offset + vsrc.stride * i).assumingMemoryBound(to: Float.self)
                let world = transform * SIMD4<Float>(p[0], p[1], p[2], 1)
                vertices.append(SIMD3(world.x, world.y, world.z))
            }

            let nsrc = geometry.normals
            if nsrc.count == vsrc.count {
                let nptr = nsrc.buffer.contents()
                for i in 0..<nsrc.count {
                    let n = (nptr + nsrc.offset + nsrc.stride * i).assumingMemoryBound(to: Float.self)
                    normals.append(simd_normalize(normalMatrix * SIMD3<Float>(n[0], n[1], n[2])))
                }
            } else {
                normals.append(contentsOf: repeatElement(SIMD3<Float>(0, 1, 0), count: vsrc.count))
            }

            let fsrc = geometry.faces
            let indexCount = fsrc.count * fsrc.indexCountPerPrimitive
            let fptr = fsrc.buffer.contents()
            faces.reserveCapacity(faces.count + indexCount)
            for i in 0..<indexCount {
                let index: UInt32
                if fsrc.bytesPerIndex == 4 {
                    index = fptr.load(fromByteOffset: i * 4, as: UInt32.self)
                } else {
                    index = UInt32(fptr.load(fromByteOffset: i * 2, as: UInt16.self))
                }
                faces.append(base + index)
            }
        }
    }

    /// Binary little-endian PLY with per-vertex normals. Opens in MeshLab, Blender, Open3D, CloudCompare.
    func writePLY(to url: URL) throws {
        var header = "ply\nformat binary_little_endian 1.0\ncomment PhoneScan ARKit scene mesh, world frame, metres\n"
        header += "element vertex \(vertices.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property float nx\nproperty float ny\nproperty float nz\n"
        header += "element face \(faceCount)\n"
        header += "property list uchar uint vertex_indices\nend_header\n"

        var data = Data(header.utf8)
        data.reserveCapacity(data.count + vertices.count * 24 + faceCount * 13)

        for i in 0..<vertices.count {
            let v = vertices[i], n = normals[i]
            for f in [v.x, v.y, v.z, n.x, n.y, n.z] {
                withUnsafeBytes(of: f.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            }
        }
        for t in 0..<faceCount {
            data.append(3)
            for k in 0..<3 {
                withUnsafeBytes(of: faces[t * 3 + k].littleEndian) { data.append(contentsOf: $0) }
            }
        }
        try data.write(to: url)
    }
}
