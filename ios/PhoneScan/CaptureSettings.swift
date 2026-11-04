import Foundation

/// User-tunable capture parameters, persisted in UserDefaults (and bound with @AppStorage in SettingsView).
struct CaptureSettings {
    enum Key {
        static let minTranslationCm = "minTranslationCm"
        static let minRotationDeg = "minRotationDeg"
        static let minIntervalSec = "minIntervalSec"
        static let jpegQuality = "jpegQuality"
        static let saveDepth = "saveDepth"
        static let saveMesh = "saveMesh"
        static let showMesh = "showMesh"
        static let use4K = "use4K"
        static let serverURL = "serverURL"
    }

    static let defaults: [String: Any] = [
        Key.minTranslationCm: 5.0,
        Key.minRotationDeg: 5.0,
        Key.minIntervalSec: 0.2,
        Key.jpegQuality: 0.9,
        Key.saveDepth: true,
        Key.saveMesh: true,
        Key.showMesh: true,
        Key.use4K: false,
        Key.serverURL: "http://192.168.1.100:8765",
    ]

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: defaults)
    }

    /// Minimum camera translation between keyframes, metres.
    var minTranslation: Float
    /// Minimum camera rotation between keyframes, radians.
    var minRotation: Float
    /// Never take keyframes faster than this, seconds.
    var minInterval: TimeInterval
    var jpegQuality: Double
    var saveDepth: Bool
    var saveMesh: Bool
    var use4K: Bool

    static func load() -> CaptureSettings {
        registerDefaults()
        let d = UserDefaults.standard
        return CaptureSettings(
            minTranslation: Float(d.double(forKey: Key.minTranslationCm) / 100.0),
            minRotation: Float(d.double(forKey: Key.minRotationDeg) * .pi / 180.0),
            minInterval: d.double(forKey: Key.minIntervalSec),
            jpegQuality: d.double(forKey: Key.jpegQuality),
            saveDepth: d.bool(forKey: Key.saveDepth),
            saveMesh: d.bool(forKey: Key.saveMesh),
            use4K: d.bool(forKey: Key.use4K)
        )
    }
}
