import SwiftUI

/// Every brand colour in the app lives here, so the look can be changed in one place.
///
/// WATonomous palette, sampled from watonomous.ca: the logo's blue, the teal of the W lanterns,
/// the night-alley navy, and the warm shop-light amber.
enum Theme {
    /// WATonomous logo blue: primary buttons and the record button.
    static let primary = Color(hex: 0x1E5E96)
    /// Lighter brand blue for text and icons on dark backgrounds (logo blue is too dark there).
    static let primaryBright = Color(hex: 0x4A8FD0)
    /// Lantern teal: the mini-map path, glows, info icons.
    static let teal = Color(hex: 0x5CC0C8)
    /// Night navy: sheet backgrounds.
    static let navy = Color(hex: 0x0E2230)
    /// Raised navy: cards and grouped rows.
    static let surface = Color(hex: 0x16303F)
    /// Shop-light amber: warnings.
    static let amber = Color(hex: 0xE3A867)

    static let secondary = teal
    static let good = Color(hex: 0x4CD38A)
    static let warning = amber
    /// Recording stays red: it's the universal "this is recording" signal.
    static let recording = Color(hex: 0xFF453A)
}

extension Color {
    /// `Color(hex: 0xRRGGBB)`
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: opacity)
    }
}

extension View {
    /// Frosted panel tinted night-navy with a faint lantern-teal edge, used for every HUD element.
    func watoGlass<S: Shape>(_ shape: S) -> some View {
        background {
            ZStack {
                shape.fill(.ultraThinMaterial)
                shape.fill(Theme.navy.opacity(0.5))
            }
        }
        .overlay(shape.stroke(Theme.teal.opacity(0.22), lineWidth: 1))
    }

    /// Wide capitals with tracking, like the WATonomous site's navigation.
    func watoLabel() -> some View {
        self.fontWidth(.expanded).textCase(.uppercase).tracking(1.2)
    }
}

/// "PIONEER HOME" wordmark in the style of the WATONOMOUS logo button.
struct Wordmark: View {
    var body: some View {
        Text("Pioneer Home")
            .font(.system(size: 15, weight: .heavy))
            .watoLabel()
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Theme.primary, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}
