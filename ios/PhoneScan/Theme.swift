import CoreText
import UIKit
import SwiftUI

/// Pioneer Home's look: minimal and product-like. Plain surfaces, sentence-case type, the robot as
/// the hero, and one blue accent for what Pioneer is doing. Every colour adapts to light and dark mode.
enum Theme {
    static let paper = Color(light: 0xF5F5F6, dark: 0x0D0E10)      // app background, map backdrop
    static let card = Color(light: 0xFFFFFF, dark: 0x18191C)       // sheets, tiles, lists
    static let ink = Color(light: 0x16181D, dark: 0xF2F2F3)        // primary text, icons, solid buttons
    static let ink2 = Color(light: 0x5F6268, dark: 0xA0A2A7)       // secondary text
    static let ink3 = Color(light: 0x9A9CA1, dark: 0x6C6E73)       // tertiary text, chevrons
    static let line = Color(light: 0xE6E6E8, dark: 0x2A2B2F)       // hairlines, outlines
    static let soft = Color(light: 0xECECEE, dark: 0x222327)       // quiet fills (round buttons, segments)
    static let chip = Color(light: 0x16181D, dark: 0xF2F2F3)       // dark status chip (inverts in dark mode)
    static let accent = Color(light: 0x3E6AE1, dark: 0x5B82F0)     // the one accent colour
    static let trail = Color(light: 0x3E6AE1, dark: 0x5B82F0)      // the scan / robot route on the 3D map
    static let chipGreen = Color(light: 0x2FA45A, dark: 0x3DBE6C)
    static let good = Color(light: 0x2FA45A, dark: 0x3DBE6C)
    static let warning = Color(light: 0xC2661C, dark: 0xE38B3D)
    static let recording = Color(hex: 0xE0372B)
    /// The phone's marker on maps (the robot uses `accent`).
    static let phone = Color(light: 0x5F6268, dark: 0xA0A2A7)

    static func mono(_ size: CGFloat = 11, _ weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    /// Headline face: the WATonomous brand font if one is bundled, else SF Pro semibold.
    static func display(_ size: CGFloat, _ weight: Font.Weight = .semibold) -> Font {
        if let name = BrandFont.postScriptName {
            return .custom(name, size: size)
        }
        return .system(size: size, weight: weight)
    }
}

/// The appearance picked in Settings.
enum Appearance: String, CaseIterable, Identifiable {
    case system = "System", light = "Light", dark = "Dark"
    var id: String { rawValue }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
}

/// Registers any .ttf / .otf shipped in the app bundle at launch, so the WATonomous font only has
/// to be dropped into ios/PhoneScan/ (no Info.plist edits). The first one found is used for headlines.
enum BrandFont {
    static let postScriptName: String? = {
        let urls = ["otf", "ttf"].flatMap { Bundle.main.urls(forResourcesWithExtension: $0, subdirectory: nil) ?? [] }
        for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
            if let descs = CTFontManagerCreateFontDescriptorsFromURL(url as CFURL) as? [CTFontDescriptor],
               let first = descs.first,
               let name = CTFontDescriptorCopyAttribute(first, kCTFontNameAttribute) as? String {
                return name
            }
        }
        return nil
    }()
}

extension Color {
    /// A colour that follows light / dark mode.
    init(light: UInt32, dark: UInt32) {
        self.init(UIColor { traits in
            UIColor(Color(hex: traits.userInterfaceStyle == .dark ? dark : light))
        })
    }

    /// `Color(hex: 0xRRGGBB)`
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: opacity)
    }
}

// MARK: - Building blocks

extension View {
    /// Solid card with a hairline border.
    func paperCard(_ radius: CGFloat = 18, fill: Color = Theme.card) -> some View {
        background(fill, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(Theme.line, lineWidth: 1))
    }

    /// Small caption, e.g. a chip or a stat's label. Sentence case, no tracking.
    func monoLabel(_ size: CGFloat = 11, color: Color = Theme.ink) -> some View {
        font(.system(size: size + 1.5, weight: .medium)).foregroundStyle(color)
    }
}

/// Status chip. `.dark` for a strong fact, `.outline` for secondary ones, `.tint` / `.solid` for a coloured state.
struct Chip: View {
    enum Style { case dark, outline, tint(Color), solid(Color) }
    let text: String
    var symbol: String?
    var style: Style = .outline

    var body: some View {
        HStack(spacing: 5) {
            if let symbol { Image(systemName: symbol).font(.system(size: 10, weight: .semibold)) }
            Text(text)
        }
        .font(.system(size: 12, weight: .medium))
        .foregroundStyle(fg)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(bg, in: Capsule())
    }

    private var fg: Color {
        switch style {
        case .dark: return Theme.card
        case .outline: return Theme.ink2
        case .tint(let c): return c
        case .solid: return .white
        }
    }
    private var bg: Color {
        switch style {
        case .dark: return Theme.chip
        case .outline: return Theme.soft
        case .tint(let c): return c.opacity(0.14)
        case .solid(let c): return c
        }
    }
}

/// Status line: a coloured dot and a short sentence, e.g. "• Localized · Living room".
struct StatusLine: View {
    let text: String
    var color: Color = Theme.good
    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(text).font(.subheadline).foregroundStyle(Theme.ink2).lineLimit(1)
        }
    }
}

/// Wide action button. `prominent` is solid (ink on light, white on dark); otherwise outlined.
struct Tile: View {
    let title: String
    let symbol: String
    var prominent = false
    var busy = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if busy {
                    ProgressView().tint(prominent ? Theme.card : Theme.ink)
                } else {
                    Image(systemName: symbol).font(.system(size: 15, weight: .medium))
                }
                Text(title).font(.system(size: 16, weight: .semibold)).lineLimit(1).minimumScaleFactor(0.8)
            }
            .foregroundStyle(prominent ? Theme.card : Theme.ink)
            .frame(maxWidth: .infinity, minHeight: 54)
            .background(prominent ? Theme.ink : Color.clear, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(prominent ? Color.clear : Theme.line, lineWidth: 1.5))
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}

/// Round shortcut with a caption under it, the row under the robot on Home.
struct QuickAction: View {
    let title: String
    let symbol: String
    var highlighted = false
    var busy = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                ZStack {
                    Circle().fill(highlighted ? Theme.accent : Theme.card)
                    Circle().stroke(highlighted ? Color.clear : Theme.line, lineWidth: 1)
                    if busy {
                        ProgressView().tint(highlighted ? .white : Theme.ink)
                    } else {
                        Image(systemName: symbol).font(.system(size: 19, weight: .regular))
                            .foregroundStyle(highlighted ? .white : Theme.ink)
                    }
                }
                .frame(width: 58, height: 58)
                Text(title).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.ink2).lineLimit(1)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
    }
}

/// A list row: icon, title, subtitle, chevron.
struct ListRow: View {
    let symbol: String
    let title: String
    var subtitle: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: symbol).font(.system(size: 18, weight: .regular)).frame(width: 26)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.system(size: 16, weight: .medium))
                    if let subtitle { Text(subtitle).font(.footnote).foregroundStyle(Theme.ink2).lineLimit(1) }
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.ink3)
            }
            .foregroundStyle(Theme.ink)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A value and its label, e.g. "0.36 m" / "X".
struct StatView: View {
    let value: String
    var unit: String = ""
    let label: String
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 22, weight: .semibold)).monospacedDigit()
                    .contentTransition(.numericText())
                if !unit.isEmpty { Text(unit).font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.ink2) }
            }
            .foregroundStyle(Theme.ink)
            Text(label).font(.caption).foregroundStyle(Theme.ink2)
        }
    }
}

/// Small square button that floats on the map.
struct SquareButton: View {
    let symbol: String
    let label: String
    var active = false
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(active ? Theme.card : Theme.ink)
                .frame(width: 38, height: 38)
                .background(active ? Theme.ink : Theme.card, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Theme.line, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Round button that floats down the right edge of the map.
struct RoundButton: View {
    let symbol: String
    let label: String
    var active = false
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(active ? Theme.card : Theme.ink)
                .frame(width: 42, height: 42)
                .background(active ? Theme.ink : Theme.card, in: Circle())
                .overlay(Circle().stroke(Theme.line, lineWidth: active ? 0 : 1))
                .shadow(color: .black.opacity(0.06), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Solid button floating above the tab bar, e.g. "+ New scan".
struct PillButton: View {
    let title: String
    let symbol: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: symbol).font(.system(size: 15, weight: .semibold))
                Text(title).font(.system(size: 16, weight: .semibold))
            }
            .foregroundStyle(Theme.card)
            .frame(maxWidth: .infinity, minHeight: 54)
            .background(Theme.ink, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
        }
        .buttonStyle(.plain)
    }
}

/// Segmented control: a quiet track with the selected segment raised.
struct Segmented<T: Hashable>: View {
    let options: [(T, String)]
    @Binding var selection: T
    var body: some View {
        HStack(spacing: 0) {
            ForEach(options.indices, id: \.self) { i in
                let value = options[i].0
                Button { selection = value } label: {
                    Text(options[i].1)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(selection == value ? Theme.ink : Theme.ink2)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(selection == value ? Theme.card : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                        .shadow(color: .black.opacity(selection == value ? 0.08 : 0), radius: 2, y: 1)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Theme.soft, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

/// Section caption above a group, e.g. "Scans".
struct SectionLabel: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.ink2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.leading, 4)
    }
}
