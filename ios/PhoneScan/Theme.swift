import CoreText
import SwiftUI

/// Pioneer Home's look: a calm, product-like UI. White surfaces, ink text, hairline borders,
/// small monospace captions, and the robot's 3D map as the hero. Colour is reserved for state:
/// solid chips (green = connected, WATonomous blue = what Pioneer is doing) and the scan trail.
enum Theme {
    static let paper = Color(hex: 0xF4F3EF)      // app background, map backdrop
    static let card = Color(hex: 0xFFFFFF)       // sheets, tiles
    static let ink = Color(hex: 0x1C1C1A)        // primary text, icons
    static let ink2 = Color(hex: 0x6E6C66)       // secondary text
    static let line = Color(hex: 0xD3D0C7)       // hairlines
    static let chip = Color(hex: 0x2A2A28)       // dark status chip
    static let accent = Color(hex: 0x1E5E96)     // WATonomous logo blue
    static let trail = Color(hex: 0x2F7FD1)      // the scan / robot route on the 3D map
    static let chipGreen = Color(hex: 0x1F9D49)
    static let good = Color(hex: 0x2E7D4F)
    static let warning = Color(hex: 0xB5651D)
    static let recording = Color(hex: 0xD93A2B)
    /// The phone's marker on maps (the robot uses `accent`).
    static let phone = Color(hex: 0x6E6C66)

    static func mono(_ size: CGFloat = 11, _ weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    /// Headline face: the WATonomous brand font if one is bundled, else SF Pro Expanded
    /// (the closest built-in match to WATonomous's wide geometric lettering).
    static func display(_ size: CGFloat, _ weight: Font.Weight = .medium) -> Font {
        if let name = BrandFont.postScriptName {
            return .custom(name, size: size)
        }
        return .system(size: size, weight: weight).width(.expanded)
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
    func paperCard(_ radius: CGFloat = 14, fill: Color = Theme.card) -> some View {
        background(fill, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(Theme.line, lineWidth: 1))
    }

    /// Small uppercase monospace label, e.g. "PIONEER-WATO-01".
    func monoLabel(_ size: CGFloat = 11, color: Color = Theme.ink) -> some View {
        font(Theme.mono(size)).textCase(.uppercase).tracking(0.6).foregroundStyle(color)
    }
}

/// Status chip. `.dark` like "✓ READY", `.outline` for secondary facts, `.tint` for a coloured state.
struct Chip: View {
    enum Style { case dark, outline, tint(Color), solid(Color) }
    let text: String
    var symbol: String?
    var style: Style = .outline

    var body: some View {
        HStack(spacing: 4) {
            if let symbol { Image(systemName: symbol).font(.system(size: 9, weight: .bold)) }
            Text(text)
        }
        .monoLabel(10, color: fg)
        .padding(.horizontal, 7)
        .padding(.vertical, 4)
        .background(bg, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(border, lineWidth: 1))
    }

    private var fg: Color {
        switch style {
        case .dark: return Theme.card
        case .outline: return Theme.ink
        case .tint(let c): return c
        case .solid: return .white
        }
    }
    private var bg: Color {
        switch style {
        case .dark: return Theme.chip
        case .outline: return .clear
        case .tint(let c): return c.opacity(0.12)
        case .solid(let c): return c
        }
    }
    private var border: Color {
        switch style {
        case .dark: return Theme.chip
        case .outline: return Theme.line
        case .tint(let c): return c.opacity(0.35)
        case .solid(let c): return c
        }
    }
}

/// Big square-ish action tile with a line icon and a mono caption, like "CLEAN ROOMS".
struct Tile: View {
    let title: String
    let symbol: String
    var prominent = false
    var busy = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 0) {
                ZStack(alignment: .leading) {
                    Image(systemName: symbol)
                        .font(.system(size: 19, weight: .light))
                        .opacity(busy ? 0 : 1)
                    if busy { ProgressView().tint(prominent ? Theme.card : Theme.ink) }
                }
                .frame(height: 24)
                Spacer(minLength: 14)
                Text(title)
                    .monoLabel(11, color: prominent ? Theme.card : Theme.ink)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(prominent ? Theme.card : Theme.ink)
            .frame(maxWidth: .infinity, minHeight: 84, alignment: .topLeading)
            .padding(12)
            .background(prominent ? Theme.ink : Theme.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(prominent ? Theme.ink : Theme.line, lineWidth: 1))
        }
        .buttonStyle(.plain)
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
                .background(active ? Theme.ink : Theme.card.opacity(0.94), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).stroke(Theme.line, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Round white button that floats down the right edge of the map.
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
                .shadow(color: .black.opacity(0.10), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Black pill floating above the tab bar, e.g. "+ NEW SCAN".
struct PillButton: View {
    let title: String
    let symbol: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: symbol).font(.system(size: 16, weight: .regular))
                Text(title).monoLabel(12, color: Theme.card)
            }
            .foregroundStyle(Theme.card)
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Theme.ink, in: Capsule())
            .shadow(color: .black.opacity(0.18), radius: 10, y: 4)
        }
        .buttonStyle(.plain)
    }
}

/// Segmented control: black selected segment, outlined others (like SCHEDULES | HISTORY).
struct Segmented<T: Hashable>: View {
    let options: [(T, String)]
    @Binding var selection: T
    var body: some View {
        HStack(spacing: 0) {
            ForEach(options.indices, id: \.self) { i in
                let value = options[i].0
                Button { selection = value } label: {
                    Text(options[i].1)
                        .monoLabel(11, color: selection == value ? Theme.card : Theme.ink)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 11)
                        .background(selection == value ? Theme.ink : Color.clear)
                }
                .buttonStyle(.plain)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(Theme.ink, lineWidth: 1))
    }
}

/// Section caption above a group, e.g. "SCANS".
struct SectionLabel: View {
    let text: String
    var body: some View {
        Text(text).monoLabel(10, color: Theme.ink2).frame(maxWidth: .infinity, alignment: .leading)
    }
}
