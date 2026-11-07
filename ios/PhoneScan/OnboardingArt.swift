import SwiftUI

/// Looping illustrations for onboarding steps 2-5, each showing what the tip is about.
/// Drawn with Canvas from the clock, so there is no state to reset between pages.
struct OnboardingArt: View {
    let step: Int

    var body: some View {
        TimelineView(.animation) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            Canvas { ctx, size in
                switch step {
                case 1: Self.pace(&ctx, size, t)
                case 2: Self.sweep(&ctx, size, t)
                case 3: Self.loop(&ctx, size, t)
                default: Self.send(&ctx, size, t)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 250)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Theme.line))
        .accessibilityHidden(true)
    }

    private static func label(_ s: String, _ color: Color, size: CGFloat = 10) -> Text {
        Text(s).font(Theme.mono(size, .semibold)).foregroundColor(color)
    }

    private static func symbol(_ name: String, _ size: CGFloat, _ color: Color) -> Text {
        Text(Image(systemName: name)).font(.system(size: size, weight: .regular)).foregroundColor(color)
    }

    // MARK: Walk slowly: keyframes pile up when slow, thin out when fast.

    private static func pace(_ ctx: inout GraphicsContext, _ size: CGSize, _ t: Double) {
        let W = size.width, floorY = size.height * 0.56
        let period = 6.5
        let p = t.truncatingRemainder(dividingBy: period)
        // Slow, a burst of fast, slow again. x in fractions of the width.
        func x(_ p: Double) -> Double {
            let slow = 0.09, fast = 0.30
            if p < 3.5 { return 0.06 + slow * p }
            if p < 5.0 { return 0.06 + slow * 3.5 + fast * (p - 3.5) }
            return 0.06 + slow * 3.5 + fast * 1.5 + slow * min(p - 5.0, 1.2)
        }
        let fast = p >= 3.5 && p < 5.0

        var floor = Path(); floor.move(to: CGPoint(x: 18, y: floorY)); floor.addLine(to: CGPoint(x: W - 18, y: floorY))
        ctx.stroke(floor, with: .color(Theme.line), lineWidth: 2)

        // A keyframe every 0.3 s: dense where slow, gaps where fast.
        var k = 0.0
        while k <= p {
            let kx = x(k) * W
            let r: CGFloat = 4
            ctx.fill(Path(ellipseIn: CGRect(x: kx - r, y: floorY + 10, width: 2 * r, height: 2 * r)),
                     with: .color(Theme.accent.opacity(0.85)))
            k += 0.3
        }
        let wx = x(p) * W
        ctx.draw(symbol("figure.walk", 46, fast ? Theme.warning : Theme.ink), at: CGPoint(x: wx, y: floorY - 26))

        // Pace meter, like the capture HUD.
        let barW = W * 0.64, barX = (W - barW) / 2, barY = size.height * 0.80
        let level = fast ? 0.86 : 0.32 + 0.04 * sin(t * 3)
        ctx.fill(Path(roundedRect: CGRect(x: barX, y: barY, width: barW, height: 8), cornerRadius: 4), with: .color(Theme.soft))
        ctx.fill(Path(roundedRect: CGRect(x: barX, y: barY, width: barW * level, height: 8), cornerRadius: 4),
                 with: .color(fast ? Theme.warning : Theme.good))
        ctx.draw(label("PACE", Theme.ink3), at: CGPoint(x: barX, y: barY - 12), anchor: .leading)
        ctx.draw(label(fast ? "TOO FAST" : "GOOD", fast ? Theme.warning : Theme.good), at: CGPoint(x: barX + barW, y: barY - 12), anchor: .trailing)
        ctx.draw(label("keyframes", Theme.ink3, size: 9), at: CGPoint(x: 18, y: floorY + 30), anchor: .leading)
    }

    // MARK: Sweep up and down: the view cone covers floor, wall and ceiling.

    private static func sweep(_ ctx: inout GraphicsContext, _ size: CGSize, _ t: Double) {
        let W = size.width, H = size.height
        let ceil = H * 0.14, floor = H * 0.86, wall = W * 0.90
        let phone = CGPoint(x: W * 0.17, y: H * 0.52)
        let tilt = sin(t * 1.3) * 0.62          // radians, + = up
        let half = 0.30

        var room = Path()
        room.move(to: CGPoint(x: 18, y: ceil)); room.addLine(to: CGPoint(x: wall, y: ceil))
        room.addLine(to: CGPoint(x: wall, y: floor)); room.addLine(to: CGPoint(x: 18, y: floor))
        ctx.stroke(room, with: .color(Theme.line), lineWidth: 2)

        func hit(_ a: Double) -> CGPoint {
            let dx = cos(a), dy = -sin(a)
            var d = (wall - phone.x) / dx
            var y = phone.y + dy * d
            if y < ceil { d = (ceil - phone.y) / dy; y = ceil }
            if y > floor { d = (floor - phone.y) / dy; y = floor }
            return CGPoint(x: phone.x + dx * d, y: y)
        }

        var cone = Path()
        cone.move(to: phone)
        for i in 0...24 { cone.addLine(to: hit(tilt - half + 2 * half * Double(i) / 24)) }
        cone.closeSubpath()
        ctx.fill(cone, with: .color(Theme.accent.opacity(0.13)))

        // Surface points: lit inside the cone.
        for i in 0..<34 {
            let a = -1.05 + 2.1 * Double(i) / 33
            let q = hit(a)
            let lit = abs(a - tilt) < half
            let r: CGFloat = lit ? 4 : 2.5
            ctx.fill(Path(ellipseIn: CGRect(x: q.x - r, y: q.y - r, width: 2 * r, height: 2 * r)),
                     with: .color(lit ? Theme.accent : Theme.ink3.opacity(0.5)))
        }

        var phoneCtx = ctx
        phoneCtx.translateBy(x: phone.x, y: phone.y)
        phoneCtx.rotate(by: .radians(-tilt))
        phoneCtx.fill(Path(roundedRect: CGRect(x: -5, y: -22, width: 10, height: 44), cornerRadius: 4), with: .color(Theme.ink))

        ctx.draw(label("CEILING", Theme.ink3, size: 9), at: CGPoint(x: 22, y: ceil - 9), anchor: .leading)
        ctx.draw(label("FLOOR", Theme.ink3, size: 9), at: CGPoint(x: 22, y: floor + 10), anchor: .leading)
        ctx.draw(label(tilt > 0.2 ? "TILT UP" : tilt < -0.2 ? "TILT DOWN" : "", Theme.accent), at: CGPoint(x: phone.x, y: phone.y + 40))
    }

    // MARK: Finish where you started: the path drifts, then snaps when the loop closes.

    private static func loop(_ ctx: inout GraphicsContext, _ size: CGSize, _ t: Double) {
        let W = size.width, H = size.height
        let c = CGPoint(x: W / 2, y: H / 2 + 4)
        let rx = W * 0.34, ry = H * 0.32
        let period = 9.0, drawTime = 6.5
        let p = t.truncatingRemainder(dividingBy: period)
        let u = min(p / drawTime, 1)
        let closed = p >= drawTime
        let snap = closed ? max(0, 1 - (p - drawTime) / 0.6) : 1   // drift fades out on close

        ctx.stroke(Path(roundedRect: CGRect(x: c.x - rx - 22, y: c.y - ry - 22, width: 2 * (rx + 22), height: 2 * (ry + 22)),
                        cornerRadius: 6), with: .color(Theme.line), lineWidth: 2)

        func pt(_ s: Double, drift: Double) -> CGPoint {
            let a = .pi / 2 + s * 2 * .pi
            let wob = 0.12 * sin(3 * a)
            return CGPoint(x: c.x + rx * (1 + wob) * cos(a) + 18 * s * drift,
                           y: c.y + ry * (1 - wob) * sin(a) - 14 * s * drift)
        }
        var path = Path()
        let n = Int(120 * u)
        for i in 0...max(n, 1) {
            let q = pt(Double(i) / 120, drift: snap)
            i == 0 ? path.move(to: q) : path.addLine(to: q)
        }
        let color = closed ? Theme.good : Theme.trail
        ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))

        let start = pt(0, drift: 0)
        ctx.stroke(Path(ellipseIn: CGRect(x: start.x - 9, y: start.y - 9, width: 18, height: 18)), with: .color(Theme.ink), lineWidth: 2)
        ctx.draw(label("START", Theme.ink), at: CGPoint(x: start.x, y: start.y + 19))

        if !closed {
            let head = pt(u, drift: 1)
            ctx.fill(Path(ellipseIn: CGRect(x: head.x - 6, y: head.y - 6, width: 12, height: 12)), with: .color(Theme.trail))
            if u > 0.75 {
                ctx.draw(label("DRIFT \(Int(u * 16)) CM", Theme.warning), at: CGPoint(x: c.x, y: c.y))
            }
        } else {
            ctx.draw(symbol("checkmark.seal.fill", 26, Theme.good), at: CGPoint(x: c.x, y: c.y - 10))
            ctx.draw(label("LOOP CLOSED", Theme.good), at: CGPoint(x: c.x, y: c.y + 16))
        }
    }

    // MARK: Send it: the scan travels to a computer and becomes a 3D model.

    private static func send(_ ctx: inout GraphicsContext, _ size: CGSize, _ t: Double) {
        let W = size.width, H = size.height
        let phone = CGPoint(x: W * 0.17, y: H * 0.56)
        let laptop = CGPoint(x: W * 0.76, y: H * 0.60)
        ctx.draw(symbol("iphone", 50, Theme.ink), at: phone)
        ctx.draw(symbol("laptopcomputer", 92, Theme.ink), at: laptop)

        // Packets along an arc, staggered.
        for i in 0..<6 {
            let f = (t * 0.45 + Double(i) / 6).truncatingRemainder(dividingBy: 1)
            let x = phone.x + 22 + (laptop.x - 40 - phone.x - 22) * f
            let y = phone.y - 10 - sin(f * .pi) * H * 0.28
            let s: CGFloat = 9
            ctx.fill(Path(roundedRect: CGRect(x: x - s / 2, y: y - s / 2, width: s, height: s), cornerRadius: 2),
                     with: .color(Theme.accent.opacity(0.35 + 0.65 * sin(f * .pi))))
        }

        // A spinning point-cloud box on the laptop screen.
        let screen = CGPoint(x: laptop.x, y: laptop.y - 10)
        let a = t * 0.9
        for i in 0..<12 {
            for j in 0...6 {
                // 12 edges of a unit cube, 7 points each
                let e = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)][i]
                func corner(_ k: Int) -> SIMD3<Double> { SIMD3(Double(k & 1), Double((k >> 1) & 1), Double((k >> 2) & 1)) - 0.5 }
                let v = corner(e.0) + (corner(e.1) - corner(e.0)) * Double(j) / 6
                let x = v.x * cos(a) + v.z * sin(a), z = -v.x * sin(a) + v.z * cos(a)
                let y = v.y * cos(0.5) - z * sin(0.5)
                let r: CGFloat = 1.6
                ctx.fill(Path(ellipseIn: CGRect(x: screen.x + x * 30 - r, y: screen.y + y * 30 - r, width: 2 * r, height: 2 * r)),
                         with: .color(Theme.accent))
            }
        }
        ctx.draw(label("3D MODEL · ROBOT MAP", Theme.ink2, size: 9), at: CGPoint(x: laptop.x, y: H - 22))
        ctx.draw(label("SCAN", Theme.ink2, size: 9), at: CGPoint(x: phone.x, y: H - 22))
    }
}

#Preview {
    ScrollView { VStack(spacing: 16) { ForEach(1..<5) { OnboardingArt(step: $0) } }.padding() }
}

/// The team's plushie mascot (rendered from blender-harness), swaying gently from its feet.
struct PlushieMascot: View {
    var height: CGFloat

    var body: some View {
        TimelineView(.animation) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            Image("PlushieWave")
                .resizable()
                .scaledToFit()
                .frame(height: height)
                .rotationEffect(.degrees(3 * sin(t * 2.2)), anchor: .bottom)
                .offset(y: -2 * abs(sin(t * 2.2)))
        }
        .accessibilityLabel("Pioneer plushie waving")
    }
}
