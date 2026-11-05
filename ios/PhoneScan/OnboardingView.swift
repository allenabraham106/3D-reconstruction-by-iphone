import SwiftUI

/// First-launch guide: how to get a scan that reconstructs well. Reopen it from Settings.
struct OnboardingView: View {
    let hasLiDAR: Bool
    let done: () -> Void
    @State private var page = 0

    private struct Step: Identifiable {
        let id: Int
        let symbol: String
        let title: String
        let body: String
    }

    private var steps: [Step] {
        [
            Step(id: 0, symbol: "cube.transparent", title: "Map your home for Pioneer",
                 body: hasLiDAR
                    ? "Walk around and Pioneer Home saves photos, exact camera positions, LiDAR depth and a 3D mesh, all in real metres."
                    : "Walk around and Pioneer Home saves photos, exact camera positions and tracked 3D points, all in real metres. Your iPhone has no LiDAR, so the 3D model is built from the photos afterwards."),
            Step(id: 1, symbol: "figure.walk", title: "Walk slowly",
                 body: "About half your normal walking speed. Keep surfaces 0.5–3 m away. The Pace meter turns orange if you go too fast."),
            Step(id: 2, symbol: "arrow.up.and.down.and.arrow.left.and.right", title: "Sweep up and down",
                 body: "Tilt the phone to catch the floor, ceiling edges and furniture from a few heights. Aim at textured things, not blank walls."),
            Step(id: 3, symbol: "arrow.triangle.capsulepath", title: "Finish where you started",
                 body: "Coming back to your start point lets ARKit correct drift. The map shows your path, and the app tells you when the loop is closed."),
            Step(id: 4, symbol: "square.and.arrow.up", title: "Send it to your computer",
                 body: "Open Captures to AirDrop a scan, or upload it over Wi-Fi to the reconstruction server. It turns into a 3D model and a robot map."),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Pioneer Home").monoLabel(11)
                Spacer()
                Button("Skip", action: done).font(.subheadline).foregroundStyle(Theme.ink2)
            }
            .padding(.horizontal, 24)
            .padding(.top, 14)

            TabView(selection: $page) {
                ForEach(steps) { step in
                    VStack(alignment: .leading, spacing: 18) {
                        Spacer()
                        Image(systemName: step.symbol)
                            .font(.system(size: 44, weight: .ultraLight))
                            .foregroundStyle(Theme.ink)
                            .frame(width: 96, height: 96)
                            .paperCard(20)
                        Text(String(format: "%02d / %02d", step.id + 1, steps.count)).monoLabel(11, color: Theme.ink2)
                        Text(step.title)
                            .font(Theme.display(32))
                            .foregroundStyle(Theme.ink)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(step.body)
                            .font(.body)
                            .foregroundStyle(Theme.ink2)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer()
                        Spacer()
                    }
                    .padding(.horizontal, 28)
                    .tag(step.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            HStack(spacing: 6) {
                ForEach(steps) { s in
                    Capsule().fill(s.id == page ? Theme.ink : Theme.line).frame(width: s.id == page ? 18 : 6, height: 6)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: page)
            .frame(maxWidth: .infinity)
            .padding(.bottom, 18)

            Button {
                if page < steps.count - 1 { withAnimation { page += 1 } } else { done() }
            } label: {
                Text(page < steps.count - 1 ? "Next" : "Start")
                    .monoLabel(13, color: Theme.card)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Theme.ink, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 24)
            .padding(.bottom, 20)
        }
        .background(Theme.paper.ignoresSafeArea())
    }
}
