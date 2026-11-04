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
        VStack(spacing: 0) {
            HStack {
                Wordmark()
                Spacer()
                Button("Skip", action: done).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 24)
            .padding(.top, 12)
            TabView(selection: $page) {
                ForEach(steps) { step in
                    VStack(spacing: 22) {
                        Spacer()
                        ZStack {
                            Circle().fill(Theme.teal.opacity(0.12)).frame(width: 170, height: 170)
                                .blur(radius: 18)
                            Circle().fill(Theme.surface).frame(width: 150, height: 150)
                                .overlay(Circle().stroke(Theme.teal.opacity(0.45), lineWidth: 1.5))
                            Image(systemName: step.symbol)
                                .font(.system(size: 60, weight: .semibold))
                                .foregroundStyle(Theme.teal)
                                .symbolRenderingMode(.hierarchical)
                        }
                        Text(step.title)
                            .font(.system(size: 30, weight: .heavy))
                            .fontWidth(.expanded)
                            .multilineTextAlignment(.center)
                        Text(step.body)
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                        Spacer()
                        Spacer()
                    }
                    .tag(step.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .indexViewStyle(.page(backgroundDisplayMode: .always))

            Button {
                if page < steps.count - 1 { withAnimation { page += 1 } } else { done() }
            } label: {
                Text(page < steps.count - 1 ? "Next" : "Start scanning")
                    .font(.headline)
                    .watoLabel()
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Theme.primary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .foregroundStyle(.white)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 24)
        }
        .background(
            LinearGradient(colors: [Theme.navy, Color(hex: 0x07131B)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
        )
    }
}
