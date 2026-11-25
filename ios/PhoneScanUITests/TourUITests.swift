import XCTest

/// Walks every screen: onboarding, Home, the 3D map, Scans, Robot, Settings.
/// Runs on the simulator (no camera, so no capture). With a robot_link server reachable at
/// ROBOT_URL (default http://127.0.0.1:8770) it also checks that the app connects and shows the map:
///
///     python -m robot_link.server --maps out --host 127.0.0.1
///     xcodebuild test -project ios/PhoneScan.xcodeproj -scheme PhoneScan -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
final class TourUITests: XCTestCase {
    /// Slows the tour down so a screen recording of it reads as a demo.
    let pause = Double(ProcessInfo.processInfo.environment["TOUR_PAUSE"] ?? "0") ?? 0
    /// Set SHOT_DIR (a Mac folder; the simulator shares the file system) to save a PNG per screen.
    let shotDir = ProcessInfo.processInfo.environment["SHOT_DIR"]

    func testTour() throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        let robot = ProcessInfo.processInfo.environment["ROBOT_URL"] ?? "http://127.0.0.1:8770"
        app.launchArguments += ["-didOnboard", "NO", "-robotURL", robot]
        app.launch()

        // Onboarding: five steps, then the app.
        XCTAssertTrue(app.staticTexts["Step 1 of 5"].waitForExistence(timeout: 10))
        shot("onboarding-1")
        for step in 2...5 {
            hold()
            app.buttons["Next"].tap()
            XCTAssertTrue(app.staticTexts["Step \(step) of 5"].waitForExistence(timeout: 5))
            shot("onboarding-\(step)")
        }
        hold()
        app.buttons["Get started"].tap()

        // Home
        let tabs = ["Home", "Scans", "Robot", "Settings"]
        for t in tabs { XCTAssertTrue(app.buttons[t].waitForExistence(timeout: 10), "tab \(t)") }
        hold(2)

        // Robot tab: connect to the server.
        app.buttons["Robot"].tap()
        let connect = app.buttons.matching(NSPredicate(format: "label IN %@", ["Connect", "Reconnect"])).firstMatch
        XCTAssertTrue(connect.waitForExistence(timeout: 5))
        connect.tap()
        let connected = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'living-room'")).firstMatch
        XCTAssertTrue(connected.waitForExistence(timeout: 15), "robot map listed")
        hold(2)

        // Home now shows the robot ready: localize it (with --camera file:... the server has a camera frame).
        app.buttons["Home"].tap()
        hold(1.5)
        let localize = app.buttons.matching(NSPredicate(format: "label IN %@", ["Localize", "Localize again"])).firstMatch
        XCTAssertTrue(localize.waitForExistence(timeout: 5))
        if localize.isEnabled {   // needs a robot camera (server --camera); without one the tile stays disabled
            localize.tap()
            let found = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'matches'")).firstMatch
            XCTAssertTrue(found.waitForExistence(timeout: 20), "robot localized")
            shot("localized")
            hold(2.5)
        }
        app.buttons["Map"].firstMatch.tap()
        let fly = app.buttons["Fly through the scan"]
        XCTAssertTrue(fly.waitForExistence(timeout: 15))
        hold(3)
        shot("map")
        fly.tap()
        hold(8)
        shot("map-fly")
        app.buttons["Map"].firstMatch.tap()   // the back button is labelled Map
        hold()

        // Scans
        app.buttons["Scans"].tap()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'frames'")).firstMatch.waitForExistence(timeout: 5))
        shot("scans")
        hold(2)

        // Settings
        app.buttons["Settings"].tap()
        shot("settings")
        hold(2)
        app.swipeUp()
        hold(2)
        app.buttons["Home"].tap()
        hold(2)
    }

    private func shot(_ name: String) {
        guard let shotDir else { return }
        Thread.sleep(forTimeInterval: 1.5)   // let the page settle and the art animate
        try? XCUIScreen.main.screenshot().pngRepresentation.write(to: URL(fileURLWithPath: shotDir).appendingPathComponent("\(name).png"))
    }

    private func hold(_ seconds: Double = 1) {
        if pause > 0 { Thread.sleep(forTimeInterval: seconds * pause) }
    }
}
