import Testing
import Foundation
@testable import KataGoOnAppleSilicon

/// Cross-engine SGF interoperability tests for `printsgf` and `loadsgf`.
///
/// Each scenario has three committed fixture files under `SGFFixtures/`:
///   - `<scenario>.export.sgf`  — our printsgf when driving the position natively
///   - `<scenario>.katago.sgf`  — KataGo's printsgf at the same position
///   - `<scenario>.import.sgf`  — our printsgf after loadsgf'ing the KataGo SGF
///
/// `interop_export_<scenario>` drives the scenario via Swift's GTPHandler
/// then byte-compares our printsgf to `<scenario>.export.sgf`.
///
/// `interop_import_<scenario>` loadsgf's `<scenario>.katago.sgf` then
/// byte-compares our printsgf to `<scenario>.import.sgf`.
///
/// Fixtures are produced offline by `Scripts/generate_sgf_interop_fixtures.sh`.
/// Missing fixtures throw `InteropError.fixtureMissing`.
struct SGFInteropTests {

    enum InteropError: Error, CustomStringConvertible {
        case fixtureMissing(String)
        var description: String {
            switch self {
            case .fixtureMissing(let path):
                return """
                Fixture not found: \(path)
                Run: Scripts/generate_sgf_interop_fixtures.sh
                """
            }
        }
    }

    // MARK: - Helpers

    /// Walks up from this source file to the repo root, then descends into
    /// `subdir/name`. Mirrors the pattern used by GTPFixtureTests so the
    /// fixture lookup logic is consistent across integration tests.
    private func repoFile(subdir: String, name: String) throws -> URL {
        let fm = FileManager.default
        var here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = here
                .appendingPathComponent(subdir, isDirectory: true)
                .appendingPathComponent(name)
            if fm.fileExists(atPath: candidate.path) { return candidate }
            here.deleteLastPathComponent()
        }
        throw InteropError.fixtureMissing("\(subdir)/\(name)")
    }

    private func loadFixture(_ name: String) throws -> String {
        let url = try repoFile(
            subdir: "Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures",
            name: name
        )
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func fixtureURL(_ name: String) throws -> URL {
        try repoFile(
            subdir: "Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures",
            name: name
        )
    }

    private func loadDriver(_ name: String) throws -> String {
        let url = try repoFile(subdir: "Scripts/SGFFixtureDrivers", name: name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Strip the GTP `= ` prefix and trailing `\n\n` to recover the SGF body.
    private func sgfPayload(_ response: String) -> String? {
        guard response.hasPrefix("= ") else { return nil }
        var body = String(response.dropFirst(2))
        if body.hasSuffix("\n\n") {
            body = String(body.dropLast(2))
        }
        return body
    }

    /// Side-by-side byte dump for a `#expect` failure message. Returns
    /// `String` rather than `Testing.Comment`; mirrors the convention
    /// already used in `GTPFixtureTests` (call sites wrap with
    /// `"\(diffMessage(...))"` so the literal string interpolation
    /// satisfies `Comment`'s `ExpressibleByStringInterpolation`).
    private func diffMessage(actual: String, expected: String, label: String) -> String {
        return """
        \(label) mismatch.
        --- actual (\(actual.count) bytes) ---
        \(actual)
        --- expected (\(expected.count) bytes) ---
        \(expected)
        """
    }

    /// Replay a driver's GTP commands through a fresh handler, then return
    /// the body of our `printsgf` reply.
    private func driveAndPrint(driver: String) throws -> String {
        let handler = GTPHandler(katago: KataGoInference())
        for raw in driver.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            _ = handler.handleCommand(line)
        }
        return try #require(sgfPayload(handler.handleCommand("printsgf")))
    }

    // MARK: - Tests

    @Test func interop_export_empty() throws {
        let expected = try loadFixture("empty.export.sgf")
        let driver = try loadDriver("empty.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_empty"))")
    }

    @Test func interop_export_captures() throws {
        let expected = try loadFixture("captures.export.sgf")
        let driver = try loadDriver("captures.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_captures"))")
    }

    @Test func interop_export_handicap_5() throws {
        let expected = try loadFixture("handicap_5.export.sgf")
        let driver = try loadDriver("handicap_5.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_handicap_5"))")
    }

    @Test func interop_export_komi_nondefault() throws {
        let expected = try loadFixture("komi_nondefault.export.sgf")
        let driver = try loadDriver("komi_nondefault.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_komi_nondefault"))")
    }

    @Test func interop_export_moves_basic() throws {
        let expected = try loadFixture("moves_basic.export.sgf")
        let driver = try loadDriver("moves_basic.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_moves_basic"))")
    }

    @Test func interop_export_pass_midgame() throws {
        let expected = try loadFixture("pass_midgame.export.sgf")
        let driver = try loadDriver("pass_midgame.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_pass_midgame"))")
    }

    @Test func interop_export_rules_chinese() throws {
        let expected = try loadFixture("rules_chinese.export.sgf")
        let driver = try loadDriver("rules_chinese.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_rules_chinese"))")
    }

    @Test func interop_import_empty() throws {
        let expected = try loadFixture("empty.import.sgf")
        let engineSGFURL = try fixtureURL("empty.katago.sgf")

        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_empty"))")
    }
}
