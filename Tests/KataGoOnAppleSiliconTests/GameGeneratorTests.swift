import Foundation
import Testing
@testable import KataGoOnAppleSilicon

/// Test for game generation and SGF export
@Suite("Game Generator Tests")
struct GameGeneratorTests {
    @Test("Generate 10-move game and export to SGF")
    func testGenerateGame() throws {
        print("\n=== KataGo Game Generator Test ===")
        print("Generating a 10-move game for debugging...\n")

        // Initialize KataGo
        let katago = KataGoInference()
        try katago.loadModel(for: "AI")
        print("✓ AI model loaded\n")

        // Initialize GTP handler
        let gtp = GTPHandler(katago: katago)

        // Clear board
        _ = gtp.handleCommand("clear_board")

        // Track moves for SGF generation
        var moves: [(Stone, Point)] = []

        print("Move | Color | GTP Coord | SGF Coord")
        print("-----|-------|-----------|----------")

        // Generate 10 moves alternating between black and white
        for moveNum in 1...10 {
            let color = moveNum % 2 == 1 ? "black" : "white"
            let stone: Stone = color == "black" ? .black : .white

            // Generate move
            let response = gtp.handleCommand("genmove \(color)")

            // Parse response
            if response.starts(with: "=") {
                // Extract move from response (format: "= D4\n\n")
                let moveStr = response
                    .replacingOccurrences(of: "=", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                // Convert to SGF
                let sgfCoord = SGFGenerator.gtpToSgf(moveStr)

                // Parse move to get Point
                if let point = parseGTPMove(moveStr) {
                    moves.append((stone, point))

                    // Print move info
                    let colorStr = color == "black" ? "Black" : "White"
                    let moveNumStr = String(format: "%4d", moveNum)
                    print("\(moveNumStr) | \(colorStr.padding(toLength: 5, withPad: " ", startingAt: 0)) | \(moveStr.padding(toLength: 9, withPad: " ", startingAt: 0)) | \(sgfCoord)")
                } else {
                    Issue.record("Failed to parse move \(moveStr)")
                    return
                }
            } else {
                Issue.record("Error generating move \(moveNum): \(response)")
                return
            }
        }

        // Generate SGF
        print("\n=== Generating SGF ===")
        let sgf = SGFGenerator.generateSGF(
            moves: moves,
            blackPlayer: "KataGo (Black)",
            whitePlayer: "KataGo (White)",
            komi: 7.5
        )

        // Save SGF file to .build/test-output/ (which is in .gitignore)
        let fileManager = FileManager.default
        let buildOutputDir = ".build/test-output"

        // Create directory if it doesn't exist
        try fileManager.createDirectory(atPath: buildOutputDir, withIntermediateDirectories: true, attributes: nil)

        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "\(buildOutputDir)/game_\(timestamp).sgf"
        try SGFGenerator.saveSGF(sgf, to: filename)

        print("✓ SGF file saved: \(filename)")
        print("\nSGF Content:")
        print(sgf)

        // Verify SGF format
        #expect(sgf.hasPrefix("(;FF[4]GM[1]SZ[19]"))
        #expect(sgf.contains("PB[KataGo (Black)]"))
        #expect(sgf.contains("PW[KataGo (White)]"))
        #expect(sgf.contains("KM[7.5]"))
        #expect(sgf.hasSuffix(")"))

        print("\n✓ Game generation test passed")
    }

    @Test("Generate 10-move game with 20k model and export to SGF")
    func testGenerateGameWith20kModel() throws {
        print("\n=== KataGo Game Generator Test (20k Model) ===")
        print("Generating a 10-move game with 20k human SL model...\n")

        // Initialize KataGo
        let katago = KataGoInference()
        try katago.loadModel(for: "20k")
        print("✓ 20k model loaded\n")

        // Initialize GTP handler
        let gtp = GTPHandler(katago: katago)
        gtp.setProfile("20k")  // Use 20k profile for inference

        // Clear board
        _ = gtp.handleCommand("clear_board")

        // Track moves for SGF generation
        var moves: [(Stone, Point)] = []

        print("Move | Color | GTP Coord | SGF Coord")
        print("-----|-------|-----------|----------")

        // Generate 10 moves alternating between black and white
        for moveNum in 1...10 {
            let color = moveNum % 2 == 1 ? "black" : "white"
            let stone: Stone = color == "black" ? .black : .white

            // Generate move
            let response = gtp.handleCommand("genmove \(color)")

            // Parse response
            if response.starts(with: "=") {
                // Extract move from response (format: "= D4\n\n")
                let moveStr = response
                    .replacingOccurrences(of: "=", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                // Convert to SGF
                let sgfCoord = SGFGenerator.gtpToSgf(moveStr)

                // Parse move to get Point
                if let point = parseGTPMove(moveStr) {
                    moves.append((stone, point))

                    // Print move info
                    let colorStr = color == "black" ? "Black" : "White"
                    let moveNumStr = String(format: "%4d", moveNum)
                    print("\(moveNumStr) | \(colorStr.padding(toLength: 5, withPad: " ", startingAt: 0)) | \(moveStr.padding(toLength: 9, withPad: " ", startingAt: 0)) | \(sgfCoord)")
                } else {
                    Issue.record("Failed to parse move \(moveStr)")
                    return
                }
            } else {
                Issue.record("Error generating move \(moveNum): \(response)")
                return
            }
        }

        // Generate SGF
        print("\n=== Generating SGF ===")
        let sgf = SGFGenerator.generateSGF(
            moves: moves,
            blackPlayer: "KataGo (Black)",
            whitePlayer: "KataGo (White)",
            komi: 7.5
        )

        // Save SGF file to .build/test-output/ (which is in .gitignore)
        let fileManager = FileManager.default
        let buildOutputDir = ".build/test-output"

        // Create directory if it doesn't exist
        try fileManager.createDirectory(atPath: buildOutputDir, withIntermediateDirectories: true, attributes: nil)

        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "\(buildOutputDir)/game_20k_\(timestamp).sgf"
        try SGFGenerator.saveSGF(sgf, to: filename)

        print("✓ SGF file saved: \(filename)")
        print("\nSGF Content:")
        print(sgf)

        // Verify SGF format
        #expect(sgf.hasPrefix("(;FF[4]GM[1]SZ[19]"))
        #expect(sgf.contains("PB[KataGo (Black)]"))
        #expect(sgf.contains("PW[KataGo (White)]"))
        #expect(sgf.contains("KM[7.5]"))
        #expect(sgf.hasSuffix(")"))

        print("\n✓ Game generation test (20k model) passed")
    }

    /// Helper function to parse GTP move string to Point
    func parseGTPMove(_ moveStr: String) -> Point? {
        guard moveStr.count >= 2 else { return nil }

        let colChar = moveStr.first!
        let rowStr = String(moveStr.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= 19 else { return nil }

        var col: Int
        if colChar >= "A" && colChar <= "H" {
            col = Int(colChar.asciiValue! - 65)  // A=0, B=1, ..., H=7
        } else if colChar >= "J" && colChar <= "T" {
            col = Int(colChar.asciiValue! - 65) - 1  // J=8, K=9, ..., T=18 (skip I)
        } else {
            return nil
        }

        // GTP: 1 is top (y=18), 19 is bottom (y=0)
        let y = 19 - row

        return Point(x: col, y: y)
    }
}
