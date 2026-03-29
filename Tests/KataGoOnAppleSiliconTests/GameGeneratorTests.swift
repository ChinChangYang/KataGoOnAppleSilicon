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

    @Test("Debug A9 move on 9x9 board - reproduce user-reported issue")
    func testDebugA9MoveOn9x9() throws {
        print("\n=== Debug A9 Move on 9x9 Board ===")
        print("Reproducing user-reported issue where White plays A9 on 9x9\n")

        // Initialize KataGo with AI model
        let katago = KataGoInference()
        try katago.loadModel(for: "AI")
        print("✓ AI model loaded\n")

        // Initialize GTP handler
        let gtp = GTPHandler(katago: katago)

        // Reproduce the exact GTP command sequence from the user's log
        print("Setting up position from user's log...")
        let boardsizeResponse = gtp.handleCommand("boardsize 9")
        print("[GTP >>] boardsize 9")
        print("[GTP <<] \(boardsizeResponse.trimmingCharacters(in: .newlines))")
        #expect(boardsizeResponse.starts(with: "= "))

        let clearResponse = gtp.handleCommand("clear_board")
        print("[GTP >>] clear_board")
        print("[GTP <<] \(clearResponse.trimmingCharacters(in: .newlines))")

        let playC3 = gtp.handleCommand("play black C3")
        print("[GTP >>] play black C3")
        print("[GTP <<] \(playC3.trimmingCharacters(in: .newlines))")
        #expect(playC3.starts(with: "= "))

        let playE3 = gtp.handleCommand("play white E3")
        print("[GTP >>] play white E3")
        print("[GTP <<] \(playE3.trimmingCharacters(in: .newlines))")
        #expect(playE3.starts(with: "= "))

        let playG3 = gtp.handleCommand("play black G3")
        print("[GTP >>] play black G3")
        print("[GTP <<] \(playG3.trimmingCharacters(in: .newlines))")
        #expect(playG3.starts(with: "= "))

        // Show board state before genmove
        let showboard = gtp.handleCommand("showboard")
        print("\nBoard state before genmove:")
        print(showboard)

        // Run genmove white and check the result
        print("Generating move for White...")
        let genmoveResponse = gtp.handleCommand("genmove white")
        let moveStr = genmoveResponse
            .replacingOccurrences(of: "=", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        print("[GTP >>] genmove white")
        print("[GTP <<] \(genmoveResponse.trimmingCharacters(in: .newlines))")
        print("White plays: \(moveStr)")

        // Check that the move is NOT a 1-1 point (corner)
        let cornerMoves9x9 = ["A1", "A9", "J1", "J9"]
        if cornerMoves9x9.contains(moveStr) {
            print("⚠ WARNING: White played a 1-1 corner point (\(moveStr)) which is almost always bad in Go!")
        }

        // Show final board state
        let finalBoard = gtp.handleCommand("showboard")
        print("\nFinal board state:")
        print(finalBoard)

        print("\n✓ Debug test completed")
    }

    @Test("Generate 10-move game on 9x9 board and export to SGF")
    func testGenerateGame9x9() throws {
        print("\n=== KataGo 9x9 Game Generator Test ===")
        print("Generating a 10-move game on 9x9 board...\n")

        // Initialize KataGo
        let katago = KataGoInference()
        try katago.loadModel(for: "AI")
        print("✓ AI model loaded\n")

        // Initialize GTP handler
        let gtp = GTPHandler(katago: katago)

        // Set board size to 9x9
        _ = gtp.handleCommand("boardsize 9")
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
                let moveStr = response
                    .replacingOccurrences(of: "=", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                if moveStr == "pass" || moveStr == "resign" {
                    let colorStr = color == "black" ? "Black" : "White"
                    let moveNumStr = String(format: "%4d", moveNum)
                    print("\(moveNumStr) | \(colorStr.padding(toLength: 5, withPad: " ", startingAt: 0)) | \(moveStr.padding(toLength: 9, withPad: " ", startingAt: 0)) | -")
                    continue
                }

                // Convert to SGF
                let sgfCoord = SGFGenerator.gtpToSgf(moveStr)

                // Parse move to get Point
                if let point = parseGTPMove(moveStr, boardSize: 9) {
                    moves.append((stone, point))

                    let colorStr = color == "black" ? "Black" : "White"
                    let moveNumStr = String(format: "%4d", moveNum)
                    print("\(moveNumStr) | \(colorStr.padding(toLength: 5, withPad: " ", startingAt: 0)) | \(moveStr.padding(toLength: 9, withPad: " ", startingAt: 0)) | \(sgfCoord)")

                    // Check for 1-1 corner moves (bad moves)
                    let cornerMoves = ["A1", "A9", "J1", "J9"]
                    if cornerMoves.contains(moveStr) {
                        print("  ⚠ WARNING: 1-1 corner move detected!")
                    }
                } else {
                    Issue.record("Failed to parse move \(moveStr)")
                    return
                }
            } else {
                Issue.record("Error generating move \(moveNum): \(response)")
                return
            }
        }

        // Show final board
        let finalBoard = gtp.handleCommand("showboard")
        print("\nFinal board state:")
        print(finalBoard)

        // Generate SGF
        print("\n=== Generating SGF ===")
        let sgf = SGFGenerator.generateSGF(
            moves: moves,
            blackPlayer: "KataGo (Black)",
            whitePlayer: "KataGo (White)",
            komi: 7.5,
            boardSize: 9
        )

        // Save SGF file
        let fileManager = FileManager.default
        let buildOutputDir = ".build/test-output"
        try fileManager.createDirectory(atPath: buildOutputDir, withIntermediateDirectories: true, attributes: nil)
        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "\(buildOutputDir)/game_9x9_\(timestamp).sgf"
        try SGFGenerator.saveSGF(sgf, to: filename)

        print("✓ SGF file saved: \(filename)")
        print("\nSGF Content:")
        print(sgf)

        #expect(sgf.contains("SZ[9]"))
        print("\n✓ 9x9 game generation test passed")
    }

    /// Helper function to parse GTP move string to Point
    func parseGTPMove(_ moveStr: String, boardSize: Int = 19) -> Point? {
        guard moveStr.count >= 2 else { return nil }

        let colChar = moveStr.first!
        let rowStr = String(moveStr.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= boardSize else { return nil }

        var col: Int
        if colChar >= "A" && colChar <= "H" {
            col = Int(colChar.asciiValue! - 65)  // A=0, B=1, ..., H=7
        } else if colChar >= "J" && colChar <= "T" {
            col = Int(colChar.asciiValue! - 65) - 1  // J=8, K=9, ..., T=18 (skip I)
        } else {
            return nil
        }

        let y = boardSize - row

        return Point(x: col, y: y)
    }
}
