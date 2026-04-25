import Foundation

/// Generates SGF (Smart Game Format) files from board history
public struct SGFGenerator {
    /// Convert GTP coordinate (e.g., "C4", "D16") to SGF coordinate (e.g., "cc", "dp")
    /// - Parameter gtpCoord: GTP coordinate string (e.g., "C4")
    /// - Returns: SGF coordinate string (e.g., "cc")
    public static func gtpToSgf(_ gtpCoord: String) -> String {
        guard gtpCoord.count >= 2 else { return "" }

        let colChar = gtpCoord.first!
        let rowStr = String(gtpCoord.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= 19 else { return "" }

        // Convert GTP column letter (A-T, skip I) to SGF column letter (a-s)
        var col: Int
        if colChar >= "A" && colChar <= "H" {
            col = Int(colChar.asciiValue! - 65)  // A=0, B=1, ..., H=7
        } else if colChar >= "J" && colChar <= "T" {
            col = Int(colChar.asciiValue! - 65) - 1  // J=8, K=9, ..., T=18 (skip I)
        } else {
            return ""
        }

        // Convert GTP row (1=top, 19=bottom) to SGF row (0=top, 18=bottom)
        let sgfRow = 19 - row

        // Convert to SGF format using lowercase letters
        let sgfChars = "abcdefghijklmnopqrs"
        let sgfColChar = sgfChars[sgfChars.index(sgfChars.startIndex, offsetBy: col)]
        let sgfRowChar = sgfChars[sgfChars.index(sgfChars.startIndex, offsetBy: sgfRow)]

        return "\(sgfColChar)\(sgfRowChar)"
    }

    /// Convert Point to SGF coordinate
    /// - Parameter point: Board point (x: 0-18, y: 0-18)
    /// - Returns: SGF coordinate string (e.g., "cc")
    public static func pointToSgf(_ point: Point) -> String {
        let sgfChars = "abcdefghijklmnopqrs"
        let sgfColChar = sgfChars[sgfChars.index(sgfChars.startIndex, offsetBy: point.x)]
        let sgfRowChar = sgfChars[sgfChars.index(sgfChars.startIndex, offsetBy: point.y)]
        return "\(sgfColChar)\(sgfRowChar)"
    }

    /// Generate SGF content from a sequence of moves
    /// - Parameters:
    ///   - moves: Array of (stone, point) tuples representing the game moves
    ///   - blackPlayer: Name of the black player (default: "Black")
    ///   - whitePlayer: Name of the white player (default: "White")
    ///   - komi: Komi value (default: 7.5)
    ///   - result: Game result string (e.g., "B+R", "W+2.5", optional)
    /// - Returns: SGF file content as a string
    public static func generateSGF(
        moves: [(Stone, Point)],
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float = 7.5,
        result: String? = nil,
        boardSize: Int = 19
    ) -> String {
        let movesWithPass: [(Stone, Point?)] = moves.map { ($0.0, $0.1) }
        return generateSGF(
            movesWithPass: movesWithPass,
            blackPlayer: blackPlayer,
            whitePlayer: whitePlayer,
            komi: komi,
            result: result,
            boardSize: boardSize,
            handicapBlack: [],
            handicapWhite: [],
            rulesName: nil,
            initialSideToMove: nil
        )
    }

    /// Generate SGF content from a sequence of moves that may include passes,
    /// optional handicap stones, an explicit rules name, and an initial side to move.
    /// - Parameters:
    ///   - movesWithPass: Each entry is (player, location-or-nil); nil means pass.
    ///   - handicapBlack: Initial black stones (added via `AB[]`).
    ///   - handicapWhite: Initial white stones (added via `AW[]`).
    ///   - rulesName: Optional SGF rules name written as `RU[<name>]`.
    ///   - initialSideToMove: Optional player-to-move written as `PL[B]` or `PL[W]`.
    public static func generateSGF(
        movesWithPass: [(Stone, Point?)],
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float = 7.5,
        result: String? = nil,
        boardSize: Int = 19,
        handicapBlack: [Point] = [],
        handicapWhite: [Point] = [],
        rulesName: String? = nil,
        initialSideToMove: Stone? = nil
    ) -> String {
        var sgf = "(;FF[4]GM[1]SZ[\(boardSize)]"
        sgf += "PB[\(blackPlayer)]"
        sgf += "PW[\(whitePlayer)]"
        sgf += "KM[\(komi)]"

        if let rulesName = rulesName {
            sgf += "RU[\(rulesName)]"
        }
        if !handicapBlack.isEmpty {
            sgf += "HA[\(handicapBlack.count)]"
        }

        if !handicapBlack.isEmpty {
            sgf += "AB"
            for p in handicapBlack {
                sgf += "[\(pointToSgf(p))]"
            }
        }
        if !handicapWhite.isEmpty {
            sgf += "AW"
            for p in handicapWhite {
                sgf += "[\(pointToSgf(p))]"
            }
        }
        if let pla = initialSideToMove {
            sgf += "PL[\(pla == .black ? "B" : "W")]"
        }

        if let result = result {
            sgf += "RE[\(result)]"
        }

        // Add moves
        for (stone, point) in movesWithPass {
            let moveColor = stone == .black ? "B" : "W"
            if let point = point {
                sgf += ";\(moveColor)[\(pointToSgf(point))]"
            } else {
                // Pass move: empty SGF coordinate.
                sgf += ";\(moveColor)[]"
            }
        }

        sgf += ")"
        return sgf
    }

    /// Generate SGF content from a board with move history
    /// - Parameters:
    ///   - board: The board containing the game state
    ///   - blackPlayer: Name of the black player (default: "Black")
    ///   - whitePlayer: Name of the white player (default: "White")
    ///   - komi: Komi value (default: 7.5)
    ///   - result: Game result string (e.g., "B+R", "W+2.5", optional)
    ///   - rulesName: Optional rules name to write as `RU[]`.
    /// - Returns: SGF file content as a string
    public static func generateSGF(
        from board: Board,
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float? = nil,
        result: String? = nil,
        rulesName: String? = nil
    ) -> String {
        // Preserve passes in the SGF output.
        let moves = board.moveHistory.map { ($0.player, $0.location) }

        // Extract handicap/setup stones from initialStones.
        var handicapBlack: [Point] = []
        var handicapWhite: [Point] = []
        for y in 0..<board.ySize {
            for x in 0..<board.xSize {
                switch board.initialStones[y][x] {
                case .black: handicapBlack.append(Point(x: x, y: y))
                case .white: handicapWhite.append(Point(x: x, y: y))
                default: break
                }
            }
        }

        // Only emit PL[] when the initial side to move differs from the SGF default
        // (black moves first, or white when there are handicap stones).
        let defaultPla: Stone = handicapBlack.isEmpty && handicapWhite.isEmpty ? .black : .white
        let initialPla: Stone? = board.initialSideToMove == defaultPla ? nil : board.initialSideToMove

        return generateSGF(
            movesWithPass: moves,
            blackPlayer: blackPlayer,
            whitePlayer: whitePlayer,
            komi: komi ?? board.komi,
            result: result,
            boardSize: board.xSize,
            handicapBlack: handicapBlack,
            handicapWhite: handicapWhite,
            rulesName: rulesName,
            initialSideToMove: initialPla
        )
    }

    /// Save SGF content to a file
    /// - Parameters:
    ///   - sgfContent: The SGF file content
    ///   - filename: The filename to save to
    /// - Throws: File I/O errors
    public static func saveSGF(_ sgfContent: String, to filename: String) throws {
        let url = URL(fileURLWithPath: filename)
        try sgfContent.write(to: url, atomically: true, encoding: .utf8)
    }
}
