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

    /// Generate SGF content from a sequence of placed moves (no passes, no
    /// setup stones). Retained for the existing callers in tests and the
    /// `KataGoPlay` executable; richer output goes through `from board:`.
    public static func generateSGF(
        moves: [(Stone, Point)],
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float = 7.5,
        result: String? = nil,
        boardSize: Int = 19
    ) -> String {
        return buildSGF(
            movesWithPass: moves.map { ($0.0, Optional<Point>.some($0.1)) },
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

    /// Generate SGF content from a board, preserving passes, setup stones,
    /// and (when non-default) the starting side to move.
    public static func generateSGF(
        from board: Board,
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float? = nil,
        result: String? = nil,
        rulesName: String? = nil
    ) -> String {
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

        // SGF default starting player is white iff black handicap stones
        // exist. Mirrors `SGFParser`'s default so a round-trip is stable.
        let defaultPla: Stone = handicapBlack.isEmpty ? .black : .white
        let initialPla: Stone? = board.initialSideToMove == defaultPla ? nil : board.initialSideToMove

        return buildSGF(
            movesWithPass: board.moveHistory.map { ($0.player, $0.location) },
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

    private static func buildSGF(
        movesWithPass: [(Stone, Point?)],
        blackPlayer: String,
        whitePlayer: String,
        komi: Float,
        result: String?,
        boardSize: Int,
        handicapBlack: [Point],
        handicapWhite: [Point],
        rulesName: String?,
        initialSideToMove: Stone?
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
            sgf += "AB"
            for p in handicapBlack { sgf += "[\(pointToSgf(p))]" }
        }
        if !handicapWhite.isEmpty {
            sgf += "AW"
            for p in handicapWhite { sgf += "[\(pointToSgf(p))]" }
        }
        if let pla = initialSideToMove {
            sgf += "PL[\(pla == .black ? "B" : "W")]"
        }
        if let result = result {
            sgf += "RE[\(result)]"
        }

        for (stone, point) in movesWithPass {
            let moveColor = stone == .black ? "B" : "W"
            if let point = point {
                sgf += ";\(moveColor)[\(pointToSgf(point))]"
            } else {
                sgf += ";\(moveColor)[]"
            }
        }

        sgf += ")"
        return sgf
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
