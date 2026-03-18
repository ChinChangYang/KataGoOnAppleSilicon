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
        var sgf = "(;FF[4]GM[1]SZ[\(boardSize)]"
        sgf += "PB[\(blackPlayer)]"
        sgf += "PW[\(whitePlayer)]"
        sgf += "KM[\(komi)]"

        if let result = result {
            sgf += "RE[\(result)]"
        }

        // Add moves
        for (stone, point) in moves {
            let sgfCoord = pointToSgf(point)
            let moveColor = stone == .black ? "B" : "W"
            sgf += ";\(moveColor)[\(sgfCoord)]"
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
    /// - Returns: SGF file content as a string
    public static func generateSGF(
        from board: Board,
        blackPlayer: String = "Black",
        whitePlayer: String = "White",
        komi: Float = 7.5,
        result: String? = nil
    ) -> String {
        // Convert Move array to (Stone, Point) array, filtering out pass moves
        let moves = board.moveHistory.compactMap { move -> (Stone, Point)? in
            guard let location = move.location else { return nil }
            return (move.player, location)
        }

        return generateSGF(
            moves: moves,
            blackPlayer: blackPlayer,
            whitePlayer: whitePlayer,
            komi: komi,
            result: result
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
