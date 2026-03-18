import Foundation
import KataGoOnAppleSilicon

enum ANSI {
    static let reset  = "\u{1B}[0m"
    static let bold   = "\u{1B}[1m"
    static let cyan   = "\u{1B}[36m"
    static let yellow = "\u{1B}[33m"
    static let red    = "\u{1B}[31m"
}

private func isStarPoint(x: Int, y: Int, boardSize: Int) -> Bool {
    switch boardSize {
    case 9:
        let stars: Set<String> = ["2,2","4,2","6,2","2,4","4,4","6,4","2,6","4,6","6,6"]
        return stars.contains("\(x),\(y)")
    case 13:
        let stars: Set<String> = ["3,3","6,3","9,3","3,6","6,6","9,6","3,9","6,9","9,9"]
        return stars.contains("\(x),\(y)")
    default: // 19
        let stars: Set<String> = [
            "3,3","15,3","3,15","15,15",
            "3,9","15,9","9,3","9,15","9,9"
        ]
        return stars.contains("\(x),\(y)")
    }
}

/// Convert a GTP coordinate string (e.g. "D4") to internal (x, y).
/// Internal: x=0 left, y=0 top (GTP row boardSize).
func gtpToPoint(_ coord: String, boardSize: Int = 19) -> Point? {
    let upper = coord.uppercased()
    guard !upper.isEmpty else { return nil }
    let col = upper.first!
    let rowStr = String(upper.dropFirst())
    guard let row = Int(rowStr), row >= 1, row <= boardSize else { return nil }
    let x: Int
    if col >= "A" && col <= "H" {
        x = Int(col.asciiValue! - 65)
    } else if col >= "J" && col <= "T" {
        x = Int(col.asciiValue! - 65) - 1
    } else {
        return nil
    }
    return Point(x: x, y: boardSize - row)
}

/// Convert internal (x, y) to GTP coordinate string.
func pointToGTP(x: Int, y: Int, boardSize: Int = 19) -> String {
    let colScalar: UnicodeScalar = x < 8
        ? UnicodeScalar(65 + x)!
        : UnicodeScalar(66 + x)!   // skip I
    let col = Character(colScalar)
    let row = boardSize - y
    return "\(col)\(row)"
}

private let allCols = "A B C D E F G H J K L M N O P Q R S T"

private func colHeaderLine(boardSize: Int) -> String {
    let cols = allCols.split(separator: " ").prefix(boardSize).joined(separator: " ")
    return "   \(cols)"
}

/// Parse a showboard GTP response into a boardSize×boardSize grid of optional Stone values.
/// Grid indexing: grid[y][x], y=0 = top row (GTP row boardSize).
func parseShowboard(_ response: String, boardSize: Int = 19) -> [[Stone?]] {
    var grid: [[Stone?]] = Array(
        repeating: Array(repeating: nil, count: boardSize),
        count: boardSize
    )

    var content = response
    if content.hasPrefix("= ") {
        content = String(content.dropFirst(2))
    }

    for line in content.components(separatedBy: "\n") {
        // Line format: "19 X O . ..." or " 1 X O . ..."
        // First 2 chars = row number (right-aligned), then space, then cells
        guard line.count > 3 else { continue }

        let prefix = String(line.prefix(2)).trimmingCharacters(in: .whitespaces)
        guard let rowNum = Int(prefix), rowNum >= 1, rowNum <= boardSize else { continue }
        let y = boardSize - rowNum

        let cellsPart = String(line.dropFirst(3))
        let cells = cellsPart.split(separator: " ", omittingEmptySubsequences: true)
        for (x, cell) in cells.enumerated() where x < boardSize {
            switch cell {
            case "X": grid[y][x] = .black
            case "O": grid[y][x] = .white
            default:  grid[y][x] = nil
            }
        }
    }

    return grid
}

/// Render the board with optional last-move highlight and hint overlays.
func renderBoard(
    grid: [[Stone?]],
    boardSize: Int = 19,
    lastMove: String? = nil,
    hints: [(coord: String, prob: Double)] = []
) {
    // Build hint rank map keyed by "x,y"
    var hintMap: [String: Int] = [:]
    for (i, hint) in hints.prefix(5).enumerated() {
        if let pt = gtpToPoint(hint.coord, boardSize: boardSize) {
            hintMap["\(pt.x),\(pt.y)"] = i + 1
        }
    }

    let lastMoveXY: (x: Int, y: Int)?
    if let lm = lastMove, lm.lowercased() != "pass", let pt = gtpToPoint(lm, boardSize: boardSize) {
        lastMoveXY = (x: pt.x, y: pt.y)
    } else {
        lastMoveXY = nil
    }

    let header = colHeaderLine(boardSize: boardSize)
    print(header)

    for y in 0..<boardSize {
        let rowNum = boardSize - y
        let prefix = rowNum < 10 ? " \(rowNum)" : "\(rowNum)"
        var cells: [String] = []

        for x in 0..<boardSize {
            let isLastMove = lastMoveXY.map { $0.x == x && $0.y == y } ?? false
            let key = "\(x),\(y)"

            let cell: String
            if let stone = grid[y][x] {
                let symbol = stone == .black ? "●" : "○"
                cell = isLastMove ? "\(ANSI.red)\(ANSI.bold)\(symbol)\(ANSI.reset)" : symbol
            } else if let rank = hintMap[key] {
                cell = "\(ANSI.cyan)\(rank)\(ANSI.reset)"
            } else if isStarPoint(x: x, y: y, boardSize: boardSize) {
                cell = "+"
            } else {
                cell = "·"
            }
            cells.append(cell)
        }

        let rowStr = cells.joined(separator: " ")
        print("\(prefix) \(rowStr) \(prefix)")
    }

    print(header)
}
