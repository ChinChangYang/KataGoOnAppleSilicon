import Foundation
import KataGoOnAppleSilicon

struct RawNNResult {
    var whiteWin: Double = 0
    var whiteLoss: Double = 0
    var whiteLead: Double = 0
    var shorttermScoreError: Double = 0
    var policyRows: [[Double]] = []
    var policyPass: Double = 0
    var ownershipRows: [[Double]] = []
}

/// Parse a kata-rawnn GTP response into a RawNNResult.
func parseRawNN(_ response: String) -> RawNNResult {
    var result = RawNNResult()

    var content = response
    if content.hasPrefix("= ") {
        content = String(content.dropFirst(2))
    }

    let lines = content.components(separatedBy: "\n")
    var i = 0
    var parsingSection: String? = nil

    while i < lines.count {
        let line = lines[i]

        if line == "policy" {
            parsingSection = "policy"
            i += 1
            continue
        }
        if line == "whiteOwnership" {
            parsingSection = "ownership"
            i += 1
            continue
        }

        if parsingSection == "policy" {
            if line.hasPrefix("policyPass ") {
                parsingSection = nil
                let parts = line.split(separator: " ", omittingEmptySubsequences: true)
                if parts.count >= 2, let val = Double(parts[1]) {
                    result.policyPass = val
                }
                i += 1
                continue
            }
            let values = line.split(separator: " ", omittingEmptySubsequences: true)
                .compactMap { s -> Double? in
                    if s == "NAN" { return -1.0 }
                    return Double(s)
                }
            if !values.isEmpty {
                result.policyRows.append(values)
            }
            i += 1
            continue
        }

        if parsingSection == "ownership" {
            let values = line.split(separator: " ", omittingEmptySubsequences: true)
                .compactMap { s -> Double? in
                    if s == "NAN" { return 0.0 }
                    return Double(s)
                }
            if !values.isEmpty {
                result.ownershipRows.append(values)
            }
            i += 1
            continue
        }

        // Parse scalar key-value pairs
        let parts = line.split(separator: " ", omittingEmptySubsequences: true)
        if parts.count >= 2, let val = Double(parts[1]) {
            switch parts[0] {
            case "whiteWin":            result.whiteWin = val
            case "whiteLoss":           result.whiteLoss = val
            case "whiteLead":           result.whiteLead = val
            case "shorttermScoreError": result.shorttermScoreError = val
            default: break
            }
        }

        i += 1
    }

    return result
}

/// Return the top-N moves by policy probability, sorted descending.
func topMoves(_ result: RawNNResult, boardSize: Int = 19, count: Int = 5) -> [(coord: String, prob: Double)] {
    var indexed: [(index: Int, prob: Double)] = []

    for (rowIdx, row) in result.policyRows.enumerated() {
        for (colIdx, prob) in row.enumerated() {
            if prob >= 0 {
                indexed.append((index: rowIdx * boardSize + colIdx, prob: prob))
            }
        }
    }
    let passIndex = boardSize * boardSize
    if result.policyPass >= 0 {
        indexed.append((index: passIndex, prob: result.policyPass))
    }

    return indexed
        .sorted { $0.prob > $1.prob }
        .prefix(count)
        .map { item in
            let coord: String
            if item.index == passIndex {
                coord = "pass"
            } else {
                let x = item.index % boardSize
                let y = item.index / boardSize
                coord = pointToGTP(x: x, y: y, boardSize: boardSize)
            }
            return (coord: coord, prob: item.prob)
        }
}

/// Print win-rate bar, score lead, and top-5 moves.
func printSummary(
    _ result: RawNNResult,
    currentPlayerName: String,
    opponentName: String,
    currentIsWhite: Bool,
    boardSize: Int = 19
) {
    let currentWin  = currentIsWhite ? result.whiteWin  : result.whiteLoss
    let opponentWin = currentIsWhite ? result.whiteLoss : result.whiteWin

    let barWidth = 30
    let filledCount = max(0, min(barWidth, Int(currentWin * Double(barWidth))))
    let filled = String(repeating: "█", count: filledCount)
    let empty  = String(repeating: "░", count: barWidth - filledCount)
    let curPct = Int((currentWin * 100).rounded())
    let oppPct = Int((opponentWin * 100).rounded())
    print("[\(filled)\(empty)] \(currentPlayerName) \(curPct)% | \(opponentName) \(oppPct)%")

    let lead = currentIsWhite ? result.whiteLead : -result.whiteLead
    let err  = result.shorttermScoreError
    let leader = lead >= 0 ? currentPlayerName : opponentName
    let absLead = abs(lead)
    print(String(format: "Score Lead: \(leader)+%.1f ±%.1f", absLead, err))

    let moves = topMoves(result, boardSize: boardSize)
    let moveStrs = moves.enumerated().map { i, m in
        String(format: "%d. %@ (%.1f%%)", i + 1, m.coord, m.prob * 100)
    }.joined(separator: "  ")
    print("Top moves: \(moveStrs)")
}

/// Print full analysis: summary + ownership heat map.
func printDetailedAnalysis(
    _ result: RawNNResult,
    currentPlayerName: String,
    opponentName: String,
    currentIsWhite: Bool,
    boardSize: Int = 19
) {
    printSummary(result, currentPlayerName: currentPlayerName, opponentName: opponentName,
                 currentIsWhite: currentIsWhite, boardSize: boardSize)

    guard !result.ownershipRows.isEmpty else { return }

    print("Ownership (█▓▒░ = \(currentPlayerName) territory, \(ANSI.cyan)░▒▓█\(ANSI.reset) = \(opponentName) territory):")
    let allCols = "A B C D E F G H J K L M N O P Q R S T"
    let colsStr = allCols.split(separator: " ").prefix(boardSize).joined(separator: " ")
    let header = "   \(colsStr)"
    print(header)

    for (yi, row) in result.ownershipRows.enumerated() {
        let rowNum = boardSize - yi
        let prefix = rowNum < 10 ? " \(rowNum)" : "\(rowNum)"
        let cells = row.map { val -> String in
            // val >= 0 always means White territory from the model.
            let isCurrentTerritory = currentIsWhite ? (val >= 0) : (val < 0)
            let absVal = abs(val)
            if isCurrentTerritory {
                if absVal > 0.75 { return "█" }
                if absVal > 0.50 { return "▓" }
                if absVal > 0.25 { return "▒" }
                if absVal > 0.05 { return "░" }
                return "·"
            } else {
                if absVal > 0.75 { return "\(ANSI.cyan)█\(ANSI.reset)" }
                if absVal > 0.50 { return "\(ANSI.cyan)▓\(ANSI.reset)" }
                if absVal > 0.25 { return "\(ANSI.cyan)▒\(ANSI.reset)" }
                if absVal > 0.05 { return "\(ANSI.cyan)░\(ANSI.reset)" }
                return "·"
            }
        }.joined(separator: " ")
        print("\(prefix) \(cells)")
    }

    print(header)
}
