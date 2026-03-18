import Foundation
import KataGoOnAppleSilicon

// MARK: - Helpers

func extractGTPValue(_ response: String) -> String? {
    guard response.hasPrefix("= ") else { return nil }
    let value = response.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
    return value.isEmpty ? nil : value
}

func stoneName(_ stone: Stone) -> String {
    stone == .black ? "Black" : "White"
}

func renderBoardFromGTP(
    _ gtp: GTPHandler,
    boardSize: Int = 19,
    lastMove: String? = nil,
    hints: [(coord: String, prob: Double)] = []
) {
    let response = gtp.handleCommand("showboard")
    let grid = parseShowboard(response, boardSize: boardSize)
    renderBoard(grid: grid, boardSize: boardSize, lastMove: lastMove, hints: hints)
}

func runAnalysis(_ gtp: GTPHandler, humanName: String, aiName: String,
                 currentIsWhite: Bool, boardSize: Int = 19) {
    let rawResp = gtp.handleCommand("kata-rawnn 0")
    guard rawResp.hasPrefix("= ") else {
        print("(analysis unavailable)")
        return
    }
    let parsed = parseRawNN(rawResp, boardSize: boardSize)
    printSummary(parsed, currentPlayerName: humanName, opponentName: aiName,
                 currentIsWhite: currentIsWhite, boardSize: boardSize)
}

// MARK: - Setup

let setup = runSetupFlow()

let katago = KataGoInference()
print("Loading \(setup.aiProfile) model — this may take a moment...")
do {
    try katago.loadModel(for: setup.aiProfile)
} catch {
    print("Failed to load model: \(error)")
    exit(1)
}
print("Model loaded.\n")

let gtp = GTPHandler(katago: katago)
gtp.setProfile(setup.aiProfile)
_ = gtp.handleCommand("boardsize \(setup.boardSize)")
_ = gtp.handleCommand("komi \(setup.komi)")

let humanColor  = setup.humanColor
let aiColor     = humanColor.opponent
let humanName   = stoneName(humanColor)
let aiName      = stoneName(aiColor)
let humanGTPStr = humanName.lowercased()
let aiGTPStr    = aiName.lowercased()

var moveHistory: [(Stone, String)] = []
var lastAIMove:  String?           = nil

// MARK: - Initial board + optional first AI move

let boardSize = setup.boardSize

renderBoardFromGTP(gtp, boardSize: boardSize)
print()

if humanColor == .white {
    print("AI (\(aiName)) is thinking...")
    let aiResp = gtp.handleCommand("genmove \(aiGTPStr)")
    if let aiMove = extractGTPValue(aiResp) {
        moveHistory.append((aiColor, aiMove))
        lastAIMove = aiMove
        print("AI plays: \(aiMove)")
        renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: aiMove)
        print()
    }
}

runAnalysis(gtp, humanName: humanName, aiName: aiName,
            currentIsWhite: humanColor == .white, boardSize: boardSize)
print()

// MARK: - Game loop

let helpText = """
Commands: <coord> (e.g. D4) | pass | hint | analysis | board | \
save | profile <name> | ai | quit
"""
print(helpText)

while true {
    print("\nYour turn (\(humanName)) > ", terminator: "")
    fflush(stdout)

    guard let rawInput = readLine() else { break }
    let input = rawInput.trimmingCharacters(in: .whitespaces)
    guard !input.isEmpty else { continue }

    switch CommandParser.parse(input) {

    case .move(let coord):
        let playResp = gtp.handleCommand("play \(humanGTPStr) \(coord)")
        if playResp.hasPrefix("? ") {
            let msg = playResp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
            print("Invalid move: \(msg)")
            continue
        }
        moveHistory.append((humanColor, coord))

        print("AI (\(aiName)) is thinking...")
        let aiResp = gtp.handleCommand("genmove \(aiGTPStr)")
        if let aiMove = extractGTPValue(aiResp) {
            if aiMove == "resign" {
                moveHistory.append((aiColor, "resign"))
                print("AI (\(aiName)) resigns. \(humanName) wins!")
                renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: coord)
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                exit(0)
            } else {
                moveHistory.append((aiColor, aiMove))
                lastAIMove = aiMove
                print("AI plays: \(aiMove)")
                renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: aiMove)
                runAnalysis(gtp, humanName: humanName, aiName: aiName,
                currentIsWhite: humanColor == .white, boardSize: boardSize)
                print()
                print(helpText)
            }
        }

    case .pass:
        _ = gtp.handleCommand("play \(humanGTPStr) pass")
        moveHistory.append((humanColor, "pass"))
        print("You passed.")

        print("AI (\(aiName)) is thinking...")
        let aiResp = gtp.handleCommand("genmove \(aiGTPStr)")
        if let aiMove = extractGTPValue(aiResp) {
            moveHistory.append((aiColor, aiMove))
            if aiMove.lowercased() != "pass" { lastAIMove = aiMove }
            print("AI plays: \(aiMove)")
            renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: aiMove == "pass" ? nil : aiMove)
            if aiMove.lowercased() == "pass" {
                print("Both players passed. Game over.")
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                exit(0)
            }
            runAnalysis(gtp, humanName: humanName, aiName: aiName,
                currentIsWhite: humanColor == .white, boardSize: boardSize)
            print()
            print(helpText)
        }

    case .hint:
        let rawResp = gtp.handleCommand("kata-rawnn 0")
        if rawResp.hasPrefix("= ") {
            let parsed = parseRawNN(rawResp, boardSize: boardSize)
            let hints = topMoves(parsed, boardSize: boardSize)
            renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: lastAIMove, hints: hints)
            print("\nTop moves for \(humanName):")
            for (i, m) in hints.enumerated() {
                print(String(format: "  %d. %@ (%.1f%%)", i + 1, m.coord, m.prob * 100))
            }
        } else {
            print("Analysis unavailable.")
        }

    case .analysis:
        let rawResp = gtp.handleCommand("kata-rawnn 0")
        if rawResp.hasPrefix("= ") {
            let parsed = parseRawNN(rawResp, boardSize: boardSize)
            let hints = topMoves(parsed, boardSize: boardSize)
            renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: lastAIMove, hints: hints)
            print()
            printDetailedAnalysis(parsed, currentPlayerName: humanName, opponentName: aiName,
                                  currentIsWhite: humanColor == .white, boardSize: boardSize)
        } else {
            print("Analysis unavailable.")
        }

    case .board:
        renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: lastAIMove)

    case .save:
        saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)

    case .profile(let name):
        do {
            try katago.loadModel(for: name)
            gtp.setProfile(name)
            print("Profile switched to \(name)")
        } catch {
            print("Error switching profile: \(error)")
        }

    case .aiMove:
        print("AI playing for \(humanName)...")
        let aiResp = gtp.handleCommand("genmove \(humanGTPStr)")
        if let move = extractGTPValue(aiResp) {
            moveHistory.append((humanColor, move))
            if move.lowercased() != "pass" { lastAIMove = move }
            print("AI plays for you: \(move)")
            renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: move == "pass" ? nil : move)
        }

    case .quit:
        print("Goodbye!")
        exit(0)

    case .unknown(let s):
        print("Unknown command '\(s)'.")
        print(helpText)
    }
}
