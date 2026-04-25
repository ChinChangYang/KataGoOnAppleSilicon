import CoreML
import Foundation

/// Handles GTP commands and responses
public class GTPHandler {
    private let katago: KataGoInference
    private var board: Board = Board()  // Placeholder board
    private var profile: String = "AI"  // Profile to use for inference
    private var rules: Rules = .defaultRules  // Default rules (backward compatible)
    private var resignWinRateThreshold: Double = 0.10
    private var resignConsecutiveMoveThreshold: Int = 10
    private var consecutiveBehindCount: [Stone: Int] = [.black: 0, .white: 0]
    private var friendlyPassEnabled: Bool = false
    private var friendlyPassWinRateDelta: Double = 0.02
    private var friendlyPassLeadDelta: Double = 1.0
    private var lastPlayPassColor: Stone? = nil
    private var friendlyPassMinimumTurn: Int = 0

    // SGF metadata preserved across loadsgf / printsgf so that printing back
    // reflects what was loaded. Defaults match the engine's identity when no
    // SGF has been loaded.
    private var blackPlayerName: String = "Black"
    private var whitePlayerName: String = "White"
    private var sgfRulesName: String? = nil

    public init(katago: KataGoInference) {
        self.katago = katago
        self.board.rules = rules
    }

    private func makeBoard(size: Int) -> Board {
        let b = Board(size: size)
        b.rules = rules
        return b
    }

    /// Set the profile to use for inference (e.g., "AI", "20k", "9d", etc.)
    public func setProfile(_ profile: String) {
        self.profile = profile
    }

    /// Get the current profile being used for inference
    public func getProfile() -> String {
        return profile
    }

    /// Configure resign thresholds and reset the consecutive-behind counters for both colors.
    /// - Parameters:
    ///   - winRate: Current player win-rate threshold (0.0–1.0). Resign triggers when win rate
    ///              stays below this value for `consecutiveMoves` consecutive genmove calls.
    ///   - consecutiveMoves: Number of consecutive below-threshold moves before resigning.
    /// - Note: Resets in-progress resign streaks for both colors, so calling mid-game restarts
    ///         the streak count from zero.
    public func setResignThreshold(winRate: Double, consecutiveMoves: Int) {
        resignWinRateThreshold = winRate
        resignConsecutiveMoveThreshold = consecutiveMoves
        consecutiveBehindCount = [.black: 0, .white: 0]
    }

    /// Configure friendly pass behavior.
    /// - Parameters:
    ///   - enabled: Whether AI may respond to an opponent pass with its own pass.
    ///   - winRateDelta: Max allowed change in win rate (0.0–1.0). Default 0.02.
    ///   - leadDelta: Max allowed change in score lead (points). Default 1.0.
    ///   - minimumTurn: Earliest turn at which friendly pass may be attempted.
    ///                  Skips the second inference when `board.turnNumber` is below
    ///                  this value. Default 0 (no restriction).
    public func setFriendlyPassOptions(
        enabled: Bool,
        winRateDelta: Double = 0.02,
        leadDelta: Double = 1.0,
        minimumTurn: Int = 0
    ) {
        friendlyPassEnabled = enabled
        friendlyPassWinRateDelta = winRateDelta
        friendlyPassLeadDelta = leadDelta
        friendlyPassMinimumTurn = minimumTurn
    }

    private func successResponse(_ value: String = "") -> String {
        return value.isEmpty ? "= \n\n" : "= \(value)\n\n"
    }

    private func errorResponse(_ message: String) -> String {
        return "? \(message)\n\n"
    }

    /// Process a GTP command and return response
    public func handleCommand(_ command: String) -> String {
        let parts = command.split(separator: " ").map { String($0) }
        guard !parts.isEmpty else { return errorResponse("") }

        let cmd = parts[0]
        switch cmd {
        case "protocol_version":   return successResponse("2")
        case "name":               return successResponse("KataGoOnAppleSilicon")
        case "version":            return successResponse("1.0")
        case "known_command":      return parts.count > 1 && knownCommands.contains(parts[1])
                                       ? successResponse("true") : successResponse("false")
        case "list_commands":      return successResponse(knownCommands.joined(separator: " "))
        case "boardsize":          return handleBoardsize(parts: parts)
        case "clear_board":
            board = makeBoard(size: board.xSize)
            resetGameState()
            resetSGFMetadata()
            return successResponse()
        case "komi":               return handleKomi(parts: parts)
        case "play":               return handlePlay(parts: parts)
        case "kata-set-rules":     return handleKataSetRules(parts: parts)
        case "genmove":            return handleGenmove(parts: parts)
        case "undo":               return handleUndo()
        case "fixed_handicap":     return handleFixedHandicap(parts: parts)
        case "set_free_handicap":  return handleSetFreeHandicap(parts: parts)
        case "showboard":          return handleShowboard()
        case "kata-rawnn":         return handleKataRawNN(parts: parts)
        case "final_score":        return handleFinalScore()
        case "loadsgf":            return handleLoadSGF(parts: parts)
        case "printsgf":           return handlePrintSGF(parts: parts)
        case "quit":               return successResponse()
        default:                   return errorResponse("unknown command")
        }
    }

    private func resetGameState() {
        consecutiveBehindCount = [.black: 0, .white: 0]
        lastPlayPassColor = nil
    }

    /// Reset SGF metadata to engine defaults (used when a new game starts via
    /// `clear_board` or `boardsize`, so a subsequent `printsgf` doesn't echo
    /// stale player/rules info from a previous `loadsgf`).
    private func resetSGFMetadata() {
        blackPlayerName = "Black"
        whitePlayerName = "White"
        sgfRulesName = nil
    }

    private func handleKomi(parts: [String]) -> String {
        guard parts.count >= 2, let komi = Float(parts[1]), komi.isFinite else {
            return errorResponse("syntax error")
        }
        board.komi = komi
        return successResponse()
    }

    private func handleBoardsize(parts: [String]) -> String {
        guard parts.count >= 2, let size = Int(parts[1]) else {
            return errorResponse("syntax error")
        }
        guard size >= 2 && size <= 19 else {
            return errorResponse("unacceptable size")
        }
        board = makeBoard(size: size)
        resetGameState()
        resetSGFMetadata()
        return successResponse()
    }

    private func handlePlay(parts: [String]) -> String {
        if parts.count >= 3 {
            let colorStr = parts[1]
            let moveStr = parts[2]
            let stone: Stone = colorStr == "black" ? .black : .white
            if moveStr.lowercased() == "pass" {
                _ = board.playPass(stone: stone)
                lastPlayPassColor = stone
                return successResponse()
            } else {
                lastPlayPassColor = nil
                if let point = parseMove(moveStr) {
                    if board.playMove(at: point, stone: stone) {
                        return successResponse()
                    } else {
                        return errorResponse("illegal move")
                    }
                } else {
                    return errorResponse("syntax error")
                }
            }
        } else {
            return errorResponse("syntax error")
        }
    }

    private func handleKataSetRules(parts: [String]) -> String {
        if parts.count < 2 {
            return errorResponse("Expected at least one argument for kata-set-rules")
        }
        // Join all arguments (skip command name at index 0)
        let preset = parts[1...].joined(separator: " ").trimmingCharacters(in: .whitespaces).lowercased()
        if preset == "chinese" {
            rules = .chineseRules
            board.rules = rules
            sgfRulesName = "Chinese"
            return successResponse()
        } else {
            return errorResponse("Unknown rules '\(preset)'")
        }
    }

    private func handleGenmove(parts: [String]) -> String {
        if parts.count >= 2 {
            let colorStr = parts[1]
            let stone: Stone = colorStr == "black" ? .black : .white
            do {
                let boardState = BoardState(board: board, nextPlayer: stone, komi: board.komi, rules: rules)
                let output = try katago.predict(board: boardState, profile: profile, boardArea: board.xSize * board.ySize)  // Use configured profile

                // Resign logic
                // postprocess() swaps win/loss for .black, so whiteWinProb == current player's win rate.
                let postOutput = output.postprocess(board: board, nextPlayer: stone)
                let currentPlayerWinRate = postOutput.whiteWinProb
                if currentPlayerWinRate < resignWinRateThreshold {
                    let count = consecutiveBehindCount[stone, default: 0] + 1
                    consecutiveBehindCount[stone] = count
                    if count >= resignConsecutiveMoveThreshold {
                        // Counter resets so the engine must accumulate a new streak if the game continues.
                        consecutiveBehindCount[stone] = 0
                        lastPlayPassColor = nil
                        return successResponse("resign")
                    }
                } else {
                    consecutiveBehindCount[stone] = 0
                }

                // Friendly pass: if opponent just passed and passing is safe, pass back
                if friendlyPassEnabled, let passColor = lastPlayPassColor, passColor != stone {
                    if let passResponse = try tryFriendlyPass(stone: stone, currentOutput: postOutput) {
                        return passResponse
                    }
                }

                let move = selectMove(from: postOutput.policyProbs)

                // Handle pass before attempting to parse as a board coordinate
                if move.lowercased() == "pass" {
                    _ = board.playPass(stone: stone)
                    lastPlayPassColor = stone
                    return successResponse("pass")
                }

                // Play the generated move on the board
                if let point = parseMove(move) {
                    if board.playMove(at: point, stone: stone) {
                        return successResponse(move)
                    } else {
                        return errorResponse("illegal move: \(move)")
                    }
                } else {
                    return errorResponse("failed to parse generated move: \(move)")
                }
            } catch {
                return errorResponse(error.localizedDescription)
            }
        } else {
            return errorResponse("syntax error")
        }
    }
    
    private func handleShowboard() -> String {
        var lines: [String] = []
        for y in 0..<board.ySize {
            let rowNum = board.ySize - y
            let prefix = rowNum < 10 ? " \(rowNum)" : "\(rowNum)"
            let cells = (0..<board.xSize).map { x -> String in
                switch board.stones[y][x] {
                case .black: return "X"
                case .white: return "O"
                default:     return "."
                }
            }.joined(separator: " ")
            lines.append("\(prefix) \(cells)")
        }
        return successResponse(lines.joined(separator: "\n"))
    }

    private func handleKataRawNN(parts: [String]) -> String {
        let symmetry = Int(parts.count > 1 ? parts[1] : "0") ?? 0
        do {
            let nextPlayer: Stone = board.sideToMove
            let boardState = BoardState(board: board, nextPlayer: nextPlayer, komi: board.komi, rules: rules)
            let result = try katago.rawNN(
                board: board, boardState: boardState,
                profile: profile, whichSymmetry: symmetry)
            return successResponse(result)
        } catch {
            return errorResponse(error.localizedDescription)
        }
    }

    private func handleFinalScore() -> String {
        do {
            let nextPlayer: Stone = board.sideToMove
            let boardState = BoardState(board: board, nextPlayer: nextPlayer, komi: board.komi, rules: rules)
            let output = try katago.predict(board: boardState, profile: "AI", boardArea: board.xSize * board.ySize)
            let postOutput = output.postprocess(board: board, nextPlayer: nextPlayer)
            // whiteLead is White's absolute lead (positive = White ahead)
            // Round to nearest 0.5 (area scoring with 7.5 komi gives half-integer results)
            let lead = postOutput.whiteLead
            let roundedLead = Foundation.round(lead + 0.5) - 0.5
            if roundedLead > 0 {
                return successResponse(String(format: "W+%.1f", roundedLead))
            } else if roundedLead < 0 {
                return successResponse(String(format: "B+%.1f", -roundedLead))
            } else {
                return successResponse("0")
            }
        } catch {
            return errorResponse(error.localizedDescription)
        }
    }

    private let knownCommands = ["protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play", "genmove", "undo", "fixed_handicap", "set_free_handicap", "kata-set-rules", "showboard", "kata-rawnn", "final_score", "loadsgf", "printsgf", "quit"]

    /// `printsgf` (no arg) returns the current game as SGF.
    /// `printsgf <filename>` writes the SGF to disk instead.
    private func handlePrintSGF(parts: [String]) -> String {
        guard parts.count == 1 || parts.count == 2 else {
            return errorResponse("syntax error")
        }
        let sgf = SGFGenerator.generateSGF(
            from: board,
            blackPlayer: blackPlayerName,
            whitePlayer: whitePlayerName,
            rulesName: sgfRulesName
        )
        guard parts.count == 2 else { return successResponse(sgf) }
        do {
            try SGFGenerator.saveSGF(sgf, to: parts[1])
            return successResponse()
        } catch {
            return errorResponse("could not write file: \(error.localizedDescription)")
        }
    }

    /// `loadsgf <filename> [<movenumber>]` replays an SGF from disk.
    /// `movenumber` is 1-based and points at the position *before* move N,
    /// so N=1 is the initial position and the default plays every move.
    private func handleLoadSGF(parts: [String]) -> String {
        guard parts.count == 2 || parts.count == 3 else {
            return errorResponse("syntax error")
        }
        let filename = parts[1]
        var moveNumber: Int? = nil
        if parts.count == 3 {
            guard let n = Int(parts[2]), n >= 1 else {
                return errorResponse("syntax error")
            }
            moveNumber = n
        }

        let text: String
        do {
            text = try String(contentsOfFile: filename, encoding: .utf8)
        } catch {
            return errorResponse("cannot load file: \(error.localizedDescription)")
        }

        let parsed: ParsedSGF
        do {
            parsed = try SGFParser.parse(text)
        } catch {
            return errorResponse(error.localizedDescription)
        }

        guard parsed.boardSize >= 2 && parsed.boardSize <= 19 else {
            return errorResponse("unacceptable size")
        }

        // Only "Chinese" is modeled today; other rule names are recorded but
        // not interpreted, so the engine keeps its current rules object.
        if parsed.rulesName?.lowercased() == "chinese" {
            rules = .chineseRules
        }
        sgfRulesName = parsed.rulesName

        let newBoard = Board(size: parsed.boardSize)
        newBoard.rules = rules
        newBoard.komi = parsed.komi

        let setup: [(Point, Stone)] =
            parsed.initialBlack.map { ($0, Stone.black) } +
            parsed.initialWhite.map { ($0, Stone.white) }
        if !setup.isEmpty {
            guard newBoard.setStonesFailIfNoLibs(setup) else {
                return errorResponse("invalid setup stones in SGF")
            }
            newBoard.initialStones = newBoard.stones
        }

        // SGF default: white moves first iff black handicap stones are present.
        let defaultPla: Stone = parsed.initialBlack.isEmpty ? .black : .white
        let startingPla = parsed.initialSideToMove ?? defaultPla
        newBoard.initialSideToMove = startingPla
        newBoard.sideToMove = startingPla

        let totalMoves = parsed.moves.count
        let target = moveNumber.map { max(0, min($0 - 1, totalMoves)) } ?? totalMoves
        for i in 0..<target {
            let move = parsed.moves[i]
            if move.isPass {
                _ = newBoard.playPass(stone: move.player)
            } else if let loc = move.location {
                guard newBoard.playMove(at: loc, stone: move.player) else {
                    return errorResponse("illegal move in SGF at move \(i + 1)")
                }
            }
        }

        board = newBoard
        blackPlayerName = parsed.blackPlayer ?? "Black"
        whitePlayerName = parsed.whitePlayer ?? "White"
        resetGameState()
        if let last = newBoard.moveHistory.last, last.isPass {
            lastPlayPassColor = last.player
        }
        return successResponse()
    }

    private func handleSetFreeHandicap(parts: [String]) -> String {
        guard board.isEmpty() else {
            return errorResponse("Board is not empty")
        }
        // KataGo's parser notes bad pieces with "Invalid handicap location: …"
        // but then *always* runs setStonesFailIfNoLibs with the partial list,
        // which fails on pass/unparseable tokens and overwrites the response
        // with "Handicap placement is invalid" (gtp.cpp:3186-3199). Short-
        // circuit to match that byte-for-byte.
        var points: [Point] = []
        var hadBad = false
        for piece in parts.dropFirst() {
            if piece.lowercased() == "pass" {
                hadBad = true
                continue
            }
            if let point = parseMove(piece) {
                points.append(point)
            } else {
                hadBad = true
            }
        }
        if hadBad {
            return errorResponse("Handicap placement is invalid")
        }
        guard board.placeFreeHandicap(points) else {
            return errorResponse("Handicap placement is invalid")
        }
        resetGameState()
        return successResponse()
    }

    private func handleFixedHandicap(parts: [String]) -> String {
        let argJoined = parts.count > 1 ? parts[1...].joined(separator: " ") : ""
        guard parts.count == 2 else {
            return errorResponse("Expected one argument for fixed_handicap but got '\(argJoined)'")
        }
        let arg = parts[1]
        guard let n = Int(arg) else {
            return errorResponse("Could not parse number of handicap stones: '\(arg)'")
        }
        if n < 2 {
            return errorResponse("Number of handicap stones less than 2: '\(arg)'")
        }
        guard board.isEmpty() else {
            return errorResponse("Board is not empty")
        }
        do {
            let placed = try board.placeFixedHandicap(n: n)
            // Reset GTP-level bookkeeping — handicap starts a fresh game.
            resetGameState()
            let vertices = placed.map { coordinateToGTP(x: $0.x, y: $0.y) }.joined(separator: " ")
            return successResponse(vertices)
        } catch KataGoError.handicapRefused(let message) {
            return errorResponse(message)
        } catch {
            return errorResponse(error.localizedDescription)
        }
    }

    private func handleUndo() -> String {
        guard board.undo() else {
            return errorResponse("cannot undo")
        }
        // Recompute lastPlayPassColor from the new tail of history, and reset
        // resign counters (no well-defined rewind for an in-progress streak).
        if let last = board.moveHistory.last, last.isPass {
            lastPlayPassColor = last.player
        } else {
            lastPlayPassColor = nil
        }
        consecutiveBehindCount = [.black: 0, .white: 0]
        return successResponse()
    }
    
    private func parseMove(_ move: String) -> Point? {
        guard move.count >= 2 else { return nil }
        let colChar = move.first!
        let rowStr = String(move.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= board.ySize else { return nil }

        var col: Int
        if colChar >= "A" && colChar <= "H" {
            col = Int(colChar.asciiValue! - 65)
        } else if colChar >= "J" && colChar <= "T" {
            col = Int(colChar.asciiValue! - 65) - 1  // Skip I
        } else {
            return nil
        }

        guard col < board.xSize else { return nil }

        return Point(x: col, y: board.ySize - row)  // GTP is 1-based, top-left
    }
    
    private static let passPolicyIndex = 361

    private func selectMove(from policyProbs: [Float]) -> String {
        // policyProbs layout: index y*19+x for board points, index 361 for pass.
        // Illegal moves have value -1.0; legal moves have softmax-normalised probabilities >= 0.
        let moves = collectMovesWithProbabilities(from: policyProbs)

        guard !moves.isEmpty else { return "pass" }

        let totalProb = moves.reduce(0.0) { $0 + $1.prob }
        guard totalProb > 0 else { return selectMoveGreedy(from: policyProbs) }

        // Normalise and sample from the distribution
        let normalizedMoves = moves.map { (x: $0.x, y: $0.y, prob: $0.prob / totalProb) }
        let random = Float.random(in: 0..<1)
        var cumulativeProb: Float = 0

        for move in normalizedMoves {
            cumulativeProb += move.prob
            if random <= cumulativeProb {
                return moveToGTP(x: move.x, y: move.y)
            }
        }

        // Fallback (shouldn't reach here)
        let lastMove = normalizedMoves.last!
        return moveToGTP(x: lastMove.x, y: lastMove.y)
    }

    func selectMoveGreedy(from policyProbs: [Float]) -> String {
        let moves = collectMovesWithProbabilities(from: policyProbs)
        guard let best = moves.max(by: { $0.prob < $1.prob }), best.prob > 0 else {
            return "pass"
        }
        return moveToGTP(x: best.x, y: best.y)
    }

    private func moveToGTP(x: Int, y: Int) -> String {
        return x == -1 ? "pass" : coordinateToGTP(x: x, y: y)
    }

    private func collectMovesWithProbabilities(from policyProbs: [Float]) -> [(x: Int, y: Int, prob: Float)] {
        var moves: [(x: Int, y: Int, prob: Float)] = []

        for y in 0..<board.ySize {
            for x in 0..<board.xSize {
                // Policy index always uses 19-wide stride regardless of board size
                // (the model output is always a fixed 362-element array in 19×19 layout)
                let prob = policyProbs[y * 19 + x]
                if prob > 0 {
                    moves.append((x: x, y: y, prob: prob))
                }
            }
        }

        let passProb = policyProbs[GTPHandler.passPolicyIndex]
        if passProb > 0 {
            moves.append((x: -1, y: -1, prob: passProb))
        }
        return moves
    }

    private func coordinateToGTP(x: Int, y: Int) -> String {
        // Convert board coordinates (x, y) to GTP format
        // GTP: columns A-T (skipping I), rows 1-boardYSize (boardYSize at top)
        let colLetter = x < 8 ? String(UnicodeScalar(65 + x)!) : String(UnicodeScalar(66 + x)!)  // Skip I
        let row = board.ySize - y  // GTP: boardYSize at top
        return "\(colLetter)\(row)"
    }

    /// Evaluate whether passing is safe after the opponent passed.
    /// Runs a second inference on the post-pass board (opponent to move) and compares
    /// metrics from the AI's perspective. Inference cost: one extra model call.
    /// - Parameters:
    ///   - stone: The AI's color.
    ///   - currentOutput: Post-processed output for current position (AI's perspective).
    /// - Returns: GTP "= pass\n\n" if passing is safe, nil otherwise.
    /// - Throws: Rethrows inference errors from katago.predict.
    private func tryFriendlyPass(
        stone: Stone,
        currentOutput: PostProcessedModelOutput
    ) throws -> String? {
        // Skip the expensive second inference when the game is too young.
        guard board.turnNumber >= friendlyPassMinimumTurn else { return nil }

        // Snapshot AI's current metrics (perspective-adjusted to AI's color).
        let currentWinRate = currentOutput.whiteWinProb
        let currentLead = currentOutput.whiteLead

        // Simulate AI passing on a copy of the board.
        let postPassBoard = board.copy()
        _ = postPassBoard.playPass(stone: stone)

        // Run inference with opponent to move next.
        let postPassBoardState = BoardState(board: postPassBoard, nextPlayer: stone.opponent, komi: postPassBoard.komi, rules: rules)
        let postPassModelOutput = try katago.predict(board: postPassBoardState, profile: profile, boardArea: postPassBoard.xSize * postPassBoard.ySize)

        // Consume the flag regardless of outcome: one evaluation per opponent pass,
        // whether we end up passing or not.
        lastPlayPassColor = nil

        // Post-process from opponent's perspective (they move next after the pass).
        let postPassOutput = postPassModelOutput.postprocess(
            board: postPassBoard,
            nextPlayer: stone.opponent
        )

        // Convert opponent-perspective metrics back to AI's perspective.
        // After the perspective flip in postprocessValueOutputs (line 159-165 of PostProcessing.swift),
        // postPassOutput.whiteLossProb = opponent's loss prob = AI's win prob from that position.
        // postPassOutput.whiteLead = opponent's lead → AI's lead = -postPassOutput.whiteLead.
        let postPassWinRate = postPassOutput.whiteLossProb
        let postPassLead = -postPassOutput.whiteLead

        // Decline if passing changes win rate or lead beyond thresholds.
        let winRateDiff = abs(currentWinRate - postPassWinRate)
        let leadDiff = abs(currentLead - postPassLead)
        guard winRateDiff <= friendlyPassWinRateDelta && leadDiff <= friendlyPassLeadDelta else {
            return nil
        }

        // Safe to pass: apply to the live board and return GTP response.
        _ = board.playPass(stone: stone)
        return successResponse("pass")
    }
}
