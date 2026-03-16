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

    public init(katago: KataGoInference) {
        self.katago = katago
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
        case "boardsize":          return successResponse()  // Assume 19x19
        case "clear_board":
            board = Board()
            consecutiveBehindCount = [.black: 0, .white: 0]
            lastPlayPassColor = nil
            return successResponse()
        case "komi":               return successResponse()  // Set komi, but placeholder
        case "play":               return handlePlay(parts: parts)
        case "kata-set-rules":     return handleKataSetRules(parts: parts)
        case "genmove":            return handleGenmove(parts: parts)
        case "quit":               return successResponse()
        default:                   return errorResponse("unknown command")
        }
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
                let boardState = BoardState(board: board, rules: rules)  // Use actual board and stored rules
                let output = try katago.predict(board: boardState, profile: profile)  // Use configured profile

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

                let move = selectMove(from: output.policy, greedy: false)

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
    
    private let knownCommands = ["protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play", "genmove", "kata-set-rules", "quit"]
    
    private func parseMove(_ move: String) -> Point? {
        guard move.count >= 2 else { return nil }
        let colChar = move.first!
        let rowStr = String(move.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= 19 else { return nil }
        
        var col: Int
        if colChar >= "A" && colChar <= "H" {
            col = Int(colChar.asciiValue! - 65)
        } else if colChar >= "J" && colChar <= "T" {
            col = Int(colChar.asciiValue! - 65) - 1  // Skip I
        } else {
            return nil
        }
        
        return Point(x: col, y: 19 - row)  // GTP is 1-based, top-left
    }
    
    private func selectMove(from policy: MLMultiArray, greedy: Bool = true) -> String {
        // Policy shape is [1, 6, 362] - access board positions as [0, 0, y*19+x]
        // For 19x19 board: position index = y * 19 + x (0-360), channel 0 is main policy
        return greedy ? selectMoveGreedy(from: policy) : selectMoveProbabilistic(from: policy)
    }
    
    private func selectMoveGreedy(from policy: MLMultiArray) -> String {
        // Greedy sampling: select the move with maximum probability
        var maxProb: Float = 0
        var maxY = 0
        var maxX = 0
        
        for y in 0..<19 {
            for x in 0..<19 {
                let prob = getPolicyProbability(policy: policy, x: x, y: y)
                if prob > maxProb {
                    maxProb = prob
                    maxY = y
                    maxX = x
                }
            }
        }
        
        let passProb = getPassPolicyProbability(policy: policy)
        if passProb > maxProb {
            return "pass"
        }
        return coordinateToGTP(x: maxX, y: maxY)
    }

    private func selectMoveProbabilistic(from policy: MLMultiArray) -> String {
        // Non-greedy sampling: sample from the policy distribution
        let moves = collectMovesWithProbabilities(from: policy)
        
        guard !moves.isEmpty else {
            // Fallback to greedy if no valid moves
            return selectMoveGreedy(from: policy)
        }
        
        let totalProb = moves.reduce(0.0) { $0 + $1.prob }
        guard totalProb > 0 else {
            return selectMoveGreedy(from: policy)
        }
        
        // Normalize probabilities
        let normalizedMoves = moves.map { (x: $0.x, y: $0.y, prob: $0.prob / totalProb) }
        
        // Sample from the distribution
        let random = Float.random(in: 0..<1)
        var cumulativeProb: Float = 0
        
        for move in normalizedMoves {
            cumulativeProb += move.prob
            if random <= cumulativeProb {
                return moveToGTP(x: move.x, y: move.y)
            }
        }

        // Fallback (shouldn't reach here, but just in case)
        let lastMove = normalizedMoves.last!
        return moveToGTP(x: lastMove.x, y: lastMove.y)
    }
    
    private static let passPolicyIndex = 361

    private func getPolicyProbability(policy: MLMultiArray, x: Int, y: Int) -> Float {
        // Access policy at position index = y * 19 + x, channel 0
        let positionIndex = y * 19 + x
        return Float(policy[[0, 0, NSNumber(value: positionIndex)]].doubleValue)
    }

    private func getPassPolicyProbability(policy: MLMultiArray) -> Float {
        return Float(policy[[0, 0, NSNumber(value: GTPHandler.passPolicyIndex)]].doubleValue)
    }

    private func moveToGTP(x: Int, y: Int) -> String {
        return x == -1 ? "pass" : coordinateToGTP(x: x, y: y)
    }
    
    private func collectMovesWithProbabilities(from policy: MLMultiArray) -> [(x: Int, y: Int, prob: Float)] {
        var moves: [(x: Int, y: Int, prob: Float)] = []
        
        for y in 0..<19 {
            for x in 0..<19 {
                let prob = getPolicyProbability(policy: policy, x: x, y: y)
                if prob > 0 {
                    moves.append((x: x, y: y, prob: prob))
                }
            }
        }
        
        let passProb = getPassPolicyProbability(policy: policy)
        if passProb > 0 {
            moves.append((x: -1, y: -1, prob: passProb))
        }
        return moves
    }

    private func coordinateToGTP(x: Int, y: Int) -> String {
        // Convert board coordinates (x, y) to GTP format
        // GTP: columns A-T (skipping I), rows 1-19 (19 at top)
        let colLetter = x < 8 ? String(UnicodeScalar(65 + x)!) : String(UnicodeScalar(66 + x)!)  // Skip I
        let row = 19 - y  // GTP: 19 at top
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
        let postPassBoardState = BoardState(board: postPassBoard, rules: rules)
        let postPassModelOutput = try katago.predict(board: postPassBoardState, profile: profile)

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
