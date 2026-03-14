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

    /// Configure resign thresholds.
    /// - Parameters:
    ///   - winRate: Current player win-rate threshold (0.0–1.0). Resign triggers when win rate
    ///              stays below this value for `consecutiveMoves` consecutive genmove calls.
    ///   - consecutiveMoves: Number of consecutive below-threshold moves before resigning.
    /// - Note: Calling this method resets the consecutive-behind counters for both colors as a
    ///         side effect, clearing any in-progress resign streak.
    public func setResignThreshold(winRate: Double, consecutiveMoves: Int) {
        resignWinRateThreshold = winRate
        resignConsecutiveMoveThreshold = consecutiveMoves
        consecutiveBehindCount = [.black: 0, .white: 0]
    }

    /// Process a GTP command and return response
    public func handleCommand(_ command: String) -> String {
        let parts = command.split(separator: " ").map { String($0) }
        guard !parts.isEmpty else { return "? \n\n" }
        
        let cmd = parts[0]
        switch cmd {
        case "protocol_version":
            return "= 2\n\n"
        case "name":
            return "= KataGoOnAppleSilicon\n\n"
        case "version":
            return "= 1.0\n\n"
        case "known_command":
            return parts.count > 1 && knownCommands.contains(parts[1]) ? "= true\n\n" : "= false\n\n"
        case "list_commands":
            return "= " + knownCommands.joined(separator: " ") + "\n\n"
        case "boardsize":
            return "= \n\n"  // Assume 19x19
        case "clear_board":
            board = Board()
            consecutiveBehindCount = [.black: 0, .white: 0]
            return "= \n\n"
        case "komi":
            // Set komi, but placeholder
            return "= \n\n"
        case "play":
            if parts.count >= 3 {
                let colorStr = parts[1]
                let moveStr = parts[2]
                let stone: Stone = colorStr == "black" ? .black : .white
                if let point = parseMove(moveStr) {
                    if board.playMove(at: point, stone: stone) {
                        return "= \n\n"
                    } else {
                        return "? illegal move\n\n"
                    }
                } else {
                    return "? syntax error\n\n"
                }
            } else {
                return "? syntax error\n\n"
            }
        case "kata-set-rules":
            if parts.count < 2 {
                return "? Expected at least one argument for kata-set-rules\n\n"
            }
            // Join all arguments (skip command name at index 0)
            let preset = parts[1...].joined(separator: " ").trimmingCharacters(in: .whitespaces).lowercased()
            if preset == "chinese" {
                rules = .chineseRules
                return "= \n\n"
            } else {
                return "? Unknown rules '\(preset)'\n\n"
            }
        case "genmove":
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
                        let count = (consecutiveBehindCount[stone] ?? 0) + 1
                        consecutiveBehindCount[stone] = count
                        if count >= resignConsecutiveMoveThreshold {
                            consecutiveBehindCount[stone] = 0
                            return "= resign\n\n"
                        }
                    } else {
                        consecutiveBehindCount[stone] = 0
                    }

                    let move = selectMove(from: output.policy, greedy: false)

                    // Play the generated move on the board
                    if let point = parseMove(move) {
                        if board.playMove(at: point, stone: stone) {
                            return "= \(move)\n\n"
                        } else {
                            return "? illegal move: \(move)\n\n"
                        }
                    } else {
                        return "? failed to parse generated move: \(move)\n\n"
                    }
                } catch {
                    return "? \(error.localizedDescription)\n\n"
                }
            } else {
                return "? syntax error\n\n"
            }
        case "quit":
            return "= \n\n"
        default:
            return "? unknown command\n\n"
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
                return coordinateToGTP(x: move.x, y: move.y)
            }
        }
        
        // Fallback (shouldn't reach here, but just in case)
        let lastMove = normalizedMoves.last!
        return coordinateToGTP(x: lastMove.x, y: lastMove.y)
    }
    
    private func getPolicyProbability(policy: MLMultiArray, x: Int, y: Int) -> Float {
        // Access policy at position index = y * 19 + x, channel 0
        let positionIndex = y * 19 + x
        return Float(policy[[0, 0, NSNumber(value: positionIndex)]].doubleValue)
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
        
        return moves
    }
    
    private func coordinateToGTP(x: Int, y: Int) -> String {
        // Convert board coordinates (x, y) to GTP format
        // GTP: columns A-T (skipping I), rows 1-19 (19 at top)
        let colLetter = x < 8 ? String(UnicodeScalar(65 + x)!) : String(UnicodeScalar(66 + x)!)  // Skip I
        let row = 19 - y  // GTP: 19 at top
        return "\(colLetter)\(row)"
    }
}