import CoreML

/// Represents the board state for model input
public struct BoardState {
    public let spatial: MLMultiArray  // [1,22,19,19]
    public let global: MLMultiArray   // [1,19]
    
    public init(spatial: MLMultiArray, global: MLMultiArray) {
        self.spatial = spatial
        self.global = global
    }
    
    /// Create from board data
    /// - Parameters:
    ///   - board: Current board state
    ///   - nextPlayer: The player to move next (default: .black)
    ///   - komi: Komi value (default: 7.5)
    ///   - turnNumber: Current turn number (default: 0)
    public init(board: Board, nextPlayer: Stone = .black, komi: Float = 7.5, turnNumber: Int = 0) {
        // KataGo features: 22 spatial planes
        let spatialShape: [NSNumber] = [1, 22, 19, 19]
        self.spatial = try! MLMultiArray(shape: spatialShape, dataType: .float16)
        
        // KataGo features: 19 global features
        let globalShape: [NSNumber] = [1, 19]
        self.global = try! MLMultiArray(shape: globalShape, dataType: .float16)
        
        // Fill spatial features (history function needs global array for pass history)
        Self.fillSpatialFeatures(spatial: spatial, board: board, nextPlayer: nextPlayer, global: global)
        // Fill global features (pass history already set by fillPlanes9To13History)
        Self.fillGlobalFeatures(global: global, board: board, nextPlayer: nextPlayer, komi: komi)
    }
    
    // MARK: - Spatial Features (22 planes)
    
    /// Fill spatial features for the neural network input
    /// - Parameters:
    ///   - spatial: MLMultiArray of shape [1, 22, 19, 19]
    ///   - board: Current board state
    ///   - nextPlayer: The player to move next (determines perspective)
    ///   - global: MLMultiArray for global features (needed for pass history in planes 9-13)
    private static func fillSpatialFeatures(spatial: MLMultiArray, board: Board, nextPlayer: Stone, global: MLMultiArray) {
        fillPlane0OnBoard(spatial: spatial)
        fillPlanes1And2Stones(spatial: spatial, board: board, nextPlayer: nextPlayer)
        fillPlanes3To5Liberties(spatial: spatial, board: board)
        fillPlane6KoBan(spatial: spatial, board: board)
        fillPlane7KoRecaptureBlocked(spatial: spatial)
        fillPlanes9To13History(spatial: spatial, global: global, board: board, nextPlayer: nextPlayer)
        fillPlanes14To17Ladders(spatial: spatial, board: board, nextPlayer: nextPlayer)
        fillPlanes18And19Area(spatial: spatial, board: board, nextPlayer: nextPlayer)
    }
    
    /// Fill plane 0: On board (always 1.0 for all valid positions)
    private static func fillPlane0OnBoard(spatial: MLMultiArray) {
        for y in 0..<19 {
            for x in 0..<19 {
                spatial[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = 1.0
            }
        }
    }
    
    /// Fill planes 1-2: Own and opponent stones (perspective-based)
    private static func fillPlanes1And2Stones(spatial: MLMultiArray, board: Board, nextPlayer: Stone) {
        let ownStone = nextPlayer
        let oppStone: Stone = (nextPlayer == .black) ? .white : .black
        
        for y in 0..<19 {
            for x in 0..<19 {
                let stone = board.stones[y][x]
                // Plane 1: Own stones (current player's perspective)
                if stone == ownStone {
                    spatial[[0, 1, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                }
                // Plane 2: Opponent stones
                else if stone == oppStone {
                    spatial[[0, 2, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                }
            }
        }
    }
    
    /// Fill planes 3-5: Liberty counts (1, 2, or 3 liberties)
    private static func fillPlanes3To5Liberties(spatial: MLMultiArray, board: Board) {
        for y in 0..<19 {
            for x in 0..<19 {
                let stone = board.stones[y][x]
                if stone != .empty {
                    let libertyCount = board.liberties(of: Point(x: x, y: y))
                    // Plane 3: Stones with exactly 1 liberty (atari)
                    if libertyCount == 1 {
                        spatial[[0, 3, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                    }
                    // Plane 4: Stones with exactly 2 liberties
                    else if libertyCount == 2 {
                        spatial[[0, 4, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                    }
                    // Plane 5: Stones with exactly 3 liberties
                    else if libertyCount == 3 {
                        spatial[[0, 5, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                    }
                }
            }
        }
    }
    
    /// Fill plane 6: Ko-ban (ko prohibition locations)
    private static func fillPlane6KoBan(spatial: MLMultiArray, board: Board) {
        if let ko = board.koPoint {
            spatial[[0, 6, NSNumber(value: ko.y), NSNumber(value: ko.x)]] = 1.0
        }
    }
    
    /// Fill plane 7: Ko recapture blocked (encore ko recapture blocked locations)
    /// For Chinese rules, this plane is all zeros (no encore features)
    /// MLMultiArray is zero-initialized, so no explicit setting needed, but documented here
    private static func fillPlane7KoRecaptureBlocked(spatial: MLMultiArray) {
        // Plane 7 is zero-initialized by MLMultiArray
        // Chinese rules have no encore phase, so all values remain 0.0
    }
    
    /// Fill planes 9-13: Move history (last 5 moves)
    /// Implements the exact algorithm from KataGo's fillRowV7() (lines 2503-2562 in nninputs.cpp)
    /// - Plane 9: Most recent move (by opponent)
    /// - Plane 10: 2 moves ago (by current player)
    /// - Plane 11: 3 moves ago (by opponent)
    /// - Plane 12: 4 moves ago (by current player)
    /// - Plane 13: 5 moves ago (by opponent)
    /// Pass moves are handled via global features 0-4 instead of spatial planes
    private static func fillPlanes9To13History(spatial: MLMultiArray, global: MLMultiArray, board: Board, nextPlayer: Stone) {
        let moveHistory = board.moveHistory
        let moveHistoryLen = moveHistory.count
        
        // For Chinese rules: simplified history tracking
        // maxTurnsOfHistoryToInclude = 5 (no game end or phase end logic needed)
        let maxTurnsOfHistoryToInclude = 5
        let amountOfHistoryToTryToUse = min(maxTurnsOfHistoryToInclude, moveHistoryLen)
        
        let pla = nextPlayer
        let opp: Stone = (nextPlayer == .black) ? .white : .black
        
        // Nested conditionals following C++ algorithm exactly
        // Move 1 ago (opponent)
        if amountOfHistoryToTryToUse >= 1 && moveHistoryLen >= 1 && moveHistory[moveHistoryLen - 1].player == opp {
            let prev1Move = moveHistory[moveHistoryLen - 1]
            if prev1Move.isPass {
                global[0] = 1.0
            } else if let prev1Loc = prev1Move.location {
                spatial[[0, 9, NSNumber(value: prev1Loc.y), NSNumber(value: prev1Loc.x)]] = 1.0
            }
            
            // Move 2 ago (player)
            if amountOfHistoryToTryToUse >= 2 && moveHistoryLen >= 2 && moveHistory[moveHistoryLen - 2].player == pla {
                let prev2Move = moveHistory[moveHistoryLen - 2]
                if prev2Move.isPass {
                    global[1] = 1.0
                } else if let prev2Loc = prev2Move.location {
                    spatial[[0, 10, NSNumber(value: prev2Loc.y), NSNumber(value: prev2Loc.x)]] = 1.0
                }
                
                // Move 3 ago (opponent)
                if amountOfHistoryToTryToUse >= 3 && moveHistoryLen >= 3 && moveHistory[moveHistoryLen - 3].player == opp {
                    let prev3Move = moveHistory[moveHistoryLen - 3]
                    if prev3Move.isPass {
                        global[2] = 1.0
                    } else if let prev3Loc = prev3Move.location {
                        spatial[[0, 11, NSNumber(value: prev3Loc.y), NSNumber(value: prev3Loc.x)]] = 1.0
                    }
                    
                    // Move 4 ago (player)
                    if amountOfHistoryToTryToUse >= 4 && moveHistoryLen >= 4 && moveHistory[moveHistoryLen - 4].player == pla {
                        let prev4Move = moveHistory[moveHistoryLen - 4]
                        if prev4Move.isPass {
                            global[3] = 1.0
                        } else if let prev4Loc = prev4Move.location {
                            spatial[[0, 12, NSNumber(value: prev4Loc.y), NSNumber(value: prev4Loc.x)]] = 1.0
                        }
                        
                        // Move 5 ago (opponent)
                        if amountOfHistoryToTryToUse >= 5 && moveHistoryLen >= 5 && moveHistory[moveHistoryLen - 5].player == opp {
                            let prev5Move = moveHistory[moveHistoryLen - 5]
                            if prev5Move.isPass {
                                global[4] = 1.0
                            } else if let prev5Loc = prev5Move.location {
                                spatial[[0, 13, NSNumber(value: prev5Loc.y), NSNumber(value: prev5Loc.x)]] = 1.0
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Fill planes 14-17: Ladder detection features
    /// - Feature 14: Current board ladders
    /// - Feature 15: Previous board ladders (1 turn ago)
    /// - Feature 16: Previous-previous board ladders (2 turns ago)
    /// - Feature 17: Ladder escape/capture moves
    /// - Parameters:
    ///   - spatial: MLMultiArray of shape [1, 22, 19, 19]
    ///   - board: Current board state
    ///   - nextPlayer: The player to move next (determines perspective)
    private static func fillPlanes14To17Ladders(spatial: MLMultiArray, board: Board, nextPlayer: Stone) {
        let opp: Stone = (nextPlayer == .black) ? .white : .black
        let numTurnsOfHistoryIncluded = 2 // For Chinese rules, use 2 turns of history
        
        // Feature 14: Current board ladders
        board.iterLadders { loc, workingMoves in
            let y = loc.y
            let x = loc.x
            spatial[[0, 14, NSNumber(value: y), NSNumber(value: x)]] = 1.0
            
            // Feature 17: Ladder escape moves (only for opponent stones with >1 liberty)
            let stone = board.stones[y][x]
            if stone == opp && board.liberties(of: loc) > 1 {
                for workingMove in workingMoves {
                    spatial[[0, 17, NSNumber(value: workingMove.y), NSNumber(value: workingMove.x)]] = 1.0
                }
            }
        }
        
        // Feature 15: Previous board ladders (1 turn ago)
        let prevBoard: Board
        if numTurnsOfHistoryIncluded < 1 {
            prevBoard = board
        } else {
            prevBoard = board.getBoardAtTurn(max(0, board.turnNumber - 1))
        }
        prevBoard.iterLadders { loc, workingMoves in
            let y = loc.y
            let x = loc.x
            spatial[[0, 15, NSNumber(value: y), NSNumber(value: x)]] = 1.0
        }
        
        // Feature 16: Previous-previous board ladders (2 turns ago)
        let prevPrevBoard: Board
        if numTurnsOfHistoryIncluded < 2 {
            prevPrevBoard = prevBoard
        } else {
            prevPrevBoard = board.getBoardAtTurn(max(0, board.turnNumber - 2))
        }
        prevPrevBoard.iterLadders { loc, workingMoves in
            let y = loc.y
            let x = loc.x
            spatial[[0, 16, NSNumber(value: y), NSNumber(value: x)]] = 1.0
        }
    }
    
    /// Fill planes 18-19: Area ownership (territory/area ownership)
    /// - Plane 18: 1.0 where area is owned by nextPlayer (current player)
    /// - Plane 19: 1.0 where area is owned by the opponent
    /// - Both planes are 0.0 for neutral/unowned territory
    private static func fillPlanes18And19Area(spatial: MLMultiArray, board: Board, nextPlayer: Stone) {
        let area = board.calculateArea()
        let oppStone: Stone = (nextPlayer == .black) ? .white : .black
        
        for y in 0..<19 {
            for x in 0..<19 {
                if let owner = area[y][x] {
                    // Plane 18: Own area (current player's perspective)
                    if owner == nextPlayer {
                        spatial[[0, 18, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                    }
                    // Plane 19: Opponent area
                    else if owner == oppStone {
                        spatial[[0, 19, NSNumber(value: y), NSNumber(value: x)]] = 1.0
                    }
                }
                // If owner is nil, both planes remain 0.0 (already zero-initialized)
            }
        }
    }
    
    // MARK: - Global Features (19 values)
    
    /// Fill global features for the neural network input
    /// - Parameters:
    ///   - global: MLMultiArray of shape [1, 19]
    ///   - board: Current board state (for move history, though pass history is set by fillPlanes9To13History)
    ///   - nextPlayer: The player to move next (determines komi perspective)
    ///   - komi: Komi value
    /// Note: Features 0-4 (pass history) are set by fillPlanes9To13History() before this function is called
    /// We don't initialize 0-4 here to preserve the pass history values
    private static func fillGlobalFeatures(global: MLMultiArray, board: Board, nextPlayer: Stone, komi: Float) {
        // Initialize features 5-18 to zero (0-4 are set by fillPlanes9To13History for pass history)
        for i in 5..<19 {
            global[i] = 0.0
        }
        
        // Features 0-4: Pass history are set by fillPlanes9To13History()
        // They remain as set (either 0.0 or 1.0 for passes)
        
        // Feature 5: Komi (from current player's perspective)
        // selfKomi is positive if komi benefits the current player
        // Standard komi benefits White, so:
        // - If White to move: selfKomi = komi
        // - If Black to move: selfKomi = -komi
        let boardArea: Float = 19.0 * 19.0
        let komiClipRadius: Float = 20.0
        var selfKomi = (nextPlayer == .white) ? komi : -komi
        
        // Clip komi to board area bounds
        let maxKomi = boardArea + komiClipRadius
        if selfKomi > maxKomi { selfKomi = maxKomi }
        if selfKomi < -maxKomi { selfKomi = -maxKomi }
        
        global[5] = NSNumber(value: selfKomi / 20.0)
        
        // Features 6-7: Ko rule encoding
        // For Chinese rules (simple ko): both are 0.0
        // - Positional/situational ko would set [6]=1.0, [7]=±0.5
        // global[6] = 0.0  // Already 0 (simple ko)
        // global[7] = 0.0  // Already 0
        
        // Feature 8: Multi-stone suicide allowed
        // Chinese rules allow multi-stone suicide
        global[8] = 1.0
        
        // Feature 9: Territory scoring
        // Chinese rules use area scoring, not territory scoring
        // global[9] = 0.0  // Already 0 (area scoring)
        
        // Features 10-11: Tax rule (seki handling for Japanese rules)
        // Chinese rules have no tax rule
        // global[10] = 0.0  // Already 0 (no tax)
        // global[11] = 0.0  // Already 0
        
        // Features 12-13: Encore phase (for Japanese rules cleanup)
        // Chinese rules have no encore phase
        // global[12] = 0.0  // Already 0
        // global[13] = 0.0  // Already 0
        
        // Feature 14: Pass would end phase (requires game state tracking)
        // global[14] = 0.0  // Already 0
        
        // Features 15-16: Playout doubling advantage (for handicap)
        // global[15] = 0.0  // Already 0
        // global[16] = 0.0  // Already 0
        
        // Feature 17: Button go variant
        // global[17] = 0.0  // Already 0
        
        // Feature 18: Komi parity wave (optional optimization)
        // global[18] = 0.0  // Already 0 (can be computed later)
    }
}

/// Represents the model output
public struct ModelOutput {
    public let policy: MLMultiArray  // [1, 19, 19]
    public let value: Float           // scalar
    public let ownership: MLMultiArray  // [1, 19, 19]
    
    public init(policy: MLMultiArray, value: Float, ownership: MLMultiArray) {
        self.policy = policy
        self.value = value
        self.ownership = ownership
    }
}