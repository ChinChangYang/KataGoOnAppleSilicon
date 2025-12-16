// Copyright (c) 2025 Chin-Chang Yang
//
// Portions of this file are derived from KataGo (https://github.com/lightvector/KataGo):
// Copyright 2025 David J Wu ("lightvector") and/or other authors of the content in that repository.
//
// This file implements the input feature encoding algorithm from KataGo's `fillRowV7()` function
// in `cpp/neuralnet/nninputs.cpp`, including the history filling algorithm (lines 2503-2562)
// and the pass ends phase algorithm.

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
        fillPlane8EncoreKoRecaptureBlocked(spatial: spatial)
        fillPlanes9To13History(spatial: spatial, global: global, board: board, nextPlayer: nextPlayer)
        fillPlanes14To17Ladders(spatial: spatial, board: board, nextPlayer: nextPlayer)
        fillPlanes18And19Area(spatial: spatial, board: board, nextPlayer: nextPlayer)
        fillPlanes20And21EncoreStones(spatial: spatial)
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
        // Explicitly zero out planes 1-2 first
        for plane in 1...2 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
        
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
        // Explicitly zero out planes 3-5 first
        for plane in 3...5 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
        
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
        // Explicitly zero out plane 6 first
        for y in 0..<19 {
            for x in 0..<19 {
                spatial[[0, 6, NSNumber(value: y), NSNumber(value: x)]] = 0.0
            }
        }
        
        if let ko = board.koPoint {
            spatial[[0, 6, NSNumber(value: ko.y), NSNumber(value: ko.x)]] = 1.0
        }
    }
    
    /// Fill plane 7: Ko recapture blocked (encore ko recapture blocked locations)
    /// For Chinese rules, this plane is all zeros (no encore features)
    private static func fillPlane7KoRecaptureBlocked(spatial: MLMultiArray) {
        // Explicitly zero out plane 7 for Chinese rules (no encore features)
        for y in 0..<19 {
            for x in 0..<19 {
                spatial[[0, 7, NSNumber(value: y), NSNumber(value: x)]] = 0.0
            }
        }
    }
    
    /// Fill plane 8: Encore ko recapture blocked (encore phase ko recapture blocked locations)
    /// For Chinese rules, this plane is all zeros (no encore features)
    private static func fillPlane8EncoreKoRecaptureBlocked(spatial: MLMultiArray) {
        // Explicitly zero out plane 8 for Chinese rules (no encore features)
        for y in 0..<19 {
            for x in 0..<19 {
                spatial[[0, 8, NSNumber(value: y), NSNumber(value: x)]] = 0.0
            }
        }
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
        // Explicitly zero out planes 9-13 first to ensure clean state
        // This is defensive programming to handle cases where previous state might affect results
        for plane in 9...13 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
        
        // Also zero out global features 0-4 (pass history) to ensure clean state
        for i in 0..<5 {
            global[i] = 0.0
        }
        
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
        // Explicitly zero out planes 14-17 first to ensure clean state
        // This is defensive programming to handle cases where previous state might affect results
        for plane in 14...17 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
        
        let opp: Stone = (nextPlayer == .black) ? .white : .black
        
        // Feature 14: Current board ladders
        // Note: We always use 2 turns of history for Chinese rules
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
        // numTurnsOfHistoryIncluded is always >= 1, so we always use history
        let prevBoard = board.getBoardAtTurn(max(0, board.turnNumber - 1))
        prevBoard.iterLadders { loc, workingMoves in
            let y = loc.y
            let x = loc.x
            spatial[[0, 15, NSNumber(value: y), NSNumber(value: x)]] = 1.0
        }
        
        // Feature 16: Previous-previous board ladders (2 turns ago)
        // numTurnsOfHistoryIncluded is always >= 2, so we always use history
        let prevPrevBoard = board.getBoardAtTurn(max(0, board.turnNumber - 2))
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
        // Explicitly zero out planes 18-19 first to ensure clean state
        // This is defensive programming to handle cases where previous state might affect results
        for plane in 18...19 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
        
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
    
    /// Fill planes 20-21: Encore stones (encore phase starting stones)
    /// For Chinese rules, these planes are all zeros (no encore features)
    private static func fillPlanes20And21EncoreStones(spatial: MLMultiArray) {
        // Explicitly zero out planes 20-21 for Chinese rules (no encore features)
        for plane in 20...21 {
            for y in 0..<19 {
                for x in 0..<19 {
                    spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]] = 0.0
                }
            }
        }
    }
    
    // MARK: - Global Features (19 values)
    
    // MARK: - Pass Ends Phase Detection (Feature 14)
    
    /// Calculate ko hash based on board state, ko point, and player to move
    /// This is a simplified hash for Chinese rules (simple ko)
    /// - Parameters:
    ///   - board: Current board state
    ///   - movePla: Player who would make the move
    /// - Returns: A hash value representing the ko situation
    private static func getKoHash(board: Board, movePla: Stone) -> UInt64 {
        var hash: UInt64 = 0
        
        // Hash board state (stones)
        for y in 0..<19 {
            for x in 0..<19 {
                let stone = board.stones[y][x]
                let stoneValue = stone.rawValue
                // Combine position and stone value into hash
                hash = hash &* 31 &+ UInt64(y * 19 + x) &* 7 &+ UInt64(stoneValue)
            }
        }
        
        // Hash ko point
        if let ko = board.koPoint {
            hash = hash &* 31 &+ UInt64(ko.y * 19 + ko.x) &* 17
        }
        
        // Hash player to move
        hash = hash &* 31 &+ UInt64(movePla.rawValue) &* 19
        
        return hash
    }
    
    /// Check if phase has spightlike ending and pass history clearing
    /// For Chinese rules (simple ko), this returns true
    /// - Returns: true if simple ko, spight ko, or encore phase
    private static func phaseHasSpightlikeEndingAndPassHistoryClearing() -> Bool {
        // For Chinese rules: simple ko rule
        // This would also be true for encore phase or spight ko, but those don't apply to Chinese rules
        return true // Simple ko rule
    }
    
    /// Calculate consecutive ending passes if a pass were made
    /// Implements KataGo's BoardHistory::newConsecutiveEndingPassesAfterPass()
    /// - Parameters:
    ///   - board: Current board state
    ///   - movePla: Player who would pass
    /// - Returns: Number of consecutive ending passes after the pass
    private static func newConsecutiveEndingPassesAfterPass(board: Board, movePla: Stone) -> Int {
        // Count current consecutive ending passes from move history
        // Only count passes from the end - any non-pass move resets the count
        var consecutiveEndingPasses = 0
        let moveHistory = board.moveHistory
        
        // Count backwards from the most recent move to find consecutive passes
        // Stop when we hit a non-pass move (which resets the count)
        var i = moveHistory.count - 1
        while i >= 0 {
            if moveHistory[i].isPass {
                consecutiveEndingPasses += 1
                i -= 1
            } else {
                // Non-pass move resets consecutive passes
                break
            }
        }
        
        // For simple ko (Chinese rules): increment count for the pass we're about to make
        // For spight ko: would reset to 0 (not applicable for Chinese rules)
        // For encore phase: would increment (not applicable for Chinese rules)
        var newConsecutiveEndingPasses = consecutiveEndingPasses
        if phaseHasSpightlikeEndingAndPassHistoryClearing() {
            // Simple ko: increment for the pass we're about to make
            newConsecutiveEndingPasses += 1
        } else {
            // For spight ko: reset to 0
            newConsecutiveEndingPasses = 0
        }
        
        return newConsecutiveEndingPasses
    }
    
    /// Extract ko hashes before passes from move history
    /// - Parameters:
    ///   - board: Current board state
    ///   - movePla: Player who would pass
    /// - Returns: Tuple of (black pass hashes, white pass hashes)
    private static func getPassHistoryHashes(board: Board, movePla: Stone) -> (blackHashes: [UInt64], whiteHashes: [UInt64]) {
        var blackHashes: [UInt64] = []
        var whiteHashes: [UInt64] = []
        
        // Reconstruct board state before each pass and calculate ko hash
        let moveHistory = board.moveHistory
        let reconstructedBoard = Board()
        var currentBoard = reconstructedBoard
        
        for move in moveHistory {
            if move.isPass {
                // Calculate ko hash BEFORE this pass (using current reconstructed board state)
                let koHash = getKoHash(board: currentBoard, movePla: move.player)
                
                if move.player == .black {
                    blackHashes.append(koHash)
                } else {
                    whiteHashes.append(koHash)
                }
                // Note: Pass doesn't change board state, so we don't need to update currentBoard
            } else if let loc = move.location {
                // Replay the move to reconstruct board state
                let newBoard = currentBoard.copy()
                _ = newBoard.playMove(at: loc, stone: move.player)
                currentBoard = newBoard
            }
        }
        
        return (blackHashes, whiteHashes)
    }
    
    /// Check if a pass would be a spight-style ending pass
    /// - Parameters:
    ///   - board: Current board state
    ///   - movePla: Player who would pass
    ///   - koHashBeforeMove: Ko hash before the pass
    /// - Returns: true if this pass would cause spight-style ending
    private static func wouldBeSpightlikeEndingPass(board: Board, movePla: Stone, koHashBeforeMove: UInt64) -> Bool {
        if !phaseHasSpightlikeEndingAndPassHistoryClearing() {
            return false
        }
        
        let (blackHashes, whiteHashes) = getPassHistoryHashes(board: board, movePla: movePla)
        
        if movePla == .black {
            return blackHashes.contains(koHashBeforeMove)
        } else {
            return whiteHashes.contains(koHashBeforeMove)
        }
    }
    
    /// Check if a pass would end the current phase
    /// Implements KataGo's BoardHistory::passWouldEndPhase() algorithm
    /// - Parameters:
    ///   - board: Current board state
    ///   - movePla: Player who would pass
    /// - Returns: true if a pass would end the phase
    private static func passWouldEndPhase(board: Board, movePla: Stone) -> Bool {
        let koHashBeforeMove = getKoHash(board: board, movePla: movePla)
        
        // Check if consecutive ending passes >= 2
        if newConsecutiveEndingPassesAfterPass(board: board, movePla: movePla) >= 2 {
            return true
        }
        
        // Check if spight-style ending
        if wouldBeSpightlikeEndingPass(board: board, movePla: movePla, koHashBeforeMove: koHashBeforeMove) {
            return true
        }
        
        return false
    }
    
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
        
        // Calculate selfKomi (perspective-aware komi) for features 5 and 18
        let selfKomi = calculateSelfKomi(nextPlayer: nextPlayer, komi: komi)
        
        // Fill individual features
        fillGlobalFeature5Komi(global: global, selfKomi: selfKomi)
        fillGlobalFeatures6To13ChineseRules(global: global)
        fillGlobalFeature14PassEndsPhase(global: global, board: board, nextPlayer: nextPlayer)
        fillGlobalFeatures15To17Unused(global: global)
        fillGlobalFeature18KomiParityWave(global: global, selfKomi: selfKomi)
    }
    
    /// Calculate selfKomi (komi from current player's perspective)
    /// - Parameters:
    ///   - nextPlayer: The player to move next
    ///   - komi: Komi value
    /// - Returns: Perspective-aware komi value, clipped to board area bounds
    private static func calculateSelfKomi(nextPlayer: Stone, komi: Float) -> Float {
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
        
        return selfKomi
    }
    
    /// Fill global feature 5: Komi (from current player's perspective)
    private static func fillGlobalFeature5Komi(global: MLMultiArray, selfKomi: Float) {
        global[5] = NSNumber(value: selfKomi / 20.0)
    }
    
    /// Fill global features 6-13: Chinese rules constants
    /// Features 6-7: Ko rule (0.0 for simple ko)
    /// Feature 8: Multi-stone suicide allowed (1.0)
    /// Feature 9: Territory scoring (0.0 for area scoring)
    /// Features 10-11: Tax rule (0.0, no tax rule)
    /// Features 12-13: Encore phase (0.0, no encore)
    private static func fillGlobalFeatures6To13ChineseRules(global: MLMultiArray) {
        // Features 6-7: Ko rule encoding
        // KataGo uses positional/situational ko encoding even for Chinese rules
        // - Feature 6: 1.0 indicates positional/situational ko
        // - Feature 7: 0.5 indicates ko rule sub-type
        global[6] = 1.0
        global[7] = 0.5
        
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
    }
    
    /// Fill global feature 14: Pass would end phase
    private static func fillGlobalFeature14PassEndsPhase(global: MLMultiArray, board: Board, nextPlayer: Stone) {
        let passEndsPhase = passWouldEndPhase(board: board, movePla: nextPlayer)
        global[14] = passEndsPhase ? 1.0 : 0.0
    }
    
    /// Fill global features 15-17: Unused features (set to 0.0)
    /// Features 15-16: Playout doubling advantage (for handicap)
    /// Feature 17: Button go variant
    private static func fillGlobalFeatures15To17Unused(global: MLMultiArray) {
        // Features 15-16: Playout doubling advantage (for handicap)
        // global[15] = 0.0  // Already 0
        // global[16] = 0.0  // Already 0
        
        // Feature 17: Button go variant
        // global[17] = 0.0  // Already 0
    }
    
    /// Fill global feature 18: Komi parity wave
    /// Triangular wave based on komi parity to help neural network understand komi effects
    /// Only computed for area scoring (Chinese rules) or encore phase >= 2
    /// Since we use area scoring, we always compute this feature
    private static func fillGlobalFeature18KomiParityWave(global: MLMultiArray, selfKomi: Float) {
        let xSize = 19
        let ySize = 19
        let boardAreaIsEven = (xSize * ySize) % 2 == 0
        // For 19x19: 361 % 2 = 1 (odd), so boardAreaIsEven = false
        
        // What is the parity of the komi values that can produce jigos?
        let drawableKomisAreEven = boardAreaIsEven
        
        // Find the difference between the komi viewed from our perspective and the nearest drawable komi below it
        let komiFloor: Float
        if drawableKomisAreEven {
            komiFloor = floor(selfKomi / 2.0) * 2.0
        } else {
            komiFloor = floor((selfKomi - 1.0) / 2.0) * 2.0 + 1.0
        }
        
        // Cap just in case we have floating point weirdness
        var delta = selfKomi - komiFloor
        // Clamp delta to [0.0, 2.0]
        if delta < 0.0 {
            delta = 0.0
        }
        if delta > 2.0 {
            delta = 2.0
        }
        
        // Create the triangle wave based on the difference
        let wave: Float
        if delta < 0.5 {
            wave = delta
        } else if delta < 1.5 {
            wave = 1.0 - delta
        } else {
            wave = delta - 2.0
        }
        
        global[18] = NSNumber(value: wave)
    }
}

/// Represents the model output
public struct ModelOutput {
    public let policy: MLMultiArray  // [1, 6, 362] - 6 channels, 362 positions (361 board + 1 pass)
    public let ownership: MLMultiArray  // [1, 19, 19] or [1, 1, 19, 19]
    
    // Full value array [1, 3]: [whiteWin, whiteLoss, noResult]
    public let valueArray: MLMultiArray
    
    // Misc value arrays for additional outputs
    public let miscValueArray: MLMultiArray?  // [1, 10] - misc value outputs
    public let moreMiscValueArray: MLMultiArray?  // [1, 8] - more misc value outputs
    
    public init(
        policy: MLMultiArray,
        ownership: MLMultiArray,
        valueArray: MLMultiArray,
        miscValueArray: MLMultiArray? = nil,
        moreMiscValueArray: MLMultiArray? = nil
    ) {
        self.policy = policy
        self.ownership = ownership
        self.valueArray = valueArray
        self.miscValueArray = miscValueArray
        self.moreMiscValueArray = moreMiscValueArray
    }
    
    /// Extract whiteWin, whiteLoss, noResult from value array
    public var whiteWin: Float {
        // Use multi-dimensional indexing for shape [1, 3]
        // Try doubleValue first (works better for float16), fallback to floatValue
        if valueArray.shape.count >= 2 {
            let value = valueArray[[0, 0]].doubleValue
            return value.isNaN ? valueArray[[0, 0]].floatValue : Float(value)
        } else {
            let value = valueArray.count > 0 ? valueArray[0].doubleValue : 0.0
            return value.isNaN ? (valueArray.count > 0 ? valueArray[0].floatValue : 0.0) : Float(value)
        }
    }
    
    public var whiteLoss: Float {
        // Use multi-dimensional indexing for shape [1, 3]
        if valueArray.shape.count >= 2 && valueArray.shape[1].intValue > 1 {
            let value = valueArray[[0, 1]].doubleValue
            return value.isNaN ? valueArray[[0, 1]].floatValue : Float(value)
        } else {
            let value = valueArray.count > 1 ? valueArray[1].doubleValue : 0.0
            return value.isNaN ? (valueArray.count > 1 ? valueArray[1].floatValue : 0.0) : Float(value)
        }
    }
    
    public var noResult: Float {
        // Use multi-dimensional indexing for shape [1, 3]
        if valueArray.shape.count >= 2 && valueArray.shape[1].intValue > 2 {
            let value = valueArray[[0, 2]].doubleValue
            return value.isNaN ? valueArray[[0, 2]].floatValue : Float(value)
        } else {
            let value = valueArray.count > 2 ? valueArray[2].doubleValue : 0.0
            return value.isNaN ? (valueArray.count > 2 ? valueArray[2].floatValue : 0.0) : Float(value)
        }
    }
    
    /// Extract whiteScoreMean from miscValueArray[0]
    public var whiteScoreMean: Float? {
        guard let array = miscValueArray, array.count > 0 else { return nil }
        // Use multi-dimensional indexing for shape [1, N]
        if array.shape.count >= 2 {
            return array[[0, 0]].floatValue
        } else {
            return array[0].floatValue
        }
    }
    
    /// Extract whiteScoreMeanSq from miscValueArray[1]
    public var whiteScoreMeanSq: Float? {
        guard let array = miscValueArray, array.count > 1 else { return nil }
        // Use multi-dimensional indexing for shape [1, N]
        if array.shape.count >= 2 && array.shape[1].intValue > 1 {
            return array[[0, 1]].floatValue
        } else {
            return array[1].floatValue
        }
    }
    
    /// Extract varTimeLeft from miscValueArray[3]
    public var varTimeLeft: Float? {
        guard let array = miscValueArray, array.count > 3 else { return nil }
        // Use multi-dimensional indexing for shape [1, N]
        if array.shape.count >= 2 && array.shape[1].intValue > 3 {
            return array[[0, 3]].floatValue
        } else {
            return array[3].floatValue
        }
    }
    
    /// Extract shorttermWinlossError - use moreMiscValueArray[0] (matches C++ coremlbackend.cpp)
    public var shorttermWinlossError: Float? {
        // C++ backend uses moreMiscValuesOutputBuf[0], not miscValuesOutputBuf[4]
        // See coremlbackend.cpp line 164-165
        guard let array = moreMiscValueArray, array.count > 0 else { return nil }
        if array.shape.count >= 2 {
            return array[[0, 0]].floatValue
        } else {
            return array[0].floatValue
        }
    }
    
    /// Extract shorttermScoreError - use moreMiscValueArray[1] (matches C++ coremlbackend.cpp)
    public var shorttermScoreError: Float? {
        // C++ backend uses moreMiscValuesOutputBuf[1], not miscValuesOutputBuf[5]
        // See coremlbackend.cpp line 166
        guard let array = moreMiscValueArray, array.count > 1 else { return nil }
        if array.shape.count >= 2 && array.shape[1].intValue > 1 {
            return array[[0, 1]].floatValue
        } else {
            return array[1].floatValue
        }
    }
    
    /// Extract whiteLead from miscValueArray[2]
    public var whiteLead: Float? {
        guard let array = miscValueArray, array.count > 2 else { return nil }
        // Use multi-dimensional indexing for shape [1, N]
        if array.shape.count >= 2 && array.shape[1].intValue > 2 {
            return array[[0, 2]].floatValue
        } else {
            return array[2].floatValue
        }
    }
    
    /// Extract raw policy values as a flat array [362] (361 board positions + 1 pass)
    private func extractRawPolicy() -> [Float] {
        var rawPolicy = Array(repeating: Float(0.0), count: 362)
        let shape = policy.shape.map { $0.intValue }
        let dimCount = shape.count
        
        // Extract board positions (0-360)
        for y in 0..<19 {
            for x in 0..<19 {
                let positionIndex = y * 19 + x
                let value: Float
                
                if dimCount == 3 && shape[1] == 6 && shape[2] == 362 {
                    // [1, 6, 362] format - channel 0 is the main policy channel
                    value = policy[[0, 0, NSNumber(value: positionIndex)]].floatValue
                } else if dimCount == 4 {
                    if shape[1] == 19 {
                        // [1, 19, 19, channels]
                        value = policy[[0, NSNumber(value: y), NSNumber(value: x), 0]].floatValue
                    } else {
                        // [1, channels, 19, 19]
                        value = policy[[0, 0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                    }
                } else if dimCount == 3 && shape[1] == 19 {
                    // [1, 19, 19]
                    value = policy[[0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                } else {
                    // Fallback
                    if positionIndex < policy.count {
                        value = policy[positionIndex].floatValue
                    } else {
                        value = 0.0
                    }
                }
                
                rawPolicy[positionIndex] = value
            }
        }
        
        // Extract pass move (index 361)
        if dimCount == 3 && shape[1] == 6 && shape[2] == 362 {
            rawPolicy[361] = policy[[0, 0, NSNumber(value: 361)]].floatValue
        } else if policy.count > 361 {
            rawPolicy[361] = policy[361].floatValue
        }
        
        return rawPolicy
    }
    
    /// Extract raw ownership values as a flat array [361] (19x19 board)
    private func extractRawOwnership() -> [Float] {
        var rawOwnership = Array(repeating: Float(0.0), count: 19 * 19)
        let shape = ownership.shape.map { $0.intValue }
        let is4D = shape.count == 4
        
        for y in 0..<19 {
            for x in 0..<19 {
                let positionIndex = y * 19 + x
                let value: Float
                
                if is4D {
                    value = ownership[[0, 0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                } else {
                    value = ownership[[0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                }
                
                rawOwnership[positionIndex] = value
            }
        }
        
        return rawOwnership
    }
    
    /// Post-process model outputs using KataGo's postprocessing logic
    public func postprocess(
        board: Board,
        nextPlayer: Stone,
        modelVersion: Int = 15,
        postProcessParams: PostProcessParams = .default
    ) -> PostProcessedModelOutput {
        // Extract raw values
        let rawWhiteWinProb = Double(whiteWin)
        let rawWhiteLossProb = Double(whiteLoss)
        let rawWhiteNoResultProb = Double(noResult)
        let rawWhiteScoreMean = Double(whiteScoreMean ?? 0.0)
        let rawWhiteScoreMeanSq = Double(whiteScoreMeanSq ?? 0.0)
        let rawWhiteLead = Double(whiteLead ?? 0.0)
        let rawVarTimeLeft = Double(varTimeLeft ?? 0.0)
        let rawShorttermWinlossError = Double(shorttermWinlossError ?? 0.0)
        let rawShorttermScoreError = Double(shorttermScoreError ?? 0.0)
        
        // Post-process value outputs
        let valueResults = postprocessValueOutputs(
            rawWhiteWinProb: rawWhiteWinProb,
            rawWhiteLossProb: rawWhiteLossProb,
            rawWhiteNoResultProb: rawWhiteNoResultProb,
            rawWhiteScoreMean: rawWhiteScoreMean,
            rawWhiteScoreMeanSq: rawWhiteScoreMeanSq,
            rawWhiteLead: rawWhiteLead,
            rawVarTimeLeft: rawVarTimeLeft,
            rawShorttermWinlossError: rawShorttermWinlossError,
            rawShorttermScoreError: rawShorttermScoreError,
            nextPlayer: nextPlayer,
            modelVersion: modelVersion,
            postProcessParams: postProcessParams
        )
        
        // Post-process policy
        let rawPolicy = extractRawPolicy()
        let policyProbs = postprocessPolicy(
            rawPolicy: rawPolicy,
            board: board,
            nextPlayer: nextPlayer,
            postProcessParams: postProcessParams
        )
        
        // Post-process ownership
        let rawOwnership = extractRawOwnership()
        let ownershipValues = postprocessOwnership(
            rawOwnership: rawOwnership,
            board: board,
            nextPlayer: nextPlayer,
            postProcessParams: postProcessParams
        )
        
        return PostProcessedModelOutput(
            whiteWinProb: valueResults.whiteWinProb,
            whiteLossProb: valueResults.whiteLossProb,
            whiteNoResultProb: valueResults.whiteNoResultProb,
            whiteScoreMean: valueResults.whiteScoreMean,
            whiteScoreMeanSq: valueResults.whiteScoreMeanSq,
            whiteLead: valueResults.whiteLead,
            varTimeLeft: valueResults.varTimeLeft,
            shorttermWinlossError: valueResults.shorttermWinlossError,
            shorttermScoreError: valueResults.shorttermScoreError,
            policyProbs: policyProbs,
            ownership: ownershipValues
        )
    }
}