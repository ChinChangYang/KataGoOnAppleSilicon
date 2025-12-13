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
        Self.fillSpatialFeatures(spatial: spatial, board: board, nextPlayer: nextPlayer)
        
        // KataGo features: 19 global features
        let globalShape: [NSNumber] = [1, 19]
        self.global = try! MLMultiArray(shape: globalShape, dataType: .float16)
        Self.fillGlobalFeatures(global: global, nextPlayer: nextPlayer, komi: komi)
    }
    
    // MARK: - Spatial Features (22 planes)
    
    /// Fill spatial features for the neural network input
    /// - Parameters:
    ///   - spatial: MLMultiArray of shape [1, 22, 19, 19]
    ///   - board: Current board state
    ///   - nextPlayer: The player to move next (determines perspective)
    private static func fillSpatialFeatures(spatial: MLMultiArray, board: Board, nextPlayer: Stone) {
        fillPlane0OnBoard(spatial: spatial)
        fillPlanes1And2Stones(spatial: spatial, board: board, nextPlayer: nextPlayer)
        fillPlanes3To5Liberties(spatial: spatial, board: board)
        fillPlane6KoBan(spatial: spatial, board: board)
        fillPlane7KoRecaptureBlocked(spatial: spatial)
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