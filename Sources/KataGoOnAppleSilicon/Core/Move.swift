// Move representation for game history tracking
public struct Move {
    /// The location of the move, or nil for pass moves
    public let location: Point?
    /// The player who made the move
    public let player: Stone
    
    public init(location: Point?, player: Stone) {
        self.location = location
        self.player = player
    }
    
    /// Create a pass move
    public static func pass(player: Stone) -> Move {
        return Move(location: nil, player: player)
    }
    
    /// Create a regular move at a point
    public static func move(at point: Point, player: Stone) -> Move {
        return Move(location: point, player: player)
    }
    
    /// Check if this is a pass move
    public var isPass: Bool {
        return location == nil
    }
}