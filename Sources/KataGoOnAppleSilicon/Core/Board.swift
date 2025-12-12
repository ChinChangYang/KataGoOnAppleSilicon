// Port of KataGo's Board logic to Swift
// Simplified for 19x19, Chinese rules

public enum Stone: Int {
    case empty = 0
    case black = 1
    case white = 2
}

public struct Point: Hashable {
    public let x: Int  // 0-18
    public let y: Int  // 0-18
    
    public init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }
    
    public var isValid: Bool {
        return x >= 0 && x < 19 && y >= 0 && y < 19
    }
}

public class Board {
    public private(set) var stones: [[Stone]]
    public private(set) var koPoint: Point?
    public private(set) var turnNumber: Int = 0
    public private(set) var komi: Float = 7.5
    
    public init() {
        stones = Array(repeating: Array(repeating: .empty, count: 19), count: 19)
    }
    
    public func copy() -> Board {
        let newBoard = Board()
        newBoard.stones = stones
        newBoard.koPoint = koPoint
        newBoard.turnNumber = turnNumber
        newBoard.komi = komi
        return newBoard
    }
    
    public func playMove(at point: Point, stone: Stone) -> Bool {
        guard point.isValid, stones[point.y][point.x] == .empty else { return false }
        
        // Place stone
        stones[point.y][point.x] = stone
        
        // Capture opponent stones
        let opponent = stone == .black ? Stone.white : .black
        var captured = false
        for neighbor in neighbors(of: point) {
            if stones[neighbor.y][neighbor.x] == opponent && liberties(of: neighbor) == 0 {
                captureGroup(at: neighbor)
                captured = true
            }
        }
        
        // Handle self-capture (suicide) - allowed per GTP spec
        if liberties(of: point) == 0 {
            // Remove own group if suicide
            captureGroup(at: point)
        }
        
        // Update ko
        koPoint = captured && capturedStones.count == 1 ? capturedStones.first : nil
        
        turnNumber += 1
        return true
    }
    
    public func isLegalMove(at point: Point, stone: Stone) -> Bool {
        guard point.isValid, stones[point.y][point.x] == .empty else { return false }
        return true
    }
    
    func liberties(of point: Point) -> Int {
        var visited = Set<Point>()
        return countLiberties(at: point, visited: &visited)
    }
    
    private func countLiberties(at point: Point, visited: inout Set<Point>) -> Int {
        guard point.isValid, !visited.contains(point) else { return 0 }
        visited.insert(point)
        
        let stone = stones[point.y][point.x]
        if stone == .empty { return 1 }
        
        var count = 0
        for neighbor in neighbors(of: point) {
            if stones[neighbor.y][neighbor.x] == .empty {
                count += 1
            } else if stones[neighbor.y][neighbor.x] == stone {
                count += countLiberties(at: neighbor, visited: &visited)
            }
        }
        return count
    }
    
    private func neighbors(of point: Point) -> [Point] {
        var neighbors = [Point]()
        if point.x > 0 { neighbors.append(Point(x: point.x - 1, y: point.y)) }
        if point.x < 18 { neighbors.append(Point(x: point.x + 1, y: point.y)) }
        if point.y > 0 { neighbors.append(Point(x: point.x, y: point.y - 1)) }
        if point.y < 18 { neighbors.append(Point(x: point.x, y: point.y + 1)) }
        return neighbors
    }
    
    private var capturedStones = [Point]()
    
    private func captureGroup(at point: Point) {
        let stone = stones[point.y][point.x]
        var group = [Point]()
        var visited = Set<Point>()
        findGroup(at: point, stone: stone, group: &group, visited: &visited)
        
        for p in group {
            stones[p.y][p.x] = .empty
            capturedStones.append(p)
        }
    }
    
    private func findGroup(at point: Point, stone: Stone, group: inout [Point], visited: inout Set<Point>) {
        guard point.isValid, !visited.contains(point), stones[point.y][point.x] == stone else { return }
        visited.insert(point)
        group.append(point)
        for neighbor in neighbors(of: point) {
            findGroup(at: neighbor, stone: stone, group: &group, visited: &visited)
        }
    }
    
    public func score() -> (black: Float, white: Float) {
        // Chinese area scoring
        var blackArea = 0
        var whiteArea = 0
        
        for y in 0..<19 {
            for x in 0..<19 {
                let stone = stones[y][x]
                if stone == .black {
                    blackArea += 1
                } else if stone == .white {
                    whiteArea += 1
                } else {
                    // Empty: check surrounded
                    if isSurroundedBy(x: x, y: y, stone: .black) {
                        blackArea += 1
                    } else if isSurroundedBy(x: x, y: y, stone: .white) {
                        whiteArea += 1
                    }
                }
            }
        }
        
        return (Float(blackArea), Float(whiteArea) + komi)
    }
    
    private func isSurroundedBy(x: Int, y: Int, stone: Stone) -> Bool {
        // Simple flood fill to check if empty area is surrounded
        var visited = Set<Point>()
        return floodFill(x: x, y: y, target: stone, visited: &visited)
    }
    
    private func floodFill(x: Int, y: Int, target: Stone, visited: inout Set<Point>) -> Bool {
        let point = Point(x: x, y: y)
        guard point.isValid, !visited.contains(point) else { return true }
        visited.insert(point)
        
        let current = stones[y][x]
        if current == .empty {
            for neighbor in neighbors(of: point) {
                if !floodFill(x: neighbor.x, y: neighbor.y, target: target, visited: &visited) {
                    return false
                }
            }
            return true
        } else {
            return current == target
        }
    }
}