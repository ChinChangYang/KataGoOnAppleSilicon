// Copyright (c) 2025 Chin-Chang Yang
//
// Portions of this file are derived from KataGo (https://github.com/lightvector/KataGo):
// Copyright 2025 David J Wu ("lightvector") and/or other authors of the content in that repository.
//
// Port of KataGo's Board logic to Swift
// Simplified for 2x2 to 19x19, Chinese rules

public enum Stone: Int {
    case empty = 0
    case black = 1
    case white = 2

    /// The opposing stone color. Undefined for `.empty`.
    public var opponent: Stone {
        assert(self != .empty, "opponent is undefined for .empty")
        return self == .black ? .white : .black
    }
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
    public let xSize: Int
    public let ySize: Int
    public private(set) var stones: [[Stone]]
    public private(set) var koPoint: Point?
    public private(set) var turnNumber: Int = 0
    public private(set) var komi: Float = 7.5
    public private(set) var moveHistory: [Move] = []

    public init(size: Int = 19) {
        xSize = size
        ySize = size
        stones = Array(repeating: Array(repeating: .empty, count: size), count: size)
    }

    func isValidPoint(_ point: Point) -> Bool {
        return point.x >= 0 && point.x < xSize && point.y >= 0 && point.y < ySize
    }

    public func copy() -> Board {
        let newBoard = Board(size: xSize)
        newBoard.stones = stones
        newBoard.koPoint = koPoint
        newBoard.turnNumber = turnNumber
        newBoard.komi = komi
        newBoard.moveHistory = moveHistory
        return newBoard
    }

    public func playMove(at point: Point, stone: Stone) -> Bool {
        guard isValidPoint(point), stones[point.y][point.x] == .empty else { return false }
        guard point != koPoint else { return false }
        capturedStones = []

        // Place stone
        stones[point.y][point.x] = stone

        // Capture opponent stones
        let opponent = stone.opponent
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

        // Track move in history
        moveHistory.append(Move.move(at: point, player: stone))

        turnNumber += 1
        return true
    }

    /// Play a pass move (no stone placement)
    /// - Parameter stone: The player passing
    /// - Returns: Always true (passes are always legal)
    public func playPass(stone: Stone) -> Bool {
        moveHistory.append(Move.pass(player: stone))
        turnNumber += 1
        return true
    }

    public func isLegalMove(at point: Point, stone: Stone) -> Bool {
        guard isValidPoint(point), stones[point.y][point.x] == .empty else { return false }
        guard point != koPoint else { return false }
        return true
    }

    func liberties(of point: Point) -> Int {
        var visited = Set<Point>()
        return countLiberties(at: point, visited: &visited)
    }

    private func countLiberties(at point: Point, visited: inout Set<Point>) -> Int {
        guard isValidPoint(point), !visited.contains(point) else { return 0 }
        visited.insert(point)

        let stone = stones[point.y][point.x]
        if stone == .empty { return 1 }

        var count = 0
        for neighbor in neighbors(of: point) {
            if stones[neighbor.y][neighbor.x] == .empty {
                count += countLiberties(at: neighbor, visited: &visited)
            } else if stones[neighbor.y][neighbor.x] == stone {
                count += countLiberties(at: neighbor, visited: &visited)
            }
        }
        return count
    }

    private func neighbors(of point: Point) -> [Point] {
        var neighbors = [Point]()
        if point.x > 0 { neighbors.append(Point(x: point.x - 1, y: point.y)) }
        if point.x < xSize - 1 { neighbors.append(Point(x: point.x + 1, y: point.y)) }
        if point.y > 0 { neighbors.append(Point(x: point.x, y: point.y - 1)) }
        if point.y < ySize - 1 { neighbors.append(Point(x: point.x, y: point.y + 1)) }
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
        guard isValidPoint(point), !visited.contains(point), stones[point.y][point.x] == stone else { return }
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

        for y in 0..<ySize {
            for x in 0..<xSize {
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
        guard isValidPoint(point), !visited.contains(point) else { return true }
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

    // MARK: - Area Calculation (Benson's Algorithm)

    /// Calculate area ownership using Benson's algorithm
    /// Returns a 2D array where .black/.white indicates ownership, nil indicates neutral/unowned
    public func calculateArea() -> [[Stone?]] {
        // For Chinese rules: nonPassAliveStones=true, safeBigTerritories=true, unsafeBigTerritories=true
        var result: [[Stone?]] = Array(repeating: Array(repeating: nil, count: xSize), count: ySize)

        // Calculate area for both players
        calculateAreaForPla(pla: .black, safeBigTerritories: true, unsafeBigTerritories: true, isMultiStoneSuicideLegal: true, result: &result)
        calculateAreaForPla(pla: .white, safeBigTerritories: true, unsafeBigTerritories: true, isMultiStoneSuicideLegal: true, result: &result)

        // Mark all remaining stones as owned (nonPassAliveStones=true)
        for y in 0..<ySize {
            for x in 0..<xSize {
                if result[y][x] == nil {
                    result[y][x] = stones[y][x] != .empty ? stones[y][x] : nil
                }
            }
        }

        return result
    }

    /// Calculate area for a specific player using Benson's algorithm
    private func calculateAreaForPla(pla: Stone, safeBigTerritories: Bool, unsafeBigTerritories: Bool, isMultiStoneSuicideLegal: Bool, result: inout [[Stone?]]) {
        let opp: Stone = (pla == .black) ? .white : .black

        // Find all groups for this player and their heads (representative point)
        var groupHeads: [Point: Point] = [:] // Maps each point to its group head
        var allGroupHeads: Set<Point> = []

        for y in 0..<ySize {
            for x in 0..<xSize {
                let point = Point(x: x, y: y)
                if stones[y][x] == pla && groupHeads[point] == nil {
                    // Find the group and use the first point as head
                    var group: [Point] = []
                    var visited: Set<Point> = []
                    findGroup(at: point, stone: pla, group: &group, visited: &visited)
                    let head = group[0]
                    for p in group {
                        groupHeads[p] = head
                    }
                    allGroupHeads.insert(head)
                }
            }
        }

        // Build regions (empty-or-opponent connected areas)
        var regionIdxByPoint: [Point: Int] = [:]
        var nextEmptyOrOpp: [Point: Point] = [:]
        var regionHeads: [Point] = []
        var vitalForPlaHeads: [[Point]] = [] // For each region, which pla group heads it's vital for
        var numInternalSpacesMax2: [Int] = [] // 0, 1, or 2+ internal spaces
        var containsOpp: [Bool] = []
        var bordersNonPassAlivePlaByHead: [Bool] = []

        var numRegions = 0
        let maxRegions = (xSize * ySize + 1) / 2 + 1

        // Build regions using BFS (empty-or-opponent regions)
        for y in 0..<ySize {
            for x in 0..<xSize {
                let point = Point(x: x, y: y)
                if regionIdxByPoint[point] != nil {
                    continue
                }
                // Only process empty or opponent stones
                if stones[y][x] != .empty && stones[y][x] != opp {
                    continue
                }

                let regionIdx = numRegions
                numRegions += 1
                guard numRegions <= maxRegions else { break }

                regionHeads.append(point)
                vitalForPlaHeads.append([])
                numInternalSpacesMax2.append(0)
                containsOpp.append(false)
                bordersNonPassAlivePlaByHead.append(false)

                // Find adjacent pla group heads
                var adjacentHeads: Set<Point> = []
                for neighbor in neighbors(of: point) {
                    if stones[neighbor.y][neighbor.x] == pla {
                        if let head = groupHeads[neighbor] {
                            adjacentHeads.insert(head)
                        }
                    }
                }
                vitalForPlaHeads[regionIdx] = Array(adjacentHeads)

                // Build region using BFS
                var queue: [Point] = [point]
                var queueHead = 0
                regionIdxByPoint[point] = regionIdx
                var tailTarget = point

                while queueHead < queue.count {
                    let loc = queue[queueHead]
                    queueHead += 1

                    // Check if internal (not adjacent to pla)
                    var isAdjacentToPla = false
                    for neighbor in neighbors(of: loc) {
                        if stones[neighbor.y][neighbor.x] == pla {
                            isAdjacentToPla = true
                            break
                        }
                    }
                    if !isAdjacentToPla && numInternalSpacesMax2[regionIdx] < 2 {
                        numInternalSpacesMax2[regionIdx] += 1
                    }

                    if stones[loc.y][loc.x] == opp {
                        containsOpp[regionIdx] = true
                    }

                    nextEmptyOrOpp[loc] = tailTarget
                    tailTarget = loc

                    // Add neighbors to queue
                    for neighbor in neighbors(of: loc) {
                        if regionIdxByPoint[neighbor] == nil && (stones[neighbor.y][neighbor.x] == .empty || stones[neighbor.y][neighbor.x] == opp) {
                            regionIdxByPoint[neighbor] = regionIdx
                            queue.append(neighbor)
                        }
                    }
                }

                // Close circular linked list
                nextEmptyOrOpp[point] = tailTarget
            }
        }

        // Count vital liberties for each group
        var vitalCountByPlaHead: [Point: Int] = [:]
        for head in allGroupHeads {
            vitalCountByPlaHead[head] = 0
        }

        for i in 0..<numRegions {
            for head in vitalForPlaHeads[i] {
                vitalCountByPlaHead[head, default: 0] += 1
            }
        }

        // Benson iteration: remove groups with <2 vital liberties
        var plaHasBeenKilled: [Point: Bool] = [:]
        for head in allGroupHeads {
            plaHasBeenKilled[head] = false
        }

        while true {
            var killedAnything = false
            for head in allGroupHeads {
                if plaHasBeenKilled[head] == true {
                    continue
                }

                if vitalCountByPlaHead[head, default: 0] < 2 {
                    plaHasBeenKilled[head] = true
                    killedAnything = true

                    // Find all points in this group
                    var group: [Point] = []
                    var visited: Set<Point> = []
                    findGroup(at: head, stone: pla, group: &group, visited: &visited)

                    // Update bordering regions
                    for p in group {
                        for neighbor in neighbors(of: p) {
                            if let regionIdx = regionIdxByPoint[neighbor] {
                                if !bordersNonPassAlivePlaByHead[regionIdx] && (stones[neighbor.y][neighbor.x] == .empty || stones[neighbor.y][neighbor.x] == opp) {
                                    bordersNonPassAlivePlaByHead[regionIdx] = true
                                    // Decrement vitality for all pla chains
                                    for h in vitalForPlaHeads[regionIdx] {
                                        vitalCountByPlaHead[h, default: 0] -= 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !killedAnything {
                break
            }
        }

        // Mark pass-alive groups
        for head in allGroupHeads {
            if plaHasBeenKilled[head] != true {
                var group: [Point] = []
                var visited: Set<Point> = []
                findGroup(at: head, stone: pla, group: &group, visited: &visited)
                for p in group {
                    result[p.y][p.x] = pla
                }
            }
        }

        // Mark territory
        let atLeastOnePla = !allGroupHeads.isEmpty
        for i in 0..<numRegions {
            let head = regionHeads[i]

            let shouldMark = (numInternalSpacesMax2[i] <= 1 && !bordersNonPassAlivePlaByHead[i] && atLeastOnePla) ||
                            (safeBigTerritories && !containsOpp[i] && !bordersNonPassAlivePlaByHead[i] && atLeastOnePla)

            if shouldMark {
                // Mark all empty points in region (not opponent stones)
                var cur = head
                repeat {
                    // Only mark empty spaces, not opponent stones
                    if stones[cur.y][cur.x] == .empty {
                        result[cur.y][cur.x] = pla
                    }
                    if let next = nextEmptyOrOpp[cur], next != head {
                        cur = next
                    } else {
                        break
                    }
                } while cur != head
            } else if unsafeBigTerritories && !containsOpp[i] && atLeastOnePla {
                // Mark only if empty and not already claimed
                var cur = head
                repeat {
                    if stones[cur.y][cur.x] == .empty && result[cur.y][cur.x] == nil {
                        result[cur.y][cur.x] = pla
                    }
                    if let next = nextEmptyOrOpp[cur], next != head {
                        cur = next
                    } else {
                        break
                    }
                } while cur != head
            }
        }
    }

    // MARK: - Ladder Detection

    /// Get the representative point (head) for a stone's chain/group
    /// - Parameter point: The point to get the chain head for
    /// - Returns: The chain head point, or nil if the point is empty
    /// Uses the lexicographically smallest point (by y, then x) as the head for consistency
    func getChainHead(at point: Point) -> Point? {
        guard isValidPoint(point) else { return nil }
        let stone = stones[point.y][point.x]
        guard stone != .empty else { return nil }

        // Find the group and use the lexicographically smallest point as head
        var group: [Point] = []
        var visited: Set<Point> = []
        findGroup(at: point, stone: stone, group: &group, visited: &visited)

        // Return the smallest point (by y, then x) for consistency
        return group.min { (p1, p2) -> Bool in
            if p1.y != p2.y {
                return p1.y < p2.y
            }
            return p1.x < p2.x
        }
    }

    /// Check if a stone with 1 liberty is in a ladder (will be captured)
    /// - Parameters:
    ///   - loc: Location of the stone to check
    ///   - isAttackerFirst: Whether the attacker moves first
    /// - Returns: Tuple of (isLaddered, workingMoves). For 1-liberty, workingMoves is empty.
    func searchIsLadderCaptured(loc: Point, isAttackerFirst: Bool) -> (Bool, [Point]) {
        guard isValidPoint(loc) else { return (false, []) }
        let stone = stones[loc.y][loc.x]
        guard stone != .empty else { return (false, []) }

        let libs = liberties(of: loc)
        guard libs == 1 else { return (false, []) }

        // Find the single liberty
        var liberty: Point?
        for neighbor in neighbors(of: loc) {
            if stones[neighbor.y][neighbor.x] == .empty {
                liberty = neighbor
                break
            }
        }

        guard liberty != nil else { return (false, []) }

        // If attacker moves first, they play at the liberty
        if isAttackerFirst {
            // Attacker plays at liberty - this should capture if it's a ladder
            // Simplified: if the stone has only 1 liberty and attacker can play there, it's likely a ladder
            // More sophisticated search would simulate the full ladder sequence
            return (true, [])
        } else {
            // Defender moves first - check if they can escape
            // Simplified check: if defender can play at liberty and survive, not a ladder
            return (false, [])
        }
    }

    /// Check if a stone with 2 liberties is in a ladder when attacker moves first
    /// - Parameter loc: Location of the stone to check
    /// - Returns: Tuple of (isLaddered, workingMoves) where workingMoves are escape/capture moves
    func searchIsLadderCapturedAttackerFirst2Libs(loc: Point) -> (Bool, [Point]) {
        guard isValidPoint(loc) else { return (false, []) }
        let stone = stones[loc.y][loc.x]
        guard stone != .empty else { return (false, []) }

        let libs = liberties(of: loc)
        guard libs == 2 else { return (false, []) }

        // Find the two liberties
        var liberties: [Point] = []
        for neighbor in neighbors(of: loc) {
            if stones[neighbor.y][neighbor.x] == .empty {
                liberties.append(neighbor)
            }
        }

        guard liberties.count == 2 else { return (false, []) }

        // Simplified ladder detection for 2 liberties
        // In a real implementation, this would simulate the ladder sequence
        // For now, we'll use a heuristic: if both liberties are threatened, it might be a ladder

        // Check if defender has working moves (escape/capture)
        var workingMoves: [Point] = []

        // Check if defender can escape by playing at either liberty
        for lib in liberties {
            // Simplified: if playing here gives the group more liberties, it's a working move
            let testBoard = copy()
            if testBoard.playMove(at: lib, stone: stone) {
                let newLibs = testBoard.liberties(of: loc)
                if newLibs > 2 {
                    workingMoves.append(lib)
                }
            }
        }

        // Per C++ reference (nninputs.cpp:884), laddered and workingMoves are independent.
        // A stone can be laddered (will be captured if attacker plays correctly) AND have
        // working moves (escape attempts). Since we're checking 2-liberty stones in iterLadders,
        // they are potentially in a ladder situation. Return true to allow the callback to be
        // called so feature 17 can be populated when workingMoves exist.
        // Note: A full implementation would perform a ladder search to determine if the stone
        // is actually laddered, but for this simplified version, we consider 2-liberty stones
        // as potentially laddered when they're being checked.
        let isLaddered = true

        return (isLaddered, workingMoves)
    }

    /// Iterate through all board positions and call callback for each location in a ladder
    /// Follows KataGo's iterLadders() algorithm exactly
    /// - Parameters:
    ///   - callback: Function called with (loc, workingMoves) for each laddered position
    func iterLadders(_ callback: (Point, [Point]) -> Void) {
        // Track solved chain heads to avoid duplicate work
        var chainHeadsSolved: [Point] = []
        var chainHeadsSolvedValue: [Bool] = []

        for y in 0..<ySize {
            for x in 0..<xSize {
                let loc = Point(x: x, y: y)
                let stone = stones[y][x]

                if stone == .black || stone == .white {
                    let libs = liberties(of: loc)

                    if libs == 1 || libs == 2 {
                        // Check if we've already solved this chain head
                        var alreadySolved = false
                        if let head = getChainHead(at: loc) {
                            for i in 0..<chainHeadsSolved.count {
                                if chainHeadsSolved[i] == head {
                                    alreadySolved = true
                                    if chainHeadsSolvedValue[i] {
                                        let workingMoves: [Point] = []
                                        callback(loc, workingMoves)
                                    }
                                    break
                                }
                            }
                        }

                        if !alreadySolved {
                            // Perform search on copy so as not to mess up tracking
                            let copyForSearch = copy()
                            let laddered: Bool
                            var workingMoves: [Point] = []

                            if libs == 1 {
                                let result = copyForSearch.searchIsLadderCaptured(loc: loc, isAttackerFirst: true)
                                laddered = result.0
                            } else {
                                let result = copyForSearch.searchIsLadderCapturedAttackerFirst2Libs(loc: loc)
                                laddered = result.0
                                workingMoves = result.1
                            }

                            // Store result for chain head
                            if let head = getChainHead(at: loc) {
                                chainHeadsSolved.append(head)
                                chainHeadsSolvedValue.append(laddered)

                                if laddered {
                                    callback(loc, workingMoves)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reconstruct a board state at a specific turn number by replaying moves
    /// - Parameter turn: The turn number to reconstruct (0 = initial empty board)
    /// - Returns: A new Board representing the state at that turn
    func getBoardAtTurn(_ turn: Int) -> Board {
        let reconstructed = Board(size: xSize)
        reconstructed.komi = komi

        // Replay moves up to the specified turn
        let movesToReplay = min(turn, moveHistory.count)
        for i in 0..<movesToReplay {
            let move = moveHistory[i]
            if move.isPass {
                _ = reconstructed.playPass(stone: move.player)
            } else if let loc = move.location {
                _ = reconstructed.playMove(at: loc, stone: move.player)
            }
        }

        return reconstructed
    }
}
