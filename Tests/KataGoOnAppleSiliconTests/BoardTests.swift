import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - Board Tests

@Test func testBoardInitialization() async throws {
    let board = Board()
    #expect(board.stones.count == 19)
    #expect(board.stones[0].count == 19)
    #expect(board.koPoint == nil)
    #expect(board.turnNumber == 0)
    #expect(board.komi == 7.5)
}

@Test func testBoardCopy() async throws {
    let board = Board()
    let point = Point(x: 3, y: 3)
    _ = board.playMove(at: point, stone: .black)
    let copy = board.copy()
    #expect(copy.stones[3][3] == .black)
    #expect(copy.turnNumber == 1)
}

@Test func testPlayMoveValid() async throws {
    let board = Board()
    let point = Point(x: 3, y: 3)
    let success = board.playMove(at: point, stone: .black)
    #expect(success)
    #expect(board.stones[3][3] == .black)
    #expect(board.turnNumber == 1)
}

@Test func testPlayMoveInvalidOccupied() async throws {
    let board = Board()
    let point = Point(x: 3, y: 3)
    _ = board.playMove(at: point, stone: .black)
    let success = board.playMove(at: point, stone: .white)
    #expect(!success)
    #expect(board.stones[3][3] == .black)
}

@Test func testPlayMoveInvalidOutOfBounds() async throws {
    let board = Board()
    let point = Point(x: 19, y: 19)
    let success = board.playMove(at: point, stone: .black)
    #expect(!success)
}

@Test func testIsLegalMove() async throws {
    let board = Board()
    let point = Point(x: 3, y: 3)
    #expect(board.isLegalMove(at: point, stone: .black))
    _ = board.playMove(at: point, stone: .black)
    #expect(!board.isLegalMove(at: point, stone: .white))
}

@Test func testCaptureSingleStone() async throws {
    let board = Board()
    // Place black stone
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    // Surround with white stones
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    // Capture with white
    let success = board.playMove(at: Point(x: 3, y: 4), stone: .white)
    #expect(success)
    #expect(board.stones[3][3] == .empty)
}

@Test func testCaptureMultipleStones() async throws {
    let board = Board()
    // Place black stones in a line
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 4), stone: .black)
    // Surround with white
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 2, y: 4), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 5), stone: .white)
    // Capture
    let success = board.playMove(at: Point(x: 3, y: 6), stone: .white)
    #expect(success)
    #expect(board.stones[3][3] == .empty)
    #expect(board.stones[4][3] == .empty)
}

@Test func testSuicideAllowed() async throws {
    let board = Board()
    // Place white stones around a point
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 4), stone: .white)
    // Play black in the center (suicide) - should succeed and stone is removed
    let success = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    #expect(success)
    #expect(board.stones[3][3] == .empty)  // Stone is removed due to self-capture
}

@Test func testLiberties() async throws {
    let board = Board()
    // Single stone
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    #expect(board.liberties(of: Point(x: 3, y: 3)) == 4)
    
    // Add adjacent stone
    _ = board.playMove(at: Point(x: 3, y: 4), stone: .black)
    #expect(board.liberties(of: Point(x: 3, y: 3)) == 6) // Shared liberties
}

@Test func testScoring() async throws {
    let board = Board()
    // Place some stones
    _ = board.playMove(at: Point(x: 0, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 18, y: 18), stone: .white)
    
    let score = board.score()
    #expect(score.black >= 1)
    #expect(score.white >= 1 + board.komi)
}

@Test func testCornerStoneLiberties() async throws {
    let board = Board()
    // Corner stone at (0, 0) should have 2 liberties
    _ = board.playMove(at: Point(x: 0, y: 0), stone: .black)
    #expect(board.liberties(of: Point(x: 0, y: 0)) == 2)
}

@Test func testEdgeStoneLiberties() async throws {
    let board = Board()
    // Edge stone at (0, 1) should have 3 liberties
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .black)
    #expect(board.liberties(of: Point(x: 0, y: 1)) == 3)
}

@Test func testEmptyPointLiberties() async throws {
    let board = Board()
    // Empty point should have 1 liberty (itself counts as empty)
    #expect(board.liberties(of: Point(x: 5, y: 5)) == 1)
}

@Test func testScoringWithSurroundedTerritory() async throws {
    let board = Board()
    // Create a small surrounded territory in the corner
    // Black surrounds corner at (0,0)
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .black)
    
    let score = board.score()
    #expect(score.black >= 3)  // 2 black stones + 1 surrounded territory
}

@Test func testScoringWithWhiteSurroundedTerritory() async throws {
    let board = Board()
    // White surrounds corner at (18,18)
    _ = board.playMove(at: Point(x: 17, y: 18), stone: .white)
    _ = board.playMove(at: Point(x: 18, y: 17), stone: .white)
    
    let score = board.score()
    #expect(score.white >= 3 + board.komi)  // 2 white stones + 1 surrounded territory + komi
}

@Test func testPointIsValid() async throws {
    let validPoint = Point(x: 10, y: 10)
    let invalidPoint1 = Point(x: -1, y: 10)
    let invalidPoint2 = Point(x: 10, y: 19)
    
    #expect(validPoint.isValid)
    #expect(!invalidPoint1.isValid)
    #expect(!invalidPoint2.isValid)
}

