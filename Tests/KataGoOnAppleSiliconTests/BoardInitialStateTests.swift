import Testing
import Foundation
@testable import KataGoOnAppleSilicon

@Test func testBoardFreshInitialStateIsEmptyAndBlackToMove() async throws {
    let board = Board(size: 19)
    for y in 0..<19 {
        for x in 0..<19 {
            #expect(board.initialStones[y][x] == .empty)
        }
    }
    #expect(board.initialSideToMove == .black)
    #expect(board.sideToMove == .black)
}

@Test func testBoardCopyPreservesInitialStateAndSideToMove() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black
    board.initialSideToMove = .white
    board.sideToMove = .white
    let clone = board.copy()
    #expect(clone.initialStones[3][3] == .black)
    #expect(clone.initialSideToMove == .white)
    #expect(clone.sideToMove == .white)
}

@Test func testIsEmpty() async throws {
    let board = Board(size: 19)
    #expect(board.isEmpty())
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(!board.isEmpty())
}

@Test func testPlayMoveAdvancesSideToMove() async throws {
    let board = Board(size: 19)
    #expect(board.sideToMove == .black)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.sideToMove == .white)
    #expect(board.playMove(at: Point(x: 15, y: 15), stone: .white))
    #expect(board.sideToMove == .black)
}

@Test func testSequentialSameColorAdvancesToOpponent() async throws {
    // GTP's play command accepts any color; after two blacks the next-to-move is white.
    let board = Board(size: 19)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .black))
    #expect(board.sideToMove == .white)
}

@Test func testPlayPassAdvancesSideToMove() async throws {
    let board = Board(size: 19)
    #expect(board.playPass(stone: .black))
    #expect(board.sideToMove == .white)
}

@Test func testUndoOnEmptyBoardReturnsFalse() async throws {
    let board = Board(size: 19)
    #expect(board.undo() == false)
}

@Test func testUndoRemovesLastMove() async throws {
    let board = Board(size: 19)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .white))
    #expect(board.undo())
    #expect(board.stones[4][4] == .empty)
    #expect(board.stones[3][3] == .black)
    #expect(board.turnNumber == 1)
    #expect(board.moveHistory.count == 1)
    #expect(board.sideToMove == .white)
}

@Test func testUndoRebuildsCapturedStones() async throws {
    let board = Board(size: 19)
    // Set up a single white stone surrounded on three sides by black; then black captures.
    #expect(board.playMove(at: Point(x: 3, y: 2), stone: .black))
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .white))
    #expect(board.playMove(at: Point(x: 2, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 3, y: 4), stone: .black))
    // White stone still has one liberty at (4, 3).
    #expect(board.stones[3][3] == .white)
    #expect(board.playMove(at: Point(x: 4, y: 3), stone: .black))
    // White is captured.
    #expect(board.stones[3][3] == .empty)
    // Undo the capture — white should return.
    #expect(board.undo())
    #expect(board.stones[3][3] == .white)
    #expect(board.stones[3][4] == .empty)
}

@Test func testUndoPreservesInitialStones() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black  // Simulate handicap stone.
    board.initialSideToMove = .white
    board.clearToInitial()
    #expect(board.playMove(at: Point(x: 15, y: 15), stone: .white))
    #expect(board.undo())
    #expect(board.stones[3][3] == .black)   // Handicap stone preserved.
    #expect(board.stones[15][15] == .empty) // Played move removed.
    #expect(board.sideToMove == .white)     // Back to initial side-to-move.
    #expect(board.moveHistory.isEmpty)
}

@Test func testClearToInitialRestoresSnapshot() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black
    board.initialSideToMove = .white
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .white))
    #expect(board.playPass(stone: .black))
    board.clearToInitial()
    #expect(board.stones[3][3] == .black)
    #expect(board.stones[4][4] == .empty)
    #expect(board.koPoint == nil)
    #expect(board.turnNumber == 0)
    #expect(board.moveHistory.isEmpty)
    #expect(board.sideToMove == .white)
}
