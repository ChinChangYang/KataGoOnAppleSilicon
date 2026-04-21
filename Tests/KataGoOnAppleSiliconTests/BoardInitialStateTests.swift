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
