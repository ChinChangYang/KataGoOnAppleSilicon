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
