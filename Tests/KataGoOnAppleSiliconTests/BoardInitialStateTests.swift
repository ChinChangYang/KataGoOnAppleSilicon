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

// MARK: - placeFixedHandicap

@Test func testPlaceFixedHandicap2On19x19() async throws {
    let board = Board(size: 19)
    let points = try board.placeFixedHandicap(n: 2)
    // Scan order: y=0..<19, x=0..<19. (15, 3) appears before (3, 15).
    #expect(points == [Point(x: 15, y: 3), Point(x: 3, y: 15)])
    #expect(board.stones[3][15] == .black)
    #expect(board.stones[15][3] == .black)
    #expect(board.initialStones[3][15] == .black)
    #expect(board.initialStones[15][3] == .black)
    #expect(board.initialSideToMove == .white)
    #expect(board.sideToMove == .white)
    #expect(board.moveHistory.isEmpty)
    #expect(board.turnNumber == 0)
}

@Test func testPlaceFixedHandicap9On19x19() async throws {
    let board = Board(size: 19)
    let points = try board.placeFixedHandicap(n: 9)
    #expect(points == [
        Point(x: 3, y: 3), Point(x: 9, y: 3), Point(x: 15, y: 3),
        Point(x: 3, y: 9), Point(x: 9, y: 9), Point(x: 15, y: 9),
        Point(x: 3, y: 15), Point(x: 9, y: 15), Point(x: 15, y: 15),
    ])
}

@Test func testPlaceFixedHandicap5On13x13() async throws {
    let board = Board(size: 13)
    let points = try board.placeFixedHandicap(n: 5)
    #expect(points == [
        Point(x: 3, y: 3), Point(x: 9, y: 3),
        Point(x: 6, y: 6),
        Point(x: 3, y: 9), Point(x: 9, y: 9),
    ])
}

@Test func testPlaceFixedHandicapRejectsTooSmallBoard() async throws {
    let board = Board(size: 6)
    do {
        _ = try board.placeFixedHandicap(n: 2)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Board is too small for fixed handicap, try place_free_handicap")
    }
}

@Test func testPlaceFixedHandicapRejectsEvenDimAboveFour() async throws {
    let board = Board(size: 8)
    do {
        _ = try board.placeFixedHandicap(n: 5)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 4 is not allowed on boards with even dimensions, try place_free_handicap")
    }
}

@Test func testPlaceFixedHandicapRejectsSize7AboveFour() async throws {
    let board = Board(size: 7)
    do {
        _ = try board.placeFixedHandicap(n: 5)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 4 is not allowed on boards with size 7, try place_free_handicap")
    }
}

@Test func testPlaceFixedHandicapRejectsAboveNine() async throws {
    let board = Board(size: 19)
    do {
        _ = try board.placeFixedHandicap(n: 10)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 9 is not allowed, try place_free_handicap")
    }
}

// MARK: - setStonesFailIfNoLibs

@Test func testSetStonesFailIfNoLibsAcceptsValid() async throws {
    let board = Board(size: 19)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 3, y: 3), .black),
        (Point(x: 15, y: 15), .black),
    ])
    #expect(ok)
    #expect(board.stones[3][3] == .black)
    #expect(board.stones[15][15] == .black)
}

@Test func testSetStonesFailIfNoLibsRejectsDuplicate() async throws {
    let board = Board(size: 19)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 3, y: 3), .black),
        (Point(x: 3, y: 3), .black),
    ])
    #expect(!ok)
    // Stones are not committed on failure.
    #expect(board.stones[3][3] == .empty)
}

@Test func testSetStonesFailIfNoLibsRejectsZeroLiberty() async throws {
    // 2x2 board fully filled with black: no liberties anywhere.
    let board = Board(size: 2)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 0, y: 0), .black),
        (Point(x: 1, y: 0), .black),
        (Point(x: 0, y: 1), .black),
        (Point(x: 1, y: 1), .black),
    ])
    #expect(!ok)
    #expect(board.stones[0][0] == .empty)
    #expect(board.stones[1][1] == .empty)
}

// MARK: - placeFreeHandicap

@Test func testPlaceFreeHandicapBasic() async throws {
    let board = Board(size: 19)
    let ok = board.placeFreeHandicap([
        Point(x: 3, y: 15),   // D4
        Point(x: 15, y: 15),  // Q4
        Point(x: 15, y: 3),   // Q16
    ])
    #expect(ok)
    #expect(board.stones[15][3] == .black)
    #expect(board.initialStones[15][3] == .black)
    #expect(board.initialSideToMove == .white)
    #expect(board.sideToMove == .white)
    #expect(board.moveHistory.isEmpty)
    #expect(board.turnNumber == 0)
}

@Test func testPlaceFreeHandicapRejectsInvalid() async throws {
    let board = Board(size: 2)
    // Fill 2x2 — every stone has zero liberties, should fail.
    let ok = board.placeFreeHandicap([
        Point(x: 0, y: 0),
        Point(x: 1, y: 0),
        Point(x: 0, y: 1),
        Point(x: 1, y: 1),
    ])
    #expect(!ok)
    #expect(board.initialStones[0][0] == .empty)
    #expect(board.initialSideToMove == .black)
}
