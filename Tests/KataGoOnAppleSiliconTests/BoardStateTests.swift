import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - BoardState Tests

@Test func testBoardStateDirectInit() async throws {
    let spatialShape: [NSNumber] = [1, 22, 19, 19]
    let spatial = try MLMultiArray(shape: spatialShape, dataType: .float16)
    let globalShape: [NSNumber] = [1, 19]
    let global = try MLMultiArray(shape: globalShape, dataType: .float16)
    
    let boardState = BoardState(spatial: spatial, global: global)
    #expect(boardState.spatial.count == spatial.count)
    #expect(boardState.global.count == global.count)
}

@Test func testBoardStateWithBlackStones() async throws {
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    let boardState = BoardState(board: board)
    // Verify spatial array is created with proper dimensions
    #expect(boardState.spatial.shape == [1, 22, 19, 19] as [NSNumber])
}

@Test func testBoardStateWithWhiteStones() async throws {
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    let boardState = BoardState(board: board)
    // Verify spatial array is created with proper dimensions
    #expect(boardState.spatial.shape == [1, 22, 19, 19] as [NSNumber])
}

@Test func testBoardStatePlane6KoBan() async throws {
    let board = Board()
    // Create a ko situation:
    // Place white stone at (1,0) that will be captured
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .white)
    // Surround it with black stones
    _ = board.playMove(at: Point(x: 0, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    // White stone at (1,0) is now captured, ko point should be (1,0)
    #expect(board.koPoint != nil)
    #expect(board.koPoint?.x == 1)
    #expect(board.koPoint?.y == 0)
    
    // Create BoardState and verify plane 6 has 1.0 at ko point
    let boardState = BoardState(board: board, nextPlayer: .white)
    let koValue = boardState.spatial[[0, 6, 0, 1]].floatValue
    #expect(koValue == 1.0)
    
    // Verify other positions on plane 6 are 0.0
    let nonKoValue = boardState.spatial[[0, 6, 5, 5]].floatValue
    #expect(nonKoValue == 0.0)
}

@Test func testBoardStatePlane6NoKo() async throws {
    let board = Board()
    // Empty board should have no ko point
    #expect(board.koPoint == nil)
    
    let boardState = BoardState(board: board)
    // Verify plane 6 is all zeros
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 6, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane7KoRecaptureBlocked() async throws {
    let board = Board()
    // Empty board should have plane 7 all zeros (Chinese rules have no encore)
    let boardState = BoardState(board: board)
    // Verify plane 7 is all zeros
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 7, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane7WithKo() async throws {
    let board = Board()
    // Create a ko situation:
    // Place white stone at (1,0) that will be captured
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .white)
    // Surround it with black stones
    _ = board.playMove(at: Point(x: 0, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    // White stone at (1,0) is now captured, ko point should be (1,0)
    #expect(board.koPoint != nil)
    
    // Create BoardState and verify plane 7 is still all zeros
    // (Chinese rules don't have encore ko, so plane 7 remains zeros even with ko)
    let boardState = BoardState(board: board, nextPlayer: .white)
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 7, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

// MARK: - Plane 3 (1 Liberty/Atari) Tests

@Test func testBoardStatePlane3OneLiberty() async throws {
    // CRITICAL: Test plane 3 with exactly 1 liberty (atari situation)
    // This covers line 84 which currently has 0% coverage
    let board = Board()
    
    // Place black stone at (1,1)
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    
    // Surround it with white stones to leave exactly 1 liberty
    // Place white at (0,1), (2,1), (1,0) - black now has 1 liberty at (1,2)
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .white)
    _ = board.playMove(at: Point(x: 2, y: 1), stone: .white)
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .white)
    
    // Verify black stone at (1,1) has exactly 1 liberty
    let libertyCount = board.liberties(of: Point(x: 1, y: 1))
    #expect(libertyCount == 1)
    
    // Create BoardState and verify plane 3 has 1.0 at (1,1)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let plane3Value = boardState.spatial[[0, 3, 1, 1]].floatValue
    #expect(plane3Value == 1.0)
    
    // Verify other positions on plane 3 are 0.0
    let otherValue = boardState.spatial[[0, 3, 5, 5]].floatValue
    #expect(otherValue == 0.0)
}

@Test func testBoardStatePlane4TwoLiberties() async throws {
    // Test plane 4 with exactly 2 liberties
    let board = Board()
    
    // Place black stone at (2,2)
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)
    
    // Surround it with white stones to leave exactly 2 liberties
    // Place white at (1,2) and (3,2) - black now has 2 liberties at (2,1) and (2,3)
    _ = board.playMove(at: Point(x: 1, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    
    // Verify black stone at (2,2) has exactly 2 liberties
    let libertyCount = board.liberties(of: Point(x: 2, y: 2))
    #expect(libertyCount == 2)
    
    // Create BoardState and verify plane 4 has 1.0 at (2,2)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let plane4Value = boardState.spatial[[0, 4, 2, 2]].floatValue
    #expect(plane4Value == 1.0)
    
    // Verify other positions on plane 4 are 0.0
    let otherValue = boardState.spatial[[0, 4, 5, 5]].floatValue
    #expect(otherValue == 0.0)
}

@Test func testBoardStatePlane5ThreeLiberties() async throws {
    // Test plane 5 with exactly 3 liberties
    let board = Board()
    
    // Place black stone at (2,2)
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)
    
    // Surround it with white stones to leave exactly 3 liberties
    // Place white at (1,2) - black now has 3 liberties
    _ = board.playMove(at: Point(x: 1, y: 2), stone: .white)
    
    // Verify black stone at (2,2) has exactly 3 liberties
    let libertyCount = board.liberties(of: Point(x: 2, y: 2))
    #expect(libertyCount == 3)
    
    // Create BoardState and verify plane 5 has 1.0 at (2,2)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let plane5Value = boardState.spatial[[0, 5, 2, 2]].floatValue
    #expect(plane5Value == 1.0)
    
    // Verify other positions on plane 5 are 0.0
    let otherValue = boardState.spatial[[0, 5, 5, 5]].floatValue
    #expect(otherValue == 0.0)
}

// MARK: - Planes 1-2 Perspective Switching Tests

@Test func testBoardStatePlane1WithWhiteNextPlayer() async throws {
    // Test perspective switching: when nextPlayer = .white, plane 1 should contain white stones
    // This increases coverage for line 65 (if stone == ownStone) which only has 1 execution
    let board = Board()
    
    // Place white stones
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .white)
    
    // Place black stones
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .black)
    
    // Create BoardState with white as nextPlayer
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // Plane 1 should contain white stones (own stones from white's perspective)
    let whiteStone1 = boardState.spatial[[0, 1, 3, 3]].floatValue
    #expect(whiteStone1 == 1.0)
    
    let whiteStone2 = boardState.spatial[[0, 1, 5, 5]].floatValue
    #expect(whiteStone2 == 1.0)
    
    // Plane 2 should contain black stones (opponent stones from white's perspective)
    let blackStone = boardState.spatial[[0, 2, 4, 4]].floatValue
    #expect(blackStone == 1.0)
    
    // Empty positions should be 0.0 on both planes
    let emptyPlane1 = boardState.spatial[[0, 1, 0, 0]].floatValue
    #expect(emptyPlane1 == 0.0)
    
    let emptyPlane2 = boardState.spatial[[0, 2, 0, 0]].floatValue
    #expect(emptyPlane2 == 0.0)
}

@Test func testBoardStatePlanes1And2PerspectiveSwitching() async throws {
    // Test that perspective correctly switches when nextPlayer changes
    let board = Board()
    
    // Place both black and white stones
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    
    // Test with black as nextPlayer
    let boardStateBlack = BoardState(board: board, nextPlayer: .black)
    #expect(boardStateBlack.spatial[[0, 1, 2, 2]].floatValue == 1.0) // Black stone in plane 1
    #expect(boardStateBlack.spatial[[0, 2, 3, 3]].floatValue == 1.0) // White stone in plane 2
    
    // Test with white as nextPlayer
    let boardStateWhite = BoardState(board: board, nextPlayer: .white)
    #expect(boardStateWhite.spatial[[0, 1, 3, 3]].floatValue == 1.0) // White stone in plane 1
    #expect(boardStateWhite.spatial[[0, 2, 2, 2]].floatValue == 1.0) // Black stone in plane 2
}

// MARK: - Global Feature 5 (Komi) Tests

@Test func testBoardStateGlobalKomiBlackToMove() async throws {
    // Test global[5] with black to move (should be negative komi)
    let board = Board()
    let komi: Float = 7.5
    let boardState = BoardState(board: board, nextPlayer: .black, komi: komi)
    
    // Black to move: selfKomi = -komi, so global[5] = -komi/20.0
    let expectedKomi = -komi / 20.0
    let actualKomi = boardState.global[5].floatValue
    #expect(actualKomi == expectedKomi)
    #expect(actualKomi == -0.375)
}

@Test func testBoardStateGlobalKomiWhiteToMove() async throws {
    // Test global[5] with white to move (should be positive komi)
    let board = Board()
    let komi: Float = 7.5
    let boardState = BoardState(board: board, nextPlayer: .white, komi: komi)
    
    // White to move: selfKomi = komi, so global[5] = komi/20.0
    let expectedKomi = komi / 20.0
    let actualKomi = boardState.global[5].floatValue
    #expect(actualKomi == expectedKomi)
    #expect(actualKomi == 0.375)
}

@Test func testBoardStateGlobalKomiCustomValue() async throws {
    // Test global[5] with custom komi value
    let board = Board()
    let customKomi: Float = 6.5
    let boardState = BoardState(board: board, nextPlayer: .black, komi: customKomi)
    
    // Black to move: selfKomi = -komi, so global[5] = -komi/20.0
    let expectedKomi = -customKomi / 20.0
    let actualKomi = boardState.global[5].floatValue
    #expect(abs(actualKomi - expectedKomi) < 0.0001)
    #expect(abs(actualKomi - (-0.325)) < 0.0001)
}

@Test func testBoardStateGlobalKomiClippingLarge() async throws {
    // CRITICAL: Test komi clipping with extreme positive value
    // maxKomi = boardArea + komiClipRadius = 19*19 + 20 = 381.0
    let board = Board()
    let largeKomi: Float = 500.0  // Exceeds maxKomi of 381.0
    
    let boardState = BoardState(board: board, nextPlayer: .white, komi: largeKomi)
    
    // Should clip to maxKomi = 381.0, then divided by 20.0
    let expectedKomi: Float = 381.0 / 20.0
    let actualKomi = boardState.global[5].floatValue
    // Use tolerance for floating point precision (float16/float32 precision)
    #expect(abs(actualKomi - expectedKomi) < 0.01)
    // Verify it's close to 19.05 (within reasonable tolerance)
    #expect(abs(actualKomi - 19.05) < 0.01)
}

@Test func testBoardStateGlobalKomiClippingLargeNegative() async throws {
    // CRITICAL: Test komi clipping with extreme negative value
    // maxKomi = boardArea + komiClipRadius = 19*19 + 20 = 381.0
    let board = Board()
    let largeNegativeKomi: Float = -500.0  // Exceeds -maxKomi of -381.0
    
    let boardState = BoardState(board: board, nextPlayer: .black, komi: largeNegativeKomi)
    
    // Black to move: selfKomi = -komi = -(-500.0) = 500.0, but should clip to 381.0
    // Then global[5] = 381.0 / 20.0
    let expectedKomi: Float = 381.0 / 20.0
    let actualKomi = boardState.global[5].floatValue
    // Use tolerance for floating point precision (float16/float32 precision)
    #expect(abs(actualKomi - expectedKomi) < 0.01)
    // Verify it's close to 19.05 (within reasonable tolerance)
    #expect(abs(actualKomi - 19.05) < 0.01)
}

@Test func testBoardStateGlobalKomiClippingNegative() async throws {
    // Test komi clipping with extreme negative komi when white to move
    // If komi = -500.0 and white to move, selfKomi = komi = -500.0
    // This is < -maxKomi (-381.0), so should clip to -381.0
    // Then global[5] = -381.0 / 20.0
    let board = Board()
    let largeNegativeKomi: Float = -500.0
    
    let boardState = BoardState(board: board, nextPlayer: .white, komi: largeNegativeKomi)
    
    let expectedKomi: Float = -381.0 / 20.0
    let actualKomi = boardState.global[5].floatValue
    // Use tolerance for floating point precision (float16/float32 precision)
    #expect(abs(actualKomi - expectedKomi) < 0.01)
    // Verify it's close to -19.05 (within reasonable tolerance)
    #expect(abs(actualKomi - (-19.05)) < 0.01)
}

// MARK: - Planes 18-19 (Area Features) Tests

@Test func testBoardStatePlanes18And19EmptyBoard() async throws {
    // Empty board should have all zeros on both planes
    let board = Board()
    let boardState = BoardState(board: board)
    
    // Verify planes 18 and 19 are all zeros
    for y in 0..<19 {
        for x in 0..<19 {
            let plane18Value = boardState.spatial[[0, 18, NSNumber(value: y), NSNumber(value: x)]].floatValue
            let plane19Value = boardState.spatial[[0, 19, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(plane18Value == 0.0)
            #expect(plane19Value == 0.0)
        }
    }
}

@Test func testBoardStatePlanes18And19WithStones() async throws {
    // Stones should be marked as owned by their color
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .white)
    
    // Test with black as nextPlayer
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Black stone at (3,3) should be in plane 18 (own area)
    let blackStonePlane18 = boardState.spatial[[0, 18, 3, 3]].floatValue
    #expect(blackStonePlane18 == 1.0)
    
    // White stone at (5,5) should be in plane 19 (opponent area)
    let whiteStonePlane19 = boardState.spatial[[0, 19, 5, 5]].floatValue
    #expect(whiteStonePlane19 == 1.0)
    
    // Black stone should not be in plane 19
    let blackStonePlane19 = boardState.spatial[[0, 19, 3, 3]].floatValue
    #expect(blackStonePlane19 == 0.0)
    
    // White stone should not be in plane 18
    let whiteStonePlane18 = boardState.spatial[[0, 18, 5, 5]].floatValue
    #expect(whiteStonePlane18 == 0.0)
}

@Test func testBoardStatePlanes18And19PerspectiveSwitching() async throws {
    // Test perspective switching: when nextPlayer changes, planes 18 and 19 should swap
    let board = Board()
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    
    // Test with black as nextPlayer
    let boardStateBlack = BoardState(board: board, nextPlayer: .black)
    #expect(boardStateBlack.spatial[[0, 18, 2, 2]].floatValue == 1.0) // Black stone in plane 18
    #expect(boardStateBlack.spatial[[0, 19, 3, 3]].floatValue == 1.0) // White stone in plane 19
    
    // Test with white as nextPlayer
    let boardStateWhite = BoardState(board: board, nextPlayer: .white)
    #expect(boardStateWhite.spatial[[0, 18, 3, 3]].floatValue == 1.0) // White stone in plane 18
    #expect(boardStateWhite.spatial[[0, 19, 2, 2]].floatValue == 1.0) // Black stone in plane 19
}

@Test func testBoardStatePlanes18And19SurroundedTerritory() async throws {
    // Create a small surrounded territory in the corner
    // Black surrounds corner at (0,0)
    let board = Board()
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .black)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // The surrounded empty space at (0,0) should be in plane 18 (black's area)
    let territoryPlane18 = boardState.spatial[[0, 18, 0, 0]].floatValue
    #expect(territoryPlane18 == 1.0)
    
    // Black stones should also be in plane 18
    #expect(boardState.spatial[[0, 18, 0, 1]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 18, 1, 0]].floatValue == 1.0)
}

@Test func testBoardStatePlanes18And19WhiteSurroundedTerritory() async throws {
    // White surrounds corner at (18,18)
    let board = Board()
    _ = board.playMove(at: Point(x: 17, y: 18), stone: .white)
    _ = board.playMove(at: Point(x: 18, y: 17), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // The surrounded empty space at (18,18) should be in plane 18 (white's area from white's perspective)
    let territoryPlane18 = boardState.spatial[[0, 18, 18, 18]].floatValue
    #expect(territoryPlane18 == 1.0)
    
    // White stones should also be in plane 18
    #expect(boardState.spatial[[0, 18, 18, 17]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 18, 17, 18]].floatValue == 1.0)
}

@Test func testBoardStatePlanes18And19MixedScenario() async throws {
    // Create a mixed scenario with both stones and territory
    let board = Board()
    // Black creates territory in top-left
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .black)
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .black)
    // White creates territory in bottom-right
    _ = board.playMove(at: Point(x: 17, y: 18), stone: .white)
    _ = board.playMove(at: Point(x: 18, y: 17), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Black's area (plane 18)
    #expect(boardState.spatial[[0, 18, 0, 0]].floatValue == 1.0) // Territory
    #expect(boardState.spatial[[0, 18, 0, 1]].floatValue == 1.0) // Stone
    #expect(boardState.spatial[[0, 18, 1, 0]].floatValue == 1.0) // Stone
    
    // White's area (plane 19)
    #expect(boardState.spatial[[0, 19, 18, 18]].floatValue == 1.0) // Territory
    #expect(boardState.spatial[[0, 19, 18, 17]].floatValue == 1.0) // Stone
    #expect(boardState.spatial[[0, 19, 17, 18]].floatValue == 1.0) // Stone
}

@Test func testBoardStatePlanes18And19MultipleStones() async throws {
    // Test with multiple stones of both colors
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .black)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .white)
    _ = board.playMove(at: Point(x: 6, y: 6), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // All black stones should be in plane 18
    #expect(boardState.spatial[[0, 18, 3, 3]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 18, 4, 4]].floatValue == 1.0)
    
    // All white stones should be in plane 19
    #expect(boardState.spatial[[0, 19, 5, 5]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 19, 6, 6]].floatValue == 1.0)
}

@Test func testBoardStatePlanes18And19LargeTerritory() async throws {
    // Test with a larger territory surrounded by one color
    let board = Board()
    // Create a 3x3 territory surrounded by black
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 1, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 1, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // The center empty space at (2,2) should be in plane 18 (black's territory)
    let territoryPlane18 = boardState.spatial[[0, 18, 2, 2]].floatValue
    #expect(territoryPlane18 == 1.0)
}

@Test func testBoardStatePlanes18And19WithOpponentStonesInRegion() async throws {
    // Test that opponent stones are correctly marked by their own color, not as territory
    let board = Board()
    // Create a simple scenario with both colors
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Black stone should be in plane 18 (own area from black's perspective)
    #expect(boardState.spatial[[0, 18, 1, 1]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 19, 1, 1]].floatValue == 0.0)
    
    // White stone should be in plane 19 (opponent area from black's perspective)
    #expect(boardState.spatial[[0, 19, 2, 2]].floatValue == 1.0)
    #expect(boardState.spatial[[0, 18, 2, 2]].floatValue == 0.0)
}

// MARK: - Planes 9-13 (Move History) Tests

@Test func testBoardStatePlanes9To13EmptyBoard() async throws {
    // Empty board should have all zeros on planes 9-13
    let board = Board()
    let boardState = BoardState(board: board)
    
    // Verify planes 9-13 are all zeros
    for plane in 9...13 {
        for y in 0..<19 {
            for x in 0..<19 {
                let value = boardState.spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]].floatValue
                #expect(value == 0.0)
            }
        }
    }
    
    // Verify global features 0-4 (pass history) are also zeros
    for i in 0..<5 {
        #expect(boardState.global[i].floatValue == 0.0)
    }
}

@Test func testBoardStatePlane9SingleMove() async throws {
    // Single move should appear in plane 9 (most recent move by opponent)
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // Opponent move (if black is nextPlayer)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Plane 9 should have 1.0 at (3,3)
    let plane9Value = boardState.spatial[[0, 9, 3, 3]].floatValue
    #expect(plane9Value == 1.0)
    
    // Other positions on plane 9 should be 0.0
    let otherValue = boardState.spatial[[0, 9, 5, 5]].floatValue
    #expect(otherValue == 0.0)
    
    // Planes 10-13 should be all zeros
    for plane in 10...13 {
        let value = boardState.spatial[[0, NSNumber(value: plane), 3, 3]].floatValue
        #expect(value == 0.0)
    }
}

@Test func testBoardStatePlanes9To13MultipleMoves() async throws {
    // Multiple moves should appear in correct planes
    let board = Board()
    // Move sequence: white, black, white, black, white (from black's perspective: opp, pla, opp, pla, opp)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // Move 5 ago (opp)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .black)  // Move 4 ago (pla)
    _ = board.playMove(at: Point(x: 7, y: 7), stone: .white)  // Move 3 ago (opp)
    _ = board.playMove(at: Point(x: 9, y: 9), stone: .black)  // Move 2 ago (pla)
    _ = board.playMove(at: Point(x: 11, y: 11), stone: .white)  // Move 1 ago (opp)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Plane 9: Move 1 ago (opponent at 11,11)
    #expect(boardState.spatial[[0, 9, 11, 11]].floatValue == 1.0)
    
    // Plane 10: Move 2 ago (player at 9,9)
    #expect(boardState.spatial[[0, 10, 9, 9]].floatValue == 1.0)
    
    // Plane 11: Move 3 ago (opponent at 7,7)
    #expect(boardState.spatial[[0, 11, 7, 7]].floatValue == 1.0)
    
    // Plane 12: Move 4 ago (player at 5,5)
    #expect(boardState.spatial[[0, 12, 5, 5]].floatValue == 1.0)
    
    // Plane 13: Move 5 ago (opponent at 3,3)
    #expect(boardState.spatial[[0, 13, 3, 3]].floatValue == 1.0)
    
    // Verify other positions are zeros
    #expect(boardState.spatial[[0, 9, 3, 3]].floatValue == 0.0)  // Old move not in plane 9
    #expect(boardState.spatial[[0, 10, 11, 11]].floatValue == 0.0)  // Wrong plane
}

@Test func testBoardStatePlanes9To13LessThan5Moves() async throws {
    // If fewer than 5 moves, only some planes should be filled
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // Move 2 ago (opp)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .black)  // Move 1 ago (pla)
    
    let boardState = BoardState(board: board, nextPlayer: .white)  // White's perspective
    
    // From white's perspective: black is opp, white is pla
    // Move 1 ago: black (opp) at (5,5) → Plane 9
    #expect(boardState.spatial[[0, 9, 5, 5]].floatValue == 1.0)
    
    // Move 2 ago: white (pla) at (3,3) → Plane 10
    #expect(boardState.spatial[[0, 10, 3, 3]].floatValue == 1.0)
    
    // Planes 11-13 should be zeros (not enough history)
    for plane in 11...13 {
        let value = boardState.spatial[[0, NSNumber(value: plane), 3, 3]].floatValue
        #expect(value == 0.0)
    }
}

@Test func testBoardStatePlanes9To13PassMoves() async throws {
    // Pass moves should set global features, not spatial planes
    let board = Board()
    _ = board.playPass(stone: .white)  // Pass 1 ago (opponent from black's perspective)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Global[0] should be 1.0 (pass 1 ago)
    #expect(boardState.global[0].floatValue == 1.0)
    
    // Plane 9 should be all zeros (pass doesn't go in spatial plane)
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 9, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
    
    // Other global features should be zeros
    for i in 1..<5 {
        #expect(boardState.global[i].floatValue == 0.0)
    }
}

@Test func testBoardStatePlanes9To13MultiplePasses() async throws {
    // Multiple passes should set corresponding global features
    let board = Board()
    _ = board.playPass(stone: .white)  // Pass 5 ago (opp)
    _ = board.playPass(stone: .black)  // Pass 4 ago (pla)
    _ = board.playPass(stone: .white)  // Pass 3 ago (opp)
    _ = board.playPass(stone: .black)  // Pass 2 ago (pla)
    _ = board.playPass(stone: .white)  // Pass 1 ago (opp)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // All global features 0-4 should be 1.0
    for i in 0..<5 {
        #expect(boardState.global[i].floatValue == 1.0)
    }
    
    // All spatial planes 9-13 should be zeros
    for plane in 9...13 {
        for y in 0..<19 {
            for x in 0..<19 {
                let value = boardState.spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]].floatValue
                #expect(value == 0.0)
            }
        }
    }
}

@Test func testBoardStatePlanes9To13MixedMovesAndPasses() async throws {
    // Mixed moves and passes
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // Move 5 ago (opp)
    _ = board.playPass(stone: .black)  // Pass 4 ago (pla)
    _ = board.playMove(at: Point(x: 7, y: 7), stone: .white)  // Move 3 ago (opp)
    _ = board.playPass(stone: .black)  // Pass 2 ago (pla)
    _ = board.playMove(at: Point(x: 11, y: 11), stone: .white)  // Move 1 ago (opp)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Plane 9: Move 1 ago at (11,11)
    #expect(boardState.spatial[[0, 9, 11, 11]].floatValue == 1.0)
    
    // Global[1]: Pass 2 ago
    #expect(boardState.global[1].floatValue == 1.0)
    
    // Plane 11: Move 3 ago at (7,7)
    #expect(boardState.spatial[[0, 11, 7, 7]].floatValue == 1.0)
    
    // Global[3]: Pass 4 ago
    #expect(boardState.global[3].floatValue == 1.0)
    
    // Plane 13: Move 5 ago at (3,3)
    #expect(boardState.spatial[[0, 13, 3, 3]].floatValue == 1.0)
    
    // Global[0] should be 0 (move, not pass)
    #expect(boardState.global[0].floatValue == 0.0)
    // Global[2] should be 0 (move, not pass)
    #expect(boardState.global[2].floatValue == 0.0)
    // Global[4] should be 0 (move, not pass)
    #expect(boardState.global[4].floatValue == 0.0)
    
    // Plane 10 should be 0 (pass, not move)
    #expect(boardState.spatial[[0, 10, 5, 5]].floatValue == 0.0)
    // Plane 12 should be 0 (pass, not move)
    #expect(boardState.spatial[[0, 12, 5, 5]].floatValue == 0.0)
}

@Test func testBoardStatePlanes9To13HistoryAlternation() async throws {
    // Test that history correctly alternates opp/pla/opp/pla/opp
    let board = Board()
    // Create sequence: white, black, white, black, white
    // From black's perspective: opp, pla, opp, pla, opp
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .white)  // Move 5: opp
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)  // Move 4: pla
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // Move 3: opp
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .black)  // Move 2: pla
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .white)  // Move 1: opp
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Verify correct alternation
    // Plane 9 (move 1): white (opp) at (5,5)
    #expect(boardState.spatial[[0, 9, 5, 5]].floatValue == 1.0)
    
    // Plane 10 (move 2): black (pla) at (4,4)
    #expect(boardState.spatial[[0, 10, 4, 4]].floatValue == 1.0)
    
    // Plane 11 (move 3): white (opp) at (3,3)
    #expect(boardState.spatial[[0, 11, 3, 3]].floatValue == 1.0)
    
    // Plane 12 (move 4): black (pla) at (2,2)
    #expect(boardState.spatial[[0, 12, 2, 2]].floatValue == 1.0)
    
    // Plane 13 (move 5): white (opp) at (1,1)
    #expect(boardState.spatial[[0, 13, 1, 1]].floatValue == 1.0)
}

@Test func testBoardStatePlanes9To13PerspectiveSwitching() async throws {
    // Test that perspective switching works correctly
    let board = Board()
    // Create sequence: black, white, black, white, black
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .white)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .black)
    
    // Test with black as nextPlayer
    let boardStateBlack = BoardState(board: board, nextPlayer: .black)
    // From black's perspective: white is opp, black is pla
    // Move 1 ago: black (pla) → should NOT be in plane 9 (plane 9 is for opp)
    // Actually wait, let me check the algorithm again...
    // The algorithm checks: moveHistory[moveHistoryLen-1].pla == opp
    // So plane 9 is only set if the most recent move was by the opponent
    // In this case, move 1 ago is black (pla), so plane 9 should be 0
    // Move 2 ago is white (opp), but that's not checked because move 1 ago wasn't opp
    
    // Actually, I need to reconsider. The algorithm is:
    // if moveHistory[len-1].pla == opp: set plane 9
    //   if moveHistory[len-2].pla == pla: set plane 10
    //     ...
    // So it's nested - if the first condition fails, nothing is set
    
    // In this case: move 1 ago is black (pla), so the first condition fails
    // All planes 9-13 should be 0
    for plane in 9...13 {
        let value = boardStateBlack.spatial[[0, NSNumber(value: plane), 5, 5]].floatValue
        #expect(value == 0.0)
    }
    
    // Test with white as nextPlayer
    let boardStateWhite = BoardState(board: board, nextPlayer: .white)
    // From white's perspective: black is opp, white is pla
    // Move 1 ago: black (opp) → Plane 9 at (5,5)
    #expect(boardStateWhite.spatial[[0, 9, 5, 5]].floatValue == 1.0)
    
    // Move 2 ago: white (pla) → Plane 10 at (4,4)
    #expect(boardStateWhite.spatial[[0, 10, 4, 4]].floatValue == 1.0)
    
    // Move 3 ago: black (opp) → Plane 11 at (3,3)
    #expect(boardStateWhite.spatial[[0, 11, 3, 3]].floatValue == 1.0)
    
    // Move 4 ago: white (pla) → Plane 12 at (2,2)
    #expect(boardStateWhite.spatial[[0, 12, 2, 2]].floatValue == 1.0)
    
    // Move 5 ago: black (opp) → Plane 13 at (1,1)
    #expect(boardStateWhite.spatial[[0, 13, 1, 1]].floatValue == 1.0)
}

@Test func testBoardStatePlanes9To13WrongPlayerSequence() async throws {
    // Test that if player sequence doesn't match expected alternation, planes aren't set
    let board = Board()
    // Sequence: black, black (same player twice) - shouldn't match expected pattern
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 5, y: 5), stone: .black)  // Same player, not opp
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    // From white's perspective, we expect: move 1 ago should be opp (black), move 2 ago should be pla (white)
    // But move 2 ago is also black (opp), so plane 10 shouldn't be set
    
    // Move 1 ago: black (opp) → Plane 9 should be set
    #expect(boardState.spatial[[0, 9, 5, 5]].floatValue == 1.0)
    
    // Move 2 ago: black (opp, not pla) → Plane 10 should NOT be set
    #expect(boardState.spatial[[0, 10, 3, 3]].floatValue == 0.0)
}

// MARK: - Planes 14-17 (Ladder Features) Tests

@Test func testBoardStatePlane14EmptyBoard() async throws {
    let board = Board()
    let boardState = BoardState(board: board)
    
    // Verify plane 14 is all zeros on empty board
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 14, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane14NoLadders() async throws {
    let board = Board()
    // Place stones with many liberties (not in ladder)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 10, y: 10), stone: .white)
    
    let boardState = BoardState(board: board)
    
    // Plane 14 should be all zeros (no ladders detected)
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 14, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane14SimpleLadder() async throws {
    let board = Board()
    // Create a simple ladder situation: stone with 1 liberty
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    // Black stone at (3,3) now has 1 liberty
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // Plane 14 may or may not be set depending on ladder detection
    // Just verify it doesn't crash
    _ = boardState.spatial[[0, 14, 3, 3]].floatValue
}

@Test func testBoardStatePlane15NoHistory() async throws {
    let board = Board()
    // Empty board or board with no history
    let boardState = BoardState(board: board)
    
    // Plane 15 should be all zeros when no history
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 15, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane15WithHistory() async throws {
    let board = Board()
    // Create a ladder situation, then make another move
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    // Now make another move
    _ = board.playMove(at: Point(x: 10, y: 10), stone: .black)
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // Plane 15 should reflect previous board state
    // Just verify it doesn't crash
    _ = boardState.spatial[[0, 15, 3, 3]].floatValue
}

@Test func testBoardStatePlane16NoHistory() async throws {
    let board = Board()
    // Board with < 2 moves
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    
    let boardState = BoardState(board: board)
    
    // Plane 16 should be all zeros when insufficient history
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 16, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane16WithHistory() async throws {
    let board = Board()
    // Create ladder, then make 2 more moves
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 10, y: 10), stone: .black)
    _ = board.playMove(at: Point(x: 11, y: 11), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Plane 16 should reflect board 2 turns ago
    // Just verify it doesn't crash
    _ = boardState.spatial[[0, 16, 3, 3]].floatValue
}

@Test func testBoardStatePlane17EmptyBoard() async throws {
    let board = Board()
    let boardState = BoardState(board: board)
    
    // Verify plane 17 is all zeros on empty board
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 17, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStatePlane17NoWorkingMoves() async throws {
    let board = Board()
    // Stone with 1 liberty (no working moves for 1-liberty)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // Plane 17 should be all zeros (1-liberty stones don't have working moves)
    for y in 0..<19 {
        for x in 0..<19 {
            let value = boardState.spatial[[0, 17, NSNumber(value: y), NSNumber(value: x)]].floatValue
            #expect(value == 0.0)
        }
    }
}

@Test func testBoardStateLadderFeaturesAllZero() async throws {
    let board = Board()
    // Board with stones but no ladders
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 10, y: 10), stone: .white)
    
    let boardState = BoardState(board: board)
    
    // All ladder features should be zero
    for plane in 14...17 {
        for y in 0..<19 {
            for x in 0..<19 {
                let value = boardState.spatial[[0, NSNumber(value: plane), NSNumber(value: y), NSNumber(value: x)]].floatValue
                #expect(value == 0.0)
            }
        }
    }
}

@Test func testBoardStateLadderFeaturesComplete() async throws {
    let board = Board()
    // Create a ladder situation
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .white)
    
    // Verify all four features exist (may or may not have values depending on detection)
    _ = boardState.spatial[[0, 14, 3, 3]].floatValue
    _ = boardState.spatial[[0, 15, 3, 3]].floatValue
    _ = boardState.spatial[[0, 16, 3, 3]].floatValue
    _ = boardState.spatial[[0, 17, 3, 3]].floatValue
}

@Test func testBoardStateLadderFeaturesInsufficientHistory() async throws {
    let board = Board()
    // Board with only 1 move
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    
    let boardState = BoardState(board: board)
    
    // Features 15-16 should handle insufficient history gracefully
    // Just verify it doesn't crash
    _ = boardState.spatial[[0, 15, 3, 3]].floatValue
    _ = boardState.spatial[[0, 16, 3, 3]].floatValue
}

@Test func testBoardStateLadderFeaturesPassMoves() async throws {
    let board = Board()
    // Create ladder, then pass
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    _ = board.playPass(stone: .black)
    _ = board.playPass(stone: .white)
    
    let boardState = BoardState(board: board, nextPlayer: .black)
    
    // Should handle pass moves in history correctly
    // Just verify it doesn't crash
    _ = boardState.spatial[[0, 14, 3, 3]].floatValue
    _ = boardState.spatial[[0, 15, 3, 3]].floatValue
}

@Test func testBoardStateLadderFeaturesPerspective() async throws {
    let board = Board()
    // Create ladder with black stone
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    
    let boardStateBlack = BoardState(board: board, nextPlayer: .black)
    let boardStateWhite = BoardState(board: board, nextPlayer: .white)
    
    // Features should work with different perspectives
    // Just verify it doesn't crash
    _ = boardStateBlack.spatial[[0, 14, 3, 3]].floatValue
    _ = boardStateWhite.spatial[[0, 14, 3, 3]].floatValue
}

// MARK: - Global Feature 14: Pass Ends Phase Tests

@Test func testGlobalFeature14EmptyBoard() async throws {
    // Empty board: no passes, pass wouldn't end phase
    let board = Board()
    let boardState = BoardState(board: board, nextPlayer: .black)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 0.0)
}

@Test func testGlobalFeature14OnePass() async throws {
    // One pass: if we pass now, that makes 2 consecutive passes, which ends phase
    let board = Board()
    _ = board.playPass(stone: .black)
    let boardState = BoardState(board: board, nextPlayer: .white)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0) // 1 pass + 1 pass = 2, which ends phase
}

@Test func testGlobalFeature14TwoConsecutivePasses() async throws {
    // Two consecutive passes: should end phase for simple ko (Chinese rules)
    let board = Board()
    _ = board.playPass(stone: .black)
    _ = board.playPass(stone: .white)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14PassAfterRegularMove() async throws {
    // Pass after regular move: last move was a pass, so if we pass now that's 2 consecutive passes
    let board = Board()
    _ = board.playPass(stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)  // This breaks consecutive passes
    _ = board.playPass(stone: .black)  // This is 1 consecutive pass
    let boardState = BoardState(board: board, nextPlayer: .white)
    let feature14 = boardState.global[14].floatValue
    // Last move was a pass, so if white passes, that's 2 consecutive passes, ending phase
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14ThreeConsecutivePasses() async throws {
    // Three consecutive passes: should end phase (>= 2)
    let board = Board()
    _ = board.playPass(stone: .black)
    _ = board.playPass(stone: .white)
    _ = board.playPass(stone: .black)
    let boardState = BoardState(board: board, nextPlayer: .white)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14BlackPerspective() async throws {
    // Test from black's perspective
    let board = Board()
    _ = board.playPass(stone: .white)
    _ = board.playPass(stone: .black)
    let boardState = BoardState(board: board, nextPlayer: .white)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14WhitePerspective() async throws {
    // Test from white's perspective
    let board = Board()
    _ = board.playPass(stone: .black)
    _ = board.playPass(stone: .white)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14MovesThenPasses() async throws {
    // Regular moves, then two passes
    let board = Board()
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .white)
    _ = board.playPass(stone: .black)
    _ = board.playPass(stone: .white)
    let boardState = BoardState(board: board, nextPlayer: .black)
    let feature14 = boardState.global[14].floatValue
    #expect(feature14 == 1.0)
}

@Test func testGlobalFeature14SpightStyleEnding() async throws {
    // Spight-style ending: same player passes in same ko situation
    // This tests ko hash matching
    // For spight-style ending to work, we need the same ko situation (same board state + ko point)
    // Since board state changes after moves, we'll test with two consecutive passes by same player
    // which should trigger spight-style ending if ko hash matches
    let board = Board()
    // Create a ko situation
    _ = board.playMove(at: Point(x: 1, y: 1), stone: .black)
    _ = board.playMove(at: Point(x: 0, y: 1), stone: .white)
    _ = board.playMove(at: Point(x: 2, y: 1), stone: .white)
    _ = board.playMove(at: Point(x: 1, y: 0), stone: .white)
    // Capture creates ko
    _ = board.playMove(at: Point(x: 1, y: 2), stone: .black)
    
    // First pass by black in this ko situation
    _ = board.playPass(stone: .black)
    
    // White also passes (maintains ko situation)
    _ = board.playPass(stone: .white)
    
    // Black passes again - same ko situation, ko hash should match
    // This should trigger spight-style ending
    let boardState = BoardState(board: board, nextPlayer: .black)
    let feature14 = boardState.global[14].floatValue
    // Should be 1.0 because either: 2 consecutive passes OR spight-style ending
    #expect(feature14 == 1.0)
}

