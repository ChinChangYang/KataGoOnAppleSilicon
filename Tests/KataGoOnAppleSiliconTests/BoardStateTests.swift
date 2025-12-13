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

