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

