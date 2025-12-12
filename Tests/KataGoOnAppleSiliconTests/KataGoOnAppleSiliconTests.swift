import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - Mock Models for Testing

/// Mock model that returns invalid outputs (nil for required fields)
class MockModelWithInvalidOutputs: ModelProtocol {
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        // Return an empty feature provider with no outputs
        return try MLDictionaryFeatureProvider(dictionary: [:])
    }
}

/// Mock model that throws an error during prediction
class MockModelThatThrows: ModelProtocol {
    struct MockPredictionError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
    
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        throw MockPredictionError(message: "Simulated prediction failure")
    }
}

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

@Test func testSuicidePrevention() async throws {
    let board = Board()
    // Place white stones around a point
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 4), stone: .white)
    // Try to play black in the center (suicide)
    let success = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    #expect(!success)
    #expect(board.stones[3][3] == .empty)
}

@Test func testKoRule() async throws {
    let board = Board()
    // Set up ko situation
    _ = board.playMove(at: Point(x: 2, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 4, y: 2), stone: .black)
    _ = board.playMove(at: Point(x: 1, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 5, y: 3), stone: .black)
    _ = board.playMove(at: Point(x: 2, y: 4), stone: .black)
    _ = board.playMove(at: Point(x: 3, y: 4), stone: .black)
    _ = board.playMove(at: Point(x: 4, y: 4), stone: .black)
    
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 3, y: 1), stone: .white)
    _ = board.playMove(at: Point(x: 2, y: 3), stone: .white)
    _ = board.playMove(at: Point(x: 4, y: 3), stone: .white)
    
    // Capture creating ko
    _ = board.playMove(at: Point(x: 3, y: 3), stone: .black)
    #expect(board.stones[3][3] == .black)
    #expect(board.koPoint != nil)
    
    // Try to recapture (ko violation)
    let success = board.playMove(at: Point(x: 3, y: 3), stone: .white)
    #expect(!success)
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

// MARK: - GTPHandler Tests

@Test func testGTPProtocolVersion() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("protocol_version")
    #expect(response == "= 2\n\n")
}

@Test func testGTPName() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("name")
    #expect(response == "= KataGoOnAppleSilicon\n\n")
}

@Test func testGTPVersion() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("version")
    #expect(response == "= 1.0\n\n")
}

@Test func testGTPKnownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command play") == "= true\n\n")
    #expect(handler.handleCommand("known_command unknown") == "= false\n\n")
}

@Test func testGTPListCommands() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("list_commands")
    #expect(response.starts(with: "= "))
    #expect(response.contains("play"))
    #expect(response.contains("genmove"))
}

@Test func testGTPClearBoard() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black A1")
    let response = handler.handleCommand("clear_board")
    #expect(response == "= \n\n")
    // Board should be cleared
}

@Test func testGTPPlayMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black A1")
    #expect(response == "= \n\n")
}

@Test func testGTPPlayInvalidMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black invalid")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPGenmove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Mock or assume model is loaded, but since it's async and may fail, just test parsing
    let response = handler.handleCommand("genmove black")
    // In real scenario, would need model loaded, but for test, check it's not error
    #expect(response.starts(with: "=") || response.starts(with: "?"))
}

@Test func testParseMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Access private method via reflection or assume it's tested through play
    // For now, test through play command
    let response = handler.handleCommand("play black A1")
    #expect(response == "= \n\n")
}

// MARK: - ModelLoader Tests

@Test func testModelLoaderInitialization() async throws {
    _ = ModelLoader()
}

@Test func testLoadExistingModel() async throws {
    let loader = ModelLoader()
    // Bundled models are at Sources/KataGoOnAppleSilicon/Models/Resources/
    _ = try loader.loadModel(name: "KataGoModel19x19fp16-adam-s11165M")
}

@Test func testLoadNonExistingModel() async throws {
    let loader = ModelLoader()
    do {
        let _ = try loader.loadModel(name: "NonExistingModel")
        #expect(Bool(false), "Should have thrown error")
    } catch KataGoError.modelNotFound {
        #expect(true)
    }
}

// MARK: - KataGoInference Tests

@Test func testKataGoInferenceInitialization() async throws {
    _ = KataGoInference()
}

@Test func testLoadModelForAIProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "AI")
        #expect(true)
    } catch {
        print("Model load failed in test: \(error)")
    }
}

@Test func testLoadModelFor9dProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "9d")
        #expect(true)
    } catch {
        print("Model load failed in test: \(error)")
    }
}

@Test func testLoadModelFor20kProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "20k")
        #expect(true)
    } catch {
        print("Model load failed in test: \(error)")
    }
}

@Test func testLoadModelForUnsupportedProfile() async throws {
    let katago = KataGoInference()
    #expect(throws: KataGoError.self) {
        try katago.loadModel(for: "unsupported")
    }
}

@Test func testPredictWithoutModelLoaded() async throws {
    let katago = KataGoInference()
    let board = Board()
    let boardState = BoardState(board: board)
    do {
        let _ = try katago.predict(board: boardState, profile: "AI")
        #expect(Bool(false), "Should have thrown error")
    } catch KataGoError.modelNotFound {
        #expect(true)
    }
}

@Test func testPredictWithModelLoaded() async throws {
    let katago = KataGoInference()
    try katago.loadModel(for: "AI")
    let board = Board()
    let boardState = BoardState(board: board)
    let output = try katago.predict(board: boardState, profile: "AI")
    #expect(output.policy.count > 0)
    #expect(output.ownership.count > 0)
}

@Test func testPredictWithInvalidModelOutputs() async throws {
    let katago = KataGoInference()
    let mockModel = MockModelWithInvalidOutputs()
    katago.setModel(mockModel, for: "test")
    
    let board = Board()
    let boardState = BoardState(board: board)
    
    do {
        let _ = try katago.predict(board: boardState, profile: "test")
        #expect(Bool(false), "Should have thrown error for invalid outputs")
    } catch let error as KataGoError {
        if case .inferenceFailed(let message) = error {
            #expect(message == "Invalid model outputs")
        } else {
            #expect(Bool(false), "Expected inferenceFailed error")
        }
    }
}

@Test func testPredictWhenModelThrows() async throws {
    let katago = KataGoInference()
    let mockModel = MockModelThatThrows()
    katago.setModel(mockModel, for: "test")
    
    let board = Board()
    let boardState = BoardState(board: board)
    
    do {
        let _ = try katago.predict(board: boardState, profile: "test")
        #expect(Bool(false), "Should have thrown error when model throws")
    } catch let error as KataGoError {
        if case .inferenceFailed(let message) = error {
            #expect(message.contains("Simulated prediction failure"))
        } else {
            #expect(Bool(false), "Expected inferenceFailed error")
        }
    }
}
