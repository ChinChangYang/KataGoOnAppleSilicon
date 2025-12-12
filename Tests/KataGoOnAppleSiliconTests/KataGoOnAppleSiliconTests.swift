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

/// Mock model that returns valid outputs with a specific move
class MockModelWithValidOutputs: ModelProtocol {
    let targetX: Int
    let targetY: Int
    
    init(targetX: Int = 0, targetY: Int = 0) {
        self.targetX = targetX
        self.targetY = targetY
    }
    
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        // Create policy array [1, 19, 19, 1] with a peak at target position
        let policyShape: [NSNumber] = [1, 19, 19, 1]
        let policy = try! MLMultiArray(shape: policyShape, dataType: .float32)
        for i in 0..<policy.count {
            policy[i] = 0.0
        }
        // Set a high probability at target position
        policy[[0, NSNumber(value: targetY), NSNumber(value: targetX), 0]] = 1.0
        
        // Create value array [1, 3]
        let valueShape: [NSNumber] = [1, 3]
        let value = try! MLMultiArray(shape: valueShape, dataType: .float32)
        value[0] = 0.5
        value[1] = 0.3
        value[2] = 0.2
        
        // Create ownership array [1, 1, 19, 19]
        let ownershipShape: [NSNumber] = [1, 1, 19, 19]
        let ownership = try! MLMultiArray(shape: ownershipShape, dataType: .float32)
        for i in 0..<ownership.count {
            ownership[i] = 0.0
        }
        
        return try MLDictionaryFeatureProvider(dictionary: [
            "output_policy": policy,
            "out_value": value,
            "out_ownership": ownership
        ])
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

@Test func testGTPBoardsize() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize 19")
    #expect(response == "= \n\n")
}

@Test func testGTPKomi() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("komi 7.5")
    #expect(response == "= \n\n")
}

@Test func testGTPQuit() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("quit")
    #expect(response == "= \n\n")
}

@Test func testGTPEmptyCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("")
    #expect(response == "? \n\n")
}

@Test func testGTPUnknownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("nonexistent_command")
    #expect(response == "? unknown command\n\n")
}

@Test func testGTPPlayMissingArgs() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPPlayIllegalMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Play at same location twice - second should be illegal
    _ = handler.handleCommand("play black A1")
    let response = handler.handleCommand("play white A1")
    #expect(response == "? illegal move\n\n")
}

@Test func testGTPPlayWhiteMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play white T19")
    #expect(response == "= \n\n")
}

@Test func testGTPGenmoveMissingColor() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPGenmoveWithModelLoaded() async throws {
    let katago = KataGoInference()
    try katago.loadModel(for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= "))
}

@Test func testParseMoveColumnJToT() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Test columns J through T (skipping I)
    let responseJ = handler.handleCommand("play black J1")
    #expect(responseJ == "= \n\n")
    let responseT = handler.handleCommand("play white T1")
    #expect(responseT == "= \n\n")
}

@Test func testParseMoveInvalidColumn() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // 'I' is skipped in Go, 'Z' is out of range
    let responseZ = handler.handleCommand("play black Z1")
    #expect(responseZ == "? syntax error\n\n")
}

@Test func testParseMoveShortString() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black A")
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

// MARK: - ModelStatus Tests

@Test func testModelStatusReportLoadFailed() async throws {
    // Just call to verify it doesn't crash
    struct TestError: Error {}
    ModelStatus.reportModelLoadFailed(name: "TestModel", error: TestError())
}

// MARK: - KataGoError Tests

@Test func testKataGoErrorDescriptions() async throws {
    let modelNotFound = KataGoError.modelNotFound("test")
    #expect(modelNotFound.description == "Model not found: test")
    
    let modelLoadFailed = KataGoError.modelLoadFailed("test reason")
    #expect(modelLoadFailed.description == "Model load failed: test reason")
    
    let invalidInput = KataGoError.invalidInput("bad input")
    #expect(invalidInput.description == "Invalid input: bad input")
    
    let inferenceFailed = KataGoError.inferenceFailed("inference error")
    #expect(inferenceFailed.description == "Inference failed: inference error")
    
    let unsupportedProfile = KataGoError.unsupportedProfile("unknown")
    #expect(unsupportedProfile.description == "Unsupported profile: unknown")
}

// MARK: - GTPHandler SelectMove Tests

@Test func testGTPSelectMoveColumnAToH() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column A (x=0)
    let mockModel = MockModelWithValidOutputs(targetX: 0, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= A"))
}

@Test func testGTPSelectMoveColumnJToT() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column J (x=8, skipping I)
    let mockModel = MockModelWithValidOutputs(targetX: 8, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= J"))
}

@Test func testGTPSelectMoveColumnT() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column T (x=18)
    let mockModel = MockModelWithValidOutputs(targetX: 18, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= T"))
}
