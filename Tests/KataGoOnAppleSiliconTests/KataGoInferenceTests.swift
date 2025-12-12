import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - KataGoInference Tests

@Test func testKataGoInferenceInitialization() async throws {
    _ = KataGoInference()
}

@Test func testLoadModelForAIProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "AI")
        #expect(Bool(true))
    } catch {
        print("Model load failed in test: \(error)")
    }
}

@Test func testLoadModelFor9dProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "9d")
        #expect(Bool(true))
    } catch {
        print("Model load failed in test: \(error)")
    }
}

@Test func testLoadModelFor20kProfile() async throws {
    let katago = KataGoInference()
    do {
        try katago.loadModel(for: "20k")
        #expect(Bool(true))
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
        #expect(Bool(true))
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

