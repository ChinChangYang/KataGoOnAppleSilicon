import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - ModelStatus Tests

@Test func testModelStatusReportLoadFailed() async throws {
    // Just call to verify it doesn't crash
    struct TestError: Error {}
    ModelStatus.reportModelLoadFailed(name: "TestModel", error: TestError())
}

@Test func testModelStatusReportModelLoaded() async throws {
    // Just call to verify it doesn't crash
    ModelStatus.reportModelLoaded(name: "TestModel", time: 1.23)
}

@Test func testModelStatusReportInferenceCompleted() async throws {
    // Just call to verify it doesn't crash
    ModelStatus.reportInferenceCompleted(time: 0.45, policyCount: 361, value: 0.5)
}

@Test func testModelStatusReportInferenceFailed() async throws {
    // Just call to verify it doesn't crash
    struct TestError: Error {}
    ModelStatus.reportInferenceFailed(error: TestError())
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

// MARK: - Move Tests

@Test func testMoveInit() async throws {
    let point = Point(x: 3, y: 3)
    let move = Move(location: point, player: .black)
    #expect(move.location == point)
    #expect(move.player == .black)
}

@Test func testMoveStaticMove() async throws {
    let point = Point(x: 5, y: 5)
    let move = Move.move(at: point, player: .white)
    #expect(move.location == point)
    #expect(move.player == .white)
    #expect(!move.isPass)
}

@Test func testMoveStaticPass() async throws {
    let move = Move.pass(player: .black)
    #expect(move.location == nil)
    #expect(move.player == .black)
    #expect(move.isPass)
}

@Test func testMoveIsPassProperty() async throws {
    let regularMove = Move.move(at: Point(x: 1, y: 1), player: .white)
    #expect(!regularMove.isPass)
    
    let passMove = Move.pass(player: .black)
    #expect(passMove.isPass)
    
    let nilLocationMove = Move(location: nil, player: .white)
    #expect(nilLocationMove.isPass)
}

