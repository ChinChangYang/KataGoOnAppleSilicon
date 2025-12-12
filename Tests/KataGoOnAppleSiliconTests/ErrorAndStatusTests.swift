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

