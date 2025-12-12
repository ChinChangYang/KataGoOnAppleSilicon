import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

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
        #expect(Bool(true))
    }
}

