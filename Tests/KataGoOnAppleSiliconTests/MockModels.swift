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
        // Create policy array [1, 6, 362] with a peak at target position
        // Shape matches actual model output: 6 channels, 362 positions (361 board + 1 pass)
        let policyShape: [NSNumber] = [1, 6, 362]
        let policy = try! MLMultiArray(shape: policyShape, dataType: .float32)
        for i in 0..<policy.count {
            policy[i] = 0.0
        }
        // Set a dominant logit at target position so it survives post-processing softmax.
        // Value 100.0 vs 0.0 gives exp(100)/Z ≈ 1.0 after softmax, making selection deterministic.
        // Access pattern: [batch, channel, positionIndex] where positionIndex = y * 19 + x
        let positionIndex = targetY * 19 + targetX
        policy[[0, 0, NSNumber(value: positionIndex)]] = 100.0
        
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

