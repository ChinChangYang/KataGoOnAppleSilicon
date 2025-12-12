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

