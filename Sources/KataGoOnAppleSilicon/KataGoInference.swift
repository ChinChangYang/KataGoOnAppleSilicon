import CoreML
import Foundation

/// Protocol for model inference - enables mocking in tests
public protocol ModelProtocol {
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider
}

/// Make MLModel conform to ModelProtocol
extension MLModel: ModelProtocol {}

/// Main class for KataGo inference
public class KataGoInference {
    private let modelLoader = ModelLoader()
    private var models: [String: any ModelProtocol] = [:]
    
    public init() {}
    
    /// Inject a model for testing purposes
    internal func setModel(_ model: any ModelProtocol, for profile: String) {
        models[profile] = model
    }
    
    /// Load a model for a specific profile
    public func loadModel(for profile: String) throws {
        let modelName: String
        switch profile {
        case "AI":
            modelName = "KataGoModel19x19fp16-adam-s11165M"  // Strongest 28b model
        case "9d", "20k":
            modelName = "KataGoModel19x19fp16m1"  // Human SL model
        default:
            throw KataGoError.unsupportedProfile(profile)
        }
        
        let model = try modelLoader.loadModel(name: modelName)
        models[profile] = model
    }
    
    /// Perform inference on the given board state
    public func predict(board: BoardState, profile: String) throws -> ModelOutput {
        guard let model = models[profile] else {
            throw KataGoError.modelNotFound("Model for profile \(profile) not loaded")
        }
        
        let startTime = Date()
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_spatial": board.spatial,
                "input_global": board.global
            ])
            let prediction = try model.prediction(from: input)
            
            // Extract outputs - model uses output_policy, out_value, out_ownership
            guard let policy = prediction.featureValue(for: "output_policy")?.multiArrayValue,
                  let valueArray = prediction.featureValue(for: "out_value")?.multiArrayValue,
                  let ownership = prediction.featureValue(for: "out_ownership")?.multiArrayValue else {
                throw KataGoError.inferenceFailed("Invalid model outputs")
            }
            
            // Extract value from array [1, 3] - index 0 is typically the win probability
            let valueNum = valueArray[0].floatValue
            let output = ModelOutput(policy: policy, value: valueNum, ownership: ownership)
            
            let inferenceTime = Date().timeIntervalSince(startTime)
            ModelStatus.reportInferenceCompleted(time: inferenceTime, policyCount: Int(policy.count), value: output.value)
            
            return output
        } catch let kataError as KataGoError {
            // Re-throw KataGoError without wrapping
            ModelStatus.reportInferenceFailed(error: kataError)
            throw kataError
        } catch {
            // Wrap other errors in KataGoError
            ModelStatus.reportInferenceFailed(error: error)
            throw KataGoError.inferenceFailed(error.localizedDescription)
        }
    }
}