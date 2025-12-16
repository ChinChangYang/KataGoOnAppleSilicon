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
    /// - Parameters:
    ///   - board: BoardState with input features
    ///   - profile: Model profile to use
    ///   - nextPlayer: The player to move next (required for human SL models, defaults to .black)
    public func predict(board: BoardState, profile: String, nextPlayer: Stone = .black) throws -> ModelOutput {
        guard let model = models[profile] else {
            throw KataGoError.modelNotFound("Model for profile \(profile) not loaded")
        }
        
        let startTime = Date()
        
        do {
            // Check if model requires input_meta (human SL models)
            let modelDescription = (model as? MLModel)?.modelDescription
            let requiresInputMeta = modelDescription?.inputDescriptionsByName["input_meta"] != nil
            
            var inputDict: [String: Any] = [
                "input_spatial": board.spatial,
                "input_global": board.global
            ]
            
            // Add input_meta for human SL models (shape: [1, 192])
            if requiresInputMeta {
                // Generate SGFMetadata from profile name
                let profileName: String
                switch profile {
                case "20k":
                    profileName = "preaz_20k"
                case "9d":
                    profileName = "preaz_9d"
                default:
                    // Default to preaz_20k for human SL models
                    profileName = "preaz_20k"
                }
                
                let sgfMeta = SGFMetadata.getProfile(profileName)
                let metadataRow = SGFMetadata.fillMetadataRow(sgfMeta, nextPlayer: nextPlayer, boardArea: 361) // 19x19 = 361
                
                // Convert to MLMultiArray
                let inputMetaShape: [NSNumber] = [1, 192]
                let inputMeta = try MLMultiArray(shape: inputMetaShape, dataType: .float16)
                for i in 0..<192 {
                    inputMeta[i] = NSNumber(value: metadataRow[i])
                }
                inputDict["input_meta"] = inputMeta
            }
            
            let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let prediction = try model.prediction(from: input)
            
            // Extract outputs - model uses output_policy, out_value, out_ownership
            guard let policy = prediction.featureValue(for: "output_policy")?.multiArrayValue,
                  let valueArray = prediction.featureValue(for: "out_value")?.multiArrayValue,
                  let ownership = prediction.featureValue(for: "out_ownership")?.multiArrayValue else {
                throw KataGoError.inferenceFailed("Invalid model outputs")
            }
            
            // Extract optional misc value arrays
            let miscValueArray = prediction.featureValue(for: "out_miscvalue")?.multiArrayValue
            let moreMiscValueArray = prediction.featureValue(for: "out_moremiscvalue")?.multiArrayValue
            
            let output = ModelOutput(
                policy: policy,
                ownership: ownership,
                valueArray: valueArray,
                miscValueArray: miscValueArray,
                moreMiscValueArray: moreMiscValueArray
            )
            
            let inferenceTime = Date().timeIntervalSince(startTime)
            ModelStatus.reportInferenceCompleted(time: inferenceTime, policyCount: Int(policy.count), value: output.whiteWin)
            
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
    
    /// Generate raw neural network output in KataGo format
    /// - Parameters:
    ///   - board: Current board state
    ///   - boardState: BoardState for model input
    ///   - profile: Model profile to use
    ///   - whichSymmetry: Symmetry index (0-7) or 8 for all symmetries
    ///   - policyOptimism: Optional policy optimism value (0.0-1.0), defaults to 0.0
    ///   - useHumanModel: Whether to use human SL model (affects output format)
    /// - Returns: Formatted string matching KataGo's kata-raw-nn output
    public func rawNN(
        board: Board,
        boardState: BoardState,
        profile: String,
        whichSymmetry: Int = 0,
        policyOptimism: Float? = nil,
        useHumanModel: Bool = false
    ) throws -> String {
        // Determine next player (black moves first, so turnNumber % 2 == 0 means black)
        let nextPlayer: Stone = board.turnNumber % 2 == 0 ? .black : .white
        
        // Get model prediction
        let output = try predict(board: boardState, profile: profile, nextPlayer: nextPlayer)
        
        // Post-process model outputs
        // Use modelVersion 15 and parameters from actual model description
        // (The model description has outputScaleMultiplier = 1.0, shorttermScoreErrorMultiplier = 150.0)
        let postProcessParams = PostProcessParams(
            outputScaleMultiplier: 1.0,  // Model uses 1.0, not 8.0
            scoreMeanMultiplier: 20.0,
            scoreStdevMultiplier: 20.0,
            leadMultiplier: 20.0,
            varianceTimeMultiplier: 40.0,
            shorttermValueErrorMultiplier: 0.25,
            shorttermScoreErrorMultiplier: 150.0  // Model uses 150.0, not 30.0
        )
        let postprocessed = output.postprocess(
            board: board,
            nextPlayer: nextPlayer,
            modelVersion: 15, // Actual model version is 15, not 8
            postProcessParams: postProcessParams
        )
        
        // Format output based on model type
        var result = ""
        
        if useHumanModel {
            // Human model format
            result += "symmetry \(whichSymmetry)\n"
            result += String(format: "whiteWin %.6f\n", postprocessed.whiteWinProb)
            result += String(format: "whiteLoss %.6f\n", postprocessed.whiteLossProb)
            result += String(format: "noResult %.6f\n", postprocessed.whiteNoResultProb)
            result += String(format: "whiteScore %.3f\n", postprocessed.whiteScoreMean)
            result += String(format: "whiteScoreSq %.3f\n", postprocessed.whiteScoreMeanSq)
            result += String(format: "shorttermWinlossError %.3f\n", postprocessed.shorttermWinlossError)
            result += String(format: "shorttermScoreError %.3f\n", postprocessed.shorttermScoreError)
        } else {
            // Regular model format
            result += "symmetry \(whichSymmetry)\n"
            result += String(format: "whiteWin %.6f\n", postprocessed.whiteWinProb)
            result += String(format: "whiteLoss %.6f\n", postprocessed.whiteLossProb)
            result += String(format: "noResult %.6f\n", postprocessed.whiteNoResultProb)
            result += String(format: "whiteLead %.3f\n", postprocessed.whiteLead)
            result += String(format: "whiteScoreSelfplay %.3f\n", postprocessed.whiteScoreMean)
            result += String(format: "whiteScoreSelfplaySq %.3f\n", postprocessed.whiteScoreMeanSq)
            result += String(format: "varTimeLeft %.3f\n", postprocessed.varTimeLeft)
            result += String(format: "shorttermWinlossError %.3f\n", postprocessed.shorttermWinlossError)
            result += String(format: "shorttermScoreError %.3f\n", postprocessed.shorttermScoreError)
        }
        
        // Format policy grid (19x19) using postprocessed probabilities
        result += "policy\n"
        result += formatPolicyGridFromPostprocessed(policyProbs: postprocessed.policyProbs)
        
        // Format policy pass
        let policyPass = postprocessed.policyProbs[361] >= 0 ? postprocessed.policyProbs[361] : 0.0
        result += String(format: "policyPass %8.6f \n", policyPass)
        
        // Format ownership grid (19x19) using postprocessed values
        result += "whiteOwnership\n"
        result += formatOwnershipGridFromPostprocessed(ownership: postprocessed.ownership)
        
        // Empty line after symmetry block
        result += "\n"
        
        return result
    }
    
    /// Format postprocessed policy grid as 19 lines of 19 values each
    private func formatPolicyGridFromPostprocessed(policyProbs: [Float]) -> String {
        var result = ""
        
        for y in 0..<19 {
            var lineValues: [String] = []
            for x in 0..<19 {
                let positionIndex = y * 19 + x
                let value = positionIndex < policyProbs.count ? policyProbs[positionIndex] : 0.0
                
                if value < 0 {
                    lineValues.append("    NAN ")
                } else {
                    lineValues.append(String(format: "%8.6f ", value))
                }
            }
            result += lineValues.joined(separator: " ") + "\n"
        }
        
        return result
    }
    
    /// Format postprocessed ownership grid as 19 lines of 19 values each
    private func formatOwnershipGridFromPostprocessed(ownership: [Float]) -> String {
        var result = ""
        
        for y in 0..<19 {
            var lineValues: [String] = []
            for x in 0..<19 {
                let positionIndex = y * 19 + x
                let value = positionIndex < ownership.count ? ownership[positionIndex] : 0.0
                
                if value.isNaN {
                    lineValues.append("     NAN ")
                } else {
                    lineValues.append(String(format: "%9.7f ", value))
                }
            }
            result += lineValues.joined(separator: " ") + "\n"
        }
        
        return result
    }
}