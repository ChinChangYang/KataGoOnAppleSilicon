# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KataGoOnAppleSilicon** is a Swift package that provides Core ML-based neural network inference for KataGo, a strong Go AI engine. The project ports KataGo's input feature encoding, post-processing logic, and board algorithms from C++ to Swift, enabling AI-powered Go move analysis on Apple Silicon Macs.

**Key characteristics**:
- Swift package targeting macOS 12.0+ on Apple Silicon (M1/M2+)
- Core ML neural network inference (no C++ compilation required)
- Supports both AI models (strongest 28b) and human SL models (1d-9d, 1k-20k skill levels)
- GTP command handling for integration with Go engines and UIs
- Full integration testing against KataGo's reference implementation

## Build & Test Commands

```bash
# Build the library
swift build

# Run all tests
swift test

# Run specific test suite
swift test --filter KataGoOnAppleSiliconTests

# Run integration tests (requires reference files and Core ML models)
swift test --filter KataGoOnAppleSiliconIntegrationTests

# Run a single test
swift test --filter GTPHandlerTests.testBasicCommands

# Run game generator test (debugging tool)
swift test --filter GameGeneratorTests.testGenerateGame
```

## Integration Testing Workflow

Integration tests validate the Swift `kata-rawnn` command against KataGo's reference output. This ensures byte-for-byte compatibility.

### Prerequisites

1. **Core ML Models**: Download from [releases](https://github.com/ChinChangYang/KataGo/releases/tag/v1.16.4-coreml1) and place in `Sources/KataGoOnAppleSilicon/Models/Resources/`:
   - `KataGoModel19x19fp16-adam-s11165M.mlpackage` (AI model)
   - `KataGoModel19x19fp16m1.mlpackage` (Human SL model)

2. **Build Tools**: `brew install ninja` (required for KataGo compilation)

3. **Binary Models**: Downloaded automatically by the reference generation script

### Generate Reference Files

```bash
# Generate reference for AI model (empty board, symmetry 0)
./Scripts/generate_kata_raw_nn_reference.sh

# Generate reference for 20k human SL model
./Scripts/generate_kata_raw_nn_reference.sh --model-type 20k

# Force rebuild KataGo before generating reference
./Scripts/generate_kata_raw_nn_reference.sh --force-rebuild

# Generate reference for a non-19x19 empty board
./Scripts/generate_kata_raw_nn_reference.sh --board-size 9
./Scripts/generate_kata_raw_nn_reference.sh --board-size 13
```

Reference files are saved to `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/`

### Run Integration Tests

```bash
# Run all integration tests
swift test --filter KataGoOnAppleSiliconIntegrationTests

# Run specific test
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard20k

# Run new non-19 board parity tests
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard9x9
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard13x13
```

Tests use **relative tolerance (0.1%)** for floating-point comparisons to account for different value scales.

## Architecture & Key Modules

### Core Components

```
Sources/KataGoOnAppleSilicon/
├── KataGoInference.swift          # Main inference API, model loading, rawNN() command formatting
├── BoardState.swift               # Input feature encoding (22 spatial planes, 19 global features)
├── GTPHandler.swift               # GTP protocol command handling
├── Errors.swift                   # Error types
├── Core/
│   ├── Board.swift                # Board representation and move logic
│   ├── Move.swift                 # Move representation
│   ├── Rules.swift                # Ko and scoring rules (Chinese rules)
│   ├── PostProcessing.swift       # Value/policy/ownership post-processing
│   ├── SGFMetadata.swift          # Human SL model metadata encoding
│   └── Encoder.swift              # (if present) Neural network input encoding utilities
├── Models/
│   ├── ModelLoader.swift          # Core ML model loading and resource management
│   ├── ModelStatus.swift          # Model availability and status tracking
│   └── Resources/                 # Bundled Core ML models
└── Search/ (placeholder for future MCTS implementation)
    ├── SearchEngine.swift
    ├── MCTS.swift
    └── Evaluator.swift
```

### Input Feature Encoding

**BoardState** creates feature tensors matching KataGo's `fillRowV7()` implementation:
- **Spatial features** (shape: [1, 22, 19, 19]): Board state history, player perspective, special planes
- **Global features** (shape: [1, 19]): Game metadata (turn count, ko info, pass history, komi)

Planes 0-8 encode current/historical board state; planes 9-13 encode pass history; planes 14-21 encode metadata.

### Model Support

**AI Model**: Strongest 28b network (`KataGoModel19x19fp16-adam-s11165M.mlpackage`)
- No special preprocessing required
- Input: spatial + global tensors

**Human SL Models**: Skill-level specific networks (`KataGoModel19x19fp16m1.mlpackage`)
- Supports: 1d-9d dan, 1k-20k kyu levels
- Requires: Additional `input_meta` tensor (192 elements encoding rank/metadata)
- Generated by `SGFMetadata.fillMetadataRow()` based on profile name

### Post-Processing

**PostProcessing** transforms raw neural network outputs:
- **Values**: Converts logits to win/loss/draw probabilities and score predictions
- **Policy**: Softmax normalization and illegal move masking
- **Ownership**: Value-based territory prediction

All post-processing parameters match KataGo's default configuration.

## Recent Development Focus

The project has recently:
- Added human skill level profile validation in `KataGoInference`
- Refactored value extraction methods in `BoardState` for clarity
- Simplified shortterm error processing in `PostProcessing`
- Enhanced integration testing framework for 20k model support
- Implemented `SGFMetadata` for human SL model compatibility

See recent commits for ongoing work on feature completeness and reference compatibility.

## Important Implementation Details

### Board Coordinates
- Internal: (x, y) with origin at bottom-left, x: 0-18 left-to-right, y: 0-18 bottom-to-top
- GTP: Algebraic notation (A1-T19), 1-based indexing, converted to internal coordinates in `GTPHandler`
- Output: Policy/ownership grids use same internal coordinate system as input features

### Profile String Format
- AI model: use string `"AI"`
- Human SL: use pattern `"{level}{suffix}"` where level is 1-9 for dan or 1-20 for kyu, suffix is 'd' or 'k'
  - Examples: `"1d"`, `"5d"`, `"1k"`, `"20k"`
  - Profiles are converted to `"preaz_{profile}"` for `SGFMetadata` encoding

### Integration Test Tolerance
- Tests use relative tolerance formula: `abs(swift - ref) / max(abs(swift), abs(ref), 1.0) <= 0.001`
- This handles values spanning multiple scales (probabilities ~0.5, scores ~2-3, squared scores ~4500+)
- Exact match required for non-numeric lines (headers, labels)

### Reference Generation
The `generate_kata_raw_nn_reference.sh` script:
1. Builds KataGo from source (if needed)
2. Downloads binary models (if needed)
3. Runs KataGo GTP with `kata-raw-nn` command
4. Extracts output to reference files

Reference files enable byte-for-byte compatibility validation without requiring KataGo in production.

## Testing Patterns

### Unit Tests
Located in `Tests/KataGoOnAppleSiliconTests/`. Mock models available in `MockModels.swift` for testing without Core ML.

```swift
// Example: Testing inference with mock model
let mockModel = MockMLModel()
let katago = KataGoInference()
katago.setModel(mockModel, for: "AI")
let output = try katago.predict(board: boardState, profile: "AI")
```

### Integration Tests
Located in `Tests/KataGoOnAppleSiliconIntegrationTests/`. Tests load reference files and compare Swift output.

```swift
// Example: Integration test pattern
let referenceOutput = try loadReferenceFile(testCase: "empty_board", symmetry: 0)
let swiftOutput = try katago.rawNN(board: board, boardState: boardState, profile: "AI", whichSymmetry: 0)
let result = compareOutputs(swift: swiftOutput, reference: referenceOutput, tolerance: 0.001)
#expect(result.matches)
```

## Game Generator (Debugging Tool)

The game generator test (`GameGeneratorTests.testGenerateGame`) is a debugging tool that:
- Generates a complete 10-move game using the AI model
- Displays GTP coordinates alongside their SGF equivalents for verification
- Saves the game to an SGF file for inspection
- Validates the full pipeline from GTP commands through move generation to SGF export

### Running the Game Generator

```bash
swift test --filter GameGeneratorTests.testGenerateGame
```

### Expected Output

The test prints a table showing move pairs:
```
Move | Color | GTP Coord | SGF Coord
-----|-------|-----------|----------
   1 | Black | C4        | cp
   2 | White | Q4        | pp
   ...
```

And saves an SGF file to `.build/test-output/` (e.g., `.build/test-output/game_1765946272.sgf`):

**Note**: SGF files are automatically saved to the `.build/` directory (which is in `.gitignore`), keeping your repository clean even when running tests multiple times.
```
(;FF[4]GM[1]SZ[19]PB[KataGo (Black)]PW[KataGo (White)]KM[7.5];B[cp];W[pp];B[pc];W[dd];B[qf];W[fq];B[dq];W[qn];B[cc];W[cd])
```

### What It Tests

- **GTP Handler**: Verifies `genmove` command generates moves and updates board state
- **SGF Generator**: Validates coordinate conversion (GTP "C4" → SGF "cp")
- **Move Tracking**: Ensures board history is correctly maintained
- **SGF Export**: Confirms valid SGF file format

### Use Cases

- Verify coordinate conversion correctness
- Debug move generation issues
- Inspect AI-generated move sequences
- Validate SGF export functionality
- Demonstrate the complete inference pipeline

## Common Development Tasks

### Adding a New Model Profile
1. Update `KataGoInference.loadModel(for:)` to recognize the profile
2. Add corresponding Core ML model file to `Sources/KataGoOnAppleSilicon/Models/Resources/`
3. For human SL models: update `isHumanSLProfile()` validation if needed
4. Add `SGFMetadata` encoding if model requires `input_meta`
5. Create integration test with reference file

### Debugging Integration Test Failures
1. Run test with verbose output to see exact mismatch: `swift test --filter <test_name> -v`
2. Compare raw outputs: Check `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/` against generated output
3. Regenerate reference file if KataGo version changed: `./Scripts/generate_kata_raw_nn_reference.sh --force-rebuild`
4. Check formatting precision (floating-point output format must match KataGo exactly)
5. Verify coordinate mapping in `GTPHandler` if move selection incorrect

### Extending GTP Handler
Add new commands to `handleCommand()` in `GTPHandler.swift`. Ensure:
- Command is added to `knownCommands` array
- Response format matches GTP protocol (success: `"= <response>\n\n"`, error: `"? <message>\n\n"`)
- Illegal moves return `"? illegal move\n\n"`
- Unknown commands return `"? unknown command\n\n"`

## Dependencies & Environment

- **Swift 6.2+** (declared in `Package.swift`)
- **CoreML framework** (Apple platform)
- **Foundation framework**

No external package dependencies; pure Swift + Apple frameworks.

## Known Limitations & Future Work

- **No MCTS yet**: Currently model-only inference. Search engine framework exists but incomplete.
- **Square boards 2x2 to 19x19, Chinese rules only**: Inference uses a 19x19-trained Core ML model for all sizes; smaller boards are encoded into the top-left of the 19x19 tensor. 9x9 and 13x13 are byte-checked against KataGo C++; other sizes work transitively.
- **Model files required**: Core ML models must be downloaded separately (~191 MB total)
- **macOS only**: Apple Silicon requirement due to Core ML optimization

## Acknowledgments

This project is a Swift port of KataGo's neural network inference algorithms:
- **KataGo**: https://github.com/lightvector/KataGo
- **Input Features**: Derived from `fillRowV7()` in `cpp/neuralnet/nninputs.cpp`
- **Input Meta (Human SL)**: Derived from `SGFMetadata::fillMetadataRow()` in `cpp/neuralnet/sgfmetadata.cpp`
- **Post-Processing**: Derived from `nneval.cpp`
- **Board Logic**: Ported from KataGo's board implementation

See README.md for full attribution and links.
