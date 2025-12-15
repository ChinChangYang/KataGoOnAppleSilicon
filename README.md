## Features

- API-only interface for GTP command handling
- Bundled Core ML models for strongest 28b and human SL networks
- Model-only inference (no search yet)
- 19x19 board with Chinese rules (area scoring, positional ko, no suicide)
- Text-based status reporting for testing

## Usage

```swift
import KataGoOnAppleSilicon

// Create board
let board = Board()
board.playMove(at: Point(x: 3, y: 3), stone: .black)

// Load model
let katago = KataGoInference()
try katago.loadModel(for: "AI")

// Create board state for inference
let boardState = BoardState(board: board)

// Predict
let output = try katago.predict(board: boardState, profile: "AI")

// GTP handling
let gtp = GTPHandler(katago: katago)
let response = gtp.handleCommand("genmove white")
print(response)  // "= D4\n\n"
```

## Requirements

- macOS 12.0+
- Apple Silicon (M1/M2+)

## Building

```bash
swift build
swift test
```

## Integration Testing

The project includes integration tests that validate the Swift `kata-raw-nn` implementation against KataGo's reference output. These tests ensure the Swift implementation produces output that exactly matches KataGo's C++ implementation.

### Quick Start

```bash
# Run all integration tests
swift test --filter KataGoOnAppleSiliconIntegrationTests

# Run specific test
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard
```

### Prerequisites

1. **Reference Files**: Generate reference files using the provided script:
   ```bash
   ./Scripts/generate_kata_raw_nn_reference.sh
   ```

2. **Build Tools**: 
   - Ninja: `brew install ninja`
   - Xcode (for building KataGo)

3. **Models**: 
   - Core ML model must be in `Sources/KataGoOnAppleSilicon/Models/Resources/`
   - Binary model will be automatically downloaded by the script (~258 MB)

The reference generation script will:
- Build KataGo from source (if needed)
- Download the binary model (if needed)
- Run KataGo GTP session with `kata-raw-nn` command
- Extract and save output to reference files in `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/`

For detailed information, see [Integration Testing Guide](docs/INTEGRATION_TESTING.md).

## Acknowledgments

This project is a Swift port of KataGo's neural network inference algorithms. The input feature encoding, post-processing logic, and board algorithms are derived from KataGo's C++ implementation:

- **KataGo**: https://github.com/lightvector/KataGo
- **Input Features**: Derived from `fillRowV7()` in `cpp/neuralnet/nninputs.cpp`
- **Post-Processing**: Derived from `nneval.cpp` (value, policy, and ownership post-processing)
- **Board Logic**: Ported from KataGo's board implementation

The Swift implementation maintains compatibility with KataGo's neural network models and produces identical output for the `kata-raw-nn` command.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Model Files

The Core ML model files are not included in this repository due to their size (~191MB total). Download them separately from the [releases](https://github.com/ChinChangYang/KataGo/releases/tag/v1.16.4-coreml1) and place them in `Sources/KataGoOnAppleSilicon/Models/Resources/`.