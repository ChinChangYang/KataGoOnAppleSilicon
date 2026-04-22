## Features

- GTP engine with model-only inference (no search yet)
- Interactive CLI application (`KataGoPlay`) for playing against the AI
- Board sizes 2x2 to 19x19 with Chinese rules (area scoring, positional ko, no suicide)
- Bundled Core ML models for strongest 28b and human SL networks
- 30 human SL profiles: 1d–9d (dan) and 1k–20k (kyu)
- Resign logic, friendly pass mode, greedy and probabilistic move selection
- Board rendering, real-time analysis display, and SGF export
- GTP commands: `protocol_version`, `name`, `version`, `known_command`, `list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `genmove`, `undo`, `fixed_handicap`, `set_free_handicap`, `kata-set-rules`, `showboard`, `kata-rawnn`, `final_score`, `quit`

## Usage

### Library API

```swift
import KataGoOnAppleSilicon

let katago = KataGoInference()
try katago.loadModel(for: "AI")    // Load AI model (strongest 28b)
try katago.loadModel(for: "1d")    // Load human SL model (1d profile)

let gtp = GTPHandler(katago: katago)

// Set board size and komi
_ = gtp.handleCommand("boardsize 13")
_ = gtp.handleCommand("komi 7.5")

// Generate a move
let response = gtp.handleCommand("genmove black")
print(response)  // "= D4\n\n"

// Switch to a different human SL profile
gtp.setProfile("5k")
let response5k = gtp.handleCommand("genmove white")

// Get estimated score
let score = gtp.handleCommand("final_score")
```

### KataGoPlay (Interactive CLI)

```bash
swift run KataGoPlay
```

KataGoPlay provides an interactive setup flow (color, AI profile, board size, komi) and a game loop with board rendering, move highlighting, win-rate analysis, and SGF export. In-game commands:

```
<coord>           Play a move (e.g. D4)
pass              Pass your turn
hint              Show top suggested moves on the board
analysis          Detailed win-rate and score analysis
board             Redraw the board
save              Export the game as SGF
profile <name>    Switch AI profile (e.g. profile 3d)
ai                Let the AI play your turn
quit              Exit
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

The project includes integration tests that validate the Swift `kata-rawnn` implementation against KataGo's reference output. These tests ensure the Swift implementation produces output that exactly matches KataGo's C++ implementation.

### Quick Start

```bash
# Run all integration tests
swift test --filter KataGoOnAppleSiliconIntegrationTests

# Run specific test
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard

# Run 20k model test
swift test --filter KataRawNNIntegrationTests.testKataRawNNEmptyBoard20k
```

### Prerequisites

1. **Reference Files**: Generate reference files using the provided script:
   ```bash
   # Generate reference for AI model (default)
   ./Scripts/generate_kata_raw_nn_reference.sh
   
   # Generate reference for 20k human SL model
   ./Scripts/generate_kata_raw_nn_reference.sh --model-type 20k
   ```

2. **Build Tools**: 
   - Ninja: `brew install ninja`
   - Xcode (for building KataGo)

3. **Models**: 
   - Core ML models must be in `Sources/KataGoOnAppleSilicon/Models/Resources/`
     - `KataGoModel19x19fp16-adam-s11165M.mlpackage` (AI model)
     - `KataGoModel19x19fp16m1.mlpackage` (Human SL model)
   - Binary models will be automatically downloaded by the script:
     - AI model: `kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz` (~258 MB)
     - Human SL model: `b18c384nbt-humanv0.bin.gz` (for 20k model)

The reference generation script will:
- Build KataGo from source (if needed)
- Download the appropriate binary model (if needed)
- Run KataGo GTP session with `kata-raw-nn` command
- Extract and save output to reference files in `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/`

For detailed information, see [Integration Testing Guide](docs/INTEGRATION_TESTING.md).

## Game Generator (Debugging Tool)

The game generator is a debugging tool that generates a complete 10-move game, demonstrating the full pipeline from GTP commands to SGF export. It validates that:
- The `genmove` command generates and plays moves correctly
- GTP coordinates are properly converted to SGF format
- Moves are tracked and exported to valid SGF files

### Running the Game Generator

```bash
swift test --filter GameGeneratorTests.testGenerateGame
```

### Expected Output

The test displays a table showing each move with both GTP and SGF coordinates for verification:

```
Move | Color | GTP Coord | SGF Coord
-----|-------|-----------|----------
   1 | Black | C4        | cp
   2 | White | Q4        | pp
   3 | Black | Q17       | pc
   4 | White | D16       | dd
   5 | Black | R14       | qf
   6 | White | F3        | fq
   7 | Black | D3        | dq
   8 | White | R6        | qn
   9 | Black | C17       | cc
  10 | White | C16       | cd
```

The generated SGF file is saved to `.build/test-output/` (e.g., `.build/test-output/game_1765946272.sgf`) and contains:

**Note**: SGF files are saved to the `.build/` directory, which is in `.gitignore`, so running tests does not make your git status dirty.

```
(;FF[4]GM[1]SZ[19]PB[KataGo (Black)]PW[KataGo (White)]KM[7.5];B[cp];W[pp];B[pc];W[dd];B[qf];W[fq];B[dq];W[qn];B[cc];W[cd])
```

This tool is useful for:
- Verifying the GTP-to-SGF coordinate conversion is correct
- Debugging move generation and board state management
- Inspecting the actual moves generated by the AI model
- Validating SGF export functionality

## Acknowledgments

This project is a Swift port of KataGo's neural network inference algorithms. The input feature encoding, post-processing logic, and board algorithms are derived from KataGo's C++ implementation:

- **KataGo**: https://github.com/lightvector/KataGo
- **Input Features**: Derived from `fillRowV7()` in `cpp/neuralnet/nninputs.cpp`
- **Input Meta (Human SL)**: Derived from `SGFMetadata::fillMetadataRow()` in `cpp/neuralnet/sgfmetadata.cpp`
- **Post-Processing**: Derived from `nneval.cpp` (value, policy, and ownership post-processing)
- **Board Logic**: Ported from KataGo's board implementation

The Swift implementation maintains compatibility with KataGo's neural network models and produces identical output for the `kata-rawnn` command, supporting both AI models and human SL models (30 profiles: 1d–9d dan, 1k–20k kyu).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Model Files

The Core ML model files are not included in this repository due to their size (~191MB total). Download them separately from the [releases](https://github.com/ChinChangYang/KataGo/releases/tag/v1.16.4-coreml1) and place them in `Sources/KataGoOnAppleSilicon/Models/Resources/`.