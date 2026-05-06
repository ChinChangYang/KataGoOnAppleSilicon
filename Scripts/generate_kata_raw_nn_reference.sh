#!/bin/bash

# Script to generate reference output files for kata-raw-nn command
# This script builds KataGo from source, sets up GTP session, and generates reference outputs
#
# Usage:
#   ./generate_kata_raw_nn_reference.sh [--force-rebuild] [--model-type AI|20k] [--board-size N]
#
# Options:
#   --force-rebuild    Force rebuilding KataGo executable even if it already exists
#   --model-type       Model type to use: "AI" (default) or "20k" (human SL)
#   --board-size       Board size: integer 2–19 (default: 19)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command-line arguments
FORCE_REBUILD=false
MODEL_TYPE="AI"
BOARD_SIZE=19
while [ $# -gt 0 ]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --model-type)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --model-type requires a value"
                echo "Usage: $0 [--force-rebuild] [--model-type AI|20k] [--board-size N]"
                exit 1
            fi
            MODEL_TYPE="$1"
            if [ "$MODEL_TYPE" != "AI" ] && [ "$MODEL_TYPE" != "20k" ]; then
                echo "Error: --model-type must be 'AI' or '20k'"
                echo "Usage: $0 [--force-rebuild] [--model-type AI|20k] [--board-size N]"
                exit 1
            fi
            shift
            ;;
        --board-size)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --board-size requires a value"
                echo "Usage: $0 [--force-rebuild] [--model-type AI|20k] [--board-size N]"
                exit 1
            fi
            BOARD_SIZE="$1"
            if ! [[ "$BOARD_SIZE" =~ ^[0-9]+$ ]] || [ "$BOARD_SIZE" -lt 2 ] || [ "$BOARD_SIZE" -gt 19 ]; then
                echo "Error: --board-size must be an integer between 2 and 19"
                echo "Usage: $0 [--force-rebuild] [--model-type AI|20k] [--board-size N]"
                exit 1
            fi
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force-rebuild] [--model-type AI|20k] [--board-size N]"
            exit 1
            ;;
    esac
done

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KATAGO_ARCHIVE_URL="https://github.com/ChinChangYang/KataGo/archive/metal-coreml-stable.tar.gz"
KATAGO_DIR="$PROJECT_ROOT/KataGo-metal-coreml-stable"
BUILD_DIR="$KATAGO_DIR/cpp/build"
KATAGO_EXE="$BUILD_DIR/katago"
REFERENCE_OUTPUT_DIR="$PROJECT_ROOT/Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs"

# Set model-specific configuration based on MODEL_TYPE
if [ "$MODEL_TYPE" = "20k" ]; then
    # Human SL model configuration
    BINARY_MODEL_URL="https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz"
    BINARY_MODEL_NAME="b18c384nbt-humanv0.bin.gz"
    CORE_ML_MODEL_PATH="$PROJECT_ROOT/Sources/KataGoOnAppleSilicon/Models/Resources/KataGoModel19x19fp16m1.mlpackage"
    REFERENCE_FILE_SUFFIX="_20k"
    HUMAN_SL_OVERRIDE="-override-config humanSLProfile=preaz_20k"
else
    # AI model configuration (default)
    BINARY_MODEL_URL="https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"
    BINARY_MODEL_NAME="kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"
    CORE_ML_MODEL_PATH="$PROJECT_ROOT/Sources/KataGoOnAppleSilicon/Models/Resources/KataGoModel19x19fp16-adam-s11165M.mlpackage"
    REFERENCE_FILE_SUFFIX=""
    HUMAN_SL_OVERRIDE=""
fi

# Board-size suffix: empty for 19x19 (default), "_NxN" otherwise
if [ "$BOARD_SIZE" = "19" ]; then
    BOARD_SIZE_SUFFIX=""
else
    BOARD_SIZE_SUFFIX="_${BOARD_SIZE}x${BOARD_SIZE}"
fi

# Create reference output directory
mkdir -p "$REFERENCE_OUTPUT_DIR"

echo -e "${GREEN}=== KataGo kata-raw-nn Reference File Generator ===${NC}"
echo ""

# Step 1: Check if KataGo is already built
if [ -f "$KATAGO_EXE" ] && [ "$FORCE_REBUILD" = false ]; then
    echo -e "${GREEN}✓ KataGo executable found at: $KATAGO_EXE${NC}"
else
    if [ "$FORCE_REBUILD" = true ]; then
        echo -e "${YELLOW}Force rebuild requested. Rebuilding KataGo from source...${NC}"
        # Remove existing build directory to force clean rebuild
        if [ -d "$BUILD_DIR" ]; then
            echo "Removing existing build directory..."
            rm -rf "$BUILD_DIR"
        fi
    else
        echo -e "${YELLOW}KataGo executable not found. Building from source...${NC}"
    fi
    
    # Download KataGo source if needed
    if [ ! -d "$KATAGO_DIR" ]; then
        echo "Downloading KataGo source..."
        cd "$PROJECT_ROOT"
        wget -q "$KATAGO_ARCHIVE_URL" -O metal-coreml-stable.tar.gz
        tar -zxf metal-coreml-stable.tar.gz
        rm metal-coreml-stable.tar.gz
    fi
    
    # Build KataGo
    echo "Building KataGo..."
    cd "$KATAGO_DIR/cpp"
    if [ -f "CMakeLists.txt-macos" ]; then
        cp CMakeLists.txt-macos CMakeLists.txt
    else
        echo -e "${RED}✗ CMakeLists.txt-macos not found. Cannot build with Core ML backend.${NC}"
        exit 1
    fi
    
    # Remove build directory to ensure clean build with Core ML backend
    if [ -d "build" ]; then
        echo "Removing existing build directory for clean rebuild..."
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    echo "Configuring CMake with Core ML backend..."
    cmake -G Ninja -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release ../
    
    echo "Building with Ninja..."
    ninja
    
    if [ ! -f "$KATAGO_EXE" ]; then
        echo -e "${RED}✗ Failed to build KataGo executable${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ KataGo built successfully${NC}"
fi

# Step 2: Download binary model if needed
# Ensure build directory exists
mkdir -p "$BUILD_DIR"
BINARY_MODEL_PATH="$BUILD_DIR/$BINARY_MODEL_NAME"
if [ ! -f "$BINARY_MODEL_PATH" ]; then
    echo "Downloading binary model for $MODEL_TYPE..."
    cd "$BUILD_DIR"
    wget -q "$BINARY_MODEL_URL" -O "$BINARY_MODEL_NAME"
fi

if [ ! -f "$BINARY_MODEL_PATH" ]; then
    echo -e "${RED}✗ Binary model not found at: $BINARY_MODEL_PATH${NC}"
    exit 1
fi

# Step 3: Check Core ML model
if [ ! -d "$CORE_ML_MODEL_PATH" ]; then
    echo -e "${RED}✗ Core ML model not found at: $CORE_ML_MODEL_PATH${NC}"
    echo "Please ensure the Core ML model is available in the Models/Resources directory"
    exit 1
fi

# Step 4: Locate GTP config
GTP_CONFIG="$KATAGO_DIR/cpp/configs/misc/coreml_gtp.cfg"
if [ ! -f "$GTP_CONFIG" ]; then
    echo -e "${RED}✗ GTP configuration file not found${NC}"
    echo "  Expected location: $GTP_CONFIG"
    echo "  Please ensure KataGo source is extracted at: $KATAGO_DIR"
    exit 1
fi

# Step 5: Create debug directory and generate reference output for empty board
echo ""
echo -e "${GREEN}Generating reference output for empty ${BOARD_SIZE}x${BOARD_SIZE} board (symmetry 0) using $MODEL_TYPE model...${NC}"

# Create debug directory for KataGo Core ML backend dumps
DEBUG_DIR="$PROJECT_ROOT/.cursor/debug"
mkdir -p "$DEBUG_DIR"
echo -e "${GREEN}Debug directory created: $DEBUG_DIR${NC}"

REFERENCE_FILE="$REFERENCE_OUTPUT_DIR/kata_raw_nn_empty_board${BOARD_SIZE_SUFFIX}_symmetry_0${REFERENCE_FILE_SUFFIX}.txt"

# Create a temporary file for GTP commands
GTP_INPUT=$(mktemp)
cat > "$GTP_INPUT" << GTPEOF
boardsize ${BOARD_SIZE}
clear_board
kata-raw-nn 0
quit
GTPEOF

# Run KataGo and extract output
# KataGo GTP format: "= <output>\n\n" for success, "? <error>\n\n" for error
echo "Running KataGo GTP session with debug dumping enabled..."
echo -e "${YELLOW}Note: KATAGO_DEBUG_DUMP=1 is set to capture Core ML backend inputs/outputs${NC}"
cd "$PROJECT_ROOT"

# Capture raw output and stderr separately for error checking
RAW_OUTPUT_FILE="$REFERENCE_OUTPUT_DIR/raw_gtp_output.txt"
STDERR_FILE="$REFERENCE_OUTPUT_DIR/gtp_stderr.txt"

# Set KATAGO_DEBUG_DUMP=1 to enable debug dumping in coremlbackend.swift
# Build command with optional human SL override
if [ -n "$HUMAN_SL_OVERRIDE" ]; then
    # For human SL model, use -human-model flag to get human format output
    # We still need a main model, so use the human model for both
    KATAGO_DEBUG_DUMP=1 "$KATAGO_EXE" gtp \
        -model "$BINARY_MODEL_PATH" \
        -human-model "$BINARY_MODEL_PATH" \
        -coreml-model "$CORE_ML_MODEL_PATH" \
        -config "$GTP_CONFIG" \
        $HUMAN_SL_OVERRIDE < "$GTP_INPUT" > "$RAW_OUTPUT_FILE" 2> "$STDERR_FILE"
else
    # For AI model, use standard command
    KATAGO_DEBUG_DUMP=1 "$KATAGO_EXE" gtp \
        -model "$BINARY_MODEL_PATH" \
        -coreml-model "$CORE_ML_MODEL_PATH" \
        -config "$GTP_CONFIG" < "$GTP_INPUT" > "$RAW_OUTPUT_FILE" 2> "$STDERR_FILE"
fi

# Check stderr for "Dummy neural net backend" error - indicates Core ML backend not enabled
if [ -f "$STDERR_FILE" ]; then
    STDERR_CONTENT=$(cat "$STDERR_FILE" 2>/dev/null || echo "")
    if echo "$STDERR_CONTENT" | grep -q "Dummy neural net backend"; then
        echo -e "${RED}✗ KataGo executable was built without Core ML backend support${NC}"
        echo -e "${YELLOW}The executable at $KATAGO_EXE needs to be rebuilt with Core ML backend.${NC}"
        echo -e "${YELLOW}Please run: $0 --force-rebuild${NC}"
        exit 1
    fi
fi

# Process raw output with awk
# GTP responds to "kata-raw-nn 0" with "= symmetry 0" followed by the output
awk '
    /^= symmetry 0$/ {
        # Print the symmetry line
        print
        # Continue reading and printing until we hit the next GTP response (empty line followed by "= ")
        while (getline > 0) {
            if (/^= $/) {
                # End of response (next GTP command response), stop capturing
                break
            }
            if (/^\?/) {
                # Error response, stop
                break
            }
            # Print the line (this is the actual output)
            print
        }
    }
' "$RAW_OUTPUT_FILE" > "$REFERENCE_FILE"

# Clean up temporary files
rm -f "$GTP_INPUT"
rm -f "$RAW_OUTPUT_FILE" "$STDERR_FILE"


if [ -f "$REFERENCE_FILE" ] && [ -s "$REFERENCE_FILE" ]; then
    echo -e "${GREEN}✓ Reference file generated: $REFERENCE_FILE${NC}"
    echo "File size: $(wc -l < "$REFERENCE_FILE") lines"
else
    echo -e "${RED}✗ Failed to generate reference file${NC}"
    exit 1
fi

# Check if debug dumps were generated
echo ""
echo -e "${GREEN}Checking for debug dumps...${NC}"
if [ -f "$DEBUG_DIR/katago_coreml_inputs_spatial.txt" ]; then
    echo -e "${GREEN}✓ Debug dumps generated in: $DEBUG_DIR${NC}"
    echo "  - katago_coreml_inputs_spatial.txt"
    echo "  - katago_coreml_inputs_global.txt"
    echo "  - katago_coreml_raw_policy.txt"
    echo "  - katago_coreml_raw_value.txt"
    echo "  - katago_coreml_raw_ownership.txt"
    echo "  - (and other output files)"
else
    echo -e "${YELLOW}⚠ No debug dumps found. Make sure KATAGO_DEBUG_DUMP=1 was set correctly.${NC}"
fi

echo ""
echo -e "${GREEN}=== Reference file generation complete ===${NC}"
