#!/bin/bash
# Generate reference-output files for GTP fixtures by piping each .gtp script
# into a locally built KataGo binary and capturing stdout.
#
# Usage:
#   ./Scripts/generate_gtp_reference.sh                 # all fixtures
#   ./Scripts/generate_gtp_reference.sh --fixture <name>  # a single fixture
#   ./Scripts/generate_gtp_reference.sh --force-rebuild   # force KataGo rebuild
#
# Each fixture <name>.gtp under Scripts/GTPFixtures produces
# Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/<name>.txt.
#
# Prereqs: same as generate_kata_raw_nn_reference.sh — ninja, cmake, Xcode CLT.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FORCE_REBUILD=false
SINGLE_FIXTURE=""

while [ $# -gt 0 ]; do
    case $1 in
        --force-rebuild) FORCE_REBUILD=true; shift ;;
        --fixture)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --fixture requires a value"; exit 1
            fi
            SINGLE_FIXTURE="$1"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KATAGO_DIR="$PROJECT_ROOT/KataGo-metal-coreml-stable"
BUILD_DIR="$KATAGO_DIR/cpp/build"
KATAGO_EXE="$BUILD_DIR/katago"
FIXTURE_DIR="$PROJECT_ROOT/Scripts/GTPFixtures"
REFERENCE_OUTPUT_DIR="$PROJECT_ROOT/Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs"

mkdir -p "$REFERENCE_OUTPUT_DIR"

# ---- KataGo build (reuse the existing script's logic) ------------------------
# Delegate to generate_kata_raw_nn_reference.sh's build step if it already
# produced the binary. Otherwise, rebuild inline.
if [ "$FORCE_REBUILD" = true ] || [ ! -x "$KATAGO_EXE" ]; then
    echo -e "${YELLOW}Building KataGo (reusing generate_kata_raw_nn_reference.sh flow)...${NC}"
    if [ "$FORCE_REBUILD" = true ]; then
        bash "$SCRIPT_DIR/generate_kata_raw_nn_reference.sh" --force-rebuild >/dev/null
    else
        bash "$SCRIPT_DIR/generate_kata_raw_nn_reference.sh" >/dev/null
    fi
fi

if [ ! -x "$KATAGO_EXE" ]; then
    echo -e "${RED}KataGo binary not found at $KATAGO_EXE${NC}"
    exit 1
fi

# ---- Config and model discovery ----------------------------------------------
GTP_CONFIG="$PROJECT_ROOT/Scripts/gtp_example.cfg"
if [ ! -f "$GTP_CONFIG" ]; then
    GTP_CONFIG="$KATAGO_DIR/cpp/configs/gtp_example.cfg"
fi
if [ ! -f "$GTP_CONFIG" ]; then
    echo -e "${RED}Could not find a KataGo GTP config${NC}"
    exit 1
fi

BIN_MODEL="$BUILD_DIR/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"
if [ ! -f "$BIN_MODEL" ]; then
    echo -e "${RED}KataGo binary model not found at $BIN_MODEL; run generate_kata_raw_nn_reference.sh first${NC}"
    exit 1
fi
COREML_MODEL="$PROJECT_ROOT/Sources/KataGoOnAppleSilicon/Models/Resources/KataGoModel19x19fp16-adam-s11165M.mlpackage"
if [ ! -d "$COREML_MODEL" ]; then
    echo -e "${RED}Core ML model not found at $COREML_MODEL${NC}"
    exit 1
fi

# ---- Generate references -----------------------------------------------------
run_one() {
    local fixture_path="$1"
    local name
    name="$(basename "$fixture_path" .gtp)"
    local out="$REFERENCE_OUTPUT_DIR/$name.txt"
    echo -e "${YELLOW}Generating $name...${NC}"
    # KataGo sends all its init/logging to stderr by default; stdout is a
    # pure GTP response stream (= … / ? … blocks separated by blank lines).
    # DO NOT set KATAGO_DEBUG_DUMP — that injects extra output.
    "$KATAGO_EXE" gtp \
        -config "$GTP_CONFIG" \
        -model "$BIN_MODEL" \
        -coreml-model "$COREML_MODEL" \
        < "$fixture_path" \
        > "$out" \
        2>/dev/null
    echo -e "${GREEN}  -> $out${NC}"
}

if [ -n "$SINGLE_FIXTURE" ]; then
    path="$FIXTURE_DIR/$SINGLE_FIXTURE.gtp"
    if [ ! -f "$path" ]; then
        echo -e "${RED}No fixture named $SINGLE_FIXTURE at $path${NC}"; exit 1
    fi
    run_one "$path"
else
    for path in "$FIXTURE_DIR"/*.gtp; do
        run_one "$path"
    done
fi

echo -e "${GREEN}All requested references generated.${NC}"
