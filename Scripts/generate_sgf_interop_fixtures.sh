#!/bin/bash
# Generate SGF interop fixtures by driving each scenario through both
# Swift's GTPHandler (via the GTPRunner executable) and KataGo, capturing
# our printsgf, KataGo's printsgf, and our printsgf-after-loading-KataGo.
#
# Usage:
#   ./Scripts/generate_sgf_interop_fixtures.sh                   # all scenarios
#   ./Scripts/generate_sgf_interop_fixtures.sh --scenario empty  # one
#   ./Scripts/generate_sgf_interop_fixtures.sh --force-rebuild   # rebuild KataGo
#
# Each driver Scripts/SGFFixtureDrivers/<name>.gtp produces three fixtures
# under Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/:
#   <name>.export.sgf  — our native printsgf at the position
#   <name>.katago.sgf  — KataGo's printsgf at the same position
#   <name>.import.sgf  — our printsgf after loadsgf'ing <name>.katago.sgf
#
# All four phases below must succeed; the script aborts on the first failure
# with a red error pointing at the failed phase.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FORCE_REBUILD=false
SINGLE_SCENARIO=""

while [ $# -gt 0 ]; do
    case $1 in
        --force-rebuild) FORCE_REBUILD=true; shift ;;
        --scenario)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --scenario requires a value"; exit 1
            fi
            SINGLE_SCENARIO="$1"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KATAGO_DIR="$PROJECT_ROOT/KataGo-metal-coreml-stable"
BUILD_DIR="$KATAGO_DIR/cpp/build"
KATAGO_EXE="$BUILD_DIR/katago"
DRIVER_DIR="$PROJECT_ROOT/Scripts/SGFFixtureDrivers"
FIXTURE_DIR="$PROJECT_ROOT/Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures"

mkdir -p "$FIXTURE_DIR"

# ---- KataGo build (delegate to the reference-gen script) --------------------
if [ "$FORCE_REBUILD" = true ] || [ ! -x "$KATAGO_EXE" ]; then
    echo -e "${YELLOW}Building KataGo (delegating to generate_kata_raw_nn_reference.sh)...${NC}"
    if [ "$FORCE_REBUILD" = true ]; then
        bash "$SCRIPT_DIR/generate_kata_raw_nn_reference.sh" --force-rebuild >/dev/null
    else
        bash "$SCRIPT_DIR/generate_kata_raw_nn_reference.sh" >/dev/null
    fi
fi

if [ ! -x "$KATAGO_EXE" ]; then
    echo -e "${RED}KataGo binary not found at $KATAGO_EXE${NC}"; exit 1
fi

# ---- Config and model discovery (parity with generate_gtp_reference.sh) -----
GTP_CONFIG="$KATAGO_DIR/cpp/configs/gtp_example.cfg"
if [ ! -f "$GTP_CONFIG" ]; then
    echo -e "${RED}KataGo GTP config not found at $GTP_CONFIG${NC}"; exit 1
fi

BIN_MODEL="$BUILD_DIR/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"
if [ ! -f "$BIN_MODEL" ]; then
    echo -e "${RED}KataGo binary model not found at $BIN_MODEL; run generate_kata_raw_nn_reference.sh first${NC}"; exit 1
fi

COREML_MODEL="$PROJECT_ROOT/Sources/KataGoOnAppleSilicon/Models/Resources/KataGoModel19x19fp16-adam-s11165M.mlpackage"
if [ ! -d "$COREML_MODEL" ]; then
    echo -e "${RED}Core ML model not found at $COREML_MODEL${NC}"; exit 1
fi

# ---- Build GTPRunner once so each scenario reuses the binary ----------------
echo -e "${YELLOW}Building GTPRunner...${NC}"
( cd "$PROJECT_ROOT" && swift build --target GTPRunner >/dev/null )
GTPRUNNER_BIN="$(cd "$PROJECT_ROOT" && swift build --target GTPRunner --show-bin-path)/GTPRunner"
if [ ! -x "$GTPRUNNER_BIN" ]; then
    echo -e "${RED}GTPRunner binary not found at $GTPRUNNER_BIN${NC}"; exit 1
fi

run_katago_gtp() {
    # stdin → KataGo gtp → stdout (stderr suppressed)
    "$KATAGO_EXE" gtp \
        -config "$GTP_CONFIG" \
        -model "$BIN_MODEL" \
        -coreml-model "$COREML_MODEL" \
        2>/dev/null
}

generate_one() {
    local driver_path="$1"
    local name; name="$(basename "$driver_path" .gtp)"
    local out_export="$FIXTURE_DIR/$name.export.sgf"
    local out_katago="$FIXTURE_DIR/$name.katago.sgf"
    local out_import="$FIXTURE_DIR/$name.import.sgf"
    local tmp_dir; tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' RETURN

    echo -e "${YELLOW}Generating $name...${NC}"

    # Phase 1 — produce our native SGF.
    {
        cat "$driver_path"
        printf 'printsgf %s\nquit\n' "$tmp_dir/export.sgf"
    } | "$GTPRUNNER_BIN" >/dev/null
    if [ ! -s "$tmp_dir/export.sgf" ]; then
        echo -e "${RED}[$name] phase 1 failed: GTPRunner did not produce export.sgf${NC}"; exit 1
    fi
    cp "$tmp_dir/export.sgf" "$out_export"

    # Phase 2 — KataGo must accept our SGF.
    local export_response
    export_response=$(
        printf 'loadsgf %s\nquit\n' "$out_export" | run_katago_gtp
    )
    if echo "$export_response" | grep -q '^?'; then
        echo -e "${RED}[$name] phase 2 failed: KataGo rejected our export SGF${NC}"
        echo "Our SGF:"; cat "$out_export"
        echo "KataGo response:"; echo "$export_response"
        exit 1
    fi

    # Phase 3 — produce KataGo's SGF.
    {
        cat "$driver_path"
        printf 'printsgf %s\nquit\n' "$tmp_dir/katago.sgf"
    } | run_katago_gtp >/dev/null
    if [ ! -s "$tmp_dir/katago.sgf" ]; then
        echo -e "${RED}[$name] phase 3 failed: KataGo did not produce katago.sgf${NC}"; exit 1
    fi
    cp "$tmp_dir/katago.sgf" "$out_katago"

    # Phase 4 — round-trip KataGo's SGF through us, capture our re-emission.
    {
        printf 'loadsgf %s\nprintsgf %s\nquit\n' "$out_katago" "$tmp_dir/import.sgf"
    } | "$GTPRUNNER_BIN" >/dev/null
    if [ ! -s "$tmp_dir/import.sgf" ]; then
        echo -e "${RED}[$name] phase 4 failed: GTPRunner did not produce import.sgf after loading KataGo's SGF${NC}"
        echo "KataGo SGF:"; cat "$out_katago"
        exit 1
    fi
    cp "$tmp_dir/import.sgf" "$out_import"

    echo -e "${GREEN}  -> $out_export${NC}"
    echo -e "${GREEN}  -> $out_katago${NC}"
    echo -e "${GREEN}  -> $out_import${NC}"
}

if [ -n "$SINGLE_SCENARIO" ]; then
    path="$DRIVER_DIR/$SINGLE_SCENARIO.gtp"
    if [ ! -f "$path" ]; then
        echo -e "${RED}No driver named $SINGLE_SCENARIO at $path${NC}"; exit 1
    fi
    generate_one "$path"
else
    for path in "$DRIVER_DIR"/*.gtp; do
        generate_one "$path"
    done
fi

echo -e "${GREEN}All requested fixtures generated.${NC}"
