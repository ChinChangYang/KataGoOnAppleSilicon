# SGF Interop Test for loadsgf / printsgf — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-engine SGF interoperability tests proving Swift's `printsgf` / `loadsgf` agree with KataGo's reference implementation on a fixed scenario corpus.

**Architecture:** Offline shell generator drives both Swift's GTPHandler (via a new `GTPRunner` executable) and KataGo (via the existing build) over the same 7 position-setup `.gtp` drivers. It commits three SGF fixtures per scenario: `<scenario>.export.sgf` (what we emit natively), `<scenario>.katago.sgf` (KataGo's printsgf), and `<scenario>.import.sgf` (what we emit after loading KataGo's SGF). Two `@Test` cases per scenario do byte-equal comparisons against the appropriate canonical.

**Tech Stack:** Swift 6.2, Swift Testing, CoreML, KataGo binary at `KataGo-metal-coreml-stable/cpp/build/katago`, bash, Swift Package Manager.

**Spec:** `docs/superpowers/specs/2026-04-26-loadsgf-printsgf-interop-test-design.md`

---

## Watchpoint resolution

The spec flagged two watchpoints. Resolved during plan-writing by probing actual KataGo output:

- **Engine-specific properties.** Probed `printsgf` for an empty board:
  - **Ours:** `(;FF[4]GM[1]SZ[19]PB[Black]PW[White]KM[7.5])`
  - **KataGo's:** `(;FF[4]GM[1]SZ[19]PB[]PW[]HA[0]KM[7.5]RU[TrompTaylor])`

  Three differences. Our `loadsgf` honours `PB`/`PW`/`RU`, so loading KataGo's SGF will overwrite our defaults. Adopting the spec's design literally (one canonical) would force direction 2 to fail.

  **Resolution:** **two canonicals per scenario.** `<scenario>.export.sgf` is what we emit natively (direction 1's expected); `<scenario>.import.sgf` is what we emit after `loadsgf <scenario>.katago.sgf` (direction 2's expected). Both are valid SGFs of the same game position; only metadata differs. The generator produces all three files (export + katago + import) per scenario.

- **`AP` stability.** Confirmed: our `printsgf` does not emit `AP` at all. No version drift risk. No code change needed.

## Acceptance criteria (revised)

The spec's mutation criteria targeted `PL[W]` for the handicap scenario and `B[]` for passes. After reviewing `SGFGenerator.swift`:

- `PL[W]` is **only** emitted when `initialSideToMove != defaultPla`. Standard `fixed_handicap N` already sets `initialSideToMove = .white` and `defaultPla = .white` (because handicap stones are non-empty), so `PL` is **omitted** in the canonical. A "remove PL emission" mutation would be a no-op for these tests.

  **Revised mutation #1:** Mutate `SGFGenerator.buildSGF` to skip the `HA[…]AB[…]` block (`if !handicapBlack.isEmpty` → `if false`). This is what handicap tests must catch.

- `B[]` / `W[]` for passes is unchanged.

  **Mutation #2 (kept):** Mutate `GTPHandler.handleLoadSGF` to skip pass moves (`if move.isPass { continue }`).

Both mutations are applied in Task 8 and reverted before commit.

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Sources/GTPRunner/main.swift` | Create | stdin → `GTPHandler` → stdout |
| `Package.swift` | Modify | Register `GTPRunner` executable target + product |
| `Scripts/SGFFixtureDrivers/empty.gtp` | Create | Driver: empty 19×19 board |
| `Scripts/SGFFixtureDrivers/moves_basic.gtp` | Create | Driver: 4 alternating plays |
| `Scripts/SGFFixtureDrivers/pass_midgame.gtp` | Create | Driver: plays + W pass + plays |
| `Scripts/SGFFixtureDrivers/handicap_5.gtp` | Create | Driver: `fixed_handicap 5` |
| `Scripts/SGFFixtureDrivers/komi_nondefault.gtp` | Create | Driver: `komi 6.5` + plays |
| `Scripts/SGFFixtureDrivers/rules_chinese.gtp` | Create | Driver: `kata-set-rules chinese` + plays |
| `Scripts/SGFFixtureDrivers/captures.gtp` | Create | Driver: 1-stone capture sequence |
| `Scripts/generate_sgf_interop_fixtures.sh` | Create | Generator script (runs GTPRunner + KataGo) |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/<scenario>.export.sgf` | Create (×7) | Direction 1 expected |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/<scenario>.katago.sgf` | Create (×7) | Direction 2 input |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/<scenario>.import.sgf` | Create (×7) | Direction 2 expected |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift` | Create | 14 `@Test` cases |

---

## Task 1: Add the `GTPRunner` executable target

**Files:**
- Create: `Sources/GTPRunner/main.swift`
- Modify: `Package.swift`

- [ ] **Step 1: Create `Sources/GTPRunner/main.swift`.**

```swift
import Foundation
import KataGoOnAppleSilicon

let katago = KataGoInference()
let handler = GTPHandler(katago: katago)

while let line = readLine(strippingNewline: true) {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty { continue }
    let response = handler.handleCommand(trimmed)
    print(response, terminator: "")
    if trimmed == "quit" { break }
}
```

- [ ] **Step 2: Update `Package.swift`.**

Insert the executable product into the existing `products:` array (right after the `KataGoPlay` product line), and the executable target into `targets:` (right after the `KataGoPlay` target):

```swift
.executable(name: "GTPRunner", targets: ["GTPRunner"]),
```

```swift
.executableTarget(
    name: "GTPRunner",
    dependencies: ["KataGoOnAppleSilicon"],
    path: "Sources/GTPRunner"
),
```

The full `products:` list now has three entries (`library`, `KataGoPlay`, `GTPRunner`); the `targets:` list now has five entries (lib, `KataGoPlay`, `GTPRunner`, two test targets).

- [ ] **Step 3: Build to verify the target compiles.**

Run: `swift build --target GTPRunner`
Expected: clean build, no errors.

- [ ] **Step 4: Smoke-test the binary against `list_commands`.**

Run:
```bash
printf 'list_commands\nquit\n' | swift run --quiet GTPRunner
```
Expected (the `=` line lists all known commands; the trailing `=` is the `quit` ack):
```
= protocol_version
name
version
known_command
list_commands
boardsize
clear_board
komi
play
genmove
undo
fixed_handicap
set_free_handicap
kata-set-rules
showboard
kata-rawnn
final_score
loadsgf
printsgf
quit

= 

```

- [ ] **Step 5: Smoke-test `printsgf` to verify file-write form works.**

Run:
```bash
TMP=$(mktemp); printf 'boardsize 19\nkomi 7.5\nprintsgf %s\nquit\n' "$TMP" | swift run --quiet GTPRunner > /dev/null && cat "$TMP"
```
Expected:
```
(;FF[4]GM[1]SZ[19]PB[Black]PW[White]KM[7.5])
```

- [ ] **Step 6: Commit.**

```bash
git add Package.swift Sources/GTPRunner/main.swift
git commit -m "feat: add GTPRunner executable for stdin-driven GTP scripting"
```

---

## Task 2: Add SGF fixture driver `.gtp` files

**Files:**
- Create: `Scripts/SGFFixtureDrivers/empty.gtp`
- Create: `Scripts/SGFFixtureDrivers/moves_basic.gtp`
- Create: `Scripts/SGFFixtureDrivers/pass_midgame.gtp`
- Create: `Scripts/SGFFixtureDrivers/handicap_5.gtp`
- Create: `Scripts/SGFFixtureDrivers/komi_nondefault.gtp`
- Create: `Scripts/SGFFixtureDrivers/rules_chinese.gtp`
- Create: `Scripts/SGFFixtureDrivers/captures.gtp`

Each file describes only the position setup. The generator appends `printsgf <tmp>` and `quit` itself.

- [ ] **Step 1: `empty.gtp` — header round-trip baseline.**

```
boardsize 19
komi 7.5
```

- [ ] **Step 2: `moves_basic.gtp` — 4 alternating plays.**

```
boardsize 19
komi 7.5
play B D4
play W Q16
play B Q4
play W D16
```

- [ ] **Step 3: `pass_midgame.gtp` — plays + W pass + plays.**

```
boardsize 19
komi 7.5
play B D4
play W pass
play B Q16
play W Q4
```

- [ ] **Step 4: `handicap_5.gtp` — fixed handicap.**

```
boardsize 19
fixed_handicap 5
```

(No `komi` — fixed_handicap traditionally pairs with reduced komi via convention, but our handler sets komi independently. We rely on the default komi 7.5. Both engines agree.)

- [ ] **Step 5: `komi_nondefault.gtp` — KM[6.5] formatting.**

```
boardsize 19
komi 6.5
play B D4
play W Q16
```

- [ ] **Step 6: `rules_chinese.gtp` — RU[Chinese] round-trip.**

```
boardsize 19
komi 7.5
kata-set-rules chinese
play B D4
play W Q16
```

- [ ] **Step 7: `captures.gtp` — 1-stone capture sequence.**

```
boardsize 19
komi 7.5
play B Q3
play W Q4
play B P4
play W S5
play B R4
play W S6
play B Q5
```

After the seven moves, W Q4 has neighbours Q3 (B), P4 (B), R4 (B), Q5 (B) and is alone (no friendly white stone adjacent). All liberties are filled by black stones, so W Q4 is captured. The capture exercises board-state replay rather than just move-list copying.

W S5 and W S6 are "tenuki" moves placed far from the capture so they have no interaction with the Q4 group.

Sanity-check the capture using the in-tree handler before relying on the driver:
```bash
printf 'boardsize 19\nkomi 7.5\nplay B Q3\nplay W Q4\nplay B P4\nplay W S5\nplay B R4\nplay W S6\nplay B Q5\nshowboard\nquit\n' | swift run --quiet GTPRunner
```
Expected: `showboard` output shows the Q4 intersection empty (W stone has been captured). If Q4 still shows white, the capture didn't fire — investigate before continuing.

- [ ] **Step 8: Verify the directory listing.**

Run: `ls Scripts/SGFFixtureDrivers/`
Expected: 7 files, alphabetically: `captures.gtp empty.gtp handicap_5.gtp komi_nondefault.gtp moves_basic.gtp pass_midgame.gtp rules_chinese.gtp`.

- [ ] **Step 9: Commit.**

```bash
git add Scripts/SGFFixtureDrivers
git commit -m "test: add SGF fixture driver scripts for cross-engine interop"
```

---

## Task 3: Implement the generator script

**Files:**
- Create: `Scripts/generate_sgf_interop_fixtures.sh`

The script delegates KataGo build/discovery to `generate_kata_raw_nn_reference.sh` (existing pattern), then runs the four-step flow per scenario.

- [ ] **Step 1: Create the script.**

```bash
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
```

- [ ] **Step 2: Make the script executable.**

Run: `chmod +x Scripts/generate_sgf_interop_fixtures.sh`

- [ ] **Step 3: Smoke-test on a single scenario.**

Run: `Scripts/generate_sgf_interop_fixtures.sh --scenario empty`
Expected: green "Generating empty..." line, then three `->` paths printed, exits 0.

- [ ] **Step 4: Verify the three fixture files exist with expected shapes.**

Run:
```bash
ls Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.*.sgf
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.export.sgf
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.katago.sgf
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.import.sgf
```
Expected (exact bytes):
```
empty.export.sgf:  (;FF[4]GM[1]SZ[19]PB[Black]PW[White]KM[7.5])
empty.katago.sgf:  (;FF[4]GM[1]SZ[19]PB[]PW[]HA[0]KM[7.5]RU[TrompTaylor])
empty.import.sgf:  (;FF[4]GM[1]SZ[19]PB[]PW[]KM[7.5]RU[TrompTaylor])
```

(`empty.import.sgf` differs from `empty.katago.sgf` in that we don't emit `HA[0]` — our SGFGenerator only emits `HA` when `handicapBlack` is non-empty, and KataGo's empty `HA[0]` has no `AB` group so our parser produces an empty handicap list.)

If the actual `import.sgf` differs from this expectation, that's a finding — investigate before proceeding. Possible causes: SGFParser handles `HA[0]` differently than expected, or KataGo's emission has changed. Document the actual output and adjust this expectation; the test in Task 7 will cement whatever the generator produces.

- [ ] **Step 5: Clean up the smoke-test fixtures (so Task 4 generates fresh).**

Run: `rm Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.*.sgf`

- [ ] **Step 6: Commit.**

```bash
git add Scripts/generate_sgf_interop_fixtures.sh
git commit -m "test: add generator script for SGF interop fixtures"
```

---

## Task 4: Run the generator and commit the fixture corpus

**Files:**
- Create: `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/<7 scenarios>.{export,katago,import}.sgf` (21 files)

- [ ] **Step 1: Run the generator for all scenarios.**

Run: `Scripts/generate_sgf_interop_fixtures.sh`
Expected: green "Generating <name>..." for each of the 7 scenarios with `->` lines, then `All requested fixtures generated.`. Exits 0.

- [ ] **Step 2: Verify the corpus.**

Run: `ls Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/ | wc -l`
Expected: `21` (7 scenarios × 3 files each).

Run: `ls Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/`
Expected (alphabetical):
```
captures.export.sgf
captures.import.sgf
captures.katago.sgf
empty.export.sgf
empty.import.sgf
empty.katago.sgf
handicap_5.export.sgf
handicap_5.import.sgf
handicap_5.katago.sgf
komi_nondefault.export.sgf
komi_nondefault.import.sgf
komi_nondefault.katago.sgf
moves_basic.export.sgf
moves_basic.import.sgf
moves_basic.katago.sgf
pass_midgame.export.sgf
pass_midgame.import.sgf
pass_midgame.katago.sgf
rules_chinese.export.sgf
rules_chinese.import.sgf
rules_chinese.katago.sgf
```

- [ ] **Step 3: Spot-check a few non-trivial fixtures.**

Run:
```bash
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/handicap_5.export.sgf
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/pass_midgame.export.sgf
cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/rules_chinese.export.sgf
```

Expected shapes (exact stones may vary by KataGo's `fixed_handicap` placement, but format must hold):
- `handicap_5.export.sgf`: contains `HA[5]AB[…][…][…][…][…]`, no `;B[…]` moves, no `PL[…]`.
- `pass_midgame.export.sgf`: contains `;B[…];W[];B[…];W[…]` (note the empty `W[]`).
- `rules_chinese.export.sgf`: contains `RU[Chinese]`.

If any spot-check fails, the issue is upstream (driver wrong, or our printsgf not honouring the feature). Do not paper over by adjusting fixtures — fix the upstream.

- [ ] **Step 4: Stage and commit the corpus.**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures
git commit -m "test: commit SGF interop fixture corpus (7 scenarios × 3 files)"
```

---

## Task 5: Add `SGFInteropTests.swift` skeleton with helpers

**Files:**
- Create: `Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift`

The file is created with helpers and one `@Test` (the simplest scenario, `empty`) for both directions, so we have an end-to-end sanity test before adding the rest.

- [ ] **Step 1: Create the file.**

```swift
import Testing
import Foundation
@testable import KataGoOnAppleSilicon

/// Cross-engine SGF interoperability tests for `printsgf` and `loadsgf`.
///
/// Each scenario has three committed fixture files under `SGFFixtures/`:
///   - `<scenario>.export.sgf`  — our printsgf when driving the position natively
///   - `<scenario>.katago.sgf`  — KataGo's printsgf at the same position
///   - `<scenario>.import.sgf`  — our printsgf after loadsgf'ing the KataGo SGF
///
/// `interop_export_<scenario>` drives the scenario via Swift's GTPHandler
/// then byte-compares our printsgf to `<scenario>.export.sgf`.
///
/// `interop_import_<scenario>` loadsgf's `<scenario>.katago.sgf` then
/// byte-compares our printsgf to `<scenario>.import.sgf`.
///
/// Fixtures are produced offline by `Scripts/generate_sgf_interop_fixtures.sh`.
/// Missing fixtures throw `InteropError.fixtureMissing`.
struct SGFInteropTests {

    enum InteropError: Error, CustomStringConvertible {
        case fixtureMissing(String)
        var description: String {
            switch self {
            case .fixtureMissing(let path):
                return """
                Fixture not found: \(path)
                Run: Scripts/generate_sgf_interop_fixtures.sh
                """
            }
        }
    }

    // MARK: - Helpers

    /// Walks up from this source file to the repo root, then descends into
    /// `subdir/name`. Mirrors the pattern used by GTPFixtureTests so the
    /// fixture lookup logic is consistent across integration tests.
    private func repoFile(subdir: String, name: String) throws -> URL {
        let fm = FileManager.default
        var here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = here
                .appendingPathComponent(subdir, isDirectory: true)
                .appendingPathComponent(name)
            if fm.fileExists(atPath: candidate.path) { return candidate }
            here.deleteLastPathComponent()
        }
        throw InteropError.fixtureMissing("\(subdir)/\(name)")
    }

    private func loadFixture(_ name: String) throws -> String {
        let url = try repoFile(
            subdir: "Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures",
            name: name
        )
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func fixtureURL(_ name: String) throws -> URL {
        try repoFile(
            subdir: "Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures",
            name: name
        )
    }

    private func loadDriver(_ name: String) throws -> String {
        let url = try repoFile(subdir: "Scripts/SGFFixtureDrivers", name: name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Strip the GTP `= ` prefix and trailing `\n\n` to recover the SGF body.
    private func sgfPayload(_ response: String) -> String? {
        guard response.hasPrefix("= ") else { return nil }
        var body = String(response.dropFirst(2))
        if body.hasSuffix("\n\n") {
            body = String(body.dropLast(2))
        }
        return body
    }

    /// Side-by-side byte dump for a `#expect` failure message. Returns
    /// `String` rather than `Testing.Comment`; mirrors the convention
    /// already used in `GTPFixtureTests` (call sites wrap with
    /// `"\(diffMessage(...))"` so the literal string interpolation
    /// satisfies `Comment`'s `ExpressibleByStringInterpolation`).
    private func diffMessage(actual: String, expected: String, label: String) -> String {
        return """
        \(label) mismatch.
        --- actual (\(actual.count) bytes) ---
        \(actual)
        --- expected (\(expected.count) bytes) ---
        \(expected)
        """
    }

    /// Replay a driver's GTP commands through a fresh handler, then return
    /// the body of our `printsgf` reply.
    private func driveAndPrint(driver: String) throws -> String {
        let handler = GTPHandler(katago: KataGoInference())
        for raw in driver.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            _ = handler.handleCommand(line)
        }
        return try #require(sgfPayload(handler.handleCommand("printsgf")))
    }

    // MARK: - Tests

    @Test func interop_export_empty() throws {
        let expected = try loadFixture("empty.export.sgf")
        let driver = try loadDriver("empty.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_empty"))")
    }

    @Test func interop_import_empty() throws {
        let expected = try loadFixture("empty.import.sgf")
        let engineSGFURL = try fixtureURL("empty.katago.sgf")

        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_empty"))")
    }
}
```

Notes:
- `SGFFixtures/<name>.sgf` files committed under the test target's directory but NOT registered as `Package.swift` resources. The test reads them via the walk-up-tree helper from the repo, mirroring `GTPFixtureTests` (which reads under `Tests/.../ReferenceOutputs/` the same way). This is intentional — see spec §"Fixture discovery — committed, not bundled."
- `loadFixture` and `loadDriver` differ only in `subdir`. They're separate so call sites read clearly at the test site.
- `#expect`'s second argument is `Testing.Comment?`. A runtime `String` doesn't auto-convert, but a string-interpolation literal at the call site (`"\(...)"`) does — so call sites wrap the helper as `"\(diffMessage(...))"`. Mirrors the pattern in `GTPFixtureTests.swift` (`#expect(diff.isEmpty, "\(diff)")`).
- `Stone` and other types referenced indirectly via `GTPHandler` are imported through `@testable import KataGoOnAppleSilicon`.

- [ ] **Step 2: Build to verify the file compiles.**

Run: `swift build --target KataGoOnAppleSiliconIntegrationTests`
Expected: clean build.

- [ ] **Step 3: Run the two empty-board tests.**

Run: `swift test --filter SGFInteropTests`
Expected: 2 tests, 2 passed.

If either fails, the `expected` byte string in Task 3 Step 4 disagrees with what the generator actually produced. Read the failing diff in the test output, update the fixture file expectations in this plan if appropriate (or fix the upstream issue if a real bug surfaced), and re-run.

- [ ] **Step 4: Commit.**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift
git commit -m "test: SGFInteropTests skeleton with empty-board export/import cases"
```

---

## Task 6: Add export-direction tests for the remaining 6 scenarios

**Files:**
- Modify: `Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift`

Add one `@Test func interop_export_<scenario>()` per remaining scenario, in alphabetical order (matching the file listing). Each test follows the `empty` template exactly — only the fixture and driver names change.

- [ ] **Step 1: Append the six new tests.**

Insert these `@Test` functions immediately after `interop_export_empty` (and before `interop_import_empty`) in `SGFInteropTests.swift`:

```swift
    @Test func interop_export_captures() throws {
        let expected = try loadFixture("captures.export.sgf")
        let driver = try loadDriver("captures.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_captures"))")
    }

    @Test func interop_export_handicap_5() throws {
        let expected = try loadFixture("handicap_5.export.sgf")
        let driver = try loadDriver("handicap_5.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_handicap_5"))")
    }

    @Test func interop_export_komi_nondefault() throws {
        let expected = try loadFixture("komi_nondefault.export.sgf")
        let driver = try loadDriver("komi_nondefault.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_komi_nondefault"))")
    }

    @Test func interop_export_moves_basic() throws {
        let expected = try loadFixture("moves_basic.export.sgf")
        let driver = try loadDriver("moves_basic.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_moves_basic"))")
    }

    @Test func interop_export_pass_midgame() throws {
        let expected = try loadFixture("pass_midgame.export.sgf")
        let driver = try loadDriver("pass_midgame.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_pass_midgame"))")
    }

    @Test func interop_export_rules_chinese() throws {
        let expected = try loadFixture("rules_chinese.export.sgf")
        let driver = try loadDriver("rules_chinese.gtp")
        let actual = try driveAndPrint(driver: driver)
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_export_rules_chinese"))")
    }
```

- [ ] **Step 2: Run all export tests.**

Run: `swift test --filter SGFInteropTests/interop_export`
Expected: 7 tests, 7 passed.

If any fails, the diff will pinpoint the discrepancy between our `printsgf` and what was committed. Investigate; do not edit the fixture to make the test pass — that hides the bug.

- [ ] **Step 3: Commit.**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift
git commit -m "test: cover remaining SGF interop export scenarios"
```

---

## Task 7: Add import-direction tests for the remaining 6 scenarios

**Files:**
- Modify: `Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift`

Add one `@Test func interop_import_<scenario>()` per remaining scenario, mirroring the export pattern.

- [ ] **Step 1: Append the six new tests.**

Insert these immediately after `interop_import_empty`:

```swift
    @Test func interop_import_captures() throws {
        let expected = try loadFixture("captures.import.sgf")
        let engineSGFURL = try fixtureURL("captures.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_captures"))")
    }

    @Test func interop_import_handicap_5() throws {
        let expected = try loadFixture("handicap_5.import.sgf")
        let engineSGFURL = try fixtureURL("handicap_5.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_handicap_5"))")
    }

    @Test func interop_import_komi_nondefault() throws {
        let expected = try loadFixture("komi_nondefault.import.sgf")
        let engineSGFURL = try fixtureURL("komi_nondefault.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_komi_nondefault"))")
    }

    @Test func interop_import_moves_basic() throws {
        let expected = try loadFixture("moves_basic.import.sgf")
        let engineSGFURL = try fixtureURL("moves_basic.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_moves_basic"))")
    }

    @Test func interop_import_pass_midgame() throws {
        let expected = try loadFixture("pass_midgame.import.sgf")
        let engineSGFURL = try fixtureURL("pass_midgame.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_pass_midgame"))")
    }

    @Test func interop_import_rules_chinese() throws {
        let expected = try loadFixture("rules_chinese.import.sgf")
        let engineSGFURL = try fixtureURL("rules_chinese.katago.sgf")
        let handler = GTPHandler(katago: KataGoInference())
        #expect(handler.handleCommand("loadsgf \(engineSGFURL.path)") == "= \n\n")
        let actual = try #require(sgfPayload(handler.handleCommand("printsgf")))
        #expect(actual == expected, "\(diffMessage(actual: actual, expected: expected, label: "interop_import_rules_chinese"))")
    }
```

- [ ] **Step 2: Run all interop tests.**

Run: `swift test --filter SGFInteropTests`
Expected: 14 tests, 14 passed.

- [ ] **Step 3: Commit.**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift
git commit -m "test: cover remaining SGF interop import scenarios"
```

---

## Task 8: Verify acceptance criteria via mutation testing

This task makes no permanent code changes — every mutation is reverted before the next step. The purpose is to confirm the test suite catches the bug classes the spec promised.

- [ ] **Step 1: Mutation 1 — break `HA[…]AB[…]` emission for handicap.**

Edit `Sources/KataGoOnAppleSilicon/Core/SGFGenerator.swift`. Find the block:

```swift
if !handicapBlack.isEmpty {
    sgf += "HA[\(handicapBlack.count)]"
    sgf += "AB"
    for p in handicapBlack { sgf += "[\(pointToSgf(p))]" }
}
```

Change the guard to `if false {` (so the handicap block is never emitted).

- [ ] **Step 2: Run handicap tests; expect both to fail.**

Run: `swift test --filter SGFInteropTests/interop_export_handicap_5`
Expected: FAIL. The diff shows actual SGF missing `HA[5]AB[…]…`.

Run: `swift test --filter SGFInteropTests/interop_import_handicap_5`
Expected: FAIL. Same shape diff.

- [ ] **Step 3: Confirm no other tests fail from this mutation.**

Run: `swift test --filter SGFInteropTests`
Expected: 14 run, exactly 2 failures (the two `handicap_5` tests). Other 12 pass.

If any *non-handicap* test also fails, the mutation is scoped wrong or fixtures were generated incorrectly — investigate before reverting.

- [ ] **Step 4: Revert mutation 1.**

Restore the original `if !handicapBlack.isEmpty {` guard.

Run: `swift test --filter SGFInteropTests`
Expected: 14 passed (back to green).

- [ ] **Step 5: Mutation 2 — break pass-move replay in `loadsgf`.**

Edit `Sources/KataGoOnAppleSilicon/GTPHandler.swift`, in `handleLoadSGF`. Find the loop:

```swift
for i in 0..<target {
    let move = parsed.moves[i]
    if move.isPass {
        _ = newBoard.playPass(stone: move.player)
    } else if let loc = move.location {
        ...
    }
}
```

Change the body to skip passes:

```swift
for i in 0..<target {
    let move = parsed.moves[i]
    if move.isPass {
        continue
    }
    if let loc = move.location {
        ...
    }
}
```

- [ ] **Step 6: Run pass tests; expect both to fail.**

Run: `swift test --filter SGFInteropTests/interop_import_pass_midgame`
Expected: FAIL. The diff shows fewer moves in the round-tripped SGF.

Run: `swift test --filter SGFInteropTests/interop_export_pass_midgame`
Expected: PASS. (The export direction never goes through `loadsgf`; only the import direction is affected by this mutation.)

If the export test also fails, the mutation has wider effect than intended — investigate.

- [ ] **Step 7: Confirm scope of breakage.**

Run: `swift test --filter SGFInteropTests`
Expected: 14 run, exactly 1 failure (`interop_import_pass_midgame`). Note: this revises the original spec criterion #4 ("breaks exactly the two pass_midgame tests") — only the *import* test is sensitive to a load-side bug, which is more accurate.

- [ ] **Step 8: Revert mutation 2.**

Restore the original pass-handling code.

Run: `swift test --filter SGFInteropTests`
Expected: 14 passed.

- [ ] **Step 9: Run the full integration suite as a final sanity check.**

Run: `swift test --filter KataGoOnAppleSiliconIntegrationTests`
Expected: all integration tests pass (existing suites + new 14).

- [ ] **Step 10: Confirm nothing is left dirty in the working tree.**

Run: `git status`
Expected: clean (all mutations reverted, all new files committed).

If `git status` shows modifications, those are remnants of mutation testing that weren't reverted properly. Use `git diff` to inspect, then `git checkout -- <file>` to restore. Do not commit mutation changes.

---

## Self-review

**Spec coverage:**
- Goal (1 export + 1 import direction): Tasks 6 + 7. ✓
- File layout: Tasks 1, 2, 3, 4, 5. ✓ (revised: 3 SGFs per scenario instead of 2; rationale documented in Watchpoint resolution.)
- GTPRunner executable: Task 1. ✓
- Generator script (4 phases, --scenario, --force-rebuild, build delegation, KataGo flag parity): Task 3. ✓
- Test file with helpers + skip-if-missing: Task 5. ✓
- Test matrix (7 scenarios × 2 directions): Tasks 6 + 7. ✓
- Acceptance criteria (mutation tests): Task 8. ✓ (mutation targets revised; rationale documented.)
- Watchpoints: Resolved at the top of this plan. ✓

**Placeholder scan:** None. No "TBD", "TODO", "implement later", "similar to". Every code block is complete.

**Type consistency:**
- `loadFixture(name) -> String`, `loadDriver(name) -> String`, `fixtureURL(name) -> URL`, `repoFile(subdir:name:) -> URL` — used consistently across Tasks 5, 6, 7.
- `sgfPayload(_:) -> String?` — used consistently.
- `diffMessage(actual:expected:label:) -> Comment` — defined in Task 5, used identically in Tasks 5, 6, 7.
- `driveAndPrint(driver:) -> String throws` — defined in Task 5, used in Task 6.
- Test names: `interop_export_<scenario>` and `interop_import_<scenario>` consistent.

**Scope check:** This plan produces working, testable software on its own. It depends on a built KataGo (already present in the repo) and the AI Core ML model (already present). No external prerequisites.

**Ambiguity check:** Three potential ambiguities resolved inline:
1. Mutation #1's exact code change (full block to alter, exact text).
2. Mutation #2's exact loop structure (full before/after).
3. Expected output for the empty-board fixtures (exact bytes, with explanation of HA[0] discrepancy).

The `import.sgf` byte expectations for non-empty scenarios are intentionally not pre-specified in this plan — they depend on KataGo's exact emission for each scenario, which is what the generator captures. Task 4's spot-checks pin format-level expectations (presence/absence of HA, RU, pass `B[]`); byte-level expectations are committed as fixture files in Task 4 and become the ground truth for Tasks 5–7.
