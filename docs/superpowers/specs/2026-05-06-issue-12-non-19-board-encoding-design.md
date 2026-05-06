# Non-19 board encoding fix (issue #12) — design

## Background

PR #7 introduced a `size N` verb in `KataGoPlay` and a `boardsize N` GTP command for any N in 2–19. The Core ML inference path (`BoardState`, `KataGoInference`) was originally written assuming a fixed 19×19 board and a fixed `[1, 22, 19, 19]` spatial tensor. Issue #12 reports that after `size 9`, commands like `genmove` and `kata-rawnn` continue to feed a 19×19-shaped tensor through a 19×19-trained model — and the user-facing output is rendered as if the board were 19×19 — producing silent failures when the board is smaller.

## Goal

After `boardsize N` (or KataGoPlay `size N`) for any N in 2–19:

1. The spatial tensor `[1, 22, 19, 19]` and the global tensor `[1, 19]` byte-match (within 0.001 relative tolerance) what KataGo C++ produces for the same board state — verified indirectly via post-network output parity (B-lite verification, see "Verification approach" below).
2. User-facing post-network output (`kata-rawnn` policy/ownership grids; KataGoPlay board rendering after a `size` change) reflects the actual played region, not a 19×19 view.

## Non-goals

- Changing the size range accepted by `handleBoardsize` (stays 2–19, as PR #7 set it).
- Changing the size range accepted by KataGoPlay's `.size` verb (stays 2–19).
- Replacing the Core ML model files (still 19×19-trained — `nnXLen = nnYLen = 19`).
- Mid-game / non-empty-board test fixtures (one empty fixture per non-19 size is enough at the encoding-parity bar).
- Byte-level coverage for sizes outside {9, 13, 19} — those work "as well as the encoding allows" but are not asserted against C++ reference.
- Rectangular boards (xSize ≠ ySize). The Swift `Board(size:)` is square-only and stays so. We support N×N for N in 2–19, not N×M.
- Inference-parity fuzzing or extending coverage to per-symmetry or human-SL fixtures for non-19 sizes.
- Any work on issue #13 (rules name reset) — separate branch.

## Verification approach (B-lite)

KataGo's `kata-raw-nn` GTP command does not emit the input feature tensor; it only prints post-network outputs (`whiteWin`, `whiteLoss`, `policy`, `policyPass`, `whiteOwnership`, etc.). We use those post-network outputs as a proxy for encoding parity: because the model is deterministic, if the post-network outputs match within the existing 0.001 relative tolerance, the input tensors were equal up to fp rounding (modulo collisions, which do not occur in practice for board encoding).

A bug strictly inside the encoder will manifest as a tolerance violation in the post-network comparison; a bug strictly inside the post-processing/format helpers will manifest at a different layer (the output text comparison). The verification is therefore indirect for encoding but still strict at the byte level for what the user observes.

## Placement convention (foundational invariant)

KataGo C++ uses `NNPos::locToPos(loc, boardXSize, nnXLen, nnYLen) = y * nnXLen + x` with the played region pinned at the top-left origin of the `nnXLen × nnYLen` tensor. For us, `nnXLen = nnYLen = 19` always (model is fixed-shape); the played board occupies the rectangle `(y, x) ∈ [0, board.ySize) × [0, board.xSize)`; positions outside that rectangle are zero on every spatial plane (plane 0 reads 0, planes 1+ read 0).

Consequence for the policy/ownership tensors that come back from the model: position `(x, y)` of the played board lives at flat index `y * 19 + x` (not `y * board.xSize + x`). Pass is at index `19 * 19 = 361` regardless of board size. Therefore:

- The existing `let modelPassIndex = 19 * 19` in `KataGoInference` is correct and stays.
- The existing `y * 19 + x` strides inside `formatPolicyGridFromPostprocessed` / `formatOwnershipGridFromPostprocessed` are correct and stay.
- What needs to change is the *iteration bounds* over those tensors when the board is smaller than 19: from `0..<19` to `0..<board.ySize` / `0..<board.xSize`.

## Components

### `Sources/KataGoOnAppleSilicon/BoardState.swift` — audit and targeted fix

Inspection suggests every fill function already either zero-fills the full 19×19 grid and writes only into `0..<board.ySize, 0..<board.xSize`, or iterates board-relative structures (move history, ko point, area map, ladder iterator) whose coordinates are already valid for the played region. The audit's job is to verify this for all 22 spatial planes and all 19 global features:

- `fillPlane0OnBoard` — already correct (`(y < board.ySize && x < board.xSize) ? 1.0 : 0.0`).
- `fillPlanes1And2Stones`, `fillPlanes3To5Liberties`, `fillPlanes18And19Area` — zero-fill 19×19, write into `0..<board.ySize, 0..<board.xSize`. Verify.
- `fillPlane6KoBan` — single ko point write at `(ko.y, ko.x)`. Verify ko point cannot have out-of-board coordinates for non-19 boards.
- `fillPlanes9To13History` — writes from `moveHistory[i].location`. Verify locations stored in the history are board-relative (`< board.xSize` etc.).
- `fillPlanes14To17Ladders` — `board.iterLadders` callback. Verify iterator coordinates are board-relative.
- `fillPlane7…`, `fillPlane8…`, `fillPlanes20And21EncoreStones` — Chinese-rules zero planes. No board-size dependence.
- `fillGlobalFeature5Komi`, `fillGlobalFeature18KomiParityWave`, komi clipping radius — already use `board.xSize * board.ySize`. Verify.
- `getKoHash`, `getPassHistoryHashes`, `passWouldEndPhase` — already use `board.xSize`/`board.ySize` and `Board(size: board.xSize)`. Verify the `Board(size:)` reconstruction handles the move-history replay correctly for non-19 boards.

Doc comments stating `[1, 22, 19, 19]` and `[1, 19]` stay as-is — the tensor shapes are literally still those, regardless of played region.

Any genuinely-wrong write found during audit gets fixed in the same change. Otherwise the file is touched only to add a brief comment near `private static func fillSpatialFeatures` clarifying the zero-fill / play-region-fill split.

### `Sources/KataGoOnAppleSilicon/KataGoInference.swift` — output paths

Two helper functions accept `boardSize: Int = 19` but their internal iteration bounds may still read `0..<19` instead of `0..<boardSize`:

- `formatPolicyGridFromPostprocessed(policyProbs:boardSize:)` — fix iteration bounds; keep `y * 19 + x` stride.
- `formatOwnershipGridFromPostprocessed(ownership:boardSize:)` — fix iteration bounds; keep `y * 19 + x` stride.

Audit all callers (the `rawNN()` flow, the `kata-rawnn` handler chain) to thread the actual `board.xSize` through. Drop the `= 19` default on these parameters so missing-pass bugs become compile errors rather than silent wrong-output.

`let modelPassIndex = 19 * 19` is correct (the model output stride is independent of played region). Add a one-line comment so the next reader understands it isn't a hardcoding bug.

### `Sources/KataGoOnAppleSilicon/GTPHandler.swift` — kata-rawnn handler

Only the `kata-rawnn` emission path of policy/ownership grids changes — it passes `board.xSize` to the format helpers above. `handleBoardsize` is unchanged. No new GTP commands.

### `Sources/KataGoPlay/` — no code changes

The `.size` verb in `Sources/KataGoPlay/main.swift` already calls `boardsize N` and re-renders the board. Once inference and formatting are correct for non-19 boards, the renderer (which reads `boardSize` from the local state already updated in `case .size(let n)`) displays the right region.

### `Scripts/generate_kata_raw_nn_reference.sh`

Add a `--board-size N` flag. Internally it issues `boardsize N` to KataGo before `kata-raw-nn 0` (validating N ∈ [2, 19] and rejecting other values with a clear error). Output filename includes the size for non-19 cases:

- `kata_raw_nn_empty_board_symmetry_0.txt` for N=19 (unchanged)
- `kata_raw_nn_empty_board_{N}x{N}_symmetry_0.txt` for N≠19

Update the script's `--help` text. The new fixtures are gitignored project-wide (`.gitignore` rule `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/*.txt`) — they are not committed; the script regenerates them on demand, the same as today's 19×19 fixture. Document the regenerate step in `CLAUDE.md` under "Generate Reference Files" so a developer running the new tests for the first time knows what command to run.

### `Tests/KataGoOnAppleSiliconIntegrationTests/KataRawNNIntegrationTests.swift`

Add two integration test cases mirroring the existing `testKataRawNNEmptyBoard`:

- `testKataRawNNEmptyBoard9x9` — instantiate `Board(size: 9)`, build `BoardState`, call `katago.rawNN(board: board, boardState: boardState, profile: "AI", whichSymmetry: 0)`, load `kata_raw_nn_empty_board_9x9_symmetry_0.txt`, run the existing `compareOutputs(...)` with relative tolerance 0.001.
- `testKataRawNNEmptyBoard13x13` — same for size 13.

The existing comparator is reused (no new comparison logic). The reference policy / ownership grids will be 9×9 (or 13×13); the comparator already handles variable widths by line-tokenized comparison, so as long as our format helpers emit the same number of lines and columns, parity holds.

## Test plan

| Test | What it covers | New / existing |
|------|----------------|----------------|
| `testKataRawNNEmptyBoard` (19×19) | Regression — boardSize-threading must not break the default path | Existing, must continue byte-identical |
| `testKataRawNNEmptyBoard9x9` | Encoding + format parity for 9×9 (B-lite) | New |
| `testKataRawNNEmptyBoard13x13` | Encoding + format parity for 13×13 (B-lite) | New |
| `BoardStateTests`, `BoardTests`, `BoardInitialStateTests`, `GTPHandlerTests`, `GTPFixtureTests`, `GameGeneratorTests`, all other unit tests | No regressions in existing suite | Existing, must continue passing |

A manual KataGoPlay smoke check is part of the implementation plan but not gating: after the fix lands, run `KataGoPlay`, type `size 9`, then `size 13`, and confirm the rendered board and a couple of `genmove`s look sensible. This is sanity, not a CI gate.

No new per-plane unit tests for `BoardState`. The B-lite verification covers the encoder transitively; piling on per-plane unit tests is YAGNI given the tolerance bar.

## Risks

- **The encoder may already be substantially correct**, in which case the only meaningful code change is to the format helpers and the script. That is acceptable — the audit and the new fixtures are still valuable as regression protection.
- **A 19-trained net asked to play on a smaller board may assign nontrivial probability to off-board positions.** `postprocessPolicy` already masks illegal moves via the `Board`. The audit verifies the masking iterates `0..<board.ySize, 0..<board.xSize` rather than `0..<19`. If it does not, that is a downstream fix in the same change.
- **Reference generation stability.** KataGo C++ output is deterministic for a fixed model + fixed inputs, but if the local KataGo build changes (different commit, different flags) the fixtures shift. Mitigation: the script header pins the source commit, same as today's 19×19 fixture flow.
- **Float16 / Float32 rounding noise on smaller boards.** The existing 0.001 relative tolerance has held for 19×19; nothing about smaller boards changes the precision math. Low risk.

## Open questions

None block the spec. Two minor implementation questions resolve naturally during execution:

1. The exact list of 19-hardcoded loop bounds inside `KataGoInference` format helpers — discovered during audit, fixed in the same change.
2. Whether the current `kata-rawnn` GTP handler already threads `boardSize` through — discovered when the audit walks the call chain.
