# GTP Completeness: `undo`, `fixed_handicap`, `set_free_handicap`

Date: 2026-04-21
Status: Draft

## Summary

Add three GTP commands whose behavior matches KataGo's reference
implementation byte-for-byte: `undo`, `fixed_handicap`, and
`set_free_handicap`. Cross-validation uses a parameterized GTP fixture
harness that drives the same command scripts through both KataGo and
Swift and diffs the response streams exactly.

## Goals

- Implement `undo`, `fixed_handicap N`, and `set_free_handicap V…` with
  behavior and error messages identical to KataGo.
- Cross-validate every command through a reference-output harness so
  future regressions surface as byte-level diffs.
- Introduce the minimum shared engine state (`initialStones`,
  `initialSideToMove`, `sideToMove`) required for these commands to
  interact correctly with each other and with existing commands.

## Non-Goals / Deferred

Deferred because KataGo's behavior requires capabilities this engine
does not yet have (MCTS, ownership sampling, Tromp-Taylor
`endAndScoreGameNow` with pass-dead detection, SGF parsing):

- `final_status_list alive|dead|seki` — requires either
  `computeAnticipatedStatusesSimple` (full area scoring with pass-dead
  recognition) or `computeAnticipatedStatusesWithOwnership`
  (100-visit MCTS + ownership head sampling).
- `place_free_handicap` — KataGo's placement strategy is coupled to
  search.
- `loadsgf`, `time_settings`, `time_left`, `reg_genmove`,
  `kgs-genmove_cleanup`.

These may become in scope once MCTS lands. This spec does not pre-commit
their design.

## Background

### Current GTP surface

`GTPHandler` (`Sources/KataGoOnAppleSilicon/GTPHandler.swift`) supports
15 commands: `protocol_version`, `name`, `version`, `known_command`,
`list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `genmove`,
`kata-set-rules`, `showboard`, `kata-rawnn`, `final_score`, `quit`.

`Board` (`Sources/KataGoOnAppleSilicon/Core/Board.swift`) already tracks
`moveHistory: [Move]`, provides `getBoardAtTurn(_:)` that replays from
an empty board, and implements Benson's algorithm for area calculation.

### KataGo reference source

All cross-checks are against the vendored KataGo checkout at
`KataGo-metal-coreml-stable/cpp/`:

- `command/gtp.cpp` — GTP command dispatch and wrapper methods
  (`undo`, `placeFixedHandicap`, `set_free_handicap` branch,
  `setPositionAndRules`, `clearBoard`).
- `program/playutils.cpp` — `PlayUtils::placeFixedHandicap`
  (handicap placement tables).
- `game/board.cpp` — `Board::setStonesFailIfNoLibs`.

### Existing reference-output pattern

`Scripts/generate_kata_raw_nn_reference.sh` already drives a GTP session
through a locally built KataGo and stores the captured output at
`Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/…`. Swift
integration tests load the reference file and diff. We extend this
pattern; we do not replace it.

## Approach

Parameterized GTP fixture harness:

- One fixture = one `.gtp` command script + one `.txt` reference
  response stream.
- One generation script generates any fixture: runs KataGo, pipes the
  `.gtp` file in, captures stdout.
- One Swift test driver replays the same `.gtp` through `GTPHandler`
  and diffs the concatenated responses byte-for-byte.

Rejected: three near-identical per-command shell scripts (drag as
fixtures multiply); a richer generic fixture language (overbuilt for
the current scope — can be introduced later without throwing away this
work).

## Shared Engine Changes

All three commands depend on a small shared addition. These are not
new user-visible commands; they are plumbing that the three new
handlers and the existing `kata-rawnn` / `final_score` commands all
rely on.

### `Board.initialStones` and `Board.initialSideToMove`

`Board` gains two snapshot fields:

- `initialStones: [[Stone]]` — the stone grid at the "start of the
  game" from the engine's point of view. For a fresh board, this is
  all `.empty`. After `fixed_handicap` or `set_free_handicap`, this
  holds the handicap stones.
- `initialSideToMove: Stone` — who moves first from the initial state.
  `.black` by default; `.white` after either handicap command.

These mirror KataGo's `initialBoard` and `initialPla`
(`gtp.cpp:367-368`). They are written by `clear_board`, `boardsize`,
`fixed_handicap`, and `set_free_handicap`. They are read only by
`undo`.

`Board` also gains live-state `Board.sideToMove: Stone`. Lifecycle:

- Initialized to `initialSideToMove` by the `Board(size:)` constructor.
- After every successful `playMove(at:, stone:)` or `playPass(stone:)`,
  `sideToMove = stone.opponent` — i.e. the opponent of whichever color
  just played, regardless of whose "turn" it was. This matches KataGo's
  `makeMove` behavior and correctly handles `play B D4; play B E4`
  (sequential same-color plays through GTP).
- `clearToInitial()` resets it back to `initialSideToMove`.
- `fixed_handicap` and `set_free_handicap`, on success, set
  `sideToMove = .white` (same as `initialSideToMove`).

### `Board.clearToInitial()`

New method that resets the live board to the initial snapshot:

- `stones ← initialStones`
- `koPoint = nil`
- `turnNumber = 0`
- `moveHistory = []`
- `sideToMove = initialSideToMove`

This is the primitive `undo()` uses to rewind before replaying.

### `Board.isEmpty()`

Returns true iff every cell in `stones` is `.empty`. Used by
`fixed_handicap` and `set_free_handicap` for the "Board is not
empty" precondition.

### `Board.copy()` — include new fields

The existing `Board.copy()` is updated to also copy `initialStones`,
`initialSideToMove`, and `sideToMove`. This matters for the
`tryFriendlyPass` code path, which clones the board for a trial
inference; without copying these fields, the clone would start with
default `.empty` / `.black` initial state regardless of any handicap
on the live board.

### Side-to-move fix for `kata-rawnn` and `final_score`

These handlers currently infer next player from `board.turnNumber % 2`.
After `fixed_handicap 4`, `turnNumber == 0` but white moves next —
so the existing heuristic is wrong. Replace
`board.turnNumber % 2 == 0 ? .black : .white` with `board.sideToMove`
in both handlers.

### Writer audit (initial state only)

Only these sites ever write `initialStones` or `initialSideToMove`:

| Site | `initialStones` | `initialSideToMove` |
|------|-----------------|---------------------|
| `Board(size:)` constructor | all `.empty` | `.black` |
| `GTPHandler.handleBoardsize` | via `Board(size:)` replacement | via `Board(size:)` replacement |
| `GTPHandler.handleCommand("clear_board")` | via `Board(size:)` replacement | via `Board(size:)` replacement |
| `Board.placeFixedHandicap` | snapshot of placed stones | `.white` |
| `Board.placeFreeHandicap` | snapshot of placed stones | `.white` |

`handleUndo`, `handlePlay`, and `handleGenmove` do not touch initial
state. They do mutate live `sideToMove` via the rule in the previous
subsection.

## Command: `undo`

### KataGo reference

- `gtp.cpp:2989-2995` — dispatch; returns `? cannot undo` when the
  underlying `engine->undo()` returns false.
- `gtp.cpp:649-670` — `GTPEngine::undo()`:
  1. Returns `false` iff `moveHistory.size() <= 0`.
  2. Copies `moveHistory` locally.
  3. Resets the live state to `initialBoard` + `initialPla`.
  4. Replays every recorded move *except the last* via `play()`.

The crucial detail is step 3: `undo` rewinds to the engine's stored
initial state, not to an empty board. This is the reason handicap
stones survive an `undo`.

### Semantics

- `undo` — no arguments.
- Success: `= \n\n` (empty body). Pops the most recent move from
  `moveHistory`; the board state reflects the replay from
  `initialStones` up to the new tail of `moveHistory`.
- Failure (no moves to undo): `? cannot undo\n\n`.

### Swift design

- New method `Board.undo() -> Bool`:
  1. If `moveHistory.isEmpty`, return `false`.
  2. Let `replay = Array(moveHistory.dropLast())`.
  3. Call `clearToInitial()`.
  4. For each move in `replay`, call `playMove` or `playPass`.
  5. Return `true`.
- New method `GTPHandler.handleUndo()`:
  1. If `board.undo()` returns false → `? cannot undo`.
  2. Recompute `lastPlayPassColor`: if `moveHistory.last?.isPass` is
     true, set to that pass's `player`; otherwise `nil`.
  3. Reset `consecutiveBehindCount = [.black: 0, .white: 0]`. The
     resign streak has no well-defined rewind semantics; clearing
     matches the spirit of "start fresh from the new position".
  4. Return `successResponse()`.
- Register `"undo"` in `knownCommands`.

### Cross-validation fixtures

- `undo_empty.gtp` — single `undo` on a fresh board. Expect
  `? cannot undo`.
- `undo_with_capture.gtp` — five-move sequence whose last move
  captures a stone; `undo` once; `showboard`. Verifies the replay
  correctly rebuilds the captured stone.
- `undo_after_pass.gtp` — play, pass, `undo`, `showboard`. Verifies
  `lastPlayPassColor` rewinds.
- `undo_after_fixed_handicap.gtp` — `fixed_handicap 4; play W D4;
  undo; showboard`. Verifies `initialStones` preservation through
  undo.

## Command: `fixed_handicap N`

### KataGo reference

- `gtp.cpp:3128-3150` — dispatch + validation order.
- `gtp.cpp:1308-1346` — `placeFixedHandicap` (engine-level: set
  `initialBoard`, switch `initialPla = P_WHITE`, emit vertex list).
- `playutils.cpp:300-335` — `PlayUtils::placeFixedHandicap` (board
  dimension checks + coordinate table + per-N placement patterns).

### Validation order and error messages (byte-exact)

1. `pieces.size() != 1` →
   `? Expected one argument for fixed_handicap but got '<joined>'`
2. `N` not an integer →
   `? Could not parse number of handicap stones: '<arg>'`
3. `N < 2` →
   `? Number of handicap stones less than 2: '<arg>'`
4. Board not empty → `? Board is not empty`
5. Either dim `< 7` →
   `? Board is too small for fixed handicap, try place_free_handicap`
6. Either dim is even and `N > 4` →
   `? Fixed handicap > 4 is not allowed on boards with even dimensions, try place_free_handicap`
7. Either dim `== 7` and `N > 4` →
   `? Fixed handicap > 4 is not allowed on boards with size 7, try place_free_handicap`
8. `N > 9` →
   `? Fixed handicap > 9 is not allowed, try place_free_handicap`

### Coordinate table (internal `(x, y)`)

- Edge offset: `size <= 12` → 2 (third line), `size >= 13` → 3
  (fourth line).
- Middle: `size / 2` (integer division).
- Ordered triples: `xCoords = [edge_low, size-1-edge_low, size/2]`,
  same shape for `yCoords`.

### Per-N placement pattern

Exactly the `(xi, yi)` pairs from `playutils.cpp:326-333` (non-monotonic
across N; port literally):

| N | Pairs |
|---|-------|
| 2 | (0,1) (1,0) |
| 3 | (0,1) (1,0) (0,0) |
| 4 | (0,1) (1,0) (0,0) (1,1) |
| 5 | (0,1) (1,0) (0,0) (1,1) (2,2) |
| 6 | (0,1) (1,0) (0,0) (1,1) (0,2) (1,2) |
| 7 | (0,1) (1,0) (0,0) (1,1) (0,2) (1,2) (2,2) |
| 8 | (0,1) (1,0) (0,0) (1,1) (0,2) (1,2) (2,0) (2,1) |
| 9 | (0,1) (1,0) (0,0) (1,1) (0,2) (1,2) (2,0) (2,1) (2,2) |

### Engine state after success (`gtp.cpp:1322-1344`)

- Black stones written onto `initialStones`.
- `initialSideToMove = .white`.
- `moveHistory = []`, `turnNumber = 0`.
- `sideToMove = .white`.
- `setInitialTurnNumber(numStonesOnBoard)` — we ignore (no search,
  no time/temperature heuristics).
- Komi is untouched (user's responsibility per GTP spec).

### Response format (`gtp.cpp:1332-1340`)

Space-separated GTP vertices in `y = 0..<ySize`, `x = 0..<xSize` scan
order, trimmed. Example, `boardsize 19; fixed_handicap 9` →
`D16 K16 Q16 D10 K10 Q10 D4 K4 Q4`.

### Swift design

- New method `Board.placeFixedHandicap(n: Int) throws -> [Point]`:
  1. Validates dimensions and N per the rules above. Throws a typed
     error carrying the exact KataGo error message.
  2. Computes `xCoords` / `yCoords` and the placement pairs.
  3. Writes stones directly into `stones` (does not go through
     `playMove` — handicap placement is not a play).
  4. Snapshots `initialStones` from `stones`; sets
     `initialSideToMove = .white`, `sideToMove = .white`.
  5. Returns points in scan order for the response formatter.
- New method `GTPHandler.handleFixedHandicap(parts: [String])`:
  1. Performs the arg-count / int-parse / `n >= 2` / empty-board
     checks in order.
  2. Calls `Board.placeFixedHandicap`; surfaces thrown errors as the
     corresponding GTP `?` responses.
  3. Formats the returned points via `coordinateToGTP` and returns
     them space-separated.
- Register `"fixed_handicap"` in `knownCommands`.

### Cross-validation fixtures

- `fixed_handicap_2_19.gtp`, `fixed_handicap_9_19.gtp`,
  `fixed_handicap_5_13.gtp` — positive cases (each ends with
  `showboard` to also verify placement).
- `fixed_handicap_err_too_small.gtp` —
  `boardsize 6; fixed_handicap 2`.
- `fixed_handicap_err_even_dim.gtp` —
  `boardsize 8; fixed_handicap 5`.
- `fixed_handicap_err_not_empty.gtp` — `play B D4` first.

## Command: `set_free_handicap V…`

### KataGo reference

- `gtp.cpp:3176-3208` — dispatch, parse, delegate.
- `board.cpp:730-751` — `Board::setStonesFailIfNoLibs`:
  1. Rejects duplicate locations.
  2. Empties all target locations first (safe with pre-existing
     stones; moot for us since we require an empty board).
  3. Places each stone; fails immediately if any placement leaves a
     group with zero liberties.

### Validation order and error messages (byte-exact)

1. Board not empty → `? Board is not empty`.
2. Iterate pieces; for each, attempt to parse as a vertex. On parse
   failure *or* the piece equals `pass` (any case), record the
   message `Invalid handicap location: <piece>`. **KataGo's loop
   overwrites the message on each bad piece, so the reported message
   is for the last bad piece.** We match that behavior.
3. If any bad piece was found → return
   `? Invalid handicap location: <last_bad_piece>`.
4. Apply via `setStonesFailIfNoLibs`. Failure (duplicate or no
   liberties) → `? Handicap placement is invalid`.
5. Success → `= \n\n` (no vertex list, unlike `fixed_handicap`).

### Engine state after success (`gtp.cpp:3202-3206`)

- Black stones written onto `initialStones`.
- `initialSideToMove = .white`, `sideToMove = .white`.
- `moveHistory = []`, `turnNumber = 0`.
- `setInitialTurnNumber(numStonesOnBoard)` — ignored.

### Swift design

- New method
  `Board.setStonesFailIfNoLibs(_ placements: [(Point, Stone)]) -> Bool`:
  1. Reject if any `Point` appears twice in `placements`.
  2. On a copy of `stones`, apply every placement.
  3. For every placed `Point`, verify `liberties(of:) > 0` against the
     updated copy. Return false on any zero-liberty stone.
  4. On success, commit the copy back to `stones`.
- New method `Board.placeFreeHandicap(_ points: [Point]) -> Bool`:
  1. Builds `(Point, .black)` pairs; calls `setStonesFailIfNoLibs`.
  2. On success: `initialStones ← stones`;
     `initialSideToMove = .white`; `sideToMove = .white`;
     `moveHistory = []`; `turnNumber = 0`. Returns true.
- New method `GTPHandler.handleSetFreeHandicap(parts: [String])`:
  1. If `!board.isEmpty()` → `? Board is not empty`.
  2. Parse loop: walk pieces `[1...]`; track a `lastBad: String?`
     — set whenever `parseMove` returns nil or the piece equals
     `"pass"` (case-insensitive). Accumulate successfully parsed
     points.
  3. If `lastBad != nil` → `? Invalid handicap location: <lastBad>`.
  4. Call `board.placeFreeHandicap(points)`. If false →
     `? Handicap placement is invalid`.
  5. Success → `successResponse()` (empty body).
- Register `"set_free_handicap"` in `knownCommands`.

### Cross-validation fixtures

- `set_free_handicap_basic.gtp` — four stones on star points;
  `showboard`.
- `set_free_handicap_undo.gtp` — handicap, play one move, `undo`,
  `showboard`. Stones from the handicap must remain.
- `set_free_handicap_err_pass.gtp` — `pass` among the vertices.
- `set_free_handicap_err_dup.gtp` — same vertex twice.
- `set_free_handicap_err_nolibs.gtp` —
  `boardsize 2; set_free_handicap A1 A2 B1 B2` (fills the 2×2 with
  black; no group liberties).
- `set_free_handicap_err_not_empty.gtp` — play a move first.

## Cross-Validation Harness

### Fixture layout

```
Scripts/
  GTPFixtures/
    undo_empty.gtp
    undo_with_capture.gtp
    undo_after_pass.gtp
    undo_after_fixed_handicap.gtp
    fixed_handicap_2_19.gtp
    fixed_handicap_9_19.gtp
    fixed_handicap_5_13.gtp
    fixed_handicap_err_too_small.gtp
    fixed_handicap_err_even_dim.gtp
    fixed_handicap_err_not_empty.gtp
    set_free_handicap_basic.gtp
    set_free_handicap_undo.gtp
    set_free_handicap_err_pass.gtp
    set_free_handicap_err_dup.gtp
    set_free_handicap_err_nolibs.gtp
    set_free_handicap_err_not_empty.gtp
  generate_gtp_reference.sh
  generate_kata_raw_nn_reference.sh        (unchanged)
Tests/KataGoOnAppleSiliconIntegrationTests/
  ReferenceOutputs/
    <fixture_name>.txt                     (one per fixture)
  GTPFixtureTests.swift                    (new)
  KataRawNNIntegrationTests.swift          (unchanged)
```

### Fixture format

Plain text, one GTP command per line, starting from a clean state.
Fixtures must declare `boardsize` before any state-dependent command.
No comment support is required; KataGo and the Swift driver both
accept only lines that parse as commands.

### Reference generation

`Scripts/generate_gtp_reference.sh` accepts
`--fixture <name> [--fixture <name> ...]` (or, with no args, generates
every fixture under `Scripts/GTPFixtures/`). It reuses the KataGo build
and model download logic already factored (and extracted, if needed,
from `generate_kata_raw_nn_reference.sh`) into a shared helper. For
each fixture, it pipes the `.gtp` file into KataGo's stdin and captures
stdout into
`Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/<name>.txt`.

### Swift test driver

`GTPFixtureTests.swift` exposes one `#Test` per fixture:

```swift
@Test func fixed_handicap_9_19() throws {
    let diff = try runFixture("fixed_handicap_9_19")
    #expect(diff.isEmpty, "Swift output differs from KataGo reference:\n\(diff)")
}
```

Shared helper `runFixture(_:)`:

1. Loads the `.gtp` file and the matching `.txt` reference.
2. Creates a fresh `KataGoInference` + `GTPHandler`.
3. For each command line, calls `handler.handleCommand(line)` and
   appends the response to a buffer.
4. Diffs the buffer against the reference file byte-for-byte; returns
   an empty string on match or a human-readable diff on mismatch.

### What the reference captures

Every response line (`= …` / `? …`) and the blank line that terminates
each response, exactly as KataGo emits them. No prefix filtering — the
Swift driver concatenates in the same way, so the streams line up.

### Interaction with `list_commands`

The addition of `undo`, `fixed_handicap`, and `set_free_handicap`
changes the output of `list_commands`. Fixtures that need to remain
compatible must avoid `list_commands`, or accept that the reference
file will be regenerated whenever a new command is added. The initial
fixture set does not use `list_commands`.

## Unit Tests

Alongside the integration fixtures, the Swift test target gains unit
tests that do not depend on KataGo:

- `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
  - `Board.clearToInitial()` resets state correctly.
  - `Board.undo()` on an empty board returns false.
  - `Board.undo()` after a capture correctly restores the captured
    stone.
  - `Board.undo()` after `placeFixedHandicap` rewinds plays but keeps
    handicap.
  - `Board.setStonesFailIfNoLibs` rejects duplicates and zero-liberty
    placements; accepts valid placements.
  - `Board.placeFixedHandicap(n:)` returns the expected vertex set for
    each N on 19×19 and 13×13.
  - `Board.placeFreeHandicap(_:)` updates `initialStones` and
    `initialSideToMove`.
- Additions to `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`:
  - Each new command appears in `list_commands` and `known_command`.
  - Each documented error path returns the exact KataGo message.
  - `sideToMove`-based `kata-rawnn` / `final_score` gives the correct
    color after `fixed_handicap`.

## Acceptance Criteria

- All cross-validation fixtures listed above exist under
  `Scripts/GTPFixtures/`, their corresponding references exist under
  `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/`, and
  every Swift test in `GTPFixtureTests.swift` passes.
- Unit tests listed above pass.
- `swift build` and `swift test` succeed cleanly.
- `Scripts/generate_gtp_reference.sh` regenerates every fixture
  reproducibly on a clean machine (given the existing KataGo build
  prerequisites).
- README is updated to list the three new GTP commands in the feature
  and command listings.
