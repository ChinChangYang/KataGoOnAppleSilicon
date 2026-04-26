# KataGoPlay GTP Coverage — Design

**Date:** 2026-04-26
**Status:** Approved (pending spec review)
**Scope:** Make KataGoPlay exercise every GTP command currently supported by `GTPHandler`, and close the one parameter-coverage gap. Do NOT add new GTP commands to the engine.

## Motivation

`GTPHandler` supports 18 commands. KataGoPlay only issues 7 of them. The other 11 are tested only via `GTPHandlerTests` and `GTPFixtureTests`. Because KataGoPlay is the project's user-facing tool — and effectively serves as an end-to-end test of the GTP layer — the unused commands have no exercised path through the play tool. Exposing them through the REPL closes that gap.

## Audit

### Supported commands in `GTPHandler.handleCommand` (18)

`protocol_version`, `name`, `version`, `known_command`, `list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `kata-set-rules`, `genmove`, `undo`, `fixed_handicap`, `set_free_handicap`, `showboard`, `kata-rawnn`, `final_score`, `quit`

### Used by KataGoPlay today (7)

| Command | Where |
|---|---|
| `boardsize` | `main.swift:55` (setup only) |
| `komi` | `main.swift:56` (setup only) |
| `showboard` | `main.swift:22` |
| `genmove` | `main.swift:77, 119, 148, 212` |
| `play` | `main.swift:110, 143` |
| `kata-rawnn` | `main.swift:29, 169, 183` (always symmetry `0`) |
| `final_score` | `main.swift:125, 156` |

### Unused by KataGoPlay (11)

- **Introspection (5):** `protocol_version`, `name`, `version`, `known_command`, `list_commands`
- **State management (3):** `clear_board`, `undo`, `quit` (KataGoPlay calls `exit(0)` directly)
- **Game configuration (3):** `kata-set-rules`, `fixed_handicap`, `set_free_handicap`

### Parameter gaps in used commands

- `kata-rawnn` — only ever called with symmetry `0`; the handler accepts `0`–`7`.
- `boardsize` / `komi` — only set once at setup, never re-issued.
- `play` — already exercises both colors, coord and `pass`.
- `genmove` — already exercises both colors via the game loop.

## Design

### REPL verbs → GTP commands

Mix approach: user-meaningful commands get dedicated REPL verbs; the introspection commands are bundled behind one `info` verb.

| New / changed REPL verb | GTP emitted | Notes |
|---|---|---|
| `new` | `clear_board` | Resets `moveHistory`, `lastAIMove`. Keeps board size, komi, profile, human color. If human is White, AI plays first move (mirrors current setup behavior in `main.swift:75-85`). |
| `undo` | `undo` | One ply. Pops `moveHistory.last`; recomputes `lastAIMove` from new tail. Surfaces `cannot undo` errors. |
| `info` | `protocol_version`, `name`, `version`, `list_commands` (issued in sequence) | Prints each labeled. Covers four commands at once. |
| `known <cmd>` | `known_command <cmd>` | Prints `true` / `false`. |
| `handicap <N>` | `fixed_handicap N` | Empty board only. See "Handicap flow" below. |
| `free-handicap <coord>…` | `set_free_handicap <coord>…` | Empty board only. See "Handicap flow" below. |
| `rules chinese` | `kata-set-rules chinese` | Only `chinese` is currently accepted by the handler (`GTPHandler.swift:171`). Verb mirrors that. |
| `size <N>` | `boardsize N` | Equivalent to "new game with different size". Resets `moveHistory`/`lastAIMove`; rebinds `boardSize` so renderers use the new size. |
| `komi <X>` | `komi X` | Allowed mid-game (handler permits it). |
| `quit` | `quit` (then `exit(0)`) | Currently calls `exit(0)` directly; route through `handleCommand("quit")` first so the command is exercised. |
| `hint [sym]` / `analysis [sym]` | `kata-rawnn <sym>` | Optional symmetry argument `0`–`7`; defaults to `0`. Closes the `kata-rawnn` parameter gap. |

After this change, every supported GTP command is reachable from the REPL.

### Handicap flow

When the user types `handicap N` or `free-handicap …`:

1. Issue `fixed_handicap N` / `set_free_handicap …` to the handler.
2. Handler places stones (always Black's, per GTP `fixed_handicap`/`set_free_handicap` semantics in `GTPHandler.swift:294-352`).
3. Do NOT mutate `humanColor` — keep whatever the human picked at setup.
4. After placement, White moves next (standard Go convention).
5. Branch on existing `humanColor`:
   - human = Black → AI is White → issue `genmove white` immediately.
   - human = White → human is to move → just prompt them; do not auto-genmove.
6. Append handicap stones to `moveHistory` as Black moves so SGF export stays correct.

### State management changes in `main.swift`

The local game-loop state must stay consistent with handler state across `new`, `undo`, `size`, `handicap`, `free-handicap`, `komi`:

- `moveHistory: [(Stone, String)]` — already `var`, append/clear/pop as appropriate.
- `lastAIMove: String?` — already `var`, recompute on `new`/`undo`/`size`.
- `boardSize` — currently `let` from setup; promote to `var` so `size <N>` can rebind it. The renderer and analysis helpers all take `boardSize` as a parameter, so threading the new value is mechanical.

`humanColor` and its derived names (`humanName`, `aiName`, `humanGTPStr`, `aiGTPStr`) stay `let` constants from setup. No verb changes them.

### `CommandParser` extensions

Extend the `UserCommand` enum and `parse(_:)` switch in `CommandParser.swift` with cases for the new verbs:

- `new`, `undo`, `info`, `quit-via-gtp` — token-only matches.
- `known <cmd>` — one string argument.
- `handicap <N>` — one int argument.
- `freeHandicap <coord>…` — variadic coord list.
- `rules <preset>` — one string argument.
- `size <N>` — one int argument.
- `komi <X>` — one float argument.
- `hint`/`analysis` — accept optional trailing int (symmetry); existing zero-arg form keeps default `0`.

Argument-validation philosophy: parser performs only shape validation (presence of args, integer/float parsing). Value-domain validation (board size range, symmetry range, coord legality) stays with `GTPHandler`, which already returns proper GTP error responses.

### Help text

Extend `helpText` in `main.swift:93-96` to document the new verbs. Two lines instead of one. Keep the wording terse.

### Quit fix

In every path that currently calls `exit(0)` (resign, both-pass, explicit `quit`, model load failure), prepend `_ = gtp.handleCommand("quit")`. This exercises the `quit` GTP command on every program-end path the user can reach from the REPL.

## Tests (minimum)

Per the explicit decision that "KataGoPlay itself is a test of the GTP handler," scaffolding stays minimal:

- New file `Tests/KataGoOnAppleSiliconTests/CommandParserTests.swift` covering parser → `UserCommand` mapping for every new verb and parameter shape:
  - `new`, `undo`, `info`, `quit`
  - `known foo`
  - `handicap 9`, `handicap` (no arg → unknown), `handicap nine` (bad arg → unknown)
  - `free-handicap C3 D4`, `free-handicap` (no args → unknown)
  - `rules chinese`, `rules` (no arg → unknown)
  - `size 13`, `size 0` (parser passes shape; engine rejects value)
  - `komi 6.5`, `komi seven` (bad arg → unknown)
  - `hint`, `hint 3`, `analysis`, `analysis 7`

Approximately 15 short tests. No new test target; lives alongside existing unit tests.

No KataGoPlayTests target. No REPL integration tests. The fact that running KataGoPlay drives every supported GTP command is the integration test.

## Out of scope

- Adding new GTP commands to `GTPHandler` (explicitly excluded by the user).
- Refactoring `GTPHandler` itself.
- Visual changes to the renderer beyond accepting a mutable `boardSize`.
- Mid-game profile-switching beyond what the existing `profile <name>` verb already does.
- Time control, SGF loading, dead-stone marking — not currently supported by the handler.

## Acceptance criteria

- Every command in `GTPHandler.handleCommand`'s switch statement is reachable from a KataGoPlay REPL session.
- `kata-rawnn` is exercised with a non-zero symmetry argument from the REPL.
- A complete REPL session — including `new`, `undo`, `handicap`, `free-handicap`, `rules`, `size`, `komi`, `info`, `known`, `hint <sym>` (with `sym ≠ 0`), and `quit` — runs without crashing and reflects the expected board state via `showboard`.
- `swift test` continues to pass; new `CommandParserTests` pass.
