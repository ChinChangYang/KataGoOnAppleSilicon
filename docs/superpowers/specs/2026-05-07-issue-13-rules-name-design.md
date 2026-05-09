# Issue #13 — `printsgf` rules-name should follow live engine state

**Status:** design
**Issue:** [#13 — `resetSGFMetadata()` erases `sgfRulesName` set by `kata-set-rules`](https://github.com/ChinChangYang/KataGoOnAppleSilicon/issues/13)
**Branch:** `fix/issue-13-rules-name-reset` (off `master @ 8ff6f42`)

## Problem

After `kata-set-rules chinese` → `clear_board` → `printsgf`, the SGF should include `RU[Chinese]` but doesn't. The board still operates under Chinese rules (carried forward by `makeBoard()`), but `resetSGFMetadata()` unconditionally clears `sgfRulesName` to `nil`.

Reproduction:

```
kata-set-rules chinese
clear_board
printsgf
```

Expected: SGF contains `RU[Chinese]`. Actual: `RU[…]` is omitted.

## Root cause

The Swift port stores SGF metadata in three GTPHandler fields — `blackPlayerName`, `whitePlayerName`, `sgfRulesName` — that exist in parallel to the live engine state. `resetSGFMetadata()` (called from `clear_board` and `boardsize`) wipes all three, even though the active `Rules` object survives via `makeBoard()`. The "what to write into `RU[]`" cache and the "what rules are actually in effect" object can therefore diverge.

## How stock KataGo handles this

Stock KataGo has no separate metadata bag:

- `printsgf` calls `WriteSgf::writeSgf(out, "", "", engine->bot->getRootHist(), …)` — empty `bName`/`wName`. KataGo never round-trips `PB`/`PW` from `loadsgf` to `printsgf`.
- The `RU[…]` value is derived from the live `Rules` object via `rules.toStringNoKomiMaybeNice()`, returning preset aliases (`"Chinese"`, `"Japanese"`, `"AGA"`, …) or a JSON-ish fallback.
- `clearBoard()` rebuilds `BoardHistory` with `currentRules`. Rules persist; nothing else needs resetting.

(References: `KataGo-metal-coreml-stable/cpp/command/gtp.cpp:590,3398` and `cpp/dataio/sgf.cpp:1923`.)

## Decision

Drop the parallel metadata bag. Derive everything `printsgf` emits from live engine state, matching KataGo. The bug then disappears for free, since `Rules` already survives `clear_board`.

## Behavior

| Input sequence | Before | After |
|---|---|---|
| `kata-set-rules chinese` → `clear_board` → `printsgf` | `RU` omitted | `RU[Chinese]` |
| `boardsize 19` → `printsgf` (fresh engine) | `RU` omitted | `RU[Chinese-OGS]` |
| `kata-set-rules chinese` → `boardsize 19` → `printsgf` | `RU` omitted | `RU[Chinese]` |
| `loadsgf game.sgf` (with `PB[Lee]PW[Hikaru]`) → `printsgf` | `PB[Lee]PW[Hikaru]` round-tripped | `PB[]PW[]` |
| `loadsgf game.sgf` (with `RU[Japanese]`) → `printsgf` | `RU[Japanese]` round-tripped (string preserved despite engine falling back to `defaultRules`) | `RU[Chinese-OGS]` (derived from the fallback `defaultRules`) |

The `loadsgf` rows are intentional behavior changes that align with KataGo: SGF metadata is no longer separately stashed, so anything not represented in the engine's live state is dropped on round-trip.

## Code

### `Sources/KataGoOnAppleSilicon/Core/Rules.swift`

Add `Equatable` conformance and a `sgfName` computed property:

```swift
public struct Rules: Sendable, Equatable {
    // … existing fields …
}

extension Rules {
    /// SGF `RU[…]` value derived from this rules object. Mirrors KataGo's
    /// `Rules::toStringNoKomiMaybeNice()` for the two presets the engine
    /// currently models. Both names round-trip through KataGo's
    /// `Rules::tryParseRules`.
    public var sgfName: String {
        if self == .chineseRules { return "Chinese" }
        return "Chinese-OGS"
    }
}
```

The `Chinese-OGS` mapping for `defaultRules` is the KataGo preset whose all four Swift-tracked fields match: `KO_POSITIONAL` (encoding flags `1.0, 0.5`), `SCORING_AREA`, `multiStoneSuicideLegal=false`. (KataGo's literal `TrompTaylor` preset differs from Swift's `defaultRules` on `multiStoneSuicideLegal`, so it would be misleading to emit.)

### `Sources/KataGoOnAppleSilicon/GTPHandler.swift`

Strip the metadata bag and the reset path:

- Delete fields: `blackPlayerName`, `whitePlayerName`, `sgfRulesName`.
- Delete method: `resetSGFMetadata()` (and its call sites in `clear_board` and `handleBoardsize`).
- `handleKataSetRules`: remove the `sgfRulesName = "Chinese"` line.
- `handleLoadSGF`: remove the three writes at the bottom of the function (`sgfRulesName = parsed.rulesName`, `blackPlayerName = parsed.blackPlayer ?? "Black"`, `whitePlayerName = parsed.whitePlayer ?? "White"`). The values are still parsed by `SGFParser`; we just stop persisting them.
- `handlePrintSGF`: emit empty player names and derive `RU` from live rules:

```swift
let sgf = SGFGenerator.generateSGF(
    from: board,
    blackPlayer: "",
    whitePlayer: "",
    rulesName: rules.sgfName
)
```

### `Sources/KataGoOnAppleSilicon/Core/SGFGenerator.swift`

No signature change. The pass-aware overload already accepts `rulesName: String?`; we just always pass a non-nil value.

### `Sources/KataGoOnAppleSilicon/Core/SGFParser.swift`

No change. `ParsedSGF.rulesName` / `.blackPlayer` / `.whitePlayer` remain in the parser surface (other callers may need them later); only `handleLoadSGF` stops reading them.

## Tests

### New regression coverage (`GTPHandlerSGFTests`)

1. `kata-set-rules chinese` → `clear_board` → `printsgf` → assert output contains `RU[Chinese]`. (The exact issue #13 scenario.)
2. `kata-set-rules chinese` → `boardsize 19` → `printsgf` → assert `RU[Chinese]` (rules survive `boardsize` too).
3. Fresh engine → `printsgf` → assert `RU[Chinese-OGS]`.
4. `loadsgf` of an SGF with non-empty `PB[Alice]PW[Bob]` → `printsgf` → assert `PB[]` and `PW[]` (KataGo-aligned, dropped from Swift's previous behavior).
5. `loadsgf` of an SGF with `RU[Japanese]` → `printsgf` → assert `RU[Chinese-OGS]` (engine falls back to `defaultRules`; the loaded string is intentionally not preserved).

### Existing tests — verified survey

`GTPHandlerSGFTests.swift` was scanned end-to-end. The relevant cases:

- `testGTPLoadSGFRulesName` (loads `RU[Chinese]`, expects `RU[Chinese]` back): **passes unchanged** — engine sets `rules = .chineseRules`; `rules.sgfName == "Chinese"`.
- `testGTPLoadSGFInvalidSetupLeavesEngineStateIntact`, `testGTPLoadSGFIllegalMoveLeavesEngineStateIntact` (use `kata-set-rules chinese` then assert `RU[Chinese]`): **pass unchanged** — same derivation.
- `testGTPPrintSGFEmptyBoard`, `testGTPPrintSGFWithMovesAndPass`, `testGTPPrintSGFRespectsBoardsize`, `testGTPPrintSGFEmitsHandicap`, `testGTPPrintSGFToFile`, `testGTPLoadSGFBasic`, `testGTPLoadSGFMoveNumber`, `testGTPLoadSGFHandicap`, `testGTPLoadSGFPassEncoding`, `testGTPLoadSGFRoundTripPreservesBoardSize`: don't assert PB/PW or RU values; **pass unchanged**.
- `testSGFParserBasicProperties`: asserts the parser still extracts `RU[Japanese]`, `PB[Alice]`, `PW[Bob]`. We're not changing the parser; **passes unchanged**.

No existing GTPHandlerSGFTests case requires changes. The new cases above are additive.

### Cross-engine `SGFInteropTests` — fixture regeneration required

`SGFInteropTests` does **byte-equal** comparisons against snapshot fixtures (`SGFFixtures/<scenario>.export.sgf` and `<scenario>.import.sgf`) that lock in *current Swift output*. After this change, Swift's output changes (PB/PW go empty; RU appears where it was absent), so all 14 snapshot fixtures need to be regenerated:

```
Scripts/generate_sgf_interop_fixtures.sh
```

The 7 `*.katago.sgf` fixtures are stock KataGo output; they don't change.

After regeneration, eyeball the new `*.export.sgf` against `*.katago.sgf`. Expected alignment with KataGo improves substantially (PB/PW empty in both, RU present in both). Two known residual divergences remain — out of scope for this issue:

- **Default rules name:** `empty.katago.sgf` has `RU[TrompTaylor]`; new Swift will emit `RU[Chinese-OGS]`. The engines genuinely differ on default rules (`multiStoneSuicideLegal` true vs false). Aligning defaults would change NN encoding and is a separate decision.
- **`HA[0]` tag:** stock KataGo emits `HA[0]` for non-handicap games; Swift's `SGFGenerator` omits `HA` when handicap is zero. Both are valid SGF; reconciling is a separate pass.

Implementation plan should include the fixture-regen step explicitly so reviewers see the diff.

## Risk

Low. The change deletes state and replaces it with derivations; there is no new data path. Risk surface is limited to:

- The 14 `SGFInteropTests` snapshot fixtures (regenerated by script — diff is auditable in the PR).
- Downstream callers of `SGFGenerator.generateSGF(from:blackPlayer:whitePlayer:…)` outside `GTPHandler` (e.g., `GameGeneratorTests`). Signatures are unchanged; existing callers continue to pass their own player names. Only `GTPHandler.handlePrintSGF` switches to empty PB/PW.

## Non-goals

- Adding more rules presets (Japanese, AGA, NewZealand) to Swift's `Rules` struct. The engine doesn't yet model them; today's two-preset surface area is enough to fix the issue.
- Reconciling Swift's `defaultRules` with KataGo's true defaults.
- Restoring `RU` round-trip for unmodeled rule strings via a "remember the loaded string" escape hatch. Stock KataGo doesn't do this; we don't need to either.
