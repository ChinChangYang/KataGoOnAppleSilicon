# Issue #13 — `printsgf` rules-name follows engine state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix issue #13 by making `printsgf` derive its SGF tags from live engine state, matching stock KataGo.

**Architecture:** Delete the GTPHandler "SGF metadata bag" (`blackPlayerName`, `whitePlayerName`, `sgfRulesName`). Add `Rules.sgfName` so `printsgf` reads `RU[…]` from the live `Rules` struct. `printsgf` emits empty `PB[]`/`PW[]` like KataGo. `clear_board`/`boardsize` no longer reset SGF metadata — the bug disappears because `rules` already survives those commands via `makeBoard()`.

**Tech Stack:** Swift 6.2, swift-testing, Apple Foundation. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-07-issue-13-rules-name-design.md`

**Branch:** `fix/issue-13-rules-name-reset` (off `master @ 8ff6f42`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `Sources/KataGoOnAppleSilicon/Core/Rules.swift` | modify | Add `Equatable` conformance and `sgfName` computed property. |
| `Sources/KataGoOnAppleSilicon/GTPHandler.swift` | modify | Delete metadata fields and `resetSGFMetadata()`; update `kata-set-rules`, `loadsgf`, `printsgf` call sites. |
| `Tests/KataGoOnAppleSiliconTests/RulesTests.swift` | create | Unit tests for `Rules.sgfName`. |
| `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift` | modify | Add 5 regression tests (issue #13 + sibling cases). |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/*.export.sgf` | regenerate | 7 snapshot fixtures (Swift export). |
| `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/*.import.sgf` | regenerate | 7 snapshot fixtures (Swift round-trip after loadsgf). |

The `*.katago.sgf` fixtures (stock KataGo output) are not regenerated — they're external references.

---

## Task 1: Add `Rules.sgfName` accessor

Foundation: a derived rule-name string the engine can use instead of a separate field. Done first because Task 2 depends on it.

**Files:**
- Create: `Tests/KataGoOnAppleSiliconTests/RulesTests.swift`
- Modify: `Sources/KataGoOnAppleSilicon/Core/Rules.swift` (line 2: add `Equatable` to protocol list; append extension at end of file)

- [ ] **Step 1: Write failing tests for `Rules.sgfName`**

Create `Tests/KataGoOnAppleSiliconTests/RulesTests.swift`:

```swift
import Testing
@testable import KataGoOnAppleSilicon

@Test func testRulesSgfNameForChineseRules() {
    #expect(Rules.chineseRules.sgfName == "Chinese")
}

@Test func testRulesSgfNameForDefaultRules() {
    // defaultRules has koRuleFlag (1.0, 0.5) → KO_POSITIONAL, area scoring,
    // multiStoneSuicideLegal=false. Among KataGo's named presets, only
    // Chinese-OGS matches all four Swift-tracked fields exactly. (TrompTaylor
    // has multiStoneSuicideLegal=true, so it doesn't fit.)
    #expect(Rules.defaultRules.sgfName == "Chinese-OGS")
}

@Test func testRulesEquatable() {
    #expect(Rules.chineseRules == Rules.chineseRules)
    #expect(Rules.defaultRules == Rules.defaultRules)
    #expect(Rules.chineseRules != Rules.defaultRules)
}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `swift test --filter RulesTests 2>&1 | tail -30`
Expected: build error or test failure — `Rules.sgfName` doesn't exist; `Rules` is not yet `Equatable`.

- [ ] **Step 3: Add `Equatable` conformance and `sgfName` to `Rules`**

In `Sources/KataGoOnAppleSilicon/Core/Rules.swift`:

Change line 2 from:

```swift
public struct Rules: Sendable {
```

to:

```swift
public struct Rules: Sendable, Equatable {
```

(Both inner enums `KoRule` and `ScoringRule` have no associated values, so Swift auto-conforms them to `Equatable`. Combined with `Float`/`Bool` stored properties, struct synthesis covers `==` automatically — no manual implementation needed.)

Append at end of file (after the closing `}` of `Rules`):

```swift
public extension Rules {
    /// SGF `RU[…]` value derived from this rules object. Mirrors KataGo's
    /// `Rules::toStringNoKomiMaybeNice()` for the two presets the engine
    /// currently models. Both names round-trip through KataGo's
    /// `Rules::tryParseRules`.
    var sgfName: String {
        if self == .chineseRules { return "Chinese" }
        return "Chinese-OGS"
    }
}
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `swift test --filter RulesTests 2>&1 | tail -20`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full unit-test suite to confirm no regression**

Run: `swift test --filter KataGoOnAppleSiliconTests 2>&1 | tail -20`
Expected: all tests PASS. (No call sites yet depend on `sgfName`; this is a pure addition.)

- [ ] **Step 6: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Rules.swift \
        Tests/KataGoOnAppleSiliconTests/RulesTests.swift
git commit -m "$(cat <<'EOF'
feat(rules): add Rules.sgfName for SGF RU[] derivation (issue #13)

Adds a computed sgfName property that maps Rules to the KataGo preset name
that round-trips through Rules::tryParseRules:
  - chineseRules → "Chinese"
  - defaultRules → "Chinese-OGS" (matches KO_POSITIONAL/area/no-suicide)

Also makes Rules conform to Equatable (auto-synthesized).

Standalone change with unit tests; no call sites yet use sgfName.
Foundation for the GTPHandler refactor in the next commit.
EOF
)"
```

---

## Task 2: Drop the SGF metadata bag from GTPHandler

This is the actual fix. Adds 5 regression tests (TDD red phase), then deletes the metadata fields and updates the three call sites that touched them. Issue #13 reproduction passes after this task.

**Files:**
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift` (append at end of file)
- Modify: `Sources/KataGoOnAppleSilicon/GTPHandler.swift` (multiple sites — see steps)

- [ ] **Step 1: Add 5 regression tests to `GTPHandlerSGFTests.swift`**

Append these tests to `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift`:

```swift
// MARK: - Issue #13: rules name follows engine state

@Test func testIssue13RulesNamePreservedAcrossClearBoard() async throws {
    // Reproduction from issue #13: kata-set-rules then clear_board must
    // not erase the rules name from subsequent printsgf output.
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("kata-set-rules chinese")
    _ = handler.handleCommand("clear_board")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("RU[Chinese]"))
}

@Test func testIssue13RulesNamePreservedAcrossBoardsize() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("kata-set-rules chinese")
    _ = handler.handleCommand("boardsize 19")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("RU[Chinese]"))
}

@Test func testIssue13DefaultRulesEmitsChineseOGS() async throws {
    // Fresh engine has rules == .defaultRules, which maps to "Chinese-OGS".
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("RU[Chinese-OGS]"))
}

@Test func testIssue13PlayerNamesNotRoundTripped() async throws {
    // KataGo-aligned: printsgf emits empty PB/PW regardless of what loadsgf
    // observed. Stock KataGo's WriteSgf::writeSgf is called with empty
    // bName/wName arguments at gtp.cpp:3412.
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[7.5]PB[Alice]PW[Bob];B[dd])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("PB[]"))
    #expect(printed.contains("PW[]"))
}

@Test func testIssue13UnmodeledRulesNameNotRoundTripped() async throws {
    // RU[Japanese] is unmodeled — engine falls back to defaultRules.
    // KataGo-aligned: we no longer stash the loaded RU string; printsgf
    // derives RU from the (fallback) rules object → "Chinese-OGS".
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[6.5]RU[Japanese];B[dd])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("RU[Chinese-OGS]"))
    #expect(!printed.contains("RU[Japanese]"))
}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `swift test --filter testIssue13 2>&1 | tail -30`
Expected: at least the first three tests FAIL (current GTPHandler omits `RU` after `clear_board` / on empty engine; emits `RU[Chinese]` only when `kata-set-rules` was the most recent rules-touching command). The PB/PW test fails because current code stores `Alice`/`Bob`. The unmodeled-rules test fails because current code stashes the raw `Japanese` string.

- [ ] **Step 3: Delete the three SGF metadata fields**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`, delete lines 18–24 (the comment block + three `private var` declarations):

```swift
    // SGF metadata preserved across loadsgf / printsgf so that printing back
    // reflects what was loaded. Defaults match the engine's identity when no
    // SGF has been loaded.
    private var blackPlayerName: String = "Black"
    private var whitePlayerName: String = "White"
    private var sgfRulesName: String? = nil
```

- [ ] **Step 4: Delete `resetSGFMetadata()` and its two call sites**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`:

(a) Remove the call inside `clear_board` (currently line 105):

Change

```swift
        case "clear_board":
            board = makeBoard(size: board.xSize)
            resetGameState()
            resetSGFMetadata()
            return successResponse()
```

to

```swift
        case "clear_board":
            board = makeBoard(size: board.xSize)
            resetGameState()
            return successResponse()
```

(b) Remove the call inside `handleBoardsize` (currently line 155):

Change

```swift
        board = makeBoard(size: size)
        resetGameState()
        resetSGFMetadata()
        return successResponse()
```

to

```swift
        board = makeBoard(size: size)
        resetGameState()
        return successResponse()
```

(c) Delete the entire `resetSGFMetadata()` method (currently lines 129–136 — the doc comment plus the function body):

```swift
    /// Reset SGF metadata to engine defaults (used when a new game starts via
    /// `clear_board` or `boardsize`, so a subsequent `printsgf` doesn't echo
    /// stale player/rules info from a previous `loadsgf`).
    private func resetSGFMetadata() {
        blackPlayerName = "Black"
        whitePlayerName = "White"
        sgfRulesName = nil
    }
```

- [ ] **Step 5: Stop writing `sgfRulesName` from `handleKataSetRules`**

In `handleKataSetRules` (currently around line 197), delete the assignment line.

Change

```swift
        if preset == "chinese" {
            rules = .chineseRules
            board.rules = rules
            sgfRulesName = "Chinese"
            return successResponse()
        } else {
```

to

```swift
        if preset == "chinese" {
            rules = .chineseRules
            board.rules = rules
            return successResponse()
        } else {
```

- [ ] **Step 6: Stop writing the three metadata fields from `handleLoadSGF`**

In `handleLoadSGF` (currently around lines 426–435), remove the three writes and update the comment to match.

Change

```swift
        // Commit engine state only once replay has fully succeeded — any
        // earlier failure leaves rules / sgfRulesName / player names alone,
        // so the engine doesn't end up with a half-applied SGF (e.g. new
        // rules name with the previous board's stones).
        board = newBoard
        rules = nextRules
        sgfRulesName = parsed.rulesName
        blackPlayerName = parsed.blackPlayer ?? "Black"
        whitePlayerName = parsed.whitePlayer ?? "White"
        resetGameState()
```

to

```swift
        // Commit engine state only once replay has fully succeeded — any
        // earlier failure leaves rules / board alone, so the engine doesn't
        // end up with a half-applied SGF.
        board = newBoard
        rules = nextRules
        resetGameState()
```

(`parsed.rulesName`, `parsed.blackPlayer`, `parsed.whitePlayer` are still parsed by `SGFParser`; we just stop persisting them on the engine. The parser surface is unchanged.)

- [ ] **Step 7: Update `handlePrintSGF` to derive PB/PW/RU from engine state**

In `handlePrintSGF` (currently around line 332), change the call to `SGFGenerator.generateSGF`:

Change

```swift
        let sgf = SGFGenerator.generateSGF(
            from: board,
            blackPlayer: blackPlayerName,
            whitePlayer: whitePlayerName,
            rulesName: sgfRulesName
        )
```

to

```swift
        let sgf = SGFGenerator.generateSGF(
            from: board,
            blackPlayer: "",
            whitePlayer: "",
            rulesName: rules.sgfName
        )
```

- [ ] **Step 8: Run the regression tests to verify they now pass**

Run: `swift test --filter testIssue13 2>&1 | tail -20`
Expected: all 5 issue-13 tests PASS.

- [ ] **Step 9: Run all unit tests to confirm no regression**

Run: `swift test --filter KataGoOnAppleSiliconTests 2>&1 | tail -30`
Expected: all tests PASS. The pre-existing `testGTPLoadSGFRulesName`, `testGTPLoadSGFInvalidSetupLeavesEngineStateIntact`, `testGTPLoadSGFIllegalMoveLeavesEngineStateIntact` should still pass — they assert `RU[Chinese]` after a sequence that ends with `rules == .chineseRules`, and `rules.sgfName == "Chinese"`.

If anything fails: stop and diagnose. Do not paper over.

- [ ] **Step 10: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/GTPHandler.swift \
        Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift
git commit -m "$(cat <<'EOF'
fix(gtp): drop SGF metadata bag, derive printsgf from engine state (#13)

resetSGFMetadata() unconditionally cleared sgfRulesName on clear_board /
boardsize, even though the active Rules object survived via makeBoard().
That caused kata-set-rules chinese → clear_board → printsgf to omit
RU[Chinese].

Following stock KataGo (cpp/command/gtp.cpp:3412 and cpp/dataio/sgf.cpp:1923),
delete the parallel metadata bag entirely and derive printsgf output from
live engine state:
  - PB[]/PW[] always empty (KataGo passes "" to WriteSgf::writeSgf).
  - RU[…] sourced from rules.sgfName.
  - clear_board / boardsize already preserve rules; nothing else to reset.

Adds 5 regression tests covering the issue scenario, sibling boardsize
case, fresh-engine default rules, and the now-dropped loadsgf round-trips
of PB/PW and unmodeled RU strings.
EOF
)"
```

---

## Task 3: Regenerate SGFInteropTests fixtures

The byte-equal snapshot fixtures lock in *current* Swift output. Task 2 changes Swift's output (PB/PW now empty; RU now present), so the 14 `*.export.sgf` and `*.import.sgf` fixtures need regeneration. The 7 `*.katago.sgf` fixtures are stock KataGo output and are not touched.

**Files:**
- Modify (regenerated): `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/*.export.sgf` (7 files)
- Modify (regenerated): `Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/*.import.sgf` (7 files)

- [ ] **Step 1: Inspect a representative pre-fix fixture for comparison**

Run: `cat Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.export.sgf`
Expected (current): `(;FF[4]GM[1]SZ[19]PB[Black]PW[White]KM[7.5])`

Take note — after regeneration this should become `(;FF[4]GM[1]SZ[19]PB[]PW[]KM[7.5]RU[Chinese-OGS])`.

- [ ] **Step 2: Run the fixture regeneration script**

Run: `Scripts/generate_sgf_interop_fixtures.sh 2>&1 | tail -20`
Expected: success message, no errors. Script writes 7 `*.export.sgf` and 7 `*.import.sgf` files.

- [ ] **Step 3: Review the diff before committing**

Run: `git diff -- Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/`
Expected — across the 14 changed files:
- `PB[Black]` / `PB[<name>]` → `PB[]`
- `PW[White]` / `PW[<name>]` → `PW[]`
- `RU` tag added where it was missing; for the empty-board / default-rules case it appears as `RU[Chinese-OGS]`; for `rules_chinese` it appears as `RU[Chinese]`.

If any other tag changes (KM, SZ, AB/AW, B/W moves, HA): stop. Something other than this fix is leaking into the diff. Investigate before committing.

- [ ] **Step 4: Run the SGFInteropTests to confirm fixtures and code agree**

Run: `swift test --filter SGFInteropTests 2>&1 | tail -30`
Expected: all 14 tests PASS.

If any fail: the diff in step 3 captured the new Swift output, so a failure here indicates the test driver doesn't match the regenerated fixture. Diagnose by running one failing test in isolation, e.g.:
`swift test --filter interop_export_empty 2>&1 | tail -30`

- [ ] **Step 5: Spot-check residual divergence from KataGo (informational)**

Compare `empty.export.sgf` against `empty.katago.sgf`:

```bash
diff Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.export.sgf \
     Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/empty.katago.sgf
```

Expected differences (these are intentional and out of scope per the spec):
- KataGo says `RU[TrompTaylor]`; Swift says `RU[Chinese-OGS]` (engines differ on default rules).
- KataGo emits `HA[0]`; Swift omits `HA` for zero handicap.

No action — just confirm these are the *only* differences. If you see other deltas (e.g. PB/PW non-empty in either, RU missing somewhere), investigate.

- [ ] **Step 6: Commit the fixture regeneration**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/
git commit -m "$(cat <<'EOF'
test: regenerate SGF interop fixtures after issue #13 fix

Snapshot fixtures (*.export.sgf / *.import.sgf) lock in Swift's printsgf
output. Following the issue #13 fix, that output now:
  - Emits empty PB[]/PW[] (KataGo-aligned)
  - Emits RU[…] derived from the live Rules struct

All 14 snapshots regenerated via Scripts/generate_sgf_interop_fixtures.sh.
The 7 *.katago.sgf reference files (stock KataGo output) are unchanged.

Residual divergence from KataGo on the default-rules case (RU[Chinese-OGS]
vs RU[TrompTaylor]) and on HA[0] emission is intentional and tracked
separately.
EOF
)"
```

---

## Final Verification

- [ ] **Step 1: Run the entire test suite**

Run: `swift test 2>&1 | tail -40`
Expected: every test PASSES.

- [ ] **Step 2: Manually verify the issue #13 reproduction with the GTPRunner executable**

Run:

```bash
echo 'kata-set-rules chinese
clear_board
printsgf
quit' | swift run GTPRunner 2>/dev/null | grep -E '^=' | tail -5
```

Expected: one of the lines contains `RU[Chinese]`.

- [ ] **Step 3: Confirm the branch is ready for review**

Run: `git log --oneline master..HEAD`
Expected: 4 commits — the existing spec commit (`b11fe44`), then Task 1, Task 2, Task 3 commits in order.
