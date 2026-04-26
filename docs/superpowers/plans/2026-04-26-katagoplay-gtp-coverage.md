# KataGoPlay GTP Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every supported `GTPHandler` command reachable from the KataGoPlay REPL, plus close the `kata-rawnn` symmetry parameter gap.

**Architecture:** Extend `UserCommand` enum and `CommandParser.parse(_:)` with new verbs; add corresponding cases to the game-loop switch in `main.swift`; promote `boardSize` to `var` so resize verbs can rebind it; route `quit` through the GTP handler before `exit(0)`.

**Tech Stack:** Swift 6.2, Swift Package Manager, no external deps.

**Reference spec:** `docs/superpowers/specs/2026-04-26-katagoplay-gtp-coverage-design.md`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `Sources/KataGoPlay/CommandParser.swift` | Modify | Extend `UserCommand` enum and `parse(_:)` for new verbs. |
| `Sources/KataGoPlay/main.swift` | Modify | Promote `boardSize` to `var`; add switch cases for new verbs; route `exit(0)` paths through GTP `quit`; pass parsed symmetry to `kata-rawnn`. |

No new files. No package changes.

---

## Task 1: Extend `UserCommand` enum and parser

**Files:**
- Modify: `Sources/KataGoPlay/CommandParser.swift`

- [ ] **Step 1: Replace the `UserCommand` enum**

Replace the enum (`CommandParser.swift:3-14`) with:

```swift
enum UserCommand {
    case move(String)
    case pass
    case hint(symmetry: Int)
    case aiMove
    case analysis(symmetry: Int)
    case save
    case board
    case profile(String)
    case newGame              // GTP: clear_board
    case undo                 // GTP: undo
    case info                 // GTP: protocol_version + name + version + list_commands
    case known(String)        // GTP: known_command <cmd>
    case handicap(Int)        // GTP: fixed_handicap N
    case freeHandicap([String])// GTP: set_free_handicap <coords>
    case rules(String)        // GTP: kata-set-rules <preset>
    case size(Int)            // GTP: boardsize N
    case komi(Float)          // GTP: komi X
    case quit
    case unknown(String)
}
```

- [ ] **Step 2: Replace `CommandParser.parse(_:)` body**

Replace the entire `parse(_:)` static func (`CommandParser.swift:17-39`) with:

```swift
static func parse(_ raw: String) -> UserCommand {
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    let lower = trimmed.lowercased()
    let tokens = trimmed.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
    let head = tokens.first?.lowercased() ?? ""

    if lower == "quit" || lower == "exit" || lower == "q" { return .quit }
    if lower == "pass" { return .pass }
    if lower == "ai" || lower == "aimove" { return .aiMove }
    if lower == "save" { return .save }
    if lower == "board" || lower == "show" { return .board }
    if lower == "new" { return .newGame }
    if lower == "undo" { return .undo }
    if lower == "info" { return .info }

    if head == "hint" {
        if tokens.count == 1 { return .hint(symmetry: 0) }
        if tokens.count == 2, let s = Int(tokens[1]), (0...7).contains(s) {
            return .hint(symmetry: s)
        }
        return .unknown(raw)
    }
    if head == "analysis" || head == "analyze" {
        if tokens.count == 1 { return .analysis(symmetry: 0) }
        if tokens.count == 2, let s = Int(tokens[1]), (0...7).contains(s) {
            return .analysis(symmetry: s)
        }
        return .unknown(raw)
    }
    if head == "profile", tokens.count >= 2 {
        return .profile(tokens.dropFirst().joined(separator: " "))
    }
    if head == "known", tokens.count == 2 {
        return .known(tokens[1])
    }
    if head == "handicap", tokens.count == 2, let n = Int(tokens[1]) {
        return .handicap(n)
    }
    if head == "free-handicap", tokens.count >= 2 {
        // Uppercase coords so the engine's parseMove (which requires
        // uppercase letters) accepts them.
        return .freeHandicap(tokens.dropFirst().map { $0.uppercased() })
    }
    if head == "rules", tokens.count == 2 {
        return .rules(tokens[1].lowercased())
    }
    if head == "size", tokens.count == 2, let n = Int(tokens[1]) {
        return .size(n)
    }
    if head == "komi", tokens.count == 2, let v = Float(tokens[1]) {
        return .komi(v)
    }

    let upper = trimmed.uppercased()
    if isValidGTPCoord(upper) { return .move(upper) }

    return .unknown(raw)
}
```

- [ ] **Step 3: Build to verify the enum extension compiles**

Run: `swift build`
Expected: build succeeds with errors only in `main.swift` (the switch is now non-exhaustive, which Task 2+ will fix). If there are errors in `CommandParser.swift` itself, fix them before continuing.

- [ ] **Step 4: Commit**

```bash
git add Sources/KataGoPlay/CommandParser.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: extend CommandParser with new GTP-backed verbs

Adds parser cases for new, undo, info, known, handicap, free-handicap,
rules, size, komi, quit, and optional symmetry args on hint/analysis.
Switch in main.swift will be wired in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Promote `boardSize` to `var` and add unimplemented switch cases

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Promote `boardSize` to `var`**

In `main.swift:70`, change:

```swift
let boardSize = setup.boardSize
```

to:

```swift
var boardSize = setup.boardSize
```

- [ ] **Step 2: Update `hint` and `analysis` cases to use parsed symmetry**

Locate the `case .hint:` block (`main.swift:168-180`). Change:

```swift
case .hint:
    let rawResp = gtp.handleCommand("kata-rawnn 0")
```

to:

```swift
case .hint(let symmetry):
    let rawResp = gtp.handleCommand("kata-rawnn \(symmetry)")
```

Then locate the `case .analysis:` block (`main.swift:182-193`). Change:

```swift
case .analysis:
    let rawResp = gtp.handleCommand("kata-rawnn 0")
```

to:

```swift
case .analysis(let symmetry):
    let rawResp = gtp.handleCommand("kata-rawnn \(symmetry)")
```

- [ ] **Step 3: Add stub cases for the new verbs**

Locate the `case .quit:` block (`main.swift:220-222`). Insert these stub cases **above** `case .quit:` (we'll fill them in in later tasks; stubs let the file compile):

```swift
case .newGame:
    print("(new game — not yet implemented)")

case .undo:
    print("(undo — not yet implemented)")

case .info:
    print("(info — not yet implemented)")

case .known:
    print("(known — not yet implemented)")

case .handicap:
    print("(handicap — not yet implemented)")

case .freeHandicap:
    print("(free-handicap — not yet implemented)")

case .rules:
    print("(rules — not yet implemented)")

case .size:
    print("(size — not yet implemented)")

case .komi:
    print("(komi — not yet implemented)")
```

- [ ] **Step 4: Build to verify the switch is now exhaustive**

Run: `swift build`
Expected: PASS with no errors.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: scaffold new verb cases and pass parsed symmetry

Promotes boardSize to var and adds stubbed switch cases for the new
verbs so the build stays green while subsequent commits implement
each verb. Also threads the parsed symmetry value into the existing
hint and analysis cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement `info` and `known` verbs

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Replace the `info` stub**

Replace:

```swift
case .info:
    print("(info — not yet implemented)")
```

with:

```swift
case .info:
    let pv = extractGTPValue(gtp.handleCommand("protocol_version")) ?? "?"
    let nm = extractGTPValue(gtp.handleCommand("name")) ?? "?"
    let vr = extractGTPValue(gtp.handleCommand("version")) ?? "?"
    let lc = extractGTPValue(gtp.handleCommand("list_commands")) ?? "?"
    print("Protocol version: \(pv)")
    print("Name:             \(nm)")
    print("Version:          \(vr)")
    print("Commands:         \(lc)")
```

- [ ] **Step 2: Replace the `known` stub**

Replace:

```swift
case .known:
    print("(known — not yet implemented)")
```

with:

```swift
case .known(let name):
    let resp = gtp.handleCommand("known_command \(name)")
    if let v = extractGTPValue(resp) {
        print("known_command \(name): \(v)")
    } else {
        print("known_command \(name): (no response)")
    }
```

- [ ] **Step 3: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: wire info and known REPL verbs

info issues protocol_version, name, version, and list_commands in
sequence and prints each labeled. known <cmd> issues known_command
and prints true/false.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Route `quit` through the GTP handler and implement `komi`

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Replace the `.quit` case**

Locate the `.quit` case (was at `main.swift:220-222`). Change:

```swift
case .quit:
    print("Goodbye!")
    exit(0)
```

to:

```swift
case .quit:
    _ = gtp.handleCommand("quit")
    print("Goodbye!")
    exit(0)
```

- [ ] **Step 2: Route the resign exit through GTP `quit`**

Locate the resign branch (`main.swift:121-129`). Change:

```swift
            if aiMove == "resign" {
                moveHistory.append((aiColor, "resign"))
                print("AI (\(aiName)) resigns. \(humanName) wins!")
                renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: coord)
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                exit(0)
```

to:

```swift
            if aiMove == "resign" {
                moveHistory.append((aiColor, "resign"))
                print("AI (\(aiName)) resigns. \(humanName) wins!")
                renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: coord)
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                _ = gtp.handleCommand("quit")
                exit(0)
```

- [ ] **Step 3: Route the both-pass exit through GTP `quit`**

Locate the both-pass branch (`main.swift:154-160`). Change:

```swift
            if aiMove.lowercased() == "pass" {
                print("Both players passed. Game over.")
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                exit(0)
            }
```

to:

```swift
            if aiMove.lowercased() == "pass" {
                print("Both players passed. Game over.")
                if let score = extractGTPValue(gtp.handleCommand("final_score")) {
                    print("Final score: \(score)")
                }
                saveSGF(moveHistory: moveHistory, komi: setup.komi, boardSize: boardSize)
                _ = gtp.handleCommand("quit")
                exit(0)
            }
```

- [ ] **Step 4: Implement the `komi` verb**

Replace:

```swift
case .komi:
    print("(komi — not yet implemented)")
```

with:

```swift
case .komi(let value):
    let resp = gtp.handleCommand("komi \(value)")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("komi error: \(msg)")
    } else {
        print("Komi set to \(value)")
    }
```

- [ ] **Step 5: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: route quit through GTP and add komi verb

Every program-end path the user can reach (resign, both-pass, explicit
quit) now sends the GTP quit command before calling exit(0). Adds a
komi REPL verb for mid-game komi changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement the `rules` verb

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Replace the `rules` stub**

Replace:

```swift
case .rules:
    print("(rules — not yet implemented)")
```

with:

```swift
case .rules(let preset):
    let resp = gtp.handleCommand("kata-set-rules \(preset)")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("rules error: \(msg)")
    } else {
        print("Rules set to \(preset)")
    }
```

- [ ] **Step 2: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: add rules verb routed to kata-set-rules

Currently only chinese is accepted by the handler; the verb mirrors
that and surfaces the engine's error message for unknown presets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `new` and `size` verbs (with shared post-reset helper)

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Add a helper for post-reset rendering and AI-first-move**

Insert this helper near the other top-level helpers (after `runAnalysis(_:humanName:aiName:currentIsWhite:boardSize:)`, around `main.swift:38`):

```swift
func startBoardAfterReset(
    gtp: GTPHandler,
    humanColor: Stone,
    aiColor: Stone,
    aiName: String,
    aiGTPStr: String,
    moveHistory: inout [(Stone, String)],
    lastAIMove: inout String?,
    boardSize: Int
) {
    renderBoardFromGTP(gtp, boardSize: boardSize)
    print()
    if humanColor == .white {
        print("AI (\(aiName)) is thinking...")
        let aiResp = gtp.handleCommand("genmove \(aiGTPStr)")
        if let aiMove = extractGTPValue(aiResp) {
            moveHistory.append((aiColor, aiMove))
            lastAIMove = aiMove
            print("AI plays: \(aiMove)")
            renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: aiMove)
            print()
        }
    }
}
```

- [ ] **Step 2: Replace the `newGame` stub**

Replace:

```swift
case .newGame:
    print("(new game — not yet implemented)")
```

with:

```swift
case .newGame:
    _ = gtp.handleCommand("clear_board")
    moveHistory = []
    lastAIMove = nil
    print("New game started.")
    startBoardAfterReset(
        gtp: gtp, humanColor: humanColor, aiColor: aiColor,
        aiName: aiName, aiGTPStr: aiGTPStr,
        moveHistory: &moveHistory, lastAIMove: &lastAIMove,
        boardSize: boardSize
    )
```

- [ ] **Step 3: Replace the `size` stub**

Replace:

```swift
case .size:
    print("(size — not yet implemented)")
```

with:

```swift
case .size(let n):
    let resp = gtp.handleCommand("boardsize \(n)")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("size error: \(msg)")
    } else {
        boardSize = n
        moveHistory = []
        lastAIMove = nil
        // Re-issue komi since boardsize wipes board state but the user's
        // chosen komi should persist across resizes.
        _ = gtp.handleCommand("komi \(setup.komi)")
        print("Board size set to \(n)x\(n).")
        startBoardAfterReset(
            gtp: gtp, humanColor: humanColor, aiColor: aiColor,
            aiName: aiName, aiGTPStr: aiGTPStr,
            moveHistory: &moveHistory, lastAIMove: &lastAIMove,
            boardSize: boardSize
        )
    }
```

- [ ] **Step 4: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: add new and size verbs

new clears the board and restarts; size resizes and restarts. Both
share a startBoardAfterReset helper that renders the empty board and
auto-genmoves if the human is White. size re-issues the user's komi
since boardsize resets it on the engine side.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement the `undo` verb

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Replace the `undo` stub**

Replace:

```swift
case .undo:
    print("(undo — not yet implemented)")
```

with:

```swift
case .undo:
    let resp = gtp.handleCommand("undo")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("undo error: \(msg)")
    } else {
        if !moveHistory.isEmpty { moveHistory.removeLast() }
        // Recompute lastAIMove from the new tail.
        if let last = moveHistory.last, last.0 == aiColor, last.1.lowercased() != "pass" {
            lastAIMove = last.1
        } else {
            lastAIMove = nil
        }
        print("Undid last move.")
        renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: lastAIMove)
    }
```

- [ ] **Step 2: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: add undo verb

One-ply undo. Pops moveHistory, recomputes lastAIMove from the new
tail, and re-renders the board. Engine's "cannot undo" error
surfaces if there is nothing to undo.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement `handicap` and `free-handicap` verbs

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Add a helper for post-handicap AI-first-move**

Insert near the other top-level helpers (after `startBoardAfterReset`):

```swift
func aiTakeMoveIfNeededAfterHandicap(
    gtp: GTPHandler,
    humanColor: Stone,
    aiColor: Stone,
    aiName: String,
    aiGTPStr: String,
    moveHistory: inout [(Stone, String)],
    lastAIMove: inout String?,
    boardSize: Int
) {
    // Handicap stones are always Black (per GTP semantics). White moves next.
    // If the AI is White, it plays now; otherwise we wait for the human.
    guard humanColor == .black else { return }
    print("AI (\(aiName)) is thinking...")
    let aiResp = gtp.handleCommand("genmove \(aiGTPStr)")
    if let aiMove = extractGTPValue(aiResp) {
        moveHistory.append((aiColor, aiMove))
        lastAIMove = aiMove
        print("AI plays: \(aiMove)")
        renderBoardFromGTP(gtp, boardSize: boardSize, lastMove: aiMove)
        print()
    }
}
```

- [ ] **Step 2: Replace the `handicap` stub**

Replace:

```swift
case .handicap:
    print("(handicap — not yet implemented)")
```

with:

```swift
case .handicap(let n):
    let resp = gtp.handleCommand("fixed_handicap \(n)")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("handicap error: \(msg)")
    } else if let vertices = extractGTPValue(resp) {
        // Engine returns the chosen vertices, e.g. "= D4 Q16 D16 Q4\n\n".
        let coords = vertices.split(separator: " ").map(String.init)
        for coord in coords {
            moveHistory.append((.black, coord))
        }
        lastAIMove = nil
        print("Handicap stones placed: \(vertices)")
        renderBoardFromGTP(gtp, boardSize: boardSize)
        print()
        aiTakeMoveIfNeededAfterHandicap(
            gtp: gtp, humanColor: humanColor, aiColor: aiColor,
            aiName: aiName, aiGTPStr: aiGTPStr,
            moveHistory: &moveHistory, lastAIMove: &lastAIMove,
            boardSize: boardSize
        )
    }
```

- [ ] **Step 3: Replace the `freeHandicap` stub**

Replace:

```swift
case .freeHandicap:
    print("(free-handicap — not yet implemented)")
```

with:

```swift
case .freeHandicap(let coords):
    let arg = coords.joined(separator: " ")
    let resp = gtp.handleCommand("set_free_handicap \(arg)")
    if resp.hasPrefix("? ") {
        let msg = resp.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        print("free-handicap error: \(msg)")
    } else {
        for coord in coords {
            moveHistory.append((.black, coord))
        }
        lastAIMove = nil
        print("Free handicap placed: \(arg)")
        renderBoardFromGTP(gtp, boardSize: boardSize)
        print()
        aiTakeMoveIfNeededAfterHandicap(
            gtp: gtp, humanColor: humanColor, aiColor: aiColor,
            aiName: aiName, aiGTPStr: aiGTPStr,
            moveHistory: &moveHistory, lastAIMove: &lastAIMove,
            boardSize: boardSize
        )
    }
```

- [ ] **Step 4: Build**

Run: `swift build`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: add handicap and free-handicap verbs

Both verbs require an empty board (engine enforces). Handicap stones
are always Black per GTP semantics; if the human is Black, the AI
plays White's first move immediately, otherwise we wait for the
human. Stones are appended to moveHistory so SGF export stays correct.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update help text and run final acceptance

**Files:**
- Modify: `Sources/KataGoPlay/main.swift`

- [ ] **Step 1: Replace `helpText`**

Locate `helpText` at `main.swift:93-96`. Replace:

```swift
let helpText = """
Commands: <coord> (e.g. D4) | pass | hint | analysis | board | \
save | profile <name> | ai | quit
"""
```

with:

```swift
let helpText = """
Commands:
  <coord>           play a stone (e.g. D4)
  pass              pass your turn
  hint [sym]        top moves at symmetry sym (0-7, default 0)
  analysis [sym]    detailed analysis at symmetry sym (0-7, default 0)
  board / show      redraw the board
  ai                let the AI play your move
  save              save the current game to SGF
  new               start a new game (clear_board)
  undo              undo the last ply
  size <N>          resize the board (2-19) and restart
  komi <X>          change komi
  rules <preset>    set rules (only "chinese" supported)
  handicap <N>      place N fixed handicap stones (empty board)
  free-handicap <coord>...  place free handicap stones (empty board)
  profile <name>    switch model profile (AI / 1d-9d / 1k-20k)
  info              show engine identity and supported commands
  known <cmd>       check whether a GTP command is known
  quit              exit
"""
```

- [ ] **Step 2: Build and run the existing test suite**

Run: `swift build && swift test`
Expected: build succeeds; all existing tests pass.

- [ ] **Step 3: Manual acceptance — run KataGoPlay end-to-end**

This step is the integration test for the work in this plan. Run:

```bash
swift run KataGoPlay
```

Walk through this script in the REPL (one verb at a time, observing that each prints a sensible result and `board` after it shows expected state):

1. At setup: pick Black, profile `AI`, board `19x19`, komi `7.5`.
2. `info` → expect protocol version `2`, name `KataGoOnAppleSilicon`, version `1.0`, and a space-separated command list.
3. `known play` → expect `true`. `known bogus` → expect `false`.
4. `D4` → play, AI replies. `undo` → expect undo confirmation; `board` shows AI's move gone (or both gone, depending on what's on top).
5. `new` → expect "New game started."; board redraws empty.
6. `handicap 4` → expect "Handicap stones placed: …"; board shows 4 black stones; AI immediately plays White's first move.
7. `new`, then `free-handicap C3 D4 E5` → expect "Free handicap placed: C3 D4 E5"; AI immediately plays White.
8. `new`, then `size 9` → expect 9x9 board redrawn.
9. `komi 6.5` → expect "Komi set to 6.5".
10. `rules chinese` → expect "Rules set to chinese". `rules japanese` → expect "rules error: Unknown rules 'japanese'".
11. `hint 3` → expect a top-moves block computed at symmetry 3. `analysis 7` → expect detailed analysis at symmetry 7.
12. `quit` → expect "Goodbye!" and clean exit.

Confirm in passing that the AI does not crash or print error responses on any verb (other than the deliberate `rules japanese` and `known bogus`).

- [ ] **Step 4: Commit**

```bash
git add Sources/KataGoPlay/main.swift
git commit -m "$(cat <<'EOF'
KataGoPlay: expand help text for all REPL verbs

Documents the new verbs added across this branch so users can
discover them at the prompt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
