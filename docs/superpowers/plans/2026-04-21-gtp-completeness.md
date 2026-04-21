# GTP Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `undo`, `fixed_handicap`, and `set_free_handicap` to the GTP surface with byte-exact cross-validation against KataGo.

**Architecture:** Three new command handlers share a small addition to `Board` (an initial-state snapshot + an explicit `sideToMove`). Each command is cross-validated by running the same `.gtp` command script through both KataGo (to produce a reference) and `GTPHandler` (to produce the candidate), then diffing byte-for-byte.

**Tech Stack:** Swift 6.2, Swift Testing (`import Testing`, `@Test` functions), Apple Core ML (unchanged for these commands — they do not touch the NN), Bash (reference-generation script).

**Spec:** `docs/superpowers/specs/2026-04-21-gtp-completeness-design.md`

---

## File Plan

**Modify:**
- `Sources/KataGoOnAppleSilicon/Core/Board.swift` — add `initialStones`, `initialSideToMove`, `sideToMove` fields; update constructor and `copy()`; add `clearToInitial`, `isEmpty`, `undo`, `placeFixedHandicap`, `setStonesFailIfNoLibs`, `placeFreeHandicap`; advance `sideToMove` in `playMove`/`playPass`.
- `Sources/KataGoOnAppleSilicon/GTPHandler.swift` — add `handleUndo`, `handleFixedHandicap`, `handleSetFreeHandicap`; register them in `knownCommands`; replace `turnNumber %2` side-to-move inference in `kata-rawnn` and `final_score` with `board.sideToMove`.
- `Sources/KataGoOnAppleSilicon/Errors.swift` — add a typed error for handicap-placement failures carrying an exact KataGo message.
- `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift` — new per-command tests.
- `README.md` — list the three new commands in Features and the GTP command list.

**Create:**
- `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift` — unit tests for new `Board` methods.
- `Scripts/GTPFixtures/<16 fixtures>.gtp`.
- `Scripts/generate_gtp_reference.sh`.
- `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/<16 fixtures>.txt` (generated).
- `Tests/KataGoOnAppleSiliconIntegrationTests/GTPFixtureTests.swift`.

---

## Task 1: Add initial-state fields to Board

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift` (around line 31-58)
- Create: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests** — create the new test file with two tests for the freshly-added fields.

`Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
import Testing
import Foundation
@testable import KataGoOnAppleSilicon

@Test func testBoardFreshInitialStateIsEmptyAndBlackToMove() async throws {
    let board = Board(size: 19)
    for y in 0..<19 {
        for x in 0..<19 {
            #expect(board.initialStones[y][x] == .empty)
        }
    }
    #expect(board.initialSideToMove == .black)
    #expect(board.sideToMove == .black)
}

@Test func testBoardCopyPreservesInitialStateAndSideToMove() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black
    board.initialSideToMove = .white
    board.sideToMove = .white
    let clone = board.copy()
    #expect(clone.initialStones[3][3] == .black)
    #expect(clone.initialSideToMove == .white)
    #expect(clone.sideToMove == .white)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testBoardFreshInitialStateIsEmptyAndBlackToMove
```

Expected: compile failure (`initialStones` / `initialSideToMove` / `sideToMove` not found).

- [ ] **Step 3: Add the three fields and update `Board(size:)` and `copy()`**

In `Sources/KataGoOnAppleSilicon/Core/Board.swift`, after line 38 (`public private(set) var moveHistory: [Move] = []`), add:
```swift
    public internal(set) var initialStones: [[Stone]]
    public internal(set) var initialSideToMove: Stone
    public internal(set) var sideToMove: Stone
```

In the `public init(size: Int = 19)` body, after `stones = Array(...)`, add:
```swift
        initialStones = Array(repeating: Array(repeating: .empty, count: size), count: size)
        initialSideToMove = .black
        sideToMove = .black
```

Update `public func copy() -> Board`:
```swift
    public func copy() -> Board {
        let newBoard = Board(size: xSize)
        newBoard.stones = stones
        newBoard.koPoint = koPoint
        newBoard.turnNumber = turnNumber
        newBoard.komi = komi
        newBoard.moveHistory = moveHistory
        newBoard.initialStones = initialStones
        newBoard.initialSideToMove = initialSideToMove
        newBoard.sideToMove = sideToMove
        return newBoard
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testBoardFreshInitialStateIsEmptyAndBlackToMove
swift test --filter testBoardCopyPreservesInitialStateAndSideToMove
swift build
```

Expected: both pass, full build succeeds.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add initial-state snapshot and sideToMove fields"
```

---

## Task 2: `Board.clearToInitial()` and `Board.isEmpty()`

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testIsEmpty() async throws {
    let board = Board(size: 19)
    #expect(board.isEmpty())
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(!board.isEmpty())
}

@Test func testClearToInitialRestoresSnapshot() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black
    board.initialSideToMove = .white
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .white))
    #expect(board.playPass(stone: .black))
    board.clearToInitial()
    #expect(board.stones[3][3] == .black)
    #expect(board.stones[4][4] == .empty)
    #expect(board.koPoint == nil)
    #expect(board.turnNumber == 0)
    #expect(board.moveHistory.isEmpty)
    #expect(board.sideToMove == .white)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testIsEmpty
swift test --filter testClearToInitialRestoresSnapshot
```

Expected: compile failure (`isEmpty` / `clearToInitial` not found).

- [ ] **Step 3: Implement**

Append to the end of `Board` class body in `Sources/KataGoOnAppleSilicon/Core/Board.swift`, just before the final `}`:
```swift
    /// Returns true iff every cell on the board is empty.
    public func isEmpty() -> Bool {
        for y in 0..<ySize {
            for x in 0..<xSize {
                if stones[y][x] != .empty { return false }
            }
        }
        return true
    }

    /// Reset the live board to the saved initial snapshot.
    /// Stones, ko, turn count, move history, and sideToMove all revert.
    public func clearToInitial() {
        stones = initialStones
        koPoint = nil
        turnNumber = 0
        moveHistory = []
        sideToMove = initialSideToMove
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testIsEmpty
swift test --filter testClearToInitialRestoresSnapshot
swift build
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add isEmpty and clearToInitial"
```

---

## Task 3: Advance `sideToMove` on play/pass; switch GTP handlers to use it

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift` (`playMove`, `playPass`)
- Modify: `Sources/KataGoOnAppleSilicon/GTPHandler.swift` (`handleKataRawNN`, `handleFinalScore`)
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testPlayMoveAdvancesSideToMove() async throws {
    let board = Board(size: 19)
    #expect(board.sideToMove == .black)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.sideToMove == .white)
    #expect(board.playMove(at: Point(x: 15, y: 15), stone: .white))
    #expect(board.sideToMove == .black)
}

@Test func testSequentialSameColorAdvancesToOpponent() async throws {
    // GTP's play command accepts any color; after two blacks the next-to-move is white.
    let board = Board(size: 19)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .black))
    #expect(board.sideToMove == .white)
}

@Test func testPlayPassAdvancesSideToMove() async throws {
    let board = Board(size: 19)
    #expect(board.playPass(stone: .black))
    #expect(board.sideToMove == .white)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testPlayMoveAdvancesSideToMove
```

Expected: `sideToMove` stays `.black` because nothing updates it yet.

- [ ] **Step 3: Update `playMove` and `playPass` to advance `sideToMove`**

In `Sources/KataGoOnAppleSilicon/Core/Board.swift`, inside `playMove(at:stone:)`, immediately before `turnNumber += 1` / `return true` (the end of the success branch), add:
```swift
        sideToMove = stone.opponent
```

Full updated tail of `playMove`:
```swift
        // Track move in history
        moveHistory.append(Move.move(at: point, player: stone))

        sideToMove = stone.opponent
        turnNumber += 1
        return true
    }
```

Inside `playPass(stone:)`, before `turnNumber += 1`:
```swift
        sideToMove = stone.opponent
```

Full updated `playPass`:
```swift
    public func playPass(stone: Stone) -> Bool {
        moveHistory.append(Move.pass(player: stone))
        sideToMove = stone.opponent
        turnNumber += 1
        return true
    }
```

- [ ] **Step 4: Run Board tests to verify they pass**

```bash
swift test --filter testPlayMoveAdvancesSideToMove
swift test --filter testSequentialSameColorAdvancesToOpponent
swift test --filter testPlayPassAdvancesSideToMove
```

Expected: all pass.

- [ ] **Step 5: Switch `handleKataRawNN` and `handleFinalScore` to use `board.sideToMove`**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`:

Replace line 248 inside `handleKataRawNN`:
```swift
        let nextPlayer: Stone = board.turnNumber % 2 == 0 ? .black : .white
```
with:
```swift
        let nextPlayer: Stone = board.sideToMove
```

Replace line 261 inside `handleFinalScore`:
```swift
            let nextPlayer: Stone = board.turnNumber % 2 == 0 ? .black : .white
```
with:
```swift
            let nextPlayer: Stone = board.sideToMove
```

- [ ] **Step 6: Run full test suite**

```bash
swift build
swift test --filter KataGoOnAppleSiliconTests
```

Expected: all pass (existing tests still work because on a plain play sequence `sideToMove` matches `turnNumber % 2`).

- [ ] **Step 7: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Sources/KataGoOnAppleSilicon/GTPHandler.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): advance sideToMove on play/pass; use it in GTP handlers"
```

---

## Task 4: `Board.undo()`

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testUndoOnEmptyBoardReturnsFalse() async throws {
    let board = Board(size: 19)
    #expect(board.undo() == false)
}

@Test func testUndoRemovesLastMove() async throws {
    let board = Board(size: 19)
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 4, y: 4), stone: .white))
    #expect(board.undo())
    #expect(board.stones[4][4] == .empty)
    #expect(board.stones[3][3] == .black)
    #expect(board.turnNumber == 1)
    #expect(board.moveHistory.count == 1)
    #expect(board.sideToMove == .white)
}

@Test func testUndoRebuildsCapturedStones() async throws {
    let board = Board(size: 19)
    // Set up a single white stone surrounded on three sides by black; then black captures.
    #expect(board.playMove(at: Point(x: 3, y: 2), stone: .black))
    #expect(board.playMove(at: Point(x: 3, y: 3), stone: .white))
    #expect(board.playMove(at: Point(x: 2, y: 3), stone: .black))
    #expect(board.playMove(at: Point(x: 3, y: 4), stone: .black))
    // White stone still has one liberty at (4, 3).
    #expect(board.stones[3][3] == .white)
    #expect(board.playMove(at: Point(x: 4, y: 3), stone: .black))
    // White is captured.
    #expect(board.stones[3][3] == .empty)
    // Undo the capture — white should return.
    #expect(board.undo())
    #expect(board.stones[3][3] == .white)
    #expect(board.stones[4][3] == .empty)
}

@Test func testUndoPreservesInitialStones() async throws {
    let board = Board(size: 19)
    board.initialStones[3][3] = .black  // Simulate handicap stone.
    board.initialSideToMove = .white
    board.clearToInitial()
    #expect(board.playMove(at: Point(x: 15, y: 15), stone: .white))
    #expect(board.undo())
    #expect(board.stones[3][3] == .black)   // Handicap stone preserved.
    #expect(board.stones[15][15] == .empty) // Played move removed.
    #expect(board.sideToMove == .white)     // Back to initial side-to-move.
    #expect(board.moveHistory.isEmpty)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testUndoOnEmptyBoardReturnsFalse
```

Expected: compile failure (`undo` not found).

- [ ] **Step 3: Implement `Board.undo()`**

Append to the `Board` class in `Sources/KataGoOnAppleSilicon/Core/Board.swift`:
```swift
    /// Undo the most recent move. Returns false iff there are no moves to undo.
    /// Rewinds to the stored initial snapshot and replays all but the last move,
    /// so stones placed via handicap (stored in initialStones) survive.
    public func undo() -> Bool {
        guard !moveHistory.isEmpty else { return false }
        let replay = Array(moveHistory.dropLast())
        clearToInitial()
        for move in replay {
            if move.isPass {
                _ = playPass(stone: move.player)
            } else if let loc = move.location {
                _ = playMove(at: loc, stone: move.player)
            }
        }
        return true
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testUndoOnEmptyBoardReturnsFalse
swift test --filter testUndoRemovesLastMove
swift test --filter testUndoRebuildsCapturedStones
swift test --filter testUndoPreservesInitialStones
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add undo that rewinds to initial snapshot"
```

---

## Task 5: Wire `undo` into GTPHandler

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/GTPHandler.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`:
```swift
@Test func testGTPUndoOnEmptyBoard() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("undo")
    #expect(response == "? cannot undo\n\n")
}

@Test func testGTPUndoAfterPlay() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    let response = handler.handleCommand("undo")
    #expect(response == "= \n\n")
    // Repeat undo now that history is empty.
    #expect(handler.handleCommand("undo") == "? cannot undo\n\n")
}

@Test func testGTPUndoClearsPassTracking() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    _ = handler.handleCommand("play white pass")
    // The pass sets lastPlayPassColor internally; undo should clear it since
    // the new last move (D4) is not a pass.
    #expect(handler.handleCommand("undo") == "= \n\n")
    // A second undo removes the D4 play.
    #expect(handler.handleCommand("undo") == "= \n\n")
    // A third undo fails.
    #expect(handler.handleCommand("undo") == "? cannot undo\n\n")
}

@Test func testGTPListCommandsIncludesUndo() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("list_commands")
    #expect(response.contains("undo"))
    #expect(handler.handleCommand("known_command undo") == "= true\n\n")
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testGTPUndoOnEmptyBoard
```

Expected: response is `? unknown command\n\n`, not `? cannot undo\n\n`.

- [ ] **Step 3: Register `undo` and add the handler**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`:

Add `undo` to `knownCommands` (line 281) so the array reads:
```swift
    private let knownCommands = ["protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play", "genmove", "undo", "kata-set-rules", "showboard", "kata-rawnn", "final_score", "quit"]
```

Add a dispatch case in `handleCommand` (the switch around line 80). Insert after the `"genmove"` case:
```swift
        case "undo":               return handleUndo()
```

Append the handler method to the class (near the other `handle*` helpers):
```swift
    private func handleUndo() -> String {
        guard board.undo() else {
            return errorResponse("cannot undo")
        }
        // Recompute lastPlayPassColor from the new tail of history, and reset
        // resign counters (no well-defined rewind for an in-progress streak).
        if let last = board.moveHistory.last, last.isPass {
            lastPlayPassColor = last.player
        } else {
            lastPlayPassColor = nil
        }
        consecutiveBehindCount = [.black: 0, .white: 0]
        return successResponse()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testGTPUndoOnEmptyBoard
swift test --filter testGTPUndoAfterPlay
swift test --filter testGTPUndoClearsPassTracking
swift test --filter testGTPListCommandsIncludesUndo
swift build
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/GTPHandler.swift Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift
git commit -m "feat(gtp): add undo command"
```

---

## Task 6: Handicap error type

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Errors.swift`

- [ ] **Step 1: Add a new error case**

In `Sources/KataGoOnAppleSilicon/Errors.swift`, add a case to `KataGoError` and a description arm. The full updated enum:
```swift
import Foundation

/// Custom errors for the KataGo library
public enum KataGoError: Error, CustomStringConvertible {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case invalidInput(String)
    case inferenceFailed(String)
    case unsupportedProfile(String)
    /// Handicap placement refused by the rules. Message is the exact GTP error
    /// text to emit (matches KataGo's gtp.cpp verbatim).
    case handicapRefused(String)

    public var description: String {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .unsupportedProfile(let profile):
            return "Unsupported profile: \(profile)"
        case .handicapRefused(let message):
            return message
        }
    }
}
```

- [ ] **Step 2: Verify build**

```bash
swift build
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Errors.swift
git commit -m "feat(errors): add handicapRefused error case"
```

---

## Task 7: `Board.placeFixedHandicap(n:)`

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testPlaceFixedHandicap2On19x19() async throws {
    let board = Board(size: 19)
    let points = try board.placeFixedHandicap(n: 2)
    // Scan order: y=0..<19, x=0..<19. (15, 3) appears before (3, 15).
    #expect(points == [Point(x: 15, y: 3), Point(x: 3, y: 15)])
    #expect(board.stones[3][15] == .black)
    #expect(board.stones[15][3] == .black)
    #expect(board.initialStones[3][15] == .black)
    #expect(board.initialStones[15][3] == .black)
    #expect(board.initialSideToMove == .white)
    #expect(board.sideToMove == .white)
    #expect(board.moveHistory.isEmpty)
    #expect(board.turnNumber == 0)
}

@Test func testPlaceFixedHandicap9On19x19() async throws {
    let board = Board(size: 19)
    let points = try board.placeFixedHandicap(n: 9)
    // Expected scan order: (3,3), (9,3), (15,3), (3,9), (9,9), (15,9), (3,15), (9,15), (15,15).
    #expect(points == [
        Point(x: 3, y: 3), Point(x: 9, y: 3), Point(x: 15, y: 3),
        Point(x: 3, y: 9), Point(x: 9, y: 9), Point(x: 15, y: 9),
        Point(x: 3, y: 15), Point(x: 9, y: 15), Point(x: 15, y: 15),
    ])
}

@Test func testPlaceFixedHandicap5On13x13() async throws {
    // 13x13: edge offset 3, middle 6. Star points used: 3, 9, 6.
    let board = Board(size: 13)
    let points = try board.placeFixedHandicap(n: 5)
    // N=5: (0,1) (1,0) (0,0) (1,1) (2,2) → (3,9) (9,3) (3,3) (9,9) (6,6).
    // Scan order sorts by y then x: (3,3) (9,3) (6,6) (3,9) (9,9).
    #expect(points == [
        Point(x: 3, y: 3), Point(x: 9, y: 3),
        Point(x: 6, y: 6),
        Point(x: 3, y: 9), Point(x: 9, y: 9),
    ])
}

@Test func testPlaceFixedHandicapRejectsTooSmallBoard() async throws {
    let board = Board(size: 6)
    #expect(throws: KataGoError.self) {
        try board.placeFixedHandicap(n: 2)
    }
    do {
        _ = try board.placeFixedHandicap(n: 2)
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Board is too small for fixed handicap, try place_free_handicap")
    } catch {
        Issue.record("Wrong error: \(error)")
    }
}

@Test func testPlaceFixedHandicapRejectsEvenDimAboveFour() async throws {
    let board = Board(size: 8)
    do {
        _ = try board.placeFixedHandicap(n: 5)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 4 is not allowed on boards with even dimensions, try place_free_handicap")
    }
}

@Test func testPlaceFixedHandicapRejectsSize7AboveFour() async throws {
    let board = Board(size: 7)
    do {
        _ = try board.placeFixedHandicap(n: 5)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 4 is not allowed on boards with size 7, try place_free_handicap")
    }
}

@Test func testPlaceFixedHandicapRejectsAboveNine() async throws {
    let board = Board(size: 19)
    do {
        _ = try board.placeFixedHandicap(n: 10)
        Issue.record("should have thrown")
    } catch KataGoError.handicapRefused(let msg) {
        #expect(msg == "Fixed handicap > 9 is not allowed, try place_free_handicap")
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testPlaceFixedHandicap2On19x19
```

Expected: compile failure (`placeFixedHandicap` not found).

- [ ] **Step 3: Implement `placeFixedHandicap`**

Append to the `Board` class in `Sources/KataGoOnAppleSilicon/Core/Board.swift`:
```swift
    /// Place a fixed handicap of `n` black stones. Mirrors
    /// PlayUtils::placeFixedHandicap in KataGo (cpp/program/playutils.cpp:300).
    /// Throws `KataGoError.handicapRefused` with the exact KataGo-compatible
    /// message for any rule violation.
    ///
    /// On success: writes stones into `stones` and `initialStones`,
    /// sets `initialSideToMove` and `sideToMove` to `.white`,
    /// clears `moveHistory`, resets `koPoint` and `turnNumber`.
    ///
    /// Caller is responsible for enforcing "board must be empty" before calling.
    ///
    /// Returns the placed points in `y=0..<ySize, x=0..<xSize` scan order so
    /// the GTP response can echo them.
    @discardableResult
    public func placeFixedHandicap(n: Int) throws -> [Point] {
        if xSize < 7 || ySize < 7 {
            throw KataGoError.handicapRefused("Board is too small for fixed handicap, try place_free_handicap")
        }
        if (xSize % 2 == 0 || ySize % 2 == 0) && n > 4 {
            throw KataGoError.handicapRefused("Fixed handicap > 4 is not allowed on boards with even dimensions, try place_free_handicap")
        }
        if (xSize <= 7 || ySize <= 7) && n > 4 {
            throw KataGoError.handicapRefused("Fixed handicap > 4 is not allowed on boards with size 7, try place_free_handicap")
        }
        if n > 9 {
            throw KataGoError.handicapRefused("Fixed handicap > 9 is not allowed, try place_free_handicap")
        }
        // Note: n < 2 is rejected by the GTP handler before calling us.

        let xLow = xSize <= 12 ? 2 : 3
        let yLow = ySize <= 12 ? 2 : 3
        let xCoords = [xLow, xSize - 1 - xLow, xSize / 2]
        let yCoords = [yLow, ySize - 1 - yLow, ySize / 2]

        // Placement patterns verbatim from playutils.cpp:326-333 (non-monotonic across N).
        let pairsByN: [Int: [(Int, Int)]] = [
            2: [(0,1),(1,0)],
            3: [(0,1),(1,0),(0,0)],
            4: [(0,1),(1,0),(0,0),(1,1)],
            5: [(0,1),(1,0),(0,0),(1,1),(2,2)],
            6: [(0,1),(1,0),(0,0),(1,1),(0,2),(1,2)],
            7: [(0,1),(1,0),(0,0),(1,1),(0,2),(1,2),(2,2)],
            8: [(0,1),(1,0),(0,0),(1,1),(0,2),(1,2),(2,0),(2,1)],
            9: [(0,1),(1,0),(0,0),(1,1),(0,2),(1,2),(2,0),(2,1),(2,2)],
        ]
        guard let pairs = pairsByN[n] else {
            // Unreachable: all n in [2, 9] are covered above.
            throw KataGoError.handicapRefused("Fixed handicap > 9 is not allowed, try place_free_handicap")
        }

        // Reset stones to empty (caller's precondition is already-empty, but
        // be explicit to match KataGo's `board = Board(xSize,ySize)` reset at
        // playutils.cpp:314.)
        stones = Array(repeating: Array(repeating: .empty, count: xSize), count: ySize)

        for (xi, yi) in pairs {
            let x = xCoords[xi]
            let y = yCoords[yi]
            stones[y][x] = .black
        }

        // Snapshot initial state and reset live bookkeeping.
        initialStones = stones
        initialSideToMove = .white
        sideToMove = .white
        koPoint = nil
        turnNumber = 0
        moveHistory = []

        // Scan-order output for the GTP response.
        var placed: [Point] = []
        for y in 0..<ySize {
            for x in 0..<xSize {
                if stones[y][x] == .black {
                    placed.append(Point(x: x, y: y))
                }
            }
        }
        return placed
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testPlaceFixedHandicap2On19x19
swift test --filter testPlaceFixedHandicap9On19x19
swift test --filter testPlaceFixedHandicap5On13x13
swift test --filter testPlaceFixedHandicapRejectsTooSmallBoard
swift test --filter testPlaceFixedHandicapRejectsEvenDimAboveFour
swift test --filter testPlaceFixedHandicapRejectsSize7AboveFour
swift test --filter testPlaceFixedHandicapRejectsAboveNine
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add placeFixedHandicap matching KataGo's placement table"
```

---

## Task 8: Wire `fixed_handicap` into GTPHandler

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/GTPHandler.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`:
```swift
@Test func testGTPFixedHandicap2On19x19() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("fixed_handicap 2")
    #expect(response == "= Q16 D4\n\n")
}

@Test func testGTPFixedHandicapArgCountError() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("fixed_handicap") == "? Expected one argument for fixed_handicap but got ''\n\n")
    #expect(handler.handleCommand("fixed_handicap 2 3") == "? Expected one argument for fixed_handicap but got '2 3'\n\n")
}

@Test func testGTPFixedHandicapIntegerParseError() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("fixed_handicap abc") == "? Could not parse number of handicap stones: 'abc'\n\n")
}

@Test func testGTPFixedHandicapBelowTwoError() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("fixed_handicap 1") == "? Number of handicap stones less than 2: '1'\n\n")
}

@Test func testGTPFixedHandicapBoardNotEmptyError() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    #expect(handler.handleCommand("fixed_handicap 2") == "? Board is not empty\n\n")
}

@Test func testGTPFixedHandicapEvenDimError() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 8")
    #expect(handler.handleCommand("fixed_handicap 5") == "? Fixed handicap > 4 is not allowed on boards with even dimensions, try place_free_handicap\n\n")
}

@Test func testGTPListCommandsIncludesFixedHandicap() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command fixed_handicap") == "= true\n\n")
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testGTPFixedHandicap2On19x19
```

Expected: `? unknown command\n\n`.

- [ ] **Step 3: Register `fixed_handicap` and add the handler**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`:

Add to `knownCommands`:
```swift
    private let knownCommands = ["protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play", "genmove", "undo", "fixed_handicap", "kata-set-rules", "showboard", "kata-rawnn", "final_score", "quit"]
```

Add a dispatch case in `handleCommand` (after `"undo"`):
```swift
        case "fixed_handicap":     return handleFixedHandicap(parts: parts)
```

Append the handler:
```swift
    private func handleFixedHandicap(parts: [String]) -> String {
        let argJoined = parts.count > 1 ? parts[1...].joined(separator: " ") : ""
        guard parts.count == 2 else {
            return errorResponse("Expected one argument for fixed_handicap but got '\(argJoined)'")
        }
        let arg = parts[1]
        guard let n = Int(arg) else {
            return errorResponse("Could not parse number of handicap stones: '\(arg)'")
        }
        if n < 2 {
            return errorResponse("Number of handicap stones less than 2: '\(arg)'")
        }
        guard board.isEmpty() else {
            return errorResponse("Board is not empty")
        }
        do {
            let placed = try board.placeFixedHandicap(n: n)
            // Reset GTP-level bookkeeping — handicap starts a fresh game.
            resetGameState()
            let vertices = placed.map { coordinateToGTP(x: $0.x, y: $0.y) }.joined(separator: " ")
            return successResponse(vertices)
        } catch KataGoError.handicapRefused(let message) {
            return errorResponse(message)
        } catch {
            return errorResponse(error.localizedDescription)
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testGTPFixedHandicap2On19x19
swift test --filter testGTPFixedHandicapArgCountError
swift test --filter testGTPFixedHandicapIntegerParseError
swift test --filter testGTPFixedHandicapBelowTwoError
swift test --filter testGTPFixedHandicapBoardNotEmptyError
swift test --filter testGTPFixedHandicapEvenDimError
swift test --filter testGTPListCommandsIncludesFixedHandicap
swift build
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/GTPHandler.swift Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift
git commit -m "feat(gtp): add fixed_handicap command"
```

---

## Task 9: `Board.setStonesFailIfNoLibs`

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testSetStonesFailIfNoLibsAcceptsValid() async throws {
    let board = Board(size: 19)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 3, y: 3), .black),
        (Point(x: 15, y: 15), .black),
    ])
    #expect(ok)
    #expect(board.stones[3][3] == .black)
    #expect(board.stones[15][15] == .black)
}

@Test func testSetStonesFailIfNoLibsRejectsDuplicate() async throws {
    let board = Board(size: 19)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 3, y: 3), .black),
        (Point(x: 3, y: 3), .black),
    ])
    #expect(!ok)
    // Stones are not committed on failure.
    #expect(board.stones[3][3] == .empty)
}

@Test func testSetStonesFailIfNoLibsRejectsZeroLiberty() async throws {
    // 2x2 board fully filled with black: no liberties anywhere.
    let board = Board(size: 2)
    let ok = board.setStonesFailIfNoLibs([
        (Point(x: 0, y: 0), .black),
        (Point(x: 1, y: 0), .black),
        (Point(x: 0, y: 1), .black),
        (Point(x: 1, y: 1), .black),
    ])
    #expect(!ok)
    #expect(board.stones[0][0] == .empty)
    #expect(board.stones[1][1] == .empty)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testSetStonesFailIfNoLibsAcceptsValid
```

Expected: compile failure (`setStonesFailIfNoLibs` not found).

- [ ] **Step 3: Implement**

Append to the `Board` class in `Sources/KataGoOnAppleSilicon/Core/Board.swift`:
```swift
    /// Place a batch of stones atomically on an otherwise empty board.
    /// Mirrors `Board::setStonesFailIfNoLibs` in KataGo (cpp/game/board.cpp:730).
    /// Returns false — and leaves `stones` untouched — if any location is
    /// listed twice, or if any placed stone would end up with zero liberties.
    /// Expects caller to require the board be empty first.
    public func setStonesFailIfNoLibs(_ placements: [(Point, Stone)]) -> Bool {
        // Duplicate detection.
        var seen = Set<Point>()
        for (p, _) in placements {
            if !seen.insert(p).inserted {
                return false
            }
            if !isValidPoint(p) {
                return false
            }
        }

        // Work on a copy so we can roll back cleanly.
        var trial = stones
        for (p, s) in placements {
            trial[p.y][p.x] = s
        }

        // Liberty check for every placed stone, using the trial grid.
        func trialLiberties(of start: Point) -> Int {
            let color = trial[start.y][start.x]
            if color == .empty { return 0 }
            var visited = Set<Point>()
            var stack = [start]
            var liberties = 0
            while let p = stack.popLast() {
                if !visited.insert(p).inserted { continue }
                for neighbor in neighbors(of: p) {
                    let nColor = trial[neighbor.y][neighbor.x]
                    if nColor == .empty {
                        liberties += 1 // Simple count; duplicates from shared liberties are fine for the >0 check.
                    } else if nColor == color, !visited.contains(neighbor) {
                        stack.append(neighbor)
                    }
                }
            }
            return liberties
        }
        for (p, _) in placements {
            if trialLiberties(of: p) == 0 {
                return false
            }
        }

        // Commit.
        stones = trial
        return true
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testSetStonesFailIfNoLibsAcceptsValid
swift test --filter testSetStonesFailIfNoLibsRejectsDuplicate
swift test --filter testSetStonesFailIfNoLibsRejectsZeroLiberty
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add setStonesFailIfNoLibs"
```

---

## Task 10: `Board.placeFreeHandicap`

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/Core/Board.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift`:
```swift
@Test func testPlaceFreeHandicapBasic() async throws {
    let board = Board(size: 19)
    let ok = board.placeFreeHandicap([
        Point(x: 3, y: 15),   // D4
        Point(x: 15, y: 15),  // Q4
        Point(x: 15, y: 3),   // Q16
    ])
    #expect(ok)
    #expect(board.stones[15][3] == .black)
    #expect(board.initialStones[15][3] == .black)
    #expect(board.initialSideToMove == .white)
    #expect(board.sideToMove == .white)
    #expect(board.moveHistory.isEmpty)
    #expect(board.turnNumber == 0)
}

@Test func testPlaceFreeHandicapRejectsInvalid() async throws {
    let board = Board(size: 2)
    // Fill 2x2 — every stone has zero liberties, should fail.
    let ok = board.placeFreeHandicap([
        Point(x: 0, y: 0),
        Point(x: 1, y: 0),
        Point(x: 0, y: 1),
        Point(x: 1, y: 1),
    ])
    #expect(!ok)
    #expect(board.initialStones[0][0] == .empty)
    #expect(board.initialSideToMove == .black)
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testPlaceFreeHandicapBasic
```

Expected: compile failure (`placeFreeHandicap` not found).

- [ ] **Step 3: Implement**

Append to `Board` in `Sources/KataGoOnAppleSilicon/Core/Board.swift`:
```swift
    /// Place user-supplied black stones as a free handicap. Mirrors the
    /// successful branch of the `set_free_handicap` handler in
    /// KataGo's cpp/command/gtp.cpp:3176-3207.
    ///
    /// On success, returns true and:
    ///   - writes stones into `stones` and `initialStones`,
    ///   - sets `initialSideToMove` and `sideToMove` to `.white`,
    ///   - clears `moveHistory`, resets `koPoint` and `turnNumber`.
    /// On failure (duplicate or zero-liberty placement), returns false and
    /// leaves the board untouched. Caller enforces "board must be empty".
    public func placeFreeHandicap(_ points: [Point]) -> Bool {
        let placements = points.map { ($0, Stone.black) }
        guard setStonesFailIfNoLibs(placements) else { return false }
        initialStones = stones
        initialSideToMove = .white
        sideToMove = .white
        koPoint = nil
        turnNumber = 0
        moveHistory = []
        return true
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testPlaceFreeHandicapBasic
swift test --filter testPlaceFreeHandicapRejectsInvalid
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/Core/Board.swift Tests/KataGoOnAppleSiliconTests/BoardInitialStateTests.swift
git commit -m "feat(board): add placeFreeHandicap"
```

---

## Task 11: Wire `set_free_handicap` into GTPHandler

**Files:**
- Modify: `Sources/KataGoOnAppleSilicon/GTPHandler.swift`
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift`:
```swift
@Test func testGTPSetFreeHandicapBasic() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap D4 Q4 Q16") == "= \n\n")
}

@Test func testGTPSetFreeHandicapBoardNotEmpty() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    #expect(handler.handleCommand("set_free_handicap Q4 Q16") == "? Board is not empty\n\n")
}

@Test func testGTPSetFreeHandicapPassRejected() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap D4 pass") == "? Invalid handicap location: pass\n\n")
}

@Test func testGTPSetFreeHandicapInvalidVertexReportsLast() async throws {
    // KataGo's parser loop overwrites the message on each bad piece.
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap zz9 qq2") == "? Invalid handicap location: qq2\n\n")
}

@Test func testGTPSetFreeHandicapDuplicate() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap D4 D4") == "? Handicap placement is invalid\n\n")
}

@Test func testGTPSetFreeHandicapNoLiberties() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 2")
    #expect(handler.handleCommand("set_free_handicap A1 A2 B1 B2") == "? Handicap placement is invalid\n\n")
}

@Test func testGTPSetFreeHandicapKnownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command set_free_handicap") == "= true\n\n")
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
swift test --filter testGTPSetFreeHandicapBasic
```

Expected: `? unknown command\n\n`.

- [ ] **Step 3: Register `set_free_handicap` and add the handler**

In `Sources/KataGoOnAppleSilicon/GTPHandler.swift`:

Add to `knownCommands`:
```swift
    private let knownCommands = ["protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play", "genmove", "undo", "fixed_handicap", "set_free_handicap", "kata-set-rules", "showboard", "kata-rawnn", "final_score", "quit"]
```

Add a dispatch case in `handleCommand` (after `"fixed_handicap"`):
```swift
        case "set_free_handicap":  return handleSetFreeHandicap(parts: parts)
```

Append the handler:
```swift
    private func handleSetFreeHandicap(parts: [String]) -> String {
        guard board.isEmpty() else {
            return errorResponse("Board is not empty")
        }
        var points: [Point] = []
        var lastBad: String? = nil
        for piece in parts.dropFirst() {
            if piece.lowercased() == "pass" {
                lastBad = piece
                continue
            }
            if let point = parseMove(piece) {
                points.append(point)
            } else {
                lastBad = piece
            }
        }
        if let bad = lastBad {
            return errorResponse("Invalid handicap location: \(bad)")
        }
        guard board.placeFreeHandicap(points) else {
            return errorResponse("Handicap placement is invalid")
        }
        resetGameState()
        return successResponse()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
swift test --filter testGTPSetFreeHandicapBasic
swift test --filter testGTPSetFreeHandicapBoardNotEmpty
swift test --filter testGTPSetFreeHandicapPassRejected
swift test --filter testGTPSetFreeHandicapInvalidVertexReportsLast
swift test --filter testGTPSetFreeHandicapDuplicate
swift test --filter testGTPSetFreeHandicapNoLiberties
swift test --filter testGTPSetFreeHandicapKnownCommand
swift build
swift test --filter KataGoOnAppleSiliconTests
```

Expected: all pass; full suite stays green.

- [ ] **Step 5: Commit**

```bash
git add Sources/KataGoOnAppleSilicon/GTPHandler.swift Tests/KataGoOnAppleSiliconTests/GTPHandlerTests.swift
git commit -m "feat(gtp): add set_free_handicap command"
```

---

## Task 12: Create the 16 GTP fixture scripts

**Files:**
- Create: `Scripts/GTPFixtures/undo_empty.gtp`
- Create: `Scripts/GTPFixtures/undo_with_capture.gtp`
- Create: `Scripts/GTPFixtures/undo_after_pass.gtp`
- Create: `Scripts/GTPFixtures/undo_after_fixed_handicap.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_2_19.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_9_19.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_5_13.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_err_too_small.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_err_even_dim.gtp`
- Create: `Scripts/GTPFixtures/fixed_handicap_err_not_empty.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_basic.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_undo.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_err_pass.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_err_dup.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_err_nolibs.gtp`
- Create: `Scripts/GTPFixtures/set_free_handicap_err_not_empty.gtp`

- [ ] **Step 1: Create directory**

```bash
mkdir -p Scripts/GTPFixtures
```

- [ ] **Step 2: Write each fixture**

Write the file contents listed below. Every fixture ends with `quit` so KataGo exits cleanly when the harness pipes the file in.

`Scripts/GTPFixtures/undo_empty.gtp`:
```
boardsize 19
komi 7.5
undo
quit
```

`Scripts/GTPFixtures/undo_with_capture.gtp`:
```
boardsize 19
komi 7.5
play B D5
play W D4
play B E4
play W C4
play B D3
undo
showboard
quit
```

`Scripts/GTPFixtures/undo_after_pass.gtp`:
```
boardsize 19
komi 7.5
play B D4
play W pass
undo
showboard
quit
```

`Scripts/GTPFixtures/undo_after_fixed_handicap.gtp`:
```
boardsize 19
komi 0.5
fixed_handicap 4
play W D5
undo
showboard
quit
```

`Scripts/GTPFixtures/fixed_handicap_2_19.gtp`:
```
boardsize 19
komi 0.5
fixed_handicap 2
showboard
quit
```

`Scripts/GTPFixtures/fixed_handicap_9_19.gtp`:
```
boardsize 19
komi 0.5
fixed_handicap 9
showboard
quit
```

`Scripts/GTPFixtures/fixed_handicap_5_13.gtp`:
```
boardsize 13
komi 0.5
fixed_handicap 5
showboard
quit
```

`Scripts/GTPFixtures/fixed_handicap_err_too_small.gtp`:
```
boardsize 6
fixed_handicap 2
quit
```

`Scripts/GTPFixtures/fixed_handicap_err_even_dim.gtp`:
```
boardsize 8
fixed_handicap 5
quit
```

`Scripts/GTPFixtures/fixed_handicap_err_not_empty.gtp`:
```
boardsize 19
play B D4
fixed_handicap 4
quit
```

`Scripts/GTPFixtures/set_free_handicap_basic.gtp`:
```
boardsize 19
komi 0.5
set_free_handicap D4 Q4 D16 Q16
showboard
quit
```

`Scripts/GTPFixtures/set_free_handicap_undo.gtp`:
```
boardsize 19
komi 0.5
set_free_handicap D4 Q4 Q16
play W K10
undo
showboard
quit
```

`Scripts/GTPFixtures/set_free_handicap_err_pass.gtp`:
```
boardsize 19
set_free_handicap D4 pass
quit
```

`Scripts/GTPFixtures/set_free_handicap_err_dup.gtp`:
```
boardsize 19
set_free_handicap D4 D4
quit
```

`Scripts/GTPFixtures/set_free_handicap_err_nolibs.gtp`:
```
boardsize 2
set_free_handicap A1 A2 B1 B2
quit
```

`Scripts/GTPFixtures/set_free_handicap_err_not_empty.gtp`:
```
boardsize 19
play B D4
set_free_handicap Q4 Q16
quit
```

- [ ] **Step 3: Sanity-check the fixtures**

```bash
ls -1 Scripts/GTPFixtures/*.gtp | wc -l
```

Expected: `16`.

- [ ] **Step 4: Commit**

```bash
git add Scripts/GTPFixtures/
git commit -m "test: add GTP fixture scripts for undo, fixed_handicap, set_free_handicap"
```

---

## Task 13: Create `Scripts/generate_gtp_reference.sh`

**Files:**
- Create: `Scripts/generate_gtp_reference.sh`

- [ ] **Step 1: Review the existing pattern**

Open `Scripts/generate_kata_raw_nn_reference.sh` to see how it builds KataGo and locates the executable + model. Reuse the same directory variables (`KATAGO_DIR`, `BUILD_DIR`, `KATAGO_EXE`). The new script does NOT need a neural-net model for the three commands in scope (none of them call the NN), so we pass a minimal KataGo config that does not load a network.

Inspect: `cat Scripts/generate_kata_raw_nn_reference.sh` and note the KataGo build steps. The new script calls `./Scripts/generate_kata_raw_nn_reference.sh --build-only` if present, else runs the same `cmake/ninja` sequence inline. If the existing script does not support `--build-only`, duplicate the build logic (minimal copy; refactoring both into a shared helper is out of scope for this plan).

- [ ] **Step 2: Write the new script**

`Scripts/generate_gtp_reference.sh`:
```bash
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
KATAGO_ARCHIVE_URL="https://github.com/ChinChangYang/KataGo/archive/metal-coreml-stable.tar.gz"
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

# ---- Minimal GTP config for commands that don't need a neural net ------------
# These three commands (undo, fixed_handicap, set_free_handicap) don't query
# the NN, but KataGo still expects a valid config and model. We reuse the
# existing GTP config + model the other script already sets up.
GTP_CONFIG="$PROJECT_ROOT/Scripts/gtp_example.cfg"
if [ ! -f "$GTP_CONFIG" ]; then
    # generate_kata_raw_nn_reference.sh writes a config into the project;
    # look for the typical location.
    GTP_CONFIG="$KATAGO_DIR/cpp/configs/gtp_example.cfg"
fi
if [ ! -f "$GTP_CONFIG" ]; then
    echo -e "${RED}Could not find a KataGo GTP config${NC}"
    exit 1
fi

# Binary model path (the other script downloads this).
BIN_MODEL="$PROJECT_ROOT/Tests/KataGoOnAppleSiliconIntegrationTests/Models/b28c512nbt-s11165M.bin.gz"
if [ ! -f "$BIN_MODEL" ]; then
    echo -e "${RED}KataGo binary model not found at $BIN_MODEL; run generate_kata_raw_nn_reference.sh first${NC}"
    exit 1
fi

# ---- Generate references -----------------------------------------------------
run_one() {
    local fixture_path="$1"
    local name
    name="$(basename "$fixture_path" .gtp)"
    local out="$REFERENCE_OUTPUT_DIR/$name.txt"
    echo -e "${YELLOW}Generating $name...${NC}"
    # -log-to-stderr keeps KataGo's init/logging off stdout so the reference
    # file is a pure GTP response stream (= … / ? … blocks separated by
    # blank lines). DO NOT set KATAGO_DEBUG_DUMP — that injects extra output.
    "$KATAGO_EXE" gtp \
        -config "$GTP_CONFIG" \
        -model "$BIN_MODEL" \
        -log-to-stderr \
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
```

- [ ] **Step 3: Make it executable**

```bash
chmod +x Scripts/generate_gtp_reference.sh
```

- [ ] **Step 4: Sanity check without running KataGo**

```bash
bash -n Scripts/generate_gtp_reference.sh
```

Expected: no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add Scripts/generate_gtp_reference.sh
git commit -m "test: add generate_gtp_reference.sh harness"
```

---

## Task 14: Generate reference `.txt` files

This is a one-time run that requires a local KataGo build. The generated files are checked into the repo so CI does not need KataGo.

- [ ] **Step 1: Ensure prereqs**

```bash
which ninja || brew install ninja
```

Verify the reference-model directory from the existing kata-rawnn flow exists (if not, run the sibling script once to seed it):
```bash
ls Tests/KataGoOnAppleSiliconIntegrationTests/Models/ 2>/dev/null || ./Scripts/generate_kata_raw_nn_reference.sh
```

- [ ] **Step 2: Run the generator for all fixtures**

```bash
./Scripts/generate_gtp_reference.sh
```

Expected: 16 `.txt` files written to `Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/`.

- [ ] **Step 3: Spot-check two references**

```bash
cat Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/fixed_handicap_2_19.txt
cat Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/undo_empty.txt
```

Expected for `fixed_handicap_2_19.txt` (leading lines): an `=` line containing `Q16 D4`, followed by KataGo's `showboard` rendering. Expected for `undo_empty.txt`: a `? cannot undo` line among the responses.

- [ ] **Step 4: Commit the generated files**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs/
git commit -m "test: add KataGo reference outputs for GTP fixtures"
```

---

## Task 15: Create `GTPFixtureTests.swift`

**Files:**
- Create: `Tests/KataGoOnAppleSiliconIntegrationTests/GTPFixtureTests.swift`

- [ ] **Step 1: Write the test file**

`Tests/KataGoOnAppleSiliconIntegrationTests/GTPFixtureTests.swift`:
```swift
import Testing
import Foundation
@testable import KataGoOnAppleSilicon

struct GTPFixtureTests {

    private func fixtureURL(_ name: String, ext: String, subdir: String) -> URL? {
        let fm = FileManager.default
        // Walk up from this source file to find the repo root.
        var here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = here
                .appendingPathComponent(subdir, isDirectory: true)
                .appendingPathComponent("\(name).\(ext)")
            if fm.fileExists(atPath: candidate.path) { return candidate }
            here.deleteLastPathComponent()
        }
        return nil
    }

    private func runFixture(_ name: String) throws -> (diff: String, swift: String, reference: String) {
        guard let fixturePath = fixtureURL(name, ext: "gtp", subdir: "Scripts/GTPFixtures") else {
            throw IntegrationTestError.referenceFileNotFound("\(name).gtp")
        }
        guard let referencePath = fixtureURL(name, ext: "txt", subdir: "Tests/KataGoOnAppleSiliconIntegrationTests/ReferenceOutputs") else {
            throw IntegrationTestError.referenceFileNotFound("\(name).txt")
        }
        let fixtureText = try String(contentsOf: fixturePath, encoding: .utf8)
        let reference = try String(contentsOf: referencePath, encoding: .utf8)

        let katago = KataGoInference()
        let handler = GTPHandler(katago: katago)
        var buffer = ""
        for rawLine in fixtureText.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            buffer += handler.handleCommand(line)
        }

        let diff = buffer == reference ? "" : """
        Swift output differs from KataGo reference.
        --- Swift ---
        \(buffer)
        --- Reference ---
        \(reference)
        """
        return (diff, buffer, reference)
    }

    @Test func fixture_undo_empty() throws {
        let r = try runFixture("undo_empty")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_undo_with_capture() throws {
        let r = try runFixture("undo_with_capture")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_undo_after_pass() throws {
        let r = try runFixture("undo_after_pass")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_undo_after_fixed_handicap() throws {
        let r = try runFixture("undo_after_fixed_handicap")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_2_19() throws {
        let r = try runFixture("fixed_handicap_2_19")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_9_19() throws {
        let r = try runFixture("fixed_handicap_9_19")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_5_13() throws {
        let r = try runFixture("fixed_handicap_5_13")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_err_too_small() throws {
        let r = try runFixture("fixed_handicap_err_too_small")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_err_even_dim() throws {
        let r = try runFixture("fixed_handicap_err_even_dim")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_fixed_handicap_err_not_empty() throws {
        let r = try runFixture("fixed_handicap_err_not_empty")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_basic() throws {
        let r = try runFixture("set_free_handicap_basic")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_undo() throws {
        let r = try runFixture("set_free_handicap_undo")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_err_pass() throws {
        let r = try runFixture("set_free_handicap_err_pass")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_err_dup() throws {
        let r = try runFixture("set_free_handicap_err_dup")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_err_nolibs() throws {
        let r = try runFixture("set_free_handicap_err_nolibs")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
    @Test func fixture_set_free_handicap_err_not_empty() throws {
        let r = try runFixture("set_free_handicap_err_not_empty")
        #expect(r.diff.isEmpty, "\(r.diff)")
    }
}
```

Note: `IntegrationTestError` is already defined in the integration test target (used by the kata-rawnn tests). Verify with `grep -n 'enum IntegrationTestError' Tests/KataGoOnAppleSiliconIntegrationTests/*.swift`. If it is not public to this file, import by adding the same helper file or promote the enum to a shared location. If absent entirely, define a local error:
```swift
enum IntegrationTestError: Error { case referenceFileNotFound(String) }
```

- [ ] **Step 2: Run the fixture tests**

```bash
swift test --filter KataGoOnAppleSiliconIntegrationTests
```

Expected: every `fixture_*` test passes. If any fail, the failure message prints both streams side by side — the spec's byte-exactness rule is the reason.

Troubleshooting matrix (apply the first that fits):

- **Extra lines at the top of the reference but not in the Swift buffer** → KataGo's `-log-to-stderr` didn't suppress everything. Inspect `head -5 <reference>.txt`. If banner lines appear, add `-quiet` to `generate_gtp_reference.sh` and regenerate, or filter the banner in the generator with a small `sed`/`awk` step that keeps only lines starting with `=` or `?` plus their bodies (GTP responses always begin with `=` or `?`; content lines continue until a blank line).
- **Trailing blank-line count differs** → every GTP response ends with `\n\n`. Confirm both sides emit exactly one blank line per response. The Swift harness already does this because `successResponse` / `errorResponse` both append `\n\n`. If the reference has only one `\n` at EOF, that's a KataGo quirk — `printf '\n'` the end of the reference when generating, not in the comparator.
- **Coordinates differ (`D4` vs `Q16` in unexpected places)** → your Swift `coordinateToGTP` uses `ySize - y` for the row, matching KataGo. Double-check by hand against one fixture.

Fix each failing fixture until the suite is green. Commit after each meaningful fix.

- [ ] **Step 3: Commit**

```bash
git add Tests/KataGoOnAppleSiliconIntegrationTests/GTPFixtureTests.swift
git commit -m "test: add GTP fixture integration tests"
```

---

## Task 16: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Features list and GTP command list**

In `README.md`, update line 10 (the GTP command list) to include the three new commands. Find:
```
- GTP commands: `protocol_version`, `name`, `version`, `known_command`, `list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `genmove`, `kata-set-rules`, `showboard`, `kata-rawnn`, `final_score`, `quit`
```
Replace with:
```
- GTP commands: `protocol_version`, `name`, `version`, `known_command`, `list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `genmove`, `undo`, `fixed_handicap`, `set_free_handicap`, `kata-set-rules`, `showboard`, `kata-rawnn`, `final_score`, `quit`
```

- [ ] **Step 2: Run final verification**

```bash
swift build
swift test
```

Expected: everything passes.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: list undo, fixed_handicap, set_free_handicap in README"
```

---

## Done-done checklist

- [ ] `swift build` is clean.
- [ ] `swift test --filter KataGoOnAppleSiliconTests` is green.
- [ ] `swift test --filter KataGoOnAppleSiliconIntegrationTests` is green — includes both the existing kata-rawnn tests and the 16 new fixture tests.
- [ ] `Scripts/generate_gtp_reference.sh` regenerates every reference reproducibly.
- [ ] `known_command undo`, `known_command fixed_handicap`, and `known_command set_free_handicap` all return `true`.
- [ ] README mentions the three new commands.
