# Issue #14 — Exercise `skipBalancedTree` in test fixture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close issue #14 by relocating the parens-bearing SGF comment from the parsed branch to the skipped branch, and add a sibling test that exercises `skipBalancedTree`'s nested-paren depth tracking — the only `skipBalancedTree` code path still untouched after the relocation.

**Architecture:** Test-only changes. No production code is modified. Two edits in one test file.

**Tech Stack:** Swift 6.2, swift-testing. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-07-issue-14-skip-balanced-tree-test-design.md`

**Branch:** `fix/issue-14-skip-balanced-tree-test` (off `master`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift` | modify | Edit existing `testSGFParserSkipsParensInsidePropertyValues` (line 245) and add new `testSGFParserSkipsNestedVariationsInsideSkippedSibling` immediately after. |

No production source changes. No new files.

---

## TDD note for this plan

This is a test-quality fix, not a feature. The standard TDD red-green cycle ("write failing test → make it pass") doesn't apply because the production code (`SGFParser.skipBalancedTree`) is already correct as far as we know — what's broken is *test coverage*, not behavior.

The cycle for each task is therefore: **write the improved test → run it → it must PASS on the unchanged parser**. If a new test fails, that is itself a finding (a real parser bug previously hidden by the weaker test) — stop, report, and do not paper over it.

---

## Task 1: Relocate parens-bearing comment to the skipped sibling

Fix the test that motivates issue #14. The existing assertions remain valid because both SGF forms parse to the same main-line moves `[B[aa], W[bb]]`.

**Files:**
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift:245-252`

- [ ] **Step 1: Edit the test SGF and refresh the comment**

Replace the existing test body. The change relocates the parens-bearing comment from the *first* (parsed) child variation `(;W[bb]…)` into the *second* (skipped) child variation `(;W[cc]…)`, and strengthens its payload to an *imbalanced* form: a stray `(` placed AFTER a `\]` escape, with no matching `)` inside the brackets. Under correct handling both characters are consumed inside `[...]` and invisible. Under any of three bracket-mode mutations (`[` entry, `\\` escape, or `]` exit), the stray `(` leaks out as a structural token, `skipBalancedTree`'s depth never returns to zero, and the parser throws "unterminated variation" — so each of the three paths is *mutation-killed*, not merely line-executed.

Old (lines 245–252):

```swift
@Test func testSGFParserSkipsParensInsidePropertyValues() throws {
    // Parens inside a comment must not unbalance the variation-skip logic.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa](;W[bb]C[note with ( and ) and \\] escape])(;W[cc]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 2)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
}
```

New:

```swift
@Test func testSGFParserSkipsParensInsidePropertyValues() throws {
    // The skipped sibling's comment contains an escaped ']' followed by a
    // stray '('. Under correct handling both are inside [...] and invisible.
    // If either bracket-mode entry, '\\' escape, or ']' exit were broken, the
    // stray '(' would leak out as a structural token and skipBalancedTree
    // would never reach depth==0 — the parser throws "unterminated variation".
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa](;W[bb])(;W[cc]C[note with \\] stray (]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 2)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
}
```

- [ ] **Step 2: Run the modified test and verify it passes**

Run: `swift test --filter GTPHandlerSGFTests/testSGFParserSkipsParensInsidePropertyValues 2>&1 | tail -20`

Expected: PASS — `skipBalancedTree` correctly enters `[…]` mode, consumes the `\]` escape sequence, and stays inside brackets through the trailing stray `(` in the comment of the skipped second sibling.

If it fails: this would mean `skipBalancedTree`'s bracket-tracking or escape handling has a real defect that the previous (weaker) form of the test had hidden. Do **not** revert the SGF — escalate. The spec calls this out as an acceptable outcome.

- [ ] **Step 3: Commit**

```bash
git add Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift
git commit -m "$(cat <<'EOF'
test(sgf): mutation-kill skipBalancedTree's bracket handling on skipped-sibling comment (issue #14)

Relocate the parens-bearing comment from the parsed first child to the
skipped second child, and strengthen its payload to an imbalanced form
(stray '(' after a '\]' escape). Now the test fails under any of three
bracket-mode mutations — '[' entry, '\\' escape, or ']' exit — making
each path mutation-killed rather than merely line-executed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add coverage for nested-paren depth tracking in `skipBalancedTree`

After Task 1, the only `skipBalancedTree` code path still uncovered by tests is the `(` → `depth += 1` branch (path 5 in the spec's coverage table) — hit when the *skipped* subtree itself contains a nested variation. This task adds one focused test for that.

**Files:**
- Modify: `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift` (insert new test directly after `testSGFParserSkipsParensInsidePropertyValues`, before `testSGFParserHandlesDeeplyNestedVariations` — currently line 254).

- [ ] **Step 1: Add the new test**

Insert the following test immediately after the closing `}` of `testSGFParserSkipsParensInsidePropertyValues` and before `testSGFParserHandlesDeeplyNestedVariations`:

```swift
@Test func testSGFParserSkipsNestedVariationsInsideSkippedSibling() throws {
    // The skipped sibling itself contains a sub-variation followed by more
    // nodes. skipBalancedTree must increment depth on the inner '(' so it
    // doesn't return early at the inner ')' and leak the trailing ';B[zz]'
    // back into the outer parser as a top-level move.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa](;W[bb])(;W[cc](;B[dd]);B[zz]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 2)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
}
```

Trace (for the reviewer):
- Main line: `B[aa]` is parsed (move 1).
- First child: `(;W[bb])` is descended into; `W[bb]` is parsed (move 2).
- Second child: `(;W[cc](;B[dd]);B[zz])` is handed to `skipBalancedTree`. Inside, the inner `(;B[dd])` causes `depth` to climb to 2 and back to 1; the trailing `;B[zz]` is then consumed inside the still-open subtree; only the outermost `)` brings `depth` to 0 and triggers the return.

If `skipBalancedTree` ever loses its `depth += 1` on the inner `(`, it would return at the first inner `)`, leaving `;B[zz])` for `parseGameTree` to consume as a top-level node. The outer parser then calls `makeMove` on coordinate `zz`, which is out of range for a 19x19 board, throwing `SGFParseError.malformed("invalid move coordinate 'zz'")` — the test fails loudly. The `depth += 1` branch is therefore mutation-killed, not merely line-executed.

- [ ] **Step 2: Run the new test and verify it passes**

Run: `swift test --filter GTPHandlerSGFTests/testSGFParserSkipsNestedVariationsInsideSkippedSibling 2>&1 | tail -20`

Expected: PASS — `parsed.moves.count == 2`, with moves at (0,0) and (1,1).

If it fails: this indicates a real defect in nested-paren depth tracking inside `skipBalancedTree`. Stop and report; do not silently weaken the test.

- [ ] **Step 3: Run the full SGF-tests suite to confirm no regressions**

Run: `swift test --filter GTPHandlerSGFTests 2>&1 | tail -10`

Expected: all `GTPHandlerSGFTests` cases PASS, including the Task 1 fix and the new Task 2 test. Look specifically for the summary line indicating zero failures.

- [ ] **Step 4: Commit**

```bash
git add Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift
git commit -m "$(cat <<'EOF'
test(sgf): mutation-kill skipBalancedTree's nested-paren depth tracking (issue #14)

Add a regression test where the skipped sibling contains a sub-variation
followed by a trailing ';B[zz]' node. Under correct depth tracking the
entire sibling is skipped as one unit; under a removed 'depth += 1',
skipBalancedTree returns early at the inner ')' and the outer parser
sees ';B[zz]' as a top-level node — throwing
"invalid move coordinate 'zz'". The depth-increment branch is now
mutation-killed, not merely line-executed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (post-implementation)

Before opening a PR, the implementer should confirm:

- [ ] Both edits sit in the same file; no production code was touched (`git diff master --stat` shows only `GTPHandlerSGFTests.swift`).
- [ ] `swift test --filter GTPHandlerSGFTests` passes.
- [ ] No other test file was changed inadvertently.
- [ ] Commit messages reference issue #14.

## Out of scope

These are explicitly **not** part of this plan (deferred per the spec's non-goals):

- Auditing `SGFParser` for other unhandled escape semantics (`\\\n` soft line breaks, multiple property values `[v1][v2]`, etc.).
- Refactoring `skipBalancedTree` itself.
- Adding fuzz / property-based tests for the SGF parser.
