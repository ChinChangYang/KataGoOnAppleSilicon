# Issue #14 — `testSGFParserSkipsParensInsidePropertyValues` doesn't actually exercise `skipBalancedTree`

**Status:** design
**Issue:** [#14 — SGF parser test coverage gap: parens-in-comment lives in the parsed branch, not the skipped one](https://github.com/ChinChangYang/KataGoOnAppleSilicon/issues/14)
**Branch:** `fix/issue-14-skip-balanced-tree-test` (off `master`)

## Problem

The test `testSGFParserSkipsParensInsidePropertyValues` (`Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift:245`) is named after `skipBalancedTree`'s behavior, but the SGF it parses places the parens-bearing comment inside the **first** child variation — which the parser descends and parses via `parseNode` / `parsePropertyValue`. The bracket-tracking logic inside `skipBalancedTree` only sees the **second** sibling, which contains no brackets or parens. A regression in `skipBalancedTree`'s `[…]` handling would not be caught.

Current SGF (escapes shown as in the Swift source):

```swift
"(;FF[4]GM[1]SZ[19];B[aa](;W[bb]C[note with ( and ) and \\] escape])(;W[cc]))"
```

Trace:

- `parseGameTree` enters with `depth = 1`, `firstChildTaken = false`.
- It descends into `(;W[bb]C[…])` — `firstChildTaken` is false, so this is the parsed branch. `parseNode` and `parsePropertyValue` see the parens-and-escape comment.
- After that branch closes, `firstChildTaken` becomes true.
- The next `(;W[cc])` is what reaches `skipBalancedTree` — and it has nothing interesting in it.

## Root cause

A test naming/data mismatch in the test fixture. No production-code defect. `skipBalancedTree` itself is correct as far as the existing tests demonstrate, but the test that should be its primary regression net doesn't exercise it.

## Decision

Two surgical edits in `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift`. No production code changes. No new files.

1. **Fix the existing test** by relocating the parens-bearing comment to the *skipped* sibling. This activates `skipBalancedTree`'s bracket-tracking and escape handling against the comment content.
2. **Add a sibling test** that exercises the one path *still* uncovered after edit 1 — the `(` → `depth += 1` branch inside `skipBalancedTree`, hit when the skipped subtree itself contains a nested variation.

## Coverage analysis

`skipBalancedTree` at `Sources/KataGoOnAppleSilicon/Core/SGFParser.swift:216` has six code paths:

| Path | Trigger | Covered after edit 1? | Covered after edits 1+2? | Mutation-killed by edit 1? | Mutation-killed by edit 2? |
|---|---|---|---|---|---|
| 1. `[` outside brackets | enter `[…]` mode | yes | yes | yes* | no |
| 2. `\` inside `[…]` | consume escaped char (2 bytes) | yes (the `\\]` in the comment) | yes | yes* | no |
| 3. `]` inside `[…]` | exit `[…]` mode | yes | yes | yes* | no |
| 4. any other char inside `[…]` | consume 1 byte | yes (comment body) | yes | no | no |
| 5. `(` outside `[…]` | `depth += 1` (nested variation) | **no** | yes (added by edit 2) | no | yes* |
| 6. `)` outside `[…]` | `depth -= 1`, return if zero | yes | yes | no | no |

After edit 1, path 5 (nested-paren depth tracking inside the skipped subtree) is the only remaining gap. Edit 2 covers it with a single small SGF.

Combining paths 2 and 5 into one mega-test was considered and rejected: it would obscure which path is broken when something fails. One path per test gives crisper diagnostics.

**Footnote:** After this iteration, paths 1, 2, 3, and 5 are not just executed but also mutation-killed by their respective tests (verified empirically via single-line reversions). Paths 4 (any-other-char-inside-brackets) and 6 (`)` exit) are exercised by every test in this section and don't have a meaningful mutation worth a dedicated test.

## Code

### Edit 1 — `Tests/KataGoOnAppleSiliconTests/GTPHandlerSGFTests.swift:245-252`

Strengthen the test by placing a stray `(` AFTER a `\]` escape inside the comment, with no matching `)` inside the brackets. Under correct bracket-mode and escape handling, both are invisible (consumed inside `[...]`). Under any of the three `[...]` mutations (entry, `\\` escape, or `]` exit), the `(` leaks out as a structural token, `skipBalancedTree`'s depth never returns to zero, and the parser throws "unterminated variation". Assertions are unchanged: both forms parse to the same main-line moves `[B[aa], W[bb]]`.

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

### Edit 2 — new test, inserted immediately after edit 1

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

If `skipBalancedTree` ever drops its `depth += 1` on inner `(`, it would return at the first inner `)`, leaving `;B[zz])` for the outer `parseGameTree` to parse as a top-level node. The outer parser then calls `makeMove` on coordinate `zz`, which is out of range for a 19x19 board, throwing `SGFParseError.malformed("invalid move coordinate 'zz'")` — the test fails loudly.

## Tests

### Verification

```
swift test --filter GTPHandlerSGFTests
```

Both edited and new tests must pass against the unchanged parser. If either fails, that itself is a finding (a real defect in `skipBalancedTree` that the prior weaker test had hidden).

### Existing tests

No other test in `GTPHandlerSGFTests` is affected. The existing `testSGFParserDescendsNestedVariations` (line 230) covers the orthogonal case where the *first* child contains a nested variation (parsed, not skipped), so it's not redundant with edit 2.

`SGFInteropTests` byte-equal fixtures are not touched — no parser behavior change, no `SGFGenerator` change, no GTP behavior change.

## Risk

Effectively zero. The change is test-only; the parser source is untouched. Worst-case downside: the new tests fail on the existing parser, which would mean we discovered a real bug — that's a *good* outcome for the test, not a regression.

## Non-goals

- Auditing the parser more broadly for other unhandled SGF escape semantics (soft line breaks `\\\n`, multiple property values `[v1][v2]`, brackets in non-comment text properties). Reserved for option C in brainstorming, deferred unless real issues surface.
- Refactoring `skipBalancedTree` itself.
- Adding property-based / fuzz tests for the SGF parser. Useful but well beyond this issue.
