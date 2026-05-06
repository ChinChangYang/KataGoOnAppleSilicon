# SGF Interop Test for `loadsgf` / `printsgf` (PR #6 follow-up)

**Status:** Design approved — ready for implementation plan.
**Date:** 2026-04-26
**Scope:** Add cross-engine SGF interoperability tests for the `loadsgf` and
`printsgf` GTP commands introduced in PR #6.

## Goal

Prove that the SGF dialect produced and consumed by Swift's `GTPHandler`
agrees with the reference implementation (KataGo). Specifically:

1. **Export direction.** SGF emitted by our `printsgf` is loadable by KataGo
   without a `?` reply.
2. **Import direction.** SGF emitted by KataGo's `printsgf` is loadable by
   our `loadsgf` and produces the same canonical position.

The existing `GTPHandlerSGFTests.swift` already covers Swift-internal
round-trip. This work adds the missing cross-engine guarantee.

## Non-goals

- gnugo (or any other engine) — KataGo only.
- Free / mixed setup stones beyond what `fixed_handicap` produces.
- `loadsgf <file> <move_n>` truncation — covered by existing in-Swift tests.
- Boards other than 19×19.
- Resign / draw SGFs.
- CI-side fixture regeneration. Fixtures are committed; CI just runs
  `swift test`. The generator is developer-run, like
  `generate_kata_raw_nn_reference.sh`.

## Design summary

**One canonical SGF per scenario.** Each scenario commits a single file
`<scenario>.sgf` whose authority comes from passing KataGo's `loadsgf`
offline. Both test directions reduce to byte-comparison against this file:

- **Direction 1 (export):** replay the scenario via Swift in-test →
  `printsgf` → byte-equal to `<scenario>.sgf`.
- **Direction 2 (import):** in-test, `loadsgf <scenario>.katago.sgf` →
  `printsgf` → byte-equal to `<scenario>.sgf`.

This is symmetric and uses the canonical SGF as the position fingerprint
for both directions. The semantic check from Q2 (after-load position
matches expected) is realized as "after-load `printsgf` equals canonical."

## File layout (new)

```
Scripts/
├── SGFFixtureDrivers/                 # position-setup .gtp scripts
│   ├── empty.gtp
│   ├── moves_basic.gtp
│   ├── pass_midgame.gtp
│   ├── handicap_5.gtp
│   ├── komi_nondefault.gtp
│   ├── rules_chinese.gtp
│   └── captures.gtp
└── generate_sgf_interop_fixtures.sh

Tests/KataGoOnAppleSiliconIntegrationTests/SGFFixtures/
├── <scenario>.sgf                     # our printsgf, validated by KataGo
└── <scenario>.katago.sgf              # KataGo's printsgf at the same position

Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift
Sources/GTPRunner/main.swift
Package.swift                          # +executableTarget("GTPRunner")
                                       # +executable product "GTPRunner"
```

Drivers describe positions, not operations. The generator appends
`printsgf <tmp>` + `quit` itself, so the same driver feeds both directions.

## `GTPRunner` executable

A new minimal executable target that pipes stdin through `GTPHandler` to
stdout. Reuses the loop already used in `Tests/.../GTPFixtureTests.swift`.

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

`Package.swift` additions, sibling to the existing `KataGoPlay` target:

```swift
.executable(name: "GTPRunner", targets: ["GTPRunner"]),
// ...
.executableTarget(
    name: "GTPRunner",
    dependencies: ["KataGoOnAppleSilicon"],
    path: "Sources/GTPRunner"
)
```

The new target is required, not optional — the generator script depends on
it to produce committed canonical SGFs offline.

## Generator script

`Scripts/generate_sgf_interop_fixtures.sh`. Per-scenario flow:

1. **Build canonical SGF.** Concatenate `<driver>.gtp` + `printsgf <tmp1>` +
   `quit`, pipe to `swift run GTPRunner`, read `$tmp1`. Save as
   `Tests/.../SGFFixtures/<scenario>.sgf`.
2. **Validate export direction.** Run `loadsgf <canonical>` + `quit`
   through KataGo. Abort the script with a red error if KataGo emits `?`.
3. **Build engine SGF.** Concatenate `<driver>.gtp` + `printsgf <tmp2>` +
   `quit`, pipe to KataGo, read `$tmp2`. Save as
   `Tests/.../SGFFixtures/<scenario>.katago.sgf`.
4. **Validate import direction (sanity).** Run
   `loadsgf <engine.sgf>` + `printsgf <tmp3>` + `quit` through
   `swift run GTPRunner`. Abort if `tmp3` differs from
   `<scenario>.sgf`. Same check runs at test time, but failing here gives
   a clearer locus during regen.

The script's clean exit is the validation record. No separate
`<scenario>.validators.txt` file.

**KataGo-invocation parity.** The script invokes KataGo with the exact
flags `Scripts/generate_gtp_reference.sh:96-103` already uses:
`-config gtp_example.cfg`, `-model <bin>`, `-coreml-model <mlpackage>`,
stderr redirected to `/dev/null`. Differing flags risk different
`printsgf` formatting (e.g., `AP[…]` engine-name).

**Reuse of build infra.** KataGo location/build is already centralized in
`Scripts/generate_kata_raw_nn_reference.sh`. The new script delegates the
same way — `bash "$SCRIPT_DIR/generate_kata_raw_nn_reference.sh"` if
`$KATAGO_EXE` is missing, then proceeds. No duplicated build code.

**Flags (mirroring `generate_gtp_reference.sh`):**

- `--scenario <name>` — regen one scenario only.
- `--force-rebuild` — pass through to the build delegate.

**`printsgf` always takes a path.** Both Swift's and KataGo's `printsgf`
accept either a stdout form or a `<filename>` form. The generator always
uses the path form and reads the file back — bypasses GTP framing concerns
when capturing multi-line SGF from a stdout stream. Tests use the stdout
form in-process where there is no framing issue.

## Test file

`Tests/KataGoOnAppleSiliconIntegrationTests/SGFInteropTests.swift`. Two
`@Test` cases per scenario:

```swift
@Test func interop_export_<scenario>() throws {
    let canonical = try loadFixture("<scenario>.sgf")
    let driver    = try loadDriver("<scenario>.gtp")

    let handler = GTPHandler(katago: KataGoInference())
    for line in driver.split(whereSeparator: \.isNewline) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        _ = handler.handleCommand(trimmed)
    }
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))

    #expect(printed == canonical, diffMessage(printed, canonical))
}

@Test func interop_import_<scenario>() throws {
    let canonical = try loadFixture("<scenario>.sgf")
    let engineSGF = try fixtureURL("<scenario>.katago.sgf")

    let handler = GTPHandler(katago: KataGoInference())
    #expect(handler.handleCommand("loadsgf \(engineSGF.path)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))

    #expect(printed == canonical, diffMessage(printed, canonical))
}
```

**Helpers (file-local):**

- `loadFixture(name) -> String` — reads `Tests/.../SGFFixtures/<name>` via
  the `repoURL(subdir:name:ext:)` walk-up-tree pattern from
  `GTPFixtureTests.swift:23-34`.
- `loadDriver(name) -> String` — same, for `Scripts/SGFFixtureDrivers/`.
- `sgfPayload(_:)` — already exists in `GTPHandlerSGFTests.swift`; lift to
  a fileprivate helper, or extract to a shared test helper.
- `diffMessage(_:_:)` — short side-by-side dump on mismatch (mirrors the
  `--- Swift / --- Reference` style in `GTPFixtureTests.swift`).

**Fixture discovery — committed, not bundled.** Fixtures are read from the
repo via the walk-up-tree helper, not the test bundle's `resources`. This
keeps generator output and test inputs in the same on-disk location and
avoids `Package.swift` resources churn. No new `resources` entry.

**Skip-if-missing.** A missing canonical fixture throws a
`fixtureMissing` error pointing at
`Scripts/generate_sgf_interop_fixtures.sh`. Same UX as the existing
`KataRawNNIntegrationTests` and `GTPFixtureTests`.

**Test count.** 7 scenarios × 2 directions = **14 tests.**

## Test matrix

The seven approved scenarios. Each driver is 3–10 lines.

| Scenario | Driver contents | Bug class it pins |
|---|---|---|
| `empty` | `boardsize 19` + `komi 7.5` | Header round-trip baseline (FF/GM/SZ/KM/CA/AP) |
| `moves_basic` | 4–6 alternating `play B …` / `play W …` | Move-list `;B[xx];W[yy]` ordering and coords |
| `pass_midgame` | Plays + `play W pass` + plays | `B[]` / `W[]` empty-coord encoding |
| `handicap_5` | `boardsize 19` + `fixed_handicap 5` | `HA[5]` + `AB[…]` group + `PL[W]` |
| `komi_nondefault` | `komi 6.5` + a few plays | `KM[6.5]` formatting, no trailing-zero collisions |
| `rules_chinese` | `kata-set-rules chinese` + a few plays | `RU[Chinese]` round-trip (KataGo-specific extension) |
| `captures` | Sequence ending in a 1-stone capture | Replay engine state, not just move list |

## Acceptance criteria

1. Fresh checkout: `Scripts/generate_sgf_interop_fixtures.sh` runs to
   completion (after KataGo is built once via
   `generate_kata_raw_nn_reference.sh`), producing 7 canonical + 7 KataGo
   SGFs.
2. `swift test --filter SGFInteropTests` passes all 14 tests against the
   committed fixtures.
3. Mutating `printsgf` to omit `PL[W]` for handicap breaks exactly
   `interop_export_handicap_5` and `interop_import_handicap_5`, with a
   readable diff.
4. Mutating `loadsgf` to ignore `B[]` breaks exactly the two
   `pass_midgame` tests.

## Alternatives considered

- **Single round-trip generator producing a combined verdict file.**
  Couples the test to KataGo's `printsgf` cosmetic dialect — drift in
  KataGo's emission formatting forces fixture regeneration even when
  interop is fine. Rejected.
- **Extend `generate_gtp_reference.sh` and `GTPFixtureTests`.** The
  existing fixture pattern is "diff GTP-script output bytes." This new
  test is "load committed SGF, then probe" — a different test shape.
  Bolting it into the same file mixes shapes and confuses future readers.
  Rejected.
- **Live cross-engine round-trip at test time.** Each `swift test` would
  spawn KataGo and drive it via stdin/stdout. KataGo cold start is slow,
  every dev's machine becomes a dependency, and CI would need the binary.
  Rejected in favor of offline fixtures (existing repo pattern).
- **Skip the new `GTPRunner` target; produce canonical SGF from a
  one-shot `@Test` that writes to disk.** Mixes "generate fixture" and
  "consume fixture" responsibilities into the test target, and produces
  fixtures via a path the user has to remember to run. The shell-shaped
  generator is the established pattern. Rejected.

## Implementation watchpoints

Surfaced during spec review; the implementation plan should resolve these
explicitly, not paper over them:

- **Engine-specific properties in the engine SGF.** KataGo's `printsgf`
  emits `AP[katago:…]` and may emit `PB` / `PW` / `DT` / `CA`. Per the
  PR #6 description, our `loadsgf` honours `PB` and `PW`, so direction 2
  may fail byte-match because the round-tripped SGF carries KataGo's
  player-name strings instead of our defaults. The implementation plan
  must inspect the actual `printsgf` output of both engines for each
  scenario and decide whether to (a) align defaults, (b) sanitize the
  engine SGF in the generator (strip non-load-honoured properties before
  committing), or (c) compare a normalized projection rather than raw
  bytes. Decision deferred to the implementation plan because it requires
  observing actual byte output that doesn't exist yet.
- **`AP` stability across our own runs.** Direction 1's byte equality
  requires our `printsgf` `AP[…]` to be deterministic. If `AP` includes a
  package version pulled from `Bundle` or git, the canonical SGF will
  drift on every release. The plan should verify our `AP` is stable, and
  fix it if not.

## Open questions

None — all design questions resolved during brainstorming. Watchpoints
above are explicit handoffs to implementation, not unresolved design
choices.
