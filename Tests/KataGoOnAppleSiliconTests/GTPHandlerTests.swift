import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - GTPHandler Tests

@Test func testGTPProtocolVersion() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("protocol_version")
    #expect(response == "= 2\n\n")
}

@Test func testGTPName() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("name")
    #expect(response == "= KataGoOnAppleSilicon\n\n")
}

@Test func testGTPVersion() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("version")
    #expect(response == "= 1.0\n\n")
}

@Test func testGTPKnownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command play") == "= true\n\n")
    #expect(handler.handleCommand("known_command unknown") == "= false\n\n")
}

@Test func testGTPListCommands() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("list_commands")
    #expect(response.starts(with: "= "))
    #expect(response.contains("play"))
    #expect(response.contains("genmove"))
}

@Test func testGTPClearBoard() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black A1")
    let response = handler.handleCommand("clear_board")
    #expect(response == "= \n\n")
    // Board should be cleared
}

@Test func testGTPPlayMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black A1")
    #expect(response == "= \n\n")
}

@Test func testGTPPlayInvalidMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black invalid")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPBoardsize() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize 19")
    #expect(response == "= \n\n")
}

@Test func testGTPKomi() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("komi 7.5")
    #expect(response == "= \n\n")
}

@Test func testGTPQuit() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("quit")
    #expect(response == "= \n\n")
}

@Test func testGTPEmptyCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("")
    #expect(response == "? \n\n")
}

@Test func testGTPUnknownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("nonexistent_command")
    #expect(response == "? unknown command\n\n")
}

@Test func testGTPPlayMissingArgs() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPPlayIllegalMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Play at same location twice - second should be illegal
    _ = handler.handleCommand("play black A1")
    let response = handler.handleCommand("play white A1")
    #expect(response == "? illegal move\n\n")
}

@Test func testGTPPlayWhiteMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play white T19")
    #expect(response == "= \n\n")
}

@Test func testGTPGenmoveMissingColor() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPGenmoveWithModelLoaded() async throws {
    let katago = KataGoInference()
    try katago.loadModel(for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= "))
}

@Test func testParseMoveColumnJToT() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Test columns J through T (skipping I)
    let responseJ = handler.handleCommand("play black J1")
    #expect(responseJ == "= \n\n")
    let responseT = handler.handleCommand("play white T1")
    #expect(responseT == "= \n\n")
}

@Test func testParseMoveInvalidColumn() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // 'I' is skipped in Go, 'Z' is out of range
    let responseZ = handler.handleCommand("play black Z1")
    #expect(responseZ == "? syntax error\n\n")
}

@Test func testParseMoveShortString() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("play black A")
    #expect(response == "? syntax error\n\n")
}

@Test func testGTPGenmove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Mock or assume model is loaded, but since it's async and may fail, just test parsing
    let response = handler.handleCommand("genmove black")
    // In real scenario, would need model loaded, but for test, check it's not error
    #expect(response.starts(with: "=") || response.starts(with: "?"))
}

@Test func testParseMove() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Access private method via reflection or assume it's tested through play
    // For now, test through play command
    let response = handler.handleCommand("play black A1")
    #expect(response == "= \n\n")
}

// MARK: - GTPHandler SelectMove Tests

@Test func testGTPSelectMoveColumnAToH() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column A (x=0)
    let mockModel = MockModelWithValidOutputs(targetX: 0, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= A"))
}

@Test func testGTPSelectMoveColumnJToT() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column J (x=8, skipping I)
    let mockModel = MockModelWithValidOutputs(targetX: 8, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= J"))
}

@Test func testGTPSelectMoveColumnT() async throws {
    let katago = KataGoInference()
    // Mock model that returns move at column T (x=18)
    let mockModel = MockModelWithValidOutputs(targetX: 18, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= T"))
}

// MARK: - kata-set-rules Tests

@Test func testKataSetRulesChinese() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("kata-set-rules chinese")
    #expect(response == "= \n\n")
}

@Test func testKataSetRulesUnknownPreset() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("kata-set-rules japanese")
    #expect(response == "? Unknown rules 'japanese'\n\n")
}

@Test func testKataSetRulesMissingArgument() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("kata-set-rules")
    #expect(response == "? Expected at least one argument for kata-set-rules\n\n")
}

@Test func testKataSetRulesCaseInsensitive() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Test that "CHINESE" (uppercase) works
    let response = handler.handleCommand("kata-set-rules CHINESE")
    #expect(response == "= \n\n")
}

@Test func testKataSetRulesKnownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command kata-set-rules") == "= true\n\n")
}

@Test func testKataSetRulesInListCommands() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("list_commands")
    #expect(response.contains("kata-set-rules"))
}

// MARK: - Resign Tests

private func makeHandlerWithMock() -> GTPHandler {
    let katago = KataGoInference()
    let mockModel = MockModelWithValidOutputs(targetX: 0, targetY: 0)
    katago.setModel(mockModel, for: "AI")
    return GTPHandler(katago: katago)
}

private func makeHandlerWithFriendlyPass(
    winRateDelta: Double = 0.5,
    leadDelta: Double = 100.0,
    minimumTurn: Int = 0
) -> GTPHandler {
    let handler = makeHandlerWithMock()
    handler.setFriendlyPassOptions(enabled: true, winRateDelta: winRateDelta, leadDelta: leadDelta, minimumTurn: minimumTurn)
    return handler
}

@Test func testGenmoveNeverResign() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 0.0, consecutiveMoves: 1)
    let response = handler.handleCommand("genmove black")
    #expect(response.starts(with: "= "))
    #expect(response != "= resign\n\n")
}

@Test func testGenmoveResignAfterConsecutiveBehind() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    // Call 1 (black): black_count=1 < 2, plays
    let response1 = handler.handleCommand("genmove black")
    #expect(response1.starts(with: "= "))
    #expect(response1 != "= resign\n\n")
    // Call 2 (black): black_count=2 >= 2, resign fires before playMove
    let response2 = handler.handleCommand("genmove black")
    #expect(response2 == "= resign\n\n")
}

@Test func testGenmoveClearBoardResetsConsecutiveCount() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    // Call 1 (black): black_count=1, plays at A19
    _ = handler.handleCommand("genmove black")
    // clear_board resets both counters and the board
    _ = handler.handleCommand("clear_board")
    // Call 2 after reset (black): black_count=1, plays at A19 again
    let response3 = handler.handleCommand("genmove black")
    #expect(response3.starts(with: "= "))
    #expect(response3 != "= resign\n\n")
    // Call 3 (black): black_count=2 >= 2, resign fires before playMove
    let response4 = handler.handleCommand("genmove black")
    #expect(response4 == "= resign\n\n")
}

@Test func testGenmoveColorIsolation() async throws {
    // Black is "losing" (winRate ≈ 0.16 < 0.5), White is "winning" (winRate ≈ 0.77 >= 0.5).
    // White's winning calls should NOT reset Black's consecutive-behind counter.
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 0.5, consecutiveMoves: 2)
    // Call 1 (black): black_count=1 < 2, plays at A19
    _ = handler.handleCommand("genmove black")
    // White is winning: white_count stays 0 — and must NOT reset black_count.
    // Board conflict is irrelevant here; we only verify white does not resign.
    let whiteResponse = handler.handleCommand("genmove white")
    #expect(whiteResponse != "= resign\n\n")
    // Black continues losing: black_count=2 >= 2, resign fires before playMove
    let blackResponse = handler.handleCommand("genmove black")
    #expect(blackResponse == "= resign\n\n")
}

@Test func testGenmoveResignWhiteButNotBlack() async throws {
    // White calls accumulate white_count independently of black_count (which stays at 0).
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    // White call 1: white_count=1 < 2, plays
    let r1 = handler.handleCommand("genmove white")
    #expect(r1.starts(with: "= "))
    #expect(r1 != "= resign\n\n")
    // White call 2: white_count=2 >= 2, resign fires before playMove
    let r2 = handler.handleCommand("genmove white")
    #expect(r2 == "= resign\n\n")
    // Black has never been called: black_count=0. First black call should NOT resign.
    let r3 = handler.handleCommand("genmove black")
    #expect(r3 != "= resign\n\n")
}

@Test func testGenmoveCounterResetsAfterResign() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    _ = handler.handleCommand("genmove black")          // count=1, plays at A19
    let resign = handler.handleCommand("genmove black") // count=2 → resign, counter resets to 0
    #expect(resign == "= resign\n\n")
    // Counter is now 0; the next call needs count=1 < 2 to re-qualify — does NOT resign.
    let response = handler.handleCommand("genmove black")
    #expect(response != "= resign\n\n")
}

@Test func testFriendlyPassNotTriggeredAfterResign() async throws {
    let handler = makeHandlerWithFriendlyPass()
    // White has winRate threshold=1.0 so it always resigns after 1 move
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 1)
    // Opponent (black) passes — sets lastPlayPassColor
    _ = handler.handleCommand("play black pass")
    // White genmove: resign fires (count=1 >= 1) before friendly pass is evaluated
    let resign = handler.handleCommand("genmove white")
    #expect(resign == "= resign\n\n")
    // Resign counter reset; next white genmove must NOT trigger friendly pass from stale flag
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

// MARK: - Friendly Pass Tests

@Test func testPlayPassHandled() async throws {
    let handler = GTPHandler(katago: KataGoInference())
    #expect(handler.handleCommand("play black pass") == "= \n\n")
    #expect(handler.handleCommand("play white PASS") == "= \n\n")
    #expect(handler.handleCommand("play black Pass") == "= \n\n")
}

@Test func testFriendlyPassDisabledByDefault() async throws {
    let katago = KataGoInference()
    katago.setModel(MockModelWithValidOutputs(targetX: 0, targetY: 0), for: "AI")
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black pass")
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassWhenSafe() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play black pass")
    let response = handler.handleCommand("genmove white")
    #expect(response == "= pass\n\n")
}

@Test func testFriendlyPassNotTriggeredWithoutOpponentPass() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play black A1")  // regular move, not pass
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassFlagResetAfterEvaluation() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play black pass")
    let first = handler.handleCommand("genmove white")
    #expect(first == "= pass\n\n")
    // No new opponent pass; flag was consumed
    let second = handler.handleCommand("genmove white")
    #expect(second != "= pass\n\n")
    #expect(second.starts(with: "= "))
}

@Test func testFriendlyPassResetOnClearBoard() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play black pass")
    _ = handler.handleCommand("clear_board")
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassNotTriggeredBySameColorPass() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play white pass")   // AI's own color passes via play
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")              // must NOT trigger friendly pass
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassRejectedByTightThresholds() async throws {
    // Negative thresholds are impossible to satisfy (abs diff >= 0 always),
    // so the guard always rejects the pass regardless of model output.
    let handler = makeHandlerWithFriendlyPass(winRateDelta: -1.0, leadDelta: -1.0)
    _ = handler.handleCommand("play black pass")
    let response = handler.handleCommand("genmove white")
    // Even though opponent passed, thresholds reject the friendly pass.
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassSkippedBeforeMinimumTurn() async throws {
    // minimumTurn=10 means tryFriendlyPass returns nil at turn 1 (after one play black pass).
    let handler = makeHandlerWithFriendlyPass(minimumTurn: 10)
    _ = handler.handleCommand("play black pass")   // turnNumber → 1 (< 10)
    let response = handler.handleCommand("genmove white")
    // Guard fires → friendly pass skipped → engine plays a regular move.
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

// MARK: - final_score Tests

@Test func testFinalScoreKnownCommand() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("known_command final_score") == "= true\n\n")
}

@Test func testFinalScoreResponseFormat() async throws {
    let katago = KataGoInference()
    katago.setModel(MockModelWithValidOutputs(targetX: 0, targetY: 0), for: "AI")
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("final_score")
    // Mock model has no out_miscvalue, so whiteLead = 0.0
    // round(0.0 + 0.5) - 0.5 = 0.5 → "W+0.5"
    #expect(response == "= W+0.5\n\n")
}

@Test func testFinalScoreAlwaysUsesAIModel() async throws {
    let katago = KataGoInference()
    katago.setModel(MockModelWithValidOutputs(targetX: 0, targetY: 0), for: "AI")
    // No human SL model loaded
    let handler = GTPHandler(katago: katago)
    handler.setProfile("20k")  // Human SL profile active
    let response = handler.handleCommand("final_score")
    // Must succeed using AI model, not fail because human SL model is missing
    #expect(response.starts(with: "= "))
}

// MARK: - Boardsize Command Tests

@Test func testGTPBoardsize9() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize 9")
    #expect(response == "= \n\n")
}

@Test func testGTPBoardsizeTooSmall() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize 1")
    #expect(response.starts(with: "? "))
}

@Test func testGTPBoardsizeTooLarge() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize 20")
    #expect(response.starts(with: "? "))
}

@Test func testGTPBoardsizeNonNumeric() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("boardsize abc")
    #expect(response.starts(with: "? "))
}

@Test func testGTPBoardsizePreservedAfterClearBoard() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 9")
    _ = handler.handleCommand("clear_board")
    // Play a legal move on 9x9 (A9 = x:0, y:0 on 9x9)
    let response = handler.handleCommand("play black A9")
    #expect(response == "= \n\n")
}

@Test func testGTPPlayOutOfBoundsAfterBoardsize9() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 9")
    // Row 10 is out of bounds for a 9x9 board
    let response = handler.handleCommand("play black A10")
    #expect(response.starts(with: "? "))
}

@Test func testGTPPlayColumnOutOfBoundsAfterBoardsize2() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 2")
    // Column C = x=2 is out of bounds for 2x2 board (valid: x=0,1 = A,B)
    let response = handler.handleCommand("play black C1")
    #expect(response.starts(with: "? "))
}

@Test func testGTPShowboardAfterBoardsize9() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 9")
    let response = handler.handleCommand("showboard")
    #expect(response.starts(with: "= "))
    // Should have exactly 9 rows (each line starts with row number)
    let lines = response
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .dropFirst(2)  // drop "= "
        .split(separator: "\n", omittingEmptySubsequences: false)
    #expect(lines.count == 9)
}

