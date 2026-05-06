import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

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
    _ = handler.handleCommand("genmove black")
    _ = handler.handleCommand("clear_board")
    let response3 = handler.handleCommand("genmove black")
    #expect(response3.starts(with: "= "))
    #expect(response3 != "= resign\n\n")
    let response4 = handler.handleCommand("genmove black")
    #expect(response4 == "= resign\n\n")
}

@Test func testGenmoveColorIsolation() async throws {
    // Black is "losing" (winRate ≈ 0.16 < 0.5), White is "winning" (winRate ≈ 0.77 >= 0.5).
    // White's winning calls must NOT reset Black's consecutive-behind counter.
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 0.5, consecutiveMoves: 2)
    _ = handler.handleCommand("genmove black")
    let whiteResponse = handler.handleCommand("genmove white")
    #expect(whiteResponse != "= resign\n\n")
    let blackResponse = handler.handleCommand("genmove black")
    #expect(blackResponse == "= resign\n\n")
}

@Test func testGenmoveResignWhiteButNotBlack() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    let r1 = handler.handleCommand("genmove white")
    #expect(r1.starts(with: "= "))
    #expect(r1 != "= resign\n\n")
    let r2 = handler.handleCommand("genmove white")
    #expect(r2 == "= resign\n\n")
    let r3 = handler.handleCommand("genmove black")
    #expect(r3 != "= resign\n\n")
}

@Test func testGenmoveCounterResetsAfterResign() async throws {
    let handler = makeHandlerWithMock()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 2)
    _ = handler.handleCommand("genmove black")          // count=1
    let resign = handler.handleCommand("genmove black") // count=2 → resign, counter resets to 0
    #expect(resign == "= resign\n\n")
    // Counter is now 0; the next call needs count=1 < 2 to re-qualify — does NOT resign.
    let response = handler.handleCommand("genmove black")
    #expect(response != "= resign\n\n")
}

@Test func testFriendlyPassNotTriggeredAfterResign() async throws {
    let handler = makeHandlerWithFriendlyPass()
    handler.setResignThreshold(winRate: 1.0, consecutiveMoves: 1)
    _ = handler.handleCommand("play black pass")
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
    _ = handler.handleCommand("play black A1")
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassFlagResetAfterEvaluation() async throws {
    let handler = makeHandlerWithFriendlyPass()
    _ = handler.handleCommand("play black pass")
    let first = handler.handleCommand("genmove white")
    #expect(first == "= pass\n\n")
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
    _ = handler.handleCommand("play white pass")
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassRejectedByTightThresholds() async throws {
    // Negative thresholds are impossible to satisfy (abs diff >= 0 always),
    // so the guard always rejects the pass regardless of model output.
    let handler = makeHandlerWithFriendlyPass(winRateDelta: -1.0, leadDelta: -1.0)
    _ = handler.handleCommand("play black pass")
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}

@Test func testFriendlyPassSkippedBeforeMinimumTurn() async throws {
    let handler = makeHandlerWithFriendlyPass(minimumTurn: 10)
    _ = handler.handleCommand("play black pass")   // turnNumber → 1 (< 10)
    let response = handler.handleCommand("genmove white")
    #expect(response != "= pass\n\n")
    #expect(response.starts(with: "= "))
}
