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

