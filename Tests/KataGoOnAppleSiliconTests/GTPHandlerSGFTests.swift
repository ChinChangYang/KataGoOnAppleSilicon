import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - printsgf / loadsgf

private func sgfPayload(_ response: String) -> String? {
    guard response.hasPrefix("= ") else { return nil }
    var body = String(response.dropFirst(2))
    if body.hasSuffix("\n\n") {
        body = String(body.dropLast(2))
    }
    return body
}

@Test func testGTPPrintSGFEmptyBoard() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("printsgf")
    let sgf = try #require(sgfPayload(response))
    #expect(sgf.hasPrefix("(;FF[4]GM[1]SZ[19]"))
    #expect(sgf.contains("KM[7.5]"))
    #expect(sgf.hasSuffix(")"))
    #expect(!sgf.contains(";B["))
    #expect(!sgf.contains(";W["))
}

@Test func testGTPPrintSGFWithMovesAndPass() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    _ = handler.handleCommand("play white pass")
    let sgf = try #require(sgfPayload(handler.handleCommand("printsgf")))
    // D4 in GTP corresponds to (x=3, y=15) → SGF "dp".
    #expect(sgf.contains(";B[dp]"))
    #expect(sgf.contains(";W[]"))
}

@Test func testGTPPrintSGFRespectsBoardsize() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("boardsize 9")
    let sgf = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(sgf.contains("SZ[9]"))
}

@Test func testGTPPrintSGFEmitsHandicap() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("set_free_handicap D4 Q16")
    let sgf = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(sgf.contains("AB["))
    #expect(sgf.contains("[dp]"))
    #expect(sgf.contains("[pd]"))
    #expect(sgf.contains("HA[2]"))
    // Default for handicap is white-to-move (matches SGF default), so PL[] is omitted.
    #expect(!sgf.contains("PL["))
}

@Test func testGTPPrintSGFToFile() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    _ = handler.handleCommand("play black D4")
    let tmp = NSTemporaryDirectory() + "katago_test_\(UUID().uuidString).sgf"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    let response = handler.handleCommand("printsgf \(tmp)")
    #expect(response == "= \n\n")
    let written = try String(contentsOfFile: tmp, encoding: .utf8)
    #expect(written.contains(";B[dp]"))
}

@Test func testGTPListCommandsIncludesPrintSGFAndLoadSGF() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("list_commands")
    #expect(response.contains("printsgf"))
    #expect(response.contains("loadsgf"))
    #expect(handler.handleCommand("known_command printsgf") == "= true\n\n")
    #expect(handler.handleCommand("known_command loadsgf") == "= true\n\n")
}

@Test func testGTPLoadSGFBasic() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[6.5];B[dd];W[pp])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")

    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("KM[6.5]"))
    #expect(printed.contains(";B[dd]"))
    #expect(printed.contains(";W[pp]"))
}

@Test func testGTPLoadSGFMoveNumber() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[7.5];B[dd];W[pp];B[pd];W[dp])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    // movenumber=3 means "position right before move 3", so two moves are played.
    #expect(handler.handleCommand("loadsgf \(tmp) 3") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains(";B[dd]"))
    #expect(printed.contains(";W[pp]"))
    #expect(!printed.contains(";B[pd]"))
}

@Test func testGTPLoadSGFHandicap() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[0.5]HA[2]AB[dd][pp];W[qd])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("AB[dd][pp]"))
    #expect(printed.contains(";W[qd]"))
    #expect(printed.contains("HA[2]"))
}

@Test func testGTPLoadSGFPassEncoding() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    // Modern empty pass + legacy "tt" pass on 19x19.
    let sgf = "(;FF[4]GM[1]SZ[19]KM[7.5];B[];W[tt])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains(";B[]"))
    #expect(printed.contains(";W[]"))
}

@Test func testGTPLoadSGFRulesName() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[19]KM[7.5]RU[Chinese])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("RU[Chinese]"))
}

@Test func testGTPLoadSGFMissingFile() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let response = handler.handleCommand("loadsgf /nonexistent/path/missing.sgf")
    #expect(response.hasPrefix("? "))
}

@Test func testGTPLoadSGFNoArgument() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("loadsgf") == "? syntax error\n\n")
}

@Test func testGTPLoadSGFMalformed() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try "not actually an sgf".write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    let response = handler.handleCommand("loadsgf \(tmp)")
    #expect(response.hasPrefix("? "))
}

@Test func testGTPLoadSGFRoundTripPreservesBoardSize() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    let sgf = "(;FF[4]GM[1]SZ[9]KM[7.5];B[ee];W[ff])"
    let tmp = NSTemporaryDirectory() + "katago_load_\(UUID().uuidString).sgf"
    try sgf.write(toFile: tmp, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(atPath: tmp) }

    #expect(handler.handleCommand("loadsgf \(tmp)") == "= \n\n")
    let printed = try #require(sgfPayload(handler.handleCommand("printsgf")))
    #expect(printed.contains("SZ[9]"))
}

// MARK: - SGFParser unit tests

@Test func testSGFParserBasicProperties() throws {
    let sgf = "(;FF[4]GM[1]SZ[19]KM[6.5]PB[Alice]PW[Bob]RU[Japanese];B[dd];W[pp])"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.boardSize == 19)
    #expect(parsed.komi == 6.5)
    #expect(parsed.rulesName == "Japanese")
    #expect(parsed.blackPlayer == "Alice")
    #expect(parsed.whitePlayer == "Bob")
    #expect(parsed.moves.count == 2)
}

@Test func testSGFParserDescendsFirstChildVariation() throws {
    // Per the SGF grammar, the main line continues into the first child of
    // every fork; sibling variations are dropped. Here B[cc];W[dd] are the
    // first child after W[bb], so they're part of the main line; B[ee] is a
    // sibling and is skipped.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa];W[bb](;B[cc];W[dd])(;B[ee]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 4)
    #expect(parsed.moves[0].player == .black)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].player == .white)
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
    #expect(parsed.moves[2].player == .black)
    #expect(parsed.moves[2].location == Point(x: 2, y: 2))
    #expect(parsed.moves[3].player == .white)
    #expect(parsed.moves[3].location == Point(x: 3, y: 3))
}

@Test func testSGFParserDropsSiblingVariations() throws {
    // Two siblings after A; only the first child (B) extends the main line.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa](;W[bb])(;W[cc]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 2)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
}

@Test func testSGFParserDescendsNestedVariations() throws {
    // Nested forks: the main line is A;B;C;D;E. Siblings F and G are dropped.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa];W[bb](;B[cc];W[dd](;B[ee])(;B[ff]))(;B[gg]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 5)
    let coords = parsed.moves.map { $0.location }
    #expect(coords == [
        Point(x: 0, y: 0),
        Point(x: 1, y: 1),
        Point(x: 2, y: 2),
        Point(x: 3, y: 3),
        Point(x: 4, y: 4),
    ])
}

@Test func testSGFParserSkipsParensInsidePropertyValues() throws {
    // Parens inside a comment must not confuse the variation-skip logic.
    let sgf = "(;FF[4]GM[1]SZ[19];B[aa](;W[bb]C[note with ( and ) and \\] escape])(;W[cc]))"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == 2)
    #expect(parsed.moves[0].location == Point(x: 0, y: 0))
    #expect(parsed.moves[1].location == Point(x: 1, y: 1))
}

@Test func testSGFParserHandlesDeeplyNestedVariations() throws {
    // Each move sits in its own nested first-child branch — pathological but
    // exercises the iterative tokenizer's resilience to deep nesting without
    // call-stack growth. 2_000 levels would blow a recursive parser.
    let depth = 2_000
    var sgf = "(;FF[4]GM[1]SZ[19]"
    for _ in 0..<depth { sgf += "(;B[aa]" }
    for _ in 0..<depth { sgf += ")" }
    sgf += ")"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.moves.count == depth)
}

@Test func testSGFParserHandicapStones() throws {
    let sgf = "(;FF[4]GM[1]SZ[19]HA[2]AB[dd][pp];W[qd])"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.handicap == 2)
    #expect(parsed.initialBlack.count == 2)
    #expect(parsed.initialBlack.contains(Point(x: 3, y: 3)))
    #expect(parsed.initialBlack.contains(Point(x: 15, y: 15)))
    #expect(parsed.moves.count == 1)
    #expect(parsed.moves[0].player == .white)
}

@Test func testSGFParserIgnoresUnknownProperties() throws {
    let sgf = "(;FF[4]GM[1]SZ[19]C[a comment with brackets \\[escaped\\]]GN[Game];B[aa])"
    let parsed = try SGFParser.parse(sgf)
    #expect(parsed.boardSize == 19)
    #expect(parsed.moves.count == 1)
}

@Test func testSGFParserEmptyInputRejected() throws {
    let response = try? SGFParser.parse("")
    #expect(response == nil)
}
