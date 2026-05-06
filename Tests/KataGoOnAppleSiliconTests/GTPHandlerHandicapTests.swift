import Testing
import Foundation
import CoreML
@testable import KataGoOnAppleSilicon

// MARK: - fixed_handicap

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

// MARK: - set_free_handicap

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
    // KataGo always falls through to setStonesFailIfNoLibs which fails on
    // PASS_LOC and overwrites the response with "Handicap placement is invalid".
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap D4 pass") == "? Handicap placement is invalid\n\n")
}

@Test func testGTPSetFreeHandicapInvalidVertex() async throws {
    let katago = KataGoInference()
    let handler = GTPHandler(katago: katago)
    #expect(handler.handleCommand("set_free_handicap zz9") == "? Handicap placement is invalid\n\n")
    #expect(handler.handleCommand("set_free_handicap zz9 qq2") == "? Handicap placement is invalid\n\n")
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
