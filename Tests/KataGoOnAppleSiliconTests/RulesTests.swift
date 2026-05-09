import Testing
@testable import KataGoOnAppleSilicon

@Test func testRulesSgfNameForChineseRules() {
    #expect(Rules.chineseRules.sgfName == "Chinese")
}

@Test func testRulesSgfNameForDefaultRules() {
    // defaultRules has koRuleFlag (1.0, 0.5) → KO_POSITIONAL, area scoring,
    // multiStoneSuicideLegal=false. Among KataGo's named presets, only
    // Chinese-OGS matches all four Swift-tracked fields exactly. (TrompTaylor
    // has multiStoneSuicideLegal=true, so it doesn't fit.)
    #expect(Rules.defaultRules.sgfName == "Chinese-OGS")
}

@Test func testRulesEquatable() {
    #expect(Rules.chineseRules == Rules.chineseRules)
    #expect(Rules.defaultRules == Rules.defaultRules)
    #expect(Rules.chineseRules != Rules.defaultRules)
}
