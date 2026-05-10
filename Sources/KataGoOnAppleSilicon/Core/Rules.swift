// Go rules configuration
public struct Rules: Sendable, Equatable {
    public enum KoRule: Sendable {
        case simple
        case positional
        case situational
    }
    
    public enum ScoringRule: Sendable {
        case area
        case territory
    }
    
    // Ko rule encoding configuration for global features 6-7
    public let koRuleFlag1: Float  // Global feature 6
    public let koRuleFlag2: Float  // Global feature 7
    public let koRule: KoRule
    public let scoringRule: ScoringRule
    // Whether suicide (placing a stone that leaves the played group with zero
    // liberties after opponent captures resolve) is legal. Chinese, Japanese,
    // and Korean rules set this false; New Zealand rules set it true.
    public let multiStoneSuicideLegal: Bool

    // Initialize with all required fields
    public init(koRuleFlag1: Float, koRuleFlag2: Float, koRule: KoRule, scoringRule: ScoringRule, multiStoneSuicideLegal: Bool = false) {
        self.koRuleFlag1 = koRuleFlag1
        self.koRuleFlag2 = koRuleFlag2
        self.koRule = koRule
        self.scoringRule = scoringRule
        self.multiStoneSuicideLegal = multiStoneSuicideLegal
    }

    // Default rules (backward compatible with current implementation)
    // Uses values (1.0, 0.5) that match existing integration test references
    // Note: This is NOT proper Chinese rules - it's the default encoding for backward compatibility
    public static let defaultRules = Rules(
        koRuleFlag1: 1.0,
        koRuleFlag2: 0.5,
        koRule: .simple,
        scoringRule: .area,
        multiStoneSuicideLegal: false
    )

    // Chinese rules (proper Chinese rules per documentation)
    // Uses values (0.0, 0.0) as documented in InputFeatures.md
    // Note: Verify against C++ reference (nninputs.cpp lines 2613-2746) to confirm
    public static let chineseRules = Rules(
        koRuleFlag1: 0.0,
        koRuleFlag2: 0.0,
        koRule: .simple,
        scoringRule: .area,
        multiStoneSuicideLegal: false
    )
}

public extension Rules {
    /// SGF `RU[…]` value derived from this rules object. Mirrors KataGo's
    /// `Rules::toStringNoKomiMaybeNice()` for the two presets the engine
    /// currently models. Both names round-trip through KataGo's
    /// `Rules::tryParseRules`.
    var sgfName: String {
        if self == .chineseRules { return "Chinese" }
        return "Chinese-OGS"
    }
}