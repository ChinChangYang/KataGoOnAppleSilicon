import Foundation

/// Result of parsing an SGF file: the game-info root properties plus the
/// linear sequence of moves from the main variation.
public struct ParsedSGF {
    public let boardSize: Int
    public let komi: Float
    public let rulesName: String?
    public let blackPlayer: String?
    public let whitePlayer: String?
    public let handicap: Int?
    public let initialBlack: [Point]
    public let initialWhite: [Point]
    public let initialSideToMove: Stone?
    public let moves: [Move]
}

/// Errors raised while parsing an SGF document.
public enum SGFParseError: LocalizedError {
    case malformed(String)

    public var errorDescription: String? {
        switch self {
        case .malformed(let msg): return "Malformed SGF: \(msg)"
        }
    }
}

/// Minimal SGF parser sufficient for `loadsgf`.
///
/// Scope: only the main variation (the first child of any branch). Recognises
/// the small set of properties needed to rebuild a game state: SZ, KM, RU,
/// PB, PW, HA, AB, AW, PL, B, W. All other properties are tolerated and
/// ignored. Variations and extensions beyond this set are not modelled.
public enum SGFParser {

    public static func parse(_ text: String) throws -> ParsedSGF {
        let nodes = try tokenizeMainVariation(text)
        guard !nodes.isEmpty else {
            throw SGFParseError.malformed("no nodes")
        }

        var boardSize = 19
        var komi: Float = 7.5
        var rulesName: String? = nil
        var blackPlayer: String? = nil
        var whitePlayer: String? = nil
        var handicap: Int? = nil
        var initialBlack: [Point] = []
        var initialWhite: [Point] = []
        var initialSideToMove: Stone? = nil
        var moves: [Move] = []

        let root = nodes[0]
        if let sz = root["SZ"]?.first, let s = Int(sz) { boardSize = s }
        if let km = root["KM"]?.first, let k = Float(km) {
            guard k.isFinite else {
                throw SGFParseError.malformed("non-finite komi '\(km)'")
            }
            komi = k
        }
        rulesName = root["RU"]?.first
        blackPlayer = root["PB"]?.first
        whitePlayer = root["PW"]?.first
        if let ha = root["HA"]?.first, let h = Int(ha) { handicap = h }

        // Setup (AB/AW/PL) and moves (B/W) may appear on any node; the spec
        // doesn't pin them to the root, so scan everything in order.
        for node in nodes {
            if let blackSetup = node["AB"] {
                for v in blackSetup {
                    guard let p = sgfToPoint(v, boardSize: boardSize) else {
                        throw SGFParseError.malformed("invalid AB coordinate '\(v)'")
                    }
                    initialBlack.append(p)
                }
            }
            if let whiteSetup = node["AW"] {
                for v in whiteSetup {
                    guard let p = sgfToPoint(v, boardSize: boardSize) else {
                        throw SGFParseError.malformed("invalid AW coordinate '\(v)'")
                    }
                    initialWhite.append(p)
                }
            }
            if let pl = node["PL"]?.first?.first {
                if pl == "B" || pl == "b" { initialSideToMove = .black }
                else if pl == "W" || pl == "w" { initialSideToMove = .white }
            }
            if let bs = node["B"] {
                for v in bs {
                    moves.append(try makeMove(player: .black, sgfValue: v, boardSize: boardSize))
                }
            }
            if let ws = node["W"] {
                for v in ws {
                    moves.append(try makeMove(player: .white, sgfValue: v, boardSize: boardSize))
                }
            }
        }

        return ParsedSGF(
            boardSize: boardSize,
            komi: komi,
            rulesName: rulesName,
            blackPlayer: blackPlayer,
            whitePlayer: whitePlayer,
            handicap: handicap,
            initialBlack: initialBlack,
            initialWhite: initialWhite,
            initialSideToMove: initialSideToMove,
            moves: moves
        )
    }

    /// Convert an SGF coordinate (e.g. "cd") to a board Point. Returns nil
    /// for both pass values and malformed input; callers that need to
    /// distinguish a true pass from garbage should consult `isPassValue`
    /// first.
    public static func sgfToPoint(_ value: String, boardSize: Int) -> Point? {
        if isPassValue(value, boardSize: boardSize) { return nil }
        guard value.count == 2 else { return nil }
        let chars = Array(value)
        guard let col = sgfLetterIndex(chars[0]),
              let row = sgfLetterIndex(chars[1]),
              col < boardSize && row < boardSize else { return nil }
        return Point(x: col, y: row)
    }

    /// True when the SGF value denotes a pass: an empty string, or "tt" on
    /// a board ≤ 19 (the legacy FF[3] convention).
    public static func isPassValue(_ value: String, boardSize: Int) -> Bool {
        if value.isEmpty { return true }
        if boardSize <= 19 && value == "tt" { return true }
        return false
    }

    private static func sgfLetterIndex(_ c: Character) -> Int? {
        guard let ascii = c.asciiValue else { return nil }
        if ascii >= 0x61 && ascii <= 0x7A { return Int(ascii - 0x61) } // a-z
        if ascii >= 0x41 && ascii <= 0x5A { return Int(ascii - 0x41) + 26 } // A-Z
        return nil
    }

    private static func makeMove(player: Stone, sgfValue: String, boardSize: Int) throws -> Move {
        if isPassValue(sgfValue, boardSize: boardSize) {
            return Move.pass(player: player)
        }
        guard let p = sgfToPoint(sgfValue, boardSize: boardSize) else {
            throw SGFParseError.malformed("invalid move coordinate '\(sgfValue)'")
        }
        return Move.move(at: p, player: player)
    }

    // MARK: - Tokenizer

    /// A node is a property-list mapping property identifier (e.g. "B", "AB")
    /// to one or more values. Property values appear in `[ ... ]` brackets,
    /// possibly repeated for list-typed properties (AB[aa][bb][cc]).
    private typealias Node = [String: [String]]

    /// Tokenize the main variation. Per the SGF grammar
    /// `GameTree := "(" Sequence GameTree* ")"`, the main line continues into
    /// the *first* child game tree at every fork; remaining children are
    /// alternative variations and are dropped.
    ///
    /// Iterative to avoid call-stack growth on deeply nested branches. Only
    /// two scalars of state are needed: every parent on the main path has, by
    /// definition, already taken its first child, so on `)` we just restore
    /// `firstChildTaken = true` rather than tracking a per-level stack.
    private static func tokenizeMainVariation(_ text: String) throws -> [Node] {
        let scalars = Array(text.unicodeScalars)
        var i = 0

        while i < scalars.count && scalars[i] != "(" { i += 1 }
        guard i < scalars.count else {
            throw SGFParseError.malformed("missing game tree '('")
        }
        i += 1

        var nodes: [Node] = []
        var depth = 1
        var firstChildTaken = false

        while i < scalars.count {
            while i < scalars.count && isWhitespace(scalars[i]) { i += 1 }
            if i >= scalars.count { break }

            let c = scalars[i]
            if c == ";" {
                i += 1
                nodes.append(try parseNode(scalars: scalars, index: &i))
            } else if c == "(" {
                i += 1
                if firstChildTaken {
                    try skipBalancedTree(scalars: scalars, index: &i)
                } else {
                    depth += 1
                }
            } else if c == ")" {
                i += 1
                depth -= 1
                if depth == 0 { return nodes }
                firstChildTaken = true
            } else {
                i += 1
            }
        }

        throw SGFParseError.malformed("unterminated game tree")
    }

    /// Skip a sibling game tree whose opening `(` has already been consumed.
    /// Maintains paren depth while honouring `[...]` property values, since
    /// `(` and `)` may appear inside comments or other text-typed values.
    private static func skipBalancedTree(scalars: [Unicode.Scalar], index: inout Int) throws {
        var depth = 1
        while index < scalars.count {
            let c = scalars[index]
            if c == "[" {
                index += 1
                while index < scalars.count {
                    let d = scalars[index]
                    if d == "\\" {
                        index += 1
                        if index < scalars.count { index += 1 }
                    } else if d == "]" {
                        index += 1
                        break
                    } else {
                        index += 1
                    }
                }
            } else if c == "(" {
                depth += 1
                index += 1
            } else if c == ")" {
                depth -= 1
                index += 1
                if depth == 0 { return }
            } else {
                index += 1
            }
        }
        throw SGFParseError.malformed("unterminated variation")
    }

    private static func parseNode(scalars: [Unicode.Scalar], index: inout Int) throws -> Node {
        var node: Node = [:]

        while index < scalars.count {
            while index < scalars.count && isWhitespace(scalars[index]) { index += 1 }
            if index >= scalars.count { break }

            let c = scalars[index]
            if c == ";" || c == "(" || c == ")" { break }

            var ident = ""
            while index < scalars.count {
                let ch = scalars[index]
                if (ch >= "A" && ch <= "Z") || (ch >= "a" && ch <= "z") {
                    ident.unicodeScalars.append(ch)
                    index += 1
                } else {
                    break
                }
            }
            if ident.isEmpty {
                throw SGFParseError.malformed("expected property identifier near offset \(index)")
            }
            let key = ident.uppercased()

            var values: [String] = []
            while true {
                while index < scalars.count && isWhitespace(scalars[index]) { index += 1 }
                if index >= scalars.count || scalars[index] != "[" { break }
                index += 1
                values.append(try parsePropertyValue(scalars: scalars, index: &index))
            }
            if values.isEmpty {
                throw SGFParseError.malformed("property \(key) has no value")
            }
            node[key, default: []].append(contentsOf: values)
        }

        return node
    }

    private static func parsePropertyValue(scalars: [Unicode.Scalar], index: inout Int) throws -> String {
        var out = ""
        while index < scalars.count {
            let c = scalars[index]
            if c == "\\" {
                index += 1
                if index < scalars.count {
                    // Backslash-newline is a line continuation; drop both per the SGF spec.
                    let next = scalars[index]
                    if next != "\n" && next != "\r" {
                        out.unicodeScalars.append(next)
                    }
                    index += 1
                }
            } else if c == "]" {
                index += 1
                return out
            } else {
                out.unicodeScalars.append(c)
                index += 1
            }
        }
        throw SGFParseError.malformed("unterminated property value")
    }

    private static func isWhitespace(_ c: Unicode.Scalar) -> Bool {
        return c == " " || c == "\t" || c == "\n" || c == "\r"
    }
}
