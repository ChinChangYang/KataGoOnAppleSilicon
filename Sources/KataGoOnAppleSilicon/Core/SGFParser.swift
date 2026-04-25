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
public enum SGFParseError: Error, CustomStringConvertible {
    case malformed(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .malformed(let msg): return "Malformed SGF: \(msg)"
        case .unsupported(let msg): return "Unsupported SGF: \(msg)"
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

        // Defaults follow KataGo's GTP loadsgf behaviour.
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

        // The root node carries game info; subsequent nodes carry moves.
        let root = nodes[0]
        if let sz = root["SZ"]?.first, let s = Int(sz) { boardSize = s }
        if let km = root["KM"]?.first, let k = Float(km) { komi = k }
        rulesName = root["RU"]?.first
        blackPlayer = root["PB"]?.first
        whitePlayer = root["PW"]?.first
        if let ha = root["HA"]?.first, let h = Int(ha) { handicap = h }

        // Setup stones (AB/AW) and starting player (PL) may also appear on
        // later nodes per the SGF spec; scan all nodes.
        for node in nodes {
            if let blackSetup = node["AB"] {
                for v in blackSetup {
                    if let p = sgfToPoint(v, boardSize: boardSize) {
                        initialBlack.append(p)
                    }
                }
            }
            if let whiteSetup = node["AW"] {
                for v in whiteSetup {
                    if let p = sgfToPoint(v, boardSize: boardSize) {
                        initialWhite.append(p)
                    }
                }
            }
            if let pl = node["PL"]?.first?.uppercased() {
                if pl == "B" { initialSideToMove = .black }
                else if pl == "W" { initialSideToMove = .white }
            }
        }

        // Move properties live on non-root nodes. SGF allows B/W on any node
        // (incl. the root), so accept moves wherever they appear in order.
        for node in nodes {
            if let bs = node["B"] {
                for v in bs {
                    moves.append(makeMove(player: .black, sgfValue: v, boardSize: boardSize))
                }
            }
            if let ws = node["W"] {
                for v in ws {
                    moves.append(makeMove(player: .white, sgfValue: v, boardSize: boardSize))
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

    /// Convert an SGF coordinate (e.g. "cd") to a board Point. A single empty
    /// string, or "tt" on a 19x19 board, both denote a pass.
    public static func sgfToPoint(_ value: String, boardSize: Int) -> Point? {
        if value.isEmpty { return nil }
        // Legacy 19x19 pass marker.
        if boardSize <= 19 && value == "tt" { return nil }
        guard value.count == 2 else { return nil }
        let chars = Array(value)
        let col = sgfLetterIndex(chars[0])
        let row = sgfLetterIndex(chars[1])
        guard let col = col, let row = row else { return nil }
        guard col < boardSize && row < boardSize else { return nil }
        return Point(x: col, y: row)
    }

    private static func sgfLetterIndex(_ c: Character) -> Int? {
        guard let ascii = c.asciiValue else { return nil }
        if ascii >= 0x61 && ascii <= 0x7A { return Int(ascii - 0x61) } // a-z
        if ascii >= 0x41 && ascii <= 0x5A { return Int(ascii - 0x41) + 26 } // A-Z
        return nil
    }

    private static func makeMove(player: Stone, sgfValue: String, boardSize: Int) -> Move {
        if let p = sgfToPoint(sgfValue, boardSize: boardSize) {
            return Move.move(at: p, player: player)
        }
        return Move.pass(player: player)
    }

    // MARK: - Tokenizer

    /// A node is a property-list mapping property identifier (e.g. "B", "AB")
    /// to one or more values. Property values appear in `[ ... ]` brackets,
    /// possibly repeated for list-typed properties (AB[aa][bb][cc]).
    private typealias Node = [String: [String]]

    /// Tokenize the main variation: descend only into the first child of any
    /// branch `(...)`. Everything outside the outermost game tree is ignored.
    private static func tokenizeMainVariation(_ text: String) throws -> [Node] {
        let scalars = Array(text.unicodeScalars)
        var i = 0

        // Find first '(' (start of game tree).
        while i < scalars.count && scalars[i] != "(" { i += 1 }
        guard i < scalars.count else {
            throw SGFParseError.malformed("missing game tree '('")
        }
        i += 1

        var nodes: [Node] = []
        var depth = 0  // Tracks nested '(' depth beyond the main branch.

        while i < scalars.count {
            let c = scalars[i]
            if c == "(" {
                // Sub-variation: skip until matching ')'.
                depth += 1
                i += 1
            } else if c == ")" {
                if depth == 0 {
                    return nodes
                }
                depth -= 1
                i += 1
            } else if c == ";" {
                i += 1
                if depth == 0 {
                    let node = try parseNode(scalars: scalars, index: &i)
                    nodes.append(node)
                } else {
                    // Skip nodes inside variations (still need to advance past
                    // their property values, since values can contain parens).
                    _ = try parseNode(scalars: scalars, index: &i)
                }
            } else {
                // Whitespace or stray characters between tokens.
                i += 1
            }
        }

        throw SGFParseError.malformed("unterminated game tree")
    }

    /// Parse a single node starting just after the leading `;`. Returns when
    /// the next non-property character is encountered (`;`, `(`, or `)`).
    private static func parseNode(scalars: [Unicode.Scalar], index: inout Int) throws -> Node {
        var node: Node = [:]

        while index < scalars.count {
            // Skip whitespace.
            while index < scalars.count && isWhitespace(scalars[index]) { index += 1 }
            if index >= scalars.count { break }

            let c = scalars[index]
            if c == ";" || c == "(" || c == ")" { break }

            // Property identifier: uppercase ASCII letters (FF, GM, B, AB, ...).
            // Some writers emit lowercase or mixed case; uppercase to be safe.
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

            // One or more `[value]` blocks may follow.
            var values: [String] = []
            while true {
                while index < scalars.count && isWhitespace(scalars[index]) { index += 1 }
                if index >= scalars.count || scalars[index] != "[" { break }
                index += 1 // consume '['
                let value = try parsePropertyValue(scalars: scalars, index: &index)
                values.append(value)
            }
            if values.isEmpty {
                throw SGFParseError.malformed("property \(key) has no value")
            }
            // Append values rather than replace, so property repetition merges.
            node[key, default: []].append(contentsOf: values)
        }

        return node
    }

    /// Parse a `[...]` property value with backslash escaping. Returns the
    /// value text (without surrounding brackets) and consumes the closing `]`.
    private static func parsePropertyValue(scalars: [Unicode.Scalar], index: inout Int) throws -> String {
        var out = ""
        while index < scalars.count {
            let c = scalars[index]
            if c == "\\" {
                index += 1
                if index < scalars.count {
                    // SGF spec: backslash-newline is a line continuation (drop both).
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
