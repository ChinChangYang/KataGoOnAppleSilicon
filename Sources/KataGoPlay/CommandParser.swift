import Foundation

enum UserCommand {
    case move(String)
    case pass
    case hint(symmetry: Int)
    case aiMove
    case analysis(symmetry: Int)
    case save
    case board
    case profile(String)
    case newGame              // GTP: clear_board
    case undo                 // GTP: undo
    case info                 // GTP: protocol_version + name + version + list_commands
    case known(String)        // GTP: known_command <cmd>
    case handicap(Int)        // GTP: fixed_handicap N
    case freeHandicap([String])// GTP: set_free_handicap <coords>
    case rules(String)        // GTP: kata-set-rules <preset>
    case size(Int)            // GTP: boardsize N
    case komi(Float)          // GTP: komi X
    case quit
    case unknown(String)
}

struct CommandParser {
    static func parse(_ raw: String) -> UserCommand {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        let lower = trimmed.lowercased()
        let tokens = trimmed.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        let head = tokens.first?.lowercased() ?? ""

        if lower == "quit" || lower == "exit" || lower == "q" { return .quit }
        if lower == "pass" { return .pass }
        if lower == "ai" || lower == "aimove" { return .aiMove }
        if lower == "save" { return .save }
        if lower == "board" || lower == "show" { return .board }
        if lower == "new" { return .newGame }
        if lower == "undo" { return .undo }
        if lower == "info" { return .info }

        if head == "hint" {
            if tokens.count == 1 { return .hint(symmetry: 0) }
            if tokens.count == 2, let s = Int(tokens[1]), (0...7).contains(s) {
                return .hint(symmetry: s)
            }
            return .unknown(raw)
        }
        if head == "analysis" || head == "analyze" {
            if tokens.count == 1 { return .analysis(symmetry: 0) }
            if tokens.count == 2, let s = Int(tokens[1]), (0...7).contains(s) {
                return .analysis(symmetry: s)
            }
            return .unknown(raw)
        }
        if head == "profile", tokens.count >= 2 {
            return .profile(tokens.dropFirst().joined(separator: " "))
        }
        if head == "known", tokens.count == 2 {
            return .known(tokens[1])
        }
        if head == "handicap", tokens.count == 2, let n = Int(tokens[1]) {
            return .handicap(n)
        }
        if head == "free-handicap", tokens.count >= 2 {
            // Uppercase coords so the engine's parseMove (which requires
            // uppercase letters) accepts them.
            return .freeHandicap(tokens.dropFirst().map { $0.uppercased() })
        }
        if head == "rules", tokens.count == 2 {
            return .rules(tokens[1].lowercased())
        }
        if head == "size", tokens.count == 2, let n = Int(tokens[1]) {
            return .size(n)
        }
        if head == "komi", tokens.count == 2, let v = Float(tokens[1]) {
            return .komi(v)
        }

        let upper = trimmed.uppercased()
        if isValidGTPCoord(upper) { return .move(upper) }

        return .unknown(raw)
    }

    static func isValidGTPCoord(_ s: String) -> Bool {
        guard !s.isEmpty else { return false }
        let col = s.first!
        let rowStr = String(s.dropFirst())
        guard let row = Int(rowStr), row >= 1, row <= 19 else { return false }
        return (col >= "A" && col <= "H") || (col >= "J" && col <= "T")
    }
}
