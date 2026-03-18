import Foundation
import KataGoOnAppleSilicon

struct GameSetup {
    var humanColor: Stone
    var aiProfile: String
    var komi: Float
    var boardSize: Int
}

func runSetupFlow() -> GameSetup {
    print("""
    ╔═══════════════════════════════════╗
    ║     KataGoPlay — Go vs AI         ║
    ╚═══════════════════════════════════╝
    """)

    print("Play as: 1) Black (first)  2) White  3) Random  [1]: ", terminator: "")
    fflush(stdout)
    let colorInput = readLine()?.trimmingCharacters(in: .whitespaces) ?? "1"

    let humanColor: Stone
    switch colorInput {
    case "2": humanColor = .white
    case "3": humanColor = Bool.random() ? .black : .white
    default:  humanColor = .black
    }

    print("AI Profile (AI / 1d-9d / 1k-20k)  [AI]: ", terminator: "")
    fflush(stdout)
    let profileInput = readLine()?.trimmingCharacters(in: .whitespaces) ?? ""
    let aiProfile = profileInput.isEmpty ? "AI" : profileInput

    print("Board size: 1) 19x19  2) 13x13  3) 9x9  [1]: ", terminator: "")
    fflush(stdout)
    let boardSizeInput = readLine()?.trimmingCharacters(in: .whitespaces) ?? "1"
    let boardSize: Int
    switch boardSizeInput {
    case "2": boardSize = 13
    case "3": boardSize = 9
    default:  boardSize = 19
    }

    print("Komi  [7.5]: ", terminator: "")
    fflush(stdout)
    let komiInput = readLine()?.trimmingCharacters(in: .whitespaces) ?? ""
    let komi = Float(komiInput) ?? 7.5

    let colorName = humanColor == .black ? "Black" : "White"
    print("\nYou play \(colorName) | AI profile: \(aiProfile) | Board: \(boardSize)x\(boardSize) | Komi: \(komi)\n")

    return GameSetup(humanColor: humanColor, aiProfile: aiProfile, komi: komi, boardSize: boardSize)
}
