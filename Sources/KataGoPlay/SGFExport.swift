import Foundation
import KataGoOnAppleSilicon

func saveSGF(moveHistory: [(Stone, String)], komi: Float, boardSize: Int = 19) {
    let sgfMoves = moveHistory.compactMap { stone, coord -> (Stone, Point)? in
        guard coord.lowercased() != "pass",
              coord.lowercased() != "resign",
              let pt = gtpToPoint(coord, boardSize: boardSize) else { return nil }
        return (stone, pt)
    }

    let sgf = SGFGenerator.generateSGF(
        moves: sgfMoves,
        blackPlayer: "Human/KataGo",
        whitePlayer: "KataGo/Human",
        komi: komi,
        boardSize: boardSize
    )

    let timestamp = Int(Date().timeIntervalSince1970)
    let outputDir = ".build/test-output"

    let fm = FileManager.default
    if !fm.fileExists(atPath: outputDir) {
        try? fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    }

    let filename = "\(outputDir)/game_\(timestamp).sgf"
    do {
        try SGFGenerator.saveSGF(sgf, to: filename)
        print("Game saved to \(filename)")
    } catch {
        print("Failed to save SGF: \(error)")
    }
}
