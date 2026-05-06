import Foundation
import KataGoOnAppleSilicon

let katago = KataGoInference()
let handler = GTPHandler(katago: katago)

while let line = readLine(strippingNewline: true) {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty { continue }
    let response = handler.handleCommand(trimmed)
    print(response, terminator: "")
    if trimmed == "quit" { break }
}
