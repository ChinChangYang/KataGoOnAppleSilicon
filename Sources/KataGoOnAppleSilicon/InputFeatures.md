# KataGo Neural Network Input Features (V7)

This document describes the input features used by the KataGo neural network model.
The feature encoding follows the `fillRowV7()` function from KataGo's [nninputs.cpp](https://github.com/ChinChangYang/KataGo/blob/metal-coreml-stable/cpp/neuralnet/nninputs.cpp).

## Overview

- **Spatial Features**: 22 planes of shape `[1, 22, 19, 19]`
- **Global Features**: 19 scalar values of shape `[1, 19]`

---

## Spatial Features (22 Planes)

Each plane is a 19x19 binary or float grid where position `(y, x)` corresponds to the board intersection.

| Plane | Name | Description |
|-------|------|-------------|
| 0 | On board | 1.0 for all valid board positions |
| 1 | Own stones | 1.0 where current player (pla) has a stone |
| 2 | Opponent stones | 1.0 where opponent (opp) has a stone |
| 3 | 1 liberty | 1.0 where stones have exactly 1 liberty (atari) |
| 4 | 2 liberties | 1.0 where stones have exactly 2 liberties |
| 5 | 3 liberties | 1.0 where stones have exactly 3 liberties |
| 6 | Ko-ban | 1.0 at ko prohibition locations (including superko) |
| 7 | Ko recapture blocked | 1.0 at encore ko recapture blocked locations |
| 8 | (Reserved) | Encore-specific feature |
| 9 | Move 1 ago | 1.0 at the location of the most recent move (by opponent) |
| 10 | Move 2 ago | 1.0 at the location of 2 moves ago (by pla) |
| 11 | Move 3 ago | 1.0 at the location of 3 moves ago (by opp) |
| 12 | Move 4 ago | 1.0 at the location of 4 moves ago (by pla) |
| 13 | Move 5 ago | 1.0 at the location of 5 moves ago (by opp) |
| 14 | Ladder captured | 1.0 where stones would be captured in a ladder |
| 15 | Ladder (prev board) | Ladder status from previous board state |
| 16 | Ladder (prev prev) | Ladder status from 2 boards ago |
| 17 | Ladder escape | 1.0 at moves that escape/capture a ladder |
| 18 | Area (own) | 1.0 where current player owns territory/area |
| 19 | Area (opp) | 1.0 where opponent owns territory/area |
| 20 | Encore stones (own) | Second encore starting stones for pla (Japanese rules) |
| 21 | Encore stones (opp) | Second encore starting stones for opp (Japanese rules) |

### Notes on Spatial Features

- **Perspective**: Planes 1-2 are relative to the current player, not absolute black/white.
  - If it's Black's turn: plane 1 = black stones, plane 2 = white stones
  - If it's White's turn: plane 1 = white stones, plane 2 = black stones

- **On Board Feature**: Plane 0 is always 1.0 for all 19x19 positions. This helps the network distinguish valid positions.

- **Liberty Features**: Planes 3-5 indicate stones with 1, 2, or 3 liberties. Stones with 4+ liberties have no feature set.
  
- **Move History**: Planes 9-13 indicate where the last 5 moves were played.
  - Pass moves set the corresponding **global** feature instead (features 0-4).
  - History alternates: opp move, pla move, opp move, pla move, opp move.
  
- **Chinese Rules Simplification**: For Chinese rules (area scoring), many planes are zeros or simplified:
  - Planes 7-8: No encore features (Chinese rules have no encore)
  - Planes 18-19: Area features (computed based on area scoring)
  - Planes 20-21: No encore starting stones

---

## Global Features (19 Values)

| Index | Name | Description | Chinese Rules Value |
|-------|------|-------------|---------------------|
| 0 | Pass 1 ago | 1.0 if the most recent move was a pass | (depends on history) |
| 1 | Pass 2 ago | 1.0 if 2 moves ago was a pass | (depends on history) |
| 2 | Pass 3 ago | 1.0 if 3 moves ago was a pass | (depends on history) |
| 3 | Pass 4 ago | 1.0 if 4 moves ago was a pass | (depends on history) |
| 4 | Pass 5 ago | 1.0 if 5 moves ago was a pass | (depends on history) |
| 5 | Komi | `selfKomi / 20.0` (clipped to board area bounds) | komi/20.0 |
| 6 | Ko rule flag 1 | Positional/situational ko indicator | 0.0 (simple ko) |
| 7 | Ko rule flag 2 | Ko rule sub-type | 0.0 |
| 8 | Suicide legal | 1.0 if multi-stone suicide is allowed | 1.0 |
| 9 | Territory scoring | 1.0 if using territory scoring | 0.0 (area scoring) |
| 10 | Tax rule flag 1 | Seki/tax rule indicator | 0.0 |
| 11 | Tax rule flag 2 | Tax rule sub-type | 0.0 |
| 12 | Encore phase 1 | 1.0 if in encore phase 1+ | 0.0 |
| 13 | Encore phase 2 | 1.0 if in encore phase 2 | 0.0 |
| 14 | Pass ends phase | 1.0 if a pass would end the current phase | (depends on state) |
| 15 | Playout flag | 1.0 if playout doubling advantage is nonzero | 0.0 |
| 16 | Playout advantage | `0.5 * playoutDoublingAdvantage` | 0.0 |
| 17 | Button | 1.0 if button go variant | 0.0 |
| 18 | Komi parity wave | Triangular wave based on komi parity | (computed) |

### Notes on Global Features

- **Self Komi**: From the perspective of the current player. For Black, it's `komi`. For White, it's `-komi`.

- **Komi Clipping**: The komi value is clipped to `±(boardArea + 20)` before dividing by 20.

- **Chinese Rules Constants**: For Chinese rules with simple ko:
  - Features 6-7: `0.0` (simple ko rule)
  - Feature 8: `1.0` (multi-stone suicide allowed)
  - Feature 9: `0.0` (area scoring, not territory)
  - Features 10-11: `0.0` (no tax rule)
  - Features 12-13: `0.0` (no encore phase)
  - Features 15-16: `0.0` (no playout doubling)
  - Feature 17: `0.0` (no button go)

- **Komi Parity Wave (Feature 18)**: A triangular wave that helps the neural network understand komi parity effects. The formula creates a wave with period 2 komi points, peaking around drawable komi values.

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Spatial 0 (on board) | ✅ Implemented | Always 1.0 for all positions |
| Spatial 1-2 (stones) | ✅ Implemented | Own/opponent perspective based on nextPlayer |
| Spatial 3-5 (liberties) | ✅ Implemented | Liberty counting per stone |
| Spatial 6 (ko-ban) | ✅ Implemented | Uses Board.koPoint for simple ko |
| Spatial 7 (ko recapture blocked) | ✅ Implemented | 0.0 for Chinese rules (no encore) |
| Spatial 8 (reserved) | ✅ Implemented | Zero-initialized (0.0 for Chinese rules, no encore) |
| Spatial 9-13 (history) | ✅ Implemented | Uses Board.moveHistory, fills planes 9-13 with move locations, sets global 0-4 for passes |
| Spatial 14-17 (ladders) | ⏳ Pending | Requires ladder detection |
| Spatial 18-19 (area) | ✅ Implemented | Uses Board.calculateArea() (Benson's algorithm) |
| Spatial 20-21 (encore stones) | ✅ Implemented | Zero-initialized (0.0 for Chinese rules, no encore) |
| Global 0-4 (pass history) | ✅ Implemented | Set by fillPlanes9To13History() when pass moves are detected |
| Global 5 (komi) | ✅ Implemented | selfKomi/20.0, perspective-aware |
| Global 6-7 (ko rule) | ✅ Implemented | 0.0 for simple ko (Chinese rules) |
| Global 8 (suicide) | ✅ Implemented | 1.0 (multi-stone suicide allowed) |
| Global 9 (scoring) | ✅ Implemented | 0.0 (area scoring, Chinese rules) |
| Global 10-11 (tax) | ✅ Implemented | 0.0 (no tax rule, Chinese rules) |
| Global 12-13 (encore) | ✅ Implemented | 0.0 (no encore phase, Chinese rules) |
| Global 14 (pass ends phase) | ⏳ Pending | Requires game state tracking |
| Global 15-17 (handicap/button) | ✅ Implemented | 0.0 (not used) |
| Global 18 (parity wave) | ⏳ Pending | Optional optimization |

---

## References

- [KataGo nninputs.cpp](https://github.com/ChinChangYang/KataGo/blob/metal-coreml-stable/cpp/neuralnet/nninputs.cpp)
- KataGo Documentation: Neural Network Input Format

