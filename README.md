# Dots & Cuts

A strategic board game where two players move and shoot pieces on a 9x9 grid. Pieces move along edges — orthogonal pieces move on rows/columns and shoot diagonally, diagonal pieces do the opposite. Capture all enemy pieces to win.

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/dots-and-cuts.git
cd dots-and-cuts

# Install dependencies
pip install -r requirements.txt

# Play
python pygame_ui/main_game.py
```

**Requirements:** Python 3.10+, pygame, numpy.

## Game Modes

- **Player vs Player** — local 1v1
- **Player vs Bot** — play against Minimax AI (configurable depth)
- **Bot vs Bot** — watch two bots play

## Controls

| Key | Action |
|-----|--------|
| **Click** | Select piece / execute move |
| **U** | Undo |
| **R** | Restart |
| **Q** | Back to menu |
| **G** | Toggle grid edges |
| **Z** | Toggle height hints |
| **B** | Toggle analysis panel |
| **E** | Toggle eval bar |
| **M** | Show my best move |
| **N** | Show opponent's best move |
| **D / Shift+D** | Decrease / increase minimax depth |
| **T / Shift+T** | Decrease / increase analysis timeout |

## Rules

- Each player starts with 2 pieces: one orthogonal (—) and one diagonal (/)
- **Orthogonal pieces** move along rows/columns, shoot along diagonals
- **Diagonal pieces** move along diagonals, shoot along rows/columns
- Moving creates an edge trail — you cannot cross your own trail
- **Towers** (+1 height) let you shoot over obstacles, **bunkers** (-1) block shots
- Eliminate all enemy pieces to win

## Project Structure

```
core/               Game logic (rules, state, actions)
pygame_ui/          Interactive PyGame interface
minimax_approach/   Minimax AI with alpha-beta pruning
```

---

Built with the help of Claude Opus 4.6 (Anthropic).
