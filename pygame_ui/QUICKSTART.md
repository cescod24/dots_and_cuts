# Dots & Cuts - PyGame UI Quick Start

## Playing the Game

### Start the Game
```bash
python3 main_game.py
```

### Menu Navigation
1. **Mode Selection**: Choose game type
   - Player vs Player (1v1)
   - Player vs Bot
   - Bot vs Bot

2. **Bot Configuration** (if bot game)
   - Select bot type:
     - Minimax v1 (classical solver)
     - Minimax v2 (enhanced)
     - RL v1 (standard Deep Q-Learning)
     - RL v2 (enhanced with reward shaping)
   - Select bot strength tier:
     - Weak: early training (1000-2000 episodes)
     - Medium: mid training (3000-4000 episodes)
     - Strong: full training (5000+ episodes)

3. **Player Selection**: Choose which player (P1 or P2) you are
   - P1 moves first
   - Each player sees their score and piece count

4. **Board Selection**: Choose starting configuration
   - Standard Game (recommended)
   - Custom maps (if available)

### Game Controls

| Key | Action |
|-----|--------|
| **Left Click** | Select piece or target |
| **ESC** | Return to menu |
| **G** | Toggle unvisited edge grid (default: OFF) |
| **Z** | Toggle z-value vertex coloring (default: OFF) |
| **B** | Toggle bot thinking display (default: ON) |
| **U** | Undo last move |
| **R** | Restart current game |
| **Q** | Quick quit to menu |

### Understanding the Display

**Board Elements:**
- **Black background**: Game board
- **White lines**: Edges (can be visited once)
- **Gray grid**: Unvisited edges (toggle with G)
- **Blue circles**: Towers (z=+1)
- **Red diamonds**: Bunkers (z=-1)
- **Light blue edges**: Lakes (special board feature)
- **Colored squares**: Your pieces
  - Green: Player 1
  - Red: Player 2

**Right Panel Shows:**
- Current player
- Piece count (O=orthogonal, D=diagonal)
- Points scored
- Next moves (if bot is thinking)
- Current toggle status

### Game Rules (Quick Reference)

**Piece Movement:**
- **Orthogonal (O)**: Moves in cardinal directions (↑↓←→)
- **Diagonal (D)**: Moves diagonally (↗↖↙↘)
- Must use unvisited edges only
- Each edge can be used exactly once

**Shooting:**
- Move pieces to capture opponent pieces
- Diagonal pieces shoot orthogonally
- Orthogonal pieces shoot diagonally
- z-value rules determine legality:
  - Shot from -1 to 0: ILLEGAL
  - Shot from 1 to 0: OK (if no midpoint blocks)
  - Shot from 1 to 1: Always OK
  - Other combinations have specific rules

**Victory:**
- Capture all opponent pieces
- Or force opponent into no-legal-moves state

### Playing Against Different Bots

**Minimax Bots (Deterministic)**
- Always make optimal move for given depth
- v1: Faster, shallower search
- v2: Stronger evaluation function
- **Pro**: Predictable, consistent
- **Con**: Can be too strong or weak (no learning)

**RL Bots (Learned Strategies)**
- Trained through self-play
- v1 Weak: Makes obvious mistakes
- v1 Medium: Reasonable player
- v1 Strong: Difficult opponent
- v2 Weak/Medium/Strong: Even more sophisticated (v2 has better reward shaping)
- **Pro**: Realistic, varied play
- **Con**: Takes time to load checkpoints

### Bot Thinking Display (Toggle with B)

Shows the bot's top 3 candidate moves:
```
Bot's Top Moves:
1. O (3,4)-(3,5)  Q=0.856  [move description]
2. D (1,2)->(4,5) Q=0.723  [shoot description]
3. O (7,1)-(6,1)  Q=0.612  [move description]
```

- **Move notation**: `O/D (from_x,from_y)-(to_x,to_y)` = movement
- **Shoot notation**: `O/D (from_x,from_y)->(target_x,target_y)` = shot
- **Q-value**: How good the bot thinks this move is (higher = better)
- **X at top**: Best move currently selected

### Tips for Playing

1. **Learn the z-values** (Toggle with Z to see):
   - Towers (blue circles) create +1 zones
   - Bunkers (red diamonds) create -1 zones
   - Affects shot legality

2. **Control the board**:
   - Use unvisited edges strategically
   - Don't waste moves on dead-end edges
   - Plan ahead for future piece positions

3. **Against minimax**:
   - Predictable moves
   - Look for forcing sequences
   - Try to limit their options

4. **Against RL bots**:
   - v1 bots may have quirks (trained with bug fix)
   - v2 bots are more consistent (reward shaping)
   - Weak bots miss tactics; medium/strong are challenging

5. **Debug mode**:
   - Use **G** to see unvisited edges (helps visualize empty spaces)
   - Use **Z** to see z-values (understand difficulty/special areas)
   - Use **B** to watch bot's decision-making

### Troubleshooting

**Bot takes forever to move**
- Turn off bot thinking display (press **B**)
- Minimax at high depth can be slow
- Try a weaker bot first

**Can't click pieces**
- Make sure it's your turn
- Click on your piece first, then click target
- Some moves might be illegal

**Shooting not allowed**
- Check z-values (press **Z** to see)
- Verify you have a clear shooting line
- Diagonal pieces shoot orthogonally and vice versa

**Game runs slowly**
- Reduce window resolution if needed
- Disable bot thinking display
- Use weaker bots (less computation)

---

Enjoy! For more details on RL training or minimax solvers, see the main README.md.
