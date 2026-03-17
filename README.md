# DOTS AND CUTS - Complete RL Research System

A comprehensive platform for training, testing, and playing the Dots & Cuts game using Deep Reinforcement Learning.

## Architecture Overview

```
dots_and_cuts/
├── core/                      # Game logic (immutable)
│   ├── dotscuts.py           # Game implementation (FIXED: z-index bug)
│   └── ai_core.py            # State/action utilities
│
├── RL_approach/              # Training pipeline with versioned models
│   ├── rl_training.py        # V1: Standard Deep Q-Learning
│   ├── rl_training_v2.py     # V2: Double DQN with enhanced rewards
│   ├── training_metrics.py   # Metrics tracking & logging
│   ├── analyze_training.py   # Post-training visualization
│   ├── training_log.csv      # V1 training output
│   ├── training_log_v2.csv   # V2 training output
│   ├── checkpoints/          # V1 saved models (weak/medium/strong)
│   ├── checkpoints_v2/       # V2 saved models (weak/medium/strong)
│   └── README_RL.md          # RL system documentation
│
├── minimax_approach/         # Classical minimax solver
│   └── minimax_ai.py         # V1 and V2 evaluation functions
│
├── pygame_ui/                # Interactive game
│   ├── main_game.py          # Game loop with toggle controls
│   ├── game_display.py       # Dynamic board rendering
│   ├── bot_player.py         # Bot factory (all versions)
│   ├── mode_selection.py     # Menu system with version selection
│   └── QUICKSTART.md         # UI usage guide
│
├── dotscuts.py               # Game logic (root, mirrored to core/)
├── PROJECT_SUMMARY.md        # Detailed completion report
└── README.md                 # This file
```

## Key Implementation: Deep Q-Learning & Minimax Solvers

The system implements two distinct approaches to Dots & Cuts AI:

### Reinforcement Learning (RL_APPROACH.md)

**Complete technical documentation** covering:
- ✅ V1: Standard Deep Q-Learning (648-dim state, 256-128-64-1 network)
- ✅ V2: Enhanced Double DQN (972-dim state, 512-256-128-64-1 network)
- ✅ Mathematical foundations (Bellman equation, loss functions)
- ✅ State representation deep dive (8 vs 12 layers, tactical features)
- ✅ Turn management & RL training dynamics
- ✅ Known issues & fixes (z-index bug corrected)

**Read**: [RL_APPROACH.md](./RL_APPROACH.md) for complete details on algorithms, math, and architecture.
  - Shoot threat bonus: 0.1 × min(shoot actions available, 2)
- Double DQN: Uses two networks to reduce overestimation
- Loss function: Huber (robust to outliers)
- Gradient clipping: max_norm=1.0
- Checkpoints: `RL_approach/checkpoints_v2/`

**Checkpoint Tiers**: Each version has weak/medium/strong models:
- Weak: ~1000-2000 episodes
- Medium: ~3000-4000 episodes  
- Strong: ~5000+ episodes

### Minimax Solver: Two Versions

**Version 1**: Logistic regression evaluation with standard alpha-beta pruning
**Version 2**: Enhanced evaluation function with deeper analysis

Both support configurable search depth (2-6 plies).

## Quick Start

```bash
# 1. Train RL V1 (standard)
cd RL_approach/
python3 rl_training.py --episodes 5000

# 2. Train RL V2 (enhanced - stronger bot)
python3 rl_training_v2.py --episodes 5000

# 3. Analyze both training runs
python3 analyze_training.py  # Creates visualization for V1
# (For V2, modify the script to read training_log_v2.csv)

# 4. Play against bot (menu lets you choose version/tier)
cd ../pygame_ui/
python3 main_game.py

# 5. Bot options in menu:
#    - Minimax v1 (fast, depth 3-6)
#    - Minimax v2 (stronger analysis)
#    - RL v1 weak/medium/strong (standard DQN)
#    - RL v2 weak/medium/strong (enhanced DQN)
```

## Training Output

Both RL versions produce similar outputs:

**Version 1 (Standard)**
1. **training_log.csv**: Full metrics per episode
2. **training_analysis.png**: 4-panel visualization
3. **checkpoints/model_epXXXX.pt**: Weak/medium/strong tiers

**Version 2 (Enhanced)**
1. **training_log_v2.csv**: Enhanced metrics with reward shaping
2. **checkpoints_v2/model_epXXXX.pt**: Weak/medium/strong tiers

Ready for pandas/R analysis with columns:
- episode, p1_wins, p2_wins, draws, game_length, epsilon, loss, q_value_mean
- rolling_p1_wr, rolling_p2_wr, rolling_draw_wr, rolling_game_length, rolling_loss

## Game Modes

### Player vs Player
```bash
python3 main_game.py
# Menu → Select "Player vs Player" → Choose player sides → Select map
```
Local 1v1 game between two human players.

### Player vs Bot
```bash
python3 main_game.py
# Menu → Select "Player vs Bot" → Choose bot type and tier → Choose sides → Select map
```
Human vs bot (4 bot types with 3 tiers each):
- Minimax v1 (classical solver)
- Minimax v2 (stronger evaluation)
- RL v1 weak/medium/strong (standard DQN)
- RL v2 weak/medium/strong (enhanced DQN)

Bot displays top candidate moves with analysis.

### Bot vs Bot
Select any two bots to watch them play (visualization for learning analysis).

### Game Controls
- **ESC**: Go back to menu
- **G**: Toggle unvisited edge grid (default: OFF)
- **Z**: Toggle z-value vertex hints (default: OFF)
- **B**: Toggle bot thinking display (default: ON)
- **U**: Undo last move
- **R**: Restart current game
- **Q**: Return to menu

## Research Workflow

### 1. Train Multiple Seeds

```bash
for seed in 1 2 3; do
    python3 experiment_cli.py new exp_seed_$seed --episodes 5000
    python3 experiment_cli.py train exp_seed_$seed
done
```

Compare win rates across runs to assess stability.

### 2. Ablation Studies

Test impact of hyperparameters:
- Different learning rates
- Different gamma values
- With/without target network
- With/without replay buffer

```bash
python3 experiment_cli.py compare model_v1.pt model_v2.pt --games 20
```

### 3. Hyperparameter Sweep

Modify hyperparameters in `rl_training.py` and run experiments:
```python
LEARNING_RATE = 0.001  # Try different values
GAMMA = 0.9            # Try 0.5, 0.7, 0.99
BATCH_SIZE = 32        # Try 16, 64, 128
```

### 4. Statistical Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log.csv')

# Plot win rate
plt.plot(df['episode'], df['rolling_p1_wr'], label='P1')
plt.plot(df['episode'], df['rolling_p2_wr'], label='P2')
plt.legend()
plt.show()

# Fairness metric
final_diff = abs(df.iloc[-1]['rolling_p1_wr'] - df.iloc[-1]['rolling_p2_wr'])
print(f"Fairness (lower is better): {final_diff:.1f}%")
```


## Documentation Map

For detailed information on specific topics:

| Topic | File |
|-------|------|
| **RL Algorithms & Math** | [RL_APPROACH.md](./RL_APPROACH.md) - Complete technical guide (state representations, Bellman, Double DQN, turn dynamics) |
| **Project Structure** | [VERSION_ORGANIZATION.md](./VERSION_ORGANIZATION.md) - How versions are separated, adding new versions |
| **RL Version Comparison** | [RL_approach/RL_VERSIONS.md](./RL_approach/RL_VERSIONS.md) - V1 vs V2 detailed comparison |
| **Game UI Guide** | [pygame_ui/QUICKSTART.md](./pygame_ui/QUICKSTART.md) - How to play, controls, menu options |
| **RL Deep Dive** | [RL_approach/README_RL.md](./RL_approach/README_RL.md) - Algorithm explanations and research methodology |

---

## Performance Metrics

After 5000 episodes, expect:

- **Win Rate**: ~50% for each player (fair game)
  - If significantly imbalanced (>60%), something might favor one player
  
- **Game Length**: ~25-30 turns
  - Starts high (50+), decreases as model learns efficient play
  
- **Loss**: Converges to ~0.001
  - Should smoothly decrease, not oscillate
  
- **Exploration**: ε decays to 0.05
  - Should follow exponential decay curve

## Troubleshooting

**Loss not converging:**
- Check that legal actions are properly saved
- Verify game rewards are being assigned (+1/-1)
- Print sample state vectors to check for NaN

**Bot plays poorly:**
- This is normal early in training
- Check training curve: win rate should increase
- Use earlier checkpoints if late ones overfit

**Game runs slow:**
- Disable bot thinking display (press 'B')
- Use smaller board in custom setup
- PyGame rendering at 60 FPS is expensive

**Can't load model:**
- Use absolute path: `/path/to/model_ep5000.pt`
- Ensure checkpoint exists: `ls checkpoints/model_*.pt`

## File Manifest

| File | Purpose |
|------|---------|
| core/dotscuts.py | Game rules, board state |
| core/ai_core.py | Actions, state vectors |
| RL_approach/rl_training.py | Training loop, Q-learning |
| RL_approach/training_metrics.py | Logging, CSV export |
| RL_approach/analyze_training.py | Matplotlib visualization |
| RL_approach/experiment_cli.py | Experiment management CLI |
| pygame_ui/main_game.py | Interactive game UI |
| pygame_ui/game_display.py | Board rendering |
| pygame_ui/bot_player.py | Load and run models |
| pygame_ui/custom_setup.py | Custom game builder |

## Next Steps

1. **Run training**: `python3 RL_approach/rl_training.py`
2. **Analyze**: `python3 RL_approach/analyze_training.py`
3. **Play**: `python3 pygame_ui/main_game.py --mode pvbot --bot-model ...`
4. **Experiment**: `python3 RL_approach/experiment_cli.py train ...`

Good luck with your research! 🚀

---

Last updated: 2026-03-17
Version: 2.1 (Consolidated Documentation with Enhanced Math & Turn Analysis)
