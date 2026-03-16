# DOTS AND CUTS - Complete RL Research System

A comprehensive platform for training, testing, and playing the Dots & Cuts game using Deep Reinforcement Learning.

## Architecture Overview

```
dots_and_cuts/
├── core/                      # Game logic (immutable)
│   ├── dotscuts.py           # Game implementation
│   └── ai_core.py            # State/action utilities
│
├── RL_approach/              # Training pipeline
│   ├── rl_training.py        # Main training loop (FIXED Bellman)
│   ├── training_metrics.py   # Metrics tracking & logging
│   ├── analyze_training.py   # Post-training visualization
│   ├── experiment_cli.py     # CLI for experiments
│   ├── training_log.csv      # Training output
│   └── checkpoints/          # Saved models
│
├── pygame_ui/                # Interactive game
│   ├── main_game.py          # Main game interface
│   ├── game_display.py       # Board rendering
│   ├── bot_player.py         # Bot implementations
│   └── custom_setup.py       # Game setup builder
│
└── README.md                 # This file
```

## Key Implementation: Deep Q-Learning (FIXED)

### The Bug That Was Fixed

OLD CODE (BROKEN):
```python
def _estimate_best_q(self, state):
    q_values = []
    for _ in range(10):
        random_action = np.random.randn(6) * 0.1  # ❌ RANDOM ACTIONS!
        q_val = self.target_network(concat(state, random_action))
        q_values.append(q_val)
    return max(q_values)  # ❌ MAX OF INVALID ACTIONS
```

Problem: Calculated Q-target using RANDOM continuous-space actions, not legal game moves.
Result: Loss oscillates, model doesn't learn.

NEW CODE (FIXED):
```python
# Store legal actions in buffer
replay_buffer.add(..., next_legal_action_vectors=legal_actions)

# During training, use ONLY legal actions
for legal_action_vec in legal_action_vectors:
    q_val = self.target_network(concat(state, legal_action_vec))
    q_values.append(q_val)
best_next_q = max(q_values)  # ✓ MAX OF REAL ACTIONS

# Proper Bellman equation
target = reward + gamma * best_next_q
```

Result: Loss converges properly, model learns strategies.

## Quick Start

```bash
# 1. Train a model (5000 episodes)
cd RL_approach/
python3 rl_training.py

# 2. Analyze results
python3 analyze_training.py

# 3. Play against the bot
cd ../pygame_ui/
python3 main_game.py --mode pvbot --bot-model ../RL_approach/checkpoints/model_ep5000.pt

# 4. Experiment management
cd ../RL_approach/
python3 experiment_cli.py list
python3 experiment_cli.py test checkpoints/model_ep5000.pt --games 10
python3 experiment_cli.py compare checkpoints/model_ep1000.pt checkpoints/model_ep5000.pt
```

## Training Output

After running `rl_training.py`, you get:

1. **training_log.csv**: Full metrics per episode
   - episode, p1_wins, p2_wins, draws, game_length, epsilon, loss, q_value_mean
   - Rolling averages (50-episode window)
   - Ready for pandas/R analysis

2. **training_analysis.png**: 4-panel visualization
   - Win Rate Trend: Shows if model learns
   - Game Length Evolution: Shows strategic complexity
   - Loss Convergence: Shows if neural network learns
   - Fairness Check: Shows if balanced between players

3. **checkpoints/**: Saved models
   - model_ep0.pt, model_ep500.pt, model_ep1000.pt, ...
   - Use for testing, comparing, resuming

## Game Modes

### Player vs Player
```bash
python3 main_game.py --mode pvp
```
Local 1v1 game between two human players.

### Player vs Bot
```bash
python3 main_game.py --mode pvbot --bot-model ../RL_approach/checkpoints/model_ep5000.pt
```
Human (Player 1) vs trained bot (Player 2).
Bot's top 3 moves displayed with Q-value scores.

### Bot vs Bot
```bash
python3 main_game.py --mode botbot --bot-model ../RL_approach/checkpoints/model_ep5000.pt
```
Watch two agents play each other.

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

## Understanding the RL Implementation

### Bellman Equation

```
Q(s,a) = r + γ × max[Q(s', a')]
```

- Q(s,a): Expected value of taking action a in state s
- r: Immediate reward (+1 for win, -1 for loss, 0 during game)
- γ (gamma): Discount factor (0.9 = future 90% as important as present)
- max[Q(s', a')]: Best possible value in next state

The network learns to estimate Q-values. Good moves have high Q-values because they lead to good future states.

### Experience Replay

Problem: Training on sequential episodes causes correlated updates.
Solution: Store experiences, sample random mini-batches.

Benefits:
- Breaks temporal correlation
- Reuses successful experiences
- Stabilizes neural network training

### Target Network

Two copies of the network:
- Main network: Updated every step (learns)
- Target network: Copied from main every 100 episodes (provides stable targets)

Without target network, the network would chase a moving target (unstable).
With target network, it learns from fixed targets (stable).

### Epsilon-Greedy Exploration

ε decays from 1.0 to 0.05:
- Early training (ε=1.0): 100% random actions (explore)
- Late training (ε=0.05): 95% best action, 5% random (exploit)

This balances discovering new strategies vs refining good ones.

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

Last updated: 2024-03-16
Version: 2.0 (Fixed Bellman Equation)
