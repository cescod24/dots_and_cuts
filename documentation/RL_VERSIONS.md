# RL Training Versions Comparison

## Overview

This project includes two distinct RL training implementations to explore different approaches to the Dots & Cuts game.

---

## Version 1: Standard Deep Q-Learning

### Architecture
```
Input (654 dims: 648-dim state + 6-dim action)
  ↓
Dense(256) + ReLU
  ↓
Dense(128) + ReLU
  ↓
Dense(64) + ReLU
  ↓
Dense(1)  [Q-value output]
```

### State Representation
- **State Vector**: `state_to_vector()` → 648 dims
- **Action Vector**: `action_to_vector()` → 6 dims
- **Total Input**: 654 dims

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0005 |
| Gamma (discount) | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.995 |
| Buffer Size | 5000 |
| Batch Size | 32 |
| Target Update Frequency | 100 episodes |

### Reward Structure
- **+1.0**: Win game
- **-1.0**: Lose game
- **0.0**: During game

### Training File
`RL_approach/rl_training.py`

### Output
- **Checkpoints**: `RL_approach/checkpoints/`
- **Log**: `RL_approach/training_log.csv`
- **Analysis**: `RL_approach/training_analysis.png`

### Training Time
~30-60 minutes for 5000 episodes

### Strengths
- Simpler architecture (faster training)
- Fewer hyperparameters to tune
- Good baseline for comparison
- Well-understood algorithm

### Weaknesses
- No reward shaping (sparse rewards)
- No double DQN (may overestimate Q-values)
- Smaller network capacity
- Can struggle with tactical opportunities

### Suitable For
- Baseline comparison
- Understanding basic DQN
- Teaching RL concepts
- Quick experiments

---

## Version 2: Enhanced Double DQN with Reward Shaping

### Architecture
```
Input (978 dims: 972-dim state + 6-dim action)
  ↓
Dense(512) + ReLU
  ↓
Dense(256) + ReLU
  ↓
Dense(128) + ReLU
  ↓
Dense(64) + ReLU
  ↓
Dense(1)  [Q-value output]
```

### State Representation
- **State Vector**: `state_to_vector_v2()` → 972 dims (vs V1's 648)
- **Action Vector**: `action_to_vector()` → 6 dims
- **Total Input**: 978 dims
- **Board Layers**: 12 layers × N² (vs V1's 8 layers)

**V2 Adds 4 Tactical Layers on top of V1:**
1. **shoot_threat**: For my pieces, normalized count of enemies that can shoot it
2. **shoot_opportunity**: For my pieces, normalized count of enemies it can shoot
3. **my_reachable**: Binary map of vertices my pieces can move to
4. **enemy_reachable**: Binary map of vertices enemy pieces can move to

These tactical features help the network understand offensive and defensive situations more clearly.

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0005 |
| Gamma (discount) | 0.95 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.997 (slower) |
| Buffer Size | 10000 (larger) |
| Batch Size | 64 (larger) |
| Target Update Frequency | 100 episodes |
| Gradient Clipping | max_norm=1.0 |

### Reward Shaping
Instead of sparse +1/-1 rewards, V2 uses shaped rewards:

```python
base_reward = +1.0 if won else (-1.0 if lost else 0.0)

# Capture bonus
capture_reward = +0.4 per piece captured
loss_reward = -0.4 per piece lost

# Mobility bonus (encourages control)
mobility_bonus = 0.05 × max(0, min(10, legal_moves_available))

# Shoot threat bonus (encourages aggressive play)
shoot_threat = 0.1 × min(shooting_actions_available, 2)

total_reward = base_reward + capture_reward + loss_reward + mobility_bonus + shoot_threat
```

### Advanced Features
1. **Double DQN**
   - Two networks: main (learns) and target (provides targets)
   - Reduces overestimation bias
   - More stable convergence

2. **Huber Loss**
   - Robust to outliers
   - Smooth gradients
   - Better than MSE for large Q-value ranges

3. **Gradient Clipping**
   - Prevents exploding gradients
   - `max_norm=1.0`
   - Stabilizes training

4. **Slower Epsilon Decay**
   - 0.997 vs 0.995
   - More exploration in later episodes
   - Better strategy discovery

### Training File
`RL_approach/rl_training_v2.py`

### Output
- **Checkpoints**: `RL_approach/checkpoints_v2/`
- **Log**: `RL_approach/training_log_v2.csv`

### Training Time
~45-90 minutes for 5000 episodes (slower due to larger network)

### Strengths
- Reward shaping guides learning
- Double DQN reduces overestimation
- Larger network capacity
- More stable convergence
- Better handling of game-specific tactics
- Encourages both mobility and aggressiveness

### Weaknesses
- Longer training time
- More hyperparameters to tune
- Requires careful reward scaling
- May overfit to reward function design

### Suitable For
- Stronger bot for playing against
- Teaching advanced RL concepts
- Production use
- Learning complex strategies

---

## Comparison Table

| Aspect | V1 | V2 |
|--------|----|----|
| **State Representation** | 8 layers (648 dims) | 12 layers (972 dims) + tactical |
| **Input Dimension** | 654 (state + action) | 978 (state + action) |
| **Network Size** | 256-128-64-1 | 512-256-128-64-1 |
| **Learning Capacity** | Smaller | Larger |
| **Training Speed** | Faster | Slower |
| **Reward Signal** | Sparse (+1/-1) | Shaped (multi-component) |
| **Algorithm** | Standard DQN | Double DQN |
| **Loss Function** | MSE | Huber |
| **Stability** | Moderate | High |
| **Gradient Clipping** | No | Yes |
| **Exploration Duration** | Standard | Extended |
| **Buffer Size** | 5000 | 10000 |
| **Batch Size** | 32 | 64 |
| **Expected Strength** | Medium | Stronger |

---

## How to Choose

### Use V1 If You Want To:
- Understand basic Deep Q-Learning
- Train quickly for testing
- Compare against a simpler baseline
- Teach RL fundamentals
- Have limited computational resources

### Use V2 If You Want To:
- Play against a stronger bot
- Study advanced RL techniques
- Understand reward shaping benefits
- Conduct research on training methods
- Compare with minimax solver

---

## Running Both Versions

### Train Both
```bash
cd RL_approach/

# V1 training
python3 rl_training.py --episodes 5000

# V2 training (after V1 completes or in parallel)
python3 rl_training_v2.py --episodes 5000
```

### Analyze Both
```bash
# V1 analysis (auto-reads training_log.csv)
python3 analyze_training.py

# For V2, you would need to modify the script to read training_log_v2.csv
# or run analysis on V2 separately
```

### Play Against Both
```bash
cd ../pygame_ui/
python3 main_game.py
# Menu will auto-discover:
# - RL v1 weak/medium/strong
# - RL v2 weak/medium/strong
# Select which version/tier to play
```

---

## Research Insights

### Expected Results from V1
- Win rates converge around 45-55% (depends on starting position)
- Game length stabilizes (strategy develops)
- Loss decreases but doesn't reach zero (function approximation)

### Expected Results from V2
- Win rates more decisive (better strategy learning)
- More consistent gameplay
- Smoother loss convergence
- Better balance between offense and defense (due to reward shaping)

### Comparative Questions You Can Answer
1. **Does reward shaping help?** Compare V1 vs V2 learning curves
2. **Is Double DQN necessary?** Compare convergence stability
3. **How important is network size?** Compare 256-128-64 vs 512-256-128-64
4. **Can DQN beat minimax?** Play V2 strong vs minimax_v2
5. **What makes a good reward function?** Study V2's reward components

---

## Integration with Pygame UI

Both versions are fully integrated into the menu system:

1. **Mode Selection** → "Player vs Bot"
2. **Bot Configuration** → Select "RL v1" or "RL v2"
3. **Strength Tier** → Choose weak/medium/strong (auto-discovered)
4. **Play** → Game loads appropriate checkpoint

The bot_player.py factory automatically:
- Discovers checkpoints in both directories
- Detects model architecture from metadata
- Ranks by episode count (weak/medium/strong)
- Loads and initializes the network
- Provides action selection and visualization

---

## Checkpoint Structure

Both versions save checkpoints in the same format:

```python
checkpoint = {
    'network_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'episode': current_episode,
    'epsilon': current_epsilon,
    'version': 'v1' or 'v2',  # For auto-detection
}
```

Files are named: `model_ep{episode_number}.pt`

Examples:
- `checkpoints/model_ep0.pt` → V1 weak
- `checkpoints/model_ep3000.pt` → V1 medium
- `checkpoints/model_ep5000.pt` → V1 strong
- `checkpoints_v2/model_ep0.pt` → V2 weak
- `checkpoints_v2/model_ep3000.pt` → V2 medium
- `checkpoints_v2/model_ep5000.pt` → V2 strong

---

## Performance Expectations

### V1 Against Minimax V1
- **Weak (1000 ep)**: Loses consistently
- **Medium (3000 ep)**: Sometimes competitive
- **Strong (5000 ep)**: Can win occasionally

### V2 Against Minimax V1
- **Weak (1000 ep)**: Loses but more strategically
- **Medium (3000 ep)**: More competitive than V1 medium
- **Strong (5000 ep)**: Can beat minimax_v1 sometimes

### V2 Against Minimax V2
- **Strong (5000 ep)**: Roughly equal or slightly worse (minimax has perfect information)

---

## Future Enhancements

Potential improvements to explore:

1. **Policy Gradient Methods** (A3C, PPO) instead of value-based
2. **Attention Mechanisms** for board analysis
3. **Transfer Learning** between game variants
4. **Multi-Agent Training** (trained both players simultaneously)
5. **Curriculum Learning** (simple boards → complex boards)

---

## References

- Mnih et al. (2013): Playing Atari with Deep Reinforcement Learning
- Mnih et al. (2015): Human-level control through deep reinforcement learning
- Van Hasselt et al. (2015): Deep Reinforcement Learning with Double Q-learning
- Schaul et al. (2015): Prioritized Experience Replay

---

See README.md for overall system documentation.
