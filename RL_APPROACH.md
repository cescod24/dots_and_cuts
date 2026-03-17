# RL System: Architecture, State Representations & Turn Dynamics

Complete technical documentation of the Reinforcement Learning system, including mathematical foundations, version differences, and turn management.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Version 1: Standard Deep Q-Learning](#version-1-standard-deep-q-learning)
3. [Version 2: Enhanced Double DQN](#version-2-enhanced-double-dqn)
4. [Mathematical Differences](#mathematical-differences)
5. [State Representation Deep Dive](#state-representation-deep-dive)
6. [Turn Management & RL Impact](#turn-management--rl-impact)
7. [Training Pipeline](#training-pipeline)
8. [Known Issues & Fixes](#known-issues--fixes)

---

## System Overview

The RL system uses Deep Q-Learning to train agents through self-play. Two versions explore different approaches:
- **V1**: Simple, fast baseline
- **V2**: Enhanced with tactical features and reward shaping

### Core Algorithm: Bellman Equation

Both versions implement the Q-learning Bellman update:

```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]
```

Where:
- **s, a**: state and action
- **r**: immediate reward
- **s'**: next state
- **γ**: discount factor (0.95)
- **α**: learning rate (0.0005)

In PyTorch, we minimize the loss:

```
Loss = (r + γ·max_a'(Q_target(s',a')) - Q_network(s,a))²
```

---

## Version 1: Standard Deep Q-Learning

### Network Architecture

```
Input: 654 dimensions
  ↓
Dense(654 → 256) + ReLU
Dense(256 → 128) + ReLU
Dense(128 → 64) + ReLU
Dense(64 → 1) [Q-value scalar output]
```

**Total parameters**: ~180k

### State Representation (648 dims)

The state is an 8-layer grid representation for a 9×9 board (8×81 = 648):

```python
def state_to_vector(game_state, current_player):
    """8-layer representation concatenated to 1D."""
    layers = [
        my_pieces,      # Layer 0: Where my pieces are (1 if present, 0 else)
        enemy_pieces,   # Layer 1: Where enemy pieces are
        orthogonal,     # Layer 2: Which of *my* pieces are orthogonal
        diagonal,       # Layer 3: Which of *my* pieces are diagonal
        z_values,       # Layer 4: Terrain elevation (-1, 0, +1)
        edge_count,     # Layer 5: Edge usage (normalized 0-1 per vertex)
        arrival_order,  # Layer 6: When each piece entered (temporal info)
        mobility,       # Layer 7: How many legal moves each piece has (normalized)
    ]
    return concatenate(layers).flatten()  # 8 × 81 = 648
```

**Each layer is normalized** (0-1 range) except z-values which are intrinsic [-1,0,+1].

### Action Representation (6 dims)

```python
def action_to_vector(action):
    """Convert action to 6-D vector."""
    return [
        action.piece.x,      # Source piece X (0-8)
        action.piece.y,      # Source piece Y (0-8)
        action.target_x,     # Destination X (0-8)
        action.target_y,     # Destination Y (0-8)
        1.0 if action.action_type == "move" else 0.0,
        1.0 if action.action_type == "shoot" else 0.0,
    ]
```

### Network Training

```python
# Experience replay buffer
replay_buffer = [(state_vec, action_vec, reward, next_state_vec, done)]

# Mini-batch training (batch_size=32)
for (s, a, r, s_next, done) in mini_batch:
    state_action = concatenate([s, a])  # 654 dims
    with torch.no_grad():
        # Stable target from target network
        target_q = r + gamma * max_a'(target_network(s_next + a'))

    # Predict with main network
    predicted_q = network(state_action)
    loss = MSE(predicted_q, target_q)
    loss.backward()
    optimizer.step()

# Every 100 episodes: target_network ← copy(network)
```

### Reward Structure

```python
if game_over:
    if current_player_won:
        reward = +1.0
    else:
        reward = -1.0
else:
    reward = 0.0  # Sparse reward during game
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 0.0005 | Conservative, prevents oscillation |
| **Gamma (γ)** | 0.95 | Values future rewards (95% of max lookahead) |
| **Epsilon Start** | 1.0 | Full exploration initially |
| **Epsilon End** | 0.05 | 5% exploration at the end |
| **Epsilon Decay** | 0.995 | ε *= 0.995 per episode (slower exploration decay) |
| **Buffer Size** | 5000 | Stores last 5000 experiences |
| **Batch Size** | 32 | Balances stability vs. computational cost |
| **Target Update Freq** | 100 eps | Sync target network every 100 episodes |

### Strengths

✅ Simple, interpretable algorithm
✅ Trains quickly (~30-60 min for 5000 eps)
✅ Good baseline for research
✅ Well-understood DQN mechanics

### Weaknesses

❌ Sparse rewards (only win/loss signals)
❌ May overestimate Q-values without Double DQN
❌ Smaller capacity for complex state features

---

## Version 2: Enhanced Double DQN

### Network Architecture

```
Input: 978 dimensions
  ↓
Dense(978 → 512) + ReLU
Dense(512 → 256) + ReLU
Dense(256 → 128) + ReLU
Dense(128 → 64) + ReLU
Dense(64 → 1) [Q-value scalar output]
```

**Total parameters**: ~600k (3.3× larger than V1)

### State Representation (972 dims)

Extends V1 with 4 **tactical layers** for strategic decision-making:

```python
def state_to_vector_v2(game_state, current_player):
    """12-layer representation: 8 base + 4 tactical."""
    base_layers = [
        my_pieces,         # Layer 0
        enemy_pieces,      # Layer 1
        orthogonal,        # Layer 2
        diagonal,          # Layer 3
        z_values,          # Layer 4
        edge_count,        # Layer 5
        arrival_order,     # Layer 6
        mobility,          # Layer 7
    ]

    tactical_layers = [
        shoot_threat,      # Layer 8: Threat density to my pieces
        shoot_opportunity, # Layer 9: My pieces' attack potential
        my_reachable,      # Layer 10: Vertices my pieces can reach (binary)
        enemy_reachable,   # Layer 11: Vertices enemy can reach (binary)
    ]

    return concatenate(base_layers + tactical_layers).flatten()  # 12 × 81 = 972
```

#### Detailed Tactical Layers

**Layer 8: shoot_threat(y, x)**
- For each vertex, normalized count of enemy pieces that can shoot that location
- Formula: `count_enemy_shoot_threats[y,x] / max_enemies` (clamped to [0,1])
- Interpretation: "How much danger is at vertex (y,x) from enemy fire?"

**Layer 9: shoot_opportunity(y, x)**
- For each *my* piece, how many different targets it can shoot
- Formula: `num_legal_shots_from_piece / max_enemies` per piece location
- Interpretation: "How much offensive power does my piece at (y,x) have?"

**Layer 10: my_reachable(y, x)**
- Binary map: 1 if any of my pieces can move to (y,x), else 0
- Computed from all move actions of all my pieces
- Interpretation: "Can I expand to this vertex?"

**Layer 11: enemy_reachable(y, x)**
- Binary map: 1 if any enemy piece can move to (y,x), else 0
- Interpretation: "Can the enemy threaten this vertex next turn?"

### Reward Shaping

Instead of sparse +1/-1, V2 uses shaped rewards to guide learning:

```python
def compute_shaped_reward(game_state, player, previous_state):
    """Multi-component reward signal."""
    reward = 0.0

    # 1. Game outcome (terminal reward)
    if game_over:
        if player_won:
            reward += 1.0
        else:
            reward -= 1.0

    # 2. Capture rewards (tactical)
    pieces_before = count_pieces(previous_state, player)
    pieces_after = count_pieces(game_state, player)
    enemy_pieces_before = count_pieces(previous_state, opponent)
    enemy_pieces_after = count_pieces(game_state, opponent)

    my_losses = pieces_before - pieces_after
    enemy_losses = enemy_pieces_before - enemy_pieces_after

    reward += 0.4 * enemy_losses   # +0.4 for each enemy piece captured
    reward -= 0.4 * my_losses      # -0.4 for each of my pieces lost

    # 3. Mobility bonus (strategic)
    my_moves_before = count_legal_moves(previous_state, player)
    my_moves_after = count_legal_moves(game_state, player)
    mobility_delta = my_moves_after - my_moves_before
    reward += 0.05 * clamp(mobility_delta, -5, +5)  # Range: -0.25 to +0.25

    # 4. Shoot threat bonus (offensive)
    # Count how many enemy pieces we can shoot
    my_shoot_actions = [a for a in generate_legal_actions(game_state, my_piece)
                        if a.action_type == "shoot"]
    shoot_count = min(len(my_shoot_actions), 2)
    reward += 0.1 * shoot_count  # +0.1 or +0.2 if we can shoot

    return reward
```

**Reward Range**: Typically [-1.4, +2.2] depending on board state

### Improvements Over V1

#### 1. Double DQN

```python
# V1: Overestimation issue
target_q = reward + gamma * max_a'(Q_network(s', a'))
           ↑ uses same network for selection AND evaluation

# V2: Decouple selection from evaluation
best_action = argmax_a'(Q_network(s', a'))  # Use main network to pick action
target_q = reward + gamma * Q_target_network(s', best_action)  # Evaluate with target
           ↑ Prevents overoptimistic Q-value estimates
```

#### 2. Huber Loss Instead of MSE

```python
def huber_loss(prediction, target, delta=1.0):
    error = prediction - target
    if |error| <= delta:
        return 0.5 * error²
    else:
        return delta * (|error| - 0.5 * delta)
```

Benefits: Less sensitive to outliers, smoother gradients.

#### 3. Gradient Clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
optimizer.step()
```

Prevents exploding gradients during training.

#### 4. Larger Capacity

- Network: 512-256-128-64-1 vs. 256-128-64-1
- State: 972 dims vs. 648 dims
- Can represent more complex strategic patterns

#### 5. Extended Exploration

- Epsilon decay: 0.997 (V2) vs. 0.995 (V1)
- Slower decay = explore longer before exploiting
- Helps escape local optima

### Hyperparameters

| Parameter | V1 | V2 | Reason for Change |
|-----------|----|----|-------------------|
| **State Dims** | 648 | 972 | Tactical features for strategy |
| **Input Dims** | 654 | 978 | 324 more dims total |
| **Network** | 256-128-64-1 | 512-256-128-64-1 | 3.3× larger capacity |
| **Learning Rate** | 0.0005 | 0.0005 | Same base rate |
| **Gamma** | 0.95 | 0.95 | Same discount factor |
| **Epsilon Decay** | 0.995 | 0.997 | Slower (explore longer) |
| **Buffer Size** | 5000 | 10000 | 2× larger (more diverse) |
| **Batch Size** | 32 | 64 | 2× larger (more stable updates) |
| **Loss Function** | MSE | Huber | More robust to outliers |
| **Gradient Clipping** | None | max_norm=1.0 | Prevent exploding gradients |

---

## Mathematical Differences

### 1. State Space Complexity

**V1**: 648-dimensional continuous space
- Encodes board position, piece types, terrain, edges
- Simple but limited representation

**V2**: 972-dimensional continuous space (+50% more information)
- Base 8 layers: same as V1
- +4 tactical layers: shoot threat, opportunity, reachability, strategic
- Enables learning offensive/defensive tactics explicitly

### 2. Value Function Estimation

**V1 Update Rule (DQN)**:
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]
                            ↑ same network (overestimation bias)
```

**V2 Update Rule (Double DQN)**:
```
Q(s,a) ← Q(s,a) + α[r + γ·Q_target(s', argmax_a'(Q(s',a'))) - Q(s,a)]
                                      ↑ different networks (debiased)
```

### 3. Loss Function Sensitivity

**V1 (MSE)**:
```
Loss = (target - predicted)²
```
Highly sensitive to outliers. Single bad prediction amplified quadratically.

**V2 (Huber)**:
```
         { 0.5·(target - predicted)²,            if error ≤ δ
Loss = {
         { δ·|target - predicted| - 0.5·δ²,    if error > δ
```
Smoother, more robust learning curve.

### 4. Exploration vs. Exploitation

**V1**: ε = 1.0 × (0.995)^episode
- After 1000 eps: ε ≈ 0.37 (37% exploration)
- After 5000 eps: ε ≈ 0.007 (0.7% exploration)

**V2**: ε = 1.0 × (0.997)^episode
- After 1000 eps: ε ≈ 0.37 (same)
- After 5000 eps: ε ≈ 0.061 (6.1% exploration) ← 8× more!

V2 explores longer, allowing it to discover more sophisticated strategies.

---

## State Representation Deep Dive

### Geographic Encoding: The [y][x] Convention

All matrices use **row-major indexing [y][x]**:
- y ∈ [0, N) = vertical position (row)
- x ∈ [0, N) = horizontal position (column)
- board[y][x] accesses row y, column x

```
Example 3×3 board indexing:
(0,0) (1,0) (2,0)     [0,0] [0,1] [0,2]
(0,1) (1,1) (2,1)  =  [1,0] [1,1] [1,2]
(0,2) (1,2) (2,2)     [2,0] [2,1] [2,2]

board[y][x] where y=row, x=col
```

### Normalization Strategy

**Bounded features** (0-1 range):
- Piece positions: binary
- Edge count: count/8 (max 8 edges per vertex)
- Mobility: action_count / max_possible_actions
- Arrival order: arrival_step / max_all_pieces
- Tactical layers: density / max_density

**Unbounded features** (preserved as-is):
- z-values: inherently [-1, 0, +1]

### Why Normalization Matters for RL

```
WITHOUT normalization:
- z-values range [-1, 1] → relatively small gradients
- Edge count ranges [0, 8] → different scale than positions [0,1]
→ Network struggles to balance conflicting scales

WITH normalization:
- All inputs [0, 1] (except z which is knowably [-1,1])
→ Activations stay in efficient operating region
→ Gradients more balanced
→ Faster convergence
```

---

## Turn Management & RL Impact

### How Dots & Cuts Turn System Works

1. **Turn sequence**: P1 → P2 → P1 → ... (alternating)
2. **Each turn**: One piece performs ONE action (move OR shoot)
3. **Game ends**: When one player has zero pieces

### Impact on RL Training

#### 1. **Opponent Modeling**

When we train P1, it sees states where it's P1's turn.
The Q-network learns: `Q(state_as_P1, action) = expected future value`

When opponent (P2) plays, **the state representation flips**:

```python
if current_player == 1:
    my_pieces, enemy_pieces = p1, p2  # P1: I see my pieces as "mine"
else:
    my_pieces, enemy_pieces = p2, p1  # P2: I see my pieces as "mine"
```

This means:
- The same board position looks different for P1 vs. P2
- Leads to **asymmetric value estimates** (fairness issue if uncorrected)
- V2's tactical layers help by explicitly encoding both perspectives

#### 2. **Reward Attribution Across Turns**

```
Turn sequence:
t=0: P1 moves (piece captured!)
t=1: P2 shoots
t=2: P1 moves (loses a piece)
t=3: Game ends, P1 loses

Reward flow (with γ=0.95):
-----
Actually, rewards only come at game end:
- t=2 end turns: P1 receives r=-1.0 (future_discount already applied by Bellman)
- Each intermediate step: r=0.0
  (OR with reward shaping: r=+0.4 for a capture, etc.)

Q-learning naturally handles the temporal credit:
Q(s_t, a_t) ← r + γ·max(Q(s_{t+1}, a))
This discount ensures capture at t=0 is valued less at t=2
```

#### 3. **Self-Play Dynamics**

During training, agent plays against itself:

```
Iteration N:
- P1 uses Q-network(episode N) → makes move
- P2 uses Q-network(episode N) → makes countering move
- Repeat until game end

Key insight:
- Both players improve together
- Creates feedback loop: "if P1 learns a tactic, P2 learns defense"
- Can converge to stable mixed strategies
- NOT asymmetric if we swap P1/P2 roles (fairness helps)
```

#### 4. **State Transitions & Immediate Validity**

The RL loop evaluates **legal actions only**:

```python
# In get_top_k_actions():
legal_actions = generate_all_actions(game_state, current_player)
for action in legal_actions:
    q_value = network([state, action])  # Only evaluate legal actions
    scores.append((action, q_value))
```

This works because:
- `state_to_vector()` encodes current board state
- Action vector encodes (piece_x, piece_y, target_x, target_y, is_move, is_shoot)
- Piece position in state + action source must match for validity
- Invalid actions are **never proposed**, only legal ones are Q-evaluated

#### 5. **Turn Dependency in Tactical Layers (V2)**

The tactical layers explicitly depend on whose turn it is:

```python
# In state_to_vector_v2():
current_player = 1  # We want state from P1's perspective

my_piece_list = [p for p in pieces if p.player == current_player]
enemy_piece_list = [p for p in pieces if p.player != current_player]

# shoot_threat: "enemy threats to MY pieces"
shoot_threat[y, x] = count_enemies_that_can_shoot(y, x, from my_piece_list perspective)

# shoot_opportunity: "MY pieces can shoot these"
shoot_opportunity[y, x] = count_legal_shoots_from(my_piece_list)
```

So if we call `state_to_vector_v2(game_state, player=2)`:
- my_piece_list = P2's pieces
- enemy_piece_list = P1's pieces
- The entire tactical layer flips perspective

### Why This Matters for Training

**Issue**: Without symmetric representation, P1 might learn "be aggressive" while P2 learns something else.

**Solution V1**: Experience replay buffer mixes P1 and P2 experiences.

**Solution V2**: Tactical layers explicitly encode "threat to me" and "opportunity for me", so the same network learns both perspectives consistently.

---

## Training Pipeline

### Data Flow

```
1. Initialize:
   agent = RLAgent(state_dim=648/972, action_dim=6)
   replay_buffer = ExperienceReplayBuffer(max_size=5000/10000)

2. For each episode:
   state = initial_game_state
   for turn in range(max_turns=~30-50):
       state_vector = state_to_vector(state, current_player)

       if random() < epsilon:
           action = random_legal_action()  # Exploration
       else:
           action = argmax_a(Q_network(state, a))  # Exploitation

       state_next, reward = execute_action(state, action)
       replay_buffer.add((state_vector, action, reward, next_state_vector, done))

       if len(replay_buffer) >= batch_size:
           mini_batch = replay_buffer.sample(batch_size)
           train_batch(mini_batch)  # Gradient descent step

       if episode % 100 == 0:
           target_network.load_state_dict(q_network.state_dict())

3. Output:
   - Checkpoint every 500 episodes
   - CSV log with metrics (loss, win_rate, game_length)
```

---

## Command-Line Usage

Both RL v1 and v2 support flexible CLI options for training and resuming:

### RL Version 1 (Standard DQN)

```bash
# Train from scratch (default: 5000 episodes)
python3 RL_approach/rl_training.py

# Train for custom duration
python3 RL_approach/rl_training.py --episodes 10000

# Resume from checkpoint, continue to 5000 total
python3 RL_approach/rl_training.py --resume RL_approach/checkpoints/model_ep2000.pt

# Resume from checkpoint, continue to 8000 total
python3 RL_approach/rl_training.py --resume RL_approach/checkpoints/model_ep2000.pt --episodes 8000
```

### RL Version 2 (Enhanced Double DQN)

```bash
# Train from scratch (default: 5000 episodes)
python3 RL_approach/rl_training_v2.py

# Train for custom duration
python3 RL_approach/rl_training_v2.py --episodes 10000

# Resume from checkpoint
python3 RL_approach/rl_training_v2.py --resume RL_approach/checkpoints_v2/model_ep3000.pt --episodes 8000
```

### Resume Details

When using `--resume`:
- Loads Q-network weights from checkpoint
- Loads target network weights
- Restores epsilon (exploration rate) from checkpoint
- Continues training from `checkpoint_episode + 1` to `target_episodes`
- Appends new results to existing CSV log
- Saves new checkpoints to the same directory

This allows:
- Extending training past original duration
- Continuing interrupted runs
- Fine-tuning trained models with different hyperparameters (by modifying code)

---

## Known Issues & Fixes

### Issue: V2 Checkpoints Trained with Wrong State Dimensions

**Status**: FIXED in bot_player.py (auto-adaptive loading)

**Problem**:
- RL v2 checkpoints were saved with input_dim=654 (incorrect) instead of 978
- This caused PyTorch state_dict loading errors: "size mismatch for net.0.weight"
- Root cause: RL v2 training code used `state_to_vector()` (648 dims) instead of `state_to_vector_v2()` (972 dims)

**Symptom**:
```
Error(s) in loading state_dict for QNetV2:
  size mismatch for net.0.weight: copying a param with shape torch.Size([512, 654])
  from checkpoint, the shape in current model is torch.Size([512, 978])
```

**Solution Implemented**:
The `bot_player.py` RLBot class now:
1. Auto-detects version from checkpoint metadata (v1/v2 field)
2. Detects actual input dimensions from network weights (ignores expected dims)
3. Selects state vector function based on actual dims, not metadata
4. Provides detailed debug output showing the mismatch and workaround being used

**How It Works**:
```python
# Bot loads v2 checkpoint with 654 dims
input_dim = 654  # Actual from checkpoint
state_dim = 654 - 6 = 648
self._state_fn = state_to_vector  # Use v1 state function (648 dims)
# Network created with input_dim=654, loads checkpoint successfully
```

**Current Behavior**:
- ✅ V1 checkpoints load correctly (input_dim=654, state_to_vector)
- ✅ V2 legacy checkpoints load correctly (input_dim=654, state_to_vector)
  - Warning message displayed about mismatch
  - Bot functions correctly despite not using enhanced features
- ❌ V2 checkpoints DON'T use tactical features (972-dim state)

**Recommendation**:
Retrain RL v2 with corrected code to generate proper 978-dim checkpoints:
```bash
rm -rf RL_approach/checkpoints_v2/*.pt  # Delete old checkpoints
python3 RL_approach/rl_training_v2.py --episodes 5000  # Retrain with correct state_to_vector_v2
```

This will create new checkpoints that properly utilize the 4 tactical layers.

---

### Bug: Z-Index Transposition in can_shoot()

**Location**: `core/dotscuts.py`, lines 484, 485, 490 (FIXED)

**Problem**: The `can_shoot()` method accessed the z-value grid with transposed indices.

**Before (WRONG)**:
```python
z_start = game_state.board.z[self.x][self.y]       # z[x][y] - WRONG
z_end = game_state.board.z[target_x][target_y]     # z[x][y] - WRONG
z_mid = game_state.board.z[current_x][current_y]   # z[x][y] - WRONG
```

**After (CORRECT)**:
```python
z_start = game_state.board.z[self.y][self.x]           # z[y][x] - RIGHT
z_end = game_state.board.z[target_y][target_x]         # z[y][x] - RIGHT
z_mid = game_state.board.z[current_y][current_x]       # z[y][x] - RIGHT
```

**Impact**:
- Bug allowed illegal shots (e.g., -1 → 0 which should fail)
- RL models trained with this bug learned incorrect shooting patterns
- **Recommend retraining** after fix for correct behavior

**Convention**: All game matrices use [y][x] indexing throughout the codebase.

---

## Summary

| Aspect | V1 | V2 |
|--------|----|----|
| **Algorithm** | DQN | Double DQN |
| **State Dims** | 648 | 972 |
| **Network** | 256-128-64-1 | 512-256-128-64-1 |
| **Reward** | Sparse (±1) | Shaped (up to ±2.2) |
| **Loss** | MSE | Huber |
| **Exploration** | Faster decay | Slower decay |
| **Training Time** | ~30-60 min | ~60-120 min |
| **Strengths** | Simple, fast | Stronger, more stable |
| **Use Case** | Baseline, teaching | Research, competitive play |

Both versions implement the same Bellman equation core, but V2 adds layers of sophistication for improved training stability and strategic understanding.

---

See also: `README.md`, `pygame_ui/QUICKSTART.md`
