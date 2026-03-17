"""
RL Training v2 - Stronger Bot
==============================
Key improvements over v1:

1. REWARD SHAPING
   - v1: only +1/-1 at game end
   - v2: intermediate rewards for captures (+0.4), losing pieces (-0.4),
     mobility advantage (+0.05 per extra action), shooting threats (+0.1)

2. DOUBLE DQN
   - v1: max Q from target network (overestimates)
   - v2: main network selects action, target network evaluates it
     (reduces overestimation bias)

3. LARGER NETWORK
   - v1: 256-128-64-1
   - v2: 512-256-128-64-1 (more capacity to learn complex strategies)

4. BETTER HYPERPARAMETERS
   - Slower epsilon decay (0.997 vs 0.995) -> more exploration
   - Larger replay buffer (10000 vs 5000) -> more diverse experience
   - Train more frequently (every 5 episodes vs 10)
   - Gradient clipping to prevent exploding gradients

5. PIECE-LEVEL REWARD TRACKING
   - Tracks piece count changes per turn to give capture/loss rewards

Usage:
    python rl_training_v2.py              # train 5000 episodes (default)
    python rl_training_v2.py --episodes 3000
    python rl_training_v2.py --resume checkpoints_v2/model_ep2000.pt --episodes 5000
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dotscuts import GameState, setup_standard_game
from ai_core import Action, generate_all_actions, execute_action, state_to_vector_v2, action_to_vector
from training_metrics import TrainingMetrics

import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Q-NETWORK v2 - Larger capacity
# ============================================================================
class QNetworkV2(nn.Module):
    """
    Larger network than v1 (512-256-128-64-1 vs 256-128-64-1).
    More capacity helps learn complex game strategies.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================
class ExperienceReplayBuffer:
    """Stores (state, action, reward, next_state, done, next_legal_action_vectors)."""

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action_vector, reward, next_state, done, next_legal_action_vectors=None):
        self.buffer.append((state, action_vector, reward, next_state, done, next_legal_action_vectors))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        next_legal = [e[5] for e in batch]
        return states, actions, rewards, next_states, dones, next_legal

    def size(self):
        return len(self.buffer)


# ============================================================================
# RL AGENT v2 - Double DQN + gradient clipping
# ============================================================================
class RLAgentV2:
    """
    Improvements:
    - Double DQN: main net selects, target net evaluates (less overestimation)
    - Gradient clipping: prevents exploding gradients
    - Huber loss: more robust than MSE for outliers
    """

    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.95, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim

        self.q_network = QNetworkV2(self.input_dim)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - more robust than MSE

        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.last_loss = 0.0
        self.last_q_values = []

    def choose_best_action(self, state_vector, legal_actions):
        if not legal_actions:
            return None

        best_action = None
        best_value = -float('inf')
        state_t = torch.tensor(state_vector, dtype=torch.float32)

        with torch.no_grad():
            for action in legal_actions:
                act_t = torch.tensor(action_to_vector(action), dtype=torch.float32)
                inp = torch.cat((state_t, act_t))
                q = self.q_network(inp).item()
                if q > best_value:
                    best_value = q
                    best_action = action

        self.last_q_values.append(best_value)
        return best_action

    def train_batch(self, batch_states, batch_actions, batch_rewards,
                    batch_next_states, batch_dones, batch_next_legal_actions):
        """
        Double DQN training:
        1. Main network selects best action in next state
        2. Target network evaluates that action
        This reduces the overestimation bias of standard DQN.
        """
        states_t = torch.tensor(batch_states, dtype=torch.float32)
        actions_t = torch.tensor(batch_actions, dtype=torch.float32)
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        next_states_t = torch.tensor(batch_next_states, dtype=torch.float32)
        dones_t = torch.tensor(batch_dones, dtype=torch.float32)

        inputs = torch.cat([states_t, actions_t], dim=1)
        q_predictions = self.q_network(inputs).squeeze()

        batch_size = batch_states.shape[0]
        targets = torch.zeros(batch_size)

        with torch.no_grad():
            for i in range(batch_size):
                if batch_dones[i]:
                    targets[i] = rewards_t[i]
                else:
                    next_state = next_states_t[i]
                    legal_vecs = batch_next_legal_actions[i]

                    if legal_vecs is None or len(legal_vecs) == 0:
                        best_next_q = 0.0
                    else:
                        # DOUBLE DQN: main net picks best action
                        best_action_vec = None
                        best_main_q = -float('inf')
                        for vec in legal_vecs:
                            vec_t = torch.tensor(vec, dtype=torch.float32)
                            inp = torch.cat([next_state, vec_t])
                            q = self.q_network(inp.unsqueeze(0)).item()
                            if q > best_main_q:
                                best_main_q = q
                                best_action_vec = vec

                        # Target net evaluates the chosen action
                        vec_t = torch.tensor(best_action_vec, dtype=torch.float32)
                        inp = torch.cat([next_state, vec_t])
                        best_next_q = self.target_network(inp.unsqueeze(0)).item()

                    targets[i] = rewards_t[i] + self.gamma * best_next_q

        loss = self.loss_fn(q_predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.last_loss = loss.item()
        self.update_counter += 1
        return loss.item()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.q_network)

    def get_average_q_value(self):
        if self.last_q_values:
            avg = np.mean(self.last_q_values[-50:])
            self.last_q_values = []
            return avg
        return 0.0


# ============================================================================
# REWARD SHAPING
# ============================================================================
def compute_shaped_reward(game_state, current_player, prev_piece_counts, game_over, winner):
    """
    Richer reward signal than just +1/-1.

    Rewards:
      +1.0  / -1.0   win / loss (terminal)
      +0.4  per enemy piece captured this turn
      -0.4  per own piece lost this turn
      +0.05 * (my_mobility - enemy_mobility)  mobility advantage
      +0.1  per shooting opportunity available
    """
    if game_over:
        return 1.0 if winner == current_player else -1.0

    reward = 0.0
    opponent = 2 if current_player == 1 else 1

    # Piece count changes (captures / losses)
    my_pieces_now = sum(1 for p in game_state.pieces if p.player == current_player)
    enemy_pieces_now = sum(1 for p in game_state.pieces if p.player == opponent)
    my_prev = prev_piece_counts.get(current_player, my_pieces_now)
    enemy_prev = prev_piece_counts.get(opponent, enemy_pieces_now)

    captures = enemy_prev - enemy_pieces_now  # positive if we captured
    losses = my_prev - my_pieces_now          # positive if we lost pieces

    reward += 0.4 * captures
    reward -= 0.4 * losses

    # Mobility advantage (subtle signal)
    my_actions = generate_all_actions(game_state, current_player)
    enemy_actions = generate_all_actions(game_state, opponent)
    mobility_diff = len(my_actions) - len(enemy_actions)
    reward += 0.05 * max(-2, min(2, mobility_diff))  # clamp to [-0.1, 0.1]

    # Shooting threat bonus
    shoot_count = sum(1 for a in my_actions if a.action_type == "shoot")
    reward += 0.1 * min(shoot_count, 2)  # cap at 0.2

    return reward


# ============================================================================
# SELF-PLAY EPISODE
# ============================================================================
def run_self_play_episode(agent, replay_buffer, starting_player, epsilon=0.1):
    """Self-play with reward shaping."""
    game_state = setup_standard_game()
    current_player = starting_player
    total_turns = 0

    # Track piece counts for reward shaping
    prev_counts = {
        1: sum(1 for p in game_state.pieces if p.player == 1),
        2: sum(1 for p in game_state.pieces if p.player == 2),
    }

    game_over, winner = game_state.is_game_over()

    while not game_over:
        state_vector = state_to_vector_v2(game_state, current_player)
        legal_actions = generate_all_actions(game_state, current_player)

        if not legal_actions:
            break

        if random.random() < epsilon:
            chosen_action = random.choice(legal_actions)
        else:
            chosen_action = agent.choose_best_action(state_vector, legal_actions)

        if chosen_action is None:
            break

        action_vector = action_to_vector(chosen_action)
        execute_action(game_state, chosen_action)
        game_over, winner = game_state.is_game_over()

        # Shaped reward
        reward = compute_shaped_reward(
            game_state, current_player, prev_counts, game_over, winner
        )

        # Update piece counts for next turn
        prev_counts = {
            1: sum(1 for p in game_state.pieces if p.player == 1),
            2: sum(1 for p in game_state.pieces if p.player == 2),
        }

        # Next state legal actions (for Bellman target)
        next_state_vector = state_to_vector_v2(game_state, current_player)
        next_legal = generate_all_actions(game_state, current_player)
        next_legal_vecs = [action_to_vector(a) for a in next_legal]

        replay_buffer.add(
            state=state_vector,
            action_vector=action_vector,
            reward=reward,
            next_state=next_state_vector,
            done=game_over,
            next_legal_action_vectors=next_legal_vecs,
        )

        total_turns += 1
        current_player = 2 if current_player == 1 else 1

    return total_turns, winner


# ============================================================================
# TRAINING LOOP
# ============================================================================
def run_training_loop(total_episodes=5000, resume_path=None):
    """
    Training loop for v2.

    Key hyperparameter changes from v1:
      - lr=0.0005 (vs 0.001) -- gentler updates with shaped rewards
      - gamma=0.95 (vs 0.9)  -- values future more (longer-horizon strategy)
      - buffer=10000 (vs 5000) -- more diverse experience
      - train_freq=5 (vs 10)   -- learn more often
      - epsilon_decay=0.997 (vs 0.995) -- explore longer
    """
    print("=" * 80)
    print("RL TRAINING v2 - Double DQN + Reward Shaping")
    print("=" * 80)

    # Dimensions
    dummy = setup_standard_game()
    state_dim = len(state_to_vector_v2(dummy, 1))
    action_dim = len(action_to_vector(generate_all_actions(dummy, 1)[0]))
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    agent = RLAgentV2(state_dim, action_dim, lr=0.0005, gamma=0.95, target_update_freq=100)
    replay_buffer = ExperienceReplayBuffer(max_size=10000)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log_v2.csv")
    metrics = TrainingMetrics(csv_path=csv_path, window_size=50)

    BATCH_SIZE = 64
    TRAIN_FREQ = 5
    CHECKPOINT_FREQ = 500
    PRINT_FREQ = 100

    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.997
    epsilon = EPSILON_START
    start_episode = 1

    # Resume from checkpoint if requested
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        agent.q_network.load_state_dict(ckpt["q_network_state"])
        agent.target_network.load_state_dict(ckpt["target_network_state"])
        epsilon = ckpt.get("epsilon", EPSILON_START)
        start_episode = ckpt.get("episode", 0) + 1
        print(f"  Resumed from {resume_path} (ep {start_episode-1}, eps={epsilon:.4f})")

    print(f"\n  Training episodes {start_episode} -> {total_episodes}")
    print(f"  Checkpoints -> {checkpoint_dir}/")
    print(f"  CSV log     -> {csv_path}")
    print()

    for episode in range(start_episode, total_episodes + 1):
        starting_player = 1 if episode % 2 == 1 else 2
        total_turns, winner = run_self_play_episode(
            agent, replay_buffer, starting_player, epsilon=epsilon
        )

        loss = 0.0
        if replay_buffer.size() >= BATCH_SIZE and episode % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones, next_legal = \
                replay_buffer.sample_batch(BATCH_SIZE)
            loss = agent.train_batch(states, actions, rewards, next_states, dones, next_legal)

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        avg_q = agent.get_average_q_value()
        metrics.record_episode(
            episode_num=episode, winner=winner, game_length=total_turns,
            epsilon=epsilon, loss=loss, q_value_mean=avg_q,
        )

        if episode % PRINT_FREQ == 0:
            metrics.print_summary(episode, total_episodes)

        if episode % CHECKPOINT_FREQ == 0:
            path = os.path.join(checkpoint_dir, f"model_ep{episode}.pt")
            torch.save({
                "episode": episode,
                "q_network_state": agent.q_network.state_dict(),
                "target_network_state": agent.target_network.state_dict(),
                "epsilon": epsilon,
                "version": "v2",
                "state_dim": state_dim,
            }, path)
            print(f"  [CHECKPOINT] {path}")

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print("\n" + "=" * 80)
    metrics.print_final_summary(total_episodes)
    print("=" * 80)
    return agent, metrics


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training v2 for Dots & Cuts")
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    run_training_loop(total_episodes=args.episodes, resume_path=args.resume)
