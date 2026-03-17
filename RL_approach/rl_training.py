"""
RL Training - FIXED VERSION with Resume Support
================================================
Fixes the loss convergence issue by implementing PROPER Bellman Equation
with LEGAL ACTIONS, not random actions in continuous space.

Key Fix: ExperienceReplayBuffer now stores next_legal_action_vectors
so train_batch() can calculate max Q-value over LEGAL actions,
not random continuous-space actions (which was the bug!)

Usage:
  # Train from scratch (default 5000 episodes)
  python3 rl_training.py

  # Train for custom number of episodes
  python3 rl_training.py --episodes 10000

  # Resume from a checkpoint and continue to 5000 episodes
  python3 rl_training.py --resume checkpoints/model_ep2000.pt

  # Resume from checkpoint and continue to 8000 episodes
  python3 rl_training.py --resume checkpoints/model_ep2000.pt --episodes 8000
"""

import sys
import os
import argparse

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, '..', 'core'))
sys.path.insert(0, os.path.join(_base, '..'))

from dotscuts import GameState, setup_standard_game
from ai_core import Action, generate_all_actions, execute_action, state_to_vector, action_to_vector
from training_metrics import TrainingMetrics

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# ============================================================================
# EXPERIENCE REPLAY BUFFER - FIXED
# ============================================================================
class ExperienceReplayBuffer:
    """
    Stores (state, action, reward, next_state, done, next_legal_action_vectors).
    
    ⭐ KEY FIX: next_legal_action_vectors is CRUCIAL for correct Bellman equation!
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action_vector, reward, next_state, done, next_legal_action_vectors=None):
        """⭐ next_legal_action_vectors MUST be provided! It's the key fix."""
        self.buffer.append((state, action_vector, reward, next_state, done, next_legal_action_vectors))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        """Returns tuple including next_legal_actions list."""
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        next_legal_actions = [exp[5] for exp in batch]

        return states, actions, rewards, next_states, dones, next_legal_actions

    def size(self):
        return len(self.buffer)


# ============================================================================
# Q-NETWORK
# ============================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# RL AGENT - FIXED
# ============================================================================
class RLAgent:
    """Agent with PROPER Deep Q-Learning (Bellman + Legal Actions)."""

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.9, target_update_freq=100):
        """
        Note: lr=0.001 (increased from 0.0005).
        With correct Q-targets, higher learning rate helps convergence.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim

        self.q_network = QNetwork(self.input_dim)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.last_loss = 0.0
        self.last_q_values = []

    def choose_best_action(self, state_vector, legal_actions):
        """Chooses action with highest Q-value."""
        if not legal_actions:
            return None

        best_action = None
        best_value = -float('inf')
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)

        with torch.no_grad():
            for action in legal_actions:
                action_vec = action_to_vector(action)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32)
                input_tensor = torch.cat((state_tensor, action_tensor))
                q_value = self.q_network(input_tensor).item()
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

        self.last_q_values.append(best_value)
        return best_action

    def train_batch(self, batch_states, batch_actions, batch_rewards, batch_next_states,
                   batch_dones, batch_next_legal_actions):
        """
        ⭐ FIXED: Uses LEGAL ACTIONS for Bellman equation.
        
        Q(s,a) = r + γ × max_{a' ∈ LEGAL(s')} Q(s', a')
        
        NOT: max over random continuous-space actions!
        """
        # Tensor conversion
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
                    legal_action_vectors = batch_next_legal_actions[i]

                    if legal_action_vectors is None or len(legal_action_vectors) == 0:
                        # Fallback (shouldn't happen)
                        best_next_q = 0.0
                    else:
                        # ⭐ KEY FIX: Evaluate ONLY legal actions
                        q_values_legal = []
                        for legal_action_vec in legal_action_vectors:
                            legal_action_t = torch.tensor(legal_action_vec, dtype=torch.float32)
                            input_next = torch.cat([next_state, legal_action_t])
                            q_val = self.target_network(input_next.unsqueeze(0)).item()
                            q_values_legal.append(q_val)
                        best_next_q = max(q_values_legal)

                    # Bellman equation
                    targets[i] = rewards_t[i] + self.gamma * best_next_q

        # Loss & optimization
        loss = self.loss_fn(q_predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()
        self.update_counter += 1
        return loss.item()

    def update_target_network(self):
        """Sync target network."""
        self.target_network = copy.deepcopy(self.q_network)

    def get_average_q_value(self):
        """Average Q-value."""
        if self.last_q_values:
            avg = np.mean(self.last_q_values[-50:])
            self.last_q_values = []
            return avg
        return 0.0


def run_self_play_episode(agent, replay_buffer, starting_player, epsilon=0.1):
    """
    Self-play episode.
    ⭐ KEY: Saves next_legal_action_vectors for each experience!
    """
    game_state = setup_standard_game()
    current_player = starting_player
    total_turns = 0

    game_over, winner = game_state.is_game_over()

    while not game_over:
        state_vector = state_to_vector(game_state, current_player)
        legal_actions = generate_all_actions(game_state, current_player)

        if random.random() < epsilon:
            chosen_action = random.choice(legal_actions)
        else:
            chosen_action = agent.choose_best_action(state_vector, legal_actions)

        if chosen_action is None:
            break

        action_vector = action_to_vector(chosen_action)

        execute_action(game_state, chosen_action)
        game_over, winner = game_state.is_game_over()

        if game_over:
            reward = 1.0 if winner == current_player else -1.0
        else:
            reward = 0.0

        # ⭐ KEY: GET LEGAL ACTIONS IN NEXT STATE BEFORE SAVING
        next_state_vector = state_to_vector(game_state, current_player)
        next_legal_actions_list = generate_all_actions(game_state, current_player)
        next_legal_action_vectors = [action_to_vector(a) for a in next_legal_actions_list]

        # Save with legal actions
        replay_buffer.add(
            state=state_vector,
            action_vector=action_vector,
            reward=reward,
            next_state=next_state_vector,
            done=game_over,
            next_legal_action_vectors=next_legal_action_vectors  # ⭐ CRUCIAL!
        )

        total_turns += 1
        current_player = 2 if current_player == 1 else 1

    return total_turns, winner


def run_training_loop(total_episodes=5000, resume_path=None):
    """
    Training loop with FIXED Bellman equation.

    Args:
        total_episodes: Target number of episodes to train
        resume_path: Optional path to checkpoint to resume from
    """

    print("="*80)
    print("STARTING RL TRAINING (FIXED - PROPER BELLMAN WITH LEGAL ACTIONS)")
    print("="*80)

    print("\n[SETUP] Initializing...")
    dummy_state = setup_standard_game()
    state_dim = len(state_to_vector(dummy_state, 1))
    dummy_actions = generate_all_actions(dummy_state, 1)
    action_dim = len(action_to_vector(dummy_actions[0]))

    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")

    agent = RLAgent(state_dim, action_dim, lr=0.001, gamma=0.9, target_update_freq=100)
    replay_buffer = ExperienceReplayBuffer(max_size=5000)
    metrics = TrainingMetrics(csv_path="training_log.csv", window_size=50)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    BATCH_SIZE = 32
    TRAIN_FREQ = 10
    CHECKPOINT_FREQ = 500
    PRINT_FREQ = 100

    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995

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
    print(f"  CSV log     -> training_log.csv")
    print()

    for episode in range(start_episode, total_episodes + 1):
        starting_player = 1 if episode % 2 == 1 else 2
        total_turns, winner = run_self_play_episode(
            agent, replay_buffer, starting_player, epsilon=epsilon
        )

        loss = 0.0
        if replay_buffer.size() >= BATCH_SIZE and episode % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones, next_legal_actions = replay_buffer.sample_batch(BATCH_SIZE)
            loss = agent.train_batch(states, actions, rewards, next_states, dones, next_legal_actions)

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        avg_q_value = agent.get_average_q_value()
        metrics.record_episode(
            episode_num=episode,
            winner=winner,
            game_length=total_turns,
            epsilon=epsilon,
            loss=loss,
            q_value_mean=avg_q_value
        )

        if episode % PRINT_FREQ == 0:
            metrics.print_summary(episode, total_episodes)

        if episode % CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'q_network_state': agent.q_network.state_dict(),
                'target_network_state': agent.target_network.state_dict(),
                'epsilon': epsilon,
                'version': 'v1',
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path}")

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print("\n" + "="*80)
    metrics.print_final_summary(total_episodes)
    print("="*80)

    return agent, metrics


if __name__ == "__main__":
    """
    Usage:
      python3 rl_training.py                                              # Train 5000 from scratch
      python3 rl_training.py --episodes 10000                             # Train 10000 from scratch
      python3 rl_training.py --resume checkpoints/model_ep2000.pt         # Resume from ep 2000, go to 5000
      python3 rl_training.py --resume checkpoints/model_ep2000.pt --episodes 8000  # Resume from ep 2000, go to 8000
    """
    parser = argparse.ArgumentParser(description="RL Training V1 for Dots & Cuts")
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes to train (default: 5000)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    agent, metrics = run_training_loop(total_episodes=args.episodes, resume_path=args.resume)
