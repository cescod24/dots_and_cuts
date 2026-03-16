"""
Bot Player Module
=================
Loads trained RL models and generates action predictions.
Provides top K moves ranked by Q-value.
"""

import sys
import os
import torch
import numpy as np

# Add paths
sys.path.insert(0, '../core')
sys.path.insert(0, '../RL_approach')

from dotscuts import GameState
from ai_core import generate_all_actions, action_to_vector, state_to_vector, Action


class BotPlayer:
    """
    Loads a trained RL model and generates bot moves.
    Can show top K candidate moves and their evaluations.
    """

    def __init__(self, model_path: str, device='cpu'):
        """
        Initialize the bot with a trained model.

        Args:
            model_path: Path to the .pt checkpoint file
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path

        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Reconstruct Q-network
        self.q_network = self._build_q_network()
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.q_network.to(device)
        self.q_network.eval()  # Evaluation mode

        self.episode = checkpoint.get('episode', 'unknown')
        self.epsilon = checkpoint.get('epsilon', 0.05)

    def _build_q_network(self):
        """
        Reconstruct the Q-network architecture.
        Must match rl_training.py's QNetwork class.
        """
        import torch.nn as nn

        # This is a standard 1-output network
        # Input: state_vector (648) + action_vector (6) = 654
        input_dim = 654  # Hardcoded for now, should match training

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

        return QNetwork(input_dim)

    def get_best_action(self, game_state: GameState, current_player: int) -> Action:
        """
        Get the single best action according to the model.
        """
        actions = generate_all_actions(game_state, current_player)

        if not actions:
            return None

        best_action = None
        best_q = -float('inf')

        state_vector = state_to_vector(game_state, current_player)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for action in actions:
                action_vec = action_to_vector(action)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32, device=self.device)

                input_tensor = torch.cat([state_tensor, action_tensor]).unsqueeze(0)
                q_value = self.q_network(input_tensor).item()

                if q_value > best_q:
                    best_q = q_value
                    best_action = action

        return best_action

    def get_top_k_actions(self, game_state: GameState, current_player: int, k: int = 3):
        """
        Get top K actions ranked by Q-value.

        Returns:
            List of tuples: (action, q_value, is_best)
        """
        actions = generate_all_actions(game_state, current_player)

        if not actions:
            return []

        action_q_pairs = []

        state_vector = state_to_vector(game_state, current_player)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for action in actions:
                action_vec = action_to_vector(action)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32, device=self.device)

                input_tensor = torch.cat([state_tensor, action_tensor]).unsqueeze(0)
                q_value = self.q_network(input_tensor).item()

                action_q_pairs.append((action, q_value))

        # Sort by Q-value descending
        action_q_pairs.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for i, (action, q_value) in enumerate(action_q_pairs[:k]):
            is_best = (i == 0)
            results.append((action, q_value, is_best))

        return results

    def action_to_readable_string(self, action: Action) -> str:
        """
        Convert an action to a readable string.
        Format: "Move from (7,5) to (6,5)" or "Shoot (7,5) -> (3,5)"
        """
        piece = action.piece
        target = (action.target_x, action.target_y)

        pos_str = f"({piece.x},{piece.y})"
        target_str = f"({action.target_x},{action.target_y})"

        if action.action_type == "move":
            return f"{piece.kind.upper()[0]} {pos_str}-{target_str}"
        else:
            return f"{piece.kind.upper()[0]} {pos_str}→{target_str}"

    def get_model_info(self) -> dict:
        """
        Return information about the loaded model.
        """
        return {
            'path': self.model_path,
            'episode_trained': self.episode,
            'epsilon': self.epsilon,
            'device': self.device
        }

    def __repr__(self):
        return f"BotPlayer(episode={self.episode}, model='{os.path.basename(self.model_path)}')"
