"""
Bot Player Module
=================
Unified interface for AI opponents.
Supports:
  - Minimax v1 / v2 (with configurable search depth)
  - RL Deep Q-Learning (from saved checkpoints)

Both bot types expose the same public API:
  - get_best_action(game_state, player) -> Action
  - get_top_k_actions(game_state, player, k) -> [(Action, score, is_best)]
  - action_to_readable_string(action) -> str
  - label  (human-readable name for UI display)
"""

import sys
import os

# Ensure core/ and minimax_approach/ are importable
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "core"))
sys.path.insert(0, os.path.join(_base, ".."))

from dotscuts import GameState
from ai_core import (Action, generate_all_actions, execute_action,
                     action_to_vector, state_to_vector, state_to_vector_v2)
from move_notation import action_to_notation


# ---------------------------------------------------------------------------
# Minimax bot
# ---------------------------------------------------------------------------
class MinimaxBot:
    """
    Wraps minimax_approach/minimax_ai.py with a clean interface.
    Supports versions 'v1' and 'v2'.
    """

    def __init__(self, version: str = "v1", depth: int = 2):
        self.version = version
        self.depth = depth
        self.label = f"Minimax {version} (depth {depth})"

        # Import lazily so the rest of pygame_ui doesn't depend on pandas/sklearn
        from minimax_approach.minimax_ai import minimax, minimax_best_move, generate_all_actions as _gen
        self._minimax = minimax
        self._minimax_best_move = minimax_best_move

    def get_best_action(self, game_state: GameState, player: int) -> Action:
        return self._minimax_best_move(game_state, player, self.depth, version=self.version)

    def get_top_k_actions(self, game_state: GameState, player: int, k: int = 3,
                          depth: int = None):
        """
        Evaluate every legal action with minimax and return the top k.
        Returns list of (Action, score, is_best).
        depth: override search depth (None = use self.depth).
        """
        from minimax_approach.minimax_ai import minimax as _mm

        d = depth if depth is not None else self.depth
        actions = generate_all_actions(game_state, player)
        if not actions:
            return []

        scored = []
        for action in actions:
            execute_action(game_state, action)
            score = _mm(game_state, d - 1,
                        alpha=float("-inf"), beta=float("inf"),
                        maximizing_player=False,
                        root_player=player,
                        version=self.version)
            game_state.undo_last_move()
            scored.append((action, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1] if scored else 0
        results = []
        for action, score in scored[:k]:
            results.append((action, score, score == best_score))
        return results

    @staticmethod
    def action_to_readable_string(action: Action) -> str:
        return _format_action(action)


# ---------------------------------------------------------------------------
# RL (Deep Q-Learning) bot
# ---------------------------------------------------------------------------
class RLBot:
    """
    Loads a trained RL checkpoint and evaluates actions via the Q-network.
    Auto-detects version from checkpoint and uses the matching state vector
    and network architecture:
      - v1: state_to_vector (648) + action(6) = 654 input, net 256-128-64-1
      - v2: state_to_vector_v2 (972) + action(6) = 978 input, net 512-256-128-64-1
    """

    # Dimensions per version (state_dim + action_dim = input_dim)
    _DIMS = {
        "v1": {"state": 648, "action": 6, "input": 654},
        "v2": {"state": 972, "action": 6, "input": 978},
    }

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.device = device
        self.checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Detect version from checkpoint: first try metadata, fallback to weight shape
        self.version = checkpoint.get("version", None)

        if self.version is None:
            # Fallback: detect version from network weight dimensions
            # First layer of q_network has shape [hidden_size, input_dim]
            # V1: [256, 654], V2: [512, 978]
            q_state = checkpoint.get("q_network_state", {})
            if q_state:
                first_weight_key = "net.0.weight"
                if first_weight_key in q_state:
                    first_weight_shape = q_state[first_weight_key].shape
                    print(f"[RLBot] Detected checkpoint weight shape: {first_weight_shape}")
                    if first_weight_shape[0] == 512:  # V2 has 512 hidden units in first layer
                        self.version = "v2"
                    else:  # V1 has 256
                        self.version = "v1"
                else:
                    print(f"[RLBot] Warning: q_network_state keys: {list(q_state.keys())}")
                    self.version = "v1"  # Default fallback
            else:
                self.version = "v1"  # Default fallback

        # Get dimensions from checkpoint if available (handles legacy checkpoints)
        checkpoint_state_dim = checkpoint.get("state_dim", None)

        # Determine actual input dimension from checkpoint weights
        actual_input_dim = None
        q_state = checkpoint.get("q_network_state", {})
        first_weight_key = "net.0.weight"
        if q_state and first_weight_key in q_state:
            actual_input_dim = q_state[first_weight_key].shape[1]

        dims = self._DIMS[self.version]

        # Use actual dimension if detected, otherwise use expected dimension
        if actual_input_dim:
            input_dim = actual_input_dim
            expected_input_dim = dims["input"]
            if input_dim != expected_input_dim:
                print(f"[RLBot] ⚠️  WARNING: Checkpoint has input_dim={input_dim}, expected {expected_input_dim}")
                print(f"[RLBot]          This checkpoint may have been trained with wrong state vector!")
                print(f"[RLBot]          Loading with actual dimensions: {input_dim}")
        else:
            input_dim = dims["input"]

        # Select state vector function based on actual dimensions
        # This handles legacy checkpoints that might have been trained with wrong state vectors
        actual_state_dim = input_dim - 6  # Remove action vector size
        if actual_state_dim == 972:
            self._state_fn = state_to_vector_v2
            print(f"[RLBot] Using state_to_vector_v2 (972 dims)")
        else:
            self._state_fn = state_to_vector
            print(f"[RLBot] Using state_to_vector (648 dims)")

        # Debug: show version and final input_dim being used
        print(f"[RLBot] Loaded {checkpoint_path} as RL {self.version.upper()}, input_dim={input_dim}")

        # Build the right architecture based on version
        if self.version == "v2":
            class QNetV2(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(dim, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64),  nn.ReLU(),
                        nn.Linear(64, 1),
                    )
                def forward(self, x):
                    return self.net(x)
            self.q_network = QNetV2(input_dim)
        else:
            class QNetV1(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(dim, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64),  nn.ReLU(),
                        nn.Linear(64, 1),
                    )
                def forward(self, x):
                    return self.net(x)
            self.q_network = QNetV1(input_dim)

        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.q_network.to(device)
        self.q_network.eval()

        ep = checkpoint.get("episode", "?")
        self.label = f"RL {self.version} ep{ep}"

    def get_best_action(self, game_state: GameState, player: int) -> Action:
        top = self.get_top_k_actions(game_state, player, k=1)
        return top[0][0] if top else None

    def get_top_k_actions(self, game_state: GameState, player: int, k: int = 3,
                          depth: int = None):
        import torch

        # RL models are trained on 9x9 boards — reject mismatched sizes
        expected_N = 9
        actual_N = game_state.board.size
        if actual_N != expected_N:
            raise ValueError(
                f"RL bot was trained on {expected_N}x{expected_N} boards, "
                f"but current board is {actual_N}x{actual_N}. "
                f"Use Minimax for non-standard board sizes."
            )

        actions = generate_all_actions(game_state, player)
        if not actions:
            return []

        state_vec = self._state_fn(game_state, player)
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device)

        scored = []
        with torch.no_grad():
            for action in actions:
                act_t = torch.tensor(action_to_vector(action), dtype=torch.float32, device=self.device)
                inp = torch.cat([state_t, act_t]).unsqueeze(0)
                q = self.q_network(inp).item()
                scored.append((action, q))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1] if scored else 0
        results = []
        for action, score in scored[:k]:
            results.append((action, score, score == best_score))
        return results

    @staticmethod
    def action_to_readable_string(action: Action) -> str:
        return _format_action(action)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_bot(config) -> "MinimaxBot | RLBot":
    """
    Build the right bot from a GameConfig (mode_selection.GameConfig).
    """
    if config.bot_type in ("minimax_v1", "minimax_v2"):
        version = config.bot_type.split("_")[1]  # "v1" or "v2"
        return MinimaxBot(version=version, depth=config.minimax_depth)
    elif config.bot_type in ("rl", "rl_v1", "rl_v2"):
        return RLBot(checkpoint_path=config.rl_checkpoint)
    else:
        raise ValueError(f"Unknown bot type: {config.bot_type}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _format_action(action: Action, game_state=None) -> str:
    """Format action using algebraic notation if game_state is available."""
    if game_state is not None:
        return action_to_notation(action, game_state)
    # Fallback: notation without disambiguation
    symbol = "/" if action.piece.kind == "diagonal" else "-"
    x_mark = "x" if action.action_type == "shoot" else ""
    tx, ty = action.target_x, action.target_y
    return f"{symbol}{x_mark}{tx}{ty}"
