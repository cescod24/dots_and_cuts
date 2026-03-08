from dotscuts import GameState, setup_standard_game
from ai_core import Action, generate_all_actions, execute_action, state_to_vector, action_to_vector
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define a deeper Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()

        # Deeper network for better pattern learning on board states
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # Output Q-value
        )

    def forward(self, x):
        return self.net(x)

# Define the RL Agent
class RLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        # The input dimension is state vector concatenated with action vector
        self.q_network = QNetwork(state_dim + action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_best_action(self, state_vector, legal_actions):
        # Given a state vector and list of legal actions, choose the best action based on Q values
        best_action = None
        best_value = -float('inf')
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        for action in legal_actions:
            action_vec = action_to_vector(action)
            action_tensor = torch.tensor(action_vec, dtype=torch.float32)
            input_tensor = torch.cat((state_tensor, action_tensor))
            q_value = self.q_network(input_tensor)
            if q_value.item() > best_value:
                best_value = q_value.item()
                best_action = action
        return best_action

    def train_step(self, input_vector, target_value):
        # Perform a single training step given input vector and target value
        self.q_network.train()
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
        self.optimizer.zero_grad()
        output = self.q_network(input_tensor)
        loss = self.loss_fn(output, target_tensor)
        loss.backward()
        self.optimizer.step()

def run_self_play_episode(agent, starting_player, epsilon=0.1):
    """
    Runs a complete self-play episode where the agent plays against itself.
    Uses generate_all_actions(game_state, player) to get legal actions.
    Stores experiences as (state_vector, action_vector, reward) for training.
    Logs detailed statistics at the end of the episode.
    """
    # Initialize a new standard game state
    game_state = setup_standard_game()
    current_player = starting_player  # Use the starting_player argument
    experiences = []  # To store (state_vector, action_vector, reward) tuples

    # Counters for statistics
    total_turns = 0
    actions_per_player = {1: 0, 2: 0}
    non_zero_reward_actions = 0

    game_over, winner = game_state.is_game_over()

    # Loop until the game is over
    while not game_over:
        # Generate all legal actions for the current player
        legal_actions = generate_all_actions(game_state, current_player)

        # Convert current game state to vector, passing current_player
        state_vector = state_to_vector(game_state, current_player)

        # Agent chooses the best action based on current policy with epsilon-greedy exploration
        if random.random() < epsilon:
            chosen_action = random.choice(legal_actions)
        else:
            chosen_action = agent.choose_best_action(state_vector, legal_actions)

        # Convert chosen action to vector
        action_vector = action_to_vector(chosen_action)

        # Execute the chosen action and update the game state
        execute_action(game_state, chosen_action)

        # Check if game is over after action
        game_over, winner = game_state.is_game_over()

        # Determine reward:
        # +1 if current player has won after this move
        # -1 if current player has lost (opponent won)
        # 0 otherwise
        if game_over:
            if winner == current_player:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        # Store the experience for training
        experiences.append((state_vector, action_vector, reward))

        # Update statistics
        total_turns += 1
        actions_per_player[current_player] += 1
        if reward != 0.0:
            non_zero_reward_actions += 1

        # Switch to the other player for the next turn
        current_player = 2 if current_player == 1 else 1

    # After the episode, train the agent on the collected experiences
    for state_vec, action_vec, reward in experiences:
        input_vector = np.concatenate([state_vec, action_vec])
        agent.train_step(input_vector, reward)

    # Log detailed statistics for the episode
    print("Episode statistics:")
    print(f"  Total number of turns: {total_turns}")
    print(f"  Number of actions taken by Player 1: {actions_per_player[1]}")
    print(f"  Number of actions taken by Player 2: {actions_per_player[2]}")
    print(f"  Number of actions with non-zero reward (final win/loss): {non_zero_reward_actions}")
    print(f"  Winner of the episode: Player {winner if winner else 'None (Draw)'}")

    # Return statistics for aggregation in training loop
    return total_turns, winner

def run_training_loop(num_episodes):
    """
    Simple test loop for debugging and checking if self-play runs correctly.
    Creates an RLAgent and runs self-play episodes, printing summary after each.
    After all episodes, prints cumulative statistics.
    """
    # Create a dummy game state to get dimensions
    dummy_state = setup_standard_game()
    current_player = 1
    state_dim = len(state_to_vector(dummy_state, current_player))
    # Generate all actions for the dummy state and current_player to get action_dim
    dummy_actions = generate_all_actions(dummy_state, current_player)
    if not dummy_actions:
        raise RuntimeError("No legal actions found in dummy state")
    action_dim = len(action_to_vector(dummy_actions[0]))

    agent = RLAgent(state_dim, action_dim)

    # Cumulative statistics
    total_turns_all = 0
    wins_per_player = {1: 0, 2: 0}

    for episode in range(1, num_episodes + 1):
        # Alternate starting player each episode
        starting_player = 1 if episode % 2 == 1 else 2
        # Run one self-play episode and get statistics
        total_turns, winner = run_self_play_episode(agent, starting_player, epsilon=0.1)
        total_turns_all += total_turns
        if winner in wins_per_player:
            wins_per_player[winner] += 1

        # Print summary after each episode
        print(f"Episode {episode} completed.")

    # After all episodes, print cumulative statistics
    avg_turns = total_turns_all / num_episodes if num_episodes > 0 else 0
    print("\nCumulative training statistics:")
    print(f"  Total episodes played: {num_episodes}")
    print(f"  Total wins for Player 1: {wins_per_player[1]}")
    print(f"  Total wins for Player 2: {wins_per_player[2]}")
    print(f"  Average number of turns per episode: {avg_turns:.2f}")

if __name__ == "__main__":
    run_training_loop(500)