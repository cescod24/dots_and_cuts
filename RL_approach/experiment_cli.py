"""
Experiment CLI
==============
Command-line interface for managing RL training experiments.

Features:
- Start new training runs
- Resume training from checkpoints
- Test trained models
- Compare different models
- View training analytics
- Organize experimental results
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '../core')
sys.path.insert(0, '../RL_approach')

from rl_training import run_training_loop
from bot_player import BotPlayer
from dotscuts import setup_standard_game
from ai_core import generate_all_actions


class ExperimentManager:
    """
    Manages RL training experiments with logging and versioning.
    """

    def __init__(self, experiment_dir: str = '../experiments'):
        """
        Initialize the experiment manager.

        Args:
            experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def list_experiments(self):
        """
        List all available experiments.
        """
        experiments = []

        for exp_path in self.experiment_dir.iterdir():
            if exp_path.is_dir():
                config_file = exp_path / 'config.json'
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                    experiments.append((exp_path.name, config))

        if not experiments:
            print("No experiments found.")
            return

        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)

        for exp_name, config in experiments:
            print(f"\n📊 {exp_name}")
            print(f"   Created: {config.get('created', 'unknown')}")
            print(f"   Episodes: {config.get('total_episodes', '?')}")
            print(f"   Status: {config.get('status', 'unknown')}")
            if config.get('best_checkpoint'):
                print(f"   Best: {config['best_checkpoint']}")

        print("\n" + "="*80)

    def create_experiment(self, name: str, total_episodes: int) -> str:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            total_episodes: Total episodes to train

        Returns:
            Experiment directory path
        """
        exp_path = self.experiment_dir / name

        if exp_path.exists():
            response = input(f"Experiment '{name}' already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                return None

        exp_path.mkdir(parents=True, exist_ok=True)

        # Create config file
        config = {
            'name': name,
            'created': datetime.now().isoformat(),
            'total_episodes': total_episodes,
            'status': 'training',
            'best_checkpoint': None,
            'best_win_rate': 0.0
        }

        with open(exp_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Created experiment: {name}")
        return str(exp_path)

    def start_training(self, exp_name: str, total_episodes: int,
                      resume_from_checkpoint: str = None):
        """
        Start a training run.

        Args:
            exp_name: Experiment name
            total_episodes: Number of episodes
            resume_from_checkpoint: Optional checkpoint to resume from
        """
        print("\n" + "="*80)
        print(f"STARTING TRAINING: {exp_name}")
        print("="*80)

        # Change to RL_approach directory
        original_cwd = os.getcwd()
        os.chdir('../RL_approach')

        try:
            agent, metrics = run_training_loop(total_episodes=total_episodes)

            # Move results to experiment dir
            exp_dir = self.experiment_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Copy training log
            import shutil
            if os.path.exists('training_log.csv'):
                shutil.copy('training_log.csv', exp_dir / 'training_log.csv')

            if os.path.exists('training_analysis.png'):
                shutil.copy('training_analysis.png', exp_dir / 'training_analysis.png')

            if os.path.exists('checkpoints'):
                shutil.copytree('checkpoints', exp_dir / 'checkpoints', dirs_exist_ok=True)

            print(f"\n✓ Results saved to {exp_dir}")

        finally:
            os.chdir(original_cwd)

    def test_model(self, model_path: str, num_games: int = 10):
        """
        Test a trained model by playing games.

        Args:
            model_path: Path to .pt model file
            num_games: Number of games to play
        """
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            return

        print(f"\n[TESTING] Model: {model_path}")
        print(f"[TESTING] Running {num_games} games...")

        try:
            bot = BotPlayer(model_path, device='cpu')

            wins = 0
            losses = 0
            draws = 0
            total_turns = 0

            for game_num in range(num_games):
                game_state = setup_standard_game()
                current_player = 1
                turn_count = 0

                while True:
                    turn_count += 1

                    # Bot's turn (player 1)
                    if current_player == 1:
                        actions = generate_all_actions(game_state, current_player)
                        if not actions:
                            break

                        action = bot.get_best_action(game_state, current_player)
                        if action:
                            from ai_core import execute_action
                            execute_action(game_state, action)
                    else:
                        # Random opponent
                        actions = generate_all_actions(game_state, current_player)
                        if not actions:
                            break

                        import random
                        action = random.choice(actions)
                        from ai_core import execute_action
                        execute_action(game_state, action)

                    game_over, winner = game_state.is_game_over()
                    if game_over:
                        if winner == 1:
                            wins += 1
                        else:
                            losses += 1
                        break

                    current_player = 2 if current_player == 1 else 1

                    if turn_count > 500:  # Timeout
                        draws += 1
                        break

                total_turns += turn_count
                print(f"  Game {game_num+1}: ", end="")
                if winner == 1:
                    print(f"✓ WIN ({turn_count} turns)")
                else:
                    print(f"✗ LOSS ({turn_count} turns)")

            # Print results
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            print(f"Wins:     {wins}/{num_games} ({wins/num_games*100:.1f}%)")
            print(f"Losses:   {losses}/{num_games} ({losses/num_games*100:.1f}%)")
            print(f"Draws:    {draws}/{num_games} ({draws/num_games*100:.1f}%)")
            print(f"Avg Turns: {total_turns/num_games:.1f}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"✗ Error during testing: {e}")

    def compare_models(self, model1_path: str, model2_path: str, num_games: int = 10):
        """
        Compare two models by having them play against each other.

        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            num_games: Number of games to play
        """
        for path in [model1_path, model2_path]:
            if not os.path.exists(path):
                print(f"✗ Model not found: {path}")
                return

        print(f"\n[COMPARISON] Model 1: {os.path.basename(model1_path)}")
        print(f"[COMPARISON] Model 2: {os.path.basename(model2_path)}")
        print(f"[COMPARISON] Playing {num_games} games...\n")

        try:
            bot1 = BotPlayer(model1_path, device='cpu')
            bot2 = BotPlayer(model2_path, device='cpu')

            wins1 = 0
            wins2 = 0
            draws = 0

            for game_num in range(num_games):
                game_state = setup_standard_game()
                turn_count = 0

                while True:
                    turn_count += 1

                    # Bot 1 (player 1)
                    actions = generate_all_actions(game_state, 1)
                    if not actions:
                        break
                    action = bot1.get_best_action(game_state, 1)
                    if action:
                        from ai_core import execute_action
                        execute_action(game_state, action)

                    game_over, winner = game_state.is_game_over()
                    if game_over:
                        if winner == 1:
                            wins1 += 1
                        else:
                            wins2 += 1
                        break

                    # Bot 2 (player 2)
                    actions = generate_all_actions(game_state, 2)
                    if not actions:
                        break
                    action = bot2.get_best_action(game_state, 2)
                    if action:
                        from ai_core import execute_action
                        execute_action(game_state, action)

                    game_over, winner = game_state.is_game_over()
                    if game_over:
                        if winner == 1:
                            wins1 += 1
                        else:
                            wins2 += 1
                        break

                    if turn_count > 500:
                        draws += 1
                        break

            # Print results
            print("\n" + "="*60)
            print("COMPARISON RESULTS")
            print("="*60)
            print(f"Model 1 Wins: {wins1}/{num_games} ({wins1/num_games*100:.1f}%)")
            print(f"Model 2 Wins: {wins2}/{num_games} ({wins2/num_games*100:.1f}%)")
            print(f"Draws:        {draws}/{num_games} ({draws/num_games*100:.1f}%)")

            if wins1 > wins2:
                print(f"\n✓ Model 1 is better!")
            elif wins2 > wins1:
                print(f"\n✓ Model 2 is better!")
            else:
                print(f"\n➡️  Models are equally matched!")

            print("="*60 + "\n")

        except Exception as e:
            print(f"✗ Error during comparison: {e}")


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Dots & Cuts RL Experiment Manager"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # List command
    subparsers.add_parser('list', help='List all experiments')

    # New experiment command
    new_parser = subparsers.add_parser('new', help='Create new experiment')
    new_parser.add_argument('name', help='Experiment name')
    new_parser.add_argument('--episodes', type=int, default=5000,
                           help='Total episodes (default: 5000)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('name', help='Experiment name')
    train_parser.add_argument('--episodes', type=int, default=5000,
                             help='Episodes to train (default: 5000)')
    train_parser.add_argument('--resume', help='Resume from checkpoint')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('model', help='Path to .pt model')
    test_parser.add_argument('--games', type=int, default=10,
                            help='Number of test games (default: 10)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('model1', help='First model path')
    compare_parser.add_argument('model2', help='Second model path')
    compare_parser.add_argument('--games', type=int, default=10,
                               help='Number of comparison games (default: 10)')

    args = parser.parse_args()

    manager = ExperimentManager()

    if args.command == 'list':
        manager.list_experiments()

    elif args.command == 'new':
        manager.create_experiment(args.name, args.episodes)

    elif args.command == 'train':
        manager.start_training(args.name, args.episodes, args.resume)

    elif args.command == 'test':
        manager.test_model(args.model, num_games=args.games)

    elif args.command == 'compare':
        manager.compare_models(args.model1, args.model2, num_games=args.games)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
