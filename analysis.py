from dotscuts import GameState, setup_standard_game
from ai_core import generate_legal_actions, execute_action
from minimax_ai import minimax_best_move
import random
import statistics
import sys
import os

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


def greedy_move(game_state: GameState, current_player: int):
    """
    Select the best move as any "shoot" action available. Greedy approach with "shoot" action.
    This function measures only the current state without lookahead.
    """
    player_pieces = [p for p in game_state.pieces if p.player == current_player]
    best_action = None

    # First try to find a shooting action
    for piece in player_pieces:
        actions = generate_legal_actions(game_state, piece)
        for action in actions:
            if action.action_type == "shoot":
                return action

    # If no shooting action found, return any legal action
    for piece in player_pieces:
        actions = generate_legal_actions(game_state, piece)
        if actions:
            return random.choice(actions)

    # No actions available
    return None

def simulate_random_game(game_state: GameState, starting_player: int):
    """
    Simulate a random game from the given game state and starting player.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            return winner, move_count, turn_count, available_moves_per_turn

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            # Current player cannot act → opponent wins
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        action = random.choice(all_actions)
        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1

def simulate_greedy_game(game_state: GameState, starting_player: int):
    """
    Simulate a game where the current player always chooses the greedy move.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            return winner, move_count, turn_count, available_moves_per_turn

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            # Current player cannot act → opponent wins
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        action = greedy_move(game_state, current_player)
        if action is None:
            # No possible action, opponent wins
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1

def simulate_greedy_vs_random_game(game_state: GameState, starting_player: int):
    """
    Simulate a game where one player uses greedy strategy and the other uses random strategy,
    alternating turns and alternating starting player each game.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []

    # Assign strategies: player 1 is greedy if starting_player == 1 else random, and vice versa
    # Actually, alternate starting player each game is handled outside, here we just assign strategies:
    # Let's say player 1 is greedy, player 2 is random
    # But starting player alternates in the run function
    def get_action(game_state, player):
        if player == 1:
            return greedy_move(game_state, player)
        else:
            player_pieces = [p for p in game_state.pieces if p.player == player]
            all_actions = []
            for piece in player_pieces:
                actions = generate_legal_actions(game_state, piece)
                all_actions.extend(actions)
            if not all_actions:
                return None
            return random.choice(all_actions)

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            return winner, move_count, turn_count, available_moves_per_turn

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            # Current player cannot act → opponent wins
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        action = get_action(game_state, current_player)
        if action is None:
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1

def simulate_minimax_vs_greedy_game(game_state: GameState, starting_player: int, depth: int):
    """
    Simulate a game where one player uses minimax strategy (with given depth) and the other uses greedy strategy,
    alternating turns.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []

    def get_action(game_state, player):
        if player == 1:
            return minimax_best_move(game_state, player, depth)
        else:
            return greedy_move(game_state, player)

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            return winner, move_count, turn_count, available_moves_per_turn

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            # Current player cannot act → opponent wins
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        action = get_action(game_state, current_player)
        if action is None:
            opponent = 1 if current_player == 2 else 2
            return opponent, move_count, turn_count, available_moves_per_turn

        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1

def run_random_simulations(game_state: GameState, starting_player: int, num_simulations: int):
    """
    Run multiple random simulations and return statistics about the results.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []
    for _ in range(num_simulations):
        sim_state = setup_standard_game()
        winner, moves, depth, available_moves_per_turn = simulate_random_game(sim_state, starting_player)
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth)
        all_available_moves_counts.extend(available_moves_per_turn)
        if winner is None:
            draws += 1

    winner_counts = {1: 0, 2: 0}
    for w in results:
        if w in winner_counts:
            winner_counts[w] += 1

    average_moves = statistics.mean(moves_list) if moves_list else 0
    max_moves = max(moves_list) if moves_list else 0
    average_depth = statistics.mean(depths_list) if depths_list else 0
    average_available_moves_per_turn = statistics.mean(all_available_moves_counts) if all_available_moves_counts else 0

    return {
        "winner_counts": winner_counts,
        "average_moves": average_moves,
        "max_moves": max_moves,
        "average_depth": average_depth,
        "draws": draws,
        "average_available_moves_per_turn": average_available_moves_per_turn
    }

def run_greedy_simulations(game_state: GameState, starting_player: int, num_simulations: int):
    """
    Run multiple greedy simulations and return statistics about the results.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []
    for _ in range(num_simulations):
        sim_state = setup_standard_game()
        winner, moves, depth, available_moves_per_turn = simulate_greedy_game(sim_state, starting_player)
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth)
        all_available_moves_counts.extend(available_moves_per_turn)
        if winner is None:
            draws += 1

    winner_counts = {1: 0, 2: 0}
    for w in results:
        if w in winner_counts:
            winner_counts[w] += 1

    average_moves = statistics.mean(moves_list) if moves_list else 0
    max_moves = max(moves_list) if moves_list else 0
    average_depth = statistics.mean(depths_list) if depths_list else 0
    average_available_moves_per_turn = statistics.mean(all_available_moves_counts) if all_available_moves_counts else 0

    return {
        "winner_counts": winner_counts,
        "average_moves": average_moves,
        "max_moves": max_moves,
        "average_depth": average_depth,
        "draws": draws,
        "average_available_moves_per_turn": average_available_moves_per_turn
    }

def run_greedy_vs_random_simulations(num_simulations: int):
    """
    Run multiple simulations where player 1 is greedy and player 2 is random,
    alternating starting player each game.
    Returns aggregated statistics and counts of wins for greedy and random players.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []
    greedy_wins = 0
    random_wins = 0

    for i in range(num_simulations):
        sim_state = setup_standard_game()
        starting_player = 1 if i % 2 == 0 else 2
        winner, moves, depth, available_moves_per_turn = simulate_greedy_vs_random_game(sim_state, starting_player)
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth)
        all_available_moves_counts.extend(available_moves_per_turn)
        if winner is None:
            draws += 1
        else:
            # Determine if winner was greedy or random player
            # Player 1 is greedy, player 2 is random
            if winner == 1:
                greedy_wins += 1
            elif winner == 2:
                random_wins += 1

    winner_counts = {1: greedy_wins, 2: random_wins}

    average_moves = statistics.mean(moves_list) if moves_list else 0
    max_moves = max(moves_list) if moves_list else 0
    average_depth = statistics.mean(depths_list) if depths_list else 0
    average_available_moves_per_turn = statistics.mean(all_available_moves_counts) if all_available_moves_counts else 0

    return {
        "winner_counts": winner_counts,
        "average_moves": average_moves,
        "max_moves": max_moves,
        "average_depth": average_depth,
        "draws": draws,
        "average_available_moves_per_turn": average_available_moves_per_turn
    }

def run_minimax_vs_greedy_simulations(num_simulations: int, depth: int):
    """
    Run multiple simulations where player 1 uses minimax strategy with given depth and player 2 uses greedy,
    alternating starting player each game.
    Returns aggregated statistics.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []

    for i in range(num_simulations):
        sim_state = setup_standard_game()
        starting_player = 1 if i % 2 == 0 else 2
        winner, moves, depth_turns, available_moves_per_turn = simulate_minimax_vs_greedy_game(sim_state, starting_player, depth)
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth_turns)
        all_available_moves_counts.extend(available_moves_per_turn)
        if winner is None:
            draws += 1

    winner_counts = {1: 0, 2: 0}
    for w in results:
        if w in winner_counts:
            winner_counts[w] += 1

    average_moves = statistics.mean(moves_list) if moves_list else 0
    max_moves = max(moves_list) if moves_list else 0
    average_depth = statistics.mean(depths_list) if depths_list else 0
    average_available_moves_per_turn = statistics.mean(all_available_moves_counts) if all_available_moves_counts else 0

    return {
        "winner_counts": winner_counts,
        "average_moves": average_moves,
        "max_moves": max_moves,
        "average_depth": average_depth,
        "draws": draws,
        "average_available_moves_per_turn": average_available_moves_per_turn
    }

if __name__ == "__main__":

    game_state = setup_standard_game()

    starting_player = 1

    num_simulations = 600
    WRITE_RESULTS_TO_FILE = False
    RESULTS_FILE_NAME = "results.txt"
    minimax_depth = 8
    block_print()
    # random_results = run_random_simulations(game_state, starting_player, num_simulations)
    # greedy_results = run_greedy_simulations(game_state, starting_player, num_simulations)
    # greedy_vs_random_results = run_greedy_vs_random_simulations(num_simulations)
    minimax_vs_greedy_results = run_minimax_vs_greedy_simulations(num_simulations, minimax_depth)
    enable_print()

    # print(f"Random strategy results over {num_simulations} games:", random_results)
    # print(f"Greedy strategy results over {num_simulations} games:", greedy_results)
    # print(f"Greedy vs Random strategy results over {num_simulations} games:", greedy_vs_random_results)
    print(f"Minimax (depth={minimax_depth}) vs Greedy strategy results over {num_simulations} games:", minimax_vs_greedy_results)

    # greedy_vs_random_greedy_wins = greedy_vs_random_results["winner_counts"].get(1,0)
    # greedy_vs_random_random_wins = greedy_vs_random_results["winner_counts"].get(2,0)

    # print(f"Comparison:")
    # print(f"Greedy vs Random - Greedy wins: {greedy_vs_random_greedy_wins}")
    # print(f"Greedy vs Random - Random wins: {greedy_vs_random_random_wins}")

    if WRITE_RESULTS_TO_FILE:
        with open(RESULTS_FILE_NAME, "a") as f:
            # f.write(f"Random strategy results over {num_simulations} games: {random_results}\n")
            # f.write(f"Greedy strategy results over {num_simulations} games: {greedy_results}\n")
            # f.write(f"Greedy vs Random strategy results over {num_simulations} games: {greedy_vs_random_results}\n")
            f.write(f"Minimax (depth={minimax_depth}) vs Greedy strategy results over {num_simulations} games: {minimax_vs_greedy_results}\n")
            f.write("Comparison:\n")
            # f.write(f"Greedy vs Random - Greedy wins: {greedy_vs_random_greedy_wins}\n")
            # f.write(f"Greedy vs Random - Random wins: {greedy_vs_random_random_wins}\n")
            f.write("-" * 60 + "\n")