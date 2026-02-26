import csv
import math
# ---- Feature Extraction ----
def compute_features(game_state, current_player):
    """
    Compute the 8 differential features for the current_player:
      1. material_diff
      2. mobility_diff
      3. shooting_diff
      4. pieces_in_danger_diff
      5. safe_pieces_diff
      6. avg_distance_to_enemy_diff
      7. clustering_diff
      8. board_centrality_diff
    Returns a dict with these feature names as keys.
    """
    # Helper functions
    def get_pieces(state, player):
        return [p for p in state.pieces if p.player == player]

    def get_all_actions(state, player):
        actions = []
        for piece in get_pieces(state, player):
            actions.extend(generate_legal_actions(state, piece))
        return actions

    def get_shoot_actions(state, player):
        actions = []
        for piece in get_pieces(state, player):
            acts = generate_legal_actions(state, piece)
            actions.extend([a for a in acts if hasattr(a, 'action_type') and a.action_type == "shoot"])
        return actions

    def is_piece_in_danger(state, piece):
        # A piece can be in danger only if it is the last arrived piece
        # among all pieces sharing the same vertex.
        same_vertex_pieces = [
            p for p in state.pieces
            if p.x == piece.x and p.y == piece.y
        ]

        # If this piece is not the one with the highest arrival_order,
        # it cannot be shot according to the stacking rule.
        last_arrived = max(same_vertex_pieces, key=lambda p: p.arrival_order)
        if piece is not last_arrived:
            return False

        # Now check if any enemy piece can legally shoot this vertex
        enemy = 1 if piece.player == 2 else 2
        for enemy_piece in get_pieces(state, enemy):
            if hasattr(enemy_piece, "can_shoot"):
                if enemy_piece.can_shoot(piece.x, piece.y, state):
                    return True

        return False

    def is_piece_safe(state, piece):
        # Not in danger
        return not is_piece_in_danger(state, piece)

    def avg_distance_to_enemy(state, player):
        my_pieces = get_pieces(state, player)
        enemy = 1 if player == 2 else 2
        enemy_pieces = get_pieces(state, enemy)
        if not my_pieces or not enemy_pieces:
            return 0.0
        dists = []
        for p in my_pieces:
            min_dist = min(manhattan_distance((p.x, p.y), (e.x, e.y)) for e in enemy_pieces)
            dists.append(min_dist)
        return sum(dists) / len(dists) if dists else 0.0

    def clustering(state, player):
        my_pieces = get_pieces(state, player)
        if len(my_pieces) < 2:
            return 0.0
        dists = []
        for i, p1 in enumerate(my_pieces):
            for j, p2 in enumerate(my_pieces):
                if j > i:
                    dists.append(manhattan_distance((p1.x, p1.y), (p2.x, p2.y)))
        return sum(dists) / len(dists) if dists else 0.0

    def board_centrality(state, player):
        my_pieces = get_pieces(state, player)
        # Assume board is 9x9, center at (4, 4)
        center = (4, 4)
        if not my_pieces:
            return 0.0
        dists = [manhattan_distance((p.x, p.y), center) for p in my_pieces]
        return -sum(dists) / len(dists) if dists else 0.0  # negative: lower distance = better

    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Get players
    me = current_player
    opp = 1 if me == 2 else 2

    # 1. material_diff
    my_material = len(get_pieces(game_state, me))
    opp_material = len(get_pieces(game_state, opp))
    material_diff = my_material - opp_material

    # 2. mobility_diff
    my_mobility = len(get_all_actions(game_state, me))
    opp_mobility = len(get_all_actions(game_state, opp))
    mobility_diff = my_mobility - opp_mobility

    # 3. shooting_diff
    my_shooting = len(get_shoot_actions(game_state, me))
    opp_shooting = len(get_shoot_actions(game_state, opp))
    shooting_diff = my_shooting - opp_shooting

    # 4. pieces_in_danger_diff
    my_pieces_in_danger = sum(1 for p in get_pieces(game_state, me) if is_piece_in_danger(game_state, p))
    opp_pieces_in_danger = sum(1 for p in get_pieces(game_state, opp) if is_piece_in_danger(game_state, p))
    pieces_in_danger_diff = my_pieces_in_danger - opp_pieces_in_danger

    # 5. safe_pieces_diff
    my_safe_pieces = sum(1 for p in get_pieces(game_state, me) if is_piece_safe(game_state, p))
    opp_safe_pieces = sum(1 for p in get_pieces(game_state, opp) if is_piece_safe(game_state, p))
    safe_pieces_diff = my_safe_pieces - opp_safe_pieces

    # 6. avg_distance_to_enemy_diff
    my_avg_dist = avg_distance_to_enemy(game_state, me)
    opp_avg_dist = avg_distance_to_enemy(game_state, opp)
    avg_distance_to_enemy_diff = my_avg_dist - opp_avg_dist

    # 7. clustering_diff
    my_clustering = clustering(game_state, me)
    opp_clustering = clustering(game_state, opp)
    clustering_diff = my_clustering - opp_clustering

    # 8. board_centrality_diff
    my_centrality = board_centrality(game_state, me)
    opp_centrality = board_centrality(game_state, opp)
    board_centrality_diff = my_centrality - opp_centrality

    return {
        "material_diff": material_diff,
        "mobility_diff": mobility_diff,
        "shooting_diff": shooting_diff,
        "pieces_in_danger_diff": pieces_in_danger_diff,
        "safe_pieces_diff": safe_pieces_diff,
        "avg_distance_to_enemy_diff": avg_distance_to_enemy_diff,
        "clustering_diff": clustering_diff,
        "board_centrality_diff": board_centrality_diff,
    }
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

def simulate_minimax_vs_greedy_game(game_state: GameState, starting_player: int, depth: int, feature_log_file=None, root_player=1):
    """
    Simulate a game where one player uses minimax strategy (with given depth) and the other uses greedy strategy,
    alternating turns.
    If feature_log_file is given, log features for root_player at each of their turns.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []
    feature_log = []

    def get_action(game_state, player):
        if player == 1:
            return minimax_best_move(game_state, player, depth)
        else:
            return greedy_move(game_state, player)

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            # Mark winner in features if feature_log not empty
            if feature_log:
                for row in feature_log:
                    row["winner"] = winner if winner is not None else 0
            return winner, move_count, turn_count, available_moves_per_turn, feature_log

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            opponent = 1 if current_player == 2 else 2
            if feature_log:
                for row in feature_log:
                    row["winner"] = opponent
            return opponent, move_count, turn_count, available_moves_per_turn, feature_log

        # Feature logging for root_player
        if current_player == root_player:
            features = compute_features(game_state, current_player)
            row = features.copy()
            row["winner"] = None
            feature_log.append(row)

        action = get_action(game_state, current_player)
        if action is None:
            opponent = 1 if current_player == 2 else 2
            if feature_log:
                for row in feature_log:
                    row["winner"] = opponent
            return opponent, move_count, turn_count, available_moves_per_turn, feature_log

        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1


# ---- NEW: Simulate Minimax vs Minimax Game ----
def simulate_minimax_vs_minimax_game(game_state: GameState, starting_player: int, depth: int, feature_log_file=None, root_player=1):
    """
    Simulate a game where both players use minimax strategy (with given depth).
    If feature_log_file is given, log features for root_player at each of their turns.
    Returns a tuple: (winner player number or None for draw, number of moves, depth (turns), list of available moves counts per turn, feature_log)
    """
    current_player = starting_player
    move_count = 0
    turn_count = 0
    available_moves_per_turn = []
    feature_log = []

    def get_action(game_state, player):
        return minimax_best_move(game_state, player, depth)

    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            # Mark winner in features if feature_log not empty
            if feature_log:
                for row in feature_log:
                    row["winner"] = winner if winner is not None else 0
            return winner, move_count, turn_count, available_moves_per_turn, feature_log

        player_pieces = [p for p in game_state.pieces if p.player == current_player]
        all_actions = []
        for piece in player_pieces:
            actions = generate_legal_actions(game_state, piece)
            all_actions.extend(actions)

        available_moves_per_turn.append(len(all_actions))

        if not all_actions:
            opponent = 1 if current_player == 2 else 2
            if feature_log:
                for row in feature_log:
                    row["winner"] = opponent
            return opponent, move_count, turn_count, available_moves_per_turn, feature_log

        # Feature logging for root_player
        if current_player == root_player:
            features = compute_features(game_state, current_player)
            row = features.copy()
            row["winner"] = None
            feature_log.append(row)

        action = get_action(game_state, current_player)
        if action is None:
            opponent = 1 if current_player == 2 else 2
            if feature_log:
                for row in feature_log:
                    row["winner"] = opponent
            return opponent, move_count, turn_count, available_moves_per_turn, feature_log

        execute_action(game_state, action)

        move_count += 1
        current_player = 2 if current_player == 1 else 1
        turn_count += 1


# ---- NEW: Run Minimax vs Minimax Simulations ----
def run_minimax_vs_minimax_simulations(num_simulations: int, depth: int, feature_log_file=None, root_player=1):
    """
    Run multiple simulations where both players use minimax strategy with given depth,
    alternating starting player each game.
    If feature_log_file is provided, log features for root_player at each of their turns.
    Returns aggregated statistics.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []
    feature_logs = []
    for i in range(num_simulations):
        sim_state = setup_standard_game()
        starting_player = 1 if i % 2 == 0 else 2
        winner, moves, depth_turns, available_moves_per_turn, feature_log = simulate_minimax_vs_minimax_game(
            sim_state, starting_player, depth, feature_log_file=None, root_player=root_player
        )
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth_turns)
        all_available_moves_counts.extend(available_moves_per_turn)
        if feature_log:
            feature_logs.extend(feature_log)
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
        "average_available_moves_per_turn": average_available_moves_per_turn,
        "feature_logs": feature_logs
    }

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

def run_minimax_vs_greedy_simulations(num_simulations: int, depth: int, feature_log_file=None, root_player=1):
    """
    Run multiple simulations where player 1 uses minimax strategy with given depth and player 2 uses greedy,
    alternating starting player each game.
    If feature_log_file is provided, log features for root_player at each of their turns.
    Returns aggregated statistics.
    """
    results = []
    moves_list = []
    depths_list = []
    draws = 0
    all_available_moves_counts = []

    feature_logs = []
    for i in range(num_simulations):
        sim_state = setup_standard_game()
        starting_player = 1 if i % 2 == 0 else 2
        winner, moves, depth_turns, available_moves_per_turn, feature_log = simulate_minimax_vs_greedy_game(
            sim_state, starting_player, depth, feature_log_file=None, root_player=root_player
        )
        results.append(winner)
        moves_list.append(moves)
        depths_list.append(depth_turns)
        all_available_moves_counts.extend(available_moves_per_turn)
        if feature_log:
            feature_logs.extend(feature_log)
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
        "average_available_moves_per_turn": average_available_moves_per_turn,
        "feature_logs": feature_logs
    }

# if __name__ == "__main__":
#     # Minimax vs Greedy block (commented out)
#     game_state = setup_standard_game()
#     starting_player = 1
#     num_simulations = 10000
#     WRITE_RESULTS_TO_FILE = False
#     RESULTS_FILE_NAME = "results.txt"
#     minimax_depth = 2
#     log_interval = 3  # Ogni quanti turni del root_player scrivere le feature nel CSV
#
#     block_print()
#     minimax_vs_greedy_results = run_minimax_vs_greedy_simulations(
#         num_simulations, minimax_depth, feature_log_file=None, root_player=1
#     )
#     enable_print()
#
#     # Print minimax vs greedy results
#     print(f"Minimax (depth={minimax_depth}) vs Greedy strategy results over {num_simulations} games:", {k: v for k, v in minimax_vs_greedy_results.items() if k != "feature_logs"})
#
#     # Write features to CSV for root_player, with proper header only if file does not exist
#     feature_log_file = ""
#     feature_logs = minimax_vs_greedy_results.get("feature_logs", [])
#     if feature_logs:
#         header = [
#             "material_diff", "mobility_diff", "shooting_diff", "pieces_in_danger_diff",
#             "safe_pieces_diff", "avg_distance_to_enemy_diff", "clustering_diff", "board_centrality_diff", "winner"
#         ]
#         file_exists = os.path.isfile(feature_log_file)
#         with open(feature_log_file, "a", newline="") as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=header)
#             if not file_exists:
#                 writer.writeheader()
#             for idx, row in enumerate(feature_logs):
#                 # Scrivi solo le righe dei turni multipli di log_interval (0 incluso)
#                 if idx % log_interval == 0:
#                     writer.writerow(row)
#
#     if WRITE_RESULTS_TO_FILE:
#         with open(RESULTS_FILE_NAME, "a") as f:
#             f.write(f"Minimax (depth={minimax_depth}) vs Greedy strategy results over {num_simulations} games: { {k: v for k, v in minimax_vs_greedy_results.items() if k != 'feature_logs'} }\n")
#             f.write("-" * 60 + "\n")

if __name__ == "__main__":
    # Parameters
    num_simulations = 300
    minimax_depth = 2
    WRITE_RESULTS_TO_FILE = False
    RESULTS_FILE_NAME = "results.txt"
    feature_log_file = "" 
    log_interval = 3  
    root_player = 1

    block_print()
    minimax_vs_minimax_results = run_minimax_vs_minimax_simulations(
        num_simulations, minimax_depth, feature_log_file=None, root_player=root_player
    )
    enable_print()

    # Print minimax vs minimax results
    print(f"Minimax (depth={minimax_depth}) vs Minimax strategy results over {num_simulations} games:", {k: v for k, v in minimax_vs_minimax_results.items() if k != "feature_logs"})

    # Write features to CSV for root_player, with proper header only if file does not exist
    feature_logs = minimax_vs_minimax_results.get("feature_logs", [])
    if feature_logs and feature_log_file:
        header = [
            "material_diff", "mobility_diff", "shooting_diff", "pieces_in_danger_diff",
            "safe_pieces_diff", "avg_distance_to_enemy_diff", "clustering_diff", "board_centrality_diff", "winner"
        ]
        file_exists = os.path.isfile(feature_log_file)
        with open(feature_log_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            for idx, row in enumerate(feature_logs):
                # Write only rows at multiples of log_interval (including 0)
                if idx % log_interval == 0:
                    writer.writerow(row)

    if WRITE_RESULTS_TO_FILE:
        with open(RESULTS_FILE_NAME, "a") as f:
            f.write(f"Minimax (depth={minimax_depth}) vs Minimax strategy results over {num_simulations} games: { {k: v for k, v in minimax_vs_minimax_results.items() if k != 'feature_logs'} }\n")
            f.write("-" * 60 + "\n")