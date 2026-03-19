from dotscuts import GameState
from ai_core import Action, generate_legal_actions, generate_all_actions, execute_action
import random
import numpy as np

def evaluate_position_v1(game_state: GameState, current_player: int,
                         weights, means, stds, intercept) -> float:
    """
    Evaluates a GameState according to fixed rules (version 1)
    In this case "v1" means that this is the first evaluate function:
    --> features, weight, logistic regression, linear score
    """
    score = 0

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
        # A piece is in danger if it can be shot by an enemy in the next turn
        enemy = 1 if piece.player == 2 else 2
        for enemy_piece in get_pieces(state, enemy):
            # Assume enemy_piece has a method can_shoot(x, y, state)
            # Use piece.x and piece.y directly
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

    # Build feature vector in the SAME ORDER used for training
    features = np.array([
        material_diff,
        mobility_diff,
        shooting_diff,
        pieces_in_danger_diff,
        safe_pieces_diff,
        avg_distance_to_enemy_diff,
        clustering_diff,
        board_centrality_diff
    ])

    # Standardize using trained scaler statistics
    features_scaled = (features - means) / stds

    # Logistic regression linear score (logit)
    score = np.dot(weights, features_scaled) + intercept

    return float(score)


# Dictionary to hold multiple minimax AI versions.
# To add new versions, define their evaluation functions and parameters,
# then add them here with a unique key.
MINIMAX_VERSIONS = {
    "v1": {
        "evaluate_position": lambda state, player: evaluate_position_v1(
            state, player,
            weights=MINIMAX_VERSIONS["v1"]["weights"],
            means=MINIMAX_VERSIONS["v1"]["means"],
            stds=MINIMAX_VERSIONS["v1"]["stds"],
            intercept=MINIMAX_VERSIONS["v1"]["intercept"]
        ),
        "weights":  np.array([
                    0.37511014,
                    0.1517887,
                    0.0344337,
                    0.13282581,
                    0.23723877,
                    0.04105137,
                    -0.21269454,
                    0.07904447
                    ]),
        "means":    np.array([
                    0.11232675,
                    1.19902684,
                    0.12217635,
                    -0.11828369,
                    0.23061044,
                    -0.41045414,
                    -2.18466529,
                    1.28327927
                    ]),
        "stds":     np.array([
                    0.53652077,
                    2.06301329,
                    0.37892051,
                    0.36825379,
                    0.64214112,
                    1.14836881,
                    4.07156428,
                    1.9309659
                    ]),
        "intercept": 0.3122567689286516
    },
    "v2": {
        "evaluate_position": lambda state, player: evaluate_position_v1(
            state, player,
            weights=MINIMAX_VERSIONS["v2"]["weights"],
            means=MINIMAX_VERSIONS["v2"]["means"],
            stds=MINIMAX_VERSIONS["v2"]["stds"],
            intercept=MINIMAX_VERSIONS["v2"]["intercept"]
        ),
        "weights":  np.array([
                    0.25779231,
                    0.33878357,
                    -0.02898758,
                    0.02717691,##
                    0.20366564,
                    0.00593692,##
                    -0.05804915,
                    0.2017183
                    ]),
        "means":    np.array([
                    -0.00836355,
                    -0.0083293,
                    -0.00945482,
                    0.01096018,
                    -0.01932372,
                    0.00276476,
                    0.01844728,
                    -0.0431036 
                    ]),
        "stds":     np.array([
                    0.33630206,
                    1.64375838,    
                    0.20803943,
                    0.19797178,
                    0.39926138,
                    0.77341806,
                    2.44407136,
                    1.74754301
                    ]),
        "intercept": -0.0990457519794764
    },
}

def minimax(game_state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool, root_player: int, version: str = "v1") -> float:
    """
    Minimax with AB pruning, supporting multiple AI versions.
    """
    evaluate_position = MINIMAX_VERSIONS[version]["evaluate_position"]

    game_over, winner = game_state.is_game_over()

    if game_over:
        # Terminal evaluation is always from root_player's perspective:
        # root_player wins → +inf, root_player loses → -inf
        if winner == root_player:
            return float("inf")
        elif winner is not None:
            return float("-inf")
        else:
            return 0  # Draw

    player = root_player if maximizing_player else (2 if root_player == 1 else 1)
        
    if depth == 0:
        return evaluate_position(game_state, root_player)
    
    if maximizing_player:
        player_actions = generate_all_actions(game_state, player)
        random.shuffle(player_actions)
        max_eval = float("-inf")

        for action in player_actions:
            # print("Game state before action")
            # game_state.print_game_state()
            execute_action(game_state, action)
            # print("Game state after action")
            # game_state.print_game_state()
            score = minimax(game_state, depth-1, alpha, beta, False, root_player, version=version)
            game_state.undo_last_move()
            # print("Game state after undo")
            # game_state.print_game_state()

            max_eval = max(max_eval, score)
            alpha = max(alpha, max_eval)
            if alpha >= beta:
                break
            
        return max_eval
    
    if not maximizing_player:
        player_actions = generate_all_actions(game_state, player)
        random.shuffle(player_actions)
        min_eval = float("inf")

        for action in player_actions:
            # print("Game state before action")
            # game_state.print_game_state()
            execute_action(game_state, action)
            # print("Game state after action")
            # game_state.print_game_state()
            score = minimax(game_state, depth-1, alpha, beta, True, root_player, version=version)
            game_state.undo_last_move()
            # print("Game state after undo")
            # game_state.print_game_state()

            min_eval = min(min_eval, score)
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        
        return min_eval
    
def minimax_best_move(game_state: GameState, player: int, depth: int, version: str = "v1") -> Action:
    """
    Returns the best action for the player using minimax search with specified version.
    """
    actions = generate_all_actions(game_state, player)
    best_score = float("-inf")
    best_actions = []
    all_scores = []
    for action in actions:
        execute_action(game_state, action)
        score = minimax(game_state, depth-1, alpha=float("-inf"), beta=float("inf"), maximizing_player=False, root_player=player, version=version)
        all_scores.append(score)
        game_state.undo_last_move()

        if score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)

    if all_scores:
        print(
            f"[DEBUG] Scores -> "
            f"min: {min(all_scores):.3f}, "
            f"max: {max(all_scores):.3f}, "
            f"avg: {sum(all_scores)/len(all_scores):.3f}, "
            f"best: {best_score:.3f}"
        )

    return random.choice(best_actions) if best_actions else None


if __name__ == "__main__":

    ###### WEIGHTS ANALYSIS ##############
    import pandas as pd

    # Carica il CSV
    df = pd.read_csv("feature_log_v2_depth2.csv")
    print(df.columns)

    # Target: 1 se vince player 1, 0 se vince player 2
    df["y"] = (df["winner"] == 1).astype(int)

    # Feature matrix
    feature_cols = [
        "material_diff",
        "mobility_diff",
        "shooting_diff",
        "pieces_in_danger_diff",
        "safe_pieces_diff",
        "avg_distance_to_enemy_diff",
        "clustering_diff",
        "board_centrality_diff"
    ]

    X = df[feature_cols].values
    y = df["y"].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Means:", scaler.mean_)
    print("Stds:", scaler.scale_)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    weights = model.coef_[0]
    intercept = model.intercept_[0]
    print("Weights raw:", model.coef_[0])
    print("Feature order:", feature_cols)
    print("Intercept:", intercept)

    # To test different AI versions, change the version string below:
    # version_to_test = "v1"
    # You can then call minimax_best_move or minimax with version=version_to_test
    # Example:
    # best_action = minimax_best_move(game_state, player=1, depth=3, version=version_to_test)