from dotscuts import GameState
from ai_core import Action, generate_legal_actions, execute_action
import copy

def generate_all_actions(game_state: GameState, current_player: int) -> list:
    """
    Generates all legal actions for a player
    """
    player_pieces = [p for p in game_state.pieces if p.player == current_player]
    all_actions = []
    for piece in player_pieces:
        actions = generate_legal_actions(game_state, piece)
        all_actions.extend(actions)
    
    return all_actions


def evaluate_position(game_state: GameState, current_player: int) -> float:
    """
    Evaluates a GameState according to fixed rules
    """
    score = 0

    player_pieces = [p for p in game_state.pieces if p.player == current_player]
    enemy_pieces = [p for p in game_state.pieces if p.player != current_player]

    player_actions = generate_all_actions(game_state, current_player=current_player)
    enemy_player = 2 if current_player == 1 else 1
    enemy_actions = generate_all_actions(game_state, current_player=enemy_player)

    # 1 piece = 1 point
    pieces_difference = 1.5 * (len(player_pieces) - len(enemy_pieces))
    score += pieces_difference

    # 1 available move = 0.1 point
    moves_difference = len(player_actions) - len(enemy_actions)
    score += 0.25 * moves_difference

    # 1 shooting opportunity = 1 point
    shooting_opportunities = [a for a in player_actions if a.action_type == "shoot"] 
    score += 5 * len(shooting_opportunities)

    return score


def minimax(game_state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool, root_player: int) -> float:
    """
    Minimax with AB pruning
    """
    game_over, winner = game_state.is_game_over()
    player = root_player if maximizing_player else (2 if root_player == 1 else 1)
    opponent = 2 if player == 1 else 1

    if game_over:
        if winner == player:
            return float("inf")
        elif winner == opponent:
            return float("-inf")
        else:
            return 0 # Draw
        
    if depth == 0:
        return evaluate_position(game_state, root_player)
    
    if maximizing_player:
        player_actions = generate_all_actions(game_state, player)
        max_eval = float("-inf")

        for action in player_actions:
            clone_state = copy.deepcopy(game_state)
            execute_action(clone_state, action)
            score = minimax(clone_state, depth-1, alpha, beta, False, root_player)
            max_eval = max(max_eval, score)
            alpha = max(alpha, max_eval)
            if alpha >= beta:
                break
            
        return max_eval
    
    if not maximizing_player:
        opponent_actions = generate_all_actions(game_state, opponent)
        min_eval = float("inf")

        for action in opponent_actions:
            clone_state = copy.deepcopy(game_state)
            execute_action(clone_state, action)
            score = minimax(clone_state, depth-1, alpha, beta, True, root_player)
            min_eval = min(min_eval, score)
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        
        return min_eval
    
def minimax_best_move(game_state: GameState, player: int, depth: int) -> Action:
    """
    Returns the best action for the player using minimax search
    """
    actions = generate_all_actions(game_state, player)
    best_score = float("-inf")
    best_action = None
    for action in actions:
        clone_state = copy.deepcopy(game_state)
        execute_action(clone_state, action)
        score = minimax(clone_state, depth-1, alpha=float("-inf"), beta=float("inf"), maximizing_player=False, root_player=player)
        if score > best_score or best_action is None:
            best_score = score
            best_action = action
    return best_action


if __name__ == "__main__":
    ...