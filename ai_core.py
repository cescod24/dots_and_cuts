from dotscuts import GameState, Piece

class Action:
    def __init__(self, piece, action_type, target_x, target_y):
        self.piece = piece
        self.action_type = action_type  # "move" or "shoot"
        self.target_x = target_x
        self.target_y = target_y

def generate_legal_actions(game_state: GameState, piece: Piece) -> list:
    """
    Generate legal actions for a given piece
    """
    
    legal_actions = []
    enemy_pieces = [p for p in game_state.pieces if p.player != piece.player]

    # Available moves
    directions = []
    if piece.kind == "diagonal":
        directions = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
    elif piece.kind == "orthogonal":
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    for dx, dy in directions:
        new_x, new_y = piece.x + dx, piece.y + dy
        if piece.can_move(new_x, new_y, game_state):
            legal_actions.append(Action(piece, "move", new_x, new_y))

    # Available shots
    for enemy_piece in enemy_pieces:
        if piece.can_shoot(enemy_piece.x, enemy_piece.y, game_state):
            new_x, new_y = enemy_piece.x, enemy_piece.y
            legal_actions.append(Action(piece, "shoot", new_x, new_y))
    
    return legal_actions

def execute_action(game_state: GameState, action: Action):
    """
    Execute the action selected
    """
    if not action or not action.piece:
        return
    
    target_x, target_y = action.target_x, action.target_y
    action_type = action.action_type
    piece = action.piece

    if action_type == "move":
        piece.move(target_x, target_y, game_state)
    elif action_type == "shoot":
        piece.shoot(target_x, target_y, game_state)
        
