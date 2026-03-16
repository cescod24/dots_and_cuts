from dotscuts import GameState, Piece
import numpy as np

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

def action_to_vector(action: Action):
    """
    Transforms an object of type Action into a numeric array for RL purposes.
    The vector contains:
    - piece_x, piece_y: coordinates of the piece to move/shoot from
    - target_x, target_y: coordinates of the target position or shot target
    - is_move: 1 if the action is a move, 0 otherwise
    - is_shoot: 1 if the action is a shoot, 0 otherwise
    """
    piece_x = action.piece.x
    piece_y = action.piece.y

    target_x = action.target_x
    target_y = action.target_y

    is_move = 1 if action.action_type == "move" else 0
    is_shoot = 1 if action.action_type == "shoot" else 0

    return np.array([
        piece_x,
        piece_y,
        target_x,
        target_y,
        is_move,
        is_shoot
    ], dtype=float)

def state_to_vector(game_state: GameState, current_player: int):
    """
    Transforms an object of type GameState into a numeric array for RL purposes.
    The resulting vector concatenates the following layers (each flattened):
    - my_pieces: binary layer marking positions of the current player's pieces
    - enemy_pieces: binary layer marking positions of the opponent's pieces
    - orth: binary layer marking pieces of kind 'orthogonal'
    - diag: binary layer marking pieces of kind 'diagonal'
    - z: numeric layer representing board heights or other board-specific values
    - edge_count: normalized count of visited edges adjacent to each cell, scaled to [0,1]
    - arrival_layer: normalized arrival order of pieces, indicating sequence in which pieces arrived on the board
    These layers collectively represent the full state of the game for input into the RL model.
    """
    N = game_state.board.size

    p1 = np.zeros((N, N))
    p2 = np.zeros((N, N))
    orth = np.zeros((N, N))
    diag = np.zeros((N, N))
    arrival_layer = np.zeros((N, N))
    mobility = np.zeros((N, N))

    max_arrival = max([p.arrival_order for p in game_state.pieces], default=1)
    enemy_pieces_count = len([p for p in game_state.pieces if p.player != current_player])

    for piece in game_state.pieces:

        x = piece.x
        y = piece.y

        if piece.player == 1:
            p1[y, x] = 1
        else:
            p2[y, x] = 1

        if piece.kind == "orthogonal":
            orth[y, x] = 1
        else:
            diag[y, x] = 1

        arrival_layer[y, x] = piece.arrival_order / max_arrival

        enemy_pieces_count = len([p for p in game_state.pieces if p.player != piece.player])
        actions = generate_legal_actions(game_state, piece)
        max_possible_actions = 4 + enemy_pieces_count
        mobility[y, x] = len(actions) / max_possible_actions if max_possible_actions > 0 else 0

    # Convert absolute player layers into perspective-based layers
    if current_player == 1:
        my_pieces = p1
        enemy_pieces = p2
    else:
        my_pieces = p2
        enemy_pieces = p1

    z = np.array(game_state.board.z)

    edge_count = np.zeros((N, N))

    for v1, v2 in game_state.visited_edges:
        x1, y1 = v1
        x2, y2 = v2
        edge_count[y1, x1] += 1
        edge_count[y2, x2] += 1

    edge_count = edge_count / 8 # v edge in edge_count edge € [0,1]

    state_vector = np.concatenate([
        my_pieces.flatten(),
        enemy_pieces.flatten(),
        orth.flatten(),
        diag.flatten(),
        z.flatten(),
        edge_count.flatten(),
        arrival_layer.flatten(),
        mobility.flatten()
    ], dtype=float)

    return state_vector
        
