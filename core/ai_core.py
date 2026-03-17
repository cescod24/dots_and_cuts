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


def state_to_vector_v2(game_state: GameState, current_player: int):
    """
    Enhanced state representation for RL v2.

    Adds 4 tactical layers on top of the v1 features:
      - shoot_threat:      for each of my pieces, how many enemies can shoot it (normalized)
      - shoot_opportunity: for each of my pieces, how many enemies it can shoot (normalized)
      - my_reachable:      binary map of vertices any of my pieces can move to
      - enemy_reachable:   binary map of vertices any enemy piece can move to

    Total: 12 layers x N^2  (v1 had 8 layers x N^2)
    For 9x9 board: 12 x 81 = 972 dims  (v1: 648)
    """
    N = game_state.board.size
    opponent = 2 if current_player == 1 else 1

    # ---- v1 layers (recomputed here to avoid double call overhead) ----
    p1 = np.zeros((N, N))
    p2 = np.zeros((N, N))
    orth = np.zeros((N, N))
    diag = np.zeros((N, N))
    arrival_layer = np.zeros((N, N))
    mobility = np.zeros((N, N))

    max_arrival = max((p.arrival_order for p in game_state.pieces), default=1)

    # Pre-compute legal actions for every piece (reused for multiple layers)
    piece_actions = {}
    for piece in game_state.pieces:
        actions = generate_legal_actions(game_state, piece)
        piece_actions[id(piece)] = actions

        x, y = piece.x, piece.y
        if piece.player == 1:
            p1[y, x] = 1
        else:
            p2[y, x] = 1
        if piece.kind == "orthogonal":
            orth[y, x] = 1
        else:
            diag[y, x] = 1
        arrival_layer[y, x] = piece.arrival_order / max_arrival

        enemy_count = sum(1 for p in game_state.pieces if p.player != piece.player)
        max_possible = 4 + enemy_count
        mobility[y, x] = len(actions) / max_possible if max_possible > 0 else 0

    if current_player == 1:
        my_pieces, enemy_pieces = p1, p2
    else:
        my_pieces, enemy_pieces = p2, p1

    z = np.array(game_state.board.z)

    edge_count = np.zeros((N, N))
    for v1, v2 in game_state.visited_edges:
        x1, y1 = v1
        x2, y2 = v2
        edge_count[y1, x1] += 1
        edge_count[y2, x2] += 1
    edge_count = edge_count / 8

    # ---- v2 tactical layers ----
    shoot_threat = np.zeros((N, N))       # danger to my pieces
    shoot_opportunity = np.zeros((N, N))  # attack potential for my pieces
    my_reachable = np.zeros((N, N))       # vertices I can move to
    enemy_reachable = np.zeros((N, N))    # vertices enemy can move to

    my_piece_list = [p for p in game_state.pieces if p.player == current_player]
    enemy_piece_list = [p for p in game_state.pieces if p.player == opponent]
    max_enemies = max(len(enemy_piece_list), 1)

    for piece in my_piece_list:
        actions = piece_actions[id(piece)]
        # Count shoot opportunities for this piece
        shoot_count = sum(1 for a in actions if a.action_type == "shoot")
        shoot_opportunity[piece.y, piece.x] = shoot_count / max_enemies
        # Reachable vertices from my pieces
        for a in actions:
            if a.action_type == "move":
                my_reachable[a.target_y, a.target_x] = 1

    for piece in enemy_piece_list:
        actions = piece_actions[id(piece)]
        # Count how many of my pieces this enemy can shoot -> threat
        for a in actions:
            if a.action_type == "shoot":
                shoot_threat[a.target_y, a.target_x] += 1
            elif a.action_type == "move":
                enemy_reachable[a.target_y, a.target_x] = 1

    # Normalize threat by max possible shooters
    shoot_threat = np.minimum(shoot_threat / max_enemies, 1.0)

    state_vector = np.concatenate([
        my_pieces.flatten(),
        enemy_pieces.flatten(),
        orth.flatten(),
        diag.flatten(),
        z.flatten(),
        edge_count.flatten(),
        arrival_layer.flatten(),
        mobility.flatten(),
        # v2 tactical layers
        shoot_threat.flatten(),
        shoot_opportunity.flatten(),
        my_reachable.flatten(),
        enemy_reachable.flatten(),
    ], dtype=float)

    return state_vector


def state_to_vector_v3(game_state: GameState, current_player: int):
    """
    Optimized state representation for RL v3.
    
    Based on V2 with strategic improvements:
    - REMOVES: arrival_order (low impact)
    - ADDS: territory_control (strategic dominance per vertex)
    - ADDS: piece_vulnerability (exposure/safety of pieces)
    
    Total: 13 layers x N^2
    For 9x9 board: 13 x 81 = 1053 dims (was 972 in v2)
    
    Territory control: 0.5 = neutral, 1.0 = fully my control, 0.0 = enemy control
    Piece vulnerability: 0.0 = safe, 1.0 = in immediate danger (can be shot)
    """
    N = game_state.board.size
    opponent = 2 if current_player == 1 else 1
    
    # ---- v2 layers (foundation) ----
    p1 = np.zeros((N, N))
    p2 = np.zeros((N, N))
    orth = np.zeros((N, N))
    diag = np.zeros((N, N))
    mobility = np.zeros((N, N))
    
    # Pre-compute legal actions for every piece
    piece_actions = {}
    for piece in game_state.pieces:
        actions = generate_legal_actions(game_state, piece)
        piece_actions[id(piece)] = actions
        
        x, y = piece.x, piece.y
        if piece.player == 1:
            p1[y, x] = 1
        else:
            p2[y, x] = 1
        if piece.kind == "orthogonal":
            orth[y, x] = 1
        else:
            diag[y, x] = 1
        
        enemy_count = sum(1 for p in game_state.pieces if p.player != piece.player)
        max_possible = 4 + enemy_count
        mobility[y, x] = len(actions) / max_possible if max_possible > 0 else 0
    
    if current_player == 1:
        my_pieces, enemy_pieces = p1, p2
    else:
        my_pieces, enemy_pieces = p2, p1
    
    z = np.array(game_state.board.z)
    
    edge_count = np.zeros((N, N))
    for v1, v2_edge in game_state.visited_edges:
        x1, y1 = v1
        x2, y2 = v2_edge
        edge_count[y1, x1] += 1
        edge_count[y2, x2] += 1
    edge_count = edge_count / 8
    
    # ---- v2 tactical layers ----
    shoot_threat = np.zeros((N, N))
    shoot_opportunity = np.zeros((N, N))
    my_reachable = np.zeros((N, N))
    enemy_reachable = np.zeros((N, N))
    
    my_piece_list = [p for p in game_state.pieces if p.player == current_player]
    enemy_piece_list = [p for p in game_state.pieces if p.player == opponent]
    max_enemies = max(len(enemy_piece_list), 1)
    
    for piece in my_piece_list:
        actions = piece_actions[id(piece)]
        shoot_count = sum(1 for a in actions if a.action_type == "shoot")
        shoot_opportunity[piece.y, piece.x] = shoot_count / max_enemies
        for a in actions:
            if a.action_type == "move":
                my_reachable[a.target_y, a.target_x] = 1
    
    for piece in enemy_piece_list:
        actions = piece_actions[id(piece)]
        for a in actions:
            if a.action_type == "shoot":
                shoot_threat[a.target_y, a.target_x] += 1
            elif a.action_type == "move":
                enemy_reachable[a.target_y, a.target_x] = 1
    
    shoot_threat = np.minimum(shoot_threat / max_enemies, 1.0)
    
    # ---- v3 new layers ----
    # Territory control: measure who has advantage at each vertex
    # 1.0 = my piece, 0.5 = neutral, 0.0 = enemy piece, or based on reachability
    territory_control = np.zeros((N, N))
    for y in range(N):
        for x in range(N):
            if my_pieces[y, x] == 1:
                territory_control[y, x] = 1.0
            elif enemy_pieces[y, x] == 1:
                territory_control[y, x] = 0.0
            else:
                # Neutral: advantage to whoever can reach first
                my_reach = my_reachable[y, x]
                enemy_reach = enemy_reachable[y, x]
                if my_reach and not enemy_reach:
                    territory_control[y, x] = 0.8
                elif enemy_reach and not my_reach:
                    territory_control[y, x] = 0.2
                elif my_reach and enemy_reach:
                    territory_control[y, x] = 0.5
                else:
                    territory_control[y, x] = 0.5
    
    # Piece vulnerability: how exposed is each of my pieces?
    # 0.0 = safe, 1.0 = directly threatened by multiple enemies
    piece_vulnerability = np.zeros((N, N))
    for piece in my_piece_list:
        y, x = piece.y, piece.x
        # Direct threat: enemy can shoot this piece
        threat_count = shoot_threat[y, x] * max_enemies
        # Exposure: how many enemy pieces can reach this location
        exposure = 0
        for enemy in enemy_piece_list:
            for a in piece_actions[id(enemy)]:
                if a.action_type == "move" and a.target_x == x and a.target_y == y:
                    exposure += 1
        # Combined vulnerability
        piece_vulnerability[y, x] = min((threat_count + exposure * 0.5) / max(max_enemies, 2), 1.0)
    
    state_vector = np.concatenate([
        my_pieces.flatten(),
        enemy_pieces.flatten(),
        orth.flatten(),
        diag.flatten(),
        z.flatten(),
        edge_count.flatten(),
        mobility.flatten(),
        # v2 tactical layers
        shoot_threat.flatten(),
        shoot_opportunity.flatten(),
        my_reachable.flatten(),
        enemy_reachable.flatten(),
        # v3 new layers (replacing arrival_order)
        territory_control.flatten(),
        piece_vulnerability.flatten(),
    ], dtype=float)
    
    return state_vector

