class Board:

    """
    size: no. of vertices
    cells: no of vertices - 1
    """
    
    def __init__(self, size):
        self.size = size
        self.towers = [[False] * (size - 1) for _ in range(size - 1)]
        self.bunkers = [[False] * (size - 1) for _ in range(size - 1)]
        self.lakes = [[False] * (size - 1) for _ in range(size - 1)]
        self.z = [[0] * size for _ in range(size)]

    def recompute_z(self):
        """
        Recompute the z grid based on current towers and bunkers.
        Note: (x, y) is user-facing, but all internal matrices are accessed as [y][x].
        """
        tower_influence = [[False] * self.size for _ in range(self.size)]
        bunker_influence = [[False] * self.size for _ in range(self.size)]

        # Loop order: y is row, x is column. Internal access is [y][x].
        for y in range(self.size - 1):
            for x in range(self.size - 1):
                if self.towers[y][x]:
                    tower_influence[y][x] = True
                    tower_influence[y+1][x] = True
                    tower_influence[y][x+1] = True
                    tower_influence[y+1][x+1] = True
                if self.bunkers[y][x]:
                    bunker_influence[y][x] = True
                    bunker_influence[y+1][x] = True
                    bunker_influence[y][x+1] = True
                    bunker_influence[y+1][x+1] = True

        for y in range(self.size):
            for x in range(self.size):
                if tower_influence[y][x] and bunker_influence[y][x]:
                    self.z[y][x] = 0
                elif tower_influence[y][x]:
                    self.z[y][x] = 1
                elif bunker_influence[y][x]:
                    self.z[y][x] = -1
                else:
                    self.z[y][x] = 0

    def place_tower(self, x, y):
        """
        Place a tower at (x, y) as provided by user.
        Internally, access self.towers as [y][x].
        """
        self.towers[y][x] = True
        self.recompute_z()

    def place_bunker(self, x, y):
        """
        Place a bunker at (x, y) as provided by user.
        Internally, access self.bunkers as [y][x].
        """
        self.bunkers[y][x] = True
        self.recompute_z()

    def place_lake(self, x, y):
        """
        Place a lake at (x, y) as provided by user.
        Internally, access self.lakes as [y][x].
        """
        self.lakes[y][x] = True

    def print_board(self):
        """
        Print the current board state.
        Note: (x, y) is user-facing, but all internal matrices are accessed as [y][x].
        """
        # Print z grid (vertices) only with + for 1, - for -1, and . for 0
        print("Vertices (z grid):")
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if self.z[y][x] == 1:
                    row.append("+")
                elif self.z[y][x] == -1:
                    row.append("-")
                else:
                    row.append(".")
            print(" ".join(row))
        print()
        # Print cell grid with towers, bunkers, lakes, empty as '.'
        print("Cells (towers: o, bunkers: v, lakes: x, empty: .):")
        for y in range(self.size - 1):
            row = []
            for x in range(self.size - 1):
                if self.towers[y][x]:
                    cell = "o"
                elif self.bunkers[y][x]:
                    cell = "v"
                elif self.lakes[y][x]:
                    cell = "x"
                else:
                    cell = "."
                row.append(cell)
            print(" ".join(row))

class GameState:

    def __init__(self, board):
        self.board = board
        self.pieces = []  # List to hold pieces
        self.visited_edges = set() 
        self.move_counter = 0  # global counter to track arrival order on vertices
        self.initialize_lake_edges()

    def initialize_lake_edges(self):
        """
        Mark all edges around all lake cells (orthogonal and diagonal) as visited.
        Note: (x, y) is user-facing, but all internal matrices are accessed as [y][x].
        """
        for y in range(self.board.size - 1):
            for x in range(self.board.size - 1):
                if self.board.lakes[y][x]:
                    # Vertices of cell (x, y): (x, y), (x+1, y), (x, y+1), (x+1, y+1)
                    v_tl = (x, y)
                    v_tr = (x+1, y)
                    v_bl = (x, y+1)
                    v_br = (x+1, y+1)
                    # Four sides and two diagonals
                    edges = [
                        (v_tl, v_tr),  # top
                        (v_tr, v_br),  # right
                        (v_br, v_bl),  # bottom
                        (v_bl, v_tl),  # left
                        (v_tl, v_br),  # diagonal
                        (v_tr, v_bl)   # diagonal
                    ]
                    for v1, v2 in edges:
                        self.add_visited_edge(v1, v2)

    def setup_board(self):
        # Placeholder for board setup logic if needed
        pass

    def setup_pieces(self, piece):
        # Placeholder for initializing pieces on the board
        self.pieces.append(piece)
        pass

    def reset(self):
        self.board = Board(self.board.size)
        self.pieces.clear()
        self.visited_edges.clear()
        self.setup_board()
        self.setup_pieces()

    def add_visited_edge(self, v1, v2):
        # sort the vertices so (v1,v2) == (v2,v1)
        edge = tuple(sorted([v1, v2]))
        self.visited_edges.add(edge)

    def edge_visited(self, v1, v2):
        edge = tuple(sorted([v1, v2]))
        return edge in self.visited_edges

    def resolve_vertex_conflict(self, vertex, attacking_piece):
        """
        Handles combat resolution on a vertex.
        Returns a list of pieces to be removed (but does not remove them).
        Rules:
        - If exactly 1 opponent is present → standard 1v1:
          the opponent dies and the attacker survives.
        - If 2 or more opponents are present → collapse rule:
          the last arrived opponent dies and the attacker also dies.
        """
        x, y = vertex
        # Find opponent pieces on the same vertex
        opponents = [
            p for p in self.pieces
            if p.x == x and p.y == y and p.player != attacking_piece.player
        ]

        if len(opponents) == 0:
            return []  # No combat, nothing to remove

        # 1v1 case
        if len(opponents) == 1:
            opponent = opponents[0]
            return [opponent]  # attacker survives, opponent dies

        # Collapse case: 2 or more opponents
        last_arrived = max(opponents, key=lambda p: p.arrival_order)
        to_remove = []
        to_remove.append(last_arrived)
        to_remove.append(attacking_piece)
        return to_remove

    def apply_conflict_resolution(self, removed_pieces):
        """
        Actually removes the pieces from the board and prints capture messages.
        """
        for piece in removed_pieces:
            if piece in self.pieces:
                self.pieces.remove(piece)
                print(f"Player {piece} was captured.")

    def place_piece_with_tail(self, position_x, position_y, tail_x, tail_y, kind, player):
        """
        Place a piece of the given kind and player at (position_x, position_y), using (tail_x, tail_y) as the tail.
        Validates that the move from tail to position is legal for the kind, and that the destination is unoccupied.
        Marks the edge from tail to position as visited.
        Prints an error if validation fails.
        """
        # Check that both the tail and the position are within board bounds
        if not (0 <= position_x < self.board.size and 0 <= position_y < self.board.size):
            print("Error: position out of bounds.")
            return
        if not (0 <= tail_x < self.board.size and 0 <= tail_y < self.board.size):
            print("Error: tail out of bounds.")
            return

        # Ensure that a piece's tail and position are not exactly the same
        if position_x == tail_x and position_y == tail_y:
            print("Error: tail and position cannot be exactly the same.")
            return

        # Validate move from tail to position according to kind
        dx = position_x - tail_x
        dy = position_y - tail_y

        kind_lower = kind.lower()
        if kind_lower == "orthogonal":
            if not ((abs(dx) == 1 and dy == 0) or (dx == 0 and abs(dy) == 1)):
                print("Error: invalid orthogonal move from tail to position.")
                return
        elif kind_lower == "diagonal":
            if not (abs(dx) == 1 and abs(dy) == 1):
                print("Error: invalid diagonal move from tail to position.")
                return
        else:
            print("Error: unknown piece kind.")
            return

        # Create the piece at the position
        piece = Piece(kind, position_x, position_y, player)
        self.move_counter += 1
        piece.arrival_order = self.move_counter
        self.pieces.append(piece)
        # Mark edge as visited
        self.add_visited_edge((tail_x, tail_y), (position_x, position_y))

    def print_game_state(self):
        # Print vertices z grid
        print("Vertices (z grid):")
        for y in range(self.board.size):
            row = []
            for x in range(self.board.size):
                val = self.board.z[y][x]
                if val == 1:
                    row.append("+")
                elif val == -1:
                    row.append("-")
                else:
                    row.append(".")
            print(" ".join(row))
        print()
        # Print cells grid
        print("Cells (towers: o, bunkers: v, lakes: x, empty: .):")
        for y in range(self.board.size - 1):
            row = []
            for x in range(self.board.size - 1):
                if self.board.towers[y][x]:
                    cell = "o"
                elif self.board.bunkers[y][x]:
                    cell = "v"
                elif self.board.lakes[y][x]:
                    cell = "x"
                else:
                    cell = "."
                row.append(cell)
            print(" ".join(row))
        print()
        # Print pieces on the board
        print("Pieces on board:")
        if not self.pieces:
            print("No pieces on the board.")
        else:
            for piece in self.pieces:
                print(f"Player {piece.player} {piece.kind} at ({piece.x}, {piece.y})")
        print()
        # Print visited edges
        print("Visited edges:")
        if not self.visited_edges:
            print("No visited edges.")
        else:
            for edge in sorted(self.visited_edges):
                print(f"{edge[0]} <-> {edge[1]}")
        print()


    def is_game_over(self):
        """
        Returns (True, winner) if the game is over, (False, None) otherwise.
        Game is over if a player has no pieces left or no legal move/shoot.
        Winner is the other player (1 or 2).
        """
        for player in [1, 2]:
            player_pieces = [p for p in self.pieces if p.player == player]
            if not player_pieces:
                # This player has no pieces, so other player wins
                return True, 2 if player == 1 else 1
            can_act = any(p.has_legal_move_or_shoot(self) for p in player_pieces)
            if not can_act:
                # This player cannot act, so other player wins
                return True, 2 if player == 1 else 1
        return False, None

class Piece:

    def __init__(self, kind, x, y, player):
        self.kind = kind
        self.x = x
        self.y = y
        self.player = player # Player 1 or Player 2
        self.arrival_order = 0  # will be updated by GameState when placed or moved

    def can_move(self, new_x, new_y, game_state):
        """
        Returns True if moving to (new_x, new_y) is legal.
        Checks:
        - Inside board bounds
        - Move distance is exactly one step
        - Direction matches piece kind
        - Edge not already visited
        """
        # Check bounds
        if not (0 <= new_x < game_state.board.size and 0 <= new_y < game_state.board.size):
            return False

        dx = new_x - self.x
        dy = new_y - self.y

        # Must move exactly one step
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        if dx == 0 and dy == 0:
            return False

        # Direction rules
        if self.kind == "orthogonal":
            if not ((abs(dx) == 1 and dy == 0) or (dx == 0 and abs(dy) == 1)):
                return False
        elif self.kind == "diagonal":
            if not (abs(dx) == 1 and abs(dy) == 1):
                return False
        else:
            return False

        # Edge must not be visited
        if game_state.edge_visited((self.x, self.y), (new_x, new_y)):
            return False

        return True

    def move(self, new_x, new_y, game_state):
        start = (self.x, self.y)
        end = (new_x, new_y)
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if not self.can_move(new_x, new_y, game_state):
            print("Invalid move.")
            return

        # Update position and edges
        self.x, self.y = new_x, new_y
        game_state.add_visited_edge(start, end)

        # Update arrival order
        game_state.move_counter += 1
        self.arrival_order = game_state.move_counter

        # Resolve conflict
        removed_pieces = game_state.resolve_vertex_conflict((new_x, new_y), self)
        game_state.apply_conflict_resolution(removed_pieces)
    
    def can_shoot(self, target_x, target_y, game_state):
        """
        Returns True if a shoot from self.x, self.y to target_x,target_y
        is legal according to z rules and piece type, AND there is an enemy piece at the target.
        Does NOT execute the shoot.
        """
        # First, check if an enemy piece exists at (target_x, target_y)
        enemy_at_target = any(
            p.x == target_x and p.y == target_y and p.player != self.player
            for p in game_state.pieces
        )
        if not enemy_at_target:
            return False

        dx = target_x - self.x
        dy = target_y - self.y

        if dx == 0 and dy == 0:
            return False  # same position

        # Direction rules
        if self.kind == "diagonal" and not (dx == 0 or dy == 0):
            return False
        if self.kind == "orthogonal" and abs(dx) != abs(dy):
            return False

        # Normalize direction
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        current_x, current_y = self.x, self.y
        z_start = game_state.board.z[self.x][self.y]
        z_end = game_state.board.z[target_x][target_y]

        while (current_x, current_y) != (target_x, target_y):
            current_x += step_x
            current_y += step_y
            z_mid = game_state.board.z[current_x][current_y]

            # z rules
            if z_start == 1 and z_end == 1:
                continue
            elif z_start == -1 and z_end == -1:
                if z_mid != -1:
                    print("Failed shooting attempt (-1->-1)")
                    return False
            elif z_start == 0 and z_end == 0:
                if z_mid not in (0, -1):
                    print("Failed shooting attempt (0->0)")
                    return False
            elif z_start == 1 and z_end == 0:
                if z_mid not in (0, -1):
                    print("Failed shooting attempt (1->0)")
                    return False
            elif z_start == 0 and z_end == 1:
                if z_mid not in (0, -1):
                    print("Failed shooting attempt (0->1)")
                    return False
            else:
                print("Failed shooting attempt (-1 -> 0, 0->-1, 1->-1, -1->1)")
                return False

        return True

    def shoot(self, new_x, new_y, game_state):
        """
        Executes the shot, if it can.
        """
        if not self.can_shoot(new_x, new_y, game_state):
            print("Invalid shoot action.")
            return

        start = (self.x, self.y)
        end = (new_x, new_y)

        dx = new_x - self.x
        dy = new_y - self.y

        # Normalize direction
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        current_x, current_y = self.x, self.y

        path_vertices = []

        # Traverse path (excluding start, including end)
        while (current_x, current_y) != (new_x, new_y):
            next_x = current_x + step_x
            next_y = current_y + step_y

            path_vertices.append((next_x, next_y)) ###  Possibile ottimizzazione qua
            current_x, current_y = next_x, next_y

        # Mark edges as visited after validation 
        current_x, current_y = start
        for vx, vy in path_vertices:
            game_state.add_visited_edge(
                (current_x, current_y),
                (vx, vy)
            )
            current_x, current_y = vx, vy

        # Move piece
        self.x = new_x
        self.y = new_y

        game_state.move_counter += 1
        self.arrival_order = game_state.move_counter

        removed_pieces = game_state.resolve_vertex_conflict((new_x, new_y), self)
        game_state.apply_conflict_resolution(removed_pieces)

    def has_legal_move_or_shoot(self, game_state):
        """
        Returns True if this piece has at least one legal move or shoot.
        Only considers shoot targets where enemy pieces are present.
        """
        directions_orthogonal = [(1,0), (-1,0), (0,1), (0,-1)]
        directions_diagonal = [(1,1), (1,-1), (-1,1), (-1,-1)]

        # Check possible moves (one step)
        if self.kind.lower() == "orthogonal":
            moves = directions_orthogonal
        elif self.kind.lower() == "diagonal":
            moves = directions_diagonal
        else:
            moves = []

        for dx, dy in moves:
            new_x = self.x + dx
            new_y = self.y + dy
            # Check bounds
            if 0 <= new_x < game_state.board.size and 0 <= new_y < game_state.board.size:
                start = (self.x, self.y)
                end = (new_x, new_y)
                if not game_state.edge_visited(start, end):
                    return True

        # Check possible shoots (only at enemy piece locations)
        for other in game_state.pieces:
            if other.player == self.player:
                continue
            if (other.x, other.y) == (self.x, self.y):
                continue
            if self.can_shoot(other.x, other.y, game_state):
                return True
        return False


import random

def setup_standard_game():
    """
    Create a 9x9 board, place a random number of towers (6-10), bunkers (6-10), and lakes (3-5) in unique positions (no overlaps),
    with lakes never in the corners. Keeps player pieces in fixed positions.
    Returns a ready-to-use GameState.
    """
    board = Board(9)
    size = 9
    cell_coords = [(x, y) for x in range(size-1) for y in range(size-1)]
    # Corners for lakes exclusion
    corners = {(0,0), (0,size-2), (size-2,0), (size-2,size-2)}
    # Random counts
    n_towers = random.randint(5, 10)
    n_bunkers = random.randint(10, 15)
    n_lakes = random.randint(0, 1)
    # First, choose lake positions (no corners)
    possible_lake_cells = [pos for pos in cell_coords if pos not in corners]
    lake_positions = set(random.sample(possible_lake_cells, n_lakes))
    # Now, choose tower positions, avoiding lakes
    remaining_for_towers = [pos for pos in cell_coords if pos not in lake_positions]
    tower_positions = set(random.sample(remaining_for_towers, n_towers))
    # Now, choose bunker positions, avoiding both lakes and towers
    remaining_for_bunkers = [pos for pos in cell_coords if pos not in lake_positions and pos not in tower_positions]
    bunker_positions = set(random.sample(remaining_for_bunkers, n_bunkers))
    # Place on board
    for x, y in tower_positions:
        board.place_tower(x, y)
    for x, y in bunker_positions:
        board.place_bunker(x, y)
    for x, y in lake_positions:
        board.place_lake(x, y)
    # Create GameState with the board
    game_state = GameState(board)
    # Place 1 orthogonal and 1 diagonal piece for each player (fixed positions)
    # Player 1:
    game_state.place_piece_with_tail(8, 7, 8, 8, "orthogonal", 1)
    game_state.place_piece_with_tail(1, 7, 0, 8, "diagonal", 1)
    # Player 2:
    game_state.place_piece_with_tail(0, 1, 0, 0, "orthogonal", 2)
    game_state.place_piece_with_tail(7, 1, 8, 0, "diagonal", 2)
    return game_state


if __name__ == "__main__":
    # Standard board setup

    game_state = setup_standard_game()

    # Main game loop
    current_player = 1
    while True:
        print("="*40)
        print(f"Player {current_player}'s turn")
        game_state.print_game_state()
        # List player's pieces
        player_pieces = [piece for piece in game_state.pieces if piece.player == current_player]
        game_over, winner = game_state.is_game_over()
        if game_over:
            print(f"Game over! Player {winner} wins!")
            break
        # Retry loop for the player's turn
        while True:
            print("Your pieces:")
            for idx, piece in enumerate(player_pieces):
                print(f"{idx}: {piece.kind} at ({piece.x}, {piece.y})")
            # Choose piece
            while True:
                try:
                    piece_idx = int(input("Select a piece by index: "))
                    if 0 <= piece_idx < len(player_pieces):
                        selected_piece = player_pieces[piece_idx]
                        break
                    else:
                        print("Invalid index.")
                except Exception:
                    print("Please enter a valid integer index.")
            # Choose action
            while True:
                action = input("Action? (move/shoot): ").strip().lower()
                if action in ("move", "shoot"):
                    break
                else:
                    print("Type 'move' or 'shoot'.")
            # Input target coordinates
            while True:
                try:
                    target = input("Target coordinates (x y): ").strip().split()
                    if len(target) != 2:
                        print("Enter two integers separated by space.")
                        continue
                    tx, ty = int(target[0]), int(target[1])
                    if 0 <= tx < game_state.board.size and 0 <= ty < game_state.board.size:
                        break
                    else:
                        print(f"Coordinates must be between 0 and {game_state.board.size - 1}.")
                except Exception:
                    print("Invalid input. Try again.")
            # Perform the action, but only switch turn if the action is successful
            action_success = False
            if action == "move":
                print(f"Player {current_player} tries to move {selected_piece.kind} from ({selected_piece.x}, {selected_piece.y}) to ({tx}, {ty})")
                prev_pos = (selected_piece.x, selected_piece.y)
                prev_edges = set(game_state.visited_edges)
                selected_piece.move(tx, ty, game_state)
                # Check if piece position or visited_edges changed
                if (selected_piece.x, selected_piece.y) != prev_pos or game_state.visited_edges != prev_edges:
                    action_success = True
                else:
                    print("Move failed or invalid. Try again.")
                    print("Your pieces:")
                    for idx, piece in enumerate(player_pieces):
                        print(f"{idx}: {piece.kind} at ({piece.x}, {piece.y})")
            else:
                print(f"Player {current_player} tries to shoot {selected_piece.kind} from ({selected_piece.x}, {selected_piece.y}) to ({tx}, {ty})")
                prev_pos = (selected_piece.x, selected_piece.y)
                prev_edges = set(game_state.visited_edges)
                selected_piece.shoot(tx, ty, game_state)
                # Check if piece moved or visited_edges changed
                if (selected_piece.x, selected_piece.y) != prev_pos or game_state.visited_edges != prev_edges:
                    action_success = True
                else:
                    print("Shoot failed or invalid. Try again.")
                    print("Your pieces:")
                    for idx, piece in enumerate(player_pieces):
                        print(f"{idx}: {piece.kind} at ({piece.x}, {piece.y})")
            # Print current board state for debugging
            print("Current board state after action attempt:")
            game_state.print_game_state()
            # Only break (and switch player) if action succeeded
            if action_success:
                break
        # Remove any dead pieces (already handled in conflict resolution)
        # Switch player
        current_player = 2 if current_player == 1 else 1







########### TODO ###############
# 
# permettere più pezzi dello stesso player nello stesso vertice, con tail diverse,
# ora si puo ancora fare tail e pos uguali... da risolverre