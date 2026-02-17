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
        tower_influence = [[False] * self.size for _ in range(self.size)]
        bunker_influence = [[False] * self.size for _ in range(self.size)]

        for i in range(self.size - 1):
            for j in range(self.size - 1):
                if self.towers[i][j]:
                    tower_influence[i][j] = True
                    tower_influence[i+1][j] = True
                    tower_influence[i][j+1] = True
                    tower_influence[i+1][j+1] = True
                if self.bunkers[i][j]:
                    bunker_influence[i][j] = True
                    bunker_influence[i+1][j] = True
                    bunker_influence[i][j+1] = True
                    bunker_influence[i+1][j+1] = True

        for i in range(self.size):
            for j in range(self.size):
                if tower_influence[i][j] and bunker_influence[i][j]:
                    self.z[i][j] = 0
                elif tower_influence[i][j]:
                    self.z[i][j] = 1
                elif bunker_influence[i][j]:
                    self.z[i][j] = -1
                else:
                    self.z[i][j] = 0

    def place_tower(self, i, j):
        self.towers[i][j] = True
        self.recompute_z()

    def place_bunker(self, i, j):
        self.bunkers[i][j] = True
        self.recompute_z()

    def place_lake(self, i, j):
        self.lakes[i][j] = True

    def print_board(self):
        # Print z grid (vertices) only with + for 1, - for -1, and . for 0
        print("Vertices (z grid):")
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if self.z[i][j] == 1:
                    row.append("+")
                elif self.z[i][j] == -1:
                    row.append("-")
                else:
                    row.append(".")
            print(" ".join(row))
        print()
        # Print cell grid with towers, bunkers, empty as '.'
        print("Cells (towers: o, bunkers: v, empty: .):")
        for i in range(self.size - 1):
            row = []
            for j in range(self.size - 1):
                if self.towers[i][j]:
                    cell = "o"
                elif self.bunkers[i][j]:
                    cell = "v"
                elif self.lakes[i][j]:
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

    def print_game_state(self):
        # Print vertices z grid
        print("Vertices (z grid):")
        for i in range(self.board.size):
            row = []
            for j in range(self.board.size):
                val = self.board.z[i][j]
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
        for i in range(self.board.size - 1):
            row = []
            for j in range(self.board.size - 1):
                if self.board.towers[i][j]:
                    cell = "o"
                elif self.board.bunkers[i][j]:
                    cell = "v"
                elif self.board.lakes[i][j]:
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

class Piece:

    def __init__(self, kind, x, y, player):
        self.kind = kind
        self.x = x
        self.y = y
        self.player = player # Player 1 or Player 2

    def move(self, new_x, new_y, game_state):

        start = (self.x, self.y)
        end = (new_x, new_y)
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) > 1 or abs(dy) > 1:
            print("Invalid move: only one tile at a time i allowed")
            return
        
        # Orthogonal can move either vertically or horizontally by one edge (one manhattan distance...)
        if self.kind == "orthogonal" and not ((abs(dx) == 1 and dy == 0) or (dx == 0 and abs(dy) == 1)):
            print("Invalid orthogonal move")
            return
        
        # Diagonal can move diagonally (two manhattan distances)
        if self.kind == "diagonal" and not (abs(dx) == 1 and abs(dy) == 1):
            print("Invalid diagonal move")
            return

        # Check if edge already visited
        if game_state.edge_visited(start, end):
            print("Move not allowed: edge already visited.")
            return

        # Update position and edges
        self.x, self.y = new_x, new_y
        game_state.add_visited_edge(start, end)

        ### TODO ###
        # implement the logic that if two pieces occupy the same edge, then the last piece that arrived is killed.
        # logically even the same piece that arrived will die because it will be killed by the one that remains.
        # In this case the turn is still of the one who did not attack.
    
    def shoot(new_x, new_y):
        ...


if __name__ == "__main__":
    board = Board(5)
    board.place_tower(1, 1)
    board.place_tower(1, 1)
    board.place_bunker(0,1)
    board.place_lake(0, 2)
    board.print_board()

    # Demonstration of GameState and print_game_state
    game_state = GameState(5)
    # Copy board state to game_state.board for demonstration
    game_state.board = board

    # Add some pieces
    p1 = Piece("Diagonal", 1, 1, 1)
    p2 = Piece("Orthogonal", 2, 2, 2)
    game_state.pieces.append(p1)
    game_state.pieces.append(p2)

    # Mark some edges visited
    game_state.add_visited_edge((1, 1), (1, 2))
    game_state.add_visited_edge((2, 2), (3, 2))

    # Print the complete game state
    game_state.print_game_state()
    