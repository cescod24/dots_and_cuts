"""
Custom Game Setup Module
========================
Allows building custom game configurations with:
- Custom board sizes
- Custom starting positions for pieces
- Custom tower/bunker/lake placements
"""

import sys
sys.path.insert(0, '../core')

from dotscuts import Board, GameState, Piece


class GameSetupBuilder:
    """
    Builds custom game configurations interactively or programmatically.
    """

    def __init__(self, size: int = 9):
        """
        Initialize the builder with a board of given size.

        Args:
            size: Board size (9x9 default)
        """
        self.size = size
        self.board = Board(size)
        self.pieces = []

    def add_tower(self, x: int, y: int):
        """
        Add a tower at cell (x, y).

        Args:
            x, y: Cell coordinates (0 to size-2)
        """
        if not (0 <= x < self.size - 1 and 0 <= y < self.size - 1):
            raise ValueError(f"Cell ({x}, {y}) out of bounds for {self.size}x{self.size} board")
        self.board.place_tower(x, y)

    def add_bunker(self, x: int, y: int):
        """
        Add a bunker at cell (x, y).
        """
        if not (0 <= x < self.size - 1 and 0 <= y < self.size - 1):
            raise ValueError(f"Cell ({x}, {y}) out of bounds for {self.size}x{self.size} board")
        self.board.place_bunker(x, y)

    def add_lake(self, x: int, y: int):
        """
        Add a lake at cell (x, y).
        Lakes block all 6 edges of a cell.
        """
        if not (0 <= x < self.size - 1 and 0 <= y < self.size - 1):
            raise ValueError(f"Cell ({x}, {y}) out of bounds for {self.size}x{self.size} board")
        self.board.place_lake(x, y)

    def add_piece(self, position_x: int, position_y: int,
                  tail_x: int, tail_y: int,
                  kind: str, player: int):
        """
        Add a piece with its tail (starting position).

        Args:
            position_x, position_y: Vertex where piece is placed
            tail_x, tail_y: Tail vertex (one step away based on kind)
            kind: "orthogonal" or "diagonal"
            player: Player 1 or 2
        """
        if not (0 <= position_x < self.size and 0 <= position_y < self.size):
            raise ValueError(f"Position ({position_x}, {position_y}) out of bounds")
        if not (0 <= tail_x < self.size and 0 <= tail_y < self.size):
            raise ValueError(f"Tail ({tail_x}, {tail_y}) out of bounds")

        self.pieces.append({
            'position': (position_x, position_y),
            'tail': (tail_x, tail_y),
            'kind': kind,
            'player': player
        })

    def build(self) -> GameState:
        """
        Build and return the configured GameState.

        Returns:
            GameState object ready to play
        """
        game_state = GameState(self.board)

        # Add all pieces
        for piece_config in self.pieces:
            pos_x, pos_y = piece_config['position']
            tail_x, tail_y = piece_config['tail']
            kind = piece_config['kind']
            player = piece_config['player']

            game_state.place_piece_with_tail(pos_x, pos_y, tail_x, tail_y, kind, player)

        return game_state

    def print_summary(self):
        """
        Print a summary of the setup.
        """
        print("\n" + "="*60)
        print(f"GAME SETUP SUMMARY ({self.size}x{self.size} board)")
        print("="*60)

        # Count obstacles
        tower_count = sum(1 for row in self.board.towers for cell in row if cell)
        bunker_count = sum(1 for row in self.board.bunkers for cell in row if cell)
        lake_count = sum(1 for row in self.board.lakes for cell in row if cell)

        print(f"Towers: {tower_count}")
        print(f"Bunkers: {bunker_count}")
        print(f"Lakes: {lake_count}")

        print(f"\nPieces ({len(self.pieces)} total):")
        for i, piece_config in enumerate(self.pieces):
            pos = piece_config['position']
            tail = piece_config['tail']
            kind = piece_config['kind']
            player = piece_config['player']
            print(f"  {i+1}. Player {player} {kind} at {pos} (tail: {tail})")

        print("="*60 + "\n")


class PrebuiltSetups:
    """
    Collection of predefined game setups.
    """

    @staticmethod
    def standard_9x9() -> GameState:
        """
        Standard 9x9 setup with random obstacles and fixed starting positions.
        """
        from dotscuts import setup_standard_game
        return setup_standard_game()

    @staticmethod
    def balanced_9x9() -> GameState:
        """
        Balanced setup with symmetrical towers and bunkers.
        """
        builder = GameSetupBuilder(size=9)

        # Add symmetrical towers
        tower_positions = [(1, 1), (7, 1), (1, 7), (7, 7)]
        for x, y in tower_positions:
            builder.add_tower(x, y)

        # Add bunkers
        bunker_positions = [(2, 2), (6, 2), (2, 6), (6, 6),
                            (3, 4), (5, 4), (4, 3), (4, 5)]
        for x, y in bunker_positions:
            builder.add_bunker(x, y)

        # Fixed starting positions
        builder.add_piece(8, 7, 8, 8, "orthogonal", 1)
        builder.add_piece(1, 7, 0, 8, "diagonal", 1)
        builder.add_piece(0, 1, 0, 0, "orthogonal", 2)
        builder.add_piece(7, 1, 8, 0, "diagonal", 2)

        return builder.build()

    @staticmethod
    def skirmish_9x9() -> GameState:
        """
        Standard 9x9 with random obstacles and 4 pieces per player.
        Extra orthogonal + diagonal in more central positions.
        """
        import random
        from dotscuts import Board
        board = Board(9)
        size = 9
        cell_coords = [(x, y) for x in range(size - 1) for y in range(size - 1)]
        corners = {(0, 0), (0, size - 2), (size - 2, 0), (size - 2, size - 2)}
        n_towers = random.randint(5, 10)
        n_bunkers = random.randint(10, 15)
        n_lakes = random.randint(0, 1)
        possible_lake_cells = [pos for pos in cell_coords if pos not in corners]
        lake_positions = set(random.sample(possible_lake_cells, n_lakes))
        remaining_for_towers = [pos for pos in cell_coords if pos not in lake_positions]
        tower_positions = set(random.sample(remaining_for_towers, n_towers))
        remaining_for_bunkers = [pos for pos in cell_coords if pos not in lake_positions and pos not in tower_positions]
        bunker_positions = set(random.sample(remaining_for_bunkers, n_bunkers))
        for x, y in tower_positions:
            board.place_tower(x, y)
        for x, y in bunker_positions:
            board.place_bunker(x, y)
        for x, y in lake_positions:
            board.place_lake(x, y)

        builder = GameSetupBuilder(size=9)
        builder.board = board
        # Standard pieces (corners)
        builder.add_piece(8, 7, 8, 8, "orthogonal", 1)
        builder.add_piece(1, 7, 0, 8, "diagonal", 1)
        builder.add_piece(0, 1, 0, 0, "orthogonal", 2)
        builder.add_piece(7, 1, 8, 0, "diagonal", 2)
        # Extra pieces (central)
        builder.add_piece(4, 6, 4, 7, "orthogonal", 1)
        builder.add_piece(5, 5, 4, 6, "diagonal", 1)
        builder.add_piece(4, 2, 4, 1, "orthogonal", 2)
        builder.add_piece(3, 3, 4, 2, "diagonal", 2)

        return builder.build()

    @staticmethod
    def mid_7x7() -> GameState:
        """
        Mid-size 7x7 with random obstacles and 3 pieces per player.
        """
        import random
        from dotscuts import Board
        board = Board(7)
        size = 7
        cell_coords = [(x, y) for x in range(size - 1) for y in range(size - 1)]
        corners = {(0, 0), (0, size - 2), (size - 2, 0), (size - 2, size - 2)}
        n_towers = random.randint(3, 6)
        n_bunkers = random.randint(5, 10)
        n_lakes = random.randint(0, 1)
        possible_lake_cells = [pos for pos in cell_coords if pos not in corners]
        lake_positions = set(random.sample(possible_lake_cells, n_lakes))
        remaining_for_towers = [pos for pos in cell_coords if pos not in lake_positions]
        tower_positions = set(random.sample(remaining_for_towers, n_towers))
        remaining_for_bunkers = [pos for pos in cell_coords if pos not in lake_positions and pos not in tower_positions]
        bunker_positions = set(random.sample(remaining_for_bunkers, n_bunkers))
        for x, y in tower_positions:
            board.place_tower(x, y)
        for x, y in bunker_positions:
            board.place_bunker(x, y)
        for x, y in lake_positions:
            board.place_lake(x, y)

        builder = GameSetupBuilder(size=7)
        builder.board = board
        # P1: corner pieces + central
        builder.add_piece(6, 5, 6, 6, "orthogonal", 1)
        builder.add_piece(1, 5, 0, 6, "diagonal", 1)
        builder.add_piece(3, 4, 3, 5, "orthogonal", 1)
        # P2: corner pieces + central
        builder.add_piece(0, 1, 0, 0, "orthogonal", 2)
        builder.add_piece(5, 1, 6, 0, "diagonal", 2)
        builder.add_piece(3, 2, 3, 1, "orthogonal", 2)

        return builder.build()

    @staticmethod
    def small_5x5() -> GameState:
        """
        Small 5x5 board for quick games.
        """
        builder = GameSetupBuilder(size=5)

        # One tower in the center
        builder.add_tower(2, 2)

        # Some bunkers
        builder.add_bunker(1, 1)
        builder.add_bunker(3, 3)

        # Pieces closer to center
        builder.add_piece(4, 3, 4, 4, "orthogonal", 1)
        builder.add_piece(1, 3, 0, 4, "diagonal", 1)
        builder.add_piece(0, 1, 0, 0, "orthogonal", 2)
        builder.add_piece(3, 1, 4, 0, "diagonal", 2)

        return builder.build()


class InteractiveSetupBuilder:
    """
    Interactive CLI for building custom setups.
    """

    def __init__(self):
        self.builder = None

    def run(self):
        """
        Run the interactive setup builder.
        """
        print("\n" + "="*60)
        print("DOTS & CUTS - CUSTOM GAME SETUP BUILDER")
        print("="*60)

        # Choose board size
        while True:
            try:
                size = int(input("\nEnter board size (default 9): ") or "9")
                if 3 <= size <= 15:
                    break
                else:
                    print("Board size must be between 3 and 15")
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.builder = GameSetupBuilder(size=size)

        # Menu loop
        while True:
            print("\n[Board Editor]")
            print("1) Add tower")
            print("2) Add bunker")
            print("3) Add lake")
            print("4) Add piece")
            print("5) Remove last piece")
            print("6) View summary")
            print("7) Load preset")
            print("8) Build and play")
            print("9) Exit")

            choice = input("\nChoice (1-9): ").strip()

            if choice == '1':
                self._add_obstacle("tower")
            elif choice == '2':
                self._add_obstacle("bunker")
            elif choice == '3':
                self._add_obstacle("lake")
            elif choice == '4':
                self._add_piece()
            elif choice == '5':
                if self.builder.pieces:
                    self.builder.pieces.pop()
                    print("✓ Removed last piece")
            elif choice == '6':
                self.builder.print_summary()
            elif choice == '7':
                self._load_preset()
            elif choice == '8':
                return self.builder.build()
            elif choice == '9':
                return None

    def _add_obstacle(self, obstacle_type: str):
        """
        Add an obstacle (tower, bunker, or lake).
        """
        try:
            x = int(input(f"Enter X coordinate (0-{self.builder.size-2}): "))
            y = int(input(f"Enter Y coordinate (0-{self.builder.size-2}): "))

            if obstacle_type == "tower":
                self.builder.add_tower(x, y)
                print(f"✓ Added tower at ({x}, {y})")
            elif obstacle_type == "bunker":
                self.builder.add_bunker(x, y)
                print(f"✓ Added bunker at ({x}, {y})")
            elif obstacle_type == "lake":
                self.builder.add_lake(x, y)
                print(f"✓ Added lake at ({x}, {y})")
        except (ValueError, IndexError) as e:
            print(f"✗ Error: {e}")

    def _add_piece(self):
        """
        Add a piece interactively.
        """
        try:
            player = int(input("Player (1 or 2): "))
            kind = input("Kind (orthogonal/diagonal): ").lower()
            pos_x = int(input(f"Position X (0-{self.builder.size-1}): "))
            pos_y = int(input(f"Position Y (0-{self.builder.size-1}): "))
            tail_x = int(input(f"Tail X (0-{self.builder.size-1}): "))
            tail_y = int(input(f"Tail Y (0-{self.builder.size-1}): "))

            self.builder.add_piece(pos_x, pos_y, tail_x, tail_y, kind, player)
            print(f"✓ Added piece for Player {player}")
        except ValueError:
            print("✗ Invalid input")

    def _load_preset(self):
        """
        Load a preset setup.
        """
        print("\nAvailable presets:")
        print("1) Standard 9x9 (random)")
        print("2) Balanced 9x9 (symmetrical)")
        print("3) Skirmish 9x9 (4 pieces each)")
        print("4) Mid 7x7 (3 pieces each)")
        print("5) Small 5x5")

        choice = input("Choose preset (1-5): ").strip()

        if choice == '1':
            self.builder = GameSetupBuilder(size=9)
            from dotscuts import setup_standard_game
            return setup_standard_game()
        elif choice == '2':
            self.builder = GameSetupBuilder(size=9)
            return PrebuiltSetups.balanced_9x9()
        elif choice == '3':
            self.builder = GameSetupBuilder(size=9)
            return PrebuiltSetups.skirmish_9x9()
        elif choice == '4':
            self.builder = GameSetupBuilder(size=7)
            return PrebuiltSetups.mid_7x7()
        elif choice == '5':
            self.builder = GameSetupBuilder(size=5)
            return PrebuiltSetups.small_5x5()


def main():
    """
    Main entry point for the setup builder.
    """
    builder = InteractiveSetupBuilder()
    game_state = builder.run()

    if game_state:
        print("\n✓ Setup complete! Ready to play.")
        # For future: launch game with this setup
        return game_state
    else:
        print("\n✗ Setup cancelled.")
        return None


if __name__ == "__main__":
    game_state = main()
