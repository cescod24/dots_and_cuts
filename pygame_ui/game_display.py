"""
Game Display Module for Dots & Cuts
====================================
Renders the game board with:
- 9x9 grid of vertices (dots/circles)
- Towers (circles inside cells)
- Bunkers (diamonds/rotated squares inside cells)
- Lakes (all edges of a cell marked as visited)
- Pieces (colored squares on vertices)
"""

import pygame
import numpy as np


class GameDisplay:
    """
    Handles all rendering of the board and game state.
    """

    # Color palette
    COLORS = {
        'background': (245, 245, 245),      # Light gray
        'grid_line': (100, 100, 100),       # Dark gray
        'vertex': (50, 50, 50),             # Black dots
        'tower': (100, 150, 255),           # Blue circles
        'bunker': (255, 100, 100),          # Red diamond
        'lake_edge': (150, 200, 255),       # Light blue edges
        'visited_edge': (200, 200, 200),    # Gray visited edges
        'unvisited_edge': (220, 220, 220),  # Light gray unvisited
        'p1_piece': (0, 100, 0),            # Dark green
        'p2_piece': (200, 0, 0),            # Red
        'p1_selected': (0, 255, 0),         # Bright green
        'p2_selected': (255, 100, 100),     # Light red
        'legal_move': (100, 200, 100),      # Light green
        'legal_shoot': (255, 200, 100),     # Orange
    }

    def __init__(self, width=1000, height=1000):
        """
        Initialize the display.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self.width = width
        self.height = height
        self.margin = 60  # Space for margins around board

        # Calculate cell size
        board_width = width - 2 * self.margin
        board_height = height - 2 * self.margin

        # For 9x9 board (9 vertices), we need 8 cells between them
        # Cell size is the distance between vertices
        self.cell_size = min(board_width, board_height) / 8.0

        # Board origin (top-left corner)
        self.board_x = self.margin
        self.board_y = self.margin

        # Initialize pygame
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dots & Cuts - Interactive Game")

        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)

        self.clock = pygame.time.Clock()

    def vertex_to_pixel(self, vertex_x, vertex_y):
        """
        Convert board vertex coordinates (0-8, 0-8) to pixel coordinates.
        """
        pixel_x = self.board_x + vertex_x * self.cell_size
        pixel_y = self.board_y + vertex_y * self.cell_size
        return int(pixel_x), int(pixel_y)

    def pixel_to_vertex(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates back to board vertex coordinates.
        Returns None if outside board.
        """
        vx = (pixel_x - self.board_x) / self.cell_size
        vy = (pixel_y - self.board_y) / self.cell_size

        # Snap to nearest vertex if close enough
        vx_snapped = round(vx)
        vy_snapped = round(vy)

        # Check if within board bounds (0-8 for 9x9)
        if 0 <= vx_snapped <= 8 and 0 <= vy_snapped <= 8:
            # Check if actually close to a vertex (within half cell size)
            if abs(vx - vx_snapped) < 0.3 and abs(vy - vy_snapped) < 0.3:
                return int(vx_snapped), int(vy_snapped)

        return None

    def draw_board(self, game_state):
        """
        Draw the complete board state.
        """
        # Clear screen
        self.screen.fill(self.COLORS['background'])

        # Draw grid and cells
        self.draw_grid(game_state)

        # Draw visited edges
        self.draw_visited_edges(game_state)

        # Draw towers and bunkers
        self.draw_obstacles(game_state)

        # Draw pieces
        self.draw_pieces(game_state)

    def draw_grid(self, game_state):
        """
        Draw the 9x9 vertex grid (dots at each vertex).
        """
        vertex_radius = 5

        # Draw vertices (the dots)
        for vy in range(game_state.board.size):
            for vx in range(game_state.board.size):
                px, py = self.vertex_to_pixel(vx, vy)
                pygame.draw.circle(self.screen, self.COLORS['vertex'], (px, py), vertex_radius)

    def draw_visited_edges(self, game_state):
        """
        Draw visited and unvisited edges between vertices.
        - Gray for visited edges
        - Light gray for unvisited edges
        - Light blue for edges adjacent to lakes
        """
        edge_width = 3

        # Pre-compute which edges are lake edges
        lake_edges = set()
        for y in range(game_state.board.size - 1):
            for x in range(game_state.board.size - 1):
                if game_state.board.lakes[y][x]:
                    # All 6 edges of this cell are lake edges
                    v_tl = (x, y)
                    v_tr = (x + 1, y)
                    v_bl = (x, y + 1)
                    v_br = (x + 1, y + 1)
                    edges = [
                        tuple(sorted([v_tl, v_tr])),
                        tuple(sorted([v_tr, v_br])),
                        tuple(sorted([v_br, v_bl])),
                        tuple(sorted([v_bl, v_tl])),
                        tuple(sorted([v_tl, v_br])),
                        tuple(sorted([v_tr, v_bl]))
                    ]
                    lake_edges.update(edges)

        # Draw all possible edges
        for vy in range(game_state.board.size):
            for vx in range(game_state.board.size):
                # Orthogonal neighbors
                for next_vx, next_vy in [(vx+1, vy), (vx, vy+1)]:
                    if next_vx < game_state.board.size and next_vy < game_state.board.size:
                        edge = tuple(sorted([(vx, vy), (next_vx, next_vy)]))

                        px1, py1 = self.vertex_to_pixel(vx, vy)
                        px2, py2 = self.vertex_to_pixel(next_vx, next_vy)

                        # Determine color
                        if edge in lake_edges:
                            color = self.COLORS['lake_edge']
                        elif edge in game_state.visited_edges:
                            color = self.COLORS['visited_edge']
                        else:
                            color = self.COLORS['unvisited_edge']

                        pygame.draw.line(self.screen, color, (px1, py1), (px2, py2), edge_width)

                # Diagonal neighbors
                for next_vx, next_vy in [(vx+1, vy+1), (vx+1, vy-1)]:
                    if 0 <= next_vx < game_state.board.size and 0 <= next_vy < game_state.board.size:
                        # Only draw the first diagonal to avoid duplicates
                        if next_vx > vx:
                            edge = tuple(sorted([(vx, vy), (next_vx, next_vy)]))

                            px1, py1 = self.vertex_to_pixel(vx, vy)
                            px2, py2 = self.vertex_to_pixel(next_vx, next_vy)

                            # Determine color
                            if edge in lake_edges:
                                color = self.COLORS['lake_edge']
                            elif edge in game_state.visited_edges:
                                color = self.COLORS['visited_edge']
                            else:
                                color = self.COLORS['unvisited_edge']

                            pygame.draw.line(self.screen, color, (px1, py1), (px2, py2), edge_width)

    def draw_obstacles(self, game_state):
        """
        Draw towers (blue circles) and bunkers (red diamonds) in cells.
        """
        tower_radius = int(self.cell_size * 0.25)

        for cy in range(game_state.board.size - 1):
            for cx in range(game_state.board.size - 1):
                # Cell center
                cell_cx = self.board_x + (cx + 0.5) * self.cell_size
                cell_cy = self.board_y + (cy + 0.5) * self.cell_size

                if game_state.board.towers[cy][cx]:
                    # Draw tower (blue circle)
                    pygame.draw.circle(
                        self.screen,
                        self.COLORS['tower'],
                        (int(cell_cx), int(cell_cy)),
                        tower_radius
                    )

                if game_state.board.bunkers[cy][cx]:
                    # Draw bunker (red diamond)
                    self.draw_diamond(
                        int(cell_cx), int(cell_cy),
                        int(tower_radius * 1.2),
                        self.COLORS['bunker']
                    )

    def draw_diamond(self, center_x, center_y, size, color):
        """
        Draw a diamond (rotated square) shape.
        """
        points = [
            (center_x, center_y - size),       # Top
            (center_x + size, center_y),       # Right
            (center_x, center_y + size),       # Bottom
            (center_x - size, center_y)        # Left
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)  # Border

    def draw_pieces(self, game_state, selected_piece=None, legal_moves=None, legal_shoots=None):
        """
        Draw all pieces on the board.

        Args:
            game_state: Current game state
            selected_piece: Currently selected piece (highlighted)
            legal_moves: Set of legal move targets
            legal_shoots: Set of legal shoot targets
        """
        piece_size = int(self.cell_size * 0.3)

        # Draw legal move/shoot targets
        if legal_moves:
            for vx, vy in legal_moves:
                px, py = self.vertex_to_pixel(vx, vy)
                pygame.draw.circle(self.screen, self.COLORS['legal_move'], (px, py), piece_size + 3, 2)

        if legal_shoots:
            for vx, vy in legal_shoots:
                px, py = self.vertex_to_pixel(vx, vy)
                pygame.draw.circle(self.screen, self.COLORS['legal_shoot'], (px, py), piece_size + 3, 2)

        # Draw pieces
        for piece in game_state.pieces:
            px, py = self.vertex_to_pixel(piece.x, piece.y)

            # Determine color based on player
            if piece.player == 1:
                color = self.COLORS['p1_piece']
                if piece == selected_piece:
                    color = self.COLORS['p1_selected']
            else:
                color = self.COLORS['p2_piece']
                if piece == selected_piece:
                    color = self.COLORS['p2_selected']

            # Draw piece as square
            rect = pygame.Rect(px - piece_size, py - piece_size, piece_size * 2, piece_size * 2)
            pygame.draw.rect(self.screen, color, rect)

            # Draw piece type (O for orthogonal, D for diagonal)
            type_char = "O" if piece.kind == "orthogonal" else "D"
            type_text = self.font_small.render(type_char, True, (255, 255, 255))
            self.screen.blit(type_text, (px - 5, py - 5))

    def draw_ui_bottom(self, current_player, game_info=""):
        """
        Draw UI information at the bottom of the screen.
        """
        ui_y = self.height - 40

        # Current player indicator
        player_color = self.COLORS['p1_piece'] if current_player == 1 else self.COLORS['p2_piece']
        player_text = f"Player {current_player}'s Turn"
        text_surface = self.font_medium.render(player_text, True, player_color)
        self.screen.blit(text_surface, (20, ui_y))

        # Game info
        if game_info:
            info_surface = self.font_small.render(game_info, True, (50, 50, 50))
            self.screen.blit(info_surface, (300, ui_y + 5))

    def draw_bot_thinking(self, top_3_moves, position=(20, 20)):
        """
        Draw the bot's top 3 candidate moves and evaluations.

        Args:
            top_3_moves: List of tuples (action_text, q_value, is_best)
            position: (x, y) position to draw at
        """
        x, y = position
        line_height = 25

        # Title
        title_surface = self.font_medium.render("Bot's Top Moves:", True, (0, 0, 0))
        self.screen.blit(title_surface, (x, y))
        y += line_height + 5

        for i, (action_text, q_value, is_best) in enumerate(top_3_moves[:3]):
            # Format: "1. Move (7,5) -> (6,5)  Q=0.85"
            rank = i + 1
            color = (255, 100, 0) if is_best else (50, 50, 50)

            move_text = f"{rank}. {action_text:30s} Q={q_value:6.3f}"
            if is_best:
                move_text += "  ← BEST"

            text_surface = self.font_small.render(move_text, True, color)
            self.screen.blit(text_surface, (x, y))
            y += line_height

    def update(self):
        """
        Update the display (flip buffers, handle timing).
        """
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def quit(self):
        """
        Clean up and close the display.
        """
        pygame.quit()
