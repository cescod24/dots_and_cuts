"""
Game Display Module for Dots & Cuts
====================================
Renders the board and UI elements using PyGame.

Supports any board size (5x5 to 15x15) by computing layout dynamically.
Left area: board.  Right panel: info + bot thinking.

Toggleable overlays (controlled by the caller):
  - show_grid:    draw unvisited edges as faint lines  (default OFF)
  - show_z_hints: colour vertices by z-value            (default OFF)
"""

import pygame


class GameDisplay:
    """Handles all rendering of the board, pieces, and UI."""

    COLORS = {
        "bg":              (24,  24,  28),
        "panel_bg":        (32,  32,  38),
        "grid_dot":        (160, 160, 160),
        "grid_dot_tower":  (90,  140, 240),
        "grid_dot_bunker": (230, 90,  90),
        "tower":           (90,  140, 240),
        "bunker":          (230, 90,  90),
        "lake_fill":       (50,  85, 120),
        "visited_edge":    (190, 190, 190),
        "unvisited_edge":  (48,  48,  54),
        "lake_edge":       (70,  110, 155),
        "p1":              (50,  180, 80),
        "p1_sel":          (100, 255, 120),
        "p2":              (210, 60,  60),
        "p2_sel":          (255, 120, 120),
        "legal_move":      (80,  200, 120),
        "legal_shoot":     (240, 180, 60),
        "text":            (220, 220, 220),
        "text_dim":        (120, 120, 130),
        "accent":          (100, 160, 255),
        "divider":         (55,  55,  62),
    }

    PANEL_WIDTH = 280

    def __init__(self, board_size: int = 9, width: int = 1080, height: int = 760):
        self.board_size = board_size
        self.width = width
        self.height = height

        board_area = min(width - self.PANEL_WIDTH - 40, height - 80)
        margin = 50
        usable = board_area - 2 * margin
        self.cell_size = usable / (board_size - 1)
        self.board_x = margin
        self.board_y = margin

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dots & Cuts")

        self.font_sm  = pygame.font.SysFont("Arial", 13)
        self.font_md  = pygame.font.SysFont("Arial", 16)
        self.font_lg  = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_xl  = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_tiny = pygame.font.SysFont("Arial", 11)

        self.clock = pygame.time.Clock()

    # ---- coordinate conversion ----

    def vertex_to_pixel(self, vx, vy):
        return (int(self.board_x + vx * self.cell_size),
                int(self.board_y + vy * self.cell_size))

    def pixel_to_vertex(self, px, py):
        vx = (px - self.board_x) / self.cell_size
        vy = (py - self.board_y) / self.cell_size
        rx, ry = round(vx), round(vy)
        if 0 <= rx < self.board_size and 0 <= ry < self.board_size:
            if abs(vx - rx) < 0.35 and abs(vy - ry) < 0.35:
                return rx, ry
        return None

    # ---- full frame ----

    def draw_frame(self, game_state, current_player, *,
                   selected_piece=None, legal_moves=None, legal_shoots=None,
                   bot_top_moves=None, bot_label="", message="",
                   game_over=False, winner=None,
                   show_grid=False, show_z_hints=False):
        """Draw one complete frame. Toggles passed from GameUI."""
        self.screen.fill(self.COLORS["bg"])

        self._draw_edges(game_state, show_grid)
        self._draw_obstacles(game_state)
        self._draw_vertices(game_state, show_z_hints)
        self._draw_highlights(legal_moves, legal_shoots)
        self._draw_pieces(game_state, selected_piece)
        self._draw_panel(current_player, bot_top_moves, bot_label,
                         message, game_over, winner, game_state,
                         show_grid, show_z_hints)

        pygame.display.flip()
        self.clock.tick(60)

    # ---- board elements ----

    def _draw_vertices(self, gs, show_z):
        r = max(3, int(self.cell_size * 0.055))
        for vy in range(gs.board.size):
            for vx in range(gs.board.size):
                px, py = self.vertex_to_pixel(vx, vy)
                if show_z:
                    z = gs.board.z[vy][vx]
                    if z == 1:
                        color = self.COLORS["grid_dot_tower"]
                    elif z == -1:
                        color = self.COLORS["grid_dot_bunker"]
                    else:
                        color = self.COLORS["grid_dot"]
                else:
                    color = self.COLORS["grid_dot"]
                pygame.draw.circle(self.screen, color, (px, py), r)

    def _draw_edges(self, gs, show_grid):
        w = max(1, int(self.cell_size * 0.025))

        lake_edges = set()
        for cy in range(gs.board.size - 1):
            for cx in range(gs.board.size - 1):
                if gs.board.lakes[cy][cx]:
                    tl, tr = (cx, cy), (cx+1, cy)
                    bl, br = (cx, cy+1), (cx+1, cy+1)
                    for e in [(tl,tr),(tr,br),(br,bl),(bl,tl),(tl,br),(tr,bl)]:
                        lake_edges.add(tuple(sorted(e)))

        drawn = set()
        for vy in range(gs.board.size):
            for vx in range(gs.board.size):
                for nx, ny in [(vx+1,vy),(vx,vy+1),(vx+1,vy+1),(vx-1,vy+1)]:
                    if 0 <= nx < gs.board.size and 0 <= ny < gs.board.size:
                        edge = tuple(sorted([(vx,vy),(nx,ny)]))
                        if edge in drawn:
                            continue
                        drawn.add(edge)

                        is_lake = edge in lake_edges
                        is_visited = edge in gs.visited_edges

                        # Decide whether to draw
                        if is_lake:
                            color = self.COLORS["lake_edge"]
                        elif is_visited:
                            color = self.COLORS["visited_edge"]
                        elif show_grid:
                            color = self.COLORS["unvisited_edge"]
                        else:
                            continue  # skip unvisited edges when grid is off

                        p1 = self.vertex_to_pixel(edge[0][0], edge[0][1])
                        p2 = self.vertex_to_pixel(edge[1][0], edge[1][1])
                        pygame.draw.line(self.screen, color, p1, p2, w)

    def _draw_obstacles(self, gs):
        r = max(5, int(self.cell_size * 0.20))
        for cy in range(gs.board.size - 1):
            for cx in range(gs.board.size - 1):
                ccx = int(self.board_x + (cx + 0.5) * self.cell_size)
                ccy = int(self.board_y + (cy + 0.5) * self.cell_size)

                if gs.board.lakes[cy][cx]:
                    s = int(r * 1.0)
                    rect = pygame.Rect(ccx - s, ccy - s, s*2, s*2)
                    pygame.draw.rect(self.screen, self.COLORS["lake_fill"], rect, border_radius=3)

                if gs.board.towers[cy][cx]:
                    pygame.draw.circle(self.screen, self.COLORS["tower"], (ccx, ccy), r)
                    pygame.draw.circle(self.screen, (200,200,200), (ccx, ccy), r, 1)

                if gs.board.bunkers[cy][cx]:
                    pts = [(ccx, ccy-r), (ccx+r, ccy), (ccx, ccy+r), (ccx-r, ccy)]
                    pygame.draw.polygon(self.screen, self.COLORS["bunker"], pts)
                    pygame.draw.polygon(self.screen, (200,200,200), pts, 1)

    def _draw_highlights(self, legal_moves, legal_shoots):
        r = max(7, int(self.cell_size * 0.16))
        if legal_moves:
            for vx, vy in legal_moves:
                px, py = self.vertex_to_pixel(vx, vy)
                pygame.draw.circle(self.screen, self.COLORS["legal_move"], (px, py), r, 2)
        if legal_shoots:
            for vx, vy in legal_shoots:
                px, py = self.vertex_to_pixel(vx, vy)
                pygame.draw.circle(self.screen, self.COLORS["legal_shoot"], (px, py), r, 2)

    # ---- pieces (smaller) ----

    def _draw_pieces(self, gs, selected_piece):
        sz = max(6, int(self.cell_size * 0.16))  # smaller than before (was 0.25)

        by_vertex = {}
        for p in gs.pieces:
            by_vertex.setdefault((p.x, p.y), []).append(p)

        for (vx, vy), pieces in by_vertex.items():
            px, py = self.vertex_to_pixel(vx, vy)
            if len(pieces) == 1:
                self._piece(pieces[0], px, py, sz, pieces[0] is selected_piece)
            else:
                off = int(sz * 0.8)
                for i, p in enumerate(pieces):
                    dx = (i % 2) * off - off // 2
                    dy = (i // 2) * off - off // 2
                    self._piece(p, px+dx, py+dy, sz, p is selected_piece)

    def _piece(self, piece, px, py, sz, sel):
        c = {
            (1, False): self.COLORS["p1"],
            (1, True):  self.COLORS["p1_sel"],
            (2, False): self.COLORS["p2"],
            (2, True):  self.COLORS["p2_sel"],
        }[(piece.player, sel)]

        rect = pygame.Rect(px - sz, py - sz, sz*2, sz*2)
        pygame.draw.rect(self.screen, c, rect, border_radius=3)
        if sel:
            pygame.draw.rect(self.screen, (255,255,255), rect, 2, border_radius=3)

        label = "O" if piece.kind == "orthogonal" else "D"
        txt = self.font_tiny.render(label, True, (255,255,255))
        self.screen.blit(txt, txt.get_rect(center=(px, py)))

    # ---- right panel ----

    def _draw_panel(self, cur_player, bot_top, bot_label,
                    msg, game_over, winner, gs, show_grid, show_z):
        px = self.width - self.PANEL_WIDTH
        pygame.draw.rect(self.screen, self.COLORS["panel_bg"],
                         (px, 0, self.PANEL_WIDTH, self.height))
        pygame.draw.line(self.screen, self.COLORS["divider"],
                         (px, 0), (px, self.height), 1)

        x, y = px + 14, 16

        # Title
        self.screen.blit(self.font_xl.render("Dots & Cuts", True, self.COLORS["text"]), (x, y))
        y += 38
        pygame.draw.line(self.screen, self.COLORS["divider"], (x, y), (x + self.PANEL_WIDTH - 28, y))
        y += 12

        # Player turn / game over
        if game_over:
            t = self.font_lg.render(f"Player {winner} wins!", True, self.COLORS["accent"])
        else:
            pc = self.COLORS["p1"] if cur_player == 1 else self.COLORS["p2"]
            t = self.font_lg.render(f"Player {cur_player}'s turn", True, pc)
        self.screen.blit(t, (x, y)); y += 30

        # Piece counts
        p1n = sum(1 for p in gs.pieces if p.player == 1)
        p2n = sum(1 for p in gs.pieces if p.player == 2)
        self.screen.blit(self.font_md.render(
            f"P1: {p1n} pieces   P2: {p2n} pieces", True, self.COLORS["text_dim"]), (x, y))
        y += 24

        if bot_label:
            self.screen.blit(self.font_md.render(
                f"Bot: {bot_label}", True, self.COLORS["text_dim"]), (x, y))
            y += 22

        y += 4
        pygame.draw.line(self.screen, self.COLORS["divider"], (x, y), (x + self.PANEL_WIDTH - 28, y))
        y += 10

        # Bot top moves
        if bot_top:
            self.screen.blit(self.font_lg.render("Bot's Top Moves", True, self.COLORS["accent"]), (x, y))
            y += 26
            for i, (astr, sc, best) in enumerate(bot_top):
                col = self.COLORS["accent"] if best else self.COLORS["text"]
                pf = ">" if best else " "
                ln = f"{pf} {i+1}. {astr:24s} {sc:+.3f}"
                self.screen.blit(self.font_md.render(ln, True, col), (x, y))
                y += 20
            y += 8

        # Toggle status indicators
        y2 = self.height - 180
        pygame.draw.line(self.screen, self.COLORS["divider"], (x, y2), (x + self.PANEL_WIDTH - 28, y2))
        y2 += 10

        # Show current toggle states
        grid_st = "ON" if show_grid else "OFF"
        z_st    = "ON" if show_z    else "OFF"
        self.screen.blit(self.font_md.render("Overlays", True, self.COLORS["text"]), (x, y2))
        y2 += 20
        self.screen.blit(self.font_sm.render(f"  G = Grid edges: {grid_st}", True, self.COLORS["text_dim"]), (x, y2))
        y2 += 16
        self.screen.blit(self.font_sm.render(f"  Z = Z-value hints: {z_st}", True, self.COLORS["text_dim"]), (x, y2))
        y2 += 22

        # Controls
        self.screen.blit(self.font_md.render("Controls", True, self.COLORS["text"]), (x, y2))
        y2 += 20
        for line in ["Click to select / act",
                      "U = Undo   R = Restart",
                      "B = Bot thinking   Q = Menu"]:
            self.screen.blit(self.font_sm.render(line, True, self.COLORS["text_dim"]), (x, y2))
            y2 += 16

        # Message
        if msg:
            self.screen.blit(self.font_md.render(msg, True, self.COLORS["accent"]),
                             (x, self.height - 24))

    def quit(self):
        pass
