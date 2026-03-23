"""
Board Builder — Visual editor for custom Dots & Cuts boards.

Allows the player to:
  - Choose board size (4-13)
  - Place / remove towers, bunkers, lakes on cells
  - Place pieces (orthogonal / diagonal) with position + tail
  - Start the game with the custom setup

Piece placement is two clicks: first the position vertex, then the tail vertex.
"""

import pygame
import sys
import os

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "core"))
sys.path.insert(0, os.path.join(_base, ".."))

from dotscuts import Board, GameState
from custom_setup import GameSetupBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 1100, 760
PANEL_W = 280

TOOLS = [
    ("tower",       "Tower",       (90,  140, 240)),
    ("bunker",      "Bunker",      (230, 90,  90)),
    ("lake",        "Lake",        (50,  120, 180)),
    ("eraser",      "Eraser",      (160, 160, 160)),
    ("p1_orth",     "P1 Orthogonal", (50, 180, 80)),
    ("p1_diag",     "P1 Diagonal", (50,  180, 80)),
    ("p2_orth",     "P2 Orthogonal", (210, 60, 60)),
    ("p2_diag",     "P2 Diagonal", (210, 60,  60)),
]

C = {
    "bg":         (24,  24,  28),
    "panel":      (32,  32,  38),
    "grid_dot":   (140, 140, 140),
    "cell_empty": (38,  38,  44),
    "cell_hover": (55,  55,  65),
    "tower":      (90,  140, 240),
    "bunker":     (230, 90,  90),
    "lake":       (50,  120, 180),
    "text":       (220, 220, 220),
    "text_dim":   (120, 120, 130),
    "accent":     (90,  145, 255),
    "btn":        (50,  50,  60),
    "btn_hover":  (65,  80, 120),
    "divider":    (55,  55,  62),
    "p1":         (50,  180, 80),
    "p2":         (210, 60,  60),
    "edge":       (180, 180, 180),
    "warn":       (220, 90,  90),
    "pending":    (255, 220, 60),
}


class BoardBuilderScreen:
    """Visual board editor. Returns a GameState or '__back__' / '__quit__'."""

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock

        # Resize window for builder
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

        # Fonts
        self.f_title = pygame.font.SysFont("Arial", 28, bold=True)
        self.f_btn   = pygame.font.SysFont("Arial", 16, bold=True)
        self.f_sm    = pygame.font.SysFont("Arial", 13)
        self.f_md    = pygame.font.SysFont("Arial", 15)
        self.f_tiny  = pygame.font.SysFont("Arial", 11)

        # State
        self.board_size = 9
        self.builder = GameSetupBuilder(size=self.board_size)
        self.tool = "tower"
        self.message = ""
        self.msg_timer = 0

        # Piece placement state: waiting for tail after position click
        self.pending_piece = None  # {kind, player, pos_x, pos_y}

        # Random obstacle counts
        self.rand_towers = 5
        self.rand_bunkers = 10
        self.rand_lakes = 0

        self._recompute_layout()

    # ----- layout -----

    def _recompute_layout(self):
        board_area = SCREEN_W - PANEL_W - 60
        margin = 45
        usable = min(board_area, SCREEN_H - 80) - 2 * margin
        self.cell_size = usable / (self.board_size - 1)
        self.board_x = margin
        self.board_y = margin + 30

    def _vertex_px(self, vx, vy):
        return (int(self.board_x + vx * self.cell_size),
                int(self.board_y + vy * self.cell_size))

    def _cell_center_px(self, cx, cy):
        return (int(self.board_x + (cx + 0.5) * self.cell_size),
                int(self.board_y + (cy + 0.5) * self.cell_size))

    def _px_to_vertex(self, px, py):
        vx = (px - self.board_x) / self.cell_size
        vy = (py - self.board_y) / self.cell_size
        rx, ry = round(vx), round(vy)
        if 0 <= rx < self.board_size and 0 <= ry < self.board_size:
            if abs(vx - rx) < 0.35 and abs(vy - ry) < 0.35:
                return rx, ry
        return None

    def _px_to_cell(self, px, py):
        cx = (px - self.board_x) / self.cell_size - 0.5
        cy = (py - self.board_y) / self.cell_size - 0.5
        rx, ry = round(cx), round(cy)
        if 0 <= rx < self.board_size - 1 and 0 <= ry < self.board_size - 1:
            return rx, ry
        return None

    # ----- messages -----

    def _msg(self, text, frames=120):
        self.message = text
        self.msg_timer = frames

    # ----- board size -----

    def _set_size(self, new_size):
        new_size = max(4, min(13, new_size))
        if new_size != self.board_size:
            self.board_size = new_size
            self.builder = GameSetupBuilder(size=new_size)
            self.pending_piece = None
            self._recompute_layout()
            self._msg(f"Board size: {new_size}x{new_size}")

    # ----- structure placement -----

    def _place_structure(self, cx, cy):
        b = self.builder.board
        if self.tool == "tower":
            if b.towers[cy][cx]:
                b.towers[cy][cx] = False
                b.recompute_z()
                self._msg(f"Removed tower at ({cx},{cy})")
            else:
                b.bunkers[cy][cx] = False
                b.lakes[cy][cx] = False
                b.place_tower(cx, cy)
                self._msg(f"Placed tower at ({cx},{cy})")
        elif self.tool == "bunker":
            if b.bunkers[cy][cx]:
                b.bunkers[cy][cx] = False
                b.recompute_z()
                self._msg(f"Removed bunker at ({cx},{cy})")
            else:
                b.towers[cy][cx] = False
                b.lakes[cy][cx] = False
                b.place_bunker(cx, cy)
                self._msg(f"Placed bunker at ({cx},{cy})")
        elif self.tool == "lake":
            if b.lakes[cy][cx]:
                b.lakes[cy][cx] = False
                self._msg(f"Removed lake at ({cx},{cy})")
            else:
                b.towers[cy][cx] = False
                b.bunkers[cy][cx] = False
                b.place_lake(cx, cy)
                self._msg(f"Placed lake at ({cx},{cy})")
        elif self.tool == "eraser":
            b.towers[cy][cx] = False
            b.bunkers[cy][cx] = False
            b.lakes[cy][cx] = False
            b.recompute_z()
            self._msg(f"Cleared cell ({cx},{cy})")

    # ----- piece placement (two-click) -----

    def _start_piece(self, vx, vy):
        tool_map = {
            "p1_orth": ("orthogonal", 1),
            "p1_diag": ("diagonal",   1),
            "p2_orth": ("orthogonal", 2),
            "p2_diag": ("diagonal",   2),
        }
        kind, player = tool_map[self.tool]
        self.pending_piece = {"kind": kind, "player": player, "pos_x": vx, "pos_y": vy}
        self._msg(f"Position ({vx},{vy}) — now click tail vertex")

    def _finish_piece(self, tx, ty):
        p = self.pending_piece
        px, py = p["pos_x"], p["pos_y"]
        dx, dy = px - tx, py - ty

        # Validate tail → position direction
        valid = False
        if p["kind"] == "orthogonal":
            valid = (abs(dx) == 1 and dy == 0) or (dx == 0 and abs(dy) == 1)
        elif p["kind"] == "diagonal":
            valid = abs(dx) == 1 and abs(dy) == 1

        if not valid:
            self._msg(f"Invalid tail for {p['kind']} piece — must be 1 step away")
            self.pending_piece = None
            return

        self.builder.add_piece(px, py, tx, ty, p["kind"], p["player"])
        self._msg(f"Placed P{p['player']} {p['kind']} at ({px},{py}) tail ({tx},{ty})")
        self.pending_piece = None

    def _handle_vertex_click(self, vx, vy):
        if self.pending_piece:
            self._finish_piece(vx, vy)
        else:
            # Check if eraser should remove a piece at this vertex
            if self.tool == "eraser":
                removed = False
                for i in range(len(self.builder.pieces) - 1, -1, -1):
                    pc = self.builder.pieces[i]
                    if pc["position"] == (vx, vy):
                        self.builder.pieces.pop(i)
                        removed = True
                        break
                if removed:
                    self._msg(f"Removed piece at ({vx},{vy})")
                return
            if self.tool in ("p1_orth", "p1_diag", "p2_orth", "p2_diag"):
                self._start_piece(vx, vy)

    # ----- randomize obstacles -----

    def _randomize_obstacles(self):
        import random
        b = self.builder.board
        sz = self.board_size
        # Clear existing obstacles
        for cy in range(sz - 1):
            for cx in range(sz - 1):
                b.towers[cy][cx] = False
                b.bunkers[cy][cx] = False
                b.lakes[cy][cx] = False
        b.recompute_z()

        cell_coords = [(x, y) for x in range(sz - 1) for y in range(sz - 1)]
        corners = {(0, 0), (0, sz - 2), (sz - 2, 0), (sz - 2, sz - 2)}
        total_cells = len(cell_coords)

        n_towers = min(self.rand_towers, total_cells)
        n_lakes = min(self.rand_lakes, total_cells - n_towers)
        n_bunkers = min(self.rand_bunkers, total_cells - n_towers - n_lakes)

        possible_lake = [p for p in cell_coords if p not in corners]
        lake_pos = set(random.sample(possible_lake, min(n_lakes, len(possible_lake))))
        remaining = [p for p in cell_coords if p not in lake_pos]
        tower_pos = set(random.sample(remaining, min(n_towers, len(remaining))))
        remaining = [p for p in remaining if p not in tower_pos]
        bunker_pos = set(random.sample(remaining, min(n_bunkers, len(remaining))))

        for x, y in tower_pos:
            b.place_tower(x, y)
        for x, y in bunker_pos:
            b.place_bunker(x, y)
        for x, y in lake_pos:
            b.place_lake(x, y)

        self._msg(f"Randomized: {len(tower_pos)}T {len(bunker_pos)}B {len(lake_pos)}L")

    # ----- click dispatch -----

    def _handle_click(self, pos):
        px, py = pos

        # Right panel clicks handled separately
        if px > SCREEN_W - PANEL_W:
            return

        # Try vertex first (piece tools + eraser on vertices)
        v = self._px_to_vertex(px, py)
        if v and self.tool in ("p1_orth", "p1_diag", "p2_orth", "p2_diag", "eraser"):
            self._handle_vertex_click(v[0], v[1])
            return

        # Cancel pending piece if clicking elsewhere
        if self.pending_piece:
            self.pending_piece = None
            self._msg("Piece placement cancelled")
            return

        # Try cell (structure tools)
        cell = self._px_to_cell(px, py)
        if cell and self.tool in ("tower", "bunker", "lake", "eraser"):
            self._place_structure(cell[0], cell[1])

    # ----- validation -----

    def _validate(self):
        pieces = self.builder.pieces
        p1 = [p for p in pieces if p["player"] == 1]
        p2 = [p for p in pieces if p["player"] == 2]
        if not p1:
            self._msg("Need at least 1 piece for Player 1")
            return False
        if not p2:
            self._msg("Need at least 1 piece for Player 2")
            return False
        return True

    # ----- drawing -----

    def _draw_board(self):
        sz = self.board_size
        b = self.builder.board

        # Draw cells
        for cy in range(sz - 1):
            for cx in range(sz - 1):
                ccx, ccy = self._cell_center_px(cx, cy)
                r = max(5, int(self.cell_size * 0.20))

                if b.lakes[cy][cx]:
                    s = int(r * 1.0)
                    rect = pygame.Rect(ccx - s, ccy - s, s * 2, s * 2)
                    pygame.draw.rect(self.screen, C["lake"], rect, border_radius=3)

                if b.towers[cy][cx]:
                    pygame.draw.circle(self.screen, C["tower"], (ccx, ccy), r)
                    pygame.draw.circle(self.screen, (200, 200, 200), (ccx, ccy), r, 1)

                if b.bunkers[cy][cx]:
                    pts = [(ccx, ccy - r), (ccx + r, ccy), (ccx, ccy + r), (ccx - r, ccy)]
                    pygame.draw.polygon(self.screen, C["bunker"], pts)
                    pygame.draw.polygon(self.screen, (200, 200, 200), pts, 1)

        # Draw grid vertices
        vr = max(3, int(self.cell_size * 0.05))
        for vy in range(sz):
            for vx in range(sz):
                px, py = self._vertex_px(vx, vy)
                # Color by z-value
                z = b.z[vy][vx]
                if z == 1:
                    col = (90, 140, 240)
                elif z == -1:
                    col = (230, 90, 90)
                else:
                    col = C["grid_dot"]
                pygame.draw.circle(self.screen, col, (px, py), vr)

        # Draw placed pieces
        psz = max(6, int(self.cell_size * 0.16))
        for pc in self.builder.pieces:
            px, py = self._vertex_px(*pc["position"])
            col = C["p1"] if pc["player"] == 1 else C["p2"]
            rect = pygame.Rect(px - psz, py - psz, psz * 2, psz * 2)
            pygame.draw.rect(self.screen, col, rect, border_radius=3)
            label = "+" if pc["kind"] == "orthogonal" else "x"
            txt = self.f_tiny.render(label, True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=(px, py)))

            # Draw tail edge
            tx, ty = self._vertex_px(*pc["tail"])
            pygame.draw.line(self.screen, C["edge"], (px, py), (tx, ty), 2)

        # Highlight pending piece position
        if self.pending_piece:
            px, py = self._vertex_px(self.pending_piece["pos_x"], self.pending_piece["pos_y"])
            pygame.draw.circle(self.screen, C["pending"], (px, py), psz + 4, 3)

    def _draw_panel(self):
        panel_x = SCREEN_W - PANEL_W
        pygame.draw.rect(self.screen, C["panel"], (panel_x, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(self.screen, C["divider"], (panel_x, 0), (panel_x, SCREEN_H), 1)

        x, y = panel_x + 14, 16

        # Title
        self.screen.blit(self.f_title.render("Board Builder", True, C["text"]), (x, y))
        y += 36

        # Board size
        pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
        y += 10
        self.screen.blit(self.f_btn.render(f"Board Size: {self.board_size}x{self.board_size}", True, C["text"]), (x, y))
        y += 24

        # Size buttons
        self._size_minus_rect = pygame.Rect(x, y, 60, 28)
        self._size_plus_rect  = pygame.Rect(x + 70, y, 60, 28)
        mouse = pygame.mouse.get_pos()
        for rect, label in [(self._size_minus_rect, "- Size"), (self._size_plus_rect, "+ Size")]:
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, C["btn_hover"] if hover else C["btn"], rect, border_radius=6)
            t = self.f_sm.render(label, True, C["text"])
            self.screen.blit(t, t.get_rect(center=rect.center))
        y += 40

        # Tool buttons
        pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
        y += 8
        self.screen.blit(self.f_md.render("Structures", True, C["text_dim"]), (x, y))
        y += 20

        self._tool_rects = {}
        for tool_id, label, color in TOOLS:
            if tool_id == "p1_orth":
                y += 6
                pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
                y += 8
                self.screen.blit(self.f_md.render("Pieces (click pos, then tail)", True, C["text_dim"]), (x, y))
                y += 20

            rect = pygame.Rect(x, y, PANEL_W - 28, 26)
            self._tool_rects[tool_id] = rect

            hover = rect.collidepoint(mouse)
            selected = self.tool == tool_id
            if selected:
                bg = (color[0] // 3, color[1] // 3, color[2] // 3)
                border_col = color
            elif hover:
                bg = C["btn_hover"]
                border_col = C["divider"]
            else:
                bg = C["btn"]
                border_col = C["divider"]

            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            pygame.draw.rect(self.screen, border_col, rect, 1, border_radius=6)

            # Color indicator
            pygame.draw.circle(self.screen, color, (x + 14, rect.centery), 5)
            t = self.f_sm.render(label, True, C["text"] if not selected else color)
            self.screen.blit(t, (x + 26, rect.centery - t.get_height() // 2))
            y += 30

        # Piece count summary
        y += 6
        pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
        y += 8
        p1_count = sum(1 for p in self.builder.pieces if p["player"] == 1)
        p2_count = sum(1 for p in self.builder.pieces if p["player"] == 2)
        towers = sum(1 for row in self.builder.board.towers for c in row if c)
        bunkers = sum(1 for row in self.builder.board.bunkers for c in row if c)
        lakes = sum(1 for row in self.builder.board.lakes for c in row if c)

        for label, val in [("Towers", towers), ("Bunkers", bunkers), ("Lakes", lakes),
                           ("P1 pieces", p1_count), ("P2 pieces", p2_count)]:
            self.screen.blit(self.f_sm.render(f"{label}: {val}", True, C["text_dim"]), (x, y))
            y += 16

        # Random obstacle controls
        y += 10
        pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
        y += 8
        self.screen.blit(self.f_md.render("Random Obstacles", True, C["text_dim"]), (x, y))
        y += 20

        self._rand_rects = {}
        max_cells = max(1, (self.board_size - 1) ** 2)
        for key, label, val in [("towers", "Towers", self.rand_towers),
                                 ("bunkers", "Bunkers", self.rand_bunkers),
                                 ("lakes", "Lakes", self.rand_lakes)]:
            self.screen.blit(self.f_sm.render(f"{label}: {val}", True, C["text"]), (x, y))
            minus_r = pygame.Rect(x + 110, y - 1, 22, 18)
            plus_r  = pygame.Rect(x + 136, y - 1, 22, 18)
            for rect, lbl in [(minus_r, "-"), (plus_r, "+")]:
                hover = rect.collidepoint(mouse)
                pygame.draw.rect(self.screen, C["btn_hover"] if hover else C["btn"], rect, border_radius=4)
                t = self.f_sm.render(lbl, True, C["text"])
                self.screen.blit(t, t.get_rect(center=rect.center))
            self._rand_rects[key] = (minus_r, plus_r)
            y += 22

        y += 4
        self._randomize_rect = pygame.Rect(x, y, PANEL_W - 28, 26)
        hover = self._randomize_rect.collidepoint(mouse)
        bg = C["btn_hover"] if hover else C["btn"]
        pygame.draw.rect(self.screen, bg, self._randomize_rect, border_radius=6)
        pygame.draw.rect(self.screen, C["accent"], self._randomize_rect, 1, border_radius=6)
        t = self.f_sm.render("Randomize", True, C["accent"])
        self.screen.blit(t, t.get_rect(center=self._randomize_rect.center))
        y += 34

        # Action buttons
        y += 6
        pygame.draw.line(self.screen, C["divider"], (x, y), (x + PANEL_W - 28, y))
        y += 10

        self._clear_rect = pygame.Rect(x, y, PANEL_W - 28, 30)
        hover = self._clear_rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, C["btn_hover"] if hover else C["btn"], self._clear_rect, border_radius=6)
        t = self.f_btn.render("Clear All", True, C["warn"])
        self.screen.blit(t, t.get_rect(center=self._clear_rect.center))
        y += 38

        self._done_rect = pygame.Rect(x, y, PANEL_W - 28, 34)
        hover = self._done_rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, C["btn_hover"] if hover else C["accent"], self._done_rect, border_radius=6)
        t = self.f_btn.render("Done — Start Game", True, (255, 255, 255))
        self.screen.blit(t, t.get_rect(center=self._done_rect.center))

        # Message
        if self.message and self.msg_timer > 0:
            t = self.f_md.render(self.message, True, C["accent"])
            self.screen.blit(t, (x, SCREEN_H - 24))

    # ----- panel click handling -----

    def _handle_panel_click(self, pos):
        if self._size_minus_rect.collidepoint(pos):
            self._set_size(self.board_size - 1)
            return
        if self._size_plus_rect.collidepoint(pos):
            self._set_size(self.board_size + 1)
            return

        for tool_id, rect in self._tool_rects.items():
            if rect.collidepoint(pos):
                self.tool = tool_id
                self.pending_piece = None
                return

        # Random obstacle +/- buttons
        max_cells = max(1, (self.board_size - 1) ** 2)
        for key, (minus_r, plus_r) in self._rand_rects.items():
            if minus_r.collidepoint(pos):
                if key == "towers":
                    self.rand_towers = max(0, self.rand_towers - 1)
                elif key == "bunkers":
                    self.rand_bunkers = max(0, self.rand_bunkers - 1)
                elif key == "lakes":
                    self.rand_lakes = max(0, self.rand_lakes - 1)
                return
            if plus_r.collidepoint(pos):
                if key == "towers":
                    self.rand_towers = min(max_cells, self.rand_towers + 1)
                elif key == "bunkers":
                    self.rand_bunkers = min(max_cells, self.rand_bunkers + 1)
                elif key == "lakes":
                    self.rand_lakes = min(max_cells, self.rand_lakes + 1)
                return

        if self._randomize_rect.collidepoint(pos):
            self._randomize_obstacles()
            return

        if self._clear_rect.collidepoint(pos):
            self.builder = GameSetupBuilder(size=self.board_size)
            self.pending_piece = None
            self._msg("Board cleared")
            return

        if self._done_rect.collidepoint(pos):
            if self._validate():
                return "__done__"

        return None

    # ----- main loop -----

    def run(self):
        # Initialize rects (drawn in first frame)
        self._size_minus_rect = pygame.Rect(0, 0, 0, 0)
        self._size_plus_rect = pygame.Rect(0, 0, 0, 0)
        self._tool_rects = {}
        self._clear_rect = pygame.Rect(0, 0, 0, 0)
        self._done_rect = pygame.Rect(0, 0, 0, 0)
        self._rand_rects = {}
        self._randomize_rect = pygame.Rect(0, 0, 0, 0)

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return "__quit__"
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        if self.pending_piece:
                            self.pending_piece = None
                            self._msg("Piece placement cancelled")
                        else:
                            return "__back__"
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    px = ev.pos[0]
                    if px > SCREEN_W - PANEL_W:
                        r = self._handle_panel_click(ev.pos)
                        if r == "__done__":
                            return self.builder.build()
                    else:
                        self._handle_click(ev.pos)

            # Tick message
            if self.msg_timer > 0:
                self.msg_timer -= 1

            # Draw
            self.screen.fill(C["bg"])
            self._draw_board()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(60)
