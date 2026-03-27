"""
Game Display Module for Dots & Cuts
====================================
Renders the board and UI elements using PyGame.

Supports any board size (5x5 to 15x15) by computing layout dynamically.
Left area: board.  Right panel: info + bot thinking + move timeline.

Toggleable overlays (controlled by the caller):
  - show_grid:    draw unvisited edges as faint lines  (default OFF)
  - show_z_hints: colour vertices by z-value            (default OFF)
"""

import math
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
        "tl_bg":           (28,  28,  33),
        "tl_current":      (45,  55,  80),
        "tl_hover":        (40,  45,  60),
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
        self.font_tl  = pygame.font.SysFont("Courier New", 14)
        # Bold piece label font — sized to fit the piece square
        piece_font_sz = max(14, int(self.cell_size * 0.22))
        self.font_piece = pygame.font.SysFont("Arial", piece_font_sz, bold=True)

        self.clock = pygame.time.Clock()

        # Timeline click regions: [(rect, move_index), ...]
        self._timeline_rects = []

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

    def timeline_hit_test(self, pos):
        """Check if a click hit a timeline entry. Returns MoveNode or None."""
        for rect, node in self._timeline_rects:
            if rect.collidepoint(pos):
                return node
        return None

    # ---- full frame ----

    def draw_frame(self, game_state, current_player, *,
                   selected_piece=None, legal_moves=None, legal_shoots=None,
                   pv_lines=None, bot_label="", analysis_label="", message="",
                   game_over=False, winner=None,
                   show_grid=False, show_z_hints=False,
                   timeline_items=None, timeline_scroll=0,
                   eval_data=None, my_best_move=None, opp_best_move=None,
                   show_eval=False, show_my_best=False, show_opp_best=False):
        """Draw one complete frame. Toggles passed from GameUI."""
        self.screen.fill(self.COLORS["bg"])

        self._draw_edges(game_state, show_grid)
        self._draw_obstacles(game_state)
        self._draw_vertices(game_state, show_z_hints)
        self._draw_highlights(legal_moves, legal_shoots)
        self._draw_pieces(game_state, selected_piece)
        self._draw_move_suggestions(my_best_move, opp_best_move)
        self._draw_eval_bar(eval_data)
        self._draw_panel(current_player, pv_lines, bot_label, analysis_label,
                         message, game_over, winner, game_state,
                         show_grid, show_z_hints,
                         timeline_items, timeline_scroll,
                         show_eval, show_my_best, show_opp_best)

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
            # Sort by arrival_order ascending so last arrived draws on top
            pieces_sorted = sorted(pieces, key=lambda p: p.arrival_order)
            if len(pieces_sorted) == 1:
                self._piece(pieces_sorted[0], px, py, sz, pieces_sorted[0] is selected_piece)
            else:
                off = int(sz * 0.8)
                for i, p in enumerate(pieces_sorted):
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

        # Draw kind symbol: + for orthogonal, x for diagonal
        line_len = max(4, int(sz * 0.65))
        line_w = max(2, int(sz * 0.18))
        w = (255, 255, 255)
        if piece.kind == "orthogonal":
            pygame.draw.line(self.screen, w,
                             (px - line_len, py), (px + line_len, py), line_w)
            pygame.draw.line(self.screen, w,
                             (px, py - line_len), (px, py + line_len), line_w)
        else:
            pygame.draw.line(self.screen, w,
                             (px - line_len, py - line_len),
                             (px + line_len, py + line_len), line_w)
            pygame.draw.line(self.screen, w,
                             (px - line_len, py + line_len),
                             (px + line_len, py - line_len), line_w)

    # ---- eval bar ----

    def _draw_eval_bar(self, eval_data):
        """Draw vertical evaluation bar between board and panel."""
        if eval_data is None:
            return

        bar_w = 16
        bar_h = int((self.board_size - 1) * self.cell_size)
        bar_x = int(self.board_x + (self.board_size - 1) * self.cell_size) + 24
        bar_y = int(self.board_y)

        pct = eval_data['bar_pct']  # 0-1, >0.5 = P1 advantage

        # Border
        pygame.draw.rect(self.screen, (50, 50, 56),
                         (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2), 1,
                         border_radius=2)

        # P2 portion (top, red)
        p2_h = int(bar_h * (1 - pct))
        if p2_h > 0:
            pygame.draw.rect(self.screen, self.COLORS["p2"],
                             (bar_x, bar_y, bar_w, p2_h))

        # P1 portion (bottom, green)
        p1_h = bar_h - p2_h
        if p1_h > 0:
            pygame.draw.rect(self.screen, self.COLORS["p1"],
                             (bar_x, bar_y + p2_h, bar_w, p1_h))

        # Center marker
        center_y = bar_y + bar_h // 2
        pygame.draw.line(self.screen, (200, 200, 200),
                         (bar_x - 3, center_y), (bar_x + bar_w + 3, center_y), 1)

        # Percentage labels
        p1_pct = int(pct * 100)
        p2_pct = 100 - p1_pct

        p2_txt = self.font_tiny.render(f"{p2_pct}%", True, self.COLORS["p2"])
        self.screen.blit(p2_txt, (bar_x + bar_w + 4, bar_y))

        p1_txt = self.font_tiny.render(f"{p1_pct}%", True, self.COLORS["p1"])
        self.screen.blit(p1_txt, (bar_x + bar_w + 4, bar_y + bar_h - 14))

        # Score label (raw eval value)
        cur_q = eval_data.get('cur_q', 0)
        q_player = eval_data.get('player', 1)
        q_col = self.COLORS["p1"] if q_player == 1 else self.COLORS["p2"]
        q_txt = self.font_tiny.render(f"{cur_q:+.3f}", True, q_col)
        self.screen.blit(q_txt, (bar_x + bar_w + 4, bar_y + bar_h // 2 - 6))

        # "EVAL" label
        eval_lbl = self.font_tiny.render("EVAL", True, self.COLORS["text_dim"])
        self.screen.blit(eval_lbl, (bar_x - 2, bar_y - 16))

    # ---- best move suggestions ----

    def _draw_move_suggestions(self, my_best, opp_best):
        """Draw arrows on the board showing suggested moves."""
        if opp_best:
            self._draw_arrow(opp_best, (240, 170, 50))    # orange
        if my_best:
            self._draw_arrow(my_best, (100, 240, 130))     # green (on top)

    def _draw_arrow(self, move_info, color):
        """Draw an arrow from piece position to target."""
        sx, sy = self.vertex_to_pixel(move_info['piece_x'], move_info['piece_y'])
        tx, ty = self.vertex_to_pixel(move_info['target_x'], move_info['target_y'])

        # Shaft
        pygame.draw.line(self.screen, color, (sx, sy), (tx, ty), 3)

        # Arrowhead
        dx, dy = tx - sx, ty - sy
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return
        angle = math.atan2(dy, dx)
        head_len = min(14, length * 0.3)
        spread = 0.4
        p1 = (tx - head_len * math.cos(angle - spread),
              ty - head_len * math.sin(angle - spread))
        p2 = (tx - head_len * math.cos(angle + spread),
              ty - head_len * math.sin(angle + spread))
        pygame.draw.polygon(self.screen, color, [(tx, ty), p1, p2])

        # Shoot: crosshair at target
        if move_info.get('action_type') == 'shoot':
            r = 8
            pygame.draw.line(self.screen, (255, 60, 60),
                             (tx - r, ty - r), (tx + r, ty + r), 3)
            pygame.draw.line(self.screen, (255, 60, 60),
                             (tx - r, ty + r), (tx + r, ty - r), 3)

    # ---- right panel ----

    def _draw_panel(self, cur_player, pv_lines, bot_label, analysis_label,
                    msg, game_over, winner, gs, show_grid, show_z,
                    timeline_items, timeline_scroll,
                    show_eval=False, show_my_best=False, show_opp_best=False):
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
            y += 20
        if analysis_label:
            self.screen.blit(self.font_md.render(
                f"Analysis: {analysis_label}", True, self.COLORS["text_dim"]), (x, y))
            y += 20

        y += 4
        pygame.draw.line(self.screen, self.COLORS["divider"], (x, y), (x + self.PANEL_WIDTH - 28, y))
        y += 10

        # PV analysis
        if pv_lines is not None:
            hdr = f"P{cur_player} Analysis"
            if pv_lines.get('computing'):
                hdr += "..."
            self.screen.blit(self.font_lg.render(hdr, True, self.COLORS["accent"]), (x, y))
            y += 26
            dot_colors = {
                '1st':   self.COLORS["p1"],
                '2nd':   (180, 180, 60),
                '3rd':   (180, 180, 60),
                'worst': self.COLORS["p2"],
            }
            for line in pv_lines.get('lines', []):
                # Colored rank dot
                dot_col = dot_colors.get(line['rank'], self.COLORS["text"])
                pygame.draw.circle(self.screen, dot_col, (x + 4, y + 7), 3)
                # PV text
                txt = self.font_tiny.render(line['text'], True, self.COLORS["text"])
                self.screen.blit(txt, (x + 12, y + 1))
                # Score right-aligned
                sc = self.font_tiny.render(f"{line['score']:+.3f}", True, self.COLORS["text_dim"])
                self.screen.blit(sc, (x + self.PANEL_WIDTH - 28 - sc.get_width(), y + 1))
                y += 15
            y += 6

        # --- Move Timeline ---
        timeline_top = y
        timeline_bottom = self.height - 140
        self._draw_timeline(x, timeline_top, timeline_bottom,
                            timeline_items, timeline_scroll)

        # --- Bottom section: toggles + controls ---
        y2 = self.height - 160
        pygame.draw.line(self.screen, self.COLORS["divider"], (x, y2), (x + self.PANEL_WIDTH - 28, y2))
        y2 += 8

        grid_st = "ON" if show_grid else "OFF"
        z_st    = "ON" if show_z    else "OFF"
        self.screen.blit(self.font_sm.render(
            f"G=Grid:{grid_st}  Z=Hints:{z_st}", True, self.COLORS["text_dim"]), (x, y2))
        y2 += 16

        eval_st = "ON" if show_eval else "OFF"
        my_st   = "ON" if show_my_best else "OFF"
        opp_st  = "ON" if show_opp_best else "OFF"
        self.screen.blit(self.font_sm.render(
            f"E=Eval:{eval_st}  M=Best:{my_st}  N=Opp:{opp_st}",
            True, self.COLORS["text_dim"]), (x, y2))
        y2 += 16

        for line in ["Click=Select/Act  U=Undo  R=Restart",
                      "B=Lines  D / Shift+D = Bot depth -/+",
                      "A / Shift+A = Analysis depth -/+",
                      "T / Shift+T = Lines timeout -/+  Q=Menu"]:
            self.screen.blit(self.font_sm.render(line, True, self.COLORS["text_dim"]), (x, y2))
            y2 += 16

        # Message
        if msg:
            self.screen.blit(self.font_md.render(msg, True, self.COLORS["accent"]),
                             (x, self.height - 24))

    # ---- move timeline ----

    def _draw_timeline(self, x, y_top, y_bottom, timeline_items, scroll):
        """Draw the scrollable move timeline with tree variations."""
        self._timeline_rects = []

        if not timeline_items:
            self.screen.blit(self.font_sm.render(
                "No moves yet", True, self.COLORS["text_dim"]), (x, y_top + 4))
            return

        # Header
        self.screen.blit(self.font_md.render("Move Log", True, self.COLORS["text"]), (x, y_top))
        y_top += 22
        pygame.draw.line(self.screen, self.COLORS["divider"],
                         (x, y_top), (x + self.PANEL_WIDTH - 28, y_top))
        y_top += 4

        clip_h = y_bottom - y_top
        if clip_h < 20:
            return

        # Build display lines from tree items.
        # Turn numbering from game_depth (actual tree position).
        # P1+P2 paired only if immediately consecutive (no variation block between).
        lines = []
        i = 0
        while i < len(timeline_items):
            item = timeline_items[i]

            if item['type'] in ('var_start', 'var_end'):
                i += 1
                continue

            if item['type'] == 'move':
                depth = item['depth']
                gd = item['game_depth']
                turn_num = (gd + 1) // 2

                if item['player'] == 1:
                    p1 = item
                    p2 = None
                    # Only pair with IMMEDIATE next item (don't skip over variations)
                    if (i + 1 < len(timeline_items) and
                            timeline_items[i + 1].get('type') == 'move' and
                            timeline_items[i + 1]['player'] == 2 and
                            timeline_items[i + 1]['depth'] == depth and
                            timeline_items[i + 1]['game_depth'] == gd + 1):
                        p2 = timeline_items[i + 1]
                        i += 2
                    else:
                        i += 1
                    lines.append({
                        'type': 'turn', 'depth': depth,
                        'turn_num': turn_num,
                        'p1': p1, 'p2': p2,
                    })
                else:
                    lines.append({
                        'type': 'turn', 'depth': depth,
                        'turn_num': turn_num,
                        'p1': None, 'p2': item,
                    })
                    i += 1
            else:
                i += 1

        # Insert gap lines at depth transitions for visual separation
        final_lines = []
        prev_depth = -1
        for line in lines:
            d = line.get('depth', 0)
            if prev_depth >= 0 and d != prev_depth:
                final_lines.append({'type': 'gap'})
            final_lines.append(line)
            prev_depth = d
        lines = final_lines

        line_h = 18
        gap_h = 7  # small spacer between depth levels
        total_h = sum(gap_h if l['type'] == 'gap' else line_h for l in lines)
        visible_lines = clip_h // line_h  # approximate
        mouse = pygame.mouse.get_pos()

        # Auto-scroll to keep current move visible
        current_line = None
        for li, line in enumerate(lines):
            if line['type'] == 'turn':
                if ((line.get('p1') and line['p1'].get('is_current')) or
                        (line.get('p2') and line['p2'].get('is_current'))):
                    current_line = li
                    break
        if current_line is not None:
            if current_line >= scroll + visible_lines:
                scroll = current_line - visible_lines + 1
            if current_line < scroll:
                scroll = current_line

        # Set clip
        clip_rect = pygame.Rect(x - 2, y_top, self.PANEL_WIDTH - 24, clip_h)
        self.screen.set_clip(clip_rect)

        indent_w = 14  # pixels per variation depth level
        var_bar_color = (70, 80, 110)

        y = y_top
        for li in range(scroll, len(lines)):
            if y >= y_top + clip_h:
                break

            line = lines[li]

            if line['type'] == 'gap':
                y += gap_h
                continue

            depth = line.get('depth', 0)
            ix = x + depth * indent_w

            # Draw vertical variation bars for each nesting level
            for d in range(1, depth + 1):
                bar_x = x + d * indent_w - indent_w + 3
                pygame.draw.line(self.screen, var_bar_color,
                                 (bar_x, y), (bar_x, y + line_h), 2)

            # Turn line
            tn = line['turn_num']
            p1 = line.get('p1')
            p2 = line.get('p2')

            if not p1 and p2:
                num_str = f"{tn:>2}. ... "
            else:
                num_str = f"{tn:>2}. "

            line_rect = pygame.Rect(ix - 2, y, self.PANEL_WIDTH - 24 - depth * indent_w, line_h)

            is_current = ((p1 and p1.get('is_current')) or (p2 and p2.get('is_current')))
            on_path = ((p1 and p1.get('on_path')) or (p2 and p2.get('on_path')))

            if is_current:
                pygame.draw.rect(self.screen, self.COLORS["tl_current"], line_rect, border_radius=3)
            elif on_path:
                pygame.draw.rect(self.screen, (35, 40, 55), line_rect, border_radius=3)
            elif line_rect.collidepoint(mouse):
                pygame.draw.rect(self.screen, self.COLORS["tl_hover"], line_rect, border_radius=3)

            dim = depth > 0
            num_surf = self.font_tl.render(num_str, True, self.COLORS["text_dim"])
            self.screen.blit(num_surf, (ix, y + 1))
            cx = ix + num_surf.get_width()

            if p1:
                is_cur = p1.get('is_current', False)
                if is_cur:
                    col = self.COLORS["p1"]
                elif dim:
                    col = self.COLORS["text_dim"]
                else:
                    col = self.COLORS["text"]
                s = self.font_tl.render(p1['notation'], True, col)
                r = pygame.Rect(cx, y, s.get_width() + 4, line_h)
                self.screen.blit(s, (cx, y + 1))
                self._timeline_rects.append((r, p1['node']))
                cx += s.get_width() + 8

            if p2:
                is_cur = p2.get('is_current', False)
                if is_cur:
                    col = self.COLORS["p2"]
                elif dim:
                    col = self.COLORS["text_dim"]
                else:
                    col = self.COLORS["text"]
                s = self.font_tl.render(p2['notation'], True, col)
                r = pygame.Rect(cx, y, s.get_width() + 4, line_h)
                self.screen.blit(s, (cx, y + 1))
                self._timeline_rects.append((r, p2['node']))

            y += line_h

        # Reset clip
        self.screen.set_clip(None)

        # Scroll indicator
        total_lines = len(lines)
        if total_lines > visible_lines:
            bar_h = max(10, int(clip_h * visible_lines / total_lines))
            bar_y = y_top + int((clip_h - bar_h) * scroll / max(1, total_lines - visible_lines))
            bar_x = x + self.PANEL_WIDTH - 32
            pygame.draw.rect(self.screen, self.COLORS["divider"],
                             (bar_x, bar_y, 4, bar_h), border_radius=2)

    def quit(self):
        pass
