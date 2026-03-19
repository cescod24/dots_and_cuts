"""
Dots & Cuts - Main Game
========================
Entry point for the PyGame interface.

Launch:
    python main_game.py

Flow:
    1. Mode selection menu  (mode_selection.ModeSelector)
    2. Game loop            (GameUI)
    3. On quit / restart -> back to menu
"""

import pygame
import sys
import os
import threading
import copy
import time

# Ensure core/ and project root are importable
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "core"))
sys.path.insert(0, os.path.join(_base, ".."))

from dotscuts import setup_standard_game
from ai_core import Action, execute_action
from move_notation import action_to_notation, notation_after_execution
from game_display import GameDisplay
from bot_player import create_bot
from custom_setup import PrebuiltSetups
from mode_selection import ModeSelector, GameConfig


# ---------------------------------------------------------------------------
# Move tree (for non-destructive timeline with variations)
# ---------------------------------------------------------------------------
class MoveNode:
    """
    Node in the move tree. children[0] is the main continuation;
    children[1:] are variations (shown in parentheses).
    """
    _next_id = 0

    def __init__(self, notation=None, player=None, parent=None, replay=None):
        self.id = MoveNode._next_id
        MoveNode._next_id += 1
        self.notation = notation
        self.player = player
        self.parent = parent
        self.children = []
        # Info to replay this move: {piece_kind, src_x, src_y, action_type, target_x, target_y, arrival_order}
        self.replay = replay

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def find_matching_child(self, notation):
        for child in self.children:
            if child.notation == notation:
                return child
        return None


def _path_from_root(node):
    """List of nodes from root (exclusive) to node (inclusive)."""
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    return list(reversed(path))


def _flatten_tree(root, current_node):
    """
    Flatten move tree into a display list for the timeline.
    Returns list of dicts:
        {'type': 'move', 'notation', 'player', 'node', 'depth', 'is_current'}
        {'type': 'var_start', 'depth'}
        {'type': 'var_end', 'depth'}
    """
    items = []
    current_path_ids = {n.id for n in _path_from_root(current_node)}
    current_path_ids.add(current_node.id)
    _flatten_children(root, items, 0, current_path_ids, current_node)
    return items


def _flatten_children(node, items, depth, current_ids, current_node, game_depth=0):
    """Flatten tree: iterate main line, insert variations at branch points.

    Variations appear RIGHT AFTER the move they branch from (not at the end).
    This matches PGN-style display: main move, then alternatives, then
    the main line continues.
    """
    current = node
    gd = game_depth

    while current.children:
        gd += 1
        main = current.children[0]

        # Output main child
        items.append({
            'type': 'move',
            'notation': main.notation,
            'player': main.player,
            'node': main,
            'depth': depth,
            'game_depth': gd,
            'is_current': main is current_node,
            'on_path': main.id in current_ids,
        })

        # Output variations HERE — right after the branch point,
        # BEFORE continuing the main line
        for var in current.children[1:]:
            items.append({'type': 'var_start', 'depth': depth + 1})
            items.append({
                'type': 'move',
                'notation': var.notation,
                'player': var.player,
                'node': var,
                'depth': depth + 1,
                'game_depth': gd,
                'is_current': var is current_node,
                'on_path': var.id in current_ids,
            })
            # Recurse into variation subtree
            _flatten_children(var, items, depth + 1, current_ids, current_node, gd)
            items.append({'type': 'var_end', 'depth': depth + 1})

        # Continue along main line
        current = main


# ---------------------------------------------------------------------------
# PV formatting
# ---------------------------------------------------------------------------
def _format_pv(moves, start_gd):
    """Format PV moves with chess-style turn numbers."""
    parts = []
    gd = start_gd
    for i, m in enumerate(moves):
        turn_num = (gd + 1) // 2
        is_p1 = gd % 2 == 1
        if is_p1:
            parts.append(f"{turn_num}.{m['notation']}")
        elif i == 0:
            parts.append(f"{turn_num}..{m['notation']}")
        else:
            parts.append(m['notation'])
        gd += 1
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Map builder helper
# ---------------------------------------------------------------------------
def build_game_state(config: GameConfig):
    """Create a GameState from the chosen map name."""
    if config.map_name == "custom" and config.custom_game_state is not None:
        return config.custom_game_state
    elif config.map_name == "standard":
        return setup_standard_game()
    elif config.map_name == "balanced":
        return PrebuiltSetups.balanced_9x9()
    elif config.map_name == "empty":
        return PrebuiltSetups.empty_board()
    elif config.map_name == "small_5x5":
        return PrebuiltSetups.small_5x5()
    else:
        return setup_standard_game()


# ---------------------------------------------------------------------------
# Main game UI
# ---------------------------------------------------------------------------
class GameUI:
    """
    Interactive game loop.
    Handles piece selection, move execution, bot turns, undo, and rendering.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.game_state = build_game_state(config)

        board_sz = self.game_state.board.size
        self.display = GameDisplay(board_size=board_sz)

        # Bot setup
        self.bot = None
        self.bot_player = None
        if config.mode == "pvbot":
            self.bot = create_bot(config)
            self.bot_player = 2 if config.human_player == 1 else 1

        # State
        self.current_player = 1
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()
        self.show_bot_thinking = True      # 0=off, 1=best only, 2=top3+worst, 3=full lines
        self.bot_thinking_mode = 3
        self.show_grid = False
        self.show_z_hints = False
        self.game_over = False
        self.winner = None
        self.message = ""
        self.message_timer = 0

        # Analysis overlays
        self.show_eval = False
        self.show_my_best = False
        self.show_opp_best = False
        self._eval_cache = None       # holds eval_data, my_best, opp_best, pv_lines

        # Background PV computation
        self._pv_generation = 0
        self._pv_thread = None
        self._pv_cancel = threading.Event()
        self._pv_timeout = 10          # seconds, adjustable with T/Shift+T
        self._pv_computing = False

        # Bot turn pending (render one frame before bot thinks)
        self._bot_pending = False

        # Move tree
        MoveNode._next_id = 0
        self.tree_root = MoveNode()
        self.current_node = self.tree_root
        self.timeline_scroll = 0

    # ----- helpers -----

    def _is_bot_turn(self) -> bool:
        return self.bot is not None and self.current_player == self.bot_player

    def _show(self, msg, frames=180):
        self.message = msg
        self.message_timer = frames

    def _get_legal_targets(self, piece):
        moves, shoots = set(), set()
        if piece.kind == "orthogonal":
            dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        else:
            dirs = [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dy in dirs:
            nx, ny = piece.x + dx, piece.y + dy
            if piece.can_move(nx, ny, self.game_state):
                moves.add((nx, ny))
        for other in self.game_state.pieces:
            if other.player != piece.player:
                if piece.can_shoot(other.x, other.y, self.game_state):
                    shoots.add((other.x, other.y))
        return moves, shoots

    # ----- move recording -----

    def _capture_replay(self, action):
        """Capture replay info BEFORE execution."""
        return {
            'piece_kind': action.piece.kind,
            'src_x': action.piece.x,
            'src_y': action.piece.y,
            'action_type': action.action_type,
            'target_x': action.target_x,
            'target_y': action.target_y,
            'arrival_order': action.piece.arrival_order,
            'player': action.piece.player,
        }

    def _record_move(self, action, replay_info):
        """Generate notation, add to tree or follow existing child."""
        notation = notation_after_execution(action, self.game_state)

        # Check if this move already exists as a child
        existing = self.current_node.find_matching_child(notation)
        if existing:
            self.current_node = existing
        else:
            node = MoveNode(notation, replay_info['player'],
                            parent=self.current_node, replay=replay_info)
            self.current_node.add_child(node)
            self.current_node = node

        return notation

    # ----- replay -----

    def _replay_node(self, node):
        """Replay a single move from a MoveNode. Returns True if successful."""
        info = node.replay
        if info is None:
            return False
        for p in self.game_state.pieces:
            if (p.kind == info['piece_kind'] and
                    p.x == info['src_x'] and p.y == info['src_y'] and
                    p.player == info['player']):
                action = Action(p, info['action_type'],
                                info['target_x'], info['target_y'])
                execute_action(self.game_state, action)
                return True
        return False

    # ----- navigation -----

    def _navigate_to_node(self, target):
        """Navigate game state to the given tree node."""
        if target is self.current_node:
            return

        current_path = _path_from_root(self.current_node)
        target_path = _path_from_root(target)

        # Find common prefix length
        common_len = 0
        for i in range(min(len(current_path), len(target_path))):
            if current_path[i] is target_path[i]:
                common_len = i + 1
            else:
                break

        # Undo from current back to common ancestor
        for _ in range(len(current_path) - common_len):
            if self.game_state.history:
                self.game_state.undo_last_move()

        # Replay from common ancestor to target
        for i in range(common_len, len(target_path)):
            if not self._replay_node(target_path[i]):
                self._show("Navigation failed!")
                return

        self.current_node = target

        # Update current player
        n = len(target_path)
        self.current_player = 1 if n % 2 == 0 else 2

        # Check game over at this position
        over, winner = self.game_state.is_game_over()
        self.game_over = over
        self.winner = winner if over else None

        # Reset UI
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()
        self._refresh_analysis()

    # ----- turn logic -----

    def _select_piece(self, piece):
        if piece.player != self.current_player:
            self._show("Not your piece!")
            return
        self.selected_piece = piece
        self.legal_moves, self.legal_shoots = self._get_legal_targets(piece)
        kind = piece.kind.capitalize()
        self._show(f"Selected {kind} at ({piece.x},{piece.y})")

    def _try_action(self, tx, ty):
        if not self.selected_piece:
            return
        if (tx, ty) in self.legal_moves:
            action = Action(self.selected_piece, "move", tx, ty)
            replay = self._capture_replay(action)
            execute_action(self.game_state, action)
            notation = self._record_move(action, replay)
            self._show(notation)
            self._end_turn()
        elif (tx, ty) in self.legal_shoots:
            action = Action(self.selected_piece, "shoot", tx, ty)
            replay = self._capture_replay(action)
            execute_action(self.game_state, action)
            notation = self._record_move(action, replay)
            self._show(notation)
            self._end_turn()
        else:
            self._show("Invalid target!")

    def _end_turn(self):
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()

        over, winner = self.game_state.is_game_over()
        if over:
            self.game_over = True
            self.winner = winner
            self._show(f"Game over! Player {winner} wins!")
            return

        self.current_player = 2 if self.current_player == 1 else 1
        self._refresh_analysis()

    def _do_bot_turn(self):
        if not self.bot or not self._is_bot_turn() or self.game_over:
            return

        action = self.bot.get_best_action(self.game_state, self.current_player)
        if action:
            replay = self._capture_replay(action)
            execute_action(self.game_state, action)
            notation = self._record_move(action, replay)
            self._show(f"Bot: {notation}")
        else:
            self._show("Bot has no legal moves!")
        self._end_turn()

    def _refresh_analysis(self):
        """Cancel running analysis and start fresh for the current position."""
        self._eval_cache = None
        self._pv_cancel.set()

        need = (self.show_bot_thinking or self.show_eval
                or self.show_my_best or self.show_opp_best)
        if not self.bot or self.game_over or not need:
            self._pv_computing = False
            return

        self._pv_generation += 1
        gen = self._pv_generation

        gs_copy = copy.deepcopy(self.game_state)
        player = self.current_player
        start_gd = len(self.game_state.history) + 1

        self._pv_cancel = threading.Event()
        self._pv_computing = True

        self._pv_thread = threading.Thread(
            target=self._analysis_worker,
            args=(gen, gs_copy, player, start_gd,
                  self._pv_cancel, self._pv_timeout),
            daemon=True,
        )
        self._pv_thread.start()

    def _analysis_worker(self, gen, gs, player, start_gd, cancel, timeout):
        """Background thread: compute eval, arrows, and PV lines."""
        import math as _math
        t0 = time.time()
        result = {}

        # --- All actions for current player (used for eval + PV ranking) ---
        try:
            all_scored = self.bot.get_top_k_actions(gs, player, k=200)
        except Exception:
            all_scored = []
        if cancel.is_set():
            self._pv_computing = False; return

        # --- Opponent's best (for eval bar + arrow) ---
        try:
            top_opp = self.bot.get_top_k_actions(gs, 3 - player, k=1)
        except Exception:
            top_opp = []
        if cancel.is_set():
            self._pv_computing = False; return

        # --- Eval bar ---
        # Use current player's Q-value as position evaluation.
        # Positive Q = current player is winning.
        # Convert to P1-relative for bar display (P1 green = bottom).
        cur_q = all_scored[0][1] if all_scored else 0
        p1_adv = cur_q if player == 1 else -cur_q
        bar_pct = 1 / (1 + _math.exp(-p1_adv * 10))
        bar_pct = max(0.03, min(0.97, bar_pct))
        result['eval_data'] = {
            'bar_pct': bar_pct,
            'cur_q': round(cur_q, 4),
            'player': player,
        }

        # --- Best-move arrows ---
        if all_scored:
            a = all_scored[0][0]
            result['my_best'] = {
                'piece_x': a.piece.x, 'piece_y': a.piece.y,
                'target_x': a.target_x, 'target_y': a.target_y,
                'action_type': a.action_type,
            }
        if top_opp:
            a = top_opp[0][0]
            result['opp_best'] = {
                'piece_x': a.piece.x, 'piece_y': a.piece.y,
                'target_x': a.target_x, 'target_y': a.target_y,
                'action_type': a.action_type,
            }

        # Early-publish eval + arrows before PV is done
        if gen == self._pv_generation and not cancel.is_set():
            self._eval_cache = result.copy()

        # --- PV lines ---
        if not self.show_bot_thinking or not all_scored:
            self._pv_computing = False; return

        pv_depth = 5
        candidates = []
        labels = ['1st', '2nd', '3rd']
        for i in range(min(3, len(all_scored))):
            candidates.append((labels[i], all_scored[i][0], all_scored[i][1]))
        if len(all_scored) > 3:
            candidates.append(('worst', all_scored[-1][0], all_scored[-1][1]))

        lines = []
        for label, first_action, first_score in candidates:
            if cancel.is_set() or time.time() - t0 > timeout:
                break

            mvs, undos = [], 0

            # First move
            n = action_to_notation(first_action, gs)
            execute_action(gs, first_action)
            undos += 1
            mvs.append({'notation': n, 'player': player})

            over, _ = gs.is_game_over()
            cur_p = 3 - player
            while not over and len(mvs) < pv_depth:
                if cancel.is_set() or time.time() - t0 > timeout:
                    break
                try:
                    top = self.bot.get_top_k_actions(gs, cur_p, k=1)
                except Exception:
                    break
                if not top:
                    break
                act = top[0][0]
                n = action_to_notation(act, gs)
                execute_action(gs, act)
                undos += 1
                mvs.append({'notation': n, 'player': cur_p})
                over, _ = gs.is_game_over()
                cur_p = 3 - cur_p

            for _ in range(undos):
                gs.undo_last_move()

            lines.append({
                'rank': label,
                'text': _format_pv(mvs, start_gd),
                'score': first_score,
            })

        result['pv_lines'] = lines
        if gen == self._pv_generation and not cancel.is_set():
            self._eval_cache = result

        self._pv_computing = False

    # ----- undo -----

    def _handle_undo(self):
        if self.game_over or self.current_node is self.tree_root:
            return

        target = self.current_node.parent
        if self.bot and target is not self.tree_root:
            # Undo both bot + human
            target = target.parent if target.parent else target

        self._navigate_to_node(target)
        self._show("Undo")

    # ----- event handling -----

    def _handle_click(self, pos):
        # Check timeline click first
        hit_node = self.display.timeline_hit_test(pos)
        if hit_node is not None:
            self._navigate_to_node(hit_node)
            return

        if self.game_over or self._is_bot_turn():
            return

        vertex = self.display.pixel_to_vertex(pos[0], pos[1])
        if vertex is None:
            return

        vx, vy = vertex
        my_pieces = [p for p in self.game_state.pieces
                     if p.x == vx and p.y == vy and p.player == self.current_player]

        is_legal_target = (vx, vy) in self.legal_moves or (vx, vy) in self.legal_shoots

        if self.selected_piece and is_legal_target:
            self._try_action(vx, vy)
        elif my_pieces:
            if (self.selected_piece and self.selected_piece.x == vx
                    and self.selected_piece.y == vy and self.selected_piece in my_pieces):
                idx = my_pieces.index(self.selected_piece)
                nxt = my_pieces[(idx + 1) % len(my_pieces)]
                self._select_piece(nxt)
            else:
                self._select_piece(my_pieces[0])
        elif self.selected_piece:
            self._try_action(vx, vy)
        else:
            self._show("Click on one of your pieces first!")

    # ----- timeline data -----

    def _build_timeline_data(self):
        """Build the flattened timeline for display."""
        return _flatten_tree(self.tree_root, self.current_node)

    # ----- main loop -----

    def run(self) -> str:
        self._refresh_analysis()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

                elif event.type == pygame.MOUSEWHEEL:
                    mx, _ = pygame.mouse.get_pos()
                    panel_x = self.display.width - self.display.PANEL_WIDTH
                    if mx >= panel_x:
                        self.timeline_scroll = max(0, self.timeline_scroll - event.y)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "menu"
                    elif event.key == pygame.K_r:
                        self.__init__(self.config)
                        self._show("Game restarted!")
                    elif event.key == pygame.K_u:
                        self._handle_undo()
                    elif event.key == pygame.K_b:
                        self.bot_thinking_mode = (self.bot_thinking_mode + 1) % 4
                        self.show_bot_thinking = self.bot_thinking_mode > 0
                        self._refresh_analysis()
                        labels = {0: 'OFF', 1: 'best move', 2: 'top 3 + worst', 3: 'full lines'}
                        self._show(f"Analysis: {labels[self.bot_thinking_mode]}")
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                        self._show(f"Grid edges: {'ON' if self.show_grid else 'OFF'}")
                    elif event.key == pygame.K_z:
                        self.show_z_hints = not self.show_z_hints
                        self._show(f"Z-value hints: {'ON' if self.show_z_hints else 'OFF'}")
                    elif event.key == pygame.K_e:
                        self.show_eval = not self.show_eval
                        self._refresh_analysis()
                        self._show(f"Eval bar: {'ON' if self.show_eval else 'OFF'}")
                    elif event.key == pygame.K_m:
                        self.show_my_best = not self.show_my_best
                        self._refresh_analysis()
                        self._show(f"My best move: {'ON' if self.show_my_best else 'OFF'}")
                    elif event.key == pygame.K_n:
                        self.show_opp_best = not self.show_opp_best
                        self._refresh_analysis()
                        self._show(f"Opponent best: {'ON' if self.show_opp_best else 'OFF'}")
                    elif event.key == pygame.K_d:
                        mods = pygame.key.get_mods()
                        if self.bot and hasattr(self.bot, 'depth'):
                            if mods & pygame.KMOD_SHIFT:
                                self.bot.depth = self.bot.depth + 1
                            else:
                                self.bot.depth = max(1, self.bot.depth - 1)
                            self.bot.label = f"Minimax {self.bot.version} (depth {self.bot.depth})"
                            self._refresh_analysis()
                            self._show(f"Minimax depth: {self.bot.depth}")
                    elif event.key == pygame.K_t:
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_SHIFT:
                            self._pv_timeout = min(60, self._pv_timeout + 5)
                        else:
                            self._pv_timeout = max(1, self._pv_timeout - 5)
                        self._show(f"Analysis timeout: {self._pv_timeout}s")

            # Bot turn: wait one frame so the human move renders first
            if self._bot_pending:
                self._bot_pending = False
                self._do_bot_turn()
            elif self._is_bot_turn() and not self.game_over:
                self._bot_pending = True
                self._show("Bot thinking...")

            # Tick message timer
            if self.message_timer > 0:
                self.message_timer -= 1
            else:
                self.message = ""

            # Build timeline
            timeline_items = self._build_timeline_data()

            # Draw
            ec = self._eval_cache or {}
            pv_data = None
            if self.show_bot_thinking:
                raw_lines = ec.get('pv_lines', [])
                if self.bot_thinking_mode == 1:
                    # Best move only: just the 1st line, first move
                    if raw_lines:
                        text = raw_lines[0].get('text', '')
                        first_move = text.split(' ')[0] if text else ''
                        raw_lines = [{**raw_lines[0], 'text': first_move}]
                    else:
                        raw_lines = []
                elif self.bot_thinking_mode == 2:
                    # Top 3 + worst: first move only for each
                    truncated = []
                    for line in raw_lines:
                        text = line.get('text', '')
                        first_move = text.split(' ')[0] if text else ''
                        truncated.append({**line, 'text': first_move})
                    raw_lines = truncated
                # mode 3: full lines, no changes
                pv_data = {
                    'lines': raw_lines,
                    'computing': self._pv_computing,
                }
            self.display.draw_frame(
                self.game_state,
                self.current_player,
                selected_piece=self.selected_piece,
                legal_moves=self.legal_moves,
                legal_shoots=self.legal_shoots,
                pv_lines=pv_data,
                bot_label=self.bot.label if self.bot else "",
                message=self.message,
                game_over=self.game_over,
                winner=self.winner,
                show_grid=self.show_grid,
                show_z_hints=self.show_z_hints,
                timeline_items=timeline_items,
                timeline_scroll=self.timeline_scroll,
                eval_data=ec.get('eval_data') if self.show_eval else None,
                my_best_move=ec.get('my_best') if self.show_my_best else None,
                opp_best_move=ec.get('opp_best') if self.show_opp_best else None,
                show_eval=self.show_eval,
                show_my_best=self.show_my_best,
                show_opp_best=self.show_opp_best,
            )

        return "quit"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    pygame.init()

    while True:
        selector = ModeSelector()
        config = selector.run()

        if config is None:
            break

        try:
            game = GameUI(config)
            result = game.run()
            if result == "quit":
                break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError: {e}")
            break

    pygame.quit()


if __name__ == "__main__":
    main()
