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

# Ensure core/ and project root are importable
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "core"))
sys.path.insert(0, os.path.join(_base, ".."))

from dotscuts import setup_standard_game
from ai_core import execute_action
from game_display import GameDisplay
from bot_player import create_bot
from custom_setup import PrebuiltSetups
from mode_selection import ModeSelector, GameConfig


# ---------------------------------------------------------------------------
# Map builder helper
# ---------------------------------------------------------------------------
def build_game_state(config: GameConfig):
    """Create a GameState from the chosen map name."""
    if config.map_name == "standard":
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
        self.bot_player = None  # which player number the bot controls
        if config.mode == "pvbot":
            self.bot = create_bot(config)
            # Bot plays the opposite side of the human
            self.bot_player = 2 if config.human_player == 1 else 1

        # State
        self.current_player = 1
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()
        self.show_bot_thinking = True
        self.show_grid = False       # unvisited edges hidden by default
        self.show_z_hints = False    # z-value vertex colours hidden by default
        self.game_over = False
        self.winner = None
        self.message = ""
        self.message_timer = 0
        self._bot_top_cache = None  # cached top-k for display

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
            self.selected_piece.move(tx, ty, self.game_state)
            self._show(f"Moved to ({tx},{ty})")
            self._end_turn()
        elif (tx, ty) in self.legal_shoots:
            self.selected_piece.shoot(tx, ty, self.game_state)
            self._show(f"Shot to ({tx},{ty})")
            self._end_turn()
        else:
            self._show("Invalid target!")

    def _end_turn(self):
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()
        self._bot_top_cache = None

        over, winner = self.game_state.is_game_over()
        if over:
            self.game_over = True
            self.winner = winner
            self._show(f"Game over! Player {winner} wins!")
            return

        self.current_player = 2 if self.current_player == 1 else 1

    def _do_bot_turn(self):
        if not self.bot or not self._is_bot_turn() or self.game_over:
            return

        action = self.bot.get_best_action(self.game_state, self.current_player)
        if action:
            desc = self.bot.action_to_readable_string(action)
            execute_action(self.game_state, action)
            self._show(f"Bot: {desc}")
        else:
            self._show("Bot has no legal moves!")
        self._end_turn()

    def _refresh_bot_thinking(self):
        """Cache the bot's top-k evaluation for the *human* player's view."""
        if not self.bot or not self.show_bot_thinking:
            self._bot_top_cache = None
            return
        if self.game_over:
            return
        # Show what the bot would do from the current player's perspective
        # (useful to see the bot's evaluation while it's the human's turn)
        if not self._is_bot_turn():
            try:
                top = self.bot.get_top_k_actions(self.game_state, self.bot_player, k=3)
                self._bot_top_cache = [
                    (self.bot.action_to_readable_string(a), s, b)
                    for a, s, b in top
                ]
            except Exception:
                self._bot_top_cache = None
        else:
            self._bot_top_cache = None

    # ----- event handling -----

    def _handle_click(self, pos):
        if self.game_over or self._is_bot_turn():
            return

        vertex = self.display.pixel_to_vertex(pos[0], pos[1])
        if vertex is None:
            return

        vx, vy = vertex
        my_pieces = [p for p in self.game_state.pieces
                     if p.x == vx and p.y == vy and p.player == self.current_player]

        if my_pieces:
            # Cycle through pieces at this vertex
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

    # ----- main loop -----

    def run(self) -> str:
        """
        Run the game loop.
        Returns:
            "menu"  -> go back to menu
            "quit"  -> exit application
        """
        self._refresh_bot_thinking()

        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                    self._refresh_bot_thinking()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "menu"

                    elif event.key == pygame.K_r:
                        # Restart same config
                        self.__init__(self.config)
                        self._show("Game restarted!")

                    elif event.key == pygame.K_u:
                        if not self.game_over and self.game_state.history:
                            if self.bot:
                                # Undo both bot + human to return to human's previous decision
                                self.game_state.undo_last_move()  # undo bot
                                if self.game_state.history:
                                    self.game_state.undo_last_move()  # undo human
                                # current_player stays (still human's turn)
                            else:
                                # PvP: undo one move, switch turn back
                                self.game_state.undo_last_move()
                                self.current_player = 2 if self.current_player == 1 else 1
                            self.selected_piece = None
                            self.legal_moves = set()
                            self.legal_shoots = set()
                            self._show("Undo")
                            self._refresh_bot_thinking()

                    elif event.key == pygame.K_b:
                        self.show_bot_thinking = not self.show_bot_thinking
                        self._refresh_bot_thinking()
                        state = "ON" if self.show_bot_thinking else "OFF"
                        self._show(f"Bot thinking: {state}")

                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                        self._show(f"Grid edges: {'ON' if self.show_grid else 'OFF'}")

                    elif event.key == pygame.K_z:
                        self.show_z_hints = not self.show_z_hints
                        self._show(f"Z-value hints: {'ON' if self.show_z_hints else 'OFF'}")

            # Bot turn (after events so it draws one frame first)
            if self._is_bot_turn() and not self.game_over:
                self._do_bot_turn()
                self._refresh_bot_thinking()

            # Tick message timer
            if self.message_timer > 0:
                self.message_timer -= 1
            else:
                self.message = ""

            # Draw
            self.display.draw_frame(
                self.game_state,
                self.current_player,
                selected_piece=self.selected_piece,
                legal_moves=self.legal_moves,
                legal_shoots=self.legal_shoots,
                bot_top_moves=self._bot_top_cache,
                bot_label=self.bot.label if self.bot else "",
                message=self.message,
                game_over=self.game_over,
                winner=self.winner,
                show_grid=self.show_grid,
                show_z_hints=self.show_z_hints,
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
            break  # user closed the menu

        try:
            game = GameUI(config)
            result = game.run()
            if result == "quit":
                break
            # result == "menu" -> loop back to selector
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError: {e}")
            break

    pygame.quit()


if __name__ == "__main__":
    main()
