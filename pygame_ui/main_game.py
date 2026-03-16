"""
Main Game UI
============
Interactive PyGame interface for playing Dots & Cuts.
Supports:
1. Player vs Player (1v1 local)
2. Player vs Bot (using trained RL model)
3. Bot vs Bot (watch two bots play)
"""

import pygame
import sys
import os

sys.path.insert(0, '../core')

from dotscuts import setup_standard_game
from ai_core import execute_action, generate_all_actions
from game_display import GameDisplay
from bot_player import BotPlayer


class GameMode:
    PLAYER_VS_PLAYER = 1
    PLAYER_VS_BOT = 2
    BOT_VS_BOT = 3


class GameUI:
    """
    Main game interface.
    Handles:
    - Piece selection and movement
    - Board display
    - Turn management
    - Bot interactions
    """

    def __init__(self, game_mode=GameMode.PLAYER_VS_PLAYER, bot_model_path=None):
        """
        Initialize the game UI.

        Args:
            game_mode: GameMode.PLAYER_VS_PLAYER, PLAYER_VS_BOT, or BOT_VS_BOT
            bot_model_path: Path to trained bot model (required for bot modes)
        """
        pygame.init()

        self.game_mode = game_mode
        self.bot_model_path = bot_model_path
        self.bot = None

        # Load bot if needed
        if game_mode in [GameMode.PLAYER_VS_BOT, GameMode.BOT_VS_BOT]:
            if not bot_model_path or not os.path.exists(bot_model_path):
                raise FileNotFoundError(f"Bot model not found: {bot_model_path}")
            self.bot = BotPlayer(bot_model_path, device='cpu')
            print(f"[BOT LOADED] {self.bot}")

        # Initialize game
        self.game_state = setup_standard_game()
        self.current_player = 1

        # Display
        self.display = GameDisplay(width=1200, height=900)

        # UI State
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()
        self.show_bot_thinking = True
        self.game_over = False
        self.winner = None
        self.message = ""
        self.message_time = 0

    def get_legal_targets(self, piece):
        """
        Get all legal move and shoot targets for a piece.

        Returns:
            (legal_moves_set, legal_shoots_set)
        """
        legal_moves = set()
        legal_shoots = set()

        # Check all possible moves
        if piece.kind == "orthogonal":
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:  # diagonal
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            new_x, new_y = piece.x + dx, piece.y + dy
            if piece.can_move(new_x, new_y, self.game_state):
                legal_moves.add((new_x, new_y))

        # Check all possible shoots
        for opponent_piece in self.game_state.pieces:
            if opponent_piece.player != piece.player:
                if piece.can_shoot(opponent_piece.x, opponent_piece.y, self.game_state):
                    legal_shoots.add((opponent_piece.x, opponent_piece.y))

        return legal_moves, legal_shoots

    def select_piece(self, piece):
        """
        Select a piece (highlights it and shows legal moves).
        """
        if piece.player != self.current_player:
            self.show_message("Not your piece!")
            return

        self.selected_piece = piece
        self.legal_moves, self.legal_shoots = self.get_legal_targets(piece)
        self.show_message(f"Selected {piece.kind} at ({piece.x},{piece.y}). Click a target.")

    def move_piece(self, target_x, target_y):
        """
        Attempt to move selected piece to target.
        """
        if not self.selected_piece:
            self.show_message("No piece selected!")
            return False

        if (target_x, target_y) in self.legal_moves:
            self.selected_piece.move(target_x, target_y, self.game_state)
            self.show_message(f"Moved {self.selected_piece.kind}")
            self.end_turn()
            return True

        elif (target_x, target_y) in self.legal_shoots:
            self.selected_piece.shoot(target_x, target_y, self.game_state)
            self.show_message(f"Shot at ({target_x},{target_y})")
            self.end_turn()
            return True

        else:
            self.show_message("Invalid target!")
            return False

    def end_turn(self):
        """
        End current player's turn and check for game over.
        """
        self.selected_piece = None
        self.legal_moves = set()
        self.legal_shoots = set()

        # Check game over
        game_over, winner = self.game_state.is_game_over()

        if game_over:
            self.game_over = True
            self.winner = winner
            self.show_message(f"GAME OVER! Player {winner} wins!")
            return

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        # If bot's turn, execute bot move
        if self._is_bot_turn():
            self.execute_bot_move()

    def execute_bot_move(self):
        """
        Let the bot execute its best move.
        """
        if not self.bot or not self._is_bot_turn():
            return

        action = self.bot.get_best_action(self.game_state, self.current_player)

        if action:
            execute_action(self.game_state, action)
            action_str = self.bot.action_to_readable_string(action)
            self.show_message(f"Bot played: {action_str}")
            self.end_turn()
        else:
            # No legal moves
            self.show_message("Bot has no legal moves!")
            self.end_turn()

    def _is_bot_turn(self) -> bool:
        """
        Check if it's the bot's turn.
        """
        if self.game_mode == GameMode.PLAYER_VS_PLAYER:
            return False
        elif self.game_mode == GameMode.PLAYER_VS_BOT:
            return self.current_player == 2  # Bot is player 2
        elif self.game_mode == GameMode.BOT_VS_BOT:
            return True  # Always bot's turn
        return False

    def show_message(self, msg: str, duration: float = 60):
        """
        Display a message at the bottom of screen.
        """
        self.message = msg
        self.message_time = duration

    def handle_click(self, pos):
        """
        Handle mouse click on the board.
        """
        if self.game_over:
            self.show_message("Game is over! Press 'R' to restart or 'Q' to quit.")
            return

        if self._is_bot_turn():
            self.show_message("Waiting for bot...")
            return

        vertex = self.display.pixel_to_vertex(pos[0], pos[1])

        if vertex is None:
            return

        vx, vy = vertex

        # Find piece at this vertex
        piece_at_target = None
        for piece in self.game_state.pieces:
            if piece.x == vx and piece.y == vy and piece.player == self.current_player:
                piece_at_target = piece
                break

        if piece_at_target:
            # Clicked on own piece - select it
            self.select_piece(piece_at_target)
        elif self.selected_piece:
            # Clicked on target
            self.move_piece(vx, vy)
        else:
            self.show_message("Click on one of your pieces first!")

    def draw(self):
        """
        Render the current game state.
        """
        # Draw board
        self.display.draw_board(self.game_state)

        # Draw pieces with highlights
        self.display.draw_pieces(
            self.game_state,
            selected_piece=self.selected_piece,
            legal_moves=self.legal_moves,
            legal_shoots=self.legal_shoots
        )

        # Draw bot thinking (top-right corner)
        if self.show_bot_thinking and self.bot and self.current_player == 1:
            top_moves = self.bot.get_top_k_actions(self.game_state, self.current_player, k=3)
            if top_moves:
                top_3_strings = [
                    (self.bot.action_to_readable_string(action), q_val, is_best)
                    for action, q_val, is_best in top_moves
                ]
                self.display.draw_bot_thinking(top_3_strings, position=(self.display.width - 350, 20))

        # Draw UI
        game_info = ""
        if self._is_bot_turn():
            game_info = "[BOT THINKING]"
        elif self.selected_piece:
            game_info = f"Moves: {len(self.legal_moves)}, Shoots: {len(self.legal_shoots)}"

        self.display.draw_ui_bottom(
            self.current_player,
            game_info=game_info
        )

        # Draw message
        if self.message and self.message_time > 0:
            msg_surface = self.display.font_medium.render(self.message, True, (200, 50, 50))
            self.display.screen.blit(msg_surface, (20, self.display.height - 80))
            self.message_time -= 1

        # Update display
        self.display.update()

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            False if user quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_click(event.pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Restart
                    self.__init__(game_mode=self.game_mode, bot_model_path=self.bot_model_path)
                    self.show_message("Game restarted!")

                elif event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_b:
                    # Toggle bot thinking display
                    self.show_bot_thinking = not self.show_bot_thinking
                    state = "ON" if self.show_bot_thinking else "OFF"
                    self.show_message(f"Bot thinking display: {state}")

                elif event.key == pygame.K_u:
                    # Undo last move
                    self.game_state.undo_last_move()
                    self.show_message("Undid last move")

        return True

    def run(self):
        """
        Main game loop.
        """
        print("\n" + "="*80)
        print("DOTS & CUTS - INTERACTIVE GAME")
        print("="*80)
        print(f"Game Mode: {self._get_mode_name()}")
        print("\nControls:")
        print("  - Click on your pieces to select them")
        print("  - Click on highlighted squares to move/shoot")
        print("  - Press 'B' to toggle bot thinking display")
        print("  - Press 'U' to undo last move")
        print("  - Press 'R' to restart")
        print("  - Press 'Q' to quit")
        print("="*80 + "\n")

        running = True
        while running:
            running = self.handle_events()

            if self._is_bot_turn() and not self.game_over:
                self.execute_bot_move()

            self.draw()

        self.display.quit()

    def _get_mode_name(self):
        """Get human-readable game mode name."""
        modes = {
            GameMode.PLAYER_VS_PLAYER: "Player vs Player",
            GameMode.PLAYER_VS_BOT: "Player vs Bot",
            GameMode.BOT_VS_BOT: "Bot vs Bot"
        }
        return modes.get(self.game_mode, "Unknown")


def main():
    """
    Main entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Dots & Cuts Interactive Game")
    parser.add_argument('--mode', type=str, default='pvp',
                        choices=['pvp', 'pvbot', 'botbot'],
                        help='Game mode: pvp=Player vs Player, pvbot=Player vs Bot, botbot=Bot vs Bot')
    parser.add_argument('--bot-model', type=str, default='../RL_approach/checkpoints/model_ep5000.pt',
                        help='Path to trained bot model')

    args = parser.parse_args()

    # Map mode strings to GameMode enum
    mode_map = {
        'pvp': GameMode.PLAYER_VS_PLAYER,
        'pvbot': GameMode.PLAYER_VS_BOT,
        'botbot': GameMode.BOT_VS_BOT
    }

    game_mode = mode_map[args.mode]

    # Create game
    try:
        game = GameUI(game_mode=game_mode, bot_model_path=args.bot_model)
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
