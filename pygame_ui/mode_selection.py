"""
Mode Selection Menu
====================
Multi-screen menu for configuring a Dots & Cuts game.

Flow:
  1. Game Mode   -> 1v1 or 1 vs Bot
  2. Bot Type    -> Minimax v1, Minimax v2, RL v1, RL v2
  3. Bot Config  -> Depth (minimax) or Strength tier (RL)
  4. Player Side -> Human plays as Player 1 or Player 2
  5. Map Select  -> Standard, Balanced, Empty, Small 5x5

Returns a GameConfig dataclass with all chosen settings.
"""

import pygame
import os
import glob
from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    """All settings chosen in the menu."""
    mode: str = "pvp"                  # "pvp" or "pvbot"
    bot_type: Optional[str] = None     # "minimax_v1", "minimax_v2", "rl_v1", "rl_v2"
    minimax_depth: int = 2
    rl_checkpoint: Optional[str] = None
    human_player: int = 1              # 1 or 2
    map_name: str = "standard"         # "standard", "balanced", "skirmish", "mid_7x7", "small_5x5", "custom"
    custom_game_state: object = None   # GameState from board builder (when map_name="custom")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 640

C = {
    "bg":         (24,  24,  28),
    "card":       (38,  38,  44),
    "card_hover": (50,  65, 110),
    "border":     (60,  60,  68),
    "title":      (240, 240, 240),
    "subtitle":   (130, 130, 145),
    "btn_text":   (220, 220, 225),
    "accent":     (90,  145, 255),
    "accent2":    (120, 220, 140),
    "dim":        (80,  80,  90),
    "warn":       (220, 90,  90),
    "tag_bg":     (55,  55,  65),
}


# ---------------------------------------------------------------------------
# Base menu class
# ---------------------------------------------------------------------------
class _Menu:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self.f_title  = pygame.font.SysFont("Arial", 42, bold=True)
        self.f_sub    = pygame.font.SysFont("Arial", 19)
        self.f_btn    = pygame.font.SysFont("Arial", 22, bold=True)
        self.f_detail = pygame.font.SysFont("Arial", 15)
        self.f_tag    = pygame.font.SysFont("Arial", 13, bold=True)

    def _cx(self, surf, y):
        self.screen.blit(surf, surf.get_rect(center=(SCREEN_W // 2, y)))

    def _bg(self):
        self.screen.fill(C["bg"])
        # Subtle top accent line
        pygame.draw.rect(self.screen, C["accent"], (0, 0, SCREEN_W, 3))

    def _title(self, text, sub=None, y_title=55):
        t = self.f_title.render(text, True, C["title"])
        self._cx(t, y_title)
        if sub:
            s = self.f_sub.render(sub, True, C["subtitle"])
            self._cx(s, y_title + 40)

    def _footer(self, text="ESC  back"):
        s = self.f_detail.render(text, True, C["dim"])
        self.screen.blit(s, (20, SCREEN_H - 26))

    def _cards(self, items, start_y=170, w=400, h=56, gap=10, tag_fn=None):
        """
        Draw card-style buttons.
        items: [(label, detail, value, enabled), ...]
        tag_fn: optional callable(value) -> str or None for a tag on the right
        Returns [(rect, value, enabled), ...]
        """
        mouse = pygame.mouse.get_pos()
        out = []
        x = (SCREEN_W - w) // 2
        for i, (label, detail, value, enabled) in enumerate(items):
            y = start_y + i * (h + gap)
            rect = pygame.Rect(x, y, w, h)
            hovered = rect.collidepoint(mouse) and enabled

            bg = C["card_hover"] if hovered else C["card"]
            if not enabled:
                bg = (30, 30, 34)
            pygame.draw.rect(self.screen, bg, rect, border_radius=10)
            pygame.draw.rect(self.screen, C["border"] if not hovered else C["accent"],
                             rect, 1, border_radius=10)

            # Label
            col = C["btn_text"] if enabled else C["dim"]
            lbl = self.f_btn.render(label, True, col)
            lbl_y = rect.centery - (10 if detail else 0)
            self.screen.blit(lbl, (x + 20, lbl_y - lbl.get_height() // 2))

            # Detail line
            if detail:
                det = self.f_detail.render(detail, True, C["subtitle"] if enabled else C["dim"])
                self.screen.blit(det, (x + 20, lbl_y + 12))

            # Tag on right
            if tag_fn:
                tag = tag_fn(value)
                if tag:
                    ts = self.f_tag.render(tag, True, C["accent"])
                    tr = ts.get_rect(midright=(rect.right - 16, rect.centery))
                    pad = pygame.Rect(tr.left - 8, tr.top - 3, tr.width + 16, tr.height + 6)
                    pygame.draw.rect(self.screen, C["tag_bg"], pad, border_radius=6)
                    self.screen.blit(ts, tr)

            out.append((rect, value, enabled))
        return out

    def _loop(self, draw_fn):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return "__quit__"
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return "__back__"
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    r = draw_fn(ev.pos)
                    if r is not None:
                        return r
            draw_fn(None)
            pygame.display.flip()
            self.clock.tick(60)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
class _ModeScreen(_Menu):
    def run(self):
        btns = []
        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Dots & Cuts", "Choose game mode")
            btns = self._cards([
                ("Player vs Player", "Local 1v1 on the same screen", "pvp", True),
                ("Player vs Bot", "Play against an AI opponent", "pvbot", True),
            ], start_y=190, h=64, gap=14)
            self._footer("ESC  quit")
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


class _BotTypeScreen(_Menu):
    """Select AI type: Minimax v1/v2 or RL v1/v2."""

    def __init__(self, screen, clock, has_rl_v1, has_rl_v2):
        super().__init__(screen, clock)
        self.has_v1 = has_rl_v1
        self.has_v2 = has_rl_v2

    def run(self):
        btns = []
        items = [
            ("Minimax v1", "Logistic regression evaluation", "minimax_v1", True),
            ("Minimax v2", "Improved weights from 100k games", "minimax_v2", True),
            ("RL v1", "Deep Q-Learning (basic)",
             "rl_v1", self.has_v1),
            ("RL v2", "Double DQN + reward shaping",
             "rl_v2", self.has_v2),
        ]

        def tag(v):
            if v == "rl_v1" and not self.has_v1:
                return "no checkpoints"
            if v == "rl_v2" and not self.has_v2:
                return "not trained yet"
            return None

        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Choose AI", "Select opponent type")
            btns = self._cards(items, start_y=170, h=62, gap=10, tag_fn=tag)
            self._footer()
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


class _DepthScreen(_Menu):
    def run(self):
        btns = []
        items = [
            ("Depth 1", "Instant — very weak", 1, True),
            ("Depth 2", "Fast — decent play", 2, True),
            ("Depth 3", "Medium — stronger, a few seconds", 3, True),
            ("Depth 4", "Slow — strong, may take a while", 4, True),
        ]
        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Search Depth", "Higher = stronger but slower")
            btns = self._cards(items, start_y=190, h=58, gap=10)
            self._footer()
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


class _RLTierScreen(_Menu):
    """
    Show 2-3 strength tiers for RL checkpoints.
    Picks early / mid / late checkpoint from the discovered set.
    """

    def __init__(self, screen, clock, checkpoints):
        super().__init__(screen, clock)
        self.checkpoints = checkpoints  # already sorted by episode
        self.tiers = self._build_tiers()

    def _build_tiers(self):
        """
        Build tier selection (weak/medium/strong) proportionally.
        Selects checkpoints at ~20%, ~50%, and 100% of max episode count,
        regardless of how many total checkpoints exist.
        """
        cks = self.checkpoints
        if not cks:
            return []

        def ep(path):
            return int(path.split("model_ep")[1].split(".pt")[0])

        episodes = sorted([ep(ck) for ck in cks])
        max_ep = episodes[-1]

        # Find closest checkpoint to target percentages
        weak_target = max_ep * 0.20    # ~20% of max
        medium_target = max_ep * 0.50  # ~50% of max
        strong_target = max_ep         # 100% (always the last)

        def find_closest(target):
            """Find checkpoint closest to target episode count."""
            return min(cks, key=lambda c: abs(ep(c) - target))

        tiers = []
        if len(cks) == 1:
            tiers.append(("Only available", f"ep {ep(cks[0])}", cks[0]))
        elif len(cks) == 2:
            weak_ck = find_closest(weak_target)
            strong_ck = find_closest(strong_target)
            tiers.append(("Weak", f"ep {ep(weak_ck)}", weak_ck))
            tiers.append(("Strong", f"ep {ep(strong_ck)}", strong_ck))
        else:
            weak_ck = find_closest(weak_target)
            medium_ck = find_closest(medium_target)
            strong_ck = find_closest(strong_target)
            tiers.append(("Weak", f"ep {ep(weak_ck)}", weak_ck))
            tiers.append(("Medium", f"ep {ep(medium_ck)}", medium_ck))
            tiers.append(("Strong", f"ep {ep(strong_ck)}", strong_ck))
        return tiers

    def run(self):
        btns = []
        items = []
        for label, detail, path in self.tiers:
            items.append((label, detail, path, True))
        if not items:
            items.append(("No checkpoints found", "", None, False))

        def tag(v):
            if v is None:
                return "missing"
            return None

        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Bot Strength", "Select training level")
            btns = self._cards(items, start_y=210, h=58, gap=12, tag_fn=tag)
            self._footer()
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


class _SideScreen(_Menu):
    def run(self):
        btns = []
        items = [
            ("Player 1", "Green — starts from the bottom", 1, True),
            ("Player 2", "Red — starts from the top", 2, True),
        ]
        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Your Side", "Choose which player you control")
            btns = self._cards(items, start_y=220, h=62, gap=14)
            self._footer()
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


class _MapScreen(_Menu):
    def run(self):
        btns = []
        items = [
            ("Standard", "Random towers & bunkers (9x9)", "standard", True),
            ("Balanced", "Symmetrical layout (9x9)", "balanced", True),
            ("Skirmish", "4 pieces each, random obstacles (9x9)", "skirmish", True),
            ("Mid 7x7", "3 pieces each, random obstacles (7x7)", "mid_7x7", True),
            ("Small 5x5", "Quick game on a smaller board", "small_5x5", True),
            ("Custom", "Build your own board", "custom", True),
        ]
        def draw(click):
            nonlocal btns
            self._bg()
            self._title("Board", "Choose the map layout")
            btns = self._cards(items, start_y=190, h=58, gap=10)
            self._footer()
            if click:
                for r, v, e in btns:
                    if r.collidepoint(click) and e:
                        return v
        return self._loop(draw)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def _discover_checkpoints(directory):
    pattern = os.path.join(directory, "model_ep*.pt")
    return sorted(glob.glob(pattern),
                  key=lambda f: int(f.split("model_ep")[1].split(".pt")[0]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
class ModeSelector:
    """Runs the full menu. Returns GameConfig or None if quit."""

    def __init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.ck_v1_dir = os.path.join(base, "..", "RL_approach", "checkpoints")
        self.ck_v2_dir = os.path.join(base, "..", "RL_approach", "checkpoints_v2")

    def run(self) -> Optional[GameConfig]:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Dots & Cuts")
        clock = pygame.time.Clock()
        config = GameConfig()

        ck_v1 = _discover_checkpoints(self.ck_v1_dir)
        ck_v2 = _discover_checkpoints(self.ck_v2_dir)

        state = "mode"
        while True:
            if state == "mode":
                r = _ModeScreen(screen, clock).run()
                if r == "__quit__":
                    return None
                config.mode = r
                state = "bot_type" if r == "pvbot" else "map"

            elif state == "bot_type":
                r = _BotTypeScreen(screen, clock,
                                   has_rl_v1=bool(ck_v1),
                                   has_rl_v2=bool(ck_v2)).run()
                if r == "__quit__":
                    return None
                if r == "__back__":
                    state = "mode"; continue
                config.bot_type = r
                state = "bot_config"

            elif state == "bot_config":
                if config.bot_type in ("minimax_v1", "minimax_v2"):
                    # Depth is adjustable in-game with D / Shift+D
                    config.minimax_depth = 3
                    state = "side"
                    continue
                else:
                    cks = ck_v1 if config.bot_type == "rl_v1" else ck_v2
                    r = _RLTierScreen(screen, clock, cks).run()
                    if r == "__quit__":
                        return None
                    if r == "__back__":
                        state = "bot_type"; continue
                    config.rl_checkpoint = r
                state = "side"

            elif state == "side":
                r = _SideScreen(screen, clock).run()
                if r == "__quit__":
                    return None
                if r == "__back__":
                    state = "bot_config" if config.mode == "pvbot" else "mode"
                    continue
                config.human_player = r
                state = "map"

            elif state == "map":
                r = _MapScreen(screen, clock).run()
                if r == "__quit__":
                    return None
                if r == "__back__":
                    state = "side" if config.mode == "pvbot" else "mode"
                    continue
                if r == "custom":
                    state = "builder"
                    continue
                config.map_name = r
                return config

            elif state == "builder":
                from board_builder import BoardBuilderScreen
                result = BoardBuilderScreen(screen, clock).run()
                if result == "__quit__":
                    return None
                if result == "__back__":
                    state = "map"; continue
                config.map_name = "custom"
                config.custom_game_state = result
                return config
