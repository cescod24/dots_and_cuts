"""
Move Notation for Dots & Cuts
==============================
Algebraic notation system for recording and displaying game moves.

Symbol:
    / = diagonal piece
    - = orthogonal piece

Coordinates: xy (column x, row y). Single digit for 0-9.
    For coords >= 10: a=10, b=11, c=12.

Format:
    <symbol>[rank][source][x]<target>[x][.]

    /33         diagonal moves to (3,3)
    -x23        orthogonal captures at (2,3)
    /x33x       diagonal captures at (3,3) and also dies (collapse)
    /2233       diagonal from (2,2) to (3,3) — disambiguated by source
    /22x33      diagonal from (2,2) captures at (3,3)
    -133        first orthogonal (by arrival order) moves to (3,3)
    -110x20x    first orth from (1,0) captures at (2,0), self dies
    -x50.       orthogonal captures at (5,0), game ends

Turn format:
    1. /33 -23       turn 1: P1 diagonal to 33, P2 orthogonal to 23
    1. ... -22       turn 1: only the second half-move shown
"""

from ai_core import generate_legal_actions, Action


# ---------------------------------------------------------------------------
# Coordinate encoding
# ---------------------------------------------------------------------------
def _coord_char(n):
    """Encode a single coordinate: 0-9 as digit, 10+ as a,b,c..."""
    if n < 10:
        return str(n)
    return chr(ord('a') + n - 10)


def _coord_str(x, y):
    """Compact coordinate pair string."""
    return _coord_char(x) + _coord_char(y)


def _piece_symbol(piece):
    """/ for diagonal, - for orthogonal."""
    return "/" if piece.kind == "diagonal" else "-"


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------
def _find_ambiguous_pieces(action, game_state):
    """
    Find other pieces of the same player and kind that could
    legally perform the same action type to the same target.
    Returns list of such pieces (excluding the acting piece).
    """
    piece = action.piece
    target = (action.target_x, action.target_y)
    at = action.action_type

    ambiguous = []
    for p in game_state.pieces:
        if p is piece:
            continue
        if p.player != piece.player or p.kind != piece.kind:
            continue
        for a in generate_legal_actions(game_state, p):
            if a.action_type == at and (a.target_x, a.target_y) == target:
                ambiguous.append(p)
                break
    return ambiguous


def _arrival_rank(piece, game_state):
    """
    Get 1-based arrival rank among same-kind, same-player pieces
    at the same vertex. Returns None if alone at vertex.
    """
    same_vertex = [
        p for p in game_state.pieces
        if p.x == piece.x and p.y == piece.y
        and p.kind == piece.kind and p.player == piece.player
    ]
    if len(same_vertex) <= 1:
        return None
    same_vertex.sort(key=lambda p: p.arrival_order)
    for i, p in enumerate(same_vertex):
        if p is piece:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def action_to_notation(action, game_state, capture_result=None, game_over=False):
    """
    Convert an Action to algebraic notation.

    Args:
        action:         Action object (piece, action_type, target_x, target_y)
        game_state:     GameState BEFORE the action is executed
        capture_result: dict {'captured': bool, 'self_died': bool} from history
                        If None, inferred from action_type and board state.
        game_over:      True if this move ends the game

    Returns: notation string (e.g. "/33", "-x23", "/2233")
    """
    piece = action.piece
    symbol = _piece_symbol(piece)
    tx, ty = action.target_x, action.target_y

    # Determine capture
    if capture_result is not None:
        is_capture = capture_result.get('captured', False)
        self_died = capture_result.get('self_died', False)
    else:
        # For bot suggestions (before execution): infer from action type
        is_capture = action.action_type == "shoot"
        # Also check if a move would land on enemies (vertex conflict)
        if not is_capture:
            is_capture = any(
                p for p in game_state.pieces
                if p.x == tx and p.y == ty and p.player != piece.player
            )
        self_died = False  # can't know before execution

    # Disambiguation
    ambiguous = _find_ambiguous_pieces(action, game_state)
    rank = _arrival_rank(piece, game_state)

    # Decide what disambiguation is needed
    need_source = False
    if ambiguous:
        # Check if all ambiguous pieces are at the same vertex as ours
        same_vertex_only = all(
            p.x == piece.x and p.y == piece.y for p in ambiguous
        )
        if same_vertex_only and rank is not None:
            need_source = False  # rank alone is sufficient
        else:
            need_source = True

    # Build notation string
    result = symbol
    if rank is not None:
        result += str(rank)
    if need_source:
        result += _coord_str(piece.x, piece.y)
    if is_capture:
        result += "x"
    result += _coord_str(tx, ty)
    if self_died:
        result += "x"
    if game_over:
        result += "."

    return result


def get_capture_result(history_entry):
    """
    Analyze a history entry to determine capture outcomes.

    Returns: {'captured': bool, 'self_died': bool}
    """
    removed = history_entry.get('removed', [])
    piece = history_entry['piece']
    captured = len(removed) > 0
    self_died = any(r[0] is piece for r in removed)
    return {'captured': captured, 'self_died': self_died}


def notation_after_execution(action, game_state):
    """
    Generate notation for an action that has JUST been executed.
    Uses the last history entry to determine capture results.
    Must be called immediately after execute_action().

    Args:
        action:     the Action that was just executed
        game_state: GameState AFTER execution (history[-1] is this action)

    Returns: notation string
    """
    if not game_state.history:
        return "?"
    entry = game_state.history[-1]
    cap = get_capture_result(entry)
    over, _ = game_state.is_game_over()

    # We need the state BEFORE execution for disambiguation.
    # Since the action is already executed, we reconstruct the needed info:
    # The piece's old position is in the history entry.
    old_x, old_y = entry['old_pos']
    old_arrival = entry['old_arrival']

    # Temporarily restore piece position for disambiguation check
    piece = action.piece
    cur_x, cur_y = piece.x, piece.y
    cur_arrival = piece.arrival_order
    piece.x, piece.y = old_x, old_y
    piece.arrival_order = old_arrival

    # Also temporarily restore captured pieces for proper disambiguation
    restored = []
    for p, px, py, pa in entry['removed']:
        if p is not piece and p not in game_state.pieces:
            p.x, p.y, p.arrival_order = px, py, pa
            game_state.pieces.append(p)
            restored.append(p)

    notation = action_to_notation(action, game_state, capture_result=cap, game_over=over)

    # Restore current state
    piece.x, piece.y = cur_x, cur_y
    piece.arrival_order = cur_arrival
    for p in restored:
        if p in game_state.pieces:
            game_state.pieces.remove(p)

    return notation


def format_turn(turn_num, p1_notation, p2_notation=None):
    """Format a complete turn: '1. /33 -23'."""
    if p1_notation and p2_notation:
        return f"{turn_num}. {p1_notation} {p2_notation}"
    elif p1_notation:
        return f"{turn_num}. {p1_notation}"
    elif p2_notation:
        return f"{turn_num}. ... {p2_notation}"
    return ""


def format_half_turn(turn_num, is_second_half, notation):
    """Format a single half-turn: '1. /33' or '1. ... -22'."""
    if is_second_half:
        return f"{turn_num}. ... {notation}"
    return f"{turn_num}. {notation}"
