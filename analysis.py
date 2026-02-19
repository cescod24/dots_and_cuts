class Action:
    def __init__(self, piece_index, action_type, target_x, target_y):
        self.piece_index = piece_index
        self.action_type = action_type  # "move" or "shoot"
        self.target_x = target_x
        self.target_y = target_y