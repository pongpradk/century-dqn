import numpy as np

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def pick_action(self, state, info):
        valid_indices = np.where(info["valid_actions"] == 1)[0]
        return np.random.choice(valid_indices)