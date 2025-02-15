from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
    
class Actions(Enum):
    rest = 0
    crystal_card = 1
    #rest, crystal, trade, acquire

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        
        self.observation_space = spaces.Dict(
            {
                # crystals in possession between 0-10
                "crystals": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                # yellow acquire card availability
                "card_available": spaces.Discrete(2),
            }
        )
        
        # Action space: Rest (0) and Crystal_card (1)
        self.action_space = spaces.Discrete(2)
        
        # Initialize the player's state
        self.crystals = 0
        self.card_available = True  # The player starts with the card available
        
        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Winning condition: Collect 5 crystals
        self.winning_crystals = 10
    
    def _get_obs(self):
        # Return the current observation as a dictionary
        return {
            "crystals": np.array([self.crystals], dtype=np.int32),
            "card_available": int(self.card_available),
        }

    def _get_info(self):
        # Return any additional information (optional, can be empty).
        return {"crystals": self.crystals, "card_available": self.card_available}

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        super().reset(seed=seed)
        
        self.crystals = 0
        self.card_available = True
        
        observation = self._get_obs()
        info = self._get_info()

        # Render the initial state
        if self.render_mode == "text":
            self.render()
            print("Game has been reset.")

        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = -1.0
        terminated = False

        if action == Actions.crystal_card.value:  # Use the card
            if self.card_available:
                self.crystals += 2
                self.card_available = False  # The card is now unavailable
                reward += 3.0  # Reward for using the card successfully
            else:
                reward -= 1.0  # Penalty for trying to use the card when unavailable

        elif action == Actions.rest.value:  # Rest
            self.card_available = True  # Reset the card availability
            reward += 0.3 # prevent spamming rest

        # Check if the player has reached the winning condition
        if self.crystals >= self.winning_crystals:
            terminated = True
            reward = 10.0  # Reward for winning

        observation = self._get_obs()
        info = self._get_info()

        # Render the state if render_mode is "text"
        if self.render_mode == "text":
            self.render()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":
            if self.crystals >= self.winning_crystals:
                print("Victory")
            else:
                card_status = "available" if self.card_available else "unavailable"
                print(
                    f"Crystals: {self.crystals} / {self.winning_crystals}, Card: {card_status}"
                )

    def close(self):
        print("Closing the Century Golem environment...")