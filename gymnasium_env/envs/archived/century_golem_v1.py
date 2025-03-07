from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class MerchantCard:
    def __init__(self, name, effect, owned=False, available=False):
        self.name = name
        self.effect = effect  # The number of crystals gained when played
        self.owned = owned
        self.available = available

class Actions(Enum):
    rest = 0
    play_merchant_card1 = 1
    acquire_merchant_card2 = 2
    play_merchant_card2 = 3

class CenturyGolemEnvV1(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        # Observation space includes crystals and status for both cards.
        self.observation_space = spaces.Dict({
            "crystals": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "merchant_card1_available": spaces.Discrete(2),  # 0: unavailable, 1: available
            "merchant_card2_owned": spaces.Discrete(2),        # 0: not owned, 1: owned
            "merchant_card2_available": spaces.Discrete(2),    # 0: unavailable, 1: available (if owned)
        })
        
        # Action space: 0: rest, 1: play merchant card 1, 2: acquire merchant card 2, 3: play merchant card 2
        self.action_space = spaces.Discrete(4)
        
        # Initialize player state
        self.crystals = 0
        
        # Merchant Card 1 is always owned and available from the start.
        self.merchant_card1 = MerchantCard("merchant card 1", effect=2, owned=True, available=True)
        # Merchant Card 2 is not owned initially.
        self.merchant_card2 = MerchantCard("merchant card 2", effect=3, owned=False, available=False)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Winning condition: Collect 10 crystals.
        self.winning_crystals = 10
    
    def _get_obs(self):
        return {
            "crystals": np.array([self.crystals], dtype=np.int32),
            "merchant_card1_available": int(self.merchant_card1.available),
            "merchant_card2_owned": int(self.merchant_card2.owned),
            "merchant_card2_available": int(self.merchant_card2.available),
        }
    
    def _get_info(self):
        return {
            "crystals": self.crystals,
            "merchant_card1_available": self.merchant_card1.available,
            "merchant_card2_owned": self.merchant_card2.owned,
            "merchant_card2_available": self.merchant_card2.available,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.crystals = 0
        # Reset card availability: Merchant Card 1 is always available at reset.
        self.merchant_card1.available = True
        # Merchant Card 2 resets to not owned.
        self.merchant_card2.owned = False
        self.merchant_card2.available = False
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "text":
            self.render()
            print("Game has been reset.")

        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = -1.0  # Base time-step penalty
        terminated = False
        
        if action == Actions.play_merchant_card1.value:
            if self.merchant_card1.available:
                self.crystals += self.merchant_card1.effect  # Add the card's effect
                self.merchant_card1.available = False
                reward += 3.0
            else:
                reward -= 1.0
        
        elif action == Actions.acquire_merchant_card2.value:
            if not self.merchant_card2.owned:
                self.merchant_card2.owned = True
                self.merchant_card2.available = True
                reward += 2.0
            else:
                reward -= 1.0
                
        elif action == Actions.play_merchant_card2.value:
            if self.merchant_card2.owned and self.merchant_card2.available:
                self.crystals += self.merchant_card2.effect  # Add card2's effect
                self.merchant_card2.available = False
                reward += 3.0
            else:
                reward -= 1.0
                
        elif action == Actions.rest.value:
            # Rest resets availability of all cards owned.
            self.merchant_card1.available = True
            if self.merchant_card2.owned:
                self.merchant_card2.available = True
            reward += 0.3
        
        # Check winning condition
        if self.crystals >= self.winning_crystals:
            terminated = True
            reward = 10.0  # Terminal reward
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "text":
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":
            card1_status = "playable" if self.merchant_card1.available else "unplayable"
            if self.merchant_card2.owned:
                card2_status = "playable" if self.merchant_card2.available else "unplayable"
            else:
                card2_status = "not owned"
            
            print(f"Crystals: {self.crystals} / {self.winning_crystals}")
            print(f"Merchant Card 1 - {card1_status}")
            print(f"Merchant Card 2 - {card2_status}")               
    
    def close(self):
        print("Closing the Century Golem environment...")