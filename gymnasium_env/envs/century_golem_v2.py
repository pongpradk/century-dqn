from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class MerchantCard:
    def __init__(self, name, effect_yellow, effect_green, owned=False, available=False):
        self.name = name
        self.effect_yellow = effect_yellow  # Number of yellow crystals gained when played
        self.effect_green = effect_green    # Number of green crystals gained when played
        self.owned = owned
        self.available = available

class Actions(Enum):
    rest = 0
    play_merchant_card1 = 1
    acquire_merchant_card2 = 2
    play_merchant_card2 = 3
    acquire_merchant_card3 = 4
    play_merchant_card3 = 5
    acquire_merchant_card4 = 6
    play_merchant_card4 = 7

class CenturyGolemEnvV2(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        # Observation space now includes crystals and statuses for all cards.
        self.observation_space = spaces.Dict({
            "yellow_crystals": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "green_crystals": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "merchant_card1_available": spaces.Discrete(2),  # 0: unavailable, 1: available
            "merchant_card2_owned": spaces.Discrete(2),        # 0: not owned, 1: owned
            "merchant_card2_available": spaces.Discrete(2),    # 0: unavailable, 1: available
            "merchant_card3_owned": spaces.Discrete(2),
            "merchant_card3_available": spaces.Discrete(2),
            "merchant_card4_owned": spaces.Discrete(2),
            "merchant_card4_available": spaces.Discrete(2),
        })
        
        # Action space:
        # 0: rest, 1: play merchant card 1, 2: acquire merchant card 2, 3: play merchant card 2,
        # 4: acquire merchant card 3, 5: play merchant card 3, 6: acquire merchant card 4, 7: play merchant card 4
        self.action_space = spaces.Discrete(8)
        
        # Initialize player state
        self.yellow_crystals = 0
        self.green_crystals = 0
        
        # Merchant Card 1 is always owned and available at the start.
        # Its effect adds 2 yellow crystals (and 0 green).
        self.merchant_card1 = MerchantCard("merchant card 1", effect_yellow=2, effect_green=0, owned=True, available=True)
        
        # Merchant Card 2 is not owned initially.
        # Its effect adds 3 yellow crystals (and 0 green).
        self.merchant_card2 = MerchantCard("merchant card 2", effect_yellow=3, effect_green=0, owned=False, available=False)
        
        # New Merchant Card 3 is not owned initially.
        # Its effect: +2 yellow crystals and +1 green crystal.
        self.merchant_card3 = MerchantCard("merchant card 3", effect_yellow=2, effect_green=1, owned=False, available=False)
        
        # New Merchant Card 4 is not owned initially.
        # Its effect: +0 yellow crystals and +2 green crystals.
        self.merchant_card4 = MerchantCard("merchant card 4", effect_yellow=0, effect_green=2, owned=False, available=False)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Winning condition: Collect 10 crystals.
        self.winning_crystals = 10
    
    def _get_obs(self):
        return {
            "yellow_crystals": np.array([self.yellow_crystals], dtype=np.int32),
            "green_crystals": np.array([self.green_crystals], dtype=np.int32),
            "merchant_card1_available": int(self.merchant_card1.available),
            "merchant_card2_owned": int(self.merchant_card2.owned),
            "merchant_card2_available": int(self.merchant_card2.available),
            "merchant_card3_owned": int(self.merchant_card3.owned),
            "merchant_card3_available": int(self.merchant_card3.available),
            "merchant_card4_owned": int(self.merchant_card4.owned),
            "merchant_card4_available": int(self.merchant_card4.available),
        }
    
    def _get_info(self):
        return {
            "yellow_crystals": self.yellow_crystals,
            "green_crystals": self.green_crystals,
            "merchant_card1_available": self.merchant_card1.available,
            "merchant_card2_owned": self.merchant_card2.owned,
            "merchant_card2_available": self.merchant_card2.available,
            "merchant_card3_owned": self.merchant_card3.owned,
            "merchant_card3_available": self.merchant_card3.available,
            "merchant_card4_owned": self.merchant_card4.owned,
            "merchant_card4_available": self.merchant_card4.available,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.yellow_crystals = 0
        self.green_crystals = 0
        
        # Reset Merchant Card 1: always available.
        self.merchant_card1.available = True
        # Reset Merchant Card 2, 3, 4: not owned.
        self.merchant_card2.owned = False
        self.merchant_card2.available = False
        self.merchant_card3.owned = False
        self.merchant_card3.available = False
        self.merchant_card4.owned = False
        self.merchant_card4.available = False
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "text":
            self.render()
            print("====================")
            print("Game start ...")
            print("====================")
            print("")

        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = -1.0  # Base time-step penalty
        terminated = False
        
        if action == Actions.play_merchant_card1.value:
            if self.merchant_card1.available:
                self.yellow_crystals += self.merchant_card1.effect_yellow
                self.green_crystals += self.merchant_card1.effect_green
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
                self.yellow_crystals += self.merchant_card2.effect_yellow
                self.green_crystals += self.merchant_card2.effect_green
                self.merchant_card2.available = False
                reward += 3.0
            else:
                reward -= 1.0
        
        elif action == Actions.acquire_merchant_card3.value:
            if not self.merchant_card3.owned:
                self.merchant_card3.owned = True
                self.merchant_card3.available = True
                reward += 2.0
            else:
                reward -= 1.0
                
        elif action == Actions.play_merchant_card3.value:
            if self.merchant_card3.owned and self.merchant_card3.available:
                self.yellow_crystals += self.merchant_card3.effect_yellow
                self.green_crystals += self.merchant_card3.effect_green
                self.merchant_card3.available = False
                reward += 3.0
            else:
                reward -= 1.0
                
        elif action == Actions.acquire_merchant_card4.value:
            if not self.merchant_card4.owned:
                self.merchant_card4.owned = True
                self.merchant_card4.available = True
                reward += 2.0
            else:
                reward -= 1.0
                
        elif action == Actions.play_merchant_card4.value:
            if self.merchant_card4.owned and self.merchant_card4.available:
                self.yellow_crystals += self.merchant_card4.effect_yellow
                self.green_crystals += self.merchant_card4.effect_green
                self.merchant_card4.available = False
                reward += 3.0
            else:
                reward -= 1.0
                
        elif action == Actions.rest.value:
            # Rest resets the availability of any owned cards.
            self.merchant_card1.available = True
            if self.merchant_card2.owned:
                self.merchant_card2.available = True
            if self.merchant_card3.owned:
                self.merchant_card3.available = True
            if self.merchant_card4.owned:
                self.merchant_card4.available = True
            reward += 0.3

        # Check winning condition (based on yellow crystals).
        if self.yellow_crystals >= self.winning_crystals:
            terminated = True
            reward = 10.0  # Terminal reward
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "text":
            self.render()
        
        return observation, reward, terminated, False, info

    def _card_status(self, card):
        """Return the status of a merchant card as a string."""
        if card.owned:
            return "playable" if card.available else "unplayable"
        return "not owned"
    
    def render(self):
        if self.render_mode == "text":            
            print(f"Yellow Crystals: {self.yellow_crystals} / {self.winning_crystals}")
            print(f"Green Crystals: {self.green_crystals}")
            print(f"Merchant Card 1 - {self._card_status(self.merchant_card1)}")
            print(f"Merchant Card 2 - {self._card_status(self.merchant_card2)}")
            print(f"Merchant Card 3 - {self._card_status(self.merchant_card3)}")
            print(f"Merchant Card 4 - {self._card_status(self.merchant_card4)}")
            print("")           
    
    def close(self):
        print("Closing the Century Golem environment...")