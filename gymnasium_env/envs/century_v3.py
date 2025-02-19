from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MerchantCard:
    def __init__(self, name, effect_yellow, effect_green, owned=False, available=False):
        self.name = name
        self.effect_yellow = effect_yellow  # Number of yellow crystals gained when played
        self.effect_green = effect_green    # Number of green crystals gained when played
        self.owned = owned
        self.available = available

class GolemCard:
    def __init__(self, name, cost_yellow, cost_green, points, owned=False):
        self.name = name
        self.cost_yellow = cost_yellow  # Cost in yellow crystals to acquire
        self.cost_green = cost_green    # Cost in green crystals to acquire
        self.points = points            # Points associated with this card
        self.owned = owned
        
class Actions(Enum):
    rest = 0
    play_merchant_card1 = 1
    acquire_merchant_card2 = 2
    play_merchant_card2 = 3
    acquire_merchant_card3 = 4
    play_merchant_card3 = 5
    acquire_merchant_card4 = 6
    play_merchant_card4 = 7
    acquire_golem_card1 = 8  
    acquire_golem_card2 = 9  
    acquire_golem_card3 = 10  
    acquire_golem_card4 = 11  
    acquire_golem_card5 = 12

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict({
            "yellow_crystals": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "green_crystals": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "merchant_card1_available": spaces.Discrete(2),
            "merchant_card2_owned": spaces.Discrete(2),
            "merchant_card2_available": spaces.Discrete(2),
            "merchant_card3_owned": spaces.Discrete(2),
            "merchant_card3_available": spaces.Discrete(2),
            "merchant_card4_owned": spaces.Discrete(2),
            "merchant_card4_available": spaces.Discrete(2),
            "golem_cards_market": spaces.MultiDiscrete([5, 5, 5, 5, 5]),  # Current golem cards in market
            "golem_cards_owned_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)  # New: number of golem cards owned
        })
        
        self.action_space = spaces.Discrete(13)
        
        # Initialize player state
        self.yellow_crystals = 0
        self.green_crystals = 0
        
        # Initialize Merchant Cards:
        self.merchant_card1 = MerchantCard("merchant card 1", effect_yellow=2, effect_green=0, owned=True, available=True)
        self.merchant_card2 = MerchantCard("merchant card 2", effect_yellow=3, effect_green=0, owned=False, available=False)
        # self.merchant_cardX = MerchantCard("merchant card 4", effect_yellow=4, effect_green=0, owned=False, available=False)
        # self.merchant_cardX = MerchantCard("merchant card 4", effect_yellow=1, effect_green=1, owned=False, available=False)
        self.merchant_card3 = MerchantCard("merchant card 3", effect_yellow=2, effect_green=1, owned=False, available=False)
        self.merchant_card4 = MerchantCard("merchant card 4", effect_yellow=0, effect_green=2, owned=False, available=False)
        
        self.golem_cards = [
            GolemCard("Golem Card 1", cost_yellow=2, cost_green=2, points=6, owned=False),
            GolemCard("Golem Card 2", cost_yellow=3, cost_green=2, points=7, owned=False),
            GolemCard("Golem Card 3", cost_yellow=2, cost_green=3, points=8, owned=False),
            GolemCard("Golem Card 4", cost_yellow=0, cost_green=4, points=8, owned=False),
            GolemCard("Golem Card 5", cost_yellow=0, cost_green=5, points=10, owned=False)
        ]
        
        # initialize the market with 3 random golem cards
        self.market = random.sample(self.golem_cards, 5)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
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
            "golem_cards_market": np.pad(
                np.array([self.golem_cards.index(card) for card in self.market], dtype=np.int32),
                (0, 5 - len(self.market)),  # Ensures 5 elements are always returned
                constant_values=-1  # Use -1 for missing values if fewer than 5 cards exist
            ),
            "golem_cards_owned_count": np.array(
                [sum(1 for card in self.golem_cards if card.owned)], dtype=np.int32
            )  # ✅ Now correctly returns owned golem card count
        }
    
    def _get_info(self):
        return {
            "yellow_crystals": self.yellow_crystals,
            "green_crystals": self.green_crystals,
            "golem_cards_market": [card.name for card in self.market],
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.yellow_crystals = 0
        self.green_crystals = 0
        
        self.merchant_card1.available = True
        self.merchant_card2.owned = False
        self.merchant_card2.available = False
        self.merchant_card3.owned = False
        self.merchant_card3.available = False
        self.merchant_card4.owned = False
        self.merchant_card4.available = False
        
        for card in self.golem_cards:
            card.owned = False
            
        self.market = random.sample(self.golem_cards, min(5, len(self.golem_cards)))
        
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
        reward = -1.5  # Base time-step penalty
        terminated = False
        
        if action == Actions.play_merchant_card1.value:
            if self.merchant_card1.available:
                self.yellow_crystals += self.merchant_card1.effect_yellow
                self.green_crystals += self.merchant_card1.effect_green
                self.merchant_card1.available = False
                reward += 1.0
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
                reward += 1.0
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
                reward += 1.0
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
                reward += 1.0
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
        
        # Handle acquiring golem cards from the market
        elif action in range(Actions.acquire_golem_card1.value, Actions.acquire_golem_card5.value + 1):
            golem_index = action - Actions.acquire_golem_card1.value
            selected_golem = self.golem_cards[golem_index]

            # Check if the selected golem is in the market before proceeding
            if (selected_golem in self.market and 
                self.yellow_crystals >= selected_golem.cost_yellow and 
                self.green_crystals >= selected_golem.cost_green and 
                not selected_golem.owned):

                self.yellow_crystals -= selected_golem.cost_yellow
                self.green_crystals -= selected_golem.cost_green
                selected_golem.owned = True
                reward += selected_golem.points

                # Remove from market and replace if possible
                self.market.remove(selected_golem)
                if len(self.market) < 5:
                    available_golems = [g for g in self.golem_cards if g not in self.market and not g.owned]
                    while len(self.market) < 5 and available_golems:
                        new_card = random.choice(available_golems)
                        self.market.append(new_card)
                        available_golems.remove(new_card)

            else:
                reward -= 1.0  # Penalize invalid action (trying to acquire a golem card not in market)

        # Enforce 10-crystal limit before moving to next step
        def enforce_crystal_limit(self):
            total_crystals = self.yellow_crystals + self.green_crystals
            if total_crystals > 10:
                excess = total_crystals - 10
                yellow_lost = 0
                green_lost = 0

                # Remove excess starting with yellow, then green
                if self.yellow_crystals >= excess:
                    yellow_lost = excess
                    self.yellow_crystals -= excess
                else:
                    yellow_lost = self.yellow_crystals
                    excess -= self.yellow_crystals
                    self.yellow_crystals = 0
                    green_lost = excess
                    self.green_crystals = max(0, self.green_crystals - excess)

                # Apply penalty for losing crystals
                penalty = - (0.5 * yellow_lost + 1.0 * green_lost)
                return penalty
            
            return 0  # No penalty if no excess crystals

        # Apply crystal limit before returning the observation
        penalty = enforce_crystal_limit(self)
        reward += penalty  # Apply the penalty to the step reward

        # Check if the player has acquired 2 golem cards (end condition)
        if sum(1 for card in self.golem_cards if card.owned) >= 2:
            terminated = True
            reward += 100.0  # Terminal reward for winning 
        
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
            print(f"Y: {self.yellow_crystals}")
            print(f"G: {self.green_crystals}")
            print(f"M1: {self._card_status(self.merchant_card1)}")
            print(f"M2: {self._card_status(self.merchant_card2)}")
            print(f"M3: {self._card_status(self.merchant_card3)}")
            print(f"M4: {self._card_status(self.merchant_card4)}")
            
            # Ensure 5 golem cards are always shown in market display
            if len(self.market) < 5:
                missing_slots = 5 - len(self.market)
                golem_available = [str(self.golem_cards.index(card) + 1) for card in self.market] + ["-"] * missing_slots
            else:
                golem_available = [str(self.golem_cards.index(card) + 1) for card in self.market]

            gm_str = ", ".join(golem_available)

            # Count the number of owned golem cards (instead of listing them)
            golem_cards_owned_count = sum(1 for card in self.golem_cards if card.owned)

            print(f"GM: {gm_str}")
            print(f"GO: {golem_cards_owned_count} owned")  # Updated to show count instead of specific cards
            print("")       
    
    def close(self):
        print("Closing the Century Golem environment...")