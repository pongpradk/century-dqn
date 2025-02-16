from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
    acquire_golem_card1 = 8  # New action: acquire golem card 1
    acquire_golem_card2 = 9  # Action to acquire golem card 2

class CenturyGolemEnvV2(gym.Env):
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
            "golem_cards_available": spaces.MultiBinary(2),   # For each golem card: 1 if available (i.e. not owned), 0 if acquired.
            "golem_cards_owned": spaces.MultiBinary(2)        # For each golem card: 1 if owned, 0 otherwise.
        })
        
        # Action space:
        # 0: rest, 1: play merchant card 1, 2: acquire merchant card 2, 3: play merchant card 2,
        # 4: acquire merchant card 3, 5: play merchant card 3, 6: acquire merchant card 4, 7: play merchant card 4
        self.action_space = spaces.Discrete(10)
        
        # Initialize player state
        self.yellow_crystals = 0
        self.green_crystals = 0
        
        # Initialize Merchant Cards:
        self.merchant_card1 = MerchantCard("merchant card 1", effect_yellow=2, effect_green=0, owned=True, available=True)
        self.merchant_card2 = MerchantCard("merchant card 2", effect_yellow=3, effect_green=0, owned=False, available=False)
        self.merchant_card3 = MerchantCard("merchant card 3", effect_yellow=2, effect_green=1, owned=False, available=False)
        self.merchant_card4 = MerchantCard("merchant card 4", effect_yellow=0, effect_green=2, owned=False, available=False)
        
        # Initialize Golem Cards:
        # Golem Card 1: costs 2 yellow and 3 green, worth 10 points.
        self.golem_card1 = GolemCard("golem card 1", cost_yellow=2, cost_green=3, points=10, owned=False)
        # Golem Card 2: costs 0 yellow and 4 green, worth 15 points.
        self.golem_card2 = GolemCard("golem card 2", cost_yellow=0, cost_green=4, points=15, owned=False)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_golem_state(self):
        if self.golem_card1.owned:
            return 1
        elif self.golem_card2.owned:
            return 2
        else:
            return 0
    
    def _get_obs(self):
        # Available golem cards: 1 if not owned, 0 if owned.
        golem_available = np.array([
            1 if not self.golem_card1.owned else 0,
            1 if not self.golem_card2.owned else 0,
        ], dtype=np.int32)
        
        # Owned golem cards: 1 if owned, 0 if not.
        golem_owned = np.array([
            1 if self.golem_card1.owned else 0,
            1 if self.golem_card2.owned else 0,
        ], dtype=np.int32)
        
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
            "golem_cards_available": golem_available,
            "golem_cards_owned": golem_owned
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
            "golem_cards_available": np.array([
                1 if not self.golem_card1.owned else 0,
                1 if not self.golem_card2.owned else 0,
            ], dtype=np.int32),
            "golem_cards_owned": np.array([
                1 if self.golem_card1.owned else 0,
                1 if self.golem_card2.owned else 0,
            ], dtype=np.int32)
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
        
        # Reset Golem Card: not owned
        self.golem_card1.owned = False
        self.golem_card2.owned = False
        
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
        
        elif action == Actions.acquire_golem_card1.value:
            if (self.yellow_crystals >= self.golem_card1.cost_yellow and 
                self.green_crystals >= self.golem_card1.cost_green and 
                not self.golem_card1.owned):
                self.yellow_crystals -= self.golem_card1.cost_yellow
                self.green_crystals -= self.golem_card1.cost_green
                self.golem_card1.owned = True
                reward += 20.0
            else:
                reward -= 1.0
                
        elif action == Actions.acquire_golem_card2.value:
            if (self.green_crystals >= self.golem_card2.cost_green and 
                not self.golem_card2.owned):
                # Note: Golem card 2 costs only green crystals.
                self.green_crystals -= self.golem_card2.cost_green
                self.golem_card2.owned = True
                reward += 21.7
            else:
                reward -= 1.0

        # Check winning condition
        if self._get_golem_state() != 0:
            terminated = True
            reward += 100.0  # Terminal reward
        
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
            
            # Compute available golem cards in the market (i.e. not owned)
            golem_available = []
            if not self.golem_card1.owned:
                golem_available.append("1")
            if not self.golem_card2.owned:
                golem_available.append("2")
            gm_str = ", ".join(golem_available) if golem_available else "None"
            
            # Determine which golem card is owned, if any
            if self.golem_card1.owned:
                go_owned = "1"
            elif self.golem_card2.owned:
                go_owned = "2"
            else:
                go_owned = "None"
            
            print(f"GM: {gm_str}")
            print(f"GO: {go_owned}")
            print("")      
    
    def close(self):
        print("Closing the Century Golem environment...")