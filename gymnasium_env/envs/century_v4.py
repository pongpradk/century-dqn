from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Player:
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.yellow = 0
        self.green = 0
        # status of each merchant card for this player
        self.merchant_cards = [2] + [0] * 5 # 0 = not owned, 1 = owned but unplayable, 2 = owned and playable
        self.golem_count = 0
        self.points = 0

class MerchantCard:
        
    def __init__(self, card_id, name, card_type, gain, cost=None, owned=False):
        self.card_id = card_id
        self.name = name
        self.card_type = card_type
        self.cost = cost
        self.gain = gain
        self.owned = owned

class GolemCard:
    def __init__(self, card_id, name, cost, points, owned=False):
        self.card_id = card_id
        self.name = name
        self.cost = cost
        self.points = points
        self.owned = owned
        
class Actions(Enum):
    rest = 0
    getM2 = 1
    getM3 = 2
    getM4 = 3
    getM5 = 4
    getM6 = 5
    useM1 = 6
    useM2 = 7
    useM3 = 8
    useM4 = 9
    useM5 = 10
    useM6 = 11
    getG1 = 12
    getG2 = 13
    getG3 = 14
    getG4 = 15
    getG5 = 16

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict({
            "player_yellow": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "player_green": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "merchant_cards": spaces.MultiDiscrete([3] * 6),
            "merchant_market": spaces.MultiDiscrete([7, 7, 7, 7, 7]),  # Allow 7 as placeholder for empty
            "golem_market": spaces.MultiDiscrete([6, 6, 6, 6, 6]),  # Allow 5 as placeholder for empty
            "golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)  # New: number of golem cards owned
        })
        
        self.action_space = spaces.Discrete(16)
        
        self.merchant_deck = {
            1: MerchantCard(1, "Y2", "crystal", {"yellow": 2, "green": 0}, None, True),
            2: MerchantCard(2, "Y3", "crystal", {"yellow": 3, "green": 0}),
            3: MerchantCard(3, "Y4", "crystal", {"yellow": 4, "green": 0}),
            4: MerchantCard(4, "Y1G1", "crystal", {"yellow": 1, "green": 1}),
            5: MerchantCard(5, "Y2G1", "crystal", {"yellow": 2, "green": 1}),
            6: MerchantCard(6, "G2", "crystal", {"yellow": 0, "green": 2}),
        }

        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 5 # draw 3 cards to market, excluding M1
        )
        
        self.golem_deck = {
            1: GolemCard(1, "Y2G2", {"yellow": 2, "green": 2}, points=6),
            2: GolemCard(2, "Y3G2", {"yellow": 3, "green": 2}, points=7),
            3: GolemCard(3, "Y2G3", {"yellow": 3, "green": 2}, points=8),
            4: GolemCard(4, "G4", {"yellow": 0, "green": 4}, points=8),
            5: GolemCard(5, "G5", {"yellow": 0, "green": 5}, points=10)
        }
        
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        self.player1 = Player(1)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def _get_obs(self):
        merchant_cards_state = np.array(self.player1.merchant_cards, dtype=np.int32)

        merchant_market_state = [card.card_id for card in self.merchant_market]
        while len(merchant_market_state) < 5:
            merchant_market_state.append(6)
        
        golem_market_state = [card.card_id for card in self.golem_market]
        while len(golem_market_state) < 5:
            golem_market_state.append(5)
        
        return {
            "player_yellow": np.array([self.player1.yellow], dtype=np.int32),
            "player_green": np.array([self.player1.green], dtype=np.int32),
            "merchant_cards": merchant_cards_state,
            "merchant_market": np.array(merchant_market_state, dtype=np.int32),
            "golem_market": np.array(golem_market_state, dtype=np.int32),
            "golem_count": np.array([self.player1.golem_count], dtype=np.int32)
        }
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 5 # draw 3 cards to market, excluding M1
        )
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.golem_deck.values()]
        
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        self.player1.yellow = 0
        self.player1.green = 0
        self.player1.merchant_cards = [2] + [0] * 5
        self.player1.golem_count = 0
        self.player1.points = 0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "text":
            self.render()

        return observation, info
    
    def step(self, action):
        reward = -1.0  # Base time-step penalty
        terminated = False
        
        # Rest
        if action == Actions.rest.value:
            self.player1.merchant_cards = [2 if card == 1 else card for card in self.player1.merchant_cards]
            reward -= 0.5
            
        # Get a merchant card
        elif Actions.getM2.value <= action <= Actions.getM6.value:
            card_id = action + 1 # e.g. action 1 = get M2
            
            # Check if card is in market
            if self.merchant_deck[card_id] in self.merchant_market:
                # Execute action
                self.player1.merchant_cards[action] = 2

                self.merchant_deck[card_id].owned = True # set taken card to owned
                self.merchant_market.remove(self.merchant_deck[card_id]) # remove taken card from market
                # Find remaining cards in deck
                cards_in_deck = [
                    card for cid, card in self.merchant_deck.items()
                    if cid != 1 and not card.owned and card not in self.merchant_market
                ]
                # Draw random card from deck
                if cards_in_deck:
                    new_card = random.choice(cards_in_deck)
                    self.merchant_market.append(new_card)
                
                reward+= 5.0
            else:
                reward = -2.0
                
        # Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM6.value:
            card_id, card_idx = action - 5, action - 6 # e.g. action 10 = use M5 = card_id 5 = card_idx 4
            if self.player1.merchant_cards[card_idx] == 2: # if card is playable
                self.player1.yellow += self.merchant_deck[card_id].gain['yellow']
                self.player1.green += self.merchant_deck[card_id].gain['green']
                self.player1.merchant_cards[card_idx] = 1 # set card status to owned but unplayable
                
                # Give reward
                reward += (0.5 * self.merchant_deck[card_id].gain['yellow'] + 1.0 * self.merchant_deck[card_id].gain['green'])
            else:
                reward -= 2.0
        
        # Get a golem card
        elif Actions.getG1.value <= action <= Actions.getG5.value:
            card_id = action - Actions.getG1.value + 1 # e.g. action 12 = get G1
            # Check if golem in market
            if self.golem_deck[card_id] in self.golem_market:
                # Check if player has enough crystals
                if (self.player1.yellow >= self.golem_deck[card_id].cost["yellow"] and self.player1.green >= self.golem_deck[card_id].cost["green"]):
                    self.player1.yellow -= self.golem_deck[card_id].cost["yellow"]
                    self.player1.green -= self.golem_deck[card_id].cost["green"]
                    self.player1.golem_count += 1
                    self.player1.points += self.golem_deck[card_id].points
                    self.golem_deck[card_id].owned = True
                    
                    self.golem_market.remove(self.golem_deck[card_id]) # remove taken card from market
                    # Find remaining cards in deck
                    cards_in_deck = [
                        card for card in self.golem_deck.values()
                        if not card.owned and card not in self.golem_market
                    ]
                    # Draw random card from deck
                    if cards_in_deck:
                        new_card = random.choice(cards_in_deck)
                        self.golem_market.append(new_card)
                        
                    reward += 20 + self.golem_deck[card_id].points
                else:   
                    reward -= 2.0
            else:
                reward -= 2.0

        # Enforce 10-crystal limit before moving to next step
        def _enforce_crystal_limit(self):
            # total_crystals = self.player1.yellow + self.player1.green
            # if total_crystals > 10:
            #     excess = total_crystals - 10
            #     yellow_lost = 0
            #     green_lost = 0

            #     # Remove excess starting with yellow, then green
            #     if self.player1.yellow >= excess:
            #         yellow_lost = excess
            #         self.player1.yellow -= excess
            #     else:
            #         yellow_lost = self.player1.yellow
            #         excess -= self.player1.yellow
            #         self.player1.yellow = 0
            #         green_lost = excess
            #         self.player1.green = max(0, self.player1.green - excess)

            #     # Apply penalty for losing crystals
            #     # penalty = - (0.5 * yellow_lost + 1.0 * green_lost)
            #     return -1.0 * (yellow_lost + green_lost)
            
            # return 0  # No penalty if no excess crystals
            total_crystals = self.player1.yellow + self.player1.green
            if total_crystals > 10:
                excess = total_crystals - 10
                self.player1.yellow = max(0, self.player1.yellow - excess)
                self.player1.green = max(0, self.player1.green - (excess - self.player1.yellow))
                return -1.0 * excess  # Penalty per excess crystal
            return 0  # No penalty if no excess crystals

        # Apply crystal limit before returning the observation
        penalty = _enforce_crystal_limit(self)
        reward += penalty  # Apply the penalty to the step reward
            
        # Check for terminating condition
        if self.player1.golem_count >= 2:
            terminated = True
            reward += 100.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "text":
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":
            print(f"Y: {self.player1.yellow}")
            print(f"G: {self.player1.green}")                
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.player1.merchant_cards):
                if card_status == 0:
                    continue
                print(f"M{i+1}: {status_map[card_status]}")
            print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
            print("GM: " + " | ".join([f"G{g.card_id}-{g.name}-{g.points}" for g in self.golem_market]))
            print(f"GC: {self.player1.golem_count}")
            print(f"P: {self.player1.points}")
            print("")     
    
    def close(self):
        print("=== CLOSE ENVIRONMENT ===")