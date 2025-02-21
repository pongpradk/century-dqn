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

class MerchantCard:
        
    def __init__(self, card_id, name, card_type, gain, cost=None, owned=False):
        self.card_id = card_id
        self.name = name
        self.card_type = card_type
        self.cost = cost
        self.gain = gain
        self.owned = owned

class GolemCard:
    def __init__(self, name, cost_yellow, cost_green, points, owned=False):
        self.name = name
        self.cost_yellow = cost_yellow  # Cost in yellow crystals to acquire
        self.cost_green = cost_green    # Cost in green crystals to acquire
        self.points = points            # Points associated with this card
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
            "merchant_market": spaces.MultiDiscrete([7, 7, 7, 7, 7]),
            "golem_cards_market": spaces.MultiDiscrete([5, 5, 5, 5, 5]),  # Current golem cards in market
            "golem_cards_owned_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)  # New: number of golem cards owned
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
        
        self.golem_cards = [
            GolemCard("Golem Card 1", cost_yellow=2, cost_green=2, points=6, owned=False),
            GolemCard("Golem Card 2", cost_yellow=3, cost_green=2, points=7, owned=False),
            GolemCard("Golem Card 3", cost_yellow=2, cost_green=3, points=8, owned=False),
            GolemCard("Golem Card 4", cost_yellow=0, cost_green=4, points=8, owned=False),
            GolemCard("Golem Card 5", cost_yellow=0, cost_green=5, points=10, owned=False)
        ]
        
        # initialize the market with 3 random golem cards
        self.market = random.sample(self.golem_cards, 5)
        
        self.player1 = Player(1)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def _get_obs(self):
        merchant_cards_state = np.array(self.player1.merchant_cards, dtype=np.int32)

        market_state = [card.card_id for card in self.merchant_market]
        while len(market_state) < 3:
            market_state.append(7)  # Fill empty slots with 7
        
        return {
            "player_yellow": np.array([self.player1.yellow], dtype=np.int32),
            "player_green": np.array([self.player1.green], dtype=np.int32),
            "merchant_cards": merchant_cards_state,
            "merchant_market": np.array(market_state, dtype=np.int32),
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
        return {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 5 # draw 3 cards to market, excluding M1
        )
        
        for card in self.golem_cards:
            card.owned = False
            
        self.market = random.sample(self.golem_cards, min(5, len(self.golem_cards)))
        
        self.player1.yellow = 0
        self.player1.green = 0
        
        self.player1.merchant_cards = [2] + [0] * 5
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "text":
            self.render()

        return observation, info
    
    def step(self, action):
        reward = -1.5  # Base time-step penalty
        terminated = False
        
        # Rest
        if action == Actions.rest.value:
            self.player1.merchant_cards = [2 if card == 1 else card for card in self.player1.merchant_cards]
            reward += 0.3
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
                
                reward+= 2.0
            else:
                reward = -1.0
        # Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM6.value:
            card_id, card_idx = action - 5, action - 6 # e.g. action 10 = use M5 = card_id 5 = card_idx 4
            if self.player1.merchant_cards[card_idx] == 2: # if card is playable
                self.player1.yellow += self.merchant_deck[card_id].gain['yellow']
                self.player1.green += self.merchant_deck[card_id].gain['green']
                self.player1.merchant_cards[card_idx] = 1 # set card status to owned but unplayable
                reward += 1.0
            else:
                reward -= 1.0
        
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
            total_crystals = self.player1.yellow + self.player1.green
            if total_crystals > 10:
                excess = total_crystals - 10
                yellow_lost = 0
                green_lost = 0

                # Remove excess starting with yellow, then green
                if self.player1.yellow >= excess:
                    yellow_lost = excess
                    self.player1.yellow -= excess
                else:
                    yellow_lost = self.player1.yellow
                    excess -= self.player1.yellow
                    self.player1.yellow = 0
                    green_lost = excess
                    self.player1.green = max(0, self.player1.green - excess)

                # Apply penalty for losing crystals
                penalty = - (0.5 * yellow_lost + 1.0 * green_lost)
                return penalty
            
            return 0  # No penalty if no excess crystals

        # Apply crystal limit before returning the observation
        # penalty = enforce_crystal_limit(self)
        # reward += penalty  # Apply the penalty to the step reward

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
            print(f"Y: {self.player1.yellow}")
            print(f"G: {self.player1.green}")                
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.player1.merchant_cards):
                if card_status == 0:
                    continue
                print(f"M{i+1}: {status_map[card_status]}")
            print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
            
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