import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import GAME_CONSTANTS
from .models import Player, MerchantCard, GolemCard
from .enums import Actions, CardStatus
from .utils import calculate_total_points, remove_excess_crystals

class InvalidActionError(Exception):
    pass

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 1}
    
    def create_merchant_deck(self):
        return {
            1: MerchantCard(1, "2Y", "crystal", gain={"yellow": 2, "green": 0}, owned=True),
            2: MerchantCard(2, "2U", "upgrade", gain=2, owned=True),
            3: MerchantCard(3, "3Y", "crystal", gain={"yellow": 3, "green": 0}),
            4: MerchantCard(4, "4Y", "crystal", gain={"yellow": 4, "green": 0}),
            5: MerchantCard(5, "1Y1G", "crystal", gain={"yellow": 1, "green": 1}),
            6: MerchantCard(6, "2Y1G", "crystal", gain={"yellow": 2, "green": 1}),
            7: MerchantCard(7, "2G", "crystal", gain={"yellow": 0, "green": 2}),
            8: MerchantCard(8, "1G:3Y", "trade", cost={"yellow": 0, "green": 1}, gain={"yellow": 3, "green": 0}),
            9: MerchantCard(9, "2Y:2G", "trade", cost={"yellow": 2, "green": 0}, gain={"yellow": 0, "green": 2}),
            10: MerchantCard(10, "3Y:3G", "trade", cost={"yellow": 3, "green": 0}, gain={"yellow": 0, "green": 3}),
            11: MerchantCard(11, "3U", "upgrade", gain=3),
            12: MerchantCard(12, "1B", "crystal", gain={"blue": 1}),
            13: MerchantCard(13, "1Y1B", "crystal", gain={"yellow": 1, "blue": 1}),
            14: MerchantCard(14, "2Y:1B", "trade", cost={"yellow": 2}, gain={"blue": 1}),
            15: MerchantCard(15, "3Y:1G1B", "trade", cost={"yellow": 3}, gain={"green": 1, "blue": 1}),
            16: MerchantCard(16, "4Y:2B", "trade", cost={"yellow": 4}, gain={"blue": 2}),
            17: MerchantCard(17, "5Y:3B", "trade", cost={"yellow": 5}, gain={"blue": 3}),
            18: MerchantCard(18, "2G:2B", "trade", cost={"green": 2}, gain={"blue": 2}),
            19: MerchantCard(19, "2G:3Y1B", "trade", cost={"green": 2}, gain={"yellow": 3, "blue": 1}),
            20: MerchantCard(20, "3G:2Y2B", "trade", cost={"green": 3}, gain={"yellow": 2, "blue": 2}),
            21: MerchantCard(21, "3G:3B", "trade", cost={"green": 3}, gain={"blue": 3}),
            22: MerchantCard(22, "1B:2G", "trade", cost={"blue": 1}, gain={"green": 2}),
            23: MerchantCard(23, "1B:4Y1G", "trade", cost={"blue": 1}, gain={"yellow": 4, "green": 1}),
            24: MerchantCard(24, "1B:1Y2G", "trade", cost={"blue": 1}, gain={"yellow": 1, "green": 2}),
            25: MerchantCard(25, "2B:2Y3G", "trade", cost={"blue": 2}, gain={"yellow": 2, "green": 3}),
        }
    
    def create_golem_deck(self):
        return {
            1: GolemCard(1, "2Y2G", {"yellow": 2, "green": 2}, points=6),
            2: GolemCard(2, "3Y2G", {"yellow": 3, "green": 2}, points=7),
            3: GolemCard(3, "2Y3G", {"yellow": 2, "green": 3}, points=8),
            4: GolemCard(4, "4G", {"yellow": 0, "green": 4}, points=8),
            5: GolemCard(5, "5G", {"yellow": 0, "green": 5}, points=10),
            6: GolemCard(6, "2Y2B", {"yellow": 2, "blue": 2}, points=8),
            7: GolemCard(7, "3Y2B", {"yellow": 3, "blue": 2}, points=9),
            8: GolemCard(8, "2G2B", {"green": 2, "blue": 2}, points=10),
            9: GolemCard(9, "2Y3B", {"yellow": 2, "blue": 3}, points=11),
            10: GolemCard(10, "3G2B", {"green": 3, "blue": 2}, points=12),
            11: GolemCard(11, "4B", {"blue": 4}, points=12),
            12: GolemCard(12, "2Y2G2B", {"yellow": 2, "green": 2, "blue": 2}, points=13),
            13: GolemCard(13, "2G3B", {"green": 2, "blue": 3}, points=13),
            14: GolemCard(14, "5B", {"blue": 5}, points=15)
        }
    
    def _draw_merchant_card(self):
        cards_in_deck = [
            card for cid, card in self.merchant_deck.items()
            if cid != 1 and cid != 2 and not card.owned and card not in self.merchant_market
        ]
        if cards_in_deck:
            new_card = random.choice(cards_in_deck)
            self.merchant_market.append(new_card)
    
    def _draw_golem_card(self):
        cards_in_deck = [
            card for card in self.golem_deck.values()
            if not card.owned and card not in self.golem_market
        ]
        if cards_in_deck:
            new_card = random.choice(cards_in_deck)
            self.golem_market.append(new_card)
    
    def _handle_rest(self):
        self.current_player.merchant_cards = [CardStatus.PLAYABLE.value if card == CardStatus.UNPLAYABLE.value else card for card in self.current_player.merchant_cards]
        return GAME_CONSTANTS['REWARDS']['REST']
    
    def _handle_get_merchant_card(self, action):
        card_id = action + 2  # e.g. M3 = action 1 + 2
        
        if self.merchant_deck[card_id] in self.merchant_market:
            self.current_player.merchant_cards[card_id - 1] = CardStatus.PLAYABLE.value
            self.merchant_deck[card_id].owned = True
            self.merchant_market.remove(self.merchant_deck[card_id])
            self._draw_merchant_card()
            
            # Progressive discount based on number of cards already owned
            owned_count = sum(1 for status in self.current_player.merchant_cards if status > 0)
            card_value = max(0.5, GAME_CONSTANTS['REWARDS']['GET_MERCHANT_CARD'] * (1 - 0.15 * (owned_count - 2)))
            
            return card_value

        raise InvalidActionError(f"Card M{card_id} is not in market")

    def _handle_crystal_card(self, card, card_index):
        for crystal, amount in card.gain.items():
            self.current_player.caravan[crystal] += amount
        self.current_player.merchant_cards[card_index] = CardStatus.UNPLAYABLE.value
        return (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.gain.get('yellow', 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.gain.get('green', 0) + GAME_CONSTANTS['CRYSTAL_VALUES']['blue'] * card.gain.get('blue', 0))

    def _handle_trade_card(self, card, card_index): 
        # Check if player has enough crystals
        if all(self.current_player.caravan[crystal] >= amount 
                for crystal, amount in card.cost.items()):
            # Apply the trade
            for crystal, amount in card.cost.items():
                self.current_player.caravan[crystal] -= amount
            for crystal, amount in card.gain.items():
                self.current_player.caravan[crystal] += amount

            self.current_player.merchant_cards[card_index] = CardStatus.UNPLAYABLE.value
            loss = (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.cost.get("yellow", 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.cost.get("green", 0) + GAME_CONSTANTS['CRYSTAL_VALUES']['blue'] * card.cost.get("blue", 0))
            gain = (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.gain.get("yellow", 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.gain.get("green", 0) + GAME_CONSTANTS['CRYSTAL_VALUES']['blue'] * card.gain.get("blue", 0))
            return (gain - loss)
        
        raise InvalidActionError(f"Not enough crystals for trade card {card.name}")

    def _handle_upgrade_card(self, card, card_index):
        upgrade_points = card.gain
        caravan = self.current_player.caravan
        upgrade_steps_taken = 0 # Track actual upgrade steps

        # Step 1: While we still have upgrade points, upgrade lowest-value crystals upward greedily
        while upgrade_points > 0:
            # Try upgrading yellow -> green -> blue (2 upgrades total)
            if caravan["yellow"] > 0 and upgrade_points >= 2:
                caravan["yellow"] -= 1
                caravan["blue"] += 1
                upgrade_points -= 2
                upgrade_steps_taken += 2 # 2 steps
            # If only 1 upgrade point, upgrade yellow -> green
            elif caravan["yellow"] > 0 and upgrade_points >= 1:
                caravan["yellow"] -= 1
                caravan["green"] += 1
                upgrade_points -= 1
                upgrade_steps_taken += 1 # 1 step
            # Upgrade green -> blue (1 upgrade)
            elif caravan["green"] > 0 and upgrade_points >= 1:
                caravan["green"] -= 1
                caravan["blue"] += 1
                upgrade_points -= 1
                upgrade_steps_taken += 1 # 1 step
            else:
                # No more crystals to upgrade or no upgrade points left
                break

        if upgrade_steps_taken == 0:
            raise InvalidActionError("No crystals can be upgraded")

        self.current_player.merchant_cards[card_index] = CardStatus.UNPLAYABLE.value
        # Reward based on number of upgrade steps performed
        return upgrade_steps_taken * 0.5 # Assign a value per step, e.g., 0.5

    def _handle_use_merchant_card(self, action):
        card_id = action - Actions.useM1.value + 1
        card = self.merchant_deck.get(card_id)
        card_index = card_id - 1

        if self.current_player.merchant_cards[card_index] == CardStatus.PLAYABLE.value and card:
            if card.card_type == "crystal":
                reward = self._handle_crystal_card(card, card_index)
            elif card.card_type == "trade":
                reward = self._handle_trade_card(card, card_index)
            elif card.card_type == "upgrade":
                reward = self._handle_upgrade_card(card, card_index)
        
        # If the player has both yellow and green crystals after using the card
        # and is close to but not over the crystal limit
        total_crystals = self.current_player.caravan["yellow"] + self.current_player.caravan["green"] + self.current_player.caravan["blue"]
        if (self.current_player.caravan["yellow"] > 0 and 
            self.current_player.caravan["green"] > 0 and
            total_crystals >= 5 and total_crystals <= GAME_CONSTANTS['MAX_CRYSTALS']):
            return reward + GAME_CONSTANTS['REWARDS']['CRYSTAL_MANAGEMENT']
        
        return reward
    
    def _handle_get_golem_card(self, action):
        card_id = action - Actions.getG1.value + 1 # e.g. G1 = 19 - 19 + 1
        golem_card = self.golem_deck[card_id]
        
        if (golem_card in self.golem_market and 
            all(self.current_player.caravan[crystal] >= amount 
                for crystal, amount in golem_card.cost.items())):
            # Execute action
            for crystal, amount in golem_card.cost.items():
                self.current_player.caravan[crystal] -= amount
            self.current_player.golem_count += 1
            self.current_player.points += golem_card.points
            golem_card.owned = True
            
            self.golem_market.remove(golem_card)
            self._draw_golem_card()
            
            # Add bonus if this brings player closer to winning
            win_proximity_bonus = 0
            if self.current_player.golem_count == 3:
                win_proximity_bonus = GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY']
            elif self.current_player.golem_count == 4:
                win_proximity_bonus = GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY'] * 2
            elif self.current_player.golem_count == 5:
                win_proximity_bonus = GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY'] * 3

            return GAME_CONSTANTS['REWARDS']['GOLEM_BASE_REWARD'] + golem_card.points + win_proximity_bonus
    
    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(63)
        
        # Open information
        crystal_types = ["yellow", "green", "blue"]  # Add more types here as needed
        
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([26, 26, 26, 26, 26, 26]),
            "golem_market": spaces.MultiDiscrete([15, 15, 15, 15, 15]),
            
            # Replace individual crystal spaces with a single Dict space
            "player1_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            "player1_merchant_cards": spaces.MultiDiscrete([3] * 25),
            "player1_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "player1_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            
            "player2_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            "player2_merchant_cards": spaces.MultiDiscrete([3] * 25),
            "player2_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "player2_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        })
        
        self.merchant_deck = self.create_merchant_deck()
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1 and cid != 2], 6
        )
        
        self.golem_deck = self.create_golem_deck()
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        # Initialize players
        self.player1, self.player2 = Player(1), Player(2)
        self.current_player, self.next_player = None, None
        
        self.endgame_triggered = False  # Tracks if endgame was triggered
        self.endgame_initiator = None   # The player who triggered the endgame
        self.starting_player = None     # Player who starts the game
        self.round_number = 0           # Current round number
        
        # Render
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.winner = None
        self.player1_final_points = 0  # Stores final DQN points
        self.player2_final_points = 0  # Stores final Random points
    
    def _get_obs(self, player):
        # Returns observation from the perspective of the player
        player1 = player
        player2 = self.player2 if player == self.player1 else self.player1
        
        # Ensure merchant cards are within [0, 2] range
        merchant_cards_state = np.clip(np.array(player1.merchant_cards, dtype=np.int32), 0, 2)

        # Ensure market states are within their bounds
        merchant_market_state = [card.card_id for card in self.merchant_market]
        while len(merchant_market_state) < 6:
            merchant_market_state.append(0) # Use 0 for padding
        
        golem_market_state = [card.card_id for card in self.golem_market]
        while len(golem_market_state) < 5:
            golem_market_state.append(0) # Use 0 for padding
        
        # Ensure caravan values are within [0, 20] range
        player1_caravan = {
            crystal: np.clip(np.array([amount], dtype=np.int32), 0, 20)
            for crystal, amount in player1.caravan.items()
        }
        
        player2_caravan = {
            crystal: np.clip(np.array([amount], dtype=np.int32), 0, 20)
            for crystal, amount in player2.caravan.items()
        }
        
        # Ensure golem count is within [0, 5] range
        player1_golem_count = np.clip(np.array([player1.golem_count], dtype=np.int32), 0, 5)
        player2_golem_count = np.clip(np.array([player2.golem_count], dtype=np.int32), 0, 5)
        
        # Ensure points are within [0, 100] range
        player1_points = np.clip(np.array([player1.points], dtype=np.int32), 0, 100)
        player2_points = np.clip(np.array([player2.points], dtype=np.int32), 0, 100)
        
        return {
            "merchant_market": np.array(merchant_market_state, dtype=np.int32),
            "golem_market": np.array(golem_market_state, dtype=np.int32),
            
            "player1_caravan": player1_caravan,
            "player1_merchant_cards": merchant_cards_state,
            "player1_golem_count": player1_golem_count,
            "player1_points": player1_points,
            
            "player2_caravan": player2_caravan,
            "player2_merchant_cards": np.clip(np.array(player2.merchant_cards, dtype=np.int32), 0, 2),
            "player2_golem_count": player2_golem_count,
            "player2_points": player2_points,
        }

    def _get_info(self):
        return {
            "valid_actions": self._get_valid_actions(self.current_player),
            "current_player": int(self.current_player.player_id - 1),  # 0 for player1, 1 for player2
            "winner": self.winner if self.winner is not None else None
        }
    
    def reset(self, seed=None, options=None):
        if self.render_mode == "text":
            print("Century: Golem Edition | Version 14\n")
        
        super().reset(seed=seed)
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1 and cid != 2], 6
        )
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.golem_deck.values()]
        
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        self.current_player = random.choice([self.player1, self.player2])
        self.next_player = self.player1 if self.current_player == self.player2 else self.player2
        
        for player in (self.player1, self.player2):
            player.caravan["green"] = 0
            player.caravan["blue"] = 0
            player.merchant_cards = [2, 2] + [0] * 23
            player.golem_count = 0
            player.points = 0
            
        self.current_player.caravan["yellow"] = 3
        self.next_player.caravan["yellow"] = 4
            
        self.endgame_triggered = False  # Tracks if endgame was triggered
        self.endgame_initiator = None   # The player who triggered the endgame
        self.starting_player = self.current_player # Track who starts
        self.round_number = 1           # Reset round counter
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()

        if self.render_mode != None:
            self.render()

        return observation, info
    
    def _get_valid_actions(self, player):
        valid_actions = np.zeros(self.action_space.n, dtype=np.int32)

        valid_actions[Actions.rest.value] = 0
        # Rest is valid if any merchant card is unplayable
        for i in range(Actions.useM1.value, Actions.useM25.value + 1):
            card_index = i - Actions.useM1.value
            if player.merchant_cards[card_index] == CardStatus.UNPLAYABLE.value:  # If owned but unplayable
                valid_actions[Actions.rest.value] = 1
                break

        # Get merchant card actions
        for i in range(Actions.getM3.value, Actions.getM25.value + 1):
            card_id = i + 2
            if self.merchant_deck[card_id] in self.merchant_market:
                valid_actions[i] = 1

        # Use merchant card actions
        for i in range(Actions.useM1.value, Actions.useM25.value + 1):
            card_index = i - Actions.useM1.value

            if player.merchant_cards[card_index] == CardStatus.PLAYABLE.value:
                card = self.merchant_deck.get(card_index + 1)
                if card.card_type == "crystal":
                    valid_actions[i] = 1  # Always valid for crystal-gaining cards
                elif card.card_type == "trade":
                    # Ensure the player has enough crystals to trade
                    if all(player.caravan[crystal] >= amount 
                          for crystal, amount in card.cost.items()):
                        valid_actions[i] = 1
                elif card.card_type == "upgrade":
                    if player.caravan["yellow"] > 0 or player.caravan["green"] > 0:
                        valid_actions[i] = 1

        # Get golem card actions
        for i in range(Actions.getG1.value, Actions.getG14.value + 1):
            card_id = i - Actions.getG1.value + 1
            golem_card = self.golem_deck.get(card_id)
            if (golem_card in self.golem_market and
                all(player.caravan[crystal] >= amount 
                    for crystal, amount in golem_card.cost.items())):
                    valid_actions[i] = 1

        return valid_actions    

    def step(self, action):
        if self.render_mode == "text":
            print(f"==== DQN | {Actions(int(action)).name} ====\n") if self.current_player.player_id == 1 else print(f"==== Random | {Actions(int(action)).name} ====\n")
            
        terminated = False
        reward = GAME_CONSTANTS['REWARDS']['STEP']
        
        # Action: Rest
        if action == Actions.rest.value:
            reward += self._handle_rest()
        # Action: Get a merchant card
        elif Actions.getM3.value <= action <= Actions.getM25.value:
            reward += self._handle_get_merchant_card(action)
        # Action: Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM25.value:
            reward += self._handle_use_merchant_card(action)
        # Action: Get a golem card
        elif Actions.getG1.value <= action <= Actions.getG14.value:
            reward += self._handle_get_golem_card(action)

        reward -= remove_excess_crystals(self.current_player)
        
        # Check if the endgame is triggered
        if not self.endgame_triggered and self.current_player.golem_count >= GAME_CONSTANTS['GOLEM_ENDGAME_THRESHOLD']:
            self.endgame_triggered = True
            self.endgame_initiator = self.current_player
        # Check if this is the last turn (player2 of endgame initiator)
        if self.endgame_triggered and self.current_player != self.endgame_initiator:
            terminated = True

            # Calculate final points for both players
            self.player1_final_points = calculate_total_points(self.player1)
            self.player2_final_points = calculate_total_points(self.player2)
            
            # If not player1's turn, the reward is reset, before calculation
            if self.current_player != self.player1:
                reward = 0
            
            score_diff = self.player1_final_points - self.player2_final_points
            if score_diff > 0:
                self.winner = "P1"
                reward += GAME_CONSTANTS['REWARDS']['WIN']
            elif score_diff == 0:
                self.winner = "P0"
                reward += GAME_CONSTANTS['REWARDS']['TIE']
            else:
                self.winner = "P2"
                reward -= GAME_CONSTANTS['REWARDS']['WIN']
        
        # Store player who just finished turn before switching
        player_finished_turn = self.current_player
        
        # Switch turn
        self.current_player, self.next_player = self.next_player, self.current_player
        
        # Increment round if the second player just finished their turn
        if player_finished_turn != self.starting_player:
            self.round_number += 1
            
        observation = self._get_obs(self.current_player)
        info = self._get_info()
        
        if self.render_mode != None:
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":               
            print(f"Round: {self.round_number}\n") # Display current round
            print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
            print("GM: " + " | ".join([f"G{g.card_id}-{g.name}-{g.points}" for g in self.golem_market]))
            print("")
            print(f"DQN Agent")
            print(f"Y:{self.player1.caravan['yellow']} | G:{self.player1.caravan['green']} | B:{self.player1.caravan['blue']}")
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.player1.merchant_cards):
                if card_status == 0:
                    continue
                card = self.merchant_deck.get(i + 1)  # Assuming card_id starts from 1
                print(f"M{i+1}-{card.name}: {status_map[card_status]}")
            print(f"GC: {self.player1.golem_count}")
            print(f"P: {self.player1.points}")
            print("")
            print(f"P{self.player2.player_id}")
            print(f"Y:{self.player2.caravan['yellow']} | G:{self.player2.caravan['green']} | B:{self.player2.caravan['blue']}")
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.player2.merchant_cards):
                if card_status == 0:
                    continue
                card = self.merchant_deck.get(i + 1)  # Assuming card_id starts from 1
                print(f"M{i+1}-{card.name}: {status_map[card_status]}")
            print(f"GC: {self.player2.golem_count}")
            print(f"P: {self.player2.points}")
            print("")
            
    def close(self):
        if self.render_mode == "text":
            print("=== CLOSE ENVIRONMENT ===") 