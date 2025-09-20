import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import GAME_CONSTANTS
from .models import Player, MerchantCard, GolemCard
from .enums import Actions, CardStatus, MerchantCardStatus
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
            26: MerchantCard(26, "3Y:1P", "trade", cost={"yellow": 3}, gain={"pink": 1}),
            27: MerchantCard(27, "4Y:1B1P", "trade", cost={"yellow": 4}, gain={"blue": 1, "pink": 1}),
            28: MerchantCard(28, "5Y:2P", "trade", cost={"yellow": 5}, gain={"pink": 2}),
            29: MerchantCard(29, "2G:2Y1P", "trade", cost={"green": 2}, gain={"yellow": 2, "pink": 1}),
            30: MerchantCard(30, "3G:1Y1B1P", "trade", cost={"green": 3}, gain={"yellow": 1, "blue": 1, "pink": 1}),
            31: MerchantCard(31, "3G:2P", "trade", cost={"green": 3}, gain={"pink": 2}),
            32: MerchantCard(32, "1Y1G:1P", "trade", cost={"yellow": 1, "green": 1}, gain={"pink": 1}),
            33: MerchantCard(33, "2B:2P", "trade", cost={"blue": 2}, gain={"pink": 2}),
            34: MerchantCard(34, "2B:2G1P", "trade", cost={"blue": 2}, gain={"green": 2, "pink": 1}),
            35: MerchantCard(35, "2B:2Y1G1P", "trade", cost={"blue": 2}, gain={"yellow": 2, "green": 1, "pink": 1}),
            36: MerchantCard(36, "3B:3P", "trade", cost={"blue": 3}, gain={"pink": 3}),
            37: MerchantCard(37, "2Y1B:2P", "trade", cost={"yellow": 2, "blue": 1}, gain={"pink": 2}),
            38: MerchantCard(38, "1P:2B", "trade", cost={"pink": 1}, gain={"blue": 2}),
            39: MerchantCard(39, "1P:3G", "trade", cost={"pink": 1}, gain={"green": 3}),
            40: MerchantCard(40, "1P:1Y1G1B", "trade", cost={"pink": 1}, gain={"yellow": 1, "green": 1, "blue": 1}),
            41: MerchantCard(41, "1P:2Y2G", "trade", cost={"pink": 1}, gain={"yellow": 2, "green": 2}),
            42: MerchantCard(42, "1P:3Y1B", "trade", cost={"pink": 1}, gain={"yellow": 3, "blue": 1}),
            43: MerchantCard(43, "2P:1Y1G3B", "trade", cost={"pink": 2}, gain={"yellow": 1, "green": 1, "blue": 3}),
            44: MerchantCard(44, "2P:3G2B", "trade", cost={"pink": 2}, gain={"green": 3, "blue": 2}),
            45: MerchantCard(45, "1P", "crystal", gain={"pink": 1}),
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
            14: GolemCard(14, "5B", {"blue": 5}, points=15),
            15: GolemCard(15, "2Y1G1P", {"yellow": 2, "green": 1, "pink": 1}, points=9),
            16: GolemCard(16, "2Y2P", {"yellow": 2, "pink": 2}, points=10),
            17: GolemCard(17, "3Y2P", {"yellow": 3, "pink": 2}, points=11),
            18: GolemCard(18, "2G2P", {"green": 2, "pink": 2}, points=12),
            19: GolemCard(19, "1Y1G1B1P", {"yellow": 1, "green": 1, "blue": 1, "pink": 1}, points=12),
            20: GolemCard(20, "1Y2B1P", {"yellow": 1, "blue": 2, "pink": 1}, points=12),
            21: GolemCard(21, "2G1B1P", {"green": 2, "blue": 1, "pink": 1}, points=12),
            22: GolemCard(22, "3G2P", {"green": 3, "pink": 2}, points=14),
            23: GolemCard(23, "2B2P", {"blue": 2, "pink": 2}, points=14),
            24: GolemCard(24, "3Y1G1B1P", {"yellow": 3, "green": 1, "blue": 1, "pink": 1}, points=14),
            25: GolemCard(25, "2Y3P", {"yellow": 2, "pink": 3}, points=14),
            26: GolemCard(26, "2Y2G2P", {"yellow": 2, "green": 2, "pink": 2}, points=15),
            27: GolemCard(27, "4P", {"pink": 4}, points=16),
            28: GolemCard(28, "2G3P", {"green": 2, "pink": 3}, points=16),
            29: GolemCard(29, "1Y3G1B1P", {"yellow": 1, "green": 3, "blue": 1, "pink": 1}, points=16),
            30: GolemCard(30, "3B2P", {"blue": 3, "pink": 2}, points=17),
            31: GolemCard(31, "2Y2B2P", {"yellow": 2, "blue": 2, "pink": 2}, points=17),
            32: GolemCard(32, "1Y1G3B1P", {"yellow": 1, "green": 1, "blue": 3, "pink": 1}, points=18),
            33: GolemCard(33, "2B3P", {"blue": 2, "pink": 3}, points=18),
            34: GolemCard(34, "2G2B2P", {"green": 2, "blue": 2, "pink": 2}, points=19),
            35: GolemCard(35, "1Y1G1B3P", {"yellow": 1, "green": 1, "blue": 1, "pink": 3}, points=20),
            36: GolemCard(36, "5P", {"pink": 5}, points=20)
        }
    
    def _draw_merchant_card(self):
        # Pick a random unowned card that's not in market
        cards_in_deck = [
            card for cid, card in self.merchant_deck.items()
            if cid != 1 and cid != 2 and not card.owned and card not in self.merchant_market
        ]
        if cards_in_deck:
            new_card = random.choice(cards_in_deck)
            self.merchant_market.append(new_card)
    
    def _draw_golem_card(self):
        # Pick a random unowned card that's not in market
        cards_in_deck = [
            card for card in self.golem_deck.values()
            if not card.owned and card not in self.golem_market
        ]
        if cards_in_deck:
            new_card = random.choice(cards_in_deck)
            self.golem_market.append(new_card)
    
    def _get_card_status_index(self, card_id, player_id):
        if card_id == 1:  # M1
            return 0 if player_id == 1 else 2
        elif card_id == 2:  # M2
            return 1 if player_id == 1 else 3
        else:  # M3-M45
            return card_id + 1  # +1 for the 4 starter card statuses
            
    def _is_starter_card(self, card_id):
        return card_id == 1 or card_id == 2
        
    def _set_card_unplayable(self, card_id, player_id):
        status_index = self._get_card_status_index(card_id, player_id)
        
        if self._is_starter_card(card_id):
            # For M1/M2
            self.merchant_cards_status[status_index] = MerchantCardStatus.P1_UNPLAYABLE.value if player_id == 1 else MerchantCardStatus.P2_UNPLAYABLE.value
        else:
            # For regular cards M3-M45
            if player_id == 1:
                self.merchant_cards_status[status_index] = MerchantCardStatus.REG_P1_UNPLAYABLE.value
            else:
                self.merchant_cards_status[status_index] = MerchantCardStatus.REG_P2_UNPLAYABLE.value
    
    def _set_card_playable(self, card_id, player_id):
        status_index = self._get_card_status_index(card_id, player_id)
        
        if self._is_starter_card(card_id):
            # For M1/M2
            self.merchant_cards_status[status_index] = MerchantCardStatus.P1_PLAYABLE.value if player_id == 1 else MerchantCardStatus.P2_PLAYABLE.value
        else:
            # For regular cards M3-M45
            if player_id == 1:
                self.merchant_cards_status[status_index] = MerchantCardStatus.REG_P1_PLAYABLE.value
            else:
                self.merchant_cards_status[status_index] = MerchantCardStatus.REG_P2_PLAYABLE.value
    
    def _is_card_playable(self, card_id, player_id):
        status_index = self._get_card_status_index(card_id, player_id)
        
        if self._is_starter_card(card_id):
            # For M1/M2
            if player_id == 1:
                return self.merchant_cards_status[status_index] == MerchantCardStatus.P1_PLAYABLE.value
            else:
                return self.merchant_cards_status[status_index] == MerchantCardStatus.P2_PLAYABLE.value
        else:
            # For regular cards M3-M45
            if player_id == 1:
                return self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P1_PLAYABLE.value
            else:
                return self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P2_PLAYABLE.value
    
    def _handle_rest(self):
        # Reset all unplayable cards for current player to playable
        player_id = self.current_player.player_id
        
        # Reset starter cards (M1, M2)
        if player_id == 1:
            self._set_card_playable(1, 1)  # M1 for P1
            self._set_card_playable(2, 1)  # M2 for P1
        else:
            self._set_card_playable(1, 2)  # M1 for P2
            self._set_card_playable(2, 2)  # M2 for P2
            
        # Reset regular cards (M3-M45)
        for card_id in range(3, 46):
            status_index = self._get_card_status_index(card_id, player_id)
            if player_id == 1 and self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P1_UNPLAYABLE.value:
                self._set_card_playable(card_id, player_id)
            elif player_id == 2 and self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P2_UNPLAYABLE.value:
                self._set_card_playable(card_id, player_id)
                    
        return GAME_CONSTANTS['REWARDS']['REST']
    
    def _handle_get_merchant_card(self, action):
        card_id = action + 2  # e.g. M3 = action 1 + 2
        
        if self.merchant_deck[card_id] in self.merchant_market:
            # Set card as playable for the current player
            player_id = self.current_player.player_id
            self._set_card_playable(card_id, player_id)
                
            self.merchant_deck[card_id].owned = True
            self.merchant_market.remove(self.merchant_deck[card_id])
            self._draw_merchant_card()
            
            # Diminishing returns for getting more cards
            owned_count = self._count_owned_merchant_cards(player_id)
            card_value = max(0.5, GAME_CONSTANTS['REWARDS']['GET_MERCHANT_CARD'] * (1 - 0.15 * (owned_count - 2)))
            
            return card_value

        raise InvalidActionError(f"Card M{card_id} is not in market")
        
    def _count_owned_merchant_cards(self, player_id):
        # Always 2 starter cards
        owned_count = 2
        
        # Count regular cards owned by the player
        if player_id == 1:
            owned_count += sum(1 for i in range(4, 47) 
                            if self.merchant_cards_status[i] == MerchantCardStatus.REG_P1_PLAYABLE.value 
                            or self.merchant_cards_status[i] == MerchantCardStatus.REG_P1_UNPLAYABLE.value)
        else:
            owned_count += sum(1 for i in range(4, 47) 
                            if self.merchant_cards_status[i] == MerchantCardStatus.REG_P2_PLAYABLE.value 
                            or self.merchant_cards_status[i] == MerchantCardStatus.REG_P2_UNPLAYABLE.value)
                
        return owned_count

    def _handle_crystal_card(self, card, card_index):
        # Add crystals to player's caravan
        for crystal, amount in card.gain.items():
            self.current_player.caravan[crystal] += amount
        
        # Set card as unplayable
        card_id = card_index + 1
        self._set_card_unplayable(card_id, self.current_player.player_id)
        
        # Calculate reward based on crystal values
        reward = 0
        for crystal, amount in card.gain.items():
            reward += GAME_CONSTANTS['CRYSTAL_VALUES'][crystal] * amount
            
        return reward

    def _handle_trade_card(self, card, card_index): 
        # Check if player has enough crystals
        if all(self.current_player.caravan[crystal] >= amount 
                for crystal, amount in card.cost.items()):
            # Apply the trade
            for crystal, amount in card.cost.items():
                self.current_player.caravan[crystal] -= amount
            for crystal, amount in card.gain.items():
                self.current_player.caravan[crystal] += amount

            # Set card as unplayable
            card_id = card_index + 1
            self._set_card_unplayable(card_id, self.current_player.player_id)
                
            # Net gain in crystal value
            loss = sum(GAME_CONSTANTS['CRYSTAL_VALUES'][crystal] * amount 
                     for crystal, amount in card.cost.items())
            gain = sum(GAME_CONSTANTS['CRYSTAL_VALUES'][crystal] * amount 
                      for crystal, amount in card.gain.items())
            
            return (gain - loss)
        
        raise InvalidActionError(f"Not enough crystals for trade card {card.name}")

    def _handle_upgrade_card(self, card, card_index):
        upgrade_points = card.gain
        caravan = self.current_player.caravan
        upgrade_steps_taken = 0

        # Greedy upgrade from lowest to highest: Y → G → B → P
        while upgrade_points > 0:
            # Y → G
            if caravan.get("yellow", 0) > 0:
                caravan["yellow"] -= 1
                caravan["green"] = caravan.get("green", 0) + 1
                upgrade_points -= 1
                upgrade_steps_taken += 1
            # G → B
            elif caravan.get("green", 0) > 0:
                caravan["green"] -= 1
                caravan["blue"] = caravan.get("blue", 0) + 1
                upgrade_points -= 1
                upgrade_steps_taken += 1
            # B → P
            elif caravan.get("blue", 0) > 0:
                caravan["blue"] -= 1
                caravan["pink"] = caravan.get("pink", 0) + 1
                upgrade_points -= 1
                upgrade_steps_taken += 1
            else:
                break  # No more crystals can be upgraded

        if upgrade_steps_taken == 0:
            raise InvalidActionError("No crystals can be upgraded")

        # Set card as unplayable
        card_id = card_index + 1
        self._set_card_unplayable(card_id, self.current_player.player_id)
            
        return upgrade_steps_taken * 0.5

    def _handle_use_merchant_card(self, action):
        card_id = action - Actions.useM1.value + 1
        card = self.merchant_deck.get(card_id)
        card_index = card_id - 1
        
        # Check if card is playable
        if not self._is_card_playable(card_id, self.current_player.player_id):
            raise InvalidActionError(f"Card M{card_id} is not playable for player {self.current_player.player_id}")
        
        # Handle by card type
        if card.card_type == "crystal":
            reward = self._handle_crystal_card(card, card_index)
        elif card.card_type == "trade":
            reward = self._handle_trade_card(card, card_index)
        elif card.card_type == "upgrade":
            reward = self._handle_upgrade_card(card, card_index)
        else:
            raise InvalidActionError(f"Unknown card type: {card.card_type}")
        
        # Strategic bonus
        strategic_reward = self._calculate_strategic_reward()
        
        return reward + strategic_reward
    
    def _calculate_strategic_reward(self):
        caravan = self.current_player.caravan
        total_crystals = sum(caravan.values())
        
        # Reward for balanced mix of crystals
        crystal_diversity = sum(1 for amount in caravan.values() if amount > 0)
        
        # Check if close to a golem in market
        potential_golem_match = False
        for golem in self.golem_market:
            # Count matching crystal types
            matching_types = sum(1 for crystal, amount in golem.cost.items() 
                               if crystal in caravan and caravan[crystal] > 0)
            
            # Check if close to acquiring
            total_needed = sum(golem.cost.values())
            if matching_types >= min(3, len(golem.cost)) and total_crystals >= total_needed * 0.7:
                potential_golem_match = True
                break
                
        strategic_reward = 0
        
        if crystal_diversity >= 3 and total_crystals <= GAME_CONSTANTS['MAX_CRYSTALS'] - 2:
            # Good diversity with room to add more
            strategic_reward += GAME_CONSTANTS['REWARDS']['CRYSTAL_MANAGEMENT'] * 0.5
        
        if potential_golem_match:
            # Extra for crystals that match market golems
            strategic_reward += GAME_CONSTANTS['REWARDS']['CRYSTAL_MANAGEMENT']
        
        # Penalty for near limit without diversity
        if total_crystals > GAME_CONSTANTS['MAX_CRYSTALS'] - 2 and crystal_diversity < 3:
            strategic_reward -= GAME_CONSTANTS['REWARDS']['CRYSTAL_MANAGEMENT'] * 0.5
            
        return strategic_reward
    
    def _handle_get_golem_card(self, action):
        card_id = action - Actions.getG1.value + 1 # e.g. G1 = 89 - 89 + 1
        golem_card = self.golem_deck[card_id]
        
        # Check golem availability and cost
        if golem_card not in self.golem_market:
            raise InvalidActionError(f"Golem card G{card_id} is not in the market")
            
        if not all(self.current_player.caravan[crystal] >= amount 
                  for crystal, amount in golem_card.cost.items()):
            raise InvalidActionError(f"Not enough crystals for golem card G{card_id}")
        
        # Spend crystals and claim golem
        for crystal, amount in golem_card.cost.items():
            self.current_player.caravan[crystal] -= amount
            
        self.current_player.golem_count += 1
        self.current_player.points += golem_card.points
        golem_card.owned = True
        
        self.golem_market.remove(golem_card)
        self._draw_golem_card()
        
        # Base reward + points + win proximity bonus
        reward = GAME_CONSTANTS['REWARDS']['GOLEM_BASE_REWARD'] + golem_card.points
        
        # Bonus for closer to winning
        if self.current_player.golem_count == 3:
            reward += GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY']
        elif self.current_player.golem_count == 4:
            reward += GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY'] * 2
        elif self.current_player.golem_count == 5:
            reward += GAME_CONSTANTS['REWARDS']['WIN_PROXIMITY'] * 3

        return reward

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(125)
        
        # Define observation space
        crystal_types = ["yellow", "green", "blue", "pink"]
        
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([46, 46, 46, 46, 46, 46]),
            "golem_market": spaces.MultiDiscrete([37, 37, 37, 37, 37]),
            
            "player1_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            # Merchant cards status: 
            # First 4: Player-specific M1/M2 (2 for each player)
            # Remaining 43: Status for cards M3-M45
            "merchant_cards_status": spaces.MultiDiscrete([2, 2, 2, 2] + [5] * 43),
            "player1_golem_count": spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            "player1_points": spaces.Box(low=0, high=140, shape=(1,), dtype=np.int32),
            
            "player2_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            "player2_golem_count": spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            "player2_points": spaces.Box(low=0, high=140, shape=(1,), dtype=np.int32),
        })
        
        # Initialize decks
        self.merchant_deck = self.create_merchant_deck()
        self.golem_deck = self.create_golem_deck()
        
        # Initialize markets (populated in reset)
        self.merchant_market = []
        self.golem_market = []
        
        # Initialize players
        self.player1 = Player(1)
        self.player2 = Player(2)
        self.current_player = None
        self.next_player = None
        
        # Card status tracking array
        # [0-1]: P1's M1/M2 status (0=unplayable, 1=playable)
        # [2-3]: P2's M1/M2 status (0=unplayable, 1=playable)
        # [4-46]: Cards M3-M45 status (0=unowned, 1=P1_unplayable, 2=P1_playable, 3=P2_unplayable, 4=P2_playable)
        self.merchant_cards_status = [0] * 47
        
        # Game state tracking
        self.endgame_triggered = False
        self.endgame_initiator = None
        self.starting_player = None
        self.round_number = 0
        self.winner = None
        self.player1_final_points = 0
        self.player2_final_points = 0
        
        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_obs(self, player):
        # Set player perspectives
        player1 = player
        player2 = self.player2 if player == self.player1 else self.player1
        
        # Keep merchant card status in range
        merchant_cards_status = np.array(self.merchant_cards_status, dtype=np.int32)
        # First 4 (starter cards) range [0,1]
        for i in range(4):
            merchant_cards_status[i] = np.clip(merchant_cards_status[i], 0, 1)
        # Rest (regular cards) range [0,4]
        for i in range(4, 47):
            merchant_cards_status[i] = np.clip(merchant_cards_status[i], 0, 4)

        # Market states with padding
        merchant_market_state = [card.card_id for card in self.merchant_market]
        merchant_market_state.extend([0] * (6 - len(merchant_market_state)))
        
        golem_market_state = [card.card_id for card in self.golem_market]
        golem_market_state.extend([0] * (5 - len(golem_market_state)))
        
        # Caravan states
        player1_caravan = {
            crystal: np.clip(np.array([amount], dtype=np.int32), 0, 20)
            for crystal, amount in player1.caravan.items()
        }
        
        player2_caravan = {
            crystal: np.clip(np.array([amount], dtype=np.int32), 0, 20)
            for crystal, amount in player2.caravan.items()
        }
        
        # Other player state values
        player1_golem_count = np.clip(np.array([player1.golem_count], dtype=np.int32), 0, 6)
        player2_golem_count = np.clip(np.array([player2.golem_count], dtype=np.int32), 0, 6)
        
        player1_points = np.clip(np.array([player1.points], dtype=np.int32), 0, 140)
        player2_points = np.clip(np.array([player2.points], dtype=np.int32), 0, 140)
        
        return {
            "merchant_market": np.array(merchant_market_state, dtype=np.int32),
            "golem_market": np.array(golem_market_state, dtype=np.int32),
            
            "player1_caravan": player1_caravan,
            "merchant_cards_status": merchant_cards_status,
            "player1_golem_count": player1_golem_count,
            "player1_points": player1_points,
            
            "player2_caravan": player2_caravan,
            "player2_golem_count": player2_golem_count,
            "player2_points": player2_points,
        }

    def _get_info(self):
        return {
            "valid_actions": self._get_valid_actions(self.current_player),
            "current_player": int(self.current_player.player_id - 1),  # 0 for P1, 1 for P2
            "winner": self.winner if self.winner is not None else None,
            "round_number": self.round_number
        }
    
    def reset(self, seed=None, options=None):
        if self.render_mode == "text":
            print("Century: Golem Edition | YGBP \n")
        
        super().reset(seed=seed)
        
        # Reset merchant deck
        for card in self.merchant_deck.values():
            card.owned = (card.card_id == 1 or card.card_id == 2)  # Only starter cards owned
        
        # Reset merchant market
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1 and cid != 2], 6
        )
        
        # Reset golem deck and market
        for card in self.golem_deck.values():
            card.owned = False
            
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        # Randomize first player
        players = [self.player1, self.player2]
        random.shuffle(players)
        self.current_player = players[0]
        self.next_player = players[1]
        self.starting_player = self.current_player
        
        # Reset merchant card status
        for i in range(4, 47):
            self.merchant_cards_status[i] = MerchantCardStatus.UNOWNED.value
            
        # Starter cards are playable
        self.merchant_cards_status[0] = MerchantCardStatus.P1_PLAYABLE.value  # P1's M1
        self.merchant_cards_status[1] = MerchantCardStatus.P1_PLAYABLE.value  # P1's M2
        self.merchant_cards_status[2] = MerchantCardStatus.P2_PLAYABLE.value  # P2's M1
        self.merchant_cards_status[3] = MerchantCardStatus.P2_PLAYABLE.value  # P2's M2
        
        # Reset player resources
        for player in (self.player1, self.player2):
            player.caravan = {
                "yellow": 0,
                "green": 0,
                "blue": 0,
                "pink": 0
            }
            player.golem_count = 0
            player.points = 0
            
        # Starting resources
        self.current_player.caravan["yellow"] = 3
        self.next_player.caravan["yellow"] = 4
            
        # Reset game state
        self.endgame_triggered = False
        self.endgame_initiator = None
        self.round_number = 1
        self.winner = None
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()

        if self.render_mode is not None:
            self.render()

        return observation, info
    
    def _get_valid_actions(self, player):
        valid_actions = np.zeros(self.action_space.n, dtype=np.int32)

        # Check if rest is needed (any unplayable cards)
        needs_rest = False
        
        # Check starter cards
        if player.player_id == 1:
            if (not self._is_card_playable(1, 1) or not self._is_card_playable(2, 1)):
                needs_rest = True
        else:
            if (not self._is_card_playable(1, 2) or not self._is_card_playable(2, 2)):
                needs_rest = True
                
        # Check regular cards
        if not needs_rest:
            for card_id in range(3, 46):
                if not self._is_card_playable(card_id, player.player_id) and self._is_card_owned(card_id, player.player_id):
                    needs_rest = True
                    break
        
        valid_actions[Actions.rest.value] = 1 if needs_rest else 0

        # Check merchant card gets
        for i in range(Actions.getM3.value, Actions.getM45.value + 1):
            card_id = i + 2  # Convert action to card ID
            if self.merchant_deck[card_id] in self.merchant_market:
                valid_actions[i] = 1

        # Check card uses
        self._check_use_merchant_card_actions(player, valid_actions)

        # Check golem gets
        self._check_get_golem_card_actions(player, valid_actions)

        return valid_actions
    
    def _is_card_owned(self, card_id, player_id):
        # M1/M2 always owned
        if card_id == 1 or card_id == 2:
            return True
            
        # Check M3-M45
        status_index = self._get_card_status_index(card_id, player_id)
        
        if player_id == 1:
            return (self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P1_PLAYABLE.value or 
                    self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P1_UNPLAYABLE.value)
        else:
            return (self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P2_PLAYABLE.value or 
                    self.merchant_cards_status[status_index] == MerchantCardStatus.REG_P2_UNPLAYABLE.value)
    
    def _check_use_merchant_card_actions(self, player, valid_actions):
        # Check starter cards
        if self._is_card_playable(1, player.player_id):
            card = self.merchant_deck.get(1)
            if self._can_use_card(card, player):
                valid_actions[Actions.useM1.value] = 1
                
        if self._is_card_playable(2, player.player_id):
            card = self.merchant_deck.get(2)
            if self._can_use_card(card, player):
                valid_actions[Actions.useM2.value] = 1

        # Check regular cards
        for card_id in range(3, 46):
            if self._is_card_playable(card_id, player.player_id):
                card = self.merchant_deck.get(card_id)
                if self._can_use_card(card, player):
                    action_index = getattr(Actions, f'useM{card_id}').value
                    valid_actions[action_index] = 1
    
    def _can_use_card(self, card, player):
        if card.card_type == "crystal":
            return True  # Always usable
        elif card.card_type == "trade":
            # Need required crystals
            return all(player.caravan[crystal] >= amount 
                      for crystal, amount in card.cost.items())
        elif card.card_type == "upgrade":
            # Need at least one crystal to upgrade
            return (player.caravan["yellow"] > 0 or 
                   player.caravan["green"] > 0 or 
                   player.caravan["blue"] > 0)
        return False
    
    def _check_get_golem_card_actions(self, player, valid_actions):
        for golem_card in self.golem_market:
            if all(player.caravan[crystal] >= amount 
                  for crystal, amount in golem_card.cost.items()):
                action_index = getattr(Actions, f'getG{golem_card.card_id}').value
                valid_actions[action_index] = 1
    
    def step(self, action):
        if self.render_mode == "text":
            player_type = "Player 1" if self.current_player.player_id == 1 else "Player 2"
            print(f"==== {player_type} | {Actions(int(action)).name} ====\n")
            
        terminated = False
        reward = GAME_CONSTANTS['REWARDS']['STEP']
        
        # Handle action
        if action == Actions.rest.value:
            reward += self._handle_rest()
        elif Actions.getM3.value <= action <= Actions.getM45.value:
            reward += self._handle_get_merchant_card(action)
        elif Actions.useM1.value <= action <= Actions.useM45.value:
            reward += self._handle_use_merchant_card(action)
        elif Actions.getG1.value <= action <= Actions.getG36.value:
            reward += self._handle_get_golem_card(action)

        # Remove excess crystals
        reward -= remove_excess_crystals(self.current_player)
        
        # Check for endgame trigger
        if not self.endgame_triggered and self.current_player.golem_count >= GAME_CONSTANTS['GOLEM_ENDGAME_THRESHOLD']:
            self.endgame_triggered = True
            self.endgame_initiator = self.current_player
            
        # Check for game end
        if self.endgame_triggered and self.current_player != self.endgame_initiator:
            terminated = True

            # Final scoring
            self.player1_final_points = calculate_total_points(self.player1)
            self.player2_final_points = calculate_total_points(self.player2)
            
            # Adjust reward based on winner
            score_diff = self.player1_final_points - self.player2_final_points
            
            # Only award win reward to P1
            if self.current_player != self.player1:
                reward = 0
            
            if score_diff > 0:
                self.winner = "P1"
                reward += GAME_CONSTANTS['REWARDS']['WIN']
            elif score_diff == 0:
                self.winner = "P0"
                reward += GAME_CONSTANTS['REWARDS']['TIE']
            else:
                self.winner = "P2"
                reward -= GAME_CONSTANTS['REWARDS']['WIN']
        
        # Store player who finished turn before switching
        player_finished_turn = self.current_player
        
        # Switch turns
        self.current_player, self.next_player = self.next_player, self.current_player
        
        # Increment round after second player
        if player_finished_turn != self.starting_player:
            self.round_number += 1
            
        observation = self._get_obs(self.current_player)
        info = self._get_info()
        
        if self.render_mode is not None:
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":               
            print(f"Round: {self.round_number}\n")
            
            # Display markets
            print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
            print("GM: " + " | ".join([f"G{g.card_id}-{g.name}-{g.points}" for g in self.golem_market]))
            print("")
            
            # Player 1
            print(f"Player {self.player1.player_id}")
            self._render_player_state(self.player1)
            print("")
            
            # Player 2
            print(f"Player {self.player2.player_id}")
            self._render_player_state(self.player2)
            print("")
            
    def _render_player_state(self, player):
        # Caravan
        print(f"Y:{player.caravan['yellow']} | G:{player.caravan['green']} | B:{player.caravan['blue']} | P:{player.caravan['pink']}")
        
        # Merchant cards
        self._render_player_merchant_cards(player)
            
        # Golem count and points
        print(f"GC: {player.golem_count}")
        print(f"P: {player.points}")
        
    def _render_player_merchant_cards(self, player):
        player_id = player.player_id
        
        # M1/M2 starter cards
        for i in range(2):
            card_id = i + 1
            status_index = self._get_card_status_index(card_id, player_id)
            
            card = self.merchant_deck.get(card_id)
            status = "playable" if self._is_card_playable(card_id, player_id) else "unplayable"
            print(f"M{card_id}-{card.name}: {status}")
                
        # Regular merchant cards
        for card_id in range(3, 46):
            if self._is_card_owned(card_id, player_id):
                card = self.merchant_deck.get(card_id)
                status = "playable" if self._is_card_playable(card_id, player_id) else "unplayable"
                print(f"M{card_id}-{card.name}: {status}")
    
    def close(self):
        if self.render_mode == "text":
            print("=== CLOSE ENVIRONMENT ===") 