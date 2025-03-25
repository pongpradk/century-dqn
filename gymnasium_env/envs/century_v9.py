import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

GAME_CONSTANTS = {
    'MAX_CRYSTALS': 10,
    'GOLEM_ENDGAME_THRESHOLD': 3,
    'CRYSTAL_VALUES': {
        'yellow': 0.5,
        'green': 1.0
    },
    'REWARDS': {
        'WIN': 100,
        'TIE': 50,
        'STEP': -0.01,
        'REST': -0.1,
        'GET_MERCHANT_CARD': 1
    }
}

class Player:
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.caravan = {
            "yellow": 3,
            "green": 0
        }
        # status of each merchant card for this player
        self.merchant_cards = [2] + [0] * 8 # 0 = not owned, 1 = owned but unplayable, 2 = owned and playable
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
    getM3 = 2 # ADD +100 and -100 for reaching max_timesteps in DQN training code, else +50 for tie
    getM4 = 3
    getM5 = 4
    getM6 = 5
    getM7 = 6
    getM8 = 7
    getM9 = 8
    useM1 = 9
    useM2 = 10
    useM3 = 11
    useM4 = 12
    useM5 = 13
    useM6 = 14
    useM7 = 15
    useM8 = 16
    useM9 = 17
    getG1 = 18
    getG2 = 19
    getG3 = 20
    getG4 = 21
    getG5 = 22

class CardStatus(Enum):
    NOT_OWNED = 0
    UNPLAYABLE = 1
    PLAYABLE = 2

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 1}
    
    def create_merchant_deck(self):
        return {
            1: MerchantCard(1, "Y2", "crystal", gain={"yellow": 2, "green": 0}, owned = True),
            2: MerchantCard(2, "Y3", "crystal", gain={"yellow": 3, "green": 0}),
            3: MerchantCard(3, "Y4", "crystal", gain={"yellow": 4, "green": 0}),
            4: MerchantCard(4, "Y1G1", "crystal", gain={"yellow": 1, "green": 1}),
            5: MerchantCard(5, "Y2G1", "crystal", gain={"yellow": 2, "green": 1}),
            6: MerchantCard(6, "G2", "crystal", gain={"yellow": 0, "green": 2}),
            7: MerchantCard(7, "G1:Y3", "trade", cost={"yellow": 0, "green": 1}, gain={"yellow": 3, "green": 0}),
            8: MerchantCard(8, "Y2:G2", "trade", cost={"yellow": 2, "green": 0}, gain={"yellow": 0, "green": 2}),
            9: MerchantCard(9, "Y3:G3", "trade", cost={"yellow": 3, "green": 0}, gain={"yellow": 0, "green": 3}),
        }
    
    def create_golem_deck(self):
        return {
            1: GolemCard(1, "Y2G2", {"yellow": 2, "green": 2}, points=6),
            2: GolemCard(2, "Y3G2", {"yellow": 3, "green": 2}, points=7),
            3: GolemCard(3, "Y2G3", {"yellow": 2, "green": 3}, points=8),
            4: GolemCard(4, "G4", {"yellow": 0, "green": 4}, points=8),
            5: GolemCard(5, "G5", {"yellow": 0, "green": 5}, points=10)
        }
    
    def _draw_merchant_card(self):
        cards_in_deck = [
            card for cid, card in self.merchant_deck.items()
            if cid != 1 and not card.owned and card not in self.merchant_market
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
        self.current_player.merchant_cards = [2 if card == 1 else card for card in self.current_player.merchant_cards]
        return GAME_CONSTANTS['REWARDS']['REST']
    
    def _handle_get_merchant_card(self, action):
        card_id = action + 1 # e.g. action 1 = get M2
        
        if self.merchant_deck[card_id] in self.merchant_market:
            self.current_player.merchant_cards[action] = CardStatus.PLAYABLE.value

            self.merchant_deck[card_id].owned = True
            
            self.merchant_market.remove(self.merchant_deck[card_id])
            self._draw_merchant_card()
            
            return GAME_CONSTANTS['REWARDS']['GET_MERCHANT_CARD']

    def _handle_use_merchant_card(self, action):
        card_idx = action - Actions.useM1.value
        card = self.merchant_deck.get(card_idx + 1)

        if self.current_player.merchant_cards[card_idx] == CardStatus.PLAYABLE.value and card:
            if card.card_type == "crystal":
                for crystal, amount in card.gain.items():
                    self.current_player.caravan[crystal] += amount
                return (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.gain.get('yellow', 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.gain.get('green', 0))
            elif card.card_type == "trade":
                # Check if player has enough crystals
                if all(self.current_player.caravan[crystal] >= amount 
                        for crystal, amount in card.cost.items()):
                    # Apply the trade
                    for crystal, amount in card.cost.items():
                        self.current_player.caravan[crystal] -= amount
                    for crystal, amount in card.gain.items():
                        self.current_player.caravan[crystal] += amount

                    loss = (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.cost.get("yellow", 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.cost.get("green", 0))
                    gain = (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * card.gain.get("yellow", 0)) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * card.gain.get("green", 0))
                    return (gain - loss)

            self.current_player.merchant_cards[card_idx] = CardStatus.UNPLAYABLE.value
    
    def _handle_get_golem_card(self, action):
        card_id = action - Actions.getG1.value + 1
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
            
            return golem_card.points
    
    def __init__(self, render_mode=None, record_session=False):
        
        self.action_space = spaces.Discrete(23)
        
        # Open information
        crystal_types = ["yellow", "green"]  # Add more types here as needed
        
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([10, 10, 10, 10, 10, 10]),
            "golem_market": spaces.MultiDiscrete([6, 6, 6, 6, 6]),
            
            # Replace individual crystal spaces with a single Dict space
            "player1_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            "player1_merchant_cards": spaces.MultiDiscrete([3] * 9),
            "player1_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "player1_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            
            "player2_caravan": spaces.Dict({
                crystal: spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
                for crystal in crystal_types
            }),
            "player2_merchant_cards": spaces.MultiDiscrete([3] * 9),
            "player2_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "player2_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        })
        
        self.merchant_deck = self.create_merchant_deck()
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 6 # draw 3 cards to market, excluding M1
        )
        
        self.golem_deck = self.create_golem_deck()
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        # Initialize players
        self.player1 = Player(1)
        self.player2 = Player(2)
        self.current_player, self.next_player = random.sample([self.player1, self.player2], 2)
        self.next_player.caravan["yellow"] += 1
        
        self.endgame_triggered = False  # Tracks if endgame was triggered
        self.endgame_initiator = None   # The player who triggered the endgame
        
        self.turn_counter = 0
        self.round = 1
        
        # Render
        self.window = None
        self.clock = None
        self.window_size = 800  # Size of the render window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Record
        self.record_session = record_session  # Enable/disable recording
        self.frame_count = 0  # Track frame number
        self.video_output_path = "session.mp4"  # Output video file name
        self.fps = 1  # Frame rate (match render_fps)
        
        if self.record_session:
            self.video_writer = None  # Initialize later when first frame is captured
        
        self.winner = None  # Stores winner's name ("DQN" or "Random")
        self.player1_final_points = 0  # Stores final DQN points
        self.player2_final_points = 0  # Stores final Random points
    
    def _get_obs(self, player):
        # Returns observation from the perspective of the player
        player1 = player
        player2 = self.player2 if player == self.player1 else self.player1
        
        merchant_cards_state = np.array(player1.merchant_cards, dtype=np.int32)

        merchant_market_state = [card.card_id for card in self.merchant_market]
        while len(merchant_market_state) < 6:
            merchant_market_state.append(9)
        
        golem_market_state = [card.card_id for card in self.golem_market]
        while len(golem_market_state) < 5:
            golem_market_state.append(5)
        
        return {
            "merchant_market": np.array(merchant_market_state, dtype=np.int32),
            "golem_market": np.array(golem_market_state, dtype=np.int32),
            
            "player1_caravan": {
                crystal: np.array([amount], dtype=np.int32)
                for crystal, amount in player1.caravan.items()
            },
            "player1_merchant_cards": merchant_cards_state,
            "player1_golem_count": np.array([player1.golem_count], dtype=np.int32),
            "player1_points": np.array([player1.points], dtype=np.int32),
            
            "player2_caravan": {
                crystal: np.array([amount], dtype=np.int32)
                for crystal, amount in player2.caravan.items()
            },
            "player2_merchant_cards": np.array(player2.merchant_cards, dtype=np.int32),
            "player2_golem_count": np.array([player2.golem_count], dtype=np.int32),
            "player2_points": np.array([player2.points], dtype=np.int32),
        }

    def _get_info(self):
        return {
            "valid_actions": self._get_valid_actions(self.current_player),
            "current_player": int(self.current_player.player_id - 1),  # 0 for player1, 1 for player2
        }
    
    def reset(self, seed=None, options=None):
        
        print("Century: Golem Edition | Version 9.0\n")
        
        super().reset(seed=seed)
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 6
        )
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.golem_deck.values()]
        
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        self.current_player = random.choice([self.player1, self.player2]) # Choose which player to play first
        self.other_player = self.player1 if self.current_player == self.player2 else self.player2
        
        # Reset players
        self.current_player.caravan["yellow"], self.other_player.caravan["yellow"] = 3, 4
        for player in (self.player1, self.player2):
            player.caravan["green"] = 0
            player.merchant_cards = [2] + [0] * 8
            player.golem_count = 0
            player.points = 0
            
        self.endgame_triggered = False  # Tracks if endgame was triggered
        self.endgame_initiator = None   # The player who triggered the endgame
        
        self.turn_counter = 0
        self.round = 1
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()

        if self.render_mode != None:
            self.render()

        return observation, info
    
    def _get_valid_actions(self, player):

        valid_actions = np.zeros(self.action_space.n, dtype=np.int32)

        valid_actions[Actions.rest.value] = 0
        # Rest is valid if any merchant card is unplayable
        for i in range(Actions.useM1.value, Actions.useM9.value + 1):
            card_idx = i - Actions.useM1.value
            if player.merchant_cards[card_idx] == 1:  # If owned but unplayable
                valid_actions[Actions.rest.value] = 1
                break

        # Get merchant card actions
        for i in range(Actions.getM2.value, Actions.getM9.value + 1):
            card_id = i + 1
            if self.merchant_deck[card_id] in self.merchant_market:
                valid_actions[i] = 1

        # Use merchant card actions
        for i in range(Actions.useM1.value, Actions.useM9.value + 1):
            card_idx = i - Actions.useM1.value

            if player.merchant_cards[card_idx] == 2:  # if playable
                card = self.merchant_deck.get(card_idx + 1)  # Get corresponding merchant card
                if card.card_type == "crystal":
                    valid_actions[i] = 1  # Always valid for crystal-gaining cards
                elif card.card_type == "trade":
                    # Ensure the player has enough crystals to trade
                    if all(player.caravan[crystal] >= amount 
                          for crystal, amount in card.cost.items()):
                        valid_actions[i] = 1

        # Get golem card actions
        for i in range(Actions.getG1.value, Actions.getG5.value + 1):
            card_id = i - Actions.getG1.value + 1
            golem_card = self.golem_deck.get(card_id)
            if (golem_card in self.golem_market and
                all(player.caravan[crystal] >= amount 
                    for crystal, amount in golem_card.cost.items())):
                    valid_actions[i] = 1

        return valid_actions    

    # Remove and penalize excess crystals
    def _remove_excess_crystals(self):
        total_crystals = sum(self.current_player.caravan.values())
        if total_crystals > GAME_CONSTANTS['MAX_CRYSTALS']:
            excess = total_crystals - GAME_CONSTANTS['MAX_CRYSTALS']
            # Remove from yellow first, then green
            excess_yellow = min(excess, self.current_player.caravan["yellow"])
            excess_green = excess - excess_yellow
            self.current_player.caravan["yellow"] -= excess_yellow
            self.current_player.caravan["green"] -= excess_green
            
            return (0.5 * excess_yellow) + (1 * excess_green)

        return 0
    
    def calculate_total_points(self, player):
        """Calculates total points for a player, including non-yellow crystals."""
        crystal_points = player.caravan["green"]  # Assuming green crystals are worth 1 point
        return player.points + crystal_points
    
    def step(self, action):
        
        if self.render_mode == "text":
            print(f"==== DQN | {Actions(int(action)).name} ====\n") if self.current_player.player_id == 1 else print(f"==== Random | {Actions(int(action)).name} ====\n")
            
        terminated = False
        reward = GAME_CONSTANTS['REWARDS']['STEP']
        
        # Store the last action for each player
        if self.current_player == self.player1:
            self.last_action_dqn = Actions(action).name  # Convert action enum to string
        else:
            self.last_action_random = Actions(action).name
        
        # Action: Rest
        if action == Actions.rest.value:
            reward += self._handle_rest()
        # Action: Get a merchant card
        elif Actions.getM2.value <= action <= Actions.getM9.value:
            reward += self._handle_get_merchant_card(action)
        # Action: Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM9.value:
            reward += self._handle_use_merchant_card(action)
        # Action: Get a golem card
        elif Actions.getG1.value <= action <= Actions.getG5.value:
            reward += self._handle_get_golem_card(action)

        reward -= self._remove_excess_crystals()
        
        # Check if the endgame is triggered
        if not self.endgame_triggered and self.current_player.golem_count >= GAME_CONSTANTS['GOLEM_ENDGAME_THRESHOLD']:
            self.endgame_triggered = True
            self.endgame_initiator = self.current_player
        # Check if this is the last turn (player2 of endgame initiator)
        if self.endgame_triggered and self.current_player != self.endgame_initiator:
            terminated = True

            # Calculate final points for both players
            self.player1_final_points = self.calculate_total_points(self.player1)
            self.player2_final_points = self.calculate_total_points(self.player2)
            
            # If not player1's turn, the reward is reset, before calculation
            if self.current_player != self.player1:
                reward = 0
            
            score_diff = self.player1_final_points - self.player2_final_points
            if score_diff > 0:
                self.winner = self.player1
                reward += GAME_CONSTANTS['REWARDS']['WIN']
            elif score_diff == 0:
                self.winner = None
                reward += GAME_CONSTANTS['REWARDS']['TIE']
            else:
                self.winner = self.player2
                reward -= GAME_CONSTANTS['REWARDS']['WIN']
        
        # Switch turn
        self.current_player, self.other_player = self.other_player, self.current_player

        # Turn and Round
        if not terminated:
            self.turn_counter += 1  
            if self.turn_counter % 2 == 0:
                self.round += 1
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()
        
        if self.render_mode != None:
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "text":               
            print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
            print("GM: " + " | ".join([f"G{g.card_id}-{g.name}-{g.points}" for g in self.golem_market]))
            print("")
            print(f"P{self.player1.player_id}")
            print(f"Y: {self.player1.caravan['yellow']}")
            print(f"G: {self.player1.caravan['green']}")
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
            print(f"Y: {self.player2.caravan['yellow']}")
            print(f"G: {self.player2.caravan['green']}")
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
        print("=== CLOSE ENVIRONMENT ===")