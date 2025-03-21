import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

class Player:
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.yellow = 3
        self.green = 0
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

class CenturyGolemEnv(gym.Env):
    metadata = {"render_modes": ["text", "human"], "render_fps": 1}
    
    def __init__(self, render_mode=None, record_session=False):
        
        self.action_space = spaces.Discrete(23)
        
        # Open information
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([10, 10, 10, 10, 10, 10]),
            "golem_market": spaces.MultiDiscrete([6, 6, 6, 6, 6]),
            
            "agent_yellow": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "agent_green": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "agent_merchant_cards": spaces.MultiDiscrete([3] * 9),
            "agent_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),  # New: number of golem cards owned
            "agent_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            
            "opponent_yellow": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "opponent_green": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "opponent_merchant_cards": spaces.MultiDiscrete([3] * 9),
            "opponent_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "opponent_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        })
        
        self.merchant_deck = {
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

        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 6 # draw 3 cards to market, excluding M1
        )
        
        self.golem_deck = {
            1: GolemCard(1, "Y2G2", {"yellow": 2, "green": 2}, points=6),
            2: GolemCard(2, "Y3G2", {"yellow": 3, "green": 2}, points=7),
            3: GolemCard(3, "Y2G3", {"yellow": 2, "green": 3}, points=8),
            4: GolemCard(4, "G4", {"yellow": 0, "green": 4}, points=8),
            5: GolemCard(5, "G5", {"yellow": 0, "green": 5}, points=10)
        }
        
        self.golem_market = random.sample(list(self.golem_deck.values()), 5)
        
        # Initialize players
        self.agent = Player(1)
        self.opponent = Player(2)
        self.current_player = random.choice([self.agent, self.opponent]) # Choose which player to play first
        self.other_player = self.agent if self.current_player == self.opponent else self.opponent
        self.other_player.yellow += 1
        
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
        self.agent_final_points = 0  # Stores final DQN points
        self.opponent_final_points = 0  # Stores final Random points
    
    def _get_obs(self, player):
        # Returns observation from the perspective of the player
        agent = player
        opponent = self.opponent if player == self.agent else self.agent
        
        merchant_cards_state = np.array(agent.merchant_cards, dtype=np.int32)

        merchant_market_state = [card.card_id for card in self.merchant_market]
        while len(merchant_market_state) < 6:
            merchant_market_state.append(9)
        
        golem_market_state = [card.card_id for card in self.golem_market]
        while len(golem_market_state) < 5:
            golem_market_state.append(5)
        
        return {
            "merchant_market": np.array(merchant_market_state, dtype=np.int32),
            "golem_market": np.array(golem_market_state, dtype=np.int32),
            
            "agent_yellow": np.array([agent.yellow], dtype=np.int32),
            "agent_green": np.array([agent.green], dtype=np.int32),
            "agent_merchant_cards": merchant_cards_state,
            "agent_golem_count": np.array([agent.golem_count], dtype=np.int32),
            "agent_points": np.array([agent.points], dtype=np.int32),
            
            "opponent_yellow": np.array([opponent.yellow], dtype=np.int32),
            "opponent_green": np.array([opponent.green], dtype=np.int32),
            "opponent_merchant_cards": np.array(opponent.merchant_cards, dtype=np.int32),
            "opponent_golem_count": np.array([opponent.golem_count], dtype=np.int32),
            "opponent_points": np.array([opponent.points], dtype=np.int32),
        }

    def _get_info(self):
        return {
            "valid_actions": self._get_valid_actions(self.current_player),
            "current_player": int(self.current_player.player_id - 1),  # 0 for agent, 1 for opponent
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
        
        self.current_player = random.choice([self.agent, self.opponent]) # Choose which player to play first
        self.other_player = self.agent if self.current_player == self.opponent else self.opponent
        
        # Reset players
        self.current_player.yellow, self.other_player.yellow = 3, 4
        for player in (self.agent, self.opponent):
            player.green = 0
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
                    if (player.yellow >= card.cost.get("yellow", 0) and
                        player.green >= card.cost.get("green", 0)):
                        valid_actions[i] = 1

        # Get golem card actions
        for i in range(Actions.getG1.value, Actions.getG5.value + 1):
            card_id = i - Actions.getG1.value + 1
            golem_card = self.golem_deck.get(card_id)
            if (golem_card in self.golem_market and
                player.yellow >= golem_card.cost.get("yellow", 0) and
                player.green >= golem_card.cost.get("green", 0)):
                    valid_actions[i] = 1

        return valid_actions    

    # Remove and penalize excess crystals
    def _remove_excess_crystals(self):
        total_crystals = self.current_player.yellow + self.current_player.green
        if total_crystals > 10:
            excess = total_crystals - 10
            excess_yellow = min(excess, self.current_player.yellow)
            excess_green = excess - excess_yellow
            self.current_player.yellow -= excess_yellow
            self.current_player.green -= excess_green
            
            return (0.5 * excess_yellow) + (1 * excess_green)

        return 0
    
    def calculate_total_points(self, player):
        """Calculates total points for a player, including non-yellow crystals."""
        crystal_points = player.green  # Assuming green crystals are worth 1 point
        return player.points + crystal_points
    
    def step(self, action):
        
        if self.render_mode == "text":
            print(f"==== DQN | {Actions(int(action)).name} ====\n") if self.current_player.player_id == 1 else print(f"==== Random | {Actions(int(action)).name} ====\n")
            
        terminated = False
        reward = -0.01
        
        # Store the last action for each player
        if self.current_player == self.agent:
            self.last_action_dqn = Actions(action).name  # Convert action enum to string
        else:
            self.last_action_random = Actions(action).name
        
        # Rest
        if action == Actions.rest.value:
            self.current_player.merchant_cards = [2 if card == 1 else card for card in self.current_player.merchant_cards]
            
            reward -= 0.1
            
        # Get a merchant card
        elif Actions.getM2.value <= action <= Actions.getM9.value:
            card_id = action + 1 # e.g. action 1 = get M2
            
            # Check if card is in market
            if self.merchant_deck[card_id] in self.merchant_market:
                # Execute action
                self.current_player.merchant_cards[action] = 2

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
                
                reward += 1

        # Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM9.value:
            card_idx = action - Actions.useM1.value
            card = self.merchant_deck.get(card_idx + 1)

            if self.current_player.merchant_cards[card_idx] == 2 and card:  # Check if playable
                if card.card_type == "crystal":
                    self.current_player.yellow += card.gain.get('yellow', 0)
                    self.current_player.green += card.gain.get('green', 0)
                    reward += (0.5 * card.gain.get('yellow', 0)) + (1 * card.gain.get('green', 0))
                elif card.card_type == "trade":
                    # Ensure the player has enough crystals to trade before applying changes
                    if (self.current_player.yellow >= card.cost.get("yellow", 0) and
                        self.current_player.green >= card.cost.get("green", 0)):
                        
                        self.current_player.yellow -= card.cost.get("yellow", 0)
                        self.current_player.green -= card.cost.get("green", 0)
                        self.current_player.yellow += card.gain.get("yellow", 0)
                        self.current_player.green += card.gain.get("green", 0)

                        loss = (0.5 * card.cost.get("yellow", 0)) + (1 * card.cost.get("green", 0))
                        gain = (0.5 * card.gain.get("yellow", 0)) + (1 * card.gain.get("green", 0))
                        reward += (gain - loss)

                self.current_player.merchant_cards[card_idx] = 1  # Set to unplayable
        
        # Get a golem card
        elif Actions.getG1.value <= action <= Actions.getG5.value:
            card_id = action - Actions.getG1.value + 1 # e.g. action 12 = get G1
            # Check if golem in market and player has enough crystals
            if (self.golem_deck[card_id] in self.golem_market) and (self.current_player.yellow >= self.golem_deck[card_id].cost["yellow"]) and (self.current_player.green >= self.golem_deck[card_id].cost["green"]):
                # Execute action
                self.current_player.yellow -= self.golem_deck[card_id].cost["yellow"]
                self.current_player.green -= self.golem_deck[card_id].cost["green"]
                self.current_player.golem_count += 1
                self.current_player.points += self.golem_deck[card_id].points
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
                
                reward += self.golem_deck[card_id].points

        reward -= self._remove_excess_crystals()
        
        # Check if the endgame is triggered
        if not self.endgame_triggered and self.current_player.golem_count >= 3:
            self.endgame_triggered = True
            self.endgame_initiator = self.current_player
        # Check if this is the last turn (opponent of endgame initiator)
        if self.endgame_triggered and self.current_player != self.endgame_initiator:
            terminated = True

            # Calculate final points for both players
            self.agent_final_points = self.calculate_total_points(self.agent)
            self.opponent_final_points = self.calculate_total_points(self.opponent)
            
            # If not agent's turn, the reward is reset, before calculation
            if self.current_player != self.agent:
                reward = 0
            
            score_diff = self.agent_final_points - self.opponent_final_points
            if score_diff > 0:
                self.winner = self.agent
                reward += 100
            elif score_diff == 0:
                self.winner = None
                reward += 50
            else:
                self.winner = self.opponent
                reward -= 100
        
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
            print(f"P{self.agent.player_id}")
            print(f"Y: {self.agent.yellow}")
            print(f"G: {self.agent.green}")
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.agent.merchant_cards):
                if card_status == 0:
                    continue
                card = self.merchant_deck.get(i + 1)  # Assuming card_id starts from 1
                print(f"M{i+1}-{card.name}: {status_map[card_status]}")
            print(f"GC: {self.agent.golem_count}")
            print(f"P: {self.agent.points}")
            print("")
            print(f"P{self.opponent.player_id}")
            print(f"Y: {self.opponent.yellow}")
            print(f"G: {self.opponent.green}")
            status_map = {1: "unplayable", 2: "playable"}
            for i, card_status in enumerate(self.opponent.merchant_cards):
                if card_status == 0:
                    continue
                card = self.merchant_deck.get(i + 1)  # Assuming card_id starts from 1
                print(f"M{i+1}-{card.name}: {status_map[card_status]}")
            print(f"GC: {self.opponent.golem_count}")
            print(f"P: {self.opponent.points}")
            print("")
            
    def close(self):
        print("=== CLOSE ENVIRONMENT ===")