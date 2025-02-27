from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

class Player:
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.yellow = 3
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
    metadata = {"render_modes": ["text", "human"], "render_fps": 1}
    
    def __init__(self, render_mode=None):
        
        self.action_space = spaces.Discrete(17)
        
        # Open information
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([7, 7, 7, 7, 7]),  # Allow 7 as placeholder for empty
            "golem_market": spaces.MultiDiscrete([6, 6, 6, 6, 6]),  # Allow 5 as placeholder for empty
            
            "agent_yellow": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "agent_green": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "agent_merchant_cards": spaces.MultiDiscrete([3] * 6),
            "agent_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),  # New: number of golem cards owned
            "agent_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            
            "opponent_yellow": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "opponent_green": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
            "opponent_merchant_cards": spaces.MultiDiscrete([3] * 6),
            "opponent_golem_count": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            "opponent_points": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            
            "current_player": spaces.Discrete(2),
            
            "valid_actions": spaces.MultiBinary(self.action_space.n)
        })
        
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
        self.window_size = 600  # Size of the render window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def _get_obs(self, player):
        # Returns observation from the perspective of the player
        agent = player
        opponent = self.opponent if player == self.agent else self.agent
        
        merchant_cards_state = np.array(agent.merchant_cards, dtype=np.int32)

        merchant_market_state = [card.card_id for card in self.merchant_market]
        while len(merchant_market_state) < 5:
            merchant_market_state.append(6)
        
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
            
            "current_player": int(self.current_player.player_id - 1),  # Convert 1/2 to 0/1
        
            "valid_actions": self._get_valid_actions(player)
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
        
        self.current_player = random.choice([self.agent, self.opponent]) # Choose which player to play first
        self.other_player = self.agent if self.current_player == self.opponent else self.opponent
        
        # Reset players
        self.current_player.yellow, self.other_player.yellow = 3, 4
        for player in (self.agent, self.opponent):
            player.green = 0
            player.merchant_cards = [2] + [0] * 5
            player.golem_count = 0
            player.points = 0
            
        self.endgame_triggered = False  # Tracks if endgame was triggered
        self.endgame_initiator = None   # The player who triggered the endgame
        
        self.turn_counter = 0
        self.round = 1
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()
        
        # Generate valid actions for the starting state
        valid_actions = self._get_valid_actions(self.current_player)
        info["valid_actions"] = valid_actions
        info["current_player"] = int(self.current_player.player_id - 1)

        if self.render_mode != None:
            self.render()

        return observation, info
    
    def _get_valid_actions(self, player):
        """Returns a binary mask indicating valid actions for the given player."""
        valid_actions = np.zeros(self.action_space.n, dtype=np.int32)

        # Rest is valid, unless any merchant card is unplayable
        valid_actions[Actions.rest.value] = 0
        for i in range(Actions.useM1.value, Actions.useM6.value + 1):
            card_idx = i - Actions.useM1.value
            if player.merchant_cards[card_idx] == 1:  # if owned but unplayable
                valid_actions[Actions.rest.value] = 1
                break

        # Get merchant card actions
        for i in range(Actions.getM2.value, Actions.getM6.value + 1):
            card_id = i + 1
            if self.merchant_deck[card_id] in self.merchant_market:
                valid_actions[i] = 1

        # Use merchant card actions
        for i in range(Actions.useM1.value, Actions.useM6.value + 1):
            card_idx = i - Actions.useM1.value

            if player.merchant_cards[card_idx] == 2:  # If playable
                valid_actions[i] = 1

        # Get golem card actions
        for i in range(Actions.getG1.value, Actions.getG5.value + 1):
            card_id = i - Actions.getG1.value + 1
            golem_card = self.golem_deck.get(card_id)
            if golem_card and golem_card.cost:
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
            penalty = 1.0 * excess
            # Remove excess starting with yellow, then green
            if self.current_player.yellow >= excess:
                self.current_player.yellow -= excess
            else:
                excess -= self.current_player.yellow
                self.current_player.yellow = 0
                self.current_player.green = max(0, self.current_player.green - excess)
            
            return penalty
        
        return 0
    
    def calculate_total_points(self, player):
        """Calculates total points for a player, including non-yellow crystals."""
        crystal_points = player.green  # Assuming green crystals are worth 1 point
        return player.points + crystal_points
    
    def step(self, action):
        
        if self.render_mode == "text":
            print(f"==== DQN | {Actions(int(action)).name} ====\n") if self.current_player.player_id == 1 else print(f"==== Random | {Actions(int(action)).name} ====\n")
            
        reward = -0.5  # Base time-step penalty
        terminated = False
        
        # Rest
        if action == Actions.rest.value:
            self.current_player.merchant_cards = [2 if card == 1 else card for card in self.current_player.merchant_cards]
            
        # Get a merchant card
        elif Actions.getM2.value <= action <= Actions.getM6.value:
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
                
                reward += 6.0 + (0.8 * self.merchant_deck[card_id].gain['yellow']) + (1.3 * self.merchant_deck[card_id].gain['green'])
            else:
                reward -= 1.0
                
        # Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM6.value:
            card_id, card_idx = action - 5, action - 6 # e.g. action 10 = use M5 = card_id 5 = card_idx 4
            if self.current_player.merchant_cards[card_idx] == 2: # if card is playable    
                self.current_player.yellow += self.merchant_deck[card_id].gain['yellow']
                self.current_player.green += self.merchant_deck[card_id].gain['green']
                self.current_player.merchant_cards[card_idx] = 1 # set card status to owned but unplayable

                reward += (0.4 * self.merchant_deck[card_id].gain['yellow'] + 1.2 * self.merchant_deck[card_id].gain['green'])
            else:
                reward -= 1.0
        
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

                # Reward for getting the golem card
                reward += 18.0 + self.golem_deck[card_id].points
                
                # Reward for blocking opponent from getting the golem card
                if self.other_player.yellow >= self.golem_deck[card_id].cost["yellow"] and self.other_player.green >= self.golem_deck[card_id].cost["green"]:
                    reward += 5.0
            else:
                reward -= 1.0

        reward -= self._remove_excess_crystals()
        
        # Check if the endgame is triggered
        if not self.endgame_triggered and self.current_player.golem_count >= 3:
            self.endgame_triggered = True
            self.endgame_initiator = self.current_player
        # Check if this is the last turn (opponent of endgame initiator)
        if self.endgame_triggered and self.current_player != self.endgame_initiator:
            terminated = True

            # Calculate final points for both players
            agent_final_points = self.calculate_total_points(self.agent)
            opponent_final_points = self.calculate_total_points(self.opponent)
            
            # If not agent's turn, the reward is reset, before calculation
            if self.current_player != self.agent:
                reward = 0
            
            # Endgame reward
            score_diff = agent_final_points - opponent_final_points
            if score_diff > 0:
                reward += 100 + (1.5 * score_diff)
            elif score_diff == 0:
                reward += 50  # Tie
            else:
                reward -= max(50 - (0.5 * abs(score_diff)), 30)
        
        # Switch turn
        self.current_player, self.other_player = self.other_player, self.current_player

        # Turn and Round
        if not terminated:
            self.turn_counter += 1  
            if self.turn_counter % 2 == 0:
                self.round += 1
        
        observation = self._get_obs(self.current_player)
        info = self._get_info()
        
        # Generate valid actions after the step
        valid_actions = self._get_valid_actions(self.current_player)
        info["valid_actions"] = valid_actions
        info["current_player"] = int(self.current_player.player_id - 1)
        
        if self.render_mode != None:
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        
        if self.render_mode == "human":
            self._render_frame()
            
        elif self.render_mode == "text":               
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

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            print("HUMAN RENDER")
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a surface to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Display round number at the top-left
        font = pygame.font.Font(None, 24)  # Set font size
        round_text = font.render(f"ROUND: {self.round}", True, (0, 0, 0))  # Render round text
        canvas.blit(round_text, (10, 10))  # Position at (10,10) from top-left

        # Define card dimensions
        card_width = 66   # Slightly wider for better layout
        card_height = 96  # Adjust height to fit points and crystals
        margin = 15       # Reduce spacing between cards

        # Draw golem cards
        for i, golem_card in enumerate(self.golem_market):
            x = margin + i * (card_width + margin)
            y = 50  # Fixed y position

            # Draw card background
            pygame.draw.rect(canvas, (200, 200, 200), (x, y, card_width, card_height), border_radius=10)
            pygame.draw.rect(canvas, (0, 0, 0), (x, y, card_width, card_height), width=3, border_radius=10)
            
            # Draw card points at the top center
            font = pygame.font.Font(None, 24)  # Set font size
            points_text = font.render(str(golem_card.points), True, (0, 0, 0))  # Render points text
            text_x = x + (card_width // 2) - (points_text.get_width() // 2)
            text_y = y + 8  # Place at top of the card
            canvas.blit(points_text, (text_x, text_y))

            # Set starting position for crystals
            cost_x = x + 15  # Left padding
            cost_y = y + 35  # Below points text
            circle_radius = 7  # Keep small size

            # Rearrange costs in two rows (up to 3 per row)
            max_per_row = 3
            row_offset = 18  # Vertical spacing between rows
            col_offset = 18  # Horizontal spacing between circles

            col = 0  # Track position in row
            row = 0  # Track row index

            for color, amount in golem_card.cost.items():
                for _ in range(amount):
                    if color == "yellow":
                        pygame.draw.circle(canvas, (255, 215, 0), (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)
                    elif color == "green":
                        pygame.draw.circle(canvas, (0, 128, 0), (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)

                    col += 1  # Move to the next column
                    if col >= max_per_row:  # If row is full, move to next row
                        col = 0
                        row += 1

        # Display updates
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def close(self):
        print("=== CLOSE ENVIRONMENT ===")