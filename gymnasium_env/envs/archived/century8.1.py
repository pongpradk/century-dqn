'''
Century: Golem Edition
Version 8.0
Changes:
- merchant trade cards
- tokens
- redefined rewards
- valid actions masking
'''

import random
import pygame
import cv2
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
    getM3 = 2 # ADD +100 and -100 for reaching max_timesteps in DQN training code, else +50 for tie
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
    
    def __init__(self, render_mode=None, record_session=False):
        
        self.action_space = spaces.Discrete(17)
        
        # Open information
        self.observation_space = spaces.Dict({
            "merchant_market": spaces.MultiDiscrete([7, 7, 7, 7, 7, 7]),
            "golem_market": spaces.MultiDiscrete([6, 6, 6, 6, 6]),
            
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
            merchant_market_state.append(7)
        
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
        }

    def _get_info(self):
        return {
            "valid_actions": self._get_valid_actions(self.current_player),
            "current_player": int(self.current_player.player_id - 1),  # 0 for agent, 1 for opponent
        }
    
    def reset(self, seed=None, options=None):
        
        print("Century: Golem Edition | Version 8.0")
        
        super().reset(seed=seed)
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 6 # draw 3 cards to market, excluding M1
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

        if self.render_mode != None:
            self.render()

        return observation, info
    
    def _get_valid_actions(self, player):

        valid_actions = np.zeros(self.action_space.n, dtype=np.int32)

        valid_actions[Actions.rest.value] = 0
        # Rest is valid if any merchant card is unplayable
        for i in range(Actions.useM1.value, Actions.useM6.value + 1):
            card_idx = i - Actions.useM1.value
            if player.merchant_cards[card_idx] == 1:  # If owned but unplayable
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
            if player.merchant_cards[card_idx] == 2:  # if playable
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
                
                reward += 1
                
        # Use a merchant card
        elif Actions.useM1.value <= action <= Actions.useM6.value:
            card_id, card_idx = action - 5, action - 6 # e.g. action 10 = use M5 = card_id 5 = card_idx 4
            if self.current_player.merchant_cards[card_idx] == 2: # if card is playable    
                self.current_player.yellow += self.merchant_deck[card_id].gain['yellow']
                self.current_player.green += self.merchant_deck[card_id].gain['green']
                self.current_player.merchant_cards[card_idx] = 1 # set card status to owned but unplayable
                
                reward += (0.5 * self.merchant_deck[card_id].gain['yellow']) + (1 * self.merchant_deck[card_id].gain['green'])
        
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
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a surface to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Light birch wood color (beige)
        
        # === ROUND & TURN ===
        
        # Display round and player turn at the top-left
        font = pygame.font.Font(None, 24)
        round_text = font.render(f"ROUND: {self.round}", True, (0, 0, 0))
        turn_text = font.render(f"TURN : {'DQN' if self.current_player == self.agent else 'Random'}", True, (0, 0, 0))

        canvas.blit(round_text, (10, 10))  # Move round number higher
        canvas.blit(turn_text, (10, 30))  # Keep turn text below round number
        
        # Ensure y is initialized before usage
        y = 75  # Fixed y position for the golem cards
        
        # === MARKET LABELS ===
        font_market = pygame.font.Font(None, 24)  # Define font size
        # Golem Market Label
        golem_market_text = font_market.render("GOLEM MARKET", True, (0, 0, 0))
        canvas.blit(golem_market_text, (self.window_size // 2 - golem_market_text.get_width() // 2, y - 25))

        # === GOLEM CARDS IN MARKET ===

        # Define card dimensions
        card_width = 66   # Slightly wider for better layout
        card_height = 96  # Adjust height to fit points and crystals
        margin = 15       # Reduce spacing between cards

        # Draw golem cards
        for i, golem_card in enumerate(self.golem_market):
            x = margin + i * (card_width + margin)

            # Draw card background
            pygame.draw.rect(canvas, (69, 69, 69), (x, y, card_width, card_height), border_radius=10)
            pygame.draw.rect(canvas, (176, 176, 176), (x, y, card_width, card_height), width=3, border_radius=10)

            # Draw letter "G" in blue at the center of the card (underneath crystals and points)
            font_large = pygame.font.Font(None, 50)  # Larger font size
            g_text = font_large.render("G", True, (92, 92, 92, 100))  # Blue color, semi-transparent
            g_text_x = x + (card_width // 2) - (g_text.get_width() // 2)
            g_text_y = y + (card_height // 2) - (g_text.get_height() // 2)
            canvas.blit(g_text, (g_text_x, g_text_y))
            
            # Draw card points at the top center
            font = pygame.font.Font(None, 24)  # Set font size
            points_text = font.render(str(golem_card.points), True, (255, 255, 255))  # Render points text
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

        # === MERCHANT CARDS IN MARKET ===

        # Define merchant card positions
        merchant_y = y + card_height + 30  # Below golem cards with spacing
        
        # Merchant Market Label
        merchant_market_text = font_market.render("MERCHANT MARKET", True, (0, 0, 0))
        canvas.blit(merchant_market_text, (self.window_size // 2 - merchant_market_text.get_width() // 2, merchant_y - 25))

        # Draw merchant cards
        for i, merchant_card in enumerate(self.merchant_market):
            x = margin + i * (card_width + margin)

            # Draw card background
            pygame.draw.rect(canvas, (69, 69, 69), (x, merchant_y, card_width, card_height), border_radius=10)
            pygame.draw.rect(canvas, (176, 176, 176), (x, merchant_y, card_width, card_height), width=3, border_radius=10)
            
            # Draw letter "G" in blue at the center of the card (underneath crystals and points)
            font_large = pygame.font.Font(None, 50)  # Larger font size
            m_text = font_large.render("M", True, (92, 92, 92, 100))  # Blue color, semi-transparent
            m_text_x = x + (card_width // 2) - (g_text.get_width() // 2)
            m_text_y = merchant_y + (card_height // 2) - (g_text.get_height() // 2)
            canvas.blit(m_text, (m_text_x, m_text_y))

            # Set starting position for crystals (at the top of the card)
            cost_x = x + 15  # Left padding
            cost_y = merchant_y + 15  # Move up since no points

            circle_radius = 7
            col = 0  # Track position in row
            row = 0  # Track row index
            max_per_row = 3  # Allow up to 3 per row
            row_offset = 18  # Vertical spacing between rows
            col_offset = 18  # Horizontal spacing between circles

            for color, amount in merchant_card.gain.items():  # Merchant cards show gain instead of cost
                for _ in range(amount):
                    if color == "yellow":
                        pygame.draw.circle(canvas, (255, 215, 0), (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)
                    elif color == "green":
                        pygame.draw.circle(canvas, (0, 128, 0), (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)

                    col += 1  # Move to next column
                    if col >= max_per_row:
                        col = 0
                        row += 1
        
        # === SEPARATOR LINE ===
        separator_y = merchant_y + card_height + 10  # Position under merchant cards
        pygame.draw.line(canvas, (0, 0, 0), (10, separator_y), (self.window_size - 10, separator_y), 3)  # Black horizontal line
        
        # === DQN PLAYER'S NAME POSITION ===
        player_start_y = separator_y + 15  # Space below the separator
        
        # === DQN PLAYER'S NAME WITH LATEST ACTION DISPLAY ===
        turn_arrow = "-> " if self.current_player == self.agent else ""  # Show arrow if it's DQN's turn
        last_action_display = f" | {self.last_action_dqn}" if self.current_player == self.opponent and hasattr(self, 'last_action_dqn') else ""
        agent_label = font.render(f"{turn_arrow}DQN{last_action_display}", True, (0, 0, 0))
        canvas.blit(agent_label, (10, player_start_y))

        # === DQN PLAYER'S CRYSTALS BELOW NAME ===
        player_crystals_y = player_start_y + 28  # Move crystals below the name
        x_offset = 18  # Align circles under the name
        y_offset = player_crystals_y
        
        circle_radius = 7
        circle_spacing = 16

        for _ in range(self.agent.yellow):
            pygame.draw.circle(canvas, (255, 215, 0), (x_offset, y_offset), circle_radius)  # Yellow
            x_offset += circle_spacing
        for _ in range(self.agent.green):
            pygame.draw.circle(canvas, (0, 128, 0), (x_offset, y_offset), circle_radius)  # Green
            x_offset += circle_spacing
        
        # === DQN's OWNED MERCHANT CARDS ===
        merchant_card_y = player_start_y + 45  # Below player crystals
        merchant_card_x = 10  # Start position for displaying cards
        merchant_card_width = 50  # Smaller than market cards
        merchant_card_height = 70
        merchant_card_margin = 10

        # Filter all owned merchant cards (both playable and unplayable)
        owned_merchant_cards = [(self.merchant_deck[i + 1], status) for i, status in enumerate(self.agent.merchant_cards) if status > 0]

        # Draw merchant cards
        for i, (merchant_card, status) in enumerate(owned_merchant_cards):
            x = merchant_card_x + i * (merchant_card_width + merchant_card_margin)

            # Determine color based on playability
            is_playable = status == 2  # 2 = playable, 1 = unplayable
            card_color = (69, 69, 69) if is_playable else (100, 100, 100)  # Darker for unplayable
            border_color = (176, 176, 176) if is_playable else (140, 140, 140)  # Greyed border

            # Draw card background
            pygame.draw.rect(canvas, card_color, (x, merchant_card_y, merchant_card_width, merchant_card_height), border_radius=10)
            pygame.draw.rect(canvas, border_color, (x, merchant_card_y, merchant_card_width, merchant_card_height), width=2, border_radius=10)

            # Draw "M" in the center of the card (Faded for unplayable)
            font_large = pygame.font.Font(None, 36)
            m_text_color = (200, 200, 200) if is_playable else (150, 150, 150)  # Lighter for unplayable
            m_text = font_large.render("M", True, m_text_color)
            text_x = x + (merchant_card_width // 2) - (m_text.get_width() // 2)
            text_y = merchant_card_y + (merchant_card_height // 2) - (m_text.get_height() // 2)
            canvas.blit(m_text, (text_x, text_y))

            # Draw the merchant card's crystals (Faded for unplayable)
            cost_x = x + 10  # Left padding
            cost_y = merchant_card_y + 10  # Top padding
            circle_radius = 5  # Smaller circle size
            col = 0  # Track column
            row = 0  # Track row index
            max_per_row = 2  # Two per row
            row_offset = 14  # Vertical spacing
            col_offset = 14  # Horizontal spacing

            for color, amount in merchant_card.gain.items():
                for _ in range(amount):
                    faded_color = (255, 215, 0) if color == "yellow" else (0, 128, 0)  # Normal for playable
                    if not is_playable:
                        faded_color = (200, 200, 100) if color == "yellow" else (100, 180, 100)  # Lightened for unplayable
                    
                    pygame.draw.circle(canvas, faded_color, (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)

                    col += 1
                    if col >= max_per_row:
                        col = 0
                        row += 1

        # === DQN's GOLEM COUNT & POINTS ===
        golem_info_y = merchant_card_y + merchant_card_height + 15  # Below merchant cards

        # Define font
        font = pygame.font.Font(None, 24)

        # Create text surfaces
        golem_count_text = font.render(f"Golem Count: {self.agent.golem_count}", True, (0, 0, 0))
        golem_points_text = font.render(f"Golem Points: {self.agent.points}", True, (0, 0, 0))

        # Display text on the canvas
        canvas.blit(golem_count_text, (10, golem_info_y))
        canvas.blit(golem_points_text, (10, golem_info_y + 25))  # Points below count
        
        # === SEPARATOR LINE ===
        separator_y = golem_info_y + 50  # Space below DQN’s info
        pygame.draw.line(canvas, (0, 0, 0), (10, separator_y), (self.window_size - 10, separator_y), 3)  # Black horizontal line
        
        # === RANDOM PLAYER'S NAME POSITION ===
        random_name_y = separator_y + 15  # Space below separator
        
        # === RANDOM PLAYER'S NAME WITH LATEST ACTION DISPLAY ===
        turn_arrow = "-> " if self.current_player == self.opponent else ""  # Show arrow if it's Random's turn
        last_action_display = f" | {self.last_action_random}" if self.current_player == self.agent and hasattr(self, 'last_action_random') else ""
        random_label = font.render(f"{turn_arrow}Random{last_action_display}", True, (0, 0, 0))
        canvas.blit(random_label, (10, random_name_y))

        # === RANDOM PLAYER'S CRYSTALS BELOW NAME ===
        random_crystals_y = random_name_y + 28  # Move crystals below the name
        x_offset = 18  # Align circles under the name
        y_offset = random_crystals_y
        
        circle_radius = 7
        circle_spacing = 16

        for _ in range(self.opponent.yellow):
            pygame.draw.circle(canvas, (255, 215, 0), (x_offset, y_offset), circle_radius)  # Yellow
            x_offset += circle_spacing
        for _ in range(self.opponent.green):
            pygame.draw.circle(canvas, (0, 128, 0), (x_offset, y_offset), circle_radius)  # Green
            x_offset += circle_spacing
        
        # === RANDOM PLAYER'S OWNED MERCHANT CARDS (PLAYABLE & UNPLAYABLE) ===
        random_merchant_card_y = random_crystals_y + 20  # Below separator
        random_merchant_card_x = 10  # Start position for displaying cards
        random_merchant_card_width = 50  # Same size as DQN's merchant cards
        random_merchant_card_height = 70
        random_merchant_card_margin = 10

        # Filter all owned merchant cards (both playable and unplayable)
        random_owned_merchant_cards = [(self.merchant_deck[i + 1], status) for i, status in enumerate(self.opponent.merchant_cards) if status > 0]

        # Draw merchant cards
        for i, (merchant_card, status) in enumerate(random_owned_merchant_cards):
            x = random_merchant_card_x + i * (random_merchant_card_width + random_merchant_card_margin)

            # Determine color based on playability
            is_playable = status == 2  # 2 = playable, 1 = unplayable
            card_color = (69, 69, 69) if is_playable else (100, 100, 100)  # Darker for unplayable
            border_color = (176, 176, 176) if is_playable else (140, 140, 140)  # Greyed border

            # Draw card background
            pygame.draw.rect(canvas, card_color, (x, random_merchant_card_y, random_merchant_card_width, random_merchant_card_height), border_radius=10)
            pygame.draw.rect(canvas, border_color, (x, random_merchant_card_y, random_merchant_card_width, random_merchant_card_height), width=2, border_radius=10)

            # Draw "M" in the center of the card (Faded for unplayable)
            font_large = pygame.font.Font(None, 36)
            m_text_color = (200, 200, 200) if is_playable else (150, 150, 150)  # Lighter for unplayable
            m_text = font_large.render("M", True, m_text_color)
            text_x = x + (random_merchant_card_width // 2) - (m_text.get_width() // 2)
            text_y = random_merchant_card_y + (random_merchant_card_height // 2) - (m_text.get_height() // 2)
            canvas.blit(m_text, (text_x, text_y))

            # Draw the merchant card's crystals (Faded for unplayable)
            cost_x = x + 10  # Left padding
            cost_y = random_merchant_card_y + 10  # Top padding
            circle_radius = 5  # Smaller circle size
            col = 0  # Track column
            row = 0  # Track row index
            max_per_row = 2  # Two per row
            row_offset = 14  # Vertical spacing
            col_offset = 14  # Horizontal spacing

            for color, amount in merchant_card.gain.items():
                faded_color = (255, 215, 0) if color == "yellow" else (0, 128, 0)  # Normal for playable
                if not is_playable:
                    faded_color = (200, 200, 100) if color == "yellow" else (100, 180, 100)  # Lightened for unplayable
                
                pygame.draw.circle(canvas, faded_color, (cost_x + col * col_offset, cost_y + row * row_offset), circle_radius)

                col += 1
                if col >= max_per_row:
                    col = 0
                    row += 1
                    
        # === RANDOM PLAYER'S GOLEM COUNT & POINTS ===
        random_golem_info_y = random_merchant_card_y + random_merchant_card_height + 15  # Below merchant cards

        # Create text surfaces
        random_golem_count_text = font.render(f"Golem Count: {self.opponent.golem_count}", True, (0, 0, 0))
        random_golem_points_text = font.render(f"Golem Points: {self.opponent.points}", True, (0, 0, 0))

        # Display text on the canvas
        canvas.blit(random_golem_count_text, (10, random_golem_info_y))
        canvas.blit(random_golem_points_text, (10, random_golem_info_y + 25))  # Points below count

        # === FINAL GAME RESULT (SHOW ONLY IF GAME HAS ENDED) ===
        if self.winner:
            result_separator_y = random_golem_info_y + 50  # Space below Random’s info
            pygame.draw.line(canvas, (0, 0, 0), (10, result_separator_y), (self.window_size - 10, result_separator_y), 3)  # Black horizontal line
            
            # Define font for final result
            font_large = pygame.font.Font(None, 24)
            font_winner = pygame.font.Font(None, 24)

            # Construct text
            score_text = f"Player DQN    vs    Player Random"
            points_text = f"{self.agent_final_points}                     {self.opponent_final_points}"
            if self.winner == self.agent:
                winner_text = f"WINNER: Player DQN"
            elif self.winner == self.opponent:
                winner_text = f"WINNER: Player Random"
            else:
                winner_text = "TIE"

            # Render text
            score_surface = font_large.render(score_text, True, (0, 0, 0))
            points_surface = font_large.render(points_text, True, (0, 0, 0))
            winner_surface = font_winner.render(winner_text, True, (201, 134, 0))  # Gold text for emphasis

            # Centered positioning
            center_x = self.window_size // 2
            result_y = result_separator_y + 20

            canvas.blit(score_surface, (center_x - score_surface.get_width() // 2, result_y))
            canvas.blit(points_surface, (center_x - points_surface.get_width() // 2, result_y + 30))
            canvas.blit(winner_surface, (center_x - winner_surface.get_width() // 2, result_y + 70))

            pygame.display.update()

            # === RECORD FINAL FRAMES TO VIDEO (EXTRA FRAMES FOR WINNER SCREEN) ===
            if self.record_session:
                for _ in range(self.fps * 3):  # Record for 3 seconds (adjust if needed)
                    frame = pygame.surfarray.array3d(pygame.display.get_surface())  # Capture screen
                    frame = np.transpose(frame, (1, 0, 2))  # Correct dimension order for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    self.video_writer.write(frame)  # Save extra frames
                    pygame.time.delay(1000 // self.fps)  # Delay to maintain frame rate
        
        # === RECORD FRAME INTO VIDEO ===
        if self.record_session:
            # Capture screen pixels properly
            frame = pygame.surfarray.array3d(pygame.display.get_surface())  # Capture full display
            frame = np.transpose(frame, (1, 0, 2))  # Correct dimension order for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            # Initialize video writer when the first frame is captured
            if self.video_writer is None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                self.video_writer = cv2.VideoWriter(self.video_output_path, fourcc, self.fps, (width, height))

            self.video_writer.write(frame)  # Write frame to video

        # Display updates
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
            
    def close(self):
        if self.record_session and self.video_writer is not None:
            print("Finalizing video recording...")

            # If game ended, add a few more frames to show winner
            if self.winner:
                for _ in range(self.fps * 3):  # Record extra 3 seconds
                    frame = pygame.surfarray.array3d(pygame.display.get_surface())
                    frame = np.transpose(frame, (1, 0, 2))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(frame)
                    pygame.time.delay(1000 // self.fps)

            self.video_writer.release()  # Save and finalize MP4
            print(f"Video saved as {self.video_output_path}")

        pygame.display.quit()
        pygame.quit()
        print("=== CLOSE ENVIRONMENT ===")