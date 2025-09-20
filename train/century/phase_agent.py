import numpy as np
import random

class StrategicAgent:
    """
    A strategic heuristic agent for the Century: Golem Edition environment.
    Uses game-specific knowledge to make better decisions than random play.
    """
    def __init__(self, action_size):
        self.action_size = action_size
        self.game_phase = "early"  # early, mid, late
        self.round_count = 0
        self.owned_merchant_count = 0  # Track how many merchant cards we own
        self.golem_count = 0  # Track how many golems we've acquired
        
        # Constants for crystal values
        self.crystal_values = {
            "yellow": 0.3,
            "green": 1.0,
            "blue": 2.5,
            "pink": 4.0
        }
        
        # Preference weights for different action types in different game phases
        self.action_weights = {
            "early": {
                "get_merchant": 1.5,
                "use_merchant": 1.0,
                "get_golem": 0.7,
                "rest": 0.5
            },
            "mid": {
                "get_merchant": 0.8,
                "use_merchant": 1.3,
                "get_golem": 1.5,
                "rest": 0.8
            },
            "late": {
                "get_merchant": 0.3,
                "use_merchant": 1.2,
                "get_golem": 2.0,
                "rest": 0.7
            }
        }

    def pick_action(self, state, info):
        """Main method to select an action based on current state and valid actions"""
        valid_actions = info["valid_actions"]
        valid_indices = np.where(valid_actions == 1)[0]
        
        # If only one valid action, take it
        if len(valid_indices) == 1:
            return valid_indices[0]
        
        # Update game phase and other tracking variables
        self._update_game_state(state, info)
        
        # Apply strategic weights to each valid action
        action_scores = self._calculate_action_scores(state, info, valid_indices)
        
        # Add some randomness to avoid being too predictable
        # Lower randomness in late game for more optimal play
        randomness = 0.3 if self.game_phase == "early" else 0.15 if self.game_phase == "mid" else 0.05
        action_scores = [score + random.uniform(0, randomness) for score in action_scores]
        
        # Select the action with the highest score
        best_action_idx = np.argmax(action_scores)
        return valid_indices[best_action_idx]

    def _update_game_state(self, state, info):
        """Update internal tracking of game state"""
        # Extract golem count from state
        # In v16, player golem count depends on current player
        if info["current_player"] == 0:  # Player 1
            self.golem_count = state[63]  # player1_golem_count - need to verify index
        else:  # Player 2
            self.golem_count = state[66]  # player2_golem_count - need to verify index
        
        # Count owned merchant cards
        # Indices 14-58 contain merchant_cards_status
        if info["current_player"] == 0:  # Player 1
            # Count cards where status is 1 (unplayable) or 2 (playable) for P1
            self.owned_merchant_count = sum(1 for i in range(14, 59) if state[i] == 1 or state[i] == 2)
        else:  # Player 2
            # Count cards where status is 3 (unplayable) or 4 (playable) for P2
            self.owned_merchant_count = sum(1 for i in range(14, 59) if state[i] == 3 or state[i] == 4)
        
        # Update game phase based on golem count and merchant cards
        if self.golem_count >= 3 or self.round_count > 12:
            self.game_phase = "late"
        elif self.golem_count >= 1 or self.owned_merchant_count >= 4 or self.round_count > 5:
            self.game_phase = "mid"
        else:
            self.game_phase = "early"
            
        # Increment round count (approximation)
        if info["current_player"] == 1:  # Player 2's turn
            self.round_count += 0.5  # Each full round is player 1 and 2 taking turns

    def _calculate_action_scores(self, state, info, valid_indices):
        """Calculate a strategic score for each valid action"""
        action_scores = []
        
        for action in valid_indices:
            score = 0.0
            
            # REST action (0)
            if action == 0:
                rest_score = self._evaluate_rest(state, info)
                score = rest_score * self.action_weights[self.game_phase]["rest"]
            
            # GET MERCHANT CARD actions (1-43)
            elif 1 <= action <= 43:
                merchant_score = self._evaluate_merchant_card(action, state, info)
                score = merchant_score * self.action_weights[self.game_phase]["get_merchant"]
            
            # USE MERCHANT CARD actions (44-88)
            elif 44 <= action <= 88:
                use_score = self._evaluate_use_merchant(action, state, info)
                score = use_score * self.action_weights[self.game_phase]["use_merchant"]
            
            # GET GOLEM CARD actions (89-124)
            elif 89 <= action <= 124:
                golem_score = self._evaluate_golem(action, state, info)
                score = golem_score * self.action_weights[self.game_phase]["get_golem"]
            
            action_scores.append(score)
        
        return action_scores

    def _evaluate_rest(self, state, info):
        """Evaluate how valuable the rest action is"""
        # Higher score if we have many unplayable cards
        if info["current_player"] == 0:  # Player 1
            unplayable_cards = sum(1 for i in range(14, 59) if state[i] == 1)  # Status 1 = P1 unplayable
        else:  # Player 2
            unplayable_cards = sum(1 for i in range(14, 59) if state[i] == 3)  # Status 3 = P2 unplayable
            
        # Base score plus bonus for each unplayable card
        score = 0.5 + (unplayable_cards * 0.2)
        
        # Discourage resting if we have few cards or in late game
        if self.owned_merchant_count < 3:
            score *= 0.5
        if self.game_phase == "late":
            score *= 0.7
            
        return score

    def _evaluate_merchant_card(self, action, state, info):
        """Evaluate how valuable a merchant card acquisition is"""
        card_id = action + 2  # action 1 = merchant card 3
        
        # Base score for any card
        score = 1.0
        
        # Discourage getting too many cards
        if self.owned_merchant_count > 5:
            score *= 0.7
        if self.owned_merchant_count > 7:
            score *= 0.5
            
        # Prioritize cards based on game phase
        if self.game_phase == "early":
            # In early game, value cards that give crystals or cheap upgrades
            if card_id in [3, 4, 5, 6, 7, 11]:  # Crystal cards or cheap upgrades
                score *= 1.5
            # Also value cards that convert yellow to green
            if card_id in [8, 9, 10]:  # Trade yellow for green
                score *= 1.3
                
        elif self.game_phase == "mid":
            # In mid game, value cards that give or trade for blue/pink
            if card_id in [12, 13, 14, 15, 16, 17, 18, 26, 27, 28]:  # Blue/Pink related
                score *= 1.4
            # Also value good upgrade cards
            if card_id == 11:  # 3 upgrades
                score *= 1.3
                
        else:  # Late game
            # In late game, value cards that directly help get high-value golems
            if card_id in [26, 27, 28, 30, 31, 33, 36]:  # Cards that give/trade for pink
                score *= 1.6
                
        return score

    def _evaluate_use_merchant(self, action, state, info):
        """Evaluate how valuable using a merchant card is"""
        card_id = action - 44 + 1  # action 44 = use card 1
        
        # Base score
        score = 1.0
        
        # Get current crystal inventory
        if info["current_player"] == 0:  # Player 1
            yellow = state[59]
            green = state[60]
            blue = state[61]
            pink = state[62]
        else:  # Player 2
            yellow = state[64]
            green = state[65]
            blue = state[66]
            pink = state[67]
            
        total_crystals = yellow + green + blue + pink
        
        # Penalize if we're close to crystal limit (10)
        if total_crystals >= 8:
            score *= 0.7
            
        # Crystal cards - always good, better when we have few crystals
        if card_id in [1, 3, 4, 5, 6, 7, 12, 13, 45]:
            score *= 1.3
            if total_crystals < 5:
                score *= 1.2
                
        # Upgrade cards - valuable when we have yellow/green to upgrade
        if card_id in [2, 11]:
            if yellow > 0 or green > 0:
                score *= 1.4
                # More valuable in mid/late game
                if self.game_phase in ["mid", "late"]:
                    score *= 1.2
                    
        # Trade cards - evaluate based on what we need
        # Pink-producing trades are very valuable in mid/late game
        if card_id in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]:
            if self.game_phase in ["mid", "late"]:
                score *= 1.5
                
        # Blue-producing trades are good in mid game
        if card_id in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
            if self.game_phase == "mid":
                score *= 1.3
                
        return score

    def _evaluate_golem(self, action, state, info):
        """Evaluate how valuable a golem acquisition is"""
        golem_id = action - 89 + 1  # action 89 = golem 1
        
        # Base score increases in later game phases
        if self.game_phase == "early":
            score = 1.0
        elif self.game_phase == "mid":
            score = 1.5
        else:  # Late game
            score = 2.0
            
        # Higher score for higher point golems
        if golem_id <= 5:  # 6-10 points (yellow/green golems)
            score *= 0.8 + (golem_id * 0.05)  # 0.85-1.05
        elif golem_id <= 14:  # 8-15 points (blue golems)
            score *= 1.0 + ((golem_id - 5) * 0.08)  # 1.08-1.72
        elif golem_id <= 27:  # 9-16 points (lower pink golems)
            score *= 1.2 + ((golem_id - 14) * 0.05)  # 1.25-1.85
        else:  # 16-20 points (higher pink golems)
            score *= 1.5 + ((golem_id - 27) * 0.1)  # 1.6-2.4
            
        # Extra value for getting 5th golem (triggers end game)
        if self.golem_count == 4:
            score *= 1.5
            
        # Extra value for getting 3rd golem (strategic point)
        if self.golem_count == 2:
            score *= 1.3
            
        return score 