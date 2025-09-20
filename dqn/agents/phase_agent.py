import numpy as np
import random

class PhaseAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.game_phase = "early"
        self.round_count = 0
        self.owned_merchant_count = 0
        self.golem_count = 0
        
        # Crystal values
        self.crystal_values = {
            "yellow": 0.3,
            "green": 1.0,
            "blue": 2.5,
            "pink": 4.0
        }
        
        # Weights for different action types in different game phases
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
        valid_actions = info["valid_actions"]
        valid_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_indices) == 1:
            return valid_indices[0]
        
        self._update_game_state(state, info)
        
        action_scores = self._calculate_action_scores(state, info, valid_indices)
        
        # Add some randomness to avoid being too predictable
        randomness = 0.3 if self.game_phase == "early" else 0.15 if self.game_phase == "mid" else 0.05
        action_scores = [score + random.uniform(0, randomness) for score in action_scores]
        
        best_action_idx = np.argmax(action_scores)
        return valid_indices[best_action_idx]

    def _update_game_state(self, state, info):
        if info["current_player"] == 0:  # Player 1
            self.golem_count = state[63]
        else:  # Player 2
            self.golem_count = state[66]
        
        # Count owned merchant cards
        if info["current_player"] == 0:  # Player 1
            self.owned_merchant_count = sum(1 for i in range(14, 59) if state[i] == 1 or state[i] == 2)
        else:  # Player 2
            self.owned_merchant_count = sum(1 for i in range(14, 59) if state[i] == 3 or state[i] == 4)
        
        # Update game phase based on golem count and merchant cards
        if self.golem_count >= 3 or self.round_count > 12:
            self.game_phase = "late"
        elif self.golem_count >= 1 or self.owned_merchant_count >= 4 or self.round_count > 5:
            self.game_phase = "mid"
        else:
            self.game_phase = "early"
            
        if info["current_player"] == 1:
            self.round_count += 0.5

    def _calculate_action_scores(self, state, info, valid_indices):
        action_scores = []
        
        for action in valid_indices:
            score = 0.0
            
            # rest
            if action == 0:
                rest_score = self._evaluate_rest(state, info)
                score = rest_score * self.action_weights[self.game_phase]["rest"]
            
            # getM
            elif 1 <= action <= 43:
                merchant_score = self._evaluate_merchant_card(action, state, info)
                score = merchant_score * self.action_weights[self.game_phase]["get_merchant"]
            
            # useM
            elif 44 <= action <= 88:
                use_score = self._evaluate_use_merchant(action, state, info)
                score = use_score * self.action_weights[self.game_phase]["use_merchant"]
            
            # getG
            elif 89 <= action <= 124:
                golem_score = self._evaluate_golem(action, state, info)
                score = golem_score * self.action_weights[self.game_phase]["get_golem"]
            
            action_scores.append(score)
        
        return action_scores

    def _evaluate_rest(self, state, info):
        # Higher score if have many unplayable cards
        if info["current_player"] == 0: # Player 1
            unplayable_cards = sum(1 for i in range(14, 59) if state[i] == 1)
        else: # Player 2
            unplayable_cards = sum(1 for i in range(14, 59) if state[i] == 3)
            
        score = 0.5 + (unplayable_cards * 0.2)
        
        if self.owned_merchant_count < 3:
            score *= 0.5
        if self.game_phase == "late":
            score *= 0.7
            
        return score

    def _evaluate_merchant_card(self, action, state, info):
        card_id = action + 2  # action 1 = merchant card 3
        
        score = 1.0
        
        # Discourage getting too many cards
        if self.owned_merchant_count > 5:
            score *= 0.7
        if self.owned_merchant_count > 7:
            score *= 0.5
            
        # Prioritise cards based on game phase
        if self.game_phase == "early":
            if card_id in [3, 4, 5, 6, 7, 11]:  # Crystal cards or cheap upgrades
                score *= 1.5
            if card_id in [8, 9, 10]:  # Trade yellow for green
                score *= 1.3
                
        elif self.game_phase == "mid":
            # Cards that give or trade for blue/pink
            if card_id in [12, 13, 14, 15, 16, 17, 18, 26, 27, 28]:  # Blue/Pink related
                score *= 1.4
            if card_id == 11:  # 3 upgrades
                score *= 1.3
                
        else:  # Late game
            # Focus cards that help get high-value golems
            if card_id in [26, 27, 28, 30, 31, 33, 36]:
                score *= 1.6
                
        return score

    def _evaluate_use_merchant(self, action, state, info):
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
        
        # Penalize if close to crystal limit
        if total_crystals >= 8:
            score *= 0.7
            
        # Crystal cards
        if card_id in [1, 3, 4, 5, 6, 7, 12, 13, 45]:
            score *= 1.3
            if total_crystals < 5:
                score *= 1.2
                
        # Upgrade cards
        if card_id in [2, 11]:
            if yellow > 0 or green > 0:
                score *= 1.4
                # More valuable in mid/late game
                if self.game_phase in ["mid", "late"]:
                    score *= 1.2
                    
        # Trade cards
        if card_id in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]:
            if self.game_phase in ["mid", "late"]:
                score *= 1.5
                
        # Blue trades
        if card_id in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
            if self.game_phase == "mid":
                score *= 1.3
                
        return score

    def _evaluate_golem(self, action, state, info):
        golem_id = action - 89 + 1  # action 89 = golem 1
        
        # Base score increases in later phases
        if self.game_phase == "early":
            score = 1.0
        elif self.game_phase == "mid":
            score = 1.5
        else:
            score = 2.0
            
        # Higher score for higher point golems
        if golem_id <= 5:
            score *= 0.8 + (golem_id * 0.05)
        elif golem_id <= 14:
            score *= 1.0 + ((golem_id - 5) * 0.08)
        elif golem_id <= 27:
            score *= 1.2 + ((golem_id - 14) * 0.05)
        else:
            score *= 1.5 + ((golem_id - 27) * 0.1)
            
        if self.golem_count == 4:
            score *= 1.5
            
        if self.golem_count == 2:
            score *= 1.3
            
        return score 