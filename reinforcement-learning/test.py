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
    
    def __str__(self):
        return f"ID: {self.card_id}, Name: {self.name}, Type: {self.card_type}, Gain: {self.gain}, Cost: {self.cost}, Owned: {self.owned}"
    
    def __repr__(self):
        return self.__str__()

class Actions():
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
    

class CenturyGolemEnv():
        
    def __init__(self):
        
        self.merchant_deck = {
            1: MerchantCard(1, "Y2", "crystal", {"yellow": 2, "green": 0}, None, True),
            2: MerchantCard(2, "Y3", "crystal", {"yellow": 3, "green": 0}),
            3: MerchantCard(3, "Y4", "crystal", {"yellow": 4, "green": 0}),
            4: MerchantCard(4, "Y1G1", "crystal", {"yellow": 1, "green": 1}),
            5: MerchantCard(5, "Y2G1", "crystal", {"yellow": 2, "green": 1}),
            6: MerchantCard(6, "G2", "crystal", {"yellow": 0, "green": 2}),
        }
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 3 # draw 3 cards to market, excluding M1
        )
        
        self.player1 = Player(1)
    
    def reset(self, seed=None, options=None):
        
        [setattr(card, 'owned', card.card_id == 1) for card in self.merchant_deck.values()]
        
        self.merchant_market = random.sample(
            [card for cid, card in self.merchant_deck.items() if cid != 1], 3 # draw 3 cards to market, excluding M1
        )
        
        self.player1.yellow = 0
        self.player1.green = 0
        
        self.player1.merchant_cards = [2] + [0] * 5
        
        self.render()
    
    def step(self, action):
        # Rest
        if action == 0:
            self.player1.merchant_cards = [2 if card == 1 else card for card in self.player1.merchant_cards]
        # Get a merchant card
        elif 1 <= action <= 5:
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
        elif 6 <= action <= 11:
            card_id, card_idx = action - 5, action - 6 # e.g. action 10 = use M5 = card_id 5 = card_idx 4
            if self.player1.merchant_cards[card_idx] == 2: # if card is playable
                self.player1.yellow += self.merchant_deck[card_id].gain['yellow']
                self.player1.green += self.merchant_deck[card_id].gain['green']
                self.player1.merchant_cards[card_idx] = 1 # set card status to owned but unplayable
                reward += 1
            else:
                reward -= 1
        
        self.render()
    
    def render(self):
        print(f"Y: {self.player1.yellow}")
        print(f"G: {self.player1.green}")                
        status_map = {1: "unplayable", 2: "playable"}
        for i, card_status in enumerate(self.player1.merchant_cards):
            if card_status == 0:
                continue
            print(f"M{i+1}: {status_map[card_status]}")
        print("MM: " + " | ".join([f"M{m.card_id}-{m.name}" for m in self.merchant_market]))
        print("")

if __name__ == "__main__":  
    test_env = CenturyGolemEnv()
    test_env.reset()
    test_env.step(1)
    test_env.reset()