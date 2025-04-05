class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.caravan = {
            "yellow": 3,
            "green": 0,
            "blue": 0
        }
        # status of each merchant card for this player
        self.merchant_cards = [2, 2] + [0] * 23  # 0 = not owned, 1 = owned but unplayable, 2 = owned and playable
        self.golem_count = 0
        self.points = 0

class MerchantCard:
    def __init__(self, card_id, name, card_type, gain=None, cost=None, owned=False):
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