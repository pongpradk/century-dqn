from .constants import GAME_CONSTANTS

def calculate_total_points(player):
    """Calculates total points for a player, including non-yellow crystals."""
    crystal_points = player.caravan["green"] + player.caravan["blue"] + player.caravan["pink"]
    return player.points + crystal_points

def remove_excess_crystals(player):
    """Remove and penalize excess crystals."""
    total_crystals = sum(player.caravan.values())
    if total_crystals > GAME_CONSTANTS['MAX_CRYSTALS']:
        excess = total_crystals - GAME_CONSTANTS['MAX_CRYSTALS']
        # Remove from yellow first, then green, then blue, then pink (highest value)
        excess_yellow = min(excess, player.caravan["yellow"])
        excess -= excess_yellow
        excess_green = min(excess, player.caravan["green"])
        excess -= excess_green
        excess_blue = min(excess, player.caravan["blue"])
        excess -= excess_blue
        excess_pink = excess
        
        player.caravan["yellow"] -= excess_yellow
        player.caravan["green"] -= excess_green
        player.caravan["blue"] -= excess_blue
        player.caravan["pink"] -= excess_pink
        
        return (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * excess_yellow) + \
               (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * excess_green) + \
               (GAME_CONSTANTS['CRYSTAL_VALUES']['blue'] * excess_blue) + \
               (GAME_CONSTANTS['CRYSTAL_VALUES']['pink'] * excess_pink)

    return 0