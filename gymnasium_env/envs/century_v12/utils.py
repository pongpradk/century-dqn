from .constants import GAME_CONSTANTS

def calculate_total_points(player):
    """Calculates total points for a player, including non-yellow crystals."""
    crystal_points = player.caravan["green"]  # Assuming green crystals are worth 1 point
    return player.points + crystal_points

def remove_excess_crystals(player):
    """Remove and penalize excess crystals."""
    total_crystals = sum(player.caravan.values())
    if total_crystals > GAME_CONSTANTS['MAX_CRYSTALS']:
        excess = total_crystals - GAME_CONSTANTS['MAX_CRYSTALS']
        # Remove from yellow first, then green
        excess_yellow = min(excess, player.caravan["yellow"])
        excess_green = excess - excess_yellow
        player.caravan["yellow"] -= excess_yellow
        player.caravan["green"] -= excess_green
        
        return (GAME_CONSTANTS['CRYSTAL_VALUES']['yellow'] * excess_yellow) + (GAME_CONSTANTS['CRYSTAL_VALUES']['green'] * excess_green)

    return 0 