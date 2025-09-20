from .constants import GAME_CONSTANTS

def calculate_total_points(player):
    crystal_points = player.caravan["green"] + player.caravan["blue"] + player.caravan["pink"]
    return player.points + crystal_points

def remove_excess_crystals(player):
    total_crystals = sum(player.caravan.values())
    
    if total_crystals <= GAME_CONSTANTS['MAX_CRYSTALS']:
        return 0
        
    excess = total_crystals - GAME_CONSTANTS['MAX_CRYSTALS']
    penalty = 0
    
    crystal_types = ["yellow", "green", "blue", "pink"]
    
    # Remove crystals in order of increasing value
    for crystal_type in crystal_types:
        if excess <= 0:
            break
            
        crystals_to_remove = min(excess, player.caravan[crystal_type])
        excess -= crystals_to_remove
        player.caravan[crystal_type] -= crystals_to_remove
        
        # Calculate penalty based on crystal value
        crystal_value = GAME_CONSTANTS['CRYSTAL_VALUES'][crystal_type]
        penalty += crystal_value * crystals_to_remove
    
    return penalty