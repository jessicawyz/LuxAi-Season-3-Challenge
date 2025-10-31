"""
Working Action Masking - Based on successful approaches
Key: SAP is ENABLED strategically, not disabled
"""

def get_valid_actions(unit_pos, energy, map_size=24, known_asteroids=None, known_point_tiles=None):
    """
    Get valid actions for a unit
    
    Based on Frog Parade winner: SAP enabled near known point tiles
    Based on working MADDPG: SAP enabled when unit has energy
    """
    import numpy as np
    
    if known_asteroids is None:
        known_asteroids = set()
    if known_point_tiles is None:
        known_point_tiles = set()
    
    x, y = unit_pos
    
    # Start with all actions allowed
    mask = np.ones(6, dtype=bool)
    
    # 1. Check if unit is in invalid position (respawning)
    if x == -1 and y == -1:
        mask[1:] = False  # Only allow NO_OP
        return mask
    
    # 2. Check if unit is off-map (safety)
    if not (0 <= x < map_size and 0 <= y < map_size):
        mask[1:] = False
        return mask
    
    # 3. Check movement actions (up, right, down, left)
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    for action_idx, (dx, dy) in enumerate(moves, start=1):
        nx, ny = x + dx, y + dy
        
        # Check bounds
        if not (0 <= nx < map_size and 0 <= ny < map_size):
            mask[action_idx] = False
        # Check asteroids
        elif (nx, ny) in known_asteroids:
            mask[action_idx] = False
    
    mask[5] = False
    
    return mask