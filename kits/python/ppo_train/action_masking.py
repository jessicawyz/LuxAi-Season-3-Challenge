"""
Simplified action masking - prevent units from dying at spawn

Rules:
1. Disallow SAP at spawn (wastes energy and kills units)
2. Allow all movement except off-map
3. Keep it simple!
"""

def get_valid_actions(unit_pos, energy, map_size=24, known_asteroids=None):
    """Get valid actions for a unit - HANDLE INVALID POSITIONS"""
    import numpy as np
    
    if known_asteroids is None:
        known_asteroids = set()
    
    x, y = unit_pos
    
    # Start with all actions allowed
    mask = np.ones(6, dtype=bool)
    
    # 1. Check if unit is in invalid position (respawning)
    if x == -1 and y == -1:
        # Unit is respawning - only allow NO_OP
        mask[1:] = False  # Disable all actions except NO_OP
        return mask
    
    # 2. Check if unit is off-map (shouldn't happen but safety)
    if not (0 <= x < map_size and 0 <= y < map_size):
        mask[1:] = False  # Disable all actions except NO_OP
        return mask
    
    # 3. Normal position handling...
    spawn_positions = {(0, 0), (23, 23)}
    
    # At spawn positions, force movement away
    if tuple(unit_pos) in spawn_positions:
        pass
    
    # Check movement actions
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    for action_idx, (dx, dy) in enumerate(moves, start=1):
        nx, ny = x + dx, y + dy
        
        if not (0 <= nx < map_size and 0 <= ny < map_size):
            mask[action_idx] = False
        elif (nx, ny) in known_asteroids:
            mask[action_idx] = False
    
    # Disallow SAP everywhere for now
    mask[5] = False
    
    return mask