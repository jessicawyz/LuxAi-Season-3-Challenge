def diagnose_unit_filtering(obs, player_id=0):
    """
    Debug why _get_unit_info only returns 1 unit when there are more active units
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSING UNIT FILTERING FOR player_{player_id}")
    print(f"{'='*70}")
    
    # Check what's in the observation
    if 'units' not in obs:
        print("❌ ERROR: 'units' not in observation!")
        return
    
    if 'position' not in obs['units']:
        print("❌ ERROR: 'position' not in obs['units']!")
        return
    
    if 'energy' not in obs['units']:
        print("❌ ERROR: 'energy' not in obs['units']!")
        return
    
    positions = obs['units']['position'][player_id]
    energies = obs['units']['energy'][player_id]
    mask = obs['units_mask'][player_id] if 'units_mask' in obs else [True] * len(positions)
    
    print(f"\n📊 RAW DATA:")
    print(f"   Total units array length: {len(positions)}")
    print(f"   Mask length: {len(mask)}")
    print(f"   Mask sum (active units): {sum(mask)}")
    
    print(f"\n🔍 UNIT BY UNIT ANALYSIS:")
    valid_count = 0
    invalid_count = 0
    
    for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
        status = "✅" if valid else "❌"
        pos_valid = "✅" if (pos[0] >= 0 and pos[1] >= 0) else "❌"
        
        print(f"   Unit {i}: mask={status} pos={pos} {pos_valid} energy={energy:.1f}")
        
        # Check the EXACT conditions from _get_unit_info
        if valid and pos[0] >= 0 and pos[1] >= 0:
            valid_count += 1
            print(f"      ✅ WOULD BE INCLUDED in unit_info")
        else:
            invalid_count += 1
            reason = []
            if not valid:
                reason.append("mask=False")
            if pos[0] < 0:
                reason.append("pos[0]<0")
            if pos[1] < 0:
                reason.append("pos[1]<0")
            print(f"      ❌ EXCLUDED because: {', '.join(reason)}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Units that PASS filter: {valid_count}")
    print(f"   Units that FAIL filter: {invalid_count}")
    print(f"   Expected in unit_info: {valid_count}")
    
    # Reproduce the exact _get_unit_info logic
    print(f"\n🔬 REPRODUCING _get_unit_info LOGIC:")
    unit_info = []
    for i, (pos, energy, valid) in enumerate(zip(positions, energies, mask)):
        if valid and pos[0] >= 0 and pos[1] >= 0:
            unit_info.append({
                'id': i,
                'position': pos,
                'energy': max(0, energy),
                'health': 100
            })
    
    print(f"   Resulting unit_info length: {len(unit_info)}")
    print(f"   Unit IDs in unit_info: {[u['id'] for u in unit_info]}")
    
    if len(unit_info) != valid_count:
        print(f"   ⚠️ WARNING: Mismatch! Expected {valid_count} but got {len(unit_info)}")
    
    return unit_info