#!/usr/bin/env python3
"""
Fix remaining blocking assignment issues in memristor_crossbar_fixed.v
"""

from pathlib import Path

def fix_remaining_memristor_issues():
    """Fix the remaining issues in memristor_crossbar_fixed.v"""
    
    filepath = Path("pentary-repo/hardware/memristor_crossbar_fixed.v")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixes = []
    
    # Line 172: weight_val = decode_pentary
    if len(lines) > 171 and 'weight_val = decode_pentary' in lines[171]:
        lines[171] = lines[171].replace('weight_val = decode_pentary', 'weight_val <= decode_pentary')
        fixes.append(172)
    
    # Line 173: input_val = decode_pentary
    if len(lines) > 172 and 'input_val = decode_pentary' in lines[172]:
        lines[172] = lines[172].replace('input_val = decode_pentary', 'input_val <= decode_pentary')
        fixes.append(173)
    
    # Line 176: accumulator[row] = accumulator[row] +
    if len(lines) > 175 and 'accumulator[row] = accumulator[row]' in lines[175]:
        lines[175] = lines[175].replace('accumulator[row] = accumulator[row]', 'accumulator[row] <= accumulator[row]')
        fixes.append(176)
    
    # Search for other blocking assignments in the file
    for i, line in enumerate(lines):
        if 'always @(posedge clk)' in line:
            # Found sequential always block, check following lines
            j = i + 1
            depth = 0
            while j < len(lines):
                if 'begin' in lines[j]:
                    depth += 1
                if 'end' in lines[j]:
                    depth -= 1
                    if depth < 0:
                        break
                
                # Check for blocking assignments
                if '=' in lines[j] and '<=' not in lines[j] and '==' not in lines[j]:
                    # Check if it's in a for loop (those are OK for loop variables)
                    if 'for' not in lines[j]:
                        # Check if it's an actual assignment
                        if 'weight_val =' in lines[j]:
                            lines[j] = lines[j].replace('weight_val =', 'weight_val <=')
                            if j+1 not in fixes:
                                fixes.append(j+1)
                        elif 'input_val =' in lines[j]:
                            lines[j] = lines[j].replace('input_val =', 'input_val <=')
                            if j+1 not in fixes:
                                fixes.append(j+1)
                        elif 'accumulator[' in lines[j] and '] =' in lines[j]:
                            lines[j] = lines[j].replace('] =', '] <=')
                            if j+1 not in fixes:
                                fixes.append(j+1)
                
                j += 1
    
    # Write fixed file
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Fixed lines: {fixes}")
    print(f"✅ Total fixes: {len(fixes)}")
    
    return len(fixes)

if __name__ == "__main__":
    print("Fixing remaining issues in memristor_crossbar_fixed.v...")
    fixes = fix_remaining_memristor_issues()
    print(f"\n✅ Applied {fixes} additional fixes")