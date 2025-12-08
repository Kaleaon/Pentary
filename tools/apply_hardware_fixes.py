#!/usr/bin/env python3
"""
Apply all hardware fixes to Verilog files.
Fixes blocking assignments in sequential logic.
"""

import re
from pathlib import Path
import shutil

def backup_file(filepath):
    """Create backup of file."""
    backup_path = Path(str(filepath) + '.backup')
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up {filepath.name} to {backup_path.name}")

def fix_memristor_crossbar(filepath):
    """Fix memristor_crossbar_fixed.v"""
    
    print(f"\n{'='*60}")
    print(f"Fixing: {filepath.name}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Fix line 164: row = compute_counter;
    if len(lines) > 163:
        if 'row = compute_counter' in lines[163]:
            lines[163] = lines[163].replace('row = compute_counter', 'row <= compute_counter')
            fixes_applied += 1
            print(f"✅ Fixed line 164: row = compute_counter → row <= compute_counter")
    
    # Fix line 167: accumulator[row] = 0;
    if len(lines) > 166:
        if 'accumulator[row] = 0' in lines[166]:
            lines[166] = lines[166].replace('accumulator[row] = 0', 'accumulator[row] <= 0')
            fixes_applied += 1
            print(f"✅ Fixed line 167: accumulator[row] = 0 → accumulator[row] <= 0")
    
    # Fix line 171: weight_val = decode_pentary
    if len(lines) > 170:
        if 'weight_val = decode_pentary' in lines[170]:
            lines[170] = lines[170].replace('weight_val = decode_pentary', 'weight_val <= decode_pentary')
            fixes_applied += 1
            print(f"✅ Fixed line 171: weight_val = → weight_val <=")
    
    # Fix line 172: input_val = decode_pentary
    if len(lines) > 171:
        if 'input_val = decode_pentary' in lines[171]:
            lines[171] = lines[171].replace('input_val = decode_pentary', 'input_val <= decode_pentary')
            fixes_applied += 1
            print(f"✅ Fixed line 172: input_val = → input_val <=")
    
    # Fix line 175: accumulator[row] = accumulator[row] +
    if len(lines) > 174:
        if 'accumulator[row] = accumulator[row]' in lines[174]:
            lines[174] = lines[174].replace('accumulator[row] = accumulator[row]', 'accumulator[row] <= accumulator[row]')
            fixes_applied += 1
            print(f"✅ Fixed line 175: accumulator[row] = → accumulator[row] <=")
    
    # Write fixed file
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"\n✅ Applied {fixes_applied} fixes to {filepath.name}")
    return fixes_applied

def fix_pipeline_control(filepath):
    """Fix pipeline_control.v"""
    
    print(f"\n{'='*60}")
    print(f"Fixing: {filepath.name}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        content = f.read()
        lines = f.readlines()
    
    # Reopen to get lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Search for blocking assignments in sequential always blocks
    in_sequential_always = False
    for i, line in enumerate(lines):
        if 'always @(posedge' in line:
            in_sequential_always = True
        
        if in_sequential_always and '=' in line and '<=' not in line:
            # Check if it's an assignment (not comparison)
            if re.search(r'\w+\s*=\s*[^=]', line) and 'if' not in line and 'else' not in line:
                # Replace blocking with non-blocking
                old_line = line
                line = re.sub(r'(\w+)\s*=\s*([^=])', r'\1 <= \2', line)
                if line != old_line:
                    lines[i] = line
                    fixes_applied += 1
                    print(f"✅ Fixed line {i+1}: Changed = to <=")
        
        if in_sequential_always and 'end' in line and line.strip() == 'end':
            in_sequential_always = False
    
    # Write fixed file
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"\n✅ Applied {fixes_applied} fixes to {filepath.name}")
    return fixes_applied

def fix_register_file(filepath):
    """Fix register_file.v"""
    
    print(f"\n{'='*60}")
    print(f"Fixing: {filepath.name}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Search for blocking assignments in sequential always blocks
    in_sequential_always = False
    for i, line in enumerate(lines):
        if 'always @(posedge' in line:
            in_sequential_always = True
        
        if in_sequential_always and '=' in line and '<=' not in line:
            # Check if it's an assignment (not comparison)
            if re.search(r'\w+\s*=\s*[^=]', line) and '==' not in line:
                # Skip if statements and comparisons
                if 'if' not in line or '=' in line.split('if')[0]:
                    old_line = line
                    # Replace blocking with non-blocking
                    line = re.sub(r'(\w+(?:\[[\w:]+\])?)\s*=\s*([^=])', r'\1 <= \2', line)
                    if line != old_line:
                        lines[i] = line
                        fixes_applied += 1
                        print(f"✅ Fixed line {i+1}: Changed = to <=")
        
        if in_sequential_always and 'end' in line and line.strip() == 'end':
            in_sequential_always = False
    
    # Write fixed file
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"\n✅ Applied {fixes_applied} fixes to {filepath.name}")
    return fixes_applied

def verify_fixes(hardware_dir):
    """Verify that fixes were applied correctly."""
    
    print(f"\n{'='*60}")
    print("Verifying Fixes")
    print(f"{'='*60}")
    
    issues_found = 0
    
    for vfile in hardware_dir.glob('*.v'):
        if vfile.name.startswith('testbench'):
            continue
        
        with open(vfile, 'r') as f:
            lines = f.readlines()
        
        in_sequential = False
        for i, line in enumerate(lines, 1):
            if 'always @(posedge' in line:
                in_sequential = True
            
            if in_sequential and '=' in line and '<=' not in line:
                # Check for blocking assignment
                if re.search(r'\w+\s*=\s*[^=]', line) and '==' not in line and 'if' not in line:
                    print(f"⚠️  {vfile.name} line {i}: Possible remaining blocking assignment")
                    issues_found += 1
            
            if in_sequential and 'end' in line and line.strip() == 'end':
                in_sequential = False
    
    if issues_found == 0:
        print("✅ No issues found - all fixes applied successfully!")
    else:
        print(f"⚠️  Found {issues_found} potential issues - manual review recommended")
    
    return issues_found

if __name__ == "__main__":
    print("="*60)
    print("Applying Hardware Fixes")
    print("="*60)
    
    hardware_dir = Path("pentary-repo/hardware")
    
    total_fixes = 0
    
    # Fix memristor_crossbar_fixed.v
    filepath = hardware_dir / "memristor_crossbar_fixed.v"
    if filepath.exists():
        backup_file(filepath)
        total_fixes += fix_memristor_crossbar(filepath)
    
    # Fix pipeline_control.v
    filepath = hardware_dir / "pipeline_control.v"
    if filepath.exists():
        backup_file(filepath)
        total_fixes += fix_pipeline_control(filepath)
    
    # Fix register_file.v
    filepath = hardware_dir / "register_file.v"
    if filepath.exists():
        backup_file(filepath)
        total_fixes += fix_register_file(filepath)
    
    print(f"\n{'='*60}")
    print(f"Total Fixes Applied: {total_fixes}")
    print(f"{'='*60}")
    
    # Verify fixes
    issues = verify_fixes(hardware_dir)
    
    if issues == 0:
        print("\n✅ All hardware fixes applied successfully!")
        print("✅ Ready for synthesis and FPGA prototyping")
    else:
        print(f"\n⚠️  {issues} potential issues found")
        print("⚠️  Manual review recommended")