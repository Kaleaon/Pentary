#!/usr/bin/env python3
"""
Fix final blocking assignment issues
"""

from pathlib import Path

filepath = Path("pentary-repo/hardware/memristor_crossbar_fixed.v")

with open(filepath, 'r') as f:
    lines = f.readlines()

# Fix line 233: error_detected = 1'b0;
if len(lines) > 232 and "error_detected = 1'b0" in lines[232]:
    lines[232] = lines[232].replace("error_detected = 1'b0", "error_detected <= 1'b0")
    print("✅ Fixed line 233: error_detected = → error_detected <=")

# Fix line 238: error_detected = 1'b1;
if len(lines) > 237 and "error_detected = 1'b1" in lines[237]:
    lines[237] = lines[237].replace("error_detected = 1'b1", "error_detected <= 1'b1")
    print("✅ Fixed line 238: error_detected = → error_detected <=")

# Write fixed file
with open(filepath, 'w') as f:
    f.writelines(lines)

print("✅ All fixes applied!")