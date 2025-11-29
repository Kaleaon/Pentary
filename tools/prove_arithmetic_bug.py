#!/usr/bin/env python3
"""
Prove Arithmetic Bug
Validates the PentaryArithmetic ADD_TABLE against mathematical truth.
"""

import sys
import os

# Add tools directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_arithmetic import PentaryArithmetic

def validate_add_table():
    print("Validating ADD_TABLE integrity...")
    print("-" * 60)
    print(f"{'Inputs (a, b, cin)':<20} {'Table (sum, cout)':<20} {'Check':<10}")
    print("-" * 60)

    errors = 0
    checked = 0

    # Range of digits {-2, -1, 0, 1, 2}
    digits = range(-2, 3)
    # Range of carry {-1, 0, 1}
    carries = range(-1, 2)

    for a in digits:
        for b in digits:
            for cin in carries:
                key = (a, b, cin)

                if key in PentaryArithmetic.ADD_TABLE:
                    sum_digit, cout = PentaryArithmetic.ADD_TABLE[key]

                    # Mathematical check
                    # value = sum_digit + 5 * cout
                    # expected = a + b + cin

                    value = sum_digit + 5 * cout
                    expected = a + b + cin

                    if value != expected:
                        print(f"{str(key):<20} {str((sum_digit, cout)):<20} FAIL")
                        print(f"  Expected {expected}, got {value} ({sum_digit} + 5*{cout})")
                        errors += 1
                    else:
                        # print(f"{str(key):<20} {str((sum_digit, cout)):<20} OK")
                        pass
                    checked += 1
                else:
                    print(f"{str(key):<20} MISSING")
                    # errors += 1

    print("-" * 60)
    print(f"Checked {checked} entries. Found {errors} errors.")

if __name__ == "__main__":
    validate_add_table()
