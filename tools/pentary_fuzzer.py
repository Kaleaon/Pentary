#!/usr/bin/env python3
"""
Pentary Fuzzer
Generates random arithmetic operations to find flaws in the Pentary implementation.
"""

import sys
import os
import random
import time
from typing import Tuple

# Add tools directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic

def fuzz_addition(iterations: int = 1000):
    """Fuzz addition operations"""
    print(f"Fuzzing addition for {iterations} iterations...")

    failures = 0
    start_time = time.time()

    for i in range(iterations):
        a = random.randint(-5000, 5000)
        b = random.randint(-5000, 5000)

        try:
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)

            # Ground truth
            expected_sum_pent = PentaryConverter.add_pentary(a_pent, b_pent)
            expected_decimal = a + b

            # Architecture logic
            calc_sum_pent, steps = PentaryArithmetic.add_pentary_detailed(a_pent, b_pent)

            # Normalize
            expected_norm = expected_sum_pent.lstrip('0') or '0'
            calc_norm = calc_sum_pent.lstrip('0') or '0'

            if expected_norm != calc_norm:
                print(f"FAILURE found at iteration {i}:")
                print(f"  Decimal: {a} + {b} = {a+b}")
                print(f"  Pentary Inputs: {a_pent} + {b_pent}")
                print(f"  Expected: {expected_norm}")
                print(f"  Calculated: {calc_norm}")
                print("  Steps:")
                for step in steps:
                    print(f"    {step}")
                failures += 1

                # Stop after finding a failure to analyze
                if failures >= 5:
                    break

        except Exception as e:
            print(f"EXCEPTION at iteration {i}: {e}")
            failures += 1

    print(f"\nFuzzing complete. Found {failures} failures in {time.time() - start_time:.2f} seconds.")

def main():
    fuzz_addition(1000)

if __name__ == "__main__":
    main()
