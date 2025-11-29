#!/usr/bin/env python3
"""
Prove Memory Inefficiency
Demonstrates the massive overhead of the string-based simulation
and the dynamic range limitations of a 16-pent word.
"""

import sys
import os
import math

# Add tools directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_converter import PentaryConverter

def compare_range():
    print("Comparison of Dynamic Range:")
    print("-" * 60)

    # Binary 32-bit (standard float/int size)
    # Signed 32-bit int: -2^31 to 2^31-1
    bin32_max = 2**31 - 1
    print(f"32-bit Binary Max: {bin32_max:,.0f} (~2e9)")

    # Pentary 16-pent (word size)
    # Range is roughly +/- (5^16 - 1) / 2 ? No, it's roughly +/- 2.5 * 5^15
    # Max value is "22...2" (16 times)
    # Sum of 2 * 5^i for i=0 to 15
    pent16_max = sum(2 * (5**i) for i in range(16))
    print(f"16-pent Pentary Max: {pent16_max:,.0f} (~1.5e11)")

    print("\nNote: 16 pents is roughly 37 bits (log2(5^16) = 37.15).")
    print("So 16 pents is comparable to a 37-bit integer.")
    print("This part is actually fine, contrary to my hypothesis.")

    # What about floating point dynamic range?
    # FP32: +/- 3.4e38
    # FP16: +/- 6.5e4
    # Pentary 16-pent is integer only.
    print("\nHowever, Pentary is integer-only.")
    print(f"FP32 Range: ~1e38")
    print(f"Pentary 16 Range: {pent16_max:.1e}")
    print("For LLM weights (often small floats), we need quantization.")
    print("But for accumulation (the 'ADD' instruction), we need high dynamic range to avoid overflow.")
    print("If we accumulate thousands of products, we might overflow 16 pents quickly.")

def measure_storage_overhead():
    print("\nStorage Overhead in Simulator:")
    print("-" * 60)

    val = 123456789
    pent = PentaryConverter.decimal_to_pentary(val)
    # Python string overhead
    overhead = sys.getsizeof(pent)

    print(f"Decimal Value: {val}")
    print(f"Pentary String: '{pent}' (Length: {len(pent)})")
    print(f"Python String Size: {overhead} bytes")
    print(f"Actual Information: {math.ceil(math.log2(val))} bits (~4 bytes)")

    print(f"Overhead Factor: {overhead / 4:.1f}x")

    print("\nThis confirms the simulator is not memory efficient for large scale simulation.")

if __name__ == "__main__":
    compare_range()
    measure_storage_overhead()
