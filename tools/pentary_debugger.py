#!/usr/bin/env python3
"""
Pentary Debugger and Visualizer
Interactive tools for debugging and visualizing pentary operations
"""

from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, '.')

from pentary_converter_optimized import PentaryConverterOptimized
from pentary_arithmetic_extended import PentaryArithmeticExtended


class PentaryDebugger:
    """Interactive debugger for pentary operations"""
    
    def __init__(self):
        self.converter = PentaryConverterOptimized()
        self.arithmetic = PentaryArithmeticExtended()
        self.history = []
    
    def step_through_addition(self, a: str, b: str, verbose: bool = True) -> str:
        """
        Step through addition digit by digit with detailed output.
        
        Args:
            a: First pentary number
            b: Second pentary number
            verbose: If True, print detailed steps
            
        Returns:
            Result of addition
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEPPING THROUGH ADDITION: {a} + {b}")
            print(f"{'='*70}\n")
        
        # Convert to arrays
        a_array = self.converter.string_to_array(a)
        b_array = self.converter.string_to_array(b)
        
        if verbose:
            print(f"Input A: {a} → {a_array}")
            print(f"Input B: {b} → {b_array}")
            print()
        
        # Pad to same length
        max_len = max(len(a_array), len(b_array))
        a_padded = [0] * (max_len - len(a_array)) + a_array
        b_padded = [0] * (max_len - len(b_array)) + b_array
        
        if verbose:
            print(f"Padded A: {a_padded}")
            print(f"Padded B: {b_padded}")
            print()
            print(f"{'Position':<10} {'A':<8} {'B':<8} {'Carry In':<12} {'Sum':<8} {'Carry Out':<12}")
            print("-" * 70)
        
        result = []
        carry = 0
        
        # Add from right to left
        for i in range(max_len - 1, -1, -1):
            a_digit = a_padded[i]
            b_digit = b_padded[i]
            
            sum_digit, carry_out = self.converter.ADD_LOOKUP[(a_digit, b_digit, carry)]
            
            if verbose:
                pos = max_len - 1 - i
                print(f"{pos:<10} {a_digit:<8} {b_digit:<8} {carry:<12} {sum_digit:<8} {carry_out:<12}")
            
            result.insert(0, sum_digit)
            carry = carry_out
        
        # Handle final carry
        if carry != 0:
            result.insert(0, carry)
            if verbose:
                print(f"{'Final':<10} {'-':<8} {'-':<8} {carry:<12} {carry:<8} {0:<12}")
        
        result_str = self.converter.array_to_string(result)
        
        if verbose:
            print()
            print(f"Result: {result} → {result_str}")
            print(f"Verification: {self.converter.pentary_to_decimal(a)} + "
                  f"{self.converter.pentary_to_decimal(b)} = "
                  f"{self.converter.pentary_to_decimal(result_str)}")
            print(f"{'='*70}\n")
        
        return result_str
    
    def step_through_multiplication(self, a: str, b: str, verbose: bool = True) -> str:
        """
        Step through multiplication with detailed output.
        
        Args:
            a: First pentary number
            b: Second pentary number
            verbose: If True, print detailed steps
            
        Returns:
            Result of multiplication
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEPPING THROUGH MULTIPLICATION: {a} × {b}")
            print(f"{'='*70}\n")
        
        a_array = self.converter.string_to_array(a)
        b_array = self.converter.string_to_array(b)
        
        if verbose:
            print(f"Input A: {a} → {a_array}")
            print(f"Input B: {b} → {b_array}")
            print()
        
        result = [0]
        
        if verbose:
            print("Partial Products:")
            print("-" * 70)
        
        for i, digit_b in enumerate(reversed(b_array)):
            if digit_b == 0:
                if verbose:
                    print(f"Position {i}: Digit is 0, skip")
                continue
            
            # Multiply a by single digit
            partial = self.converter.multiply_array_by_constant(a_array, digit_b)
            
            # Shift left by position
            partial_shifted = self.converter.shift_left_array(partial, i)
            
            if verbose:
                partial_str = self.converter.array_to_string(partial)
                partial_shifted_str = self.converter.array_to_string(partial_shifted)
                print(f"Position {i}: {a} × {digit_b} = {partial_str}")
                print(f"  Shifted: {partial_shifted_str}")
            
            # Add to result
            old_result = result.copy()
            result = self.converter.add_arrays(result, partial_shifted)
            
            if verbose:
                old_result_str = self.converter.array_to_string(old_result)
                result_str = self.converter.array_to_string(result)
                print(f"  Running total: {old_result_str} + {partial_shifted_str} = {result_str}")
                print()
        
        result_str = self.converter.array_to_string(result)
        
        if verbose:
            print(f"Final Result: {result} → {result_str}")
            print(f"Verification: {self.converter.pentary_to_decimal(a)} × "
                  f"{self.converter.pentary_to_decimal(b)} = "
                  f"{self.converter.pentary_to_decimal(result_str)}")
            print(f"{'='*70}\n")
        
        return result_str
    
    def analyze_number(self, pentary_str: str) -> Dict:
        """
        Analyze a pentary number and return detailed information.
        
        Args:
            pentary_str: Pentary number to analyze
            
        Returns:
            Dictionary with analysis results
        """
        array = self.converter.string_to_array(pentary_str)
        decimal = self.converter.pentary_to_decimal(pentary_str)
        
        # Count digit frequencies
        digit_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for digit in array:
            digit_counts[digit] += 1
        
        # Calculate sparsity (percentage of zeros)
        sparsity = digit_counts[0] / len(array) * 100 if array else 0
        
        # Determine sign
        if array[0] < 0:
            sign = "negative"
        elif array[0] > 0:
            sign = "positive"
        else:
            sign = "zero"
        
        return {
            'pentary': pentary_str,
            'array': array,
            'decimal': decimal,
            'length': len(array),
            'digit_counts': digit_counts,
            'sparsity': sparsity,
            'sign': sign
        }
    
    def print_analysis(self, pentary_str: str):
        """Print detailed analysis of a pentary number"""
        analysis = self.analyze_number(pentary_str)
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS: {pentary_str}")
        print(f"{'='*70}\n")
        
        print(f"Pentary:  {analysis['pentary']}")
        print(f"Array:    {analysis['array']}")
        print(f"Decimal:  {analysis['decimal']}")
        print(f"Length:   {analysis['length']} digits")
        print(f"Sign:     {analysis['sign']}")
        print(f"Sparsity: {analysis['sparsity']:.1f}% zeros")
        print()
        
        print("Digit Distribution:")
        print("-" * 70)
        symbols = {-2: '⊖', -1: '-', 0: '0', 1: '+', 2: '⊕'}
        for digit in [-2, -1, 0, 1, 2]:
            count = analysis['digit_counts'][digit]
            pct = count / analysis['length'] * 100 if analysis['length'] > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"{symbols[digit]} ({digit:>2}): {count:>3} ({pct:>5.1f}%) {bar}")
        
        print(f"{'='*70}\n")
    
    def compare_numbers(self, a: str, b: str):
        """Compare two pentary numbers with detailed output"""
        print(f"\n{'='*70}")
        print(f"COMPARING: {a} vs {b}")
        print(f"{'='*70}\n")
        
        a_dec = self.converter.pentary_to_decimal(a)
        b_dec = self.converter.pentary_to_decimal(b)
        
        cmp = self.converter.compare_pentary(a, b)
        
        if cmp < 0:
            relation = "<"
            result = f"{a} is less than {b}"
        elif cmp > 0:
            relation = ">"
            result = f"{a} is greater than {b}"
        else:
            relation = "="
            result = f"{a} is equal to {b}"
        
        print(f"Pentary:  {a} {relation} {b}")
        print(f"Decimal:  {a_dec} {relation} {b_dec}")
        print(f"Result:   {result}")
        print(f"{'='*70}\n")


class PentaryVisualizer:
    """Visualizer for pentary numbers and operations"""
    
    def __init__(self):
        self.converter = PentaryConverterOptimized()
    
    def visualize_number(self, pentary_str: str, width: int = 60):
        """
        Create visual representation of pentary number.
        
        Args:
            pentary_str: Pentary number to visualize
            width: Width of visualization
        """
        array = self.converter.string_to_array(pentary_str)
        decimal = self.converter.pentary_to_decimal(pentary_str)
        
        print(f"\n{'='*width}")
        print(f"VISUALIZING: {pentary_str}")
        print(f"{'='*width}\n")
        
        # Color mapping (using ANSI colors)
        colors = {
            -2: '\033[91m',  # Red
            -1: '\033[93m',  # Yellow
            0: '\033[90m',   # Gray
            1: '\033[92m',   # Green
            2: '\033[94m',   # Blue
        }
        reset = '\033[0m'
        
        symbols = {-2: '⊖', -1: '-', 0: '0', 1: '+', 2: '⊕'}
        
        # Print colored digits
        print("Colored representation:")
        for digit in array:
            print(f"{colors[digit]}{symbols[digit]}{reset}", end='')
        print(f"  (Decimal: {decimal})")
        print()
        
        # Print magnitude bars
        print("Magnitude bars:")
        for i, digit in enumerate(array):
            bar_len = abs(digit) * 5
            bar = '█' * bar_len if digit != 0 else '·'
            sign = '+' if digit > 0 else ('-' if digit < 0 else ' ')
            print(f"Position {i}: {sign} {bar}")
        
        print(f"{'='*width}\n")
    
    def visualize_operation(self, a: str, b: str, operation: str, result: str):
        """
        Visualize an arithmetic operation.
        
        Args:
            a: First operand
            b: Second operand
            operation: Operation symbol (+, -, ×, ÷)
            result: Result
        """
        print(f"\n{'='*70}")
        print(f"OPERATION VISUALIZATION")
        print(f"{'='*70}\n")
        
        a_dec = self.converter.pentary_to_decimal(a)
        b_dec = self.converter.pentary_to_decimal(b)
        r_dec = self.converter.pentary_to_decimal(result)
        
        # Align for display
        max_len = max(len(a), len(b), len(result))
        
        print(f"  {a:>{max_len}}  ({a_dec})")
        print(f"{operation} {b:>{max_len}}  ({b_dec})")
        print(f"  {'-' * max_len}")
        print(f"  {result:>{max_len}}  ({r_dec})")
        
        print(f"\n{'='*70}\n")


def main():
    """Demo and testing of debugger and visualizer"""
    print("=" * 70)
    print("Pentary Debugger and Visualizer")
    print("=" * 70)
    print()
    
    debugger = PentaryDebugger()
    visualizer = PentaryVisualizer()
    
    # Test step-through addition
    result = debugger.step_through_addition("⊕+", "+⊖")
    
    # Test step-through multiplication
    result = debugger.step_through_multiplication("+⊕", "⊕")
    
    # Test number analysis
    debugger.print_analysis("⊕+0-⊖")
    
    # Test comparison
    debugger.compare_numbers("⊕+", "+⊕")
    
    # Test visualization
    visualizer.visualize_number("⊕+0-⊖")
    
    # Test operation visualization
    visualizer.visualize_operation("⊕+", "+0", "+", "⊕⊕")
    
    print("=" * 70)


if __name__ == "__main__":
    main()