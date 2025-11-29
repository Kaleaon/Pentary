#!/usr/bin/env python3
"""
Pentary Arithmetic Calculator
Implements pentary arithmetic operations at the digit level
"""

from typing import Tuple, List
from pentary_converter import PentaryConverter


class PentaryArithmetic:
    """Low-level pentary arithmetic operations"""
    
    # Digit addition lookup table (without carry)
    # Returns (sum_digit, carry_out)
    ADD_TABLE = {
        # Format: (a, b, carry_in) -> (sum, carry_out)
        # For balanced pentary: digits are {-2, -1, 0, 1, 2}
        # If sum < -2: sum_digit = sum + 5, carry = -1
        # If sum > 2: sum_digit = sum - 5, carry = 1
        # Otherwise: sum_digit = sum, carry = 0
        (-2, -2, -1): (0, -1),   # -2 + -2 + -1 = -5 = 0 + (-1)*5
        (-2, -2, 0): (1, -1),    # -2 + -2 + 0 = -4 = 1 + (-1)*5
        (-2, -2, 1): (2, -1),    # -2 + -2 + 1 = -3 = 2 + (-1)*5
        (-2, -1, -1): (1, -1),   # -2 + -1 + -1 = -4 = 1 + (-1)*5
        (-2, -1, 0): (2, -1),    # -2 + -1 + 0 = -3 = 2 + (-1)*5
        (-2, -1, 1): (-2, 0),    # -2 + -1 + 1 = -2
        (-2, 0, -1): (2, -1),    # -2 + 0 + -1 = -3 = 2 + (-1)*5
        (-2, 0, 0): (-2, 0),     # -2 + 0 + 0 = -2
        (-2, 0, 1): (-1, 0),     # -2 + 0 + 1 = -1
        (-2, 1, -1): (-2, 0),    # -2 + 1 + -1 = -2
        (-2, 1, 0): (-1, 0),     # -2 + 1 + 0 = -1
        (-2, 1, 1): (0, 0),      # -2 + 1 + 1 = 0
        (-2, 2, -1): (-1, 0),    # -2 + 2 + -1 = -1
        (-2, 2, 0): (0, 0),      # -2 + 2 + 0 = 0
        (-2, 2, 1): (1, 0),      # -2 + 2 + 1 = 1
        
        (-1, -2, -1): (1, -1),   # -1 + -2 + -1 = -4 = 1 + (-1)*5
        (-1, -2, 0): (2, -1),    # -1 + -2 + 0 = -3 = 2 + (-1)*5
        (-1, -2, 1): (-2, 0),    # -1 + -2 + 1 = -2
        (-1, -1, -1): (2, -1),   # -1 + -1 + -1 = -3 = 2 + (-1)*5
        (-1, -1, 0): (-2, 0),    # -1 + -1 + 0 = -2
        (-1, -1, 1): (-1, 0),    # -1 + -1 + 1 = -1
        (-1, 0, -1): (-2, 0),    # -1 + 0 + -1 = -2
        (-1, 0, 0): (-1, 0),     # -1 + 0 + 0 = -1
        (-1, 0, 1): (0, 0),      # -1 + 0 + 1 = 0
        (-1, 1, -1): (-1, 0),    # -1 + 1 + -1 = -1
        (-1, 1, 0): (0, 0),      # -1 + 1 + 0 = 0
        (-1, 1, 1): (1, 0),      # -1 + 1 + 1 = 1
        (-1, 2, -1): (0, 0),     # -1 + 2 + -1 = 0
        (-1, 2, 0): (1, 0),      # -1 + 2 + 0 = 1
        (-1, 2, 1): (2, 0),      # -1 + 2 + 1 = 2
        
        (0, -2, -1): (2, -1),    # 0 + -2 + -1 = -3 = 2 + (-1)*5
        (0, -2, 0): (-2, 0),     # 0 + -2 + 0 = -2
        (0, -2, 1): (-1, 0),     # 0 + -2 + 1 = -1
        (0, -1, -1): (-2, 0),    # 0 + -1 + -1 = -2
        (0, -1, 0): (-1, 0),     # 0 + -1 + 0 = -1
        (0, -1, 1): (0, 0),      # 0 + -1 + 1 = 0
        (0, 0, -1): (-1, 0),     # 0 + 0 + -1 = -1
        (0, 0, 0): (0, 0),       # 0 + 0 + 0 = 0
        (0, 0, 1): (1, 0),       # 0 + 0 + 1 = 1
        (0, 1, -1): (0, 0),      # 0 + 1 + -1 = 0
        (0, 1, 0): (1, 0),       # 0 + 1 + 0 = 1
        (0, 1, 1): (2, 0),       # 0 + 1 + 1 = 2
        (0, 2, -1): (1, 0),      # 0 + 2 + -1 = 1
        (0, 2, 0): (2, 0),       # 0 + 2 + 0 = 2
        (0, 2, 1): (-2, 1),      # 0 + 2 + 1 = 3 = -2 + 1*5
        
        (1, -2, -1): (-2, 0),    # 1 + -2 + -1 = -2
        (1, -2, 0): (-1, 0),     # 1 + -2 + 0 = -1
        (1, -2, 1): (0, 0),      # 1 + -2 + 1 = 0
        (1, -1, -1): (-1, 0),    # 1 + -1 + -1 = -1
        (1, -1, 0): (0, 0),      # 1 + -1 + 0 = 0
        (1, -1, 1): (1, 0),      # 1 + -1 + 1 = 1
        (1, 0, -1): (0, 0),      # 1 + 0 + -1 = 0
        (1, 0, 0): (1, 0),       # 1 + 0 + 0 = 1
        (1, 0, 1): (2, 0),       # 1 + 0 + 1 = 2
        (1, 1, -1): (1, 0),      # 1 + 1 + -1 = 1
        (1, 1, 0): (2, 0),       # 1 + 1 + 0 = 2
        (1, 1, 1): (-2, 1),      # 1 + 1 + 1 = 3 = -2 + 1*5
        (1, 2, -1): (2, 0),      # 1 + 2 + -1 = 2
        (1, 2, 0): (-2, 1),      # 1 + 2 + 0 = 3 = -2 + 1*5
        (1, 2, 1): (-1, 1),      # 1 + 2 + 1 = 4 = -1 + 1*5
        
        (2, -2, -1): (-1, 0),    # 2 + -2 + -1 = -1
        (2, -2, 0): (0, 0),      # 2 + -2 + 0 = 0
        (2, -2, 1): (1, 0),      # 2 + -2 + 1 = 1
        (2, -1, -1): (0, 0),     # 2 + -1 + -1 = 0
        (2, -1, 0): (1, 0),      # 2 + -1 + 0 = 1
        (2, -1, 1): (2, 0),      # 2 + -1 + 1 = 2
        (2, 0, -1): (1, 0),      # 2 + 0 + -1 = 1
        (2, 0, 0): (2, 0),       # 2 + 0 + 0 = 2
        (2, 0, 1): (-2, 1),      # 2 + 0 + 1 = 3 = -2 + 1*5
        (2, 1, -1): (2, 0),      # 2 + 1 + -1 = 2
        (2, 1, 0): (-2, 1),      # 2 + 1 + 0 = 3 = -2 + 1*5
        (2, 1, 1): (-1, 1),      # 2 + 1 + 1 = 4 = -1 + 1*5
        (2, 2, -1): (-2, 1),     # 2 + 2 + -1 = 3 = -2 + 1*5
        (2, 2, 0): (-1, 1),      # 2 + 2 + 0 = 4 = -1 + 1*5
        (2, 2, 1): (0, 1),       # 2 + 2 + 1 = 5 = 0 + 1*5
    }
    
    @staticmethod
    def add_digits(a: int, b: int, carry_in: int = 0) -> Tuple[int, int]:
        """
        Add two pentary digits with carry.
        
        Args:
            a: First digit {-2, -1, 0, 1, 2}
            b: Second digit {-2, -1, 0, 1, 2}
            carry_in: Carry in {-1, 0, 1}
            
        Returns:
            (sum_digit, carry_out)
        """
        if (a, b, carry_in) in PentaryArithmetic.ADD_TABLE:
            return PentaryArithmetic.ADD_TABLE[(a, b, carry_in)]
        
        # Fallback: compute directly
        total = a + b + carry_in
        
        # Convert to balanced pentary
        if total >= -2 and total <= 2:
            return (total, 0)
        elif total > 2:
            # Need positive carry
            return (total - 5, 1)
        else:  # total < -2
            # Need negative carry
            return (total + 5, -1)
    
    @staticmethod
    def add_pentary_detailed(a_str: str, b_str: str) -> Tuple[str, List[dict]]:
        """
        Add two pentary numbers with detailed step-by-step trace.
        
        Args:
            a_str: First pentary number
            b_str: Second pentary number
            
        Returns:
            (result, steps) where steps is a list of operation details
        """
        # Convert to digit lists
        a_digits = [PentaryConverter.REVERSE_SYMBOLS[c] for c in a_str]
        b_digits = [PentaryConverter.REVERSE_SYMBOLS[c] for c in b_str]
        
        # Pad to same length
        max_len = max(len(a_digits), len(b_digits))
        a_digits = [0] * (max_len - len(a_digits)) + a_digits
        b_digits = [0] * (max_len - len(b_digits)) + b_digits
        
        result_digits = []
        carry = 0
        steps = []
        
        # Add from right to left
        for i in range(max_len - 1, -1, -1):
            a_digit = a_digits[i]
            b_digit = b_digits[i]
            
            sum_digit, carry_out = PentaryArithmetic.add_digits(a_digit, b_digit, carry)
            
            steps.append({
                'position': max_len - 1 - i,
                'a': a_digit,
                'b': b_digit,
                'carry_in': carry,
                'sum': sum_digit,
                'carry_out': carry_out
            })
            
            result_digits.insert(0, sum_digit)
            carry = carry_out
        
        # Handle final carry
        if carry != 0:
            result_digits.insert(0, carry)
            steps.append({
                'position': max_len,
                'a': 0,
                'b': 0,
                'carry_in': carry,
                'sum': carry,
                'carry_out': 0
            })
        
        # Convert back to string
        result_str = ''.join(PentaryConverter.SYMBOLS[d] for d in result_digits)
        
        # Remove leading zeros
        result_str = result_str.lstrip('0') or '0'
        
        return result_str, steps
    
    @staticmethod
    def multiply_digit_by_constant(digit: int, constant: int) -> Tuple[int, int]:
        """
        Multiply a pentary digit by a constant {-2, -1, 0, 1, 2}.
        
        Args:
            digit: Pentary digit {-2, -1, 0, 1, 2}
            constant: Multiplier {-2, -1, 0, 1, 2}
            
        Returns:
            (result_digit, carry) in balanced pentary
        """
        product = digit * constant
        
        # Convert to balanced pentary
        if product >= -2 and product <= 2:
            return (product, 0)
        elif product > 2:
            # Positive overflow
            if product == 3:
                return (-2, 1)
            elif product == 4:
                return (-1, 1)
            else:  # product >= 5
                carry = product // 5
                remainder = product % 5
                if remainder > 2:
                    remainder -= 5
                    carry += 1
                return (remainder, carry)
        else:  # product < -2
            # Negative overflow
            if product == -3:
                return (2, -1)
            elif product == -4:
                return (1, -1)
            else:  # product <= -5
                carry = -((-product) // 5)
                remainder = -((-product) % 5)
                if remainder < -2:
                    remainder += 5
                    carry -= 1
                return (remainder, carry)
    
    @staticmethod
    def compare_pentary(a_str: str, b_str: str) -> int:
        """
        Compare two pentary numbers.
        
        Args:
            a_str: First pentary number
            b_str: Second pentary number
            
        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        a_dec = PentaryConverter.pentary_to_decimal(a_str)
        b_dec = PentaryConverter.pentary_to_decimal(b_str)
        
        if a_dec < b_dec:
            return -1
        elif a_dec > b_dec:
            return 1
        else:
            return 0


def main():
    """Demo and testing of pentary arithmetic"""
    print("=" * 70)
    print("Pentary Arithmetic Calculator")
    print("=" * 70)
    print()
    
    # Test detailed addition
    test_cases = [
        ("+", "+"),      # 1 + 1 = 2
        ("⊕", "⊕"),      # 2 + 2 = 4
        ("+⊕", "+"),     # 7 + 1 = 8
        ("⊕⊕", "++"),    # 12 + 6 = 18
        ("+0", "-0"),    # 5 + -5 = 0
        ("⊕-", "+-"),    # 9 + 4 = 13
    ]
    
    print("Detailed Addition Examples:")
    print("-" * 70)
    
    for a, b in test_cases:
        result, steps = PentaryArithmetic.add_pentary_detailed(a, b)
        a_dec = PentaryConverter.pentary_to_decimal(a)
        b_dec = PentaryConverter.pentary_to_decimal(b)
        r_dec = PentaryConverter.pentary_to_decimal(result)
        
        print(f"\n{a} + {b} = {result}")
        print(f"({a_dec} + {b_dec} = {r_dec})")
        print("\nStep-by-step:")
        print(f"{'Pos':<5} {'A':<5} {'B':<5} {'C_in':<7} {'Sum':<5} {'C_out':<7}")
        print("-" * 40)
        
        for step in steps:
            pos = step['position']
            a_sym = PentaryConverter.SYMBOLS[step['a']]
            b_sym = PentaryConverter.SYMBOLS[step['b']]
            cin_sym = PentaryConverter.SYMBOLS.get(step['carry_in'], str(step['carry_in']))
            sum_sym = PentaryConverter.SYMBOLS[step['sum']]
            cout_sym = PentaryConverter.SYMBOLS.get(step['carry_out'], str(step['carry_out']))
            
            print(f"{pos:<5} {a_sym:<5} {b_sym:<5} {cin_sym:<7} {sum_sym:<5} {cout_sym:<7}")
    
    print("\n" + "=" * 70)
    print("Multiplication by Constants:")
    print("-" * 70)
    
    test_digit = 2  # ⊕
    print(f"\nMultiplying {PentaryConverter.SYMBOLS[test_digit]} by constants:")
    print(f"{'Constant':<12} {'Result':<15} {'Decimal Check':<20}")
    print("-" * 50)
    
    for const in [-2, -1, 0, 1, 2]:
        result_digit, carry = PentaryArithmetic.multiply_digit_by_constant(test_digit, const)
        result_str = ""
        if carry != 0:
            result_str = PentaryConverter.SYMBOLS[carry]
        result_str += PentaryConverter.SYMBOLS[result_digit]
        
        decimal_result = test_digit * const
        print(f"{const:<12} {result_str:<15} ({test_digit} × {const} = {decimal_result})")
    
    print("\n" + "=" * 70)
    print("Comparison Operations:")
    print("-" * 70)
    
    compare_tests = [
        ("+⊕", "⊕-"),    # 7 vs 9
        ("⊕⊕", "⊕⊕"),    # 12 vs 12
        ("-0", "+"),     # -5 vs 1
    ]
    
    for a, b in compare_tests:
        cmp = PentaryArithmetic.compare_pentary(a, b)
        a_dec = PentaryConverter.pentary_to_decimal(a)
        b_dec = PentaryConverter.pentary_to_decimal(b)
        
        if cmp < 0:
            op = "<"
        elif cmp > 0:
            op = ">"
        else:
            op = "="
        
        print(f"{a} {op} {b}  ({a_dec} {op} {b_dec})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()