#!/usr/bin/env python3
"""
Extended Pentary Arithmetic Operations
Implements multiplication, division, and advanced operations
"""

from typing import List, Tuple, Optional
from pentary_converter_optimized import PentaryConverterOptimized


class PentaryArithmeticExtended:
    """Extended arithmetic operations for pentary numbers"""
    
    def __init__(self):
        self.converter = PentaryConverterOptimized()
    
    def multiply_arrays(self, a: List[int], b: List[int]) -> List[int]:
        """
        Multiply two pentary digit arrays using shift-and-add algorithm.
        
        Args:
            a: First pentary digit array
            b: Second pentary digit array
            
        Returns:
            Product as pentary digit array
        """
        # Handle zero
        if a == [0] or b == [0]:
            return [0]
        
        # Start with zero
        result = [0]
        
        # Multiply digit by digit
        for i, digit_b in enumerate(reversed(b)):
            if digit_b == 0:
                continue
            
            # Multiply a by single digit
            partial = self.converter.multiply_array_by_constant(a, digit_b)
            
            # Shift left by position
            partial = self.converter.shift_left_array(partial, i)
            
            # Add to result
            result = self.converter.add_arrays(result, partial)
        
        return result
    
    def multiply_pentary(self, a: str, b: str) -> str:
        """
        Multiply two pentary numbers.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Product in pentary
        """
        a_array = self.converter.string_to_array(a)
        b_array = self.converter.string_to_array(b)
        result_array = self.multiply_arrays(a_array, b_array)
        return self.converter.array_to_string(result_array)
    
    def divide_arrays(self, dividend: List[int], divisor: List[int]) -> Tuple[List[int], List[int]]:
        """
        Divide two pentary digit arrays using long division.
        
        Args:
            dividend: Dividend as pentary digit array
            divisor: Divisor as pentary digit array
            
        Returns:
            (quotient, remainder) as pentary digit arrays
        """
        # Handle division by zero
        if divisor == [0]:
            raise ValueError("Division by zero")
        
        # Handle zero dividend
        if dividend == [0]:
            return [0], [0]
        
        # Handle signs
        dividend_neg = dividend[0] < 0
        divisor_neg = divisor[0] < 0
        result_neg = dividend_neg != divisor_neg
        
        # Work with absolute values
        if dividend_neg:
            dividend = self.converter.negate_array(dividend)
        if divisor_neg:
            divisor = self.converter.negate_array(divisor)
        
        # Check if dividend < divisor
        if self.converter.compare_arrays(dividend, divisor) < 0:
            remainder = self.converter.negate_array(dividend) if dividend_neg else dividend
            return [0], remainder
        
        quotient = [0]
        remainder = [0]
        
        # Long division
        for digit in dividend:
            # Shift remainder left and add current digit
            remainder = self.converter.shift_left_array(remainder, 1)
            remainder[-1] = digit
            
            # Remove leading zeros
            while len(remainder) > 1 and remainder[0] == 0:
                remainder.pop(0)
            
            # Find how many times divisor fits into remainder
            count = 0
            while self.converter.compare_arrays(remainder, divisor) >= 0 and count < 2:
                remainder = self.converter.add_arrays(
                    remainder, 
                    self.converter.negate_array(divisor)
                )
                count += 1
            
            # Shift quotient left and add count
            quotient = self.converter.shift_left_array(quotient, 1)
            if count <= 2:
                quotient[-1] = count
            else:
                # Handle overflow - this shouldn't happen with proper algorithm
                quotient[-1] = 2
        
        # Remove leading zeros from quotient
        while len(quotient) > 1 and quotient[0] == 0:
            quotient.pop(0)
        
        # Apply sign to quotient
        if result_neg:
            quotient = self.converter.negate_array(quotient)
        
        # Remainder has same sign as dividend
        if dividend_neg:
            remainder = self.converter.negate_array(remainder)
        
        return quotient, remainder
    
    def divide_pentary(self, a: str, b: str) -> Tuple[str, str]:
        """
        Divide two pentary numbers.
        
        Args:
            a: Dividend (pentary number)
            b: Divisor (pentary number)
            
        Returns:
            (quotient, remainder) in pentary
        """
        a_array = self.converter.string_to_array(a)
        b_array = self.converter.string_to_array(b)
        quotient, remainder = self.divide_arrays(a_array, b_array)
        return (self.converter.array_to_string(quotient), 
                self.converter.array_to_string(remainder))
    
    def power_pentary(self, base: str, exponent: int) -> str:
        """
        Raise pentary number to an integer power.
        
        Args:
            base: Base (pentary number)
            exponent: Exponent (non-negative integer)
            
        Returns:
            Result in pentary
        """
        if exponent < 0:
            raise ValueError("Negative exponents not supported")
        
        if exponent == 0:
            return "+"  # 1 in pentary
        
        if exponent == 1:
            return base
        
        # Use binary exponentiation
        result = "+"  # 1
        current = base
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = self.multiply_pentary(result, current)
            current = self.multiply_pentary(current, current)
            exponent //= 2
        
        return result
    
    def modulo_pentary(self, a: str, m: str) -> str:
        """
        Compute a mod m in pentary.
        
        Args:
            a: Number (pentary)
            m: Modulus (pentary)
            
        Returns:
            a mod m in pentary
        """
        _, remainder = self.divide_pentary(a, m)
        return remainder
    
    def gcd_pentary(self, a: str, b: str) -> str:
        """
        Compute GCD of two pentary numbers using Euclidean algorithm.
        
        Args:
            a: First number (pentary)
            b: Second number (pentary)
            
        Returns:
            GCD in pentary
        """
        while b != "0":
            a, b = b, self.modulo_pentary(a, b)
        return a
    
    def abs_pentary(self, a: str) -> str:
        """
        Compute absolute value of pentary number.
        
        Args:
            a: Pentary number
            
        Returns:
            Absolute value in pentary
        """
        a_array = self.converter.string_to_array(a)
        if a_array[0] < 0:
            return self.converter.negate_pentary(a)
        return a
    
    def min_pentary(self, a: str, b: str) -> str:
        """Return minimum of two pentary numbers"""
        return a if self.converter.compare_pentary(a, b) <= 0 else b
    
    def max_pentary(self, a: str, b: str) -> str:
        """Return maximum of two pentary numbers"""
        return a if self.converter.compare_pentary(a, b) >= 0 else b


def main():
    """Demo and testing of extended pentary arithmetic"""
    print("=" * 70)
    print("Extended Pentary Arithmetic Operations")
    print("=" * 70)
    print()
    
    arith = PentaryArithmeticExtended()
    
    # Test multiplication
    print("Multiplication Tests:")
    print("-" * 70)
    
    test_cases = [
        ("⊕", "⊕", "2 × 2 = 4"),
        ("+⊕", "+", "7 × 1 = 7"),
        ("+0", "⊕", "5 × 2 = 10"),
        ("++", "++", "6 × 6 = 36"),
        ("+⊕", "-", "7 × -1 = -7"),
        ("-0", "-", "-5 × -1 = 5"),
    ]
    
    for a, b, description in test_cases:
        result = arith.multiply_pentary(a, b)
        a_dec = arith.converter.pentary_to_decimal(a)
        b_dec = arith.converter.pentary_to_decimal(b)
        r_dec = arith.converter.pentary_to_decimal(result)
        print(f"{description}")
        print(f"  {a} × {b} = {result}")
        print(f"  Verification: {a_dec} × {b_dec} = {r_dec}")
        print()
    
    # Test division
    print("=" * 70)
    print("Division Tests:")
    print("-" * 70)
    
    test_cases = [
        ("+0", "⊕", "5 ÷ 2"),
        ("⊕⊕", "+", "12 ÷ 1"),
        ("+⊕", "⊕", "7 ÷ 2"),
        ("++", "⊕", "6 ÷ 2"),
        ("+00", "+0", "25 ÷ 5"),
        ("-0", "⊕", "-5 ÷ 2"),
    ]
    
    for a, b, description in test_cases:
        quotient, remainder = arith.divide_pentary(a, b)
        a_dec = arith.converter.pentary_to_decimal(a)
        b_dec = arith.converter.pentary_to_decimal(b)
        q_dec = arith.converter.pentary_to_decimal(quotient)
        r_dec = arith.converter.pentary_to_decimal(remainder)
        print(f"{description}")
        print(f"  {a} ÷ {b} = {quotient} remainder {remainder}")
        print(f"  Verification: {a_dec} ÷ {b_dec} = {q_dec} R {r_dec}")
        print(f"  Check: {q_dec} × {b_dec} + {r_dec} = {q_dec * b_dec + r_dec}")
        print()
    
    # Test power
    print("=" * 70)
    print("Power Tests:")
    print("-" * 70)
    
    test_cases = [
        ("⊕", 2, "2^2"),
        ("+", 5, "1^5"),
        ("⊕", 3, "2^3"),
        ("+0", 2, "5^2"),
        ("++", 2, "6^2"),
    ]
    
    for base, exp, description in test_cases:
        result = arith.power_pentary(base, exp)
        base_dec = arith.converter.pentary_to_decimal(base)
        r_dec = arith.converter.pentary_to_decimal(result)
        print(f"{description}")
        print(f"  {base}^{exp} = {result}")
        print(f"  Verification: {base_dec}^{exp} = {r_dec}")
        print()
    
    # Test GCD
    print("=" * 70)
    print("GCD Tests:")
    print("-" * 70)
    
    test_cases = [
        ("⊕⊕", "++", "GCD(12, 6)"),
        ("+0", "+⊕", "GCD(5, 7)"),
        ("+00", "+0", "GCD(25, 5)"),
    ]
    
    for a, b, description in test_cases:
        result = arith.gcd_pentary(a, b)
        a_dec = arith.converter.pentary_to_decimal(a)
        b_dec = arith.converter.pentary_to_decimal(b)
        r_dec = arith.converter.pentary_to_decimal(result)
        print(f"{description}")
        print(f"  GCD({a}, {b}) = {result}")
        print(f"  Verification: GCD({a_dec}, {b_dec}) = {r_dec}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()