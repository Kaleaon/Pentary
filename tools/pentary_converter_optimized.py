#!/usr/bin/env python3
"""
Optimized Pentary Number System Converter
Uses integer arrays internally for better performance
"""

from typing import List, Tuple, Optional
import functools

class PentaryConverterOptimized:
    """Optimized converter for balanced pentary number system {-2, -1, 0, +1, +2}"""
    
    # Symbol mappings
    SYMBOLS = {
        -2: '⊖',  # Strong negative
        -1: '-',  # Weak negative
        0: '0',   # Zero
        1: '+',   # Weak positive
        2: '⊕'    # Strong positive
    }
    
    REVERSE_SYMBOLS = {v: k for k, v in SYMBOLS.items()}
    
    # Lookup table for digit addition (digit_a, digit_b, carry_in) -> (sum_digit, carry_out)
    ADD_LOOKUP = {}
    
    # Lookup table for digit multiplication (digit, constant) -> (result_digit, carry)
    MUL_LOOKUP = {}
    
    @classmethod
    def _initialize_lookup_tables(cls):
        """Initialize lookup tables for fast operations"""
        if cls.ADD_LOOKUP:
            return  # Already initialized
        
        # Build addition lookup table
        for a in range(-2, 3):
            for b in range(-2, 3):
                for carry_in in range(-1, 2):
                    total = a + b + carry_in
                    
                    # Convert to balanced pentary
                    if -2 <= total <= 2:
                        cls.ADD_LOOKUP[(a, b, carry_in)] = (total, 0)
                    elif total > 2:
                        # Need positive carry
                        cls.ADD_LOOKUP[(a, b, carry_in)] = (total - 5, 1)
                    else:  # total < -2
                        # Need negative carry
                        cls.ADD_LOOKUP[(a, b, carry_in)] = (total + 5, -1)
        
        # Build multiplication lookup table
        for digit in range(-2, 3):
            for constant in range(-2, 3):
                product = digit * constant
                
                if -2 <= product <= 2:
                    cls.MUL_LOOKUP[(digit, constant)] = (product, 0)
                elif product > 2:
                    if product == 3:
                        cls.MUL_LOOKUP[(digit, constant)] = (-2, 1)
                    elif product == 4:
                        cls.MUL_LOOKUP[(digit, constant)] = (-1, 1)
                    else:  # product >= 5
                        carry = product // 5
                        remainder = product % 5
                        if remainder > 2:
                            remainder -= 5
                            carry += 1
                        cls.MUL_LOOKUP[(digit, constant)] = (remainder, carry)
                else:  # product < -2
                    if product == -3:
                        cls.MUL_LOOKUP[(digit, constant)] = (2, -1)
                    elif product == -4:
                        cls.MUL_LOOKUP[(digit, constant)] = (1, -1)
                    else:  # product <= -5
                        carry = -((-product) // 5)
                        remainder = -((-product) % 5)
                        if remainder < -2:
                            remainder += 5
                            carry -= 1
                        cls.MUL_LOOKUP[(digit, constant)] = (remainder, carry)
    
    def __init__(self):
        """Initialize the optimized converter"""
        self._initialize_lookup_tables()
        self._conversion_cache = {}  # Cache for frequent conversions
    
    @staticmethod
    def decimal_to_array(n: int) -> List[int]:
        """
        Convert decimal integer to pentary digit array.
        
        Args:
            n: Decimal integer
            
        Returns:
            List of pentary digits (most significant first)
        """
        if n == 0:
            return [0]
        
        digits = []
        negative = n < 0
        n = abs(n)
        
        while n > 0:
            remainder = n % 5
            n = n // 5
            
            # Convert to balanced form
            if remainder <= 2:
                digits.append(remainder)
            else:
                # remainder is 3 or 4
                digits.append(remainder - 5)
                n += 1
        
        # Reverse to get most significant first
        digits.reverse()
        
        # Apply negation if needed
        if negative:
            digits = [-d for d in digits]
        
        return digits
    
    @staticmethod
    def array_to_decimal(digits: List[int]) -> int:
        """
        Convert pentary digit array to decimal integer.
        
        Args:
            digits: List of pentary digits (most significant first)
            
        Returns:
            Decimal integer
        """
        result = 0
        power = 1
        
        for digit in reversed(digits):
            result += digit * power
            power *= 5
        
        return result
    
    @staticmethod
    def array_to_string(digits: List[int]) -> str:
        """Convert digit array to symbol string"""
        return ''.join(PentaryConverterOptimized.SYMBOLS[d] for d in digits)
    
    @staticmethod
    def string_to_array(pentary_str: str) -> List[int]:
        """Convert symbol string to digit array"""
        return [PentaryConverterOptimized.REVERSE_SYMBOLS[c] for c in pentary_str]
    
    def decimal_to_pentary(self, n: int, use_cache: bool = True) -> str:
        """
        Convert decimal integer to pentary string (with optional caching).
        
        Args:
            n: Decimal integer
            use_cache: Whether to use conversion cache
            
        Returns:
            Pentary string
        """
        if use_cache and n in self._conversion_cache:
            return self._conversion_cache[n]
        
        digits = self.decimal_to_array(n)
        result = self.array_to_string(digits)
        
        if use_cache and len(self._conversion_cache) < 10000:  # Limit cache size
            self._conversion_cache[n] = result
        
        return result
    
    def pentary_to_decimal(self, pentary_str: str) -> int:
        """
        Convert pentary string to decimal integer.
        
        Args:
            pentary_str: Pentary string
            
        Returns:
            Decimal integer
        """
        digits = self.string_to_array(pentary_str)
        return self.array_to_decimal(digits)
    
    def add_arrays(self, a: List[int], b: List[int]) -> List[int]:
        """
        Add two pentary digit arrays using lookup table.
        
        Args:
            a: First pentary digit array
            b: Second pentary digit array
            
        Returns:
            Sum as pentary digit array
        """
        # Pad to same length
        max_len = max(len(a), len(b))
        a_padded = [0] * (max_len - len(a)) + a
        b_padded = [0] * (max_len - len(b)) + b
        
        result = []
        carry = 0
        
        # Add from right to left
        for i in range(max_len - 1, -1, -1):
            sum_digit, carry_out = self.ADD_LOOKUP[(a_padded[i], b_padded[i], carry)]
            result.insert(0, sum_digit)
            carry = carry_out
        
        # Handle final carry
        if carry != 0:
            result.insert(0, carry)
        
        # Remove leading zeros
        while len(result) > 1 and result[0] == 0:
            result.pop(0)
        
        return result
    
    def add_pentary(self, a: str, b: str) -> str:
        """
        Add two pentary numbers using optimized array operations.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Sum in pentary
        """
        a_array = self.string_to_array(a)
        b_array = self.string_to_array(b)
        result_array = self.add_arrays(a_array, b_array)
        return self.array_to_string(result_array)
    
    def negate_array(self, digits: List[int]) -> List[int]:
        """Negate a pentary digit array"""
        return [-d for d in digits]
    
    def negate_pentary(self, pentary_str: str) -> str:
        """
        Negate a pentary number.
        
        Args:
            pentary_str: Pentary string
            
        Returns:
            Negated pentary string
        """
        digits = self.string_to_array(pentary_str)
        negated = self.negate_array(digits)
        return self.array_to_string(negated)
    
    def subtract_pentary(self, a: str, b: str) -> str:
        """
        Subtract two pentary numbers.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Difference in pentary
        """
        return self.add_pentary(a, self.negate_pentary(b))
    
    def multiply_array_by_constant(self, digits: List[int], constant: int) -> List[int]:
        """
        Multiply pentary digit array by a constant {-2, -1, 0, 1, 2}.
        
        Args:
            digits: Pentary digit array
            constant: Multiplier in {-2, -1, 0, 1, 2}
            
        Returns:
            Product as pentary digit array
        """
        if constant == 0:
            return [0]
        elif constant == 1:
            return digits.copy()
        elif constant == -1:
            return self.negate_array(digits)
        
        result = []
        carry = 0
        
        # Multiply from right to left
        for i in range(len(digits) - 1, -1, -1):
            prod_digit, carry_out = self.MUL_LOOKUP[(digits[i], constant)]
            
            # Add previous carry
            if carry != 0:
                sum_digit, new_carry = self.ADD_LOOKUP[(prod_digit, carry, 0)]
                prod_digit = sum_digit
                carry = carry_out + new_carry
            else:
                carry = carry_out
            
            result.insert(0, prod_digit)
        
        # Handle final carry
        while carry != 0:
            if -2 <= carry <= 2:
                result.insert(0, carry)
                carry = 0
            else:
                digit = carry % 5
                if digit > 2:
                    digit -= 5
                    carry = carry // 5 + 1
                else:
                    carry = carry // 5
                result.insert(0, digit)
        
        # Remove leading zeros
        while len(result) > 1 and result[0] == 0:
            result.pop(0)
        
        return result
    
    def multiply_pentary_by_constant(self, pentary_str: str, constant: int) -> str:
        """
        Multiply pentary number by a constant {-2, -1, 0, 1, 2}.
        
        Args:
            pentary_str: Pentary number
            constant: Multiplier in {-2, -1, 0, 1, 2}
            
        Returns:
            Product in pentary
        """
        digits = self.string_to_array(pentary_str)
        result = self.multiply_array_by_constant(digits, constant)
        return self.array_to_string(result)
    
    def shift_left_array(self, digits: List[int], positions: int = 1) -> List[int]:
        """Shift pentary array left (multiply by 5^positions)"""
        return digits + [0] * positions
    
    def shift_right_array(self, digits: List[int], positions: int = 1) -> List[int]:
        """Shift pentary array right (divide by 5^positions)"""
        if positions >= len(digits):
            return [0]
        return digits[:-positions] if positions > 0 else digits
    
    def shift_left_pentary(self, pentary_str: str, positions: int = 1) -> str:
        """Shift pentary number left (multiply by 5^positions)"""
        return pentary_str + '0' * positions
    
    def shift_right_pentary(self, pentary_str: str, positions: int = 1) -> str:
        """Shift pentary number right (divide by 5^positions)"""
        if positions >= len(pentary_str):
            return "0"
        return pentary_str[:-positions] if positions > 0 else pentary_str
    
    def compare_arrays(self, a: List[int], b: List[int]) -> int:
        """
        Compare two pentary digit arrays.
        
        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        # Pad to same length
        max_len = max(len(a), len(b))
        a_padded = [0] * (max_len - len(a)) + a
        b_padded = [0] * (max_len - len(b)) + b
        
        # Compare digit by digit from most significant
        for i in range(max_len):
            if a_padded[i] < b_padded[i]:
                return -1
            elif a_padded[i] > b_padded[i]:
                return 1
        
        return 0
    
    def compare_pentary(self, a: str, b: str) -> int:
        """
        Compare two pentary numbers.
        
        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        a_array = self.string_to_array(a)
        b_array = self.string_to_array(b)
        return self.compare_arrays(a_array, b_array)
    
    def clear_cache(self):
        """Clear the conversion cache"""
        self._conversion_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'size': len(self._conversion_cache),
            'max_size': 10000
        }


def main():
    """Demo and testing of optimized pentary converter"""
    print("=" * 70)
    print("Optimized Pentary Number System Converter")
    print("=" * 70)
    print()
    
    converter = PentaryConverterOptimized()
    
    # Test conversions
    test_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, -5, -10]
    
    print("Decimal to Pentary Conversions:")
    print("-" * 70)
    print(f"{'Decimal':<10} {'Pentary':<15} {'Array':<25} {'Back':<10}")
    print("-" * 70)
    
    for num in test_numbers:
        pentary = converter.decimal_to_pentary(num)
        array = converter.decimal_to_array(num)
        back = converter.pentary_to_decimal(pentary)
        array_str = str(array)
        print(f"{num:<10} {pentary:<15} {array_str:<25} {back:<10}")
    
    print()
    print("=" * 70)
    print("Arithmetic Operations (Optimized):")
    print("-" * 70)
    
    # Test addition
    a, b = "⊕+", "+⊖"  # 7 + 3 = 10
    result = converter.add_pentary(a, b)
    print(f"{a} + {b} = {result}")
    print(f"  ({converter.pentary_to_decimal(a)} + "
          f"{converter.pentary_to_decimal(b)} = "
          f"{converter.pentary_to_decimal(result)})")
    print()
    
    # Test subtraction
    a, b = "⊕⊕", "+0"  # 12 - 5 = 7
    result = converter.subtract_pentary(a, b)
    print(f"{a} - {b} = {result}")
    print(f"  ({converter.pentary_to_decimal(a)} - "
          f"{converter.pentary_to_decimal(b)} = "
          f"{converter.pentary_to_decimal(result)})")
    print()
    
    # Test multiplication by constants
    a = "+⊕"  # 7
    for const in [-2, -1, 0, 1, 2]:
        result = converter.multiply_pentary_by_constant(a, const)
        print(f"{a} × {const} = {result}")
        print(f"  ({converter.pentary_to_decimal(a)} × {const} = "
              f"{converter.pentary_to_decimal(result)})")
    
    print()
    print("=" * 70)
    print("Cache Statistics:")
    print("-" * 70)
    stats = converter.get_cache_stats()
    print(f"Cache size: {stats['size']}/{stats['max_size']}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()