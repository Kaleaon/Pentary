#!/usr/bin/env python3
"""
Pentary Number System Converter
Converts between decimal, binary, and balanced pentary representations
"""

class PentaryConverter:
    """Converter for balanced pentary number system {-2, -1, 0, +1, +2}"""
    
    # Symbol mappings
    SYMBOLS = {
        -2: '⊖',  # Strong negative
        -1: '-',  # Weak negative
        0: '0',   # Zero
        1: '+',   # Weak positive
        2: '⊕'    # Strong positive
    }
    
    REVERSE_SYMBOLS = {v: k for k, v in SYMBOLS.items()}
    
    @staticmethod
    def decimal_to_pentary(n: int) -> str:
        """
        Convert decimal integer to balanced pentary string.
        
        Args:
            n: Decimal integer
            
        Returns:
            Balanced pentary string using symbols {⊖, -, 0, +, ⊕}
        """
        if n == 0:
            return "0"
        
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
                # 3 = 5 - 2, so we use -2 and carry 1
                # 4 = 5 - 1, so we use -1 and carry 1
                digits.append(remainder - 5)
                n += 1
        
        # Convert to symbols
        result = ''.join(PentaryConverter.SYMBOLS[d] for d in reversed(digits))
        
        # Apply negation if needed
        if negative:
            result = PentaryConverter.negate_pentary(result)
        
        return result
    
    @staticmethod
    def pentary_to_decimal(pentary_str: str) -> int:
        """
        Convert balanced pentary string to decimal integer.
        
        Args:
            pentary_str: Balanced pentary string
            
        Returns:
            Decimal integer
        """
        result = 0
        power = 1
        
        for symbol in reversed(pentary_str):
            if symbol not in PentaryConverter.REVERSE_SYMBOLS:
                raise ValueError(f"Invalid pentary symbol: {symbol}")
            
            digit_value = PentaryConverter.REVERSE_SYMBOLS[symbol]
            result += digit_value * power
            power *= 5
        
        return result
    
    @staticmethod
    def negate_pentary(pentary_str: str) -> str:
        """
        Negate a balanced pentary number (flip all digits).
        
        Args:
            pentary_str: Balanced pentary string
            
        Returns:
            Negated pentary string
        """
        negation_map = {
            '⊖': '⊕',
            '-': '+',
            '0': '0',
            '+': '-',
            '⊕': '⊖'
        }
        
        return ''.join(negation_map[c] for c in pentary_str)
    
    @staticmethod
    def binary_to_pentary(binary_str: str) -> str:
        """
        Convert binary string to balanced pentary.
        
        Args:
            binary_str: Binary string (e.g., "1010")
            
        Returns:
            Balanced pentary string
        """
        decimal = int(binary_str, 2)
        return PentaryConverter.decimal_to_pentary(decimal)
    
    @staticmethod
    def pentary_to_binary(pentary_str: str, bits: int = None) -> str:
        """
        Convert balanced pentary to binary string.
        
        Args:
            pentary_str: Balanced pentary string
            bits: Number of bits (optional, for padding)
            
        Returns:
            Binary string
        """
        decimal = PentaryConverter.pentary_to_decimal(pentary_str)
        
        if decimal < 0:
            raise ValueError("Cannot convert negative pentary to unsigned binary")
        
        binary = bin(decimal)[2:]  # Remove '0b' prefix
        
        if bits:
            binary = binary.zfill(bits)
        
        return binary
    
    @staticmethod
    def add_pentary(a: str, b: str) -> str:
        """
        Add two balanced pentary numbers.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Sum in pentary
        """
        # Convert to decimal, add, convert back
        dec_a = PentaryConverter.pentary_to_decimal(a)
        dec_b = PentaryConverter.pentary_to_decimal(b)
        return PentaryConverter.decimal_to_pentary(dec_a + dec_b)
    
    @staticmethod
    def subtract_pentary(a: str, b: str) -> str:
        """
        Subtract two balanced pentary numbers.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Difference in pentary
        """
        # Subtraction is addition with negation
        return PentaryConverter.add_pentary(a, PentaryConverter.negate_pentary(b))
    
    @staticmethod
    def multiply_pentary_by_constant(pentary_str: str, constant: int) -> str:
        """
        Multiply pentary number by a constant {-2, -1, 0, 1, 2}.
        This is efficient as it only requires shifts and negations.
        
        Args:
            pentary_str: Pentary number
            constant: Multiplier in {-2, -1, 0, 1, 2}
            
        Returns:
            Product in pentary
        """
        if constant not in [-2, -1, 0, 1, 2]:
            raise ValueError("Constant must be in {-2, -1, 0, 1, 2}")
        
        if constant == 0:
            return "0"
        elif constant == 1:
            return pentary_str
        elif constant == -1:
            return PentaryConverter.negate_pentary(pentary_str)
        elif constant == 2:
            # Shift left (multiply by 5) then divide by 2.5
            # Actually, for pentary: x * 2 requires proper arithmetic
            dec = PentaryConverter.pentary_to_decimal(pentary_str)
            return PentaryConverter.decimal_to_pentary(dec * 2)
        elif constant == -2:
            dec = PentaryConverter.pentary_to_decimal(pentary_str)
            return PentaryConverter.decimal_to_pentary(dec * -2)
    
    @staticmethod
    def shift_left_pentary(pentary_str: str, positions: int = 1) -> str:
        """
        Shift pentary number left (multiply by 5^positions).
        
        Args:
            pentary_str: Pentary number
            positions: Number of positions to shift
            
        Returns:
            Shifted pentary number
        """
        return pentary_str + '0' * positions
    
    @staticmethod
    def shift_right_pentary(pentary_str: str, positions: int = 1) -> str:
        """
        Shift pentary number right (divide by 5^positions).
        
        Args:
            pentary_str: Pentary number
            positions: Number of positions to shift
            
        Returns:
            Shifted pentary number (truncated)
        """
        if positions >= len(pentary_str):
            return "0"
        return pentary_str[:-positions] if positions > 0 else pentary_str
    
    @staticmethod
    def multiply_pentary(a: str, b: str) -> str:
        """
        Multiply two balanced pentary numbers.
        
        Args:
            a: First pentary number
            b: Second pentary number
            
        Returns:
            Product in pentary
        """
        # Convert to decimal, multiply, convert back
        dec_a = PentaryConverter.pentary_to_decimal(a)
        dec_b = PentaryConverter.pentary_to_decimal(b)
        return PentaryConverter.decimal_to_pentary(dec_a * dec_b)
    
    @staticmethod
    def divide_pentary(a: str, b: str) -> str:
        """
        Divide two balanced pentary numbers (integer division).
        
        Args:
            a: Dividend (pentary number)
            b: Divisor (pentary number)
            
        Returns:
            Quotient in pentary (truncated toward zero)
        """
        dec_a = PentaryConverter.pentary_to_decimal(a)
        dec_b = PentaryConverter.pentary_to_decimal(b)
        
        if dec_b == 0:
            raise ValueError("Division by zero")
        
        # Integer division truncated toward zero
        quotient = int(dec_a / dec_b)
        return PentaryConverter.decimal_to_pentary(quotient)
    
    @staticmethod
    def modulo_pentary(a: str, b: str) -> str:
        """
        Calculate modulo of two balanced pentary numbers.
        
        Args:
            a: Dividend (pentary number)
            b: Divisor (pentary number)
            
        Returns:
            Remainder in pentary
        """
        dec_a = PentaryConverter.pentary_to_decimal(a)
        dec_b = PentaryConverter.pentary_to_decimal(b)
        
        if dec_b == 0:
            raise ValueError("Division by zero")
        
        remainder = dec_a % dec_b
        return PentaryConverter.decimal_to_pentary(remainder)
    
    @staticmethod
    def format_pentary(pentary_str: str, width: int = None, align: str = 'right') -> str:
        """
        Format pentary string with padding.
        
        Args:
            pentary_str: Pentary number
            width: Desired width
            align: 'left' or 'right'
            
        Returns:
            Formatted pentary string
        """
        if width is None:
            return pentary_str
        
        if len(pentary_str) >= width:
            return pentary_str
        
        padding = '0' * (width - len(pentary_str))
        
        if align == 'left':
            return pentary_str + padding
        else:
            return padding + pentary_str


def main():
    """Demo and testing of pentary converter"""
    print("=" * 60)
    print("Pentary Number System Converter")
    print("=" * 60)
    print()
    
    # Test conversions
    test_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, -5, -10]
    
    print("Decimal to Pentary Conversions:")
    print("-" * 60)
    print(f"{'Decimal':<10} {'Pentary':<15} {'Back to Decimal':<20}")
    print("-" * 60)
    
    for num in test_numbers:
        pentary = PentaryConverter.decimal_to_pentary(num)
        back = PentaryConverter.pentary_to_decimal(pentary)
        print(f"{num:<10} {pentary:<15} {back:<20}")
    
    print()
    print("=" * 60)
    print("Arithmetic Operations:")
    print("-" * 60)
    
    # Test addition
    a, b = "⊕+", "+⊖"  # 7 + 3 = 10
    result = PentaryConverter.add_pentary(a, b)
    print(f"{a} + {b} = {result}")
    print(f"  ({PentaryConverter.pentary_to_decimal(a)} + "
          f"{PentaryConverter.pentary_to_decimal(b)} = "
          f"{PentaryConverter.pentary_to_decimal(result)})")
    print()
    
    # Test subtraction
    a, b = "⊕⊕", "+0"  # 12 - 5 = 7
    result = PentaryConverter.subtract_pentary(a, b)
    print(f"{a} - {b} = {result}")
    print(f"  ({PentaryConverter.pentary_to_decimal(a)} - "
          f"{PentaryConverter.pentary_to_decimal(b)} = "
          f"{PentaryConverter.pentary_to_decimal(result)})")
    print()
    
    # Test negation
    a = "+⊕-"
    neg = PentaryConverter.negate_pentary(a)
    print(f"Negate({a}) = {neg}")
    print(f"  ({PentaryConverter.pentary_to_decimal(a)} → "
          f"{PentaryConverter.pentary_to_decimal(neg)})")
    print()
    
    # Test multiplication by constants
    a = "+⊕"  # 7
    for const in [-2, -1, 0, 1, 2]:
        result = PentaryConverter.multiply_pentary_by_constant(a, const)
        print(f"{a} × {const} = {result}")
        print(f"  ({PentaryConverter.pentary_to_decimal(a)} × {const} = "
              f"{PentaryConverter.pentary_to_decimal(result)})")
    print()
    
    # Test shifts
    a = "+⊕"  # 7
    left = PentaryConverter.shift_left_pentary(a, 1)
    print(f"{a} << 1 = {left}")
    print(f"  ({PentaryConverter.pentary_to_decimal(a)} × 5 = "
          f"{PentaryConverter.pentary_to_decimal(left)})")
    print()
    
    a = "+⊕0"  # 35
    right = PentaryConverter.shift_right_pentary(a, 1)
    print(f"{a} >> 1 = {right}")
    print(f"  ({PentaryConverter.pentary_to_decimal(a)} ÷ 5 = "
          f"{PentaryConverter.pentary_to_decimal(right)})")
    print()
    
    print("=" * 60)
    print("Binary to Pentary Conversions:")
    print("-" * 60)
    
    binary_tests = ["1010", "11111111", "10000000", "1111"]
    for binary in binary_tests:
        pentary = PentaryConverter.binary_to_pentary(binary)
        decimal = int(binary, 2)
        print(f"Binary: {binary:<12} → Pentary: {pentary:<10} (Decimal: {decimal})")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()