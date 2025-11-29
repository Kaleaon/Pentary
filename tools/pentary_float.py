#!/usr/bin/env python3
"""
Pentary Floating Point Number System
Implements floating-point arithmetic in balanced pentary
"""

from typing import Tuple, Optional
import math
from pentary_converter_optimized import PentaryConverterOptimized


class PentaryFloat:
    """
    Pentary floating-point number.
    
    Format: mantissa × 5^exponent
    where mantissa is a balanced pentary number
    """
    
    # Special values
    NAN = "NaN"
    INF = "Inf"
    NEG_INF = "-Inf"
    
    def __init__(self, mantissa: str = "0", exponent: int = 0, special: Optional[str] = None):
        """
        Initialize pentary float.
        
        Args:
            mantissa: Mantissa in pentary (normalized: first digit non-zero)
            exponent: Exponent (power of 5)
            special: Special value (NaN, Inf, -Inf) or None
        """
        self.converter = PentaryConverterOptimized()
        self.special = special
        
        if special:
            self.mantissa = "0"
            self.exponent = 0
        else:
            self.mantissa = mantissa
            self.exponent = exponent
            self._normalize()
    
    def _normalize(self):
        """Normalize the mantissa (remove leading zeros, adjust exponent)"""
        if self.special:
            return
        
        # Remove leading zeros
        mantissa_stripped = self.mantissa.lstrip('0')
        
        if not mantissa_stripped or mantissa_stripped == '':
            self.mantissa = "0"
            self.exponent = 0
            return
        
        # Adjust exponent based on removed zeros
        zeros_removed = len(self.mantissa) - len(mantissa_stripped)
        self.mantissa = mantissa_stripped
        self.exponent += zeros_removed
    
    @classmethod
    def from_decimal(cls, value: float, precision: int = 10) -> 'PentaryFloat':
        """
        Create PentaryFloat from decimal float.
        
        Args:
            value: Decimal floating-point value
            precision: Number of mantissa digits to keep
            
        Returns:
            PentaryFloat instance
        """
        converter = PentaryConverterOptimized()
        
        # Handle special values
        if math.isnan(value):
            return cls(special=cls.NAN)
        if math.isinf(value):
            return cls(special=cls.INF if value > 0 else cls.NEG_INF)
        if value == 0:
            return cls("0", 0)
        
        # Handle sign
        negative = value < 0
        value = abs(value)
        
        # Find exponent to normalize value to [1, 5)
        exponent = 0
        while value >= 5:
            value /= 5
            exponent += 1
        while value < 1:
            value *= 5
            exponent -= 1
        
        # Convert mantissa to pentary with desired precision
        # Scale up to integer, convert, then adjust exponent
        scale = 5 ** precision
        mantissa_int = int(value * scale)
        mantissa_pent = converter.decimal_to_pentary(mantissa_int)
        
        # Adjust exponent for scaling
        exponent -= precision
        
        # Apply sign
        if negative:
            mantissa_pent = converter.negate_pentary(mantissa_pent)
        
        return cls(mantissa_pent, exponent)
    
    def to_decimal(self) -> float:
        """
        Convert PentaryFloat to decimal float.
        
        Returns:
            Decimal floating-point value
        """
        if self.special == self.NAN:
            return float('nan')
        if self.special == self.INF:
            return float('inf')
        if self.special == self.NEG_INF:
            return float('-inf')
        
        mantissa_dec = self.converter.pentary_to_decimal(self.mantissa)
        return mantissa_dec * (5 ** self.exponent)
    
    def add(self, other: 'PentaryFloat') -> 'PentaryFloat':
        """Add two pentary floats"""
        # Handle special values
        if self.special or other.special:
            if self.special == self.NAN or other.special == self.NAN:
                return PentaryFloat(special=self.NAN)
            if self.special == self.INF:
                if other.special == self.NEG_INF:
                    return PentaryFloat(special=self.NAN)
                return PentaryFloat(special=self.INF)
            if self.special == self.NEG_INF:
                if other.special == self.INF:
                    return PentaryFloat(special=self.NAN)
                return PentaryFloat(special=self.NEG_INF)
            # other has special value, self doesn't
            return PentaryFloat(special=other.special)
        
        # Align exponents
        if self.exponent > other.exponent:
            # Shift other's mantissa right
            diff = self.exponent - other.exponent
            other_mantissa = self.converter.shift_right_pentary(other.mantissa, diff)
            result_exp = self.exponent
        elif other.exponent > self.exponent:
            # Shift self's mantissa right
            diff = other.exponent - self.exponent
            self_mantissa = self.converter.shift_right_pentary(self.mantissa, diff)
            result_exp = other.exponent
            other_mantissa = other.mantissa
            self_mantissa_to_use = self_mantissa
        else:
            # Same exponent
            other_mantissa = other.mantissa
            self_mantissa_to_use = self.mantissa
            result_exp = self.exponent
        
        # Add mantissas
        if self.exponent > other.exponent:
            result_mantissa = self.converter.add_pentary(self.mantissa, other_mantissa)
        else:
            result_mantissa = self.converter.add_pentary(
                self_mantissa_to_use if 'self_mantissa_to_use' in locals() else self.mantissa,
                other_mantissa
            )
        
        return PentaryFloat(result_mantissa, result_exp)
    
    def subtract(self, other: 'PentaryFloat') -> 'PentaryFloat':
        """Subtract two pentary floats"""
        # Negate other and add
        negated_other = PentaryFloat(
            self.converter.negate_pentary(other.mantissa) if not other.special else "0",
            other.exponent,
            self.NEG_INF if other.special == self.INF else (self.INF if other.special == self.NEG_INF else other.special)
        )
        return self.add(negated_other)
    
    def multiply(self, other: 'PentaryFloat') -> 'PentaryFloat':
        """Multiply two pentary floats"""
        # Handle special values
        if self.special or other.special:
            if self.special == self.NAN or other.special == self.NAN:
                return PentaryFloat(special=self.NAN)
            # Inf × 0 = NaN
            if (self.special in [self.INF, self.NEG_INF] and other.mantissa == "0") or \
               (other.special in [self.INF, self.NEG_INF] and self.mantissa == "0"):
                return PentaryFloat(special=self.NAN)
            # Determine sign of infinity
            self_neg = self.special == self.NEG_INF or (not self.special and self.mantissa[0] == '⊖' or self.mantissa[0] == '-')
            other_neg = other.special == self.NEG_INF or (not other.special and other.mantissa[0] == '⊖' or other.mantissa[0] == '-')
            result_neg = self_neg != other_neg
            return PentaryFloat(special=self.NEG_INF if result_neg else self.INF)
        
        # Multiply mantissas (convert to decimal for simplicity)
        self_dec = self.converter.pentary_to_decimal(self.mantissa)
        other_dec = self.converter.pentary_to_decimal(other.mantissa)
        result_dec = self_dec * other_dec
        result_mantissa = self.converter.decimal_to_pentary(result_dec)
        
        # Add exponents
        result_exp = self.exponent + other.exponent
        
        return PentaryFloat(result_mantissa, result_exp)
    
    def divide(self, other: 'PentaryFloat') -> 'PentaryFloat':
        """Divide two pentary floats"""
        # Handle special values
        if self.special or other.special:
            if self.special == self.NAN or other.special == self.NAN:
                return PentaryFloat(special=self.NAN)
            # Inf / Inf = NaN
            if self.special in [self.INF, self.NEG_INF] and other.special in [self.INF, self.NEG_INF]:
                return PentaryFloat(special=self.NAN)
            # x / Inf = 0
            if other.special in [self.INF, self.NEG_INF]:
                return PentaryFloat("0", 0)
            # Inf / x = Inf
            if self.special in [self.INF, self.NEG_INF]:
                other_neg = other.mantissa[0] in ['⊖', '-']
                self_neg = self.special == self.NEG_INF
                result_neg = self_neg != other_neg
                return PentaryFloat(special=self.NEG_INF if result_neg else self.INF)
        
        # Division by zero
        if other.mantissa == "0":
            self_neg = self.mantissa[0] in ['⊖', '-']
            return PentaryFloat(special=self.NEG_INF if self_neg else self.INF)
        
        # Divide mantissas (convert to decimal for simplicity)
        self_dec = self.converter.pentary_to_decimal(self.mantissa)
        other_dec = self.converter.pentary_to_decimal(other.mantissa)
        
        # Scale to maintain precision
        scale = 5 ** 10  # 10 digits of precision
        result_dec = (self_dec * scale) // other_dec
        result_mantissa = self.converter.decimal_to_pentary(result_dec)
        
        # Subtract exponents and adjust for scaling
        result_exp = self.exponent - other.exponent - 10
        
        return PentaryFloat(result_mantissa, result_exp)
    
    def __str__(self) -> str:
        """String representation"""
        if self.special:
            return self.special
        return f"{self.mantissa} × 5^{self.exponent}"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        if self.special:
            return f"PentaryFloat({self.special})"
        return f"PentaryFloat('{self.mantissa}', {self.exponent})"
    
    def __eq__(self, other: 'PentaryFloat') -> bool:
        """Equality comparison"""
        if self.special or other.special:
            return self.special == other.special
        return self.mantissa == other.mantissa and self.exponent == other.exponent
    
    def __lt__(self, other: 'PentaryFloat') -> bool:
        """Less than comparison"""
        return self.to_decimal() < other.to_decimal()
    
    def __le__(self, other: 'PentaryFloat') -> bool:
        """Less than or equal comparison"""
        return self.to_decimal() <= other.to_decimal()
    
    def __gt__(self, other: 'PentaryFloat') -> bool:
        """Greater than comparison"""
        return self.to_decimal() > other.to_decimal()
    
    def __ge__(self, other: 'PentaryFloat') -> bool:
        """Greater than or equal comparison"""
        return self.to_decimal() >= other.to_decimal()


def main():
    """Demo and testing of pentary floating point"""
    print("=" * 70)
    print("Pentary Floating Point Number System")
    print("=" * 70)
    print()
    
    # Test creation from decimal
    print("Creating Pentary Floats from Decimal:")
    print("-" * 70)
    
    test_values = [0.0, 1.0, 2.5, 5.0, 10.0, 0.1, 0.01, 100.0, -5.5, 3.14159]
    
    for value in test_values:
        pf = PentaryFloat.from_decimal(value, precision=8)
        back = pf.to_decimal()
        print(f"Decimal: {value:>10.5f} → Pentary: {pf} → Back: {back:>10.5f}")
    
    print()
    print("=" * 70)
    print("Arithmetic Operations:")
    print("-" * 70)
    
    # Test addition
    a = PentaryFloat.from_decimal(2.5, precision=6)
    b = PentaryFloat.from_decimal(3.5, precision=6)
    result = a.add(b)
    print(f"Addition: {a.to_decimal()} + {b.to_decimal()} = {result.to_decimal()}")
    print(f"  Pentary: {a} + {b} = {result}")
    print()
    
    # Test subtraction
    a = PentaryFloat.from_decimal(10.0, precision=6)
    b = PentaryFloat.from_decimal(3.5, precision=6)
    result = a.subtract(b)
    print(f"Subtraction: {a.to_decimal()} - {b.to_decimal()} = {result.to_decimal()}")
    print(f"  Pentary: {a} - {b} = {result}")
    print()
    
    # Test multiplication
    a = PentaryFloat.from_decimal(2.5, precision=6)
    b = PentaryFloat.from_decimal(4.0, precision=6)
    result = a.multiply(b)
    print(f"Multiplication: {a.to_decimal()} × {b.to_decimal()} = {result.to_decimal()}")
    print(f"  Pentary: {a} × {b} = {result}")
    print()
    
    # Test division
    a = PentaryFloat.from_decimal(10.0, precision=6)
    b = PentaryFloat.from_decimal(2.5, precision=6)
    result = a.divide(b)
    print(f"Division: {a.to_decimal()} ÷ {b.to_decimal()} = {result.to_decimal()}")
    print(f"  Pentary: {a} ÷ {b} = {result}")
    print()
    
    # Test special values
    print("=" * 70)
    print("Special Values:")
    print("-" * 70)
    
    nan = PentaryFloat(special=PentaryFloat.NAN)
    inf = PentaryFloat(special=PentaryFloat.INF)
    neg_inf = PentaryFloat(special=PentaryFloat.NEG_INF)
    
    print(f"NaN: {nan} → {nan.to_decimal()}")
    print(f"Inf: {inf} → {inf.to_decimal()}")
    print(f"-Inf: {neg_inf} → {neg_inf.to_decimal()}")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()