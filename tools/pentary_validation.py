#!/usr/bin/env python3
"""
Pentary Validation and Error Handling
Provides input validation, error recovery, and safe operations
"""

import logging
from typing import Optional, Callable, Any, Union
from functools import wraps


class PentaryValidationError(Exception):
    """Base exception for pentary validation errors"""
    pass


class InvalidPentaryStringError(PentaryValidationError):
    """Raised when pentary string contains invalid characters"""
    pass


class InvalidPentaryDigitError(PentaryValidationError):
    """Raised when pentary digit is out of range"""
    pass


class PentaryOperationError(Exception):
    """Base exception for pentary operation errors"""
    pass


class DivisionByZeroError(PentaryOperationError):
    """Raised when attempting division by zero"""
    pass


class OverflowError(PentaryOperationError):
    """Raised when operation results in overflow (if limits are set)"""
    pass


class PentaryValidator:
    """Validator for pentary numbers and operations"""
    
    VALID_SYMBOLS = {'⊖', '-', '0', '+', '⊕'}
    VALID_DIGITS = {-2, -1, 0, 1, 2}
    
    # Symbol to digit mapping
    SYMBOL_TO_DIGIT = {
        '⊖': -2,
        '-': -1,
        '0': 0,
        '+': 1,
        '⊕': 2
    }
    
    def __init__(self, strict: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise exceptions on validation errors
            logger: Optional logger for validation messages
        """
        self.strict = strict
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_pentary_string(self, pentary_str: str) -> bool:
        """
        Validate pentary string format.
        
        Args:
            pentary_str: String to validate
            
        Returns:
            True if valid
            
        Raises:
            InvalidPentaryStringError: If string is invalid and strict mode is on
        """
        if not pentary_str:
            msg = "Empty pentary string"
            self.logger.error(msg)
            if self.strict:
                raise InvalidPentaryStringError(msg)
            return False
        
        invalid_chars = set(pentary_str) - self.VALID_SYMBOLS
        if invalid_chars:
            msg = f"Invalid characters in pentary string: {invalid_chars}"
            self.logger.error(msg)
            if self.strict:
                raise InvalidPentaryStringError(msg)
            return False
        
        return True
    
    def validate_pentary_digit(self, digit: int) -> bool:
        """
        Validate pentary digit value.
        
        Args:
            digit: Digit to validate
            
        Returns:
            True if valid
            
        Raises:
            InvalidPentaryDigitError: If digit is invalid and strict mode is on
        """
        if digit not in self.VALID_DIGITS:
            msg = f"Invalid pentary digit: {digit} (must be in {self.VALID_DIGITS})"
            self.logger.error(msg)
            if self.strict:
                raise InvalidPentaryDigitError(msg)
            return False
        
        return True
    
    def validate_pentary_array(self, digits: list) -> bool:
        """
        Validate pentary digit array.
        
        Args:
            digits: List of digits to validate
            
        Returns:
            True if valid
        """
        if not digits:
            msg = "Empty pentary digit array"
            self.logger.error(msg)
            if self.strict:
                raise InvalidPentaryDigitError(msg)
            return False
        
        for i, digit in enumerate(digits):
            if not isinstance(digit, int):
                msg = f"Non-integer digit at position {i}: {digit}"
                self.logger.error(msg)
                if self.strict:
                    raise InvalidPentaryDigitError(msg)
                return False
            
            if not self.validate_pentary_digit(digit):
                return False
        
        return True
    
    def sanitize_pentary_string(self, pentary_str: str) -> str:
        """
        Attempt to sanitize pentary string by removing invalid characters.
        
        Args:
            pentary_str: String to sanitize
            
        Returns:
            Sanitized string
        """
        sanitized = ''.join(c for c in pentary_str if c in self.VALID_SYMBOLS)
        
        if not sanitized:
            self.logger.warning(f"Sanitization resulted in empty string from: {pentary_str}")
            return "0"
        
        return sanitized
    
    def normalize_pentary_string(self, pentary_str: str) -> str:
        """
        Normalize pentary string (remove leading zeros, handle empty).
        
        Args:
            pentary_str: String to normalize
            
        Returns:
            Normalized string
        """
        if not pentary_str:
            return "0"
        
        # Remove leading zeros
        normalized = pentary_str.lstrip('0')
        
        if not normalized:
            return "0"
        
        return normalized


class SafePentaryOperations:
    """Wrapper for safe pentary operations with error handling"""
    
    def __init__(self, validator: Optional[PentaryValidator] = None):
        """
        Initialize safe operations wrapper.
        
        Args:
            validator: Optional validator instance
        """
        self.validator = validator or PentaryValidator(strict=False)
        self.logger = logging.getLogger(__name__)
    
    def safe_operation(self, 
                      operation: Callable,
                      *args,
                      default: Any = None,
                      validate_inputs: bool = True,
                      **kwargs) -> Union[Any, None]:
        """
        Execute operation with error handling.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            default: Default value to return on error
            validate_inputs: Whether to validate inputs
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result or default value on error
        """
        try:
            # Validate inputs if requested
            if validate_inputs:
                for arg in args:
                    if isinstance(arg, str):
                        self.validator.validate_pentary_string(arg)
                    elif isinstance(arg, list):
                        self.validator.validate_pentary_array(arg)
            
            # Execute operation
            result = operation(*args, **kwargs)
            return result
            
        except PentaryValidationError as e:
            self.logger.error(f"Validation error in {operation.__name__}: {e}")
            return default
            
        except PentaryOperationError as e:
            self.logger.error(f"Operation error in {operation.__name__}: {e}")
            return default
            
        except Exception as e:
            self.logger.error(f"Unexpected error in {operation.__name__}: {e}")
            return default
    
    def safe_add(self, a: str, b: str, converter) -> Optional[str]:
        """Safe pentary addition"""
        return self.safe_operation(converter.add_pentary, a, b, default="0")
    
    def safe_subtract(self, a: str, b: str, converter) -> Optional[str]:
        """Safe pentary subtraction"""
        return self.safe_operation(converter.subtract_pentary, a, b, default="0")
    
    def safe_multiply(self, a: str, b: str, arithmetic) -> Optional[str]:
        """Safe pentary multiplication"""
        return self.safe_operation(arithmetic.multiply_pentary, a, b, default="0")
    
    def safe_divide(self, a: str, b: str, arithmetic) -> Optional[tuple]:
        """Safe pentary division"""
        # Check for division by zero
        if b == "0":
            self.logger.error("Division by zero attempted")
            return ("0", "0")
        
        return self.safe_operation(arithmetic.divide_pentary, a, b, default=("0", "0"))


def validate_pentary(strict: bool = True):
    """
    Decorator for validating pentary function inputs.
    
    Args:
        strict: If True, raise exceptions on validation errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = PentaryValidator(strict=strict)
            
            # Validate string arguments
            for arg in args:
                if isinstance(arg, str) and any(c in PentaryValidator.VALID_SYMBOLS for c in arg):
                    validator.validate_pentary_string(arg)
            
            # Validate string keyword arguments
            for key, value in kwargs.items():
                if isinstance(value, str) and any(c in PentaryValidator.VALID_SYMBOLS for c in value):
                    validator.validate_pentary_string(value)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def main():
    """Demo and testing of validation and error handling"""
    print("=" * 70)
    print("Pentary Validation and Error Handling")
    print("=" * 70)
    print()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test validation
    print("Validation Tests:")
    print("-" * 70)
    
    validator = PentaryValidator(strict=False)
    
    test_strings = [
        ("⊕+0", True, "Valid pentary string"),
        ("⊕+0-⊖", True, "Valid pentary string with all symbols"),
        ("123", False, "Invalid: contains decimal digits"),
        ("⊕+X", False, "Invalid: contains letter"),
        ("", False, "Invalid: empty string"),
        ("000", True, "Valid: all zeros"),
    ]
    
    for string, expected, description in test_strings:
        result = validator.validate_pentary_string(string)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: '{string}' → {result}")
    
    print()
    print("=" * 70)
    print("Sanitization Tests:")
    print("-" * 70)
    
    test_strings = [
        ("⊕+0123", "⊕+0", "Remove decimal digits"),
        ("⊕+X0", "⊕+0", "Remove letters"),
        ("  ⊕+0  ", "⊕+0", "Remove spaces"),
        ("123", "0", "All invalid → 0"),
    ]
    
    for string, expected, description in test_strings:
        result = validator.sanitize_pentary_string(string)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: '{string}' → '{result}'")
    
    print()
    print("=" * 70)
    print("Safe Operations Tests:")
    print("-" * 70)
    
    from pentary_converter_optimized import PentaryConverterOptimized
    
    converter = PentaryConverterOptimized()
    safe_ops = SafePentaryOperations()
    
    # Test safe addition with valid inputs
    result = safe_ops.safe_add("⊕+", "+0", converter)
    print(f"Safe add (valid): ⊕+ + +0 = {result}")
    
    # Test safe addition with invalid inputs
    result = safe_ops.safe_add("⊕+X", "+0", converter)
    print(f"Safe add (invalid): ⊕+X + +0 = {result} (default)")
    
    # Test safe division by zero
    from pentary_arithmetic_extended import PentaryArithmeticExtended
    arithmetic = PentaryArithmeticExtended()
    
    result = safe_ops.safe_divide("⊕+", "0", arithmetic)
    print(f"Safe divide by zero: ⊕+ ÷ 0 = {result} (default)")
    
    print()
    print("=" * 70)
    print("Decorator Tests:")
    print("-" * 70)
    
    @validate_pentary(strict=False)
    def test_function(a: str, b: str) -> str:
        """Test function with validation decorator"""
        return f"{a} and {b}"
    
    # Test with valid inputs
    result = test_function("⊕+", "+0")
    print(f"Decorated function (valid): {result}")
    
    # Test with invalid inputs (will log warning but not raise)
    result = test_function("⊕+X", "+0")
    print(f"Decorated function (invalid): {result}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()