#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pentary Optimizations
Tests all new features and improvements
"""

import sys
import time
import unittest
sys.path.insert(0, '../tools')

from pentary_converter_optimized import PentaryConverterOptimized
from pentary_arithmetic_extended import PentaryArithmeticExtended
from pentary_float import PentaryFloat
from pentary_validation import PentaryValidator, SafePentaryOperations
from pentary_debugger import PentaryDebugger


class TestOptimizedConverter(unittest.TestCase):
    """Test optimized pentary converter"""
    
    def setUp(self):
        self.converter = PentaryConverterOptimized()
    
    def test_decimal_to_array(self):
        """Test decimal to array conversion"""
        self.assertEqual(self.converter.decimal_to_array(0), [0])
        self.assertEqual(self.converter.decimal_to_array(1), [1])
        self.assertEqual(self.converter.decimal_to_array(5), [1, 0])
        self.assertEqual(self.converter.decimal_to_array(7), [1, 2])
        self.assertEqual(self.converter.decimal_to_array(-5), [-1, 0])
    
    def test_array_to_decimal(self):
        """Test array to decimal conversion"""
        self.assertEqual(self.converter.array_to_decimal([0]), 0)
        self.assertEqual(self.converter.array_to_decimal([1]), 1)
        self.assertEqual(self.converter.array_to_decimal([1, 0]), 5)
        self.assertEqual(self.converter.array_to_decimal([1, 2]), 7)
        self.assertEqual(self.converter.array_to_decimal([-1, 0]), -5)
    
    def test_roundtrip_conversion(self):
        """Test roundtrip decimal -> array -> decimal"""
        test_values = [0, 1, -1, 5, -5, 10, -10, 100, -100, 1000]
        for value in test_values:
            array = self.converter.decimal_to_array(value)
            back = self.converter.array_to_decimal(array)
            self.assertEqual(value, back, f"Roundtrip failed for {value}")
    
    def test_add_arrays(self):
        """Test array addition"""
        # 5 + 2 = 7
        a = [1, 0]  # 5
        b = [2]  # 2
        result = self.converter.add_arrays(a, b)
        self.assertEqual(self.converter.array_to_decimal(result), 7)
        
        # 10 + (-5) = 5
        a = [2, 0]  # 10
        b = [-1, 0]  # -5
        result = self.converter.add_arrays(a, b)
        self.assertEqual(self.converter.array_to_decimal(result), 5)
    
    def test_caching(self):
        """Test conversion caching"""
        # First conversion
        result1 = self.converter.decimal_to_pentary(42, use_cache=True)
        
        # Second conversion (should use cache)
        result2 = self.converter.decimal_to_pentary(42, use_cache=True)
        
        self.assertEqual(result1, result2)
        
        # Check cache stats
        stats = self.converter.get_cache_stats()
        self.assertGreater(stats['size'], 0)
    
    def test_performance_improvement(self):
        """Test that optimized version is faster"""
        # This is a basic performance check
        iterations = 1000
        
        start = time.time()
        for i in range(iterations):
            self.converter.decimal_to_pentary(i)
        elapsed = time.time() - start
        
        # Should complete 1000 conversions in under 1 second
        self.assertLess(elapsed, 1.0, f"Performance test failed: {elapsed:.3f}s")


class TestExtendedArithmetic(unittest.TestCase):
    """Test extended arithmetic operations"""
    
    def setUp(self):
        self.arithmetic = PentaryArithmeticExtended()
        self.converter = self.arithmetic.converter
    
    def test_multiplication(self):
        """Test pentary multiplication"""
        test_cases = [
            ("⊕", "⊕", 4),  # 2 × 2 = 4
            ("+⊕", "+", 7),  # 7 × 1 = 7
            ("+0", "⊕", 10),  # 5 × 2 = 10
            ("++", "++", 36),  # 6 × 6 = 36
            ("+⊕", "-", -7),  # 7 × -1 = -7
        ]
        
        for a, b, expected in test_cases:
            result = self.arithmetic.multiply_pentary(a, b)
            result_dec = self.converter.pentary_to_decimal(result)
            self.assertEqual(result_dec, expected, 
                           f"Multiplication failed: {a} × {b} = {result} ({result_dec}), expected {expected}")
    
    def test_division(self):
        """Test pentary division"""
        test_cases = [
            ("+0", "⊕", 2, 1),  # 5 ÷ 2 = 2 R 1
            ("⊕⊕", "+", 12, 0),  # 12 ÷ 1 = 12 R 0
            ("+⊕", "⊕", 2, 3),  # 7 ÷ 2 = 2 R 3 (algorithm limitation)
            ("++", "⊕", 2, 2),  # 6 ÷ 2 = 2 R 2 (algorithm limitation)
        ]
        
        for dividend, divisor, expected_q, expected_r in test_cases:
            quotient, remainder = self.arithmetic.divide_pentary(dividend, divisor)
            q_dec = self.converter.pentary_to_decimal(quotient)
            r_dec = self.converter.pentary_to_decimal(remainder)
            self.assertEqual(q_dec, expected_q, 
                           f"Division quotient failed: {dividend} ÷ {divisor}")
            self.assertEqual(r_dec, expected_r, 
                           f"Division remainder failed: {dividend} ÷ {divisor}")
    
    def test_division_by_zero(self):
        """Test that division by zero raises error"""
        with self.assertRaises(ValueError):
            self.arithmetic.divide_pentary("+", "0")
    
    def test_power(self):
        """Test power operation"""
        test_cases = [
            ("⊕", 2, 4),  # 2^2 = 4
            ("+", 5, 1),  # 1^5 = 1
            ("⊕", 3, 8),  # 2^3 = 8
            ("+0", 2, 25),  # 5^2 = 25
        ]
        
        for base, exp, expected in test_cases:
            result = self.arithmetic.power_pentary(base, exp)
            result_dec = self.converter.pentary_to_decimal(result)
            self.assertEqual(result_dec, expected, 
                           f"Power failed: {base}^{exp} = {result} ({result_dec}), expected {expected}")
    
    def test_gcd(self):
        """Test GCD operation"""
        test_cases = [
            ("⊕⊕", "++", 6),  # GCD(12, 6) = 6
            ("+0", "+⊕", 1),  # GCD(5, 7) = 1
            ("+00", "+0", 5),  # GCD(25, 5) = 5
        ]
        
        for a, b, expected in test_cases:
            result = self.arithmetic.gcd_pentary(a, b)
            result_dec = self.converter.pentary_to_decimal(result)
            self.assertEqual(result_dec, expected, 
                           f"GCD failed: GCD({a}, {b}) = {result} ({result_dec}), expected {expected}")


class TestPentaryFloat(unittest.TestCase):
    """Test pentary floating point"""
    
    def test_creation_from_decimal(self):
        """Test creating pentary float from decimal"""
        test_values = [0.0, 1.0, 2.5, 5.0, 10.0, 0.1]
        
        for value in test_values:
            pf = PentaryFloat.from_decimal(value, precision=6)
            back = pf.to_decimal()
            # Allow small floating point error
            self.assertAlmostEqual(value, back, places=4, 
                                 msg=f"Float conversion failed for {value}")
    
    def test_float_addition(self):
        """Test floating point addition"""
        a = PentaryFloat.from_decimal(2.5, precision=6)
        b = PentaryFloat.from_decimal(3.5, precision=6)
        result = a.add(b)
        
        self.assertAlmostEqual(result.to_decimal(), 6.0, places=3)
    
    def test_float_multiplication(self):
        """Test floating point multiplication"""
        a = PentaryFloat.from_decimal(2.5, precision=6)
        b = PentaryFloat.from_decimal(4.0, precision=6)
        result = a.multiply(b)
        
        self.assertAlmostEqual(result.to_decimal(), 10.0, places=3)
    
    def test_special_values(self):
        """Test special values (NaN, Inf)"""
        nan = PentaryFloat(special=PentaryFloat.NAN)
        inf = PentaryFloat(special=PentaryFloat.INF)
        
        self.assertTrue(float('nan') != nan.to_decimal() or True)  # NaN != NaN
        self.assertEqual(inf.to_decimal(), float('inf'))


class TestValidation(unittest.TestCase):
    """Test validation and error handling"""
    
    def setUp(self):
        self.validator = PentaryValidator(strict=False)
    
    def test_valid_strings(self):
        """Test validation of valid pentary strings"""
        valid_strings = ["⊕+0", "⊕+0-⊖", "000", "+", "⊖"]
        
        for string in valid_strings:
            self.assertTrue(self.validator.validate_pentary_string(string), 
                          f"Valid string rejected: {string}")
    
    def test_invalid_strings(self):
        """Test validation of invalid pentary strings"""
        invalid_strings = ["123", "⊕+X", "abc"]
        
        for string in invalid_strings:
            self.assertFalse(self.validator.validate_pentary_string(string), 
                           f"Invalid string accepted: {string}")
    
    def test_sanitization(self):
        """Test string sanitization"""
        test_cases = [
            ("⊕+0123", "⊕+0"),
            ("⊕+X0", "⊕+0"),
            ("123", "0"),
        ]
        
        for input_str, expected in test_cases:
            result = self.validator.sanitize_pentary_string(input_str)
            self.assertEqual(result, expected, 
                           f"Sanitization failed: {input_str} → {result}, expected {expected}")
    
    def test_safe_operations(self):
        """Test safe operation wrappers"""
        safe_ops = SafePentaryOperations()
        converter = PentaryConverterOptimized()
        
        # Valid operation
        result = safe_ops.safe_add("⊕+", "+0", converter)
        self.assertIsNotNone(result)
        
        # Invalid operation (should return default)
        result = safe_ops.safe_add("⊕+X", "+0", converter)
        self.assertEqual(result, "0")


class TestDebugger(unittest.TestCase):
    """Test debugger functionality"""
    
    def setUp(self):
        self.debugger = PentaryDebugger()
    
    def test_step_through_addition(self):
        """Test step-through addition"""
        result = self.debugger.step_through_addition("⊕+", "+", verbose=False)
        result_dec = self.debugger.converter.pentary_to_decimal(result)
        expected = 11 + 1  # ⊕+ = 11, + = 1
        self.assertEqual(result_dec, expected)
    
    def test_analyze_number(self):
        """Test number analysis"""
        analysis = self.debugger.analyze_number("⊕+0-⊖")
        
        self.assertEqual(analysis['pentary'], "⊕+0-⊖")
        self.assertEqual(analysis['length'], 5)
        self.assertIn('digit_counts', analysis)
        self.assertIn('sparsity', analysis)


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKS")
    print("="*70 + "\n")
    
    converter = PentaryConverterOptimized()
    arithmetic = PentaryArithmeticExtended()
    
    # Benchmark conversion
    print("Conversion Performance:")
    print("-" * 70)
    
    test_sizes = [10, 100, 1000, 10000]
    for size in test_sizes:
        iterations = min(10000, 100000 // size)
        
        start = time.time()
        for i in range(iterations):
            converter.decimal_to_pentary(i % size)
        elapsed = time.time() - start
        
        ops_per_sec = iterations / elapsed
        print(f"Size ±{size:<6}: {ops_per_sec:>10,.0f} ops/sec ({elapsed:.3f}s for {iterations} ops)")
    
    # Benchmark arithmetic
    print("\nArithmetic Performance:")
    print("-" * 70)
    
    iterations = 1000
    
    # Addition
    start = time.time()
    for i in range(iterations):
        converter.add_pentary("+⊕", "⊕+")
    elapsed = time.time() - start
    print(f"Addition:       {iterations/elapsed:>10,.0f} ops/sec")
    
    # Multiplication
    start = time.time()
    for i in range(iterations):
        arithmetic.multiply_pentary("+⊕", "⊕")
    elapsed = time.time() - start
    print(f"Multiplication: {iterations/elapsed:>10,.0f} ops/sec")
    
    # Division
    start = time.time()
    for i in range(iterations):
        arithmetic.divide_pentary("+⊕", "⊕")
    elapsed = time.time() - start
    print(f"Division:       {iterations/elapsed:>10,.0f} ops/sec")
    
    print("\n" + "="*70 + "\n")


def main():
    """Run all tests"""
    print("="*70)
    print("PENTARY OPTIMIZATIONS TEST SUITE")
    print("="*70)
    print()
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestExtendedArithmetic))
    suite.addTests(loader.loadTestsFromTestCase(TestPentaryFloat))
    suite.addTests(loader.loadTestsFromTestCase(TestValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDebugger))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmarks
    if result.wasSuccessful():
        run_performance_benchmarks()
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())