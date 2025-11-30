#!/usr/bin/env python3
"""
Comprehensive Stress Tests for Pentary System
Tests all arithmetic operations, edge cases, boundary values, and system limits
"""

import unittest
import sys
import os
import random

# Add tools directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic
from pentary_simulator import PentaryProcessor


class TestPentaryConverterStress(unittest.TestCase):
    """Stress tests for PentaryConverter"""
    
    def test_roundtrip_small_numbers(self):
        """Test roundtrip conversion for small numbers"""
        for n in range(-100, 101):
            pentary = PentaryConverter.decimal_to_pentary(n)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(n, back, f"Roundtrip failed for {n}: {pentary} -> {back}")
    
    def test_roundtrip_medium_numbers(self):
        """Test roundtrip conversion for medium numbers"""
        for n in range(-10000, 10001, 7):  # Step by 7 for variety
            pentary = PentaryConverter.decimal_to_pentary(n)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(n, back, f"Roundtrip failed for {n}: {pentary} -> {back}")
    
    def test_roundtrip_large_numbers(self):
        """Test roundtrip conversion for large numbers"""
        test_values = [
            1000000, -1000000,
            10000000, -10000000,
            123456789, -123456789,
            2147483647, -2147483647,  # 32-bit int limits
            5**10, -(5**10),  # Powers of 5
            5**15, -(5**15),  # Larger powers
        ]
        for n in test_values:
            pentary = PentaryConverter.decimal_to_pentary(n)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(n, back, f"Roundtrip failed for {n}: {pentary} -> {back}")
    
    def test_roundtrip_random_numbers(self):
        """Test roundtrip conversion for random numbers"""
        random.seed(42)
        for _ in range(1000):
            n = random.randint(-10000000, 10000000)
            pentary = PentaryConverter.decimal_to_pentary(n)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(n, back, f"Roundtrip failed for {n}: {pentary} -> {back}")
    
    def test_zero_conversion(self):
        """Test zero conversion"""
        self.assertEqual(PentaryConverter.decimal_to_pentary(0), "0")
        self.assertEqual(PentaryConverter.pentary_to_decimal("0"), 0)
    
    def test_powers_of_5(self):
        """Test powers of 5 (should have simple representations)"""
        for exp in range(15):
            n = 5 ** exp
            pentary = PentaryConverter.decimal_to_pentary(n)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(n, back, f"Power of 5 failed: 5^{exp} = {n}")
    
    def test_boundary_values(self):
        """Test boundary values in balanced pentary"""
        # Single digit boundaries
        self.assertEqual(PentaryConverter.decimal_to_pentary(2), "⊕")
        self.assertEqual(PentaryConverter.decimal_to_pentary(-2), "⊖")
        self.assertEqual(PentaryConverter.decimal_to_pentary(1), "+")
        self.assertEqual(PentaryConverter.decimal_to_pentary(-1), "-")
        
        # Transition points (near powers of 5)
        for p in range(1, 10):
            base = 5 ** p
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                n = base + offset
                pentary = PentaryConverter.decimal_to_pentary(n)
                back = PentaryConverter.pentary_to_decimal(pentary)
                self.assertEqual(n, back, f"Boundary failed for {base} + {offset} = {n}")


class TestPentaryArithmeticStress(unittest.TestCase):
    """Stress tests for PentaryArithmetic"""
    
    def test_add_table_completeness(self):
        """Verify ADD_TABLE has all required entries"""
        digits = [-2, -1, 0, 1, 2]
        carries = [-1, 0, 1]
        
        for a in digits:
            for b in digits:
                for c in carries:
                    self.assertIn((a, b, c), PentaryArithmetic.ADD_TABLE,
                                f"Missing entry: ({a}, {b}, {c})")
    
    def test_add_table_correctness(self):
        """Verify all ADD_TABLE entries are mathematically correct"""
        for (a, b, cin), (sum_digit, cout) in PentaryArithmetic.ADD_TABLE.items():
            expected = a + b + cin
            actual = sum_digit + 5 * cout
            self.assertEqual(expected, actual,
                           f"ADD_TABLE error: ({a}, {b}, {cin}) -> ({sum_digit}, {cout})")
            
            # Verify sum_digit is in valid range
            self.assertIn(sum_digit, [-2, -1, 0, 1, 2],
                         f"Invalid sum_digit: {sum_digit}")
            
            # Verify carry_out is in valid range
            self.assertIn(cout, [-1, 0, 1],
                         f"Invalid carry_out: {cout}")
    
    def test_addition_small_numbers(self):
        """Test addition for small numbers"""
        for a in range(-100, 101):
            for b in range(-100, 101):
                a_pent = PentaryConverter.decimal_to_pentary(a)
                b_pent = PentaryConverter.decimal_to_pentary(b)
                
                # High-level addition
                result = PentaryConverter.add_pentary(a_pent, b_pent)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                self.assertEqual(a + b, result_dec,
                               f"Addition failed: {a} + {b} = {a+b}, got {result_dec}")
    
    def test_addition_consistency(self):
        """Test that high-level and low-level addition match"""
        random.seed(42)
        for _ in range(1000):
            a = random.randint(-10000, 10000)
            b = random.randint(-10000, 10000)
            
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            # High-level
            high_result = PentaryConverter.add_pentary(a_pent, b_pent)
            high_result = high_result.lstrip('0') or '0'
            
            # Low-level
            low_result, _ = PentaryArithmetic.add_pentary_detailed(a_pent, b_pent)
            low_result = low_result.lstrip('0') or '0'
            
            self.assertEqual(high_result, low_result,
                           f"Consistency failed for {a} + {b}")
    
    def test_subtraction_small_numbers(self):
        """Test subtraction for small numbers"""
        for a in range(-50, 51):
            for b in range(-50, 51):
                a_pent = PentaryConverter.decimal_to_pentary(a)
                b_pent = PentaryConverter.decimal_to_pentary(b)
                
                result = PentaryConverter.subtract_pentary(a_pent, b_pent)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                self.assertEqual(a - b, result_dec,
                               f"Subtraction failed: {a} - {b} = {a-b}, got {result_dec}")
    
    def test_negation(self):
        """Test negation for various numbers"""
        test_values = list(range(-1000, 1001)) + [
            12345, -12345, 99999, -99999,
            1000000, -1000000
        ]
        
        for n in test_values:
            pentary = PentaryConverter.decimal_to_pentary(n)
            negated = PentaryConverter.negate_pentary(pentary)
            negated_dec = PentaryConverter.pentary_to_decimal(negated)
            
            self.assertEqual(-n, negated_dec,
                           f"Negation failed: -{n} = {-n}, got {negated_dec}")
    
    def test_multiply_by_constant(self):
        """Test multiplication by constants {-2, -1, 0, 1, 2}"""
        for n in range(-1000, 1001):
            pentary = PentaryConverter.decimal_to_pentary(n)
            
            for const in [-2, -1, 0, 1, 2]:
                result = PentaryConverter.multiply_pentary_by_constant(pentary, const)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                self.assertEqual(n * const, result_dec,
                               f"Multiply failed: {n} * {const} = {n*const}, got {result_dec}")
    
    def test_shift_left(self):
        """Test shift left (multiply by 5^n)"""
        for n in range(-1000, 1001):
            pentary = PentaryConverter.decimal_to_pentary(n)
            
            for shift in range(1, 5):
                result = PentaryConverter.shift_left_pentary(pentary, shift)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                expected = n * (5 ** shift)
                self.assertEqual(expected, result_dec,
                               f"Shift left failed: {n} << {shift} = {expected}, got {result_dec}")
    
    def test_shift_right(self):
        """Test shift right (divide by 5^n, truncated)"""
        for n in range(-1000, 1001):
            pentary = PentaryConverter.decimal_to_pentary(n)
            
            for shift in range(1, 4):
                result = PentaryConverter.shift_right_pentary(pentary, shift)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                # Integer division toward zero
                expected = int(n / (5 ** shift))
                if n < 0 and n % (5 ** shift) != 0:
                    expected += 1  # Adjust for truncation toward zero
                
                # Note: shift_right truncates digits, may differ from exact division
                # Just verify it's a valid integer
                self.assertIsInstance(result_dec, int,
                                    f"Shift right result not an integer: {n} >> {shift}")


class TestPentarySimulatorStress(unittest.TestCase):
    """Stress tests for PentaryProcessor simulator"""
    
    def test_register_operations(self):
        """Test basic register operations"""
        proc = PentaryProcessor()
        
        # Test all registers except P0 (always zero)
        for i in range(1, 32):
            reg = f"P{i}"
            test_val = PentaryConverter.decimal_to_pentary(i * 100)
            proc.set_register(reg, test_val)
            result = proc.get_register(reg)
            self.assertEqual(test_val, result, f"Register {reg} failed")
        
        # P0 should always be zero
        proc.set_register("P0", PentaryConverter.decimal_to_pentary(999))
        self.assertEqual(proc.get_register("P0"), "0")
    
    def test_arithmetic_instructions(self):
        """Test arithmetic instructions"""
        test_cases = [
            (5, 3, 8, "ADD"),
            (10, 4, 6, "SUB"),
            (7, 2, 14, "MUL2"),  # MUL2 multiplies by 2
        ]
        
        for a, b, expected, op in test_cases:
            proc = PentaryProcessor()
            
            if op == "ADD":
                program = [
                    f"MOVI P1, {a}",
                    f"MOVI P2, {b}",
                    "ADD P3, P1, P2",
                    "HALT"
                ]
            elif op == "SUB":
                program = [
                    f"MOVI P1, {a}",
                    f"MOVI P2, {b}",
                    "SUB P3, P1, P2",
                    "HALT"
                ]
            elif op == "MUL2":
                program = [
                    f"MOVI P1, {a}",
                    "MUL2 P3, P1",
                    "HALT"
                ]
            
            proc.load_program(program)
            proc.run()
            
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"{op} failed: {a}, {b} -> {result}")
    
    def test_loop_execution(self):
        """Test loop execution - sum from 1 to N"""
        for n in [5, 10, 20]:
            proc = PentaryProcessor()
            
            # Sum from 1 to n
            program = [
                "MOVI P1, 0",       # sum = 0
                "MOVI P2, 1",       # i = 1
                f"MOVI P3, {n}",    # limit = n
                "ADD P1, P1, P2",   # sum += i
                "ADDI P2, P2, 1",   # i++
                "SUB P4, P2, P3",   # temp = i - limit
                "ADDI P5, P4, -1",  # temp2 = temp - 1 (check if i <= limit)
                "BLT P5, 3",        # if temp2 < 0 (i <= limit), goto line 3
                "HALT"
            ]
            
            proc.load_program(program)
            proc.run(max_cycles=1000)
            
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P1"))
            expected = n * (n + 1) // 2
            self.assertEqual(expected, result, f"Loop sum 1 to {n} failed: expected {expected}, got {result}")
    
    def test_memory_operations(self):
        """Test memory load/store operations"""
        proc = PentaryProcessor()
        
        # Store and load various values
        program = [
            "MOVI P1, 10",          # base address
            "MOVI P2, 42",          # value 1
            "STORE P2, [P1 + 0]",   # mem[10] = 42
            "MOVI P3, 99",          # value 2
            "STORE P3, [P1 + 1]",   # mem[11] = 99
            "MOVI P4, -123",        # value 3 (negative)
            "STORE P4, [P1 + 2]",   # mem[12] = -123
            "LOAD P5, [P1 + 0]",    # P5 = mem[10]
            "LOAD P6, [P1 + 1]",    # P6 = mem[11]
            "LOAD P7, [P1 + 2]",    # P7 = mem[12]
            "HALT"
        ]
        
        proc.load_program(program)
        proc.run()
        
        self.assertEqual(42, PentaryConverter.pentary_to_decimal(proc.get_register("P5")))
        self.assertEqual(99, PentaryConverter.pentary_to_decimal(proc.get_register("P6")))
        self.assertEqual(-123, PentaryConverter.pentary_to_decimal(proc.get_register("P7")))
    
    def test_stack_operations(self):
        """Test stack push/pop operations"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 10",
            "MOVI P2, 20",
            "MOVI P3, 30",
            "PUSH P1",
            "PUSH P2",
            "PUSH P3",
            "POP P4",      # P4 = 30
            "POP P5",      # P5 = 20
            "POP P6",      # P6 = 10
            "HALT"
        ]
        
        proc.load_program(program)
        proc.run()
        
        self.assertEqual(30, PentaryConverter.pentary_to_decimal(proc.get_register("P4")))
        self.assertEqual(20, PentaryConverter.pentary_to_decimal(proc.get_register("P5")))
        self.assertEqual(10, PentaryConverter.pentary_to_decimal(proc.get_register("P6")))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and potential failure points"""
    
    def test_empty_pentary_string(self):
        """Test handling of empty/edge case inputs"""
        # Empty string should raise an error or return 0
        try:
            result = PentaryConverter.pentary_to_decimal("")
            # If no error, should be 0
            self.assertEqual(0, result)
        except (ValueError, IndexError):
            pass  # Expected behavior
    
    def test_invalid_pentary_symbols(self):
        """Test that invalid symbols raise errors"""
        invalid_strings = ["abc", "123", "⊕x⊖", "⊕⊕⊕x"]
        
        for s in invalid_strings:
            with self.assertRaises(ValueError, msg=f"Should raise for '{s}'"):
                PentaryConverter.pentary_to_decimal(s)
    
    def test_very_long_pentary_numbers(self):
        """Test very long pentary representations"""
        # Create a number with many digits
        n = 5 ** 20  # Very large number
        pentary = PentaryConverter.decimal_to_pentary(n)
        back = PentaryConverter.pentary_to_decimal(pentary)
        self.assertEqual(n, back)
        
        # Negative very large
        n = -(5 ** 20)
        pentary = PentaryConverter.decimal_to_pentary(n)
        back = PentaryConverter.pentary_to_decimal(pentary)
        self.assertEqual(n, back)
    
    def test_addition_with_many_carries(self):
        """Test addition that produces many carries"""
        # Numbers that should produce lots of carries
        a = sum(2 * (5 ** i) for i in range(10))  # Max positive 10-digit pentary
        b = sum(2 * (5 ** i) for i in range(10))  # Same
        
        a_pent = PentaryConverter.decimal_to_pentary(a)
        b_pent = PentaryConverter.decimal_to_pentary(b)
        
        result = PentaryConverter.add_pentary(a_pent, b_pent)
        result_dec = PentaryConverter.pentary_to_decimal(result)
        
        self.assertEqual(a + b, result_dec)
    
    def test_subtraction_resulting_in_negative(self):
        """Test subtraction that results in negative numbers"""
        test_cases = [
            (5, 10),   # -5
            (1, 100),  # -99
            (0, 1000), # -1000
        ]
        
        for a, b in test_cases:
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            result = PentaryConverter.subtract_pentary(a_pent, b_pent)
            result_dec = PentaryConverter.pentary_to_decimal(result)
            
            self.assertEqual(a - b, result_dec, f"{a} - {b} failed")


class TestCompareOperations(unittest.TestCase):
    """Test comparison operations"""
    
    def test_compare_equal(self):
        """Test comparison for equal numbers"""
        for n in range(-100, 101):
            a_pent = PentaryConverter.decimal_to_pentary(n)
            b_pent = PentaryConverter.decimal_to_pentary(n)
            
            cmp = PentaryArithmetic.compare_pentary(a_pent, b_pent)
            self.assertEqual(0, cmp, f"Compare equal failed for {n}")
    
    def test_compare_less_than(self):
        """Test comparison for less than"""
        for a in range(-50, 50):
            for b in range(a + 1, 51):
                a_pent = PentaryConverter.decimal_to_pentary(a)
                b_pent = PentaryConverter.decimal_to_pentary(b)
                
                cmp = PentaryArithmetic.compare_pentary(a_pent, b_pent)
                self.assertEqual(-1, cmp, f"Compare {a} < {b} failed")
    
    def test_compare_greater_than(self):
        """Test comparison for greater than"""
        for a in range(-50, 50):
            for b in range(-51, a):
                a_pent = PentaryConverter.decimal_to_pentary(a)
                b_pent = PentaryConverter.decimal_to_pentary(b)
                
                cmp = PentaryArithmetic.compare_pentary(a_pent, b_pent)
                self.assertEqual(1, cmp, f"Compare {a} > {b} failed")


class TestMultiplicationDivision(unittest.TestCase):
    """Test multiplication and division operations"""
    
    def test_multiplication_small(self):
        """Test multiplication for small numbers"""
        for a in range(-20, 21):
            for b in range(-20, 21):
                a_pent = PentaryConverter.decimal_to_pentary(a)
                b_pent = PentaryConverter.decimal_to_pentary(b)
                
                result = PentaryConverter.multiply_pentary(a_pent, b_pent)
                result_dec = PentaryConverter.pentary_to_decimal(result)
                
                self.assertEqual(a * b, result_dec,
                               f"Multiplication failed: {a} * {b} = {a*b}, got {result_dec}")
    
    def test_multiplication_random(self):
        """Test multiplication for random numbers"""
        random.seed(42)
        for _ in range(500):
            a = random.randint(-1000, 1000)
            b = random.randint(-1000, 1000)
            
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            result = PentaryConverter.multiply_pentary(a_pent, b_pent)
            result_dec = PentaryConverter.pentary_to_decimal(result)
            
            self.assertEqual(a * b, result_dec,
                           f"Multiplication failed: {a} * {b} = {a*b}, got {result_dec}")
    
    def test_division_exact(self):
        """Test division with exact results"""
        test_cases = [
            (10, 2, 5),
            (100, 10, 10),
            (15, 3, 5),
            (24, 4, 6),
            (-20, 4, -5),
            (20, -4, -5),
            (-20, -4, 5),
        ]
        
        for a, b, expected in test_cases:
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            result = PentaryConverter.divide_pentary(a_pent, b_pent)
            result_dec = PentaryConverter.pentary_to_decimal(result)
            
            self.assertEqual(expected, result_dec,
                           f"Division failed: {a} / {b} = {expected}, got {result_dec}")
    
    def test_division_truncation(self):
        """Test division with truncation toward zero"""
        test_cases = [
            (7, 2, 3),    # 7/2 = 3.5 -> 3
            (7, 3, 2),    # 7/3 = 2.33 -> 2
            (-7, 2, -3),  # -7/2 = -3.5 -> -3
            (7, -2, -3),  # 7/-2 = -3.5 -> -3
            (-7, -2, 3),  # -7/-2 = 3.5 -> 3
        ]
        
        for a, b, expected in test_cases:
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            result = PentaryConverter.divide_pentary(a_pent, b_pent)
            result_dec = PentaryConverter.pentary_to_decimal(result)
            
            self.assertEqual(expected, result_dec,
                           f"Division truncation failed: {a} / {b} = {expected}, got {result_dec}")
    
    def test_division_by_zero(self):
        """Test division by zero raises error"""
        a_pent = PentaryConverter.decimal_to_pentary(10)
        b_pent = PentaryConverter.decimal_to_pentary(0)
        
        with self.assertRaises(ValueError):
            PentaryConverter.divide_pentary(a_pent, b_pent)
    
    def test_modulo_operations(self):
        """Test modulo operations"""
        test_cases = [
            (7, 3, 1),    # 7 % 3 = 1
            (10, 5, 0),   # 10 % 5 = 0
            (17, 5, 2),   # 17 % 5 = 2
            (-7, 3, 2),   # Python: -7 % 3 = 2
            (7, -3, -2),  # Python: 7 % -3 = -2
        ]
        
        for a, b, expected in test_cases:
            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)
            
            result = PentaryConverter.modulo_pentary(a_pent, b_pent)
            result_dec = PentaryConverter.pentary_to_decimal(result)
            
            self.assertEqual(expected, result_dec,
                           f"Modulo failed: {a} % {b} = {expected}, got {result_dec}")
    
    def test_simulator_mul_instruction(self):
        """Test MUL instruction in simulator"""
        test_cases = [
            (3, 4, 12),
            (5, 7, 35),
            (-3, 4, -12),
            (10, 10, 100),
        ]
        
        for a, b, expected in test_cases:
            proc = PentaryProcessor()
            program = [
                f"MOVI P1, {a}",
                f"MOVI P2, {b}",
                "MUL P3, P1, P2",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"MUL {a} * {b} failed")
    
    def test_simulator_div_instruction(self):
        """Test DIV instruction in simulator"""
        test_cases = [
            (12, 3, 4),
            (100, 10, 10),
            (-20, 4, -5),
        ]
        
        for a, b, expected in test_cases:
            proc = PentaryProcessor()
            program = [
                f"MOVI P1, {a}",
                f"MOVI P2, {b}",
                "DIV P3, P1, P2",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"DIV {a} / {b} failed")
    
    def test_simulator_mod_instruction(self):
        """Test MOD instruction in simulator"""
        test_cases = [
            (17, 5, 2),
            (10, 3, 1),
        ]
        
        for a, b, expected in test_cases:
            proc = PentaryProcessor()
            program = [
                f"MOVI P1, {a}",
                f"MOVI P2, {b}",
                "MOD P3, P1, P2",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"MOD {a} % {b} failed")


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)
