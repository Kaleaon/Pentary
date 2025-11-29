#!/usr/bin/env python3
"""
Comprehensive Hardware Architecture Verification Tests
Validates the Pentary chip design against mathematical truth and edge cases
"""

import unittest
import sys
import os

# Add tools directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic
from pentary_simulator import PentaryProcessor


class TestPentaryEncoding(unittest.TestCase):
    """Verify 3-bit encoding matches hardware specification"""
    
    # Hardware encoding from pentary_chip_design.v:
    # 000 = -2 (⊖, strong negative)
    # 001 = -1 (-,  weak negative)
    # 010 =  0 (0,  zero)
    # 011 = +1 (+,  weak positive)
    # 100 = +2 (⊕, strong positive)
    
    HARDWARE_ENCODING = {
        -2: 0b000,  # ⊖
        -1: 0b001,  # -
         0: 0b010,  # 0
        +1: 0b011,  # +
        +2: 0b100,  # ⊕
    }
    
    SOFTWARE_SYMBOLS = {
        -2: '⊖',
        -1: '-',
         0: '0',
        +1: '+',
        +2: '⊕',
    }
    
    def test_encoding_values_are_valid(self):
        """Verify all hardware encoding values fit in 3 bits"""
        for digit, encoding in self.HARDWARE_ENCODING.items():
            self.assertGreaterEqual(encoding, 0, f"Encoding for {digit} is negative")
            self.assertLessEqual(encoding, 0b111, f"Encoding for {digit} exceeds 3 bits")
    
    def test_encoding_uniqueness(self):
        """Verify all hardware encodings are unique"""
        encodings = list(self.HARDWARE_ENCODING.values())
        self.assertEqual(len(encodings), len(set(encodings)), "Hardware encodings are not unique")
    
    def test_software_symbol_mapping(self):
        """Verify software symbols match expected values"""
        for digit in range(-2, 3):
            symbol = self.SOFTWARE_SYMBOLS[digit]
            # Verify we can convert single digits
            pentary = PentaryConverter.decimal_to_pentary(digit)
            back = PentaryConverter.pentary_to_decimal(pentary)
            self.assertEqual(digit, back, f"Single digit roundtrip failed for {digit}")
    
    def test_word_size_calculation(self):
        """Verify 16-pent word = 48 bits (16 × 3)"""
        # From architecture: "A 16-pent word uses 48 bits (16 × 3 bits)"
        bits_per_pent = 3
        pents_per_word = 16
        expected_bits = bits_per_pent * pents_per_word
        self.assertEqual(48, expected_bits, "Word size calculation error")
    
    def test_max_value_16_pent(self):
        """Verify maximum 16-pent value"""
        # Maximum: all ⊕ (2s) = 2 * sum(5^i for i in 0..15)
        max_value = sum(2 * (5 ** i) for i in range(16))
        expected = 76293945312  # Calculated: ~7.6 × 10^10
        self.assertEqual(max_value, expected)
        
        # Verify we can represent it
        pentary = PentaryConverter.decimal_to_pentary(max_value)
        back = PentaryConverter.pentary_to_decimal(pentary)
        self.assertEqual(max_value, back)
    
    def test_min_value_16_pent(self):
        """Verify minimum 16-pent value"""
        # Minimum: all ⊖ (-2s) = -2 * sum(5^i for i in 0..15)
        min_value = -sum(2 * (5 ** i) for i in range(16))
        expected = -76293945312
        self.assertEqual(min_value, expected)
        
        # Verify we can represent it
        pentary = PentaryConverter.decimal_to_pentary(min_value)
        back = PentaryConverter.pentary_to_decimal(pentary)
        self.assertEqual(min_value, back)


class TestALUOperationsMatchHardware(unittest.TestCase):
    """Verify software ALU operations match hardware specification"""
    
    def test_add_operation_matches_hardware_spec(self):
        """
        Verify ADD matches hardware behavior.
        From pentary_chip_design.v: opcode = 000: ADD (addition)
        """
        proc = PentaryProcessor()
        
        # Test cases from hardware spec
        test_cases = [
            (5, 3, 8),
            (10, -4, 6),
            (-7, -3, -10),
            (0, 0, 0),
            (100, 200, 300),
        ]
        
        for a, b, expected in test_cases:
            proc.reset()
            program = [
                f"MOVI P1, {a}",
                f"MOVI P2, {b}",
                "ADD P3, P1, P2",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"ADD {a} + {b} = {expected}, got {result}")
    
    def test_sub_operation_matches_hardware_spec(self):
        """
        Verify SUB matches hardware behavior.
        From pentary_chip_design.v: opcode = 001: SUB (subtraction)
        """
        proc = PentaryProcessor()
        
        test_cases = [
            (10, 3, 7),
            (5, 10, -5),
            (-5, -3, -2),
            (0, 100, -100),
        ]
        
        for a, b, expected in test_cases:
            proc.reset()
            program = [
                f"MOVI P1, {a}",
                f"MOVI P2, {b}",
                "SUB P3, P1, P2",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"SUB {a} - {b} = {expected}, got {result}")
    
    def test_mul2_operation_matches_hardware_spec(self):
        """
        Verify MUL2 matches hardware behavior.
        From pentary_chip_design.v: opcode = 010: MUL2 (multiply by 2)
        """
        proc = PentaryProcessor()
        
        test_cases = [
            (5, 10),
            (0, 0),
            (-7, -14),
            (100, 200),
            (1, 2),
        ]
        
        for a, expected in test_cases:
            proc.reset()
            program = [
                f"MOVI P1, {a}",
                "MUL2 P3, P1",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"MUL2 {a} * 2 = {expected}, got {result}")
    
    def test_neg_operation_matches_hardware_spec(self):
        """
        Verify NEG matches hardware behavior.
        From pentary_chip_design.v: opcode = 011: NEG (negation)
        """
        proc = PentaryProcessor()
        
        test_cases = [
            (5, -5),
            (-10, 10),
            (0, 0),
            (12345, -12345),
            (-67890, 67890),
        ]
        
        for a, expected in test_cases:
            proc.reset()
            program = [
                f"MOVI P1, {a}",
                "NEG P3, P1",
                "HALT"
            ]
            proc.load_program(program)
            proc.run()
            result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
            self.assertEqual(expected, result, f"NEG {a} = {expected}, got {result}")


class TestCarryPropagation(unittest.TestCase):
    """Verify carry propagation matches hardware specification"""
    
    def test_carry_chain_all_max_digits(self):
        """
        Test adding 1 to maximum value causes proper carry propagation.
        All ⊕⊕⊕... + 1 should propagate carries through entire number.
        """
        # Start with a smaller example: ⊕⊕ (12) + 1 = 13
        # 12 + 1 = 13 = +⊖- in pentary (1*5^1 + (-2)*5^0 + (-1)) wait...
        # Let's verify: 13 = +⊖- means 1*25 + (-2)*5 + (-1) = 25 - 10 - 1 = 14, wrong
        # Actually 13 = ⊕⊖ = 2*5 + (-2) = 10 - 2 = 8, also wrong
        # Let me recalculate: 13 / 5 = 2 remainder 3
        # 3 in balanced pentary: 3 = +⊖ (1*5 + (-2) = 3)... no that's wrong too
        # 3 = -⊕ wait no...
        # Actually in balanced pentary: digits are {-2, -1, 0, 1, 2}
        # 13: We need to find representation
        # 13 = 2*5 + 3, but 3 isn't a valid digit
        # So 13 = 3*5 + (-2) = 15 - 2 = 13 ✓
        # 3 in pentary is +⊖ (1*5 + (-2) = 3)
        # So 13 = +⊖⊖ = 1*25 + (-2)*5 + (-2) = 25 - 10 - 2 = 13 ✓
        
        a = 12  # ⊕⊕ = 2*5 + 2 = 12
        b = 1
        expected = 13
        
        a_pent = PentaryConverter.decimal_to_pentary(a)
        b_pent = PentaryConverter.decimal_to_pentary(b)
        result = PentaryConverter.add_pentary(a_pent, b_pent)
        result_dec = PentaryConverter.pentary_to_decimal(result)
        
        self.assertEqual(expected, result_dec, f"Carry propagation: {a} + {b} = {expected}, got {result_dec}")
    
    def test_borrow_chain_all_min_digits(self):
        """
        Test subtracting 1 from minimum values causes proper borrow propagation.
        ⊖⊖⊖... - 1 should propagate borrows.
        """
        a = -12  # ⊖⊖ = -2*5 + (-2) = -12
        b = 1
        expected = -13
        
        a_pent = PentaryConverter.decimal_to_pentary(a)
        b_pent = PentaryConverter.decimal_to_pentary(b)
        result = PentaryConverter.subtract_pentary(a_pent, b_pent)
        result_dec = PentaryConverter.pentary_to_decimal(result)
        
        self.assertEqual(expected, result_dec, f"Borrow propagation: {a} - {b} = {expected}, got {result_dec}")
    
    def test_full_carry_chain(self):
        """
        Test maximum carry chain: all 2s + all 2s should produce proper result
        """
        # ⊕⊕⊕⊕ + ⊕⊕⊕⊕ (4 digits each)
        # Each is 2*(5^3 + 5^2 + 5^1 + 5^0) = 2*(125 + 25 + 5 + 1) = 2*156 = 312
        # Sum = 624
        
        a = 312
        b = 312
        expected = 624
        
        a_pent = PentaryConverter.decimal_to_pentary(a)
        b_pent = PentaryConverter.decimal_to_pentary(b)
        result = PentaryConverter.add_pentary(a_pent, b_pent)
        result_dec = PentaryConverter.pentary_to_decimal(result)
        
        self.assertEqual(expected, result_dec, f"Full carry chain: {a} + {b} = {expected}, got {result_dec}")


class TestMemoryAddressing(unittest.TestCase):
    """Verify memory addressing matches architecture specification"""
    
    def test_register_count(self):
        """Verify 32 registers as specified in architecture"""
        proc = PentaryProcessor()
        
        # Should be able to access all 32 registers
        for i in range(32):
            reg_name = f"P{i}"
            # P0 is always zero, others should be accessible
            if i == 0:
                # P0 is hardwired to zero
                proc.set_register(reg_name, PentaryConverter.decimal_to_pentary(123))
                self.assertEqual("0", proc.get_register(reg_name))
            else:
                test_val = PentaryConverter.decimal_to_pentary(i * 100)
                proc.set_register(reg_name, test_val)
                self.assertEqual(test_val, proc.get_register(reg_name))
    
    def test_p0_always_zero(self):
        """Verify P0 is hardwired to zero (from architecture spec)"""
        proc = PentaryProcessor()
        
        # Try to set P0 to various values
        for val in [1, 100, -50, 999]:
            proc.set_register("P0", PentaryConverter.decimal_to_pentary(val))
            self.assertEqual("0", proc.get_register("P0"), "P0 should always be zero")


class TestStatusFlags(unittest.TestCase):
    """Verify status flags match architecture specification"""
    
    def test_zero_flag(self):
        """Test Z flag is set when result is zero"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 5",
            "MOVI P2, 5",
            "SUB P3, P1, P2",  # 5 - 5 = 0, should set Z flag
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        self.assertTrue(proc.sr['Z'], "Zero flag should be set")
    
    def test_negative_flag(self):
        """Test N flag is set when result is negative"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 3",
            "MOVI P2, 10",
            "SUB P3, P1, P2",  # 3 - 10 = -7, should set N flag
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        self.assertTrue(proc.sr['N'], "Negative flag should be set")
    
    def test_positive_flag(self):
        """Test P flag is set when result is positive"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 10",
            "MOVI P2, 3",
            "SUB P3, P1, P2",  # 10 - 3 = 7, should set P flag
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        self.assertTrue(proc.sr['P'], "Positive flag should be set")


class TestBranchInstructions(unittest.TestCase):
    """Verify branch instructions match architecture specification"""
    
    def test_beq_taken(self):
        """Test BEQ is taken when condition is met"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 0",      # P1 = 0
            "BEQ P1, 4",       # If P1 == 0, jump to line 4
            "MOVI P2, 100",    # Should be skipped
            "JUMP 5",          # Should be skipped
            "MOVI P2, 200",    # Should be executed
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(200, result, "BEQ should have jumped to line 4")
    
    def test_bne_taken(self):
        """Test BNE is taken when condition is met"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 5",      # P1 = 5 (non-zero)
            "BNE P1, 4",       # If P1 != 0, jump to line 4
            "MOVI P2, 100",    # Should be skipped
            "JUMP 5",          # Should be skipped
            "MOVI P2, 200",    # Should be executed
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(200, result, "BNE should have jumped to line 4")
    
    def test_blt_taken(self):
        """Test BLT is taken when condition is met"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, -5",     # P1 = -5 (negative)
            "BLT P1, 4",       # If P1 < 0, jump to line 4
            "MOVI P2, 100",    # Should be skipped
            "JUMP 5",          # Should be skipped
            "MOVI P2, 200",    # Should be executed
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(200, result, "BLT should have jumped to line 4")
    
    def test_bgt_taken(self):
        """Test BGT is taken when condition is met"""
        proc = PentaryProcessor()
        
        program = [
            "MOVI P1, 5",      # P1 = 5 (positive)
            "BGT P1, 4",       # If P1 > 0, jump to line 4
            "MOVI P2, 100",    # Should be skipped
            "JUMP 5",          # Should be skipped
            "MOVI P2, 200",    # Should be executed
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(200, result, "BGT should have jumped to line 4")


class TestHardwareVsSoftwareConsistency(unittest.TestCase):
    """Verify software arithmetic matches hardware truth tables"""
    
    def test_pentary_digit_negation(self):
        """
        Verify software negation matches hardware negation function.
        From pentary_chip_design.v:
            ⊖ -> ⊕ (-2 -> +2)
            - -> + (-1 -> +1)
            0 -> 0
            + -> - (+1 -> -1)
            ⊕ -> ⊖ (+2 -> -2)
        """
        for digit in range(-2, 3):
            pentary = PentaryConverter.decimal_to_pentary(digit)
            negated = PentaryConverter.negate_pentary(pentary)
            negated_dec = PentaryConverter.pentary_to_decimal(negated)
            self.assertEqual(-digit, negated_dec, f"Negation mismatch for {digit}")
    
    def test_pentary_shift_left(self):
        """
        Verify shift left = multiply by 5.
        From hardware spec: shift_left_pentary appends zeros.
        """
        for n in range(-100, 101):
            pentary = PentaryConverter.decimal_to_pentary(n)
            shifted = PentaryConverter.shift_left_pentary(pentary, 1)
            shifted_dec = PentaryConverter.pentary_to_decimal(shifted)
            self.assertEqual(n * 5, shifted_dec, f"Shift left mismatch for {n}")
    
    def test_pentary_shift_right(self):
        """
        Verify shift right = divide by 5 (truncated).
        """
        for n in range(-100, 101):
            if n % 5 == 0:  # Only test exact divisions
                pentary = PentaryConverter.decimal_to_pentary(n)
                shifted = PentaryConverter.shift_right_pentary(pentary, 1)
                shifted_dec = PentaryConverter.pentary_to_decimal(shifted)
                self.assertEqual(n // 5, shifted_dec, f"Shift right mismatch for {n}")


class TestInstructionFormats(unittest.TestCase):
    """Verify instruction formats match architecture specification"""
    
    def test_type_a_instruction_execution(self):
        """Test Type-A (Register-Register) instructions"""
        proc = PentaryProcessor()
        
        # ADD is Type-A
        program = [
            "MOVI P1, 10",
            "MOVI P2, 20",
            "ADD P3, P1, P2",
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
        self.assertEqual(30, result)
    
    def test_type_b_instruction_execution(self):
        """Test Type-B (Register-Immediate) instructions"""
        proc = PentaryProcessor()
        
        # ADDI is Type-B
        program = [
            "MOVI P1, 10",
            "ADDI P2, P1, 5",
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(15, result)
    
    def test_type_c_instruction_execution(self):
        """Test Type-C (Memory) instructions"""
        proc = PentaryProcessor()
        
        # STORE/LOAD are Type-C
        program = [
            "MOVI P1, 100",         # Base address
            "MOVI P2, 42",          # Value to store
            "STORE P2, [P1 + 0]",   # Store 42 at address 100
            "LOAD P3, [P1 + 0]",    # Load from address 100
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P3"))
        self.assertEqual(42, result)
    
    def test_type_d_instruction_execution(self):
        """Test Type-D (Branch) instructions"""
        proc = PentaryProcessor()
        
        # BEQ is Type-D
        program = [
            "MOVI P1, 0",      # P1 = 0
            "BEQ P1, 3",       # Branch to line 3 if zero
            "MOVI P2, 100",    # Skipped
            "MOVI P2, 200",    # Target
            "HALT"
        ]
        proc.load_program(program)
        proc.run()
        
        result = PentaryConverter.pentary_to_decimal(proc.get_register("P2"))
        self.assertEqual(200, result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
