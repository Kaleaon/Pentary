
import unittest
import sys
import os
import random

# Add tools directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic
from pentary_simulator import PentaryProcessor

class TestPentaryFailures(unittest.TestCase):

    def test_simulator_infinite_precision_flaw(self):
        """
        The architecture supposedly uses fixed-width words (e.g. 16 pents),
        but the simulator allows infinite precision.
        This test demonstrates that the simulator fails to emulate overflow.
        """
        proc = PentaryProcessor()

        # 5^20 is much larger than 5^16 (approx limit for 16 pents)
        large_val = 5**20
        large_pent = PentaryConverter.decimal_to_pentary(large_val)

        # Load a program that creates a large number
        # We cheat a bit by using MOVI with a large decimal, but we could also add loop
        proc.set_register("P1", large_pent)

        val_in_reg = proc.get_register("P1")

        # In a real 16-pent architecture, this should have overflowed or been truncated.
        # Check length of the string
        self.assertGreater(len(val_in_reg), 16, "Register held more than 16 pents, violating architecture limits")

        print(f"\n[Proof 1] Simulator allows {len(val_in_reg)} pents, ignoring hardware limits.")

    def test_arithmetic_consistency(self):
        """
        Compare high-level conversion arithmetic (Truth) vs Low-level digit arithmetic (Architecture).
        If they differ, the architecture logic is flawed.
        """
        # Fuzzing
        for _ in range(100):
            a = random.randint(-1000, 1000)
            b = random.randint(-1000, 1000)

            a_pent = PentaryConverter.decimal_to_pentary(a)
            b_pent = PentaryConverter.decimal_to_pentary(b)

            # Ground truth via decimal conversion
            expected_sum_pent = PentaryConverter.add_pentary(a_pent, b_pent)

            # Architecture logic via digit-by-digit
            calc_sum_pent, _ = PentaryArithmetic.add_pentary_detailed(a_pent, b_pent)

            # Strip leading zeros for comparison
            expected_sum_pent = expected_sum_pent.lstrip('0') or '0'
            calc_sum_pent = calc_sum_pent.lstrip('0') or '0'

            self.assertEqual(calc_sum_pent, expected_sum_pent,
                             f"Mismatch for {a} + {b}: {calc_sum_pent} vs {expected_sum_pent}")

    def test_multiplication_missing_instruction(self):
        """
        Prove that generic multiplication is impossible with a single instruction,
        forcing O(N) software emulation which kills performance.
        """
        proc = PentaryProcessor()

        # Try to execute a generic MUL instruction
        program = ["MUL P1, P2, P3", "HALT"]
        proc.load_program(program)

        # Capture stdout to check for error
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output

        proc.run()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertIn("Unknown instruction: MUL", output, "MUL instruction should not exist")
        print("\n[Proof 2] 'Multiplication is Obsolete' means generic multiplication is missing.")

if __name__ == '__main__':
    unittest.main()
