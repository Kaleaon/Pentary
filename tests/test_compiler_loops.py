#!/usr/bin/env python3
"""
Test script for Pent compiler loop generation.
"""
import sys
import os
import unittest

# Ensure import paths are correct
sys.path.insert(0, os.path.abspath('language'))
sys.path.insert(0, os.path.abspath('tools'))

from pent_compiler import Compiler
from pentary_simulator import PentaryProcessor
from pentary_converter import PentaryConverter

class TestCompilerLoops(unittest.TestCase):

    def test_range_syntax_operator(self):
        """Test 'for i in 0..5' syntax"""
        source = """
        fn main() {
            let sum = 0;
            for i in 0..5 {
                sum = sum + i;
            }
        }
        """

        compiler = Compiler()
        assembly = compiler.compile(source)

        proc = PentaryProcessor()
        proc.load_program(assembly)
        proc.run()

        # P1 = sum
        sum_val = proc.get_register("P1")
        sum_dec = PentaryConverter.pentary_to_decimal(sum_val)

        # 0+1+2+3+4 = 10
        self.assertEqual(sum_dec, 10, f"Expected sum 10, got {sum_dec}")

    def test_range_function_call(self):
        """Test 'for i in range(0, 5)' syntax"""
        source = """
        fn main() {
            let sum = 0;
            for i in range(0, 5) {
                sum = sum + i;
            }
        }
        """

        compiler = Compiler()
        assembly = compiler.compile(source)

        proc = PentaryProcessor()
        proc.load_program(assembly)
        proc.run()

        sum_val = proc.get_register("P1")
        sum_dec = PentaryConverter.pentary_to_decimal(sum_val)

        self.assertEqual(sum_dec, 10, f"Expected sum 10, got {sum_dec}")

    def test_range_function_single_arg(self):
        """Test 'for i in range(5)' syntax (0..5)"""
        source = """
        fn main() {
            let sum = 0;
            for i in range(5) {
                sum = sum + i;
            }
        }
        """

        compiler = Compiler()
        assembly = compiler.compile(source)

        proc = PentaryProcessor()
        proc.load_program(assembly)
        proc.run()

        sum_val = proc.get_register("P1")
        sum_dec = PentaryConverter.pentary_to_decimal(sum_val)

        self.assertEqual(sum_dec, 10, f"Expected sum 10, got {sum_dec}")

if __name__ == "__main__":
    unittest.main()
