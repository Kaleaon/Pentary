#!/usr/bin/env python3
"""
Comprehensive Pentary Stress Testing Suite
Tests overflow handling, edge cases, and system robustness
"""

import random
import sys
import time
from typing import List, Dict, Tuple
sys.path.insert(0, 'tools')

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic

class PentaryStressTest:
    def __init__(self):
        self.converter = PentaryConverter()
        self.arithmetic = PentaryArithmetic()
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }

    def log_error(self, test_name: str, details: str):
        """Log an error with details"""
        self.test_results['failed'] += 1
        self.test_results['errors'].append({
            'test': test_name,
            'details': details
        })

    def log_pass(self):
        """Log a passed test"""
        self.test_results['passed'] += 1

    def fuzz_arithmetic_core(self, iterations=10000):
        """
        Phase 1: Logical Correctness (The Fuzzer)
        Throw random garbage at the ALU and see if it matches Python's math.
        """
        print(f"\n{'='*70}")
        print(f"[*] PHASE 1: Fuzzing Arithmetic Core ({iterations} operations)")
        print(f"{'='*70}")
        
        errors = []
        edge_cases = [0, 1, -1, 2, -2, 5, -5, 10, -10, 100, -100, 1000, -1000, 
                      2**10, -2**10, 2**16, -2**16]
        
        for i in range(iterations):
            # Mix edge cases with random values
            if i % 10 == 0:
                a_dec = random.choice(edge_cases)
                b_dec = random.choice(edge_cases)
            else:
                a_dec = random.randint(-10000, 10000)
                b_dec = random.randint(-10000, 10000)
            
            try:
                # Convert to pentary
                a_pent = self.converter.decimal_to_pentary(a_dec)
                b_pent = self.converter.decimal_to_pentary(b_dec)
                
                # Perform Pentary Addition
                res_pent = self.converter.add_pentary(a_pent, b_pent)
                res_dec = self.converter.pentary_to_decimal(res_pent)
                
                # Verify
                expected = a_dec + b_dec
                if res_dec != expected:
                    error_msg = f"{a_dec} + {b_dec} = {res_dec} (expected {expected})\n"
                    error_msg += f"  Pentary: {a_pent} + {b_pent} = {res_pent}"
                    errors.append(error_msg)
                    self.log_error("fuzz_arithmetic", error_msg)
                    if len(errors) <= 5:
                        print(f"  [FAIL] {error_msg}")
                else:
                    self.log_pass()
                    
            except Exception as e:
                error_msg = f"Exception with {a_dec} + {b_dec}: {str(e)}"
                errors.append(error_msg)
                self.log_error("fuzz_arithmetic", error_msg)
                if len(errors) <= 5:
                    print(f"  [ERROR] {error_msg}")
        
        if len(errors) == 0:
            print(f"  ✓ [PASS] All {iterations} arithmetic operations correct!")
        else:
            print(f"  ✗ [FAIL] {len(errors)} errors out of {iterations} operations ({len(errors)/iterations*100:.2f}%)")
            if len(errors) > 5:
                print(f"  (Showing first 5 errors, {len(errors)-5} more suppressed)")

    def test_overflow_scenarios(self):
        """
        Phase 2: Overflow and Underflow Testing
        Test extreme values and boundary conditions
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 2: Overflow and Underflow Testing")
        print(f"{'='*70}")
        
        test_cases = [
            # (a, b, description)
            (2**20, 2**20, "Large positive overflow"),
            (-2**20, -2**20, "Large negative overflow"),
            (2**30, 1, "Near max positive"),
            (-2**30, -1, "Near max negative"),
            (2**15, -2**15, "Large positive + large negative"),
            (0, 2**20, "Zero + large positive"),
            (0, -2**20, "Zero + large negative"),
        ]
        
        print(f"\n{'Test Case':<40} {'Status':<10} {'Details'}")
        print("-" * 70)
        
        for a_dec, b_dec, description in test_cases:
            try:
                a_pent = self.converter.decimal_to_pentary(a_dec)
                b_pent = self.converter.decimal_to_pentary(b_dec)
                res_pent = self.converter.add_pentary(a_pent, b_pent)
                res_dec = self.converter.pentary_to_decimal(res_pent)
                
                expected = a_dec + b_dec
                if res_dec == expected:
                    print(f"{description:<40} {'✓ PASS':<10} {res_dec}")
                    self.log_pass()
                else:
                    print(f"{description:<40} {'✗ FAIL':<10} Got {res_dec}, expected {expected}")
                    self.log_error("overflow", f"{description}: {res_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<40} {'✗ ERROR':<10} {str(e)}")
                self.log_error("overflow", f"{description}: {str(e)}")

    def test_edge_cases(self):
        """
        Phase 3: Edge Case Testing
        Test special values and boundary conditions
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 3: Edge Case Testing")
        print(f"{'='*70}")
        
        edge_tests = [
            (0, 0, "Zero + Zero"),
            (1, -1, "Positive + Negative (cancel)"),
            (2, -2, "Max digit + Min digit"),
            (5, -5, "Base + Negative base"),
            (25, -25, "Base^2 + Negative base^2"),
            (125, -125, "Base^3 + Negative base^3"),
            (1, 1, "Smallest positive + Smallest positive"),
            (-1, -1, "Smallest negative + Smallest negative"),
            (2, 2, "Max digit + Max digit"),
            (-2, -2, "Min digit + Min digit"),
        ]
        
        print(f"\n{'Test Case':<40} {'Status':<10} {'Result'}")
        print("-" * 70)
        
        for a_dec, b_dec, description in edge_tests:
            try:
                a_pent = self.converter.decimal_to_pentary(a_dec)
                b_pent = self.converter.decimal_to_pentary(b_dec)
                res_pent = self.converter.add_pentary(a_pent, b_pent)
                res_dec = self.converter.pentary_to_decimal(res_pent)
                
                expected = a_dec + b_dec
                if res_dec == expected:
                    print(f"{description:<40} {'✓ PASS':<10} {res_pent} ({res_dec})")
                    self.log_pass()
                else:
                    print(f"{description:<40} {'✗ FAIL':<10} Got {res_dec}, expected {expected}")
                    self.log_error("edge_case", f"{description}: {res_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<40} {'✗ ERROR':<10} {str(e)}")
                self.log_error("edge_case", f"{description}: {str(e)}")

    def test_carry_propagation(self):
        """
        Phase 4: Carry Propagation Testing
        Test scenarios with extensive carry chains
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 4: Carry Propagation Testing")
        print(f"{'='*70}")
        
        # Create numbers that will cause long carry chains
        carry_tests = [
            ("⊕⊕⊕⊕", "+", "Multiple max digits + 1"),
            ("++++++", "+", "Multiple +1 digits + 1"),
            ("⊖⊖⊖⊖", "-", "Multiple min digits - 1"),
            ("⊕⊕⊕", "⊕⊕⊕", "Max digits + Max digits"),
            ("⊕+0-⊖", "⊕+0-⊖", "Mixed pattern + same pattern"),
        ]
        
        print(f"\n{'Test Case':<40} {'Status':<10} {'Details'}")
        print("-" * 70)
        
        for a_pent, b_pent, description in carry_tests:
            try:
                a_dec = self.converter.pentary_to_decimal(a_pent)
                b_dec = self.converter.pentary_to_decimal(b_pent)
                
                res_pent = self.converter.add_pentary(a_pent, b_pent)
                res_dec = self.converter.pentary_to_decimal(res_pent)
                
                expected = a_dec + b_dec
                if res_dec == expected:
                    print(f"{description:<40} {'✓ PASS':<10} {res_pent}")
                    self.log_pass()
                else:
                    print(f"{description:<40} {'✗ FAIL':<10} Got {res_dec}, expected {expected}")
                    self.log_error("carry_propagation", f"{description}: {res_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<40} {'✗ ERROR':<10} {str(e)}")
                self.log_error("carry_propagation", f"{description}: {str(e)}")

    def simulate_resistance_drift(self, pent_value: str, noise_prob: float = 0.01) -> str:
        """
        Phase 5: Hardware Simulation
        Simulates memristor resistance drifting to a neighboring state.
        """
        chars = list(pent_value)
        states = ['⊖', '-', '0', '+', '⊕']  # -2, -1, 0, 1, 2
        
        corrupted = []
        for char in chars:
            if char not in states:
                corrupted.append(char)
                continue
                
            if random.random() < noise_prob:
                # Drift to neighbor (resistance slightly up or down)
                current_idx = states.index(char)
                drift = random.choice([-1, 1])
                new_idx = max(0, min(4, current_idx + drift))
                corrupted.append(states[new_idx])
            else:
                corrupted.append(char)
        return "".join(corrupted)

    def torture_test_memory(self, data_size=1000, noise_levels=[0.001, 0.01, 0.05]):
        """
        Phase 5: Memory Integrity Testing
        Write data -> Inject Noise -> Read data -> Measure corruption
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 5: Memristor Drift Simulation (Memory Integrity)")
        print(f"{'='*70}")
        
        # Create "Perfect" Data
        print(f"\nGenerating {data_size} random test values...")
        data_dec = [random.randint(-5000, 5000) for _ in range(data_size)]
        data_pent = [self.converter.decimal_to_pentary(x) for x in data_dec]
        
        print(f"\n{'Noise Level':<15} {'Intact Data':<20} {'Sparsity Drift':<30}")
        print("-" * 70)
        
        for noise in noise_levels:
            corrupted_pent = [self.simulate_resistance_drift(x, noise) for x in data_pent]
            
            # Try to decode corrupted data
            corrupted_dec = []
            decode_errors = 0
            for pent in corrupted_pent:
                try:
                    corrupted_dec.append(self.converter.pentary_to_decimal(pent))
                except:
                    corrupted_dec.append(None)
                    decode_errors += 1
            
            # Check how many values survived intact
            intact = sum(1 for i in range(len(data_dec)) 
                        if corrupted_dec[i] is not None and data_dec[i] == corrupted_dec[i])
            
            # Verify if "Zero" states drifted (Power impact)
            zeros_original = sum(x.count('0') for x in data_pent)
            zeros_corrupt = sum(x.count('0') for x in corrupted_pent)
            sparsity_change = zeros_corrupt - zeros_original
            
            intact_pct = intact/data_size*100
            print(f"{noise*100:>6.2f}%{'':<8} {intact}/{data_size} ({intact_pct:.1f}%){'':<5} "
                  f"{zeros_original} → {zeros_corrupt} ({sparsity_change:+d})")
            
            if decode_errors > 0:
                print(f"{'':15} Decode errors: {decode_errors}")

    def test_conversion_roundtrip(self, iterations=1000):
        """
        Phase 6: Conversion Roundtrip Testing
        Test decimal -> pentary -> decimal conversions
        """
        print(f"\n{'='*70}")
        print(f"[*] PHASE 6: Conversion Roundtrip Testing ({iterations} iterations)")
        print(f"{'='*70}")
        
        errors = []
        test_ranges = [
            (-100, 100, "Small range"),
            (-10000, 10000, "Medium range"),
            (-100000, 100000, "Large range"),
        ]
        
        for min_val, max_val, description in test_ranges:
            range_errors = 0
            for _ in range(iterations // len(test_ranges)):
                dec_original = random.randint(min_val, max_val)
                
                try:
                    pent = self.converter.decimal_to_pentary(dec_original)
                    dec_back = self.converter.pentary_to_decimal(pent)
                    
                    if dec_original != dec_back:
                        range_errors += 1
                        if len(errors) < 5:
                            errors.append(f"{description}: {dec_original} -> {pent} -> {dec_back}")
                            self.log_error("roundtrip", errors[-1])
                    else:
                        self.log_pass()
                        
                except Exception as e:
                    range_errors += 1
                    if len(errors) < 5:
                        errors.append(f"{description}: {dec_original} raised {str(e)}")
                        self.log_error("roundtrip", errors[-1])
            
            if range_errors == 0:
                print(f"  ✓ {description}: All conversions successful")
            else:
                print(f"  ✗ {description}: {range_errors} errors")
        
        if errors:
            print(f"\nFirst few errors:")
            for err in errors[:5]:
                print(f"    {err}")

    def test_subtraction_operations(self, iterations=1000):
        """
        Phase 7: Subtraction Testing
        Test pentary subtraction operations
        """
        print(f"\n{'='*70}")
        print(f"[*] PHASE 7: Subtraction Operations Testing ({iterations} iterations)")
        print(f"{'='*70}")
        
        errors = []
        for i in range(iterations):
            a_dec = random.randint(-5000, 5000)
            b_dec = random.randint(-5000, 5000)
            
            try:
                a_pent = self.converter.decimal_to_pentary(a_dec)
                b_pent = self.converter.decimal_to_pentary(b_dec)
                
                res_pent = self.converter.subtract_pentary(a_pent, b_pent)
                res_dec = self.converter.pentary_to_decimal(res_pent)
                
                expected = a_dec - b_dec
                if res_dec != expected:
                    if len(errors) < 5:
                        errors.append(f"{a_dec} - {b_dec} = {res_dec} (expected {expected})")
                        self.log_error("subtraction", errors[-1])
                else:
                    self.log_pass()
                    
            except Exception as e:
                if len(errors) < 5:
                    errors.append(f"Exception with {a_dec} - {b_dec}: {str(e)}")
                    self.log_error("subtraction", errors[-1])
        
        if len(errors) == 0:
            print(f"  ✓ [PASS] All {iterations} subtraction operations correct!")
        else:
            print(f"  ✗ [FAIL] {len(errors)} errors detected")
            print(f"\nFirst few errors:")
            for err in errors[:5]:
                print(f"    {err}")

    def test_multiplication_by_constants(self):
        """
        Phase 8: Multiplication by Constants Testing
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 8: Multiplication by Constants Testing")
        print(f"{'='*70}")
        
        test_values = [0, 1, -1, 2, -2, 5, -5, 10, -10, 100, -100]
        constants = [-2, -1, 0, 1, 2]
        
        errors = []
        for val in test_values:
            for const in constants:
                try:
                    pent = self.converter.decimal_to_pentary(val)
                    result_pent = self.converter.multiply_pentary_by_constant(pent, const)
                    result_dec = self.converter.pentary_to_decimal(result_pent)
                    
                    expected = val * const
                    if result_dec != expected:
                        errors.append(f"{val} × {const} = {result_dec} (expected {expected})")
                        self.log_error("multiplication", errors[-1])
                    else:
                        self.log_pass()
                        
                except Exception as e:
                    errors.append(f"Exception with {val} × {const}: {str(e)}")
                    self.log_error("multiplication", errors[-1])
        
        if len(errors) == 0:
            print(f"  ✓ [PASS] All multiplication operations correct!")
        else:
            print(f"  ✗ [FAIL] {len(errors)} errors detected")
            for err in errors[:10]:
                print(f"    {err}")

    def test_shift_operations(self):
        """
        Phase 9: Shift Operations Testing
        """
        print(f"\n{'='*70}")
        print("[*] PHASE 9: Shift Operations Testing")
        print(f"{'='*70}")
        
        test_values = [1, 2, 5, 7, 10, 25, 50, 100]
        
        print(f"\n{'Value':<10} {'Left Shift':<20} {'Right Shift':<20} {'Status'}")
        print("-" * 70)
        
        for val in test_values:
            try:
                pent = self.converter.decimal_to_pentary(val)
                
                # Test left shift (multiply by 5)
                left_pent = self.converter.shift_left_pentary(pent, 1)
                left_dec = self.converter.pentary_to_decimal(left_pent)
                left_ok = (left_dec == val * 5)
                
                # Test right shift (divide by 5)
                right_pent = self.converter.shift_right_pentary(left_pent, 1)
                right_dec = self.converter.pentary_to_decimal(right_pent)
                right_ok = (right_dec == val)
                
                status = "✓ PASS" if (left_ok and right_ok) else "✗ FAIL"
                print(f"{val:<10} {left_dec:<20} {right_dec:<20} {status}")
                
                if left_ok and right_ok:
                    self.log_pass()
                else:
                    self.log_error("shift", f"Value {val}: left={left_dec}, right={right_dec}")
                    
            except Exception as e:
                print(f"{val:<10} {'ERROR':<20} {'ERROR':<20} ✗ ERROR")
                self.log_error("shift", f"Value {val}: {str(e)}")

    def generate_report(self):
        """Generate final test report"""
        print(f"\n{'='*70}")
        print("FINAL TEST REPORT")
        print(f"{'='*70}")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        pass_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"Passed: {self.test_results['passed']} ({pass_rate:.2f}%)")
        print(f"Failed: {self.test_results['failed']} ({100-pass_rate:.2f}%)")
        
        if self.test_results['failed'] > 0:
            print(f"\n{'='*70}")
            print("ERROR SUMMARY")
            print(f"{'='*70}")
            
            # Group errors by test type
            error_groups = {}
            for error in self.test_results['errors']:
                test_type = error['test']
                if test_type not in error_groups:
                    error_groups[test_type] = []
                error_groups[test_type].append(error['details'])
            
            for test_type, errors in error_groups.items():
                print(f"\n{test_type.upper()} ({len(errors)} errors):")
                for i, err in enumerate(errors[:5], 1):
                    print(f"  {i}. {err}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors)-5} more")
        
        return self.test_results

def main():
    """Run comprehensive stress tests"""
    print(f"\n{'#'*70}")
    print("PENTARY SYSTEM COMPREHENSIVE STRESS TEST SUITE")
    print(f"{'#'*70}")
    print("\nThis suite will test:")
    print("  1. Arithmetic correctness under random inputs")
    print("  2. Overflow and underflow handling")
    print("  3. Edge case behavior")
    print("  4. Carry propagation")
    print("  5. Memory integrity with simulated hardware noise")
    print("  6. Conversion roundtrip accuracy")
    print("  7. Subtraction operations")
    print("  8. Multiplication by constants")
    print("  9. Shift operations")
    
    tester = PentaryStressTest()
    
    start_time = time.time()
    
    # Run all test phases
    tester.fuzz_arithmetic_core(iterations=10000)
    tester.test_overflow_scenarios()
    tester.test_edge_cases()
    tester.test_carry_propagation()
    tester.torture_test_memory(data_size=1000)
    tester.test_conversion_roundtrip(iterations=1000)
    tester.test_subtraction_operations(iterations=1000)
    tester.test_multiplication_by_constants()
    tester.test_shift_operations()
    
    elapsed_time = time.time() - start_time
    
    # Generate final report
    results = tester.generate_report()
    
    print(f"\n{'='*70}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"{'='*70}\n")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with error code if tests failed
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)