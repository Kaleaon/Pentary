#!/usr/bin/env python3
"""
Advanced Pentary Stress Testing Suite
Tests extreme edge cases, performance limits, and potential vulnerabilities
"""

import random
import sys
import time
import math
from typing import List, Dict, Tuple
sys.path.insert(0, 'tools')

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic

class AdvancedPentaryStressTest:
    def __init__(self):
        self.converter = PentaryConverter()
        self.arithmetic = PentaryArithmetic()
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'performance': {}
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

    def test_extreme_values(self):
        """
        Test 1: Extreme Value Handling
        Test with very large numbers to find breaking points
        """
        print(f"\n{'='*70}")
        print("[*] TEST 1: Extreme Value Handling")
        print(f"{'='*70}")
        
        extreme_values = [
            (2**32, "2^32"),
            (2**40, "2^40"),
            (2**50, "2^50"),
            (2**60, "2^60"),
            (-2**32, "-2^32"),
            (-2**40, "-2^40"),
            (10**10, "10^10"),
            (10**15, "10^15"),
            (-10**10, "-10^10"),
        ]
        
        print(f"\n{'Value':<20} {'Conversion Time':<20} {'Status'}")
        print("-" * 70)
        
        for value, description in extreme_values:
            try:
                start = time.time()
                pent = self.converter.decimal_to_pentary(value)
                elapsed = time.time() - start
                
                # Verify roundtrip
                back = self.converter.pentary_to_decimal(pent)
                
                if back == value:
                    print(f"{description:<20} {elapsed*1000:.2f}ms{'':<13} ✓ PASS")
                    self.log_pass()
                else:
                    print(f"{description:<20} {elapsed*1000:.2f}ms{'':<13} ✗ FAIL (roundtrip)")
                    self.log_error("extreme_values", f"{description}: {value} != {back}")
                    
            except Exception as e:
                print(f"{description:<20} {'ERROR':<20} ✗ ERROR: {str(e)[:30]}")
                self.log_error("extreme_values", f"{description}: {str(e)}")

    def test_precision_limits(self):
        """
        Test 2: Precision and Rounding Behavior
        Test numbers near representation boundaries
        """
        print(f"\n{'='*70}")
        print("[*] TEST 2: Precision Limits and Boundary Conditions")
        print(f"{'='*70}")
        
        # Test powers of 5 (base of pentary)
        print("\nPowers of 5 (base boundaries):")
        print(f"{'Power':<15} {'Decimal':<20} {'Pentary':<25} {'Status'}")
        print("-" * 70)
        
        for i in range(0, 15):
            value = 5**i
            try:
                pent = self.converter.decimal_to_pentary(value)
                back = self.converter.pentary_to_decimal(pent)
                
                status = "✓ PASS" if back == value else "✗ FAIL"
                print(f"5^{i:<13} {value:<20} {pent:<25} {status}")
                
                if back == value:
                    self.log_pass()
                else:
                    self.log_error("precision", f"5^{i}: {value} != {back}")
                    
            except Exception as e:
                print(f"5^{i:<13} {value:<20} {'ERROR':<25} ✗ ERROR")
                self.log_error("precision", f"5^{i}: {str(e)}")
        
        # Test numbers with all same digits
        print("\nHomogeneous digit patterns:")
        print(f"{'Pattern':<15} {'Decimal':<20} {'Pentary':<25} {'Status'}")
        print("-" * 70)
        
        patterns = [
            ("⊕" * 5, "All max positive"),
            ("+" * 5, "All weak positive"),
            ("0" * 5, "All zeros"),
            ("-" * 5, "All weak negative"),
            ("⊖" * 5, "All max negative"),
        ]
        
        for pattern, description in patterns:
            try:
                dec = self.converter.pentary_to_decimal(pattern)
                back_pent = self.converter.decimal_to_pentary(dec)
                back_dec = self.converter.pentary_to_decimal(back_pent)
                
                status = "✓ PASS" if back_dec == dec else "✗ FAIL"
                print(f"{description:<15} {dec:<20} {pattern:<25} {status}")
                
                if back_dec == dec:
                    self.log_pass()
                else:
                    self.log_error("precision", f"{description}: {dec} != {back_dec}")
                    
            except Exception as e:
                print(f"{description:<15} {'ERROR':<20} {pattern:<25} ✗ ERROR")
                self.log_error("precision", f"{description}: {str(e)}")

    def test_cascading_carries(self):
        """
        Test 3: Cascading Carry Propagation
        Test scenarios that cause maximum carry propagation
        """
        print(f"\n{'='*70}")
        print("[*] TEST 3: Cascading Carry Propagation")
        print(f"{'='*70}")
        
        # Create numbers that will cause maximum carry chains
        print("\nMaximum carry propagation scenarios:")
        print(f"{'Scenario':<30} {'Carries':<15} {'Status'}")
        print("-" * 70)
        
        test_cases = [
            # Adding 1 to all 2's should cascade
            ("⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕", "+", "10 digits all ⊕ + 1"),
            # Subtracting 1 from all -2's should cascade
            ("⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖", "-", "10 digits all ⊖ - 1"),
            # Adding two numbers with alternating max digits
            ("⊕0⊕0⊕0⊕0⊕0", "⊕0⊕0⊕0⊕0⊕0", "Alternating pattern"),
            # Maximum positive + 1
            ("⊕" * 20, "+", "20 digits all ⊕ + 1"),
        ]
        
        for a, b, description in test_cases:
            try:
                a_dec = self.converter.pentary_to_decimal(a)
                b_dec = self.converter.pentary_to_decimal(b)
                
                result_pent = self.converter.add_pentary(a, b)
                result_dec = self.converter.pentary_to_decimal(result_pent)
                
                expected = a_dec + b_dec
                
                # Count potential carries (rough estimate)
                carries = min(len(a), len(b))
                
                if result_dec == expected:
                    print(f"{description:<30} {carries:<15} ✓ PASS")
                    self.log_pass()
                else:
                    print(f"{description:<30} {carries:<15} ✗ FAIL")
                    self.log_error("cascading_carries", 
                                 f"{description}: {result_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<30} {'ERROR':<15} ✗ ERROR")
                self.log_error("cascading_carries", f"{description}: {str(e)}")

    def test_sign_transitions(self):
        """
        Test 4: Sign Transition Behavior
        Test operations that cross zero boundary
        """
        print(f"\n{'='*70}")
        print("[*] TEST 4: Sign Transition Behavior")
        print(f"{'='*70}")
        
        print("\nOperations crossing zero:")
        print(f"{'Operation':<40} {'Result':<15} {'Status'}")
        print("-" * 70)
        
        test_cases = [
            (1, -1, "1 + (-1) = 0"),
            (100, -100, "100 + (-100) = 0"),
            (1000, -1000, "1000 + (-1000) = 0"),
            (-1, 2, "-1 + 2 = 1"),
            (-100, 101, "-100 + 101 = 1"),
            (50, -51, "50 + (-51) = -1"),
            (0, 0, "0 + 0 = 0"),
            (1, -2, "1 + (-2) = -1"),
            (-1, 2, "-1 + 2 = 1"),
        ]
        
        for a, b, description in test_cases:
            try:
                a_pent = self.converter.decimal_to_pentary(a)
                b_pent = self.converter.decimal_to_pentary(b)
                
                result_pent = self.converter.add_pentary(a_pent, b_pent)
                result_dec = self.converter.pentary_to_decimal(result_pent)
                
                expected = a + b
                
                if result_dec == expected:
                    print(f"{description:<40} {result_pent:<15} ✓ PASS")
                    self.log_pass()
                else:
                    print(f"{description:<40} {result_dec:<15} ✗ FAIL")
                    self.log_error("sign_transitions", 
                                 f"{description}: {result_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<40} {'ERROR':<15} ✗ ERROR")
                self.log_error("sign_transitions", f"{description}: {str(e)}")

    def test_performance_scaling(self):
        """
        Test 5: Performance Scaling
        Measure performance with increasing number sizes
        """
        print(f"\n{'='*70}")
        print("[*] TEST 5: Performance Scaling Analysis")
        print(f"{'='*70}")
        
        print("\nConversion performance vs number size:")
        print(f"{'Magnitude':<20} {'Avg Time (ms)':<20} {'Ops/sec':<15}")
        print("-" * 70)
        
        magnitudes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        
        for magnitude in magnitudes:
            times = []
            iterations = max(10, min(1000, 100000 // magnitude))  # At least 10 iterations
            
            for _ in range(iterations):
                value = random.randint(-magnitude, magnitude)
                
                start = time.time()
                pent = self.converter.decimal_to_pentary(value)
                back = self.converter.pentary_to_decimal(pent)
                elapsed = time.time() - start
                
                times.append(elapsed)
                
                if back != value:
                    self.log_error("performance", f"Roundtrip failed for {value}")
            
            avg_time = sum(times) / len(times) * 1000  # Convert to ms
            ops_per_sec = 1000 / avg_time if avg_time > 0 else float('inf')
            
            print(f"±{magnitude:<19} {avg_time:.4f}{'':<15} {ops_per_sec:.0f}")
            
            self.test_results['performance'][f'magnitude_{magnitude}'] = {
                'avg_time_ms': avg_time,
                'ops_per_sec': ops_per_sec
            }

    def test_arithmetic_commutativity(self):
        """
        Test 6: Arithmetic Properties
        Verify commutativity, associativity, and distributivity
        """
        print(f"\n{'='*70}")
        print("[*] TEST 6: Arithmetic Properties Verification")
        print(f"{'='*70}")
        
        print("\nCommutativity (a + b = b + a):")
        errors = 0
        for _ in range(100):
            a = random.randint(-1000, 1000)
            b = random.randint(-1000, 1000)
            
            a_pent = self.converter.decimal_to_pentary(a)
            b_pent = self.converter.decimal_to_pentary(b)
            
            result1 = self.converter.pentary_to_decimal(
                self.converter.add_pentary(a_pent, b_pent))
            result2 = self.converter.pentary_to_decimal(
                self.converter.add_pentary(b_pent, a_pent))
            
            if result1 != result2:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: {a} + {b} = {result1}, but {b} + {a} = {result2}")
                self.log_error("commutativity", f"{a} + {b} != {b} + {a}")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 100 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")
        
        print("\nAssociativity ((a + b) + c = a + (b + c)):")
        errors = 0
        for _ in range(100):
            a = random.randint(-500, 500)
            b = random.randint(-500, 500)
            c = random.randint(-500, 500)
            
            a_pent = self.converter.decimal_to_pentary(a)
            b_pent = self.converter.decimal_to_pentary(b)
            c_pent = self.converter.decimal_to_pentary(c)
            
            # (a + b) + c
            temp1 = self.converter.add_pentary(a_pent, b_pent)
            result1 = self.converter.pentary_to_decimal(
                self.converter.add_pentary(temp1, c_pent))
            
            # a + (b + c)
            temp2 = self.converter.add_pentary(b_pent, c_pent)
            result2 = self.converter.pentary_to_decimal(
                self.converter.add_pentary(a_pent, temp2))
            
            if result1 != result2:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: ({a} + {b}) + {c} = {result1}, but {a} + ({b} + {c}) = {result2}")
                self.log_error("associativity", f"({a} + {b}) + {c} != {a} + ({b} + {c})")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 100 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")

    def test_negation_properties(self):
        """
        Test 7: Negation Properties
        Verify that negation works correctly
        """
        print(f"\n{'='*70}")
        print("[*] TEST 7: Negation Properties")
        print(f"{'='*70}")
        
        print("\nDouble negation (-(-a) = a):")
        errors = 0
        for _ in range(100):
            a = random.randint(-1000, 1000)
            
            a_pent = self.converter.decimal_to_pentary(a)
            neg_once = self.converter.negate_pentary(a_pent)
            neg_twice = self.converter.negate_pentary(neg_once)
            
            result = self.converter.pentary_to_decimal(neg_twice)
            
            if result != a:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: -(-{a}) = {result}, expected {a}")
                self.log_error("negation", f"-(-{a}) != {a}")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 100 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")
        
        print("\nAdditive inverse (a + (-a) = 0):")
        errors = 0
        for _ in range(100):
            a = random.randint(-1000, 1000)
            
            a_pent = self.converter.decimal_to_pentary(a)
            neg_a_pent = self.converter.negate_pentary(a_pent)
            
            result_pent = self.converter.add_pentary(a_pent, neg_a_pent)
            result = self.converter.pentary_to_decimal(result_pent)
            
            if result != 0:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: {a} + (-{a}) = {result}, expected 0")
                self.log_error("additive_inverse", f"{a} + (-{a}) != 0")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 100 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")

    def test_shift_consistency(self):
        """
        Test 8: Shift Operation Consistency
        Verify shift operations maintain mathematical relationships
        """
        print(f"\n{'='*70}")
        print("[*] TEST 8: Shift Operation Consistency")
        print(f"{'='*70}")
        
        print("\nLeft shift = multiply by 5:")
        errors = 0
        for _ in range(100):
            a = random.randint(1, 10000)
            
            a_pent = self.converter.decimal_to_pentary(a)
            shifted = self.converter.shift_left_pentary(a_pent, 1)
            shifted_dec = self.converter.pentary_to_decimal(shifted)
            
            expected = a * 5
            
            if shifted_dec != expected:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: {a} << 1 = {shifted_dec}, expected {expected}")
                self.log_error("shift_left", f"{a} << 1 != {a * 5}")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 100 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")
        
        print("\nMultiple shifts:")
        errors = 0
        for _ in range(50):
            a = random.randint(1, 1000)
            shifts = random.randint(1, 5)
            
            a_pent = self.converter.decimal_to_pentary(a)
            shifted = self.converter.shift_left_pentary(a_pent, shifts)
            shifted_dec = self.converter.pentary_to_decimal(shifted)
            
            expected = a * (5 ** shifts)
            
            if shifted_dec != expected:
                errors += 1
                if errors <= 3:
                    print(f"  ✗ FAIL: {a} << {shifts} = {shifted_dec}, expected {expected}")
                self.log_error("multi_shift", f"{a} << {shifts} != {expected}")
            else:
                self.log_pass()
        
        if errors == 0:
            print("  ✓ PASS: All 50 tests passed")
        else:
            print(f"  ✗ FAIL: {errors} failures")

    def test_zero_handling(self):
        """
        Test 9: Zero Handling
        Comprehensive tests for zero in various contexts
        """
        print(f"\n{'='*70}")
        print("[*] TEST 9: Zero Handling")
        print(f"{'='*70}")
        
        print("\nZero in various operations:")
        print(f"{'Operation':<40} {'Result':<15} {'Status'}")
        print("-" * 70)
        
        test_cases = [
            (0, 0, "0 + 0"),
            (0, 1, "0 + 1"),
            (1, 0, "1 + 0"),
            (0, -1, "0 + (-1)"),
            (-1, 0, "(-1) + 0"),
            (0, 1000, "0 + 1000"),
            (1000, 0, "1000 + 0"),
        ]
        
        for a, b, description in test_cases:
            try:
                a_pent = self.converter.decimal_to_pentary(a)
                b_pent = self.converter.decimal_to_pentary(b)
                
                result_pent = self.converter.add_pentary(a_pent, b_pent)
                result_dec = self.converter.pentary_to_decimal(result_pent)
                
                expected = a + b
                
                if result_dec == expected:
                    print(f"{description:<40} {result_pent:<15} ✓ PASS")
                    self.log_pass()
                else:
                    print(f"{description:<40} {result_dec:<15} ✗ FAIL")
                    self.log_error("zero_handling", f"{description}: {result_dec} != {expected}")
                    
            except Exception as e:
                print(f"{description:<40} {'ERROR':<15} ✗ ERROR")
                self.log_error("zero_handling", f"{description}: {str(e)}")

    def generate_report(self):
        """Generate final test report"""
        print(f"\n{'='*70}")
        print("ADVANCED TEST REPORT")
        print(f"{'='*70}")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        pass_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"Passed: {self.test_results['passed']} ({pass_rate:.2f}%)")
        print(f"Failed: {self.test_results['failed']} ({100-pass_rate:.2f}%)")
        
        if self.test_results['performance']:
            print(f"\n{'='*70}")
            print("PERFORMANCE SUMMARY")
            print(f"{'='*70}")
            
            for key, metrics in self.test_results['performance'].items():
                magnitude = key.replace('magnitude_', '')
                print(f"\nMagnitude ±{magnitude}:")
                print(f"  Average time: {metrics['avg_time_ms']:.4f} ms")
                print(f"  Operations/sec: {metrics['ops_per_sec']:.0f}")
        
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
    """Run advanced stress tests"""
    print(f"\n{'#'*70}")
    print("PENTARY SYSTEM ADVANCED STRESS TEST SUITE")
    print(f"{'#'*70}")
    print("\nThis suite will test:")
    print("  1. Extreme value handling")
    print("  2. Precision limits and boundaries")
    print("  3. Cascading carry propagation")
    print("  4. Sign transition behavior")
    print("  5. Performance scaling")
    print("  6. Arithmetic properties (commutativity, associativity)")
    print("  7. Negation properties")
    print("  8. Shift operation consistency")
    print("  9. Zero handling")
    
    tester = AdvancedPentaryStressTest()
    
    start_time = time.time()
    
    # Run all test phases
    tester.test_extreme_values()
    tester.test_precision_limits()
    tester.test_cascading_carries()
    tester.test_sign_transitions()
    tester.test_performance_scaling()
    tester.test_arithmetic_commutativity()
    tester.test_negation_properties()
    tester.test_shift_consistency()
    tester.test_zero_handling()
    
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