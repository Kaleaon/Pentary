/**
 * =============================================================================
 * Verilator Testbench for Pentary ALU
 * =============================================================================
 * 
 * This testbench uses Verilator to simulate the Pentary ALU module.
 * It tests all ALU operations with comprehensive coverage.
 * 
 * Build: verilator -Wall -cc PentaryALU.v --exe tb_pentary_alu.cpp
 * Run:   ./obj_dir/VPentaryALU
 * 
 * =============================================================================
 */

#include <verilated.h>
#include "VPentaryALU.h"

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <string>

// Pentary digit encoding
#define PENT_NEG2   0b000
#define PENT_NEG1   0b001
#define PENT_ZERO   0b010
#define PENT_POS1   0b011
#define PENT_POS2   0b100

// ALU opcodes
#define OP_ADD      0b000
#define OP_SUB      0b001
#define OP_MUL2     0b010
#define OP_DIV2     0b011
#define OP_NEG      0b100
#define OP_ABS      0b101
#define OP_CMP      0b110
#define OP_MAX      0b111

class PentaryALUTestbench {
private:
    VPentaryALU* dut;
    int tests_passed;
    int tests_failed;
    
public:
    PentaryALUTestbench() {
        dut = new VPentaryALU;
        tests_passed = 0;
        tests_failed = 0;
    }
    
    ~PentaryALUTestbench() {
        delete dut;
    }
    
    // Convert decimal to pentary (48-bit representation)
    uint64_t decimal_to_pentary(int value) {
        uint64_t result = 0;
        bool negative = value < 0;
        if (negative) value = -value;
        
        for (int i = 0; i < 16 && value > 0; i++) {
            int digit = value % 5;
            value /= 5;
            
            // Map 0-4 to pentary encoding
            uint64_t encoded;
            switch (digit) {
                case 0: encoded = PENT_ZERO; break;
                case 1: encoded = PENT_POS1; break;
                case 2: encoded = PENT_POS2; break;
                case 3: encoded = PENT_NEG2; value++; break; // Balanced representation
                case 4: encoded = PENT_NEG1; value++; break;
                default: encoded = PENT_ZERO;
            }
            
            result |= (encoded << (i * 3));
        }
        
        // Fill remaining digits with zero
        for (int i = 0; i < 16; i++) {
            if (((result >> (i * 3)) & 0x7) == 0) {
                result |= (uint64_t(PENT_ZERO) << (i * 3));
            }
        }
        
        if (negative) {
            result = negate_pentary(result);
        }
        
        return result;
    }
    
    // Negate a pentary number
    uint64_t negate_pentary(uint64_t value) {
        uint64_t result = 0;
        for (int i = 0; i < 16; i++) {
            uint64_t digit = (value >> (i * 3)) & 0x7;
            uint64_t negated;
            switch (digit) {
                case PENT_NEG2: negated = PENT_POS2; break;
                case PENT_NEG1: negated = PENT_POS1; break;
                case PENT_ZERO: negated = PENT_ZERO; break;
                case PENT_POS1: negated = PENT_NEG1; break;
                case PENT_POS2: negated = PENT_NEG2; break;
                default: negated = PENT_ZERO;
            }
            result |= (negated << (i * 3));
        }
        return result;
    }
    
    // Convert pentary to decimal
    int pentary_to_decimal(uint64_t value) {
        int result = 0;
        int multiplier = 1;
        
        for (int i = 0; i < 16; i++) {
            uint64_t digit = (value >> (i * 3)) & 0x7;
            int digit_value;
            switch (digit) {
                case PENT_NEG2: digit_value = -2; break;
                case PENT_NEG1: digit_value = -1; break;
                case PENT_ZERO: digit_value =  0; break;
                case PENT_POS1: digit_value =  1; break;
                case PENT_POS2: digit_value =  2; break;
                default: digit_value = 0;
            }
            result += digit_value * multiplier;
            multiplier *= 5;
        }
        
        return result;
    }
    
    // Print pentary number
    std::string pentary_to_string(uint64_t value) {
        std::string result;
        bool leading_zero = true;
        
        for (int i = 15; i >= 0; i--) {
            uint64_t digit = (value >> (i * 3)) & 0x7;
            char c;
            switch (digit) {
                case PENT_NEG2: c = 'N'; break; // -2
                case PENT_NEG1: c = '-'; break; // -1
                case PENT_ZERO: c = '0'; break; //  0
                case PENT_POS1: c = '+'; break; // +1
                case PENT_POS2: c = 'P'; break; // +2
                default: c = '?';
            }
            
            if (c != '0' || !leading_zero || i == 0) {
                leading_zero = false;
                result += c;
            }
        }
        
        return result;
    }
    
    // Run a single test
    bool run_test(const std::string& name, uint64_t a, uint64_t b, uint8_t op, 
                  uint64_t expected_result, bool check_flags = false,
                  bool expected_zero = false, bool expected_negative = false) {
        
        // Set inputs
        dut->operand_a = a;
        dut->operand_b = b;
        dut->opcode = op;
        
        // Evaluate
        dut->eval();
        
        // Check result
        bool pass = (dut->result == expected_result);
        
        if (check_flags) {
            pass = pass && (dut->zero_flag == expected_zero);
            pass = pass && (dut->negative_flag == expected_negative);
        }
        
        if (pass) {
            tests_passed++;
            std::cout << "[PASS] " << name << std::endl;
        } else {
            tests_failed++;
            std::cout << "[FAIL] " << name << std::endl;
            std::cout << "       Input A: " << pentary_to_string(a) 
                      << " (" << pentary_to_decimal(a) << ")" << std::endl;
            std::cout << "       Input B: " << pentary_to_string(b) 
                      << " (" << pentary_to_decimal(b) << ")" << std::endl;
            std::cout << "       Expected: " << pentary_to_string(expected_result) 
                      << " (" << pentary_to_decimal(expected_result) << ")" << std::endl;
            std::cout << "       Got:      " << pentary_to_string(dut->result) 
                      << " (" << pentary_to_decimal(dut->result) << ")" << std::endl;
            if (check_flags) {
                std::cout << "       Flags: zero=" << (int)dut->zero_flag 
                          << " negative=" << (int)dut->negative_flag << std::endl;
            }
        }
        
        return pass;
    }
    
    // Test addition
    void test_addition() {
        std::cout << "\n=== Testing Addition ===" << std::endl;
        
        // 0 + 0 = 0
        uint64_t zero = decimal_to_pentary(0);
        run_test("ADD: 0 + 0 = 0", zero, zero, OP_ADD, zero, true, true, false);
        
        // 1 + 1 = 2
        uint64_t one = decimal_to_pentary(1);
        uint64_t two = decimal_to_pentary(2);
        run_test("ADD: 1 + 1 = 2", one, one, OP_ADD, two);
        
        // 2 + 3 = 5
        uint64_t three = decimal_to_pentary(3);
        uint64_t five = decimal_to_pentary(5);
        run_test("ADD: 2 + 3 = 5", two, three, OP_ADD, five);
        
        // -1 + 1 = 0
        uint64_t neg_one = decimal_to_pentary(-1);
        run_test("ADD: -1 + 1 = 0", neg_one, one, OP_ADD, zero, true, true, false);
        
        // 10 + 15 = 25
        uint64_t ten = decimal_to_pentary(10);
        uint64_t fifteen = decimal_to_pentary(15);
        uint64_t twentyfive = decimal_to_pentary(25);
        run_test("ADD: 10 + 15 = 25", ten, fifteen, OP_ADD, twentyfive);
        
        // -5 + (-3) = -8
        uint64_t neg_five = decimal_to_pentary(-5);
        uint64_t neg_three = decimal_to_pentary(-3);
        uint64_t neg_eight = decimal_to_pentary(-8);
        run_test("ADD: -5 + (-3) = -8", neg_five, neg_three, OP_ADD, neg_eight, true, false, true);
    }
    
    // Test subtraction
    void test_subtraction() {
        std::cout << "\n=== Testing Subtraction ===" << std::endl;
        
        uint64_t zero = decimal_to_pentary(0);
        uint64_t one = decimal_to_pentary(1);
        uint64_t two = decimal_to_pentary(2);
        uint64_t five = decimal_to_pentary(5);
        uint64_t three = decimal_to_pentary(3);
        
        // 5 - 3 = 2
        run_test("SUB: 5 - 3 = 2", five, three, OP_SUB, two);
        
        // 3 - 5 = -2
        uint64_t neg_two = decimal_to_pentary(-2);
        run_test("SUB: 3 - 5 = -2", three, five, OP_SUB, neg_two, true, false, true);
        
        // 10 - 10 = 0
        uint64_t ten = decimal_to_pentary(10);
        run_test("SUB: 10 - 10 = 0", ten, ten, OP_SUB, zero, true, true, false);
        
        // 0 - 1 = -1
        uint64_t neg_one = decimal_to_pentary(-1);
        run_test("SUB: 0 - 1 = -1", zero, one, OP_SUB, neg_one);
    }
    
    // Test negation
    void test_negation() {
        std::cout << "\n=== Testing Negation ===" << std::endl;
        
        uint64_t zero = decimal_to_pentary(0);
        uint64_t one = decimal_to_pentary(1);
        uint64_t neg_one = decimal_to_pentary(-1);
        uint64_t five = decimal_to_pentary(5);
        uint64_t neg_five = decimal_to_pentary(-5);
        
        // -0 = 0
        run_test("NEG: -0 = 0", zero, zero, OP_NEG, zero, true, true, false);
        
        // -1 = -1
        run_test("NEG: -(1) = -1", one, zero, OP_NEG, neg_one);
        
        // -(-1) = 1
        run_test("NEG: -(-1) = 1", neg_one, zero, OP_NEG, one);
        
        // -5 = -5
        run_test("NEG: -(5) = -5", five, zero, OP_NEG, neg_five);
        
        // -(-5) = 5
        run_test("NEG: -(-5) = 5", neg_five, zero, OP_NEG, five);
    }
    
    // Test absolute value
    void test_absolute() {
        std::cout << "\n=== Testing Absolute Value ===" << std::endl;
        
        uint64_t zero = decimal_to_pentary(0);
        uint64_t five = decimal_to_pentary(5);
        uint64_t neg_five = decimal_to_pentary(-5);
        
        // |0| = 0
        run_test("ABS: |0| = 0", zero, zero, OP_ABS, zero);
        
        // |5| = 5
        run_test("ABS: |5| = 5", five, zero, OP_ABS, five);
        
        // |-5| = 5
        run_test("ABS: |-5| = 5", neg_five, zero, OP_ABS, five);
    }
    
    // Test comparison
    void test_comparison() {
        std::cout << "\n=== Testing Comparison ===" << std::endl;
        
        uint64_t five = decimal_to_pentary(5);
        uint64_t three = decimal_to_pentary(3);
        uint64_t neg_three = decimal_to_pentary(-3);
        
        // Set up for comparison
        dut->operand_a = five;
        dut->operand_b = three;
        dut->opcode = OP_CMP;
        dut->eval();
        
        bool pass = (dut->greater_flag == 1) && (dut->equal_flag == 0);
        std::cout << (pass ? "[PASS]" : "[FAIL]") << " CMP: 5 > 3" << std::endl;
        if (pass) tests_passed++; else tests_failed++;
        
        // 3 < 5
        dut->operand_a = three;
        dut->operand_b = five;
        dut->eval();
        
        pass = (dut->greater_flag == 0) && (dut->equal_flag == 0);
        std::cout << (pass ? "[PASS]" : "[FAIL]") << " CMP: 3 < 5" << std::endl;
        if (pass) tests_passed++; else tests_failed++;
        
        // 5 == 5
        dut->operand_a = five;
        dut->operand_b = five;
        dut->eval();
        
        pass = (dut->equal_flag == 1);
        std::cout << (pass ? "[PASS]" : "[FAIL]") << " CMP: 5 == 5" << std::endl;
        if (pass) tests_passed++; else tests_failed++;
    }
    
    // Test maximum
    void test_maximum() {
        std::cout << "\n=== Testing Maximum ===" << std::endl;
        
        uint64_t five = decimal_to_pentary(5);
        uint64_t three = decimal_to_pentary(3);
        uint64_t neg_five = decimal_to_pentary(-5);
        
        // max(5, 3) = 5
        run_test("MAX: max(5, 3) = 5", five, three, OP_MAX, five);
        
        // max(3, 5) = 5
        run_test("MAX: max(3, 5) = 5", three, five, OP_MAX, five);
        
        // max(-5, 3) = 3
        run_test("MAX: max(-5, 3) = 3", neg_five, three, OP_MAX, three);
        
        // max(5, 5) = 5
        run_test("MAX: max(5, 5) = 5", five, five, OP_MAX, five);
    }
    
    // Test shift operations
    void test_shift() {
        std::cout << "\n=== Testing Shift (Multiply/Divide by 5) ===" << std::endl;
        
        uint64_t one = decimal_to_pentary(1);
        uint64_t five = decimal_to_pentary(5);
        uint64_t twentyfive = decimal_to_pentary(25);
        
        // 1 << 1 = 5 (multiply by 5)
        run_test("MUL2: 1 * 5 = 5", one, decimal_to_pentary(0), OP_MUL2, five);
        
        // 5 << 1 = 25 (multiply by 5)
        run_test("MUL2: 5 * 5 = 25", five, decimal_to_pentary(0), OP_MUL2, twentyfive);
        
        // 25 >> 1 = 5 (divide by 5)
        run_test("DIV2: 25 / 5 = 5", twentyfive, decimal_to_pentary(0), OP_DIV2, five);
        
        // 5 >> 1 = 1 (divide by 5)
        run_test("DIV2: 5 / 5 = 1", five, decimal_to_pentary(0), OP_DIV2, one);
    }
    
    // Run all tests
    void run_all_tests() {
        std::cout << "========================================" << std::endl;
        std::cout << "  Pentary ALU Verilator Testbench" << std::endl;
        std::cout << "========================================" << std::endl;
        
        test_addition();
        test_subtraction();
        test_negation();
        test_absolute();
        test_comparison();
        test_maximum();
        test_shift();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Results: " << tests_passed << " passed, " 
                  << tests_failed << " failed" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    int get_exit_code() {
        return (tests_failed > 0) ? 1 : 0;
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    PentaryALUTestbench tb;
    tb.run_all_tests();
    
    return tb.get_exit_code();
}
