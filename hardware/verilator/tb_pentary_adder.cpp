/**
 * =============================================================================
 * Verilator Testbench for Pentary Adder
 * =============================================================================
 * 
 * Tests the 16-digit pentary adder module.
 * 
 * =============================================================================
 */

#include <verilated.h>
#include "VPentaryAdder16.h"

#include <iostream>
#include <iomanip>
#include <cstdint>

// Pentary digit encoding
#define PENT_NEG2   0b000
#define PENT_NEG1   0b001
#define PENT_ZERO   0b010
#define PENT_POS1   0b011
#define PENT_POS2   0b100

class PentaryAdderTestbench {
private:
    VPentaryAdder16* dut;
    int tests_passed;
    int tests_failed;
    
public:
    PentaryAdderTestbench() {
        dut = new VPentaryAdder16;
        tests_passed = 0;
        tests_failed = 0;
    }
    
    ~PentaryAdderTestbench() {
        delete dut;
    }
    
    // Convert decimal to pentary representation
    uint64_t decimal_to_pentary(int value) {
        uint64_t result = 0;
        bool negative = value < 0;
        if (negative) value = -value;
        
        for (int i = 0; i < 16 && value > 0; i++) {
            int digit = value % 5;
            value /= 5;
            
            uint64_t encoded;
            switch (digit) {
                case 0: encoded = PENT_ZERO; break;
                case 1: encoded = PENT_POS1; break;
                case 2: encoded = PENT_POS2; break;
                case 3: encoded = PENT_NEG2; value++; break;
                case 4: encoded = PENT_NEG1; value++; break;
                default: encoded = PENT_ZERO;
            }
            
            result |= (encoded << (i * 3));
        }
        
        // Fill remaining with zero
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
    
    bool run_test(const std::string& name, int a_val, int b_val) {
        uint64_t a = decimal_to_pentary(a_val);
        uint64_t b = decimal_to_pentary(b_val);
        int expected = a_val + b_val;
        
        dut->a = a;
        dut->b = b;
        dut->carry_in = PENT_ZERO;
        dut->eval();
        
        int result = pentary_to_decimal(dut->sum);
        
        bool pass = (result == expected);
        
        if (pass) {
            tests_passed++;
            std::cout << "[PASS] " << name << ": " << a_val << " + " << b_val 
                      << " = " << result << std::endl;
        } else {
            tests_failed++;
            std::cout << "[FAIL] " << name << ": " << a_val << " + " << b_val 
                      << " expected " << expected << " got " << result << std::endl;
        }
        
        return pass;
    }
    
    void run_all_tests() {
        std::cout << "========================================" << std::endl;
        std::cout << "  Pentary Adder Verilator Testbench" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::cout << "\n=== Basic Addition Tests ===" << std::endl;
        run_test("Zero + Zero", 0, 0);
        run_test("One + Zero", 1, 0);
        run_test("One + One", 1, 1);
        run_test("Two + Three", 2, 3);
        run_test("Five + Five", 5, 5);
        
        std::cout << "\n=== Carry Tests ===" << std::endl;
        run_test("Two + Two", 2, 2);
        run_test("Four + Four", 4, 4);
        run_test("12 + 13", 12, 13);
        run_test("100 + 100", 100, 100);
        
        std::cout << "\n=== Negative Number Tests ===" << std::endl;
        run_test("-1 + 1", -1, 1);
        run_test("-5 + 5", -5, 5);
        run_test("-3 + (-2)", -3, -2);
        run_test("10 + (-15)", 10, -15);
        
        std::cout << "\n=== Large Number Tests ===" << std::endl;
        run_test("1000 + 2000", 1000, 2000);
        run_test("-500 + 500", -500, 500);
        run_test("12345 + 67890", 12345, 67890);
        
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
    
    PentaryAdderTestbench tb;
    tb.run_all_tests();
    
    return tb.get_exit_code();
}
