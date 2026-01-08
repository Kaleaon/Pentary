/**
 * =============================================================================
 * Verilator Testbench for Pentary Register File
 * =============================================================================
 * 
 * Tests the 32-register pentary register file with dual read ports.
 * 
 * =============================================================================
 */

#include <verilated.h>
#include "VRegisterFile.h"

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>

// Pentary digit encoding
#define PENT_NEG2   0b000
#define PENT_NEG1   0b001
#define PENT_ZERO   0b010
#define PENT_POS1   0b011
#define PENT_POS2   0b100

class RegisterFileTestbench {
private:
    VRegisterFile* dut;
    int tests_passed;
    int tests_failed;
    uint64_t time_step;
    
public:
    RegisterFileTestbench() {
        dut = new VRegisterFile;
        tests_passed = 0;
        tests_failed = 0;
        time_step = 0;
    }
    
    ~RegisterFileTestbench() {
        delete dut;
    }
    
    // Clock tick
    void tick() {
        dut->clk = 0;
        dut->eval();
        time_step++;
        
        dut->clk = 1;
        dut->eval();
        time_step++;
    }
    
    // Reset the DUT
    void reset() {
        dut->rst = 1;
        dut->write_enable = 0;
        tick();
        tick();
        dut->rst = 0;
        tick();
    }
    
    // Write to a register
    void write_register(uint8_t reg_addr, uint64_t value) {
        dut->write_addr = reg_addr;
        dut->write_data = value;
        dut->write_enable = 1;
        tick();
        dut->write_enable = 0;
    }
    
    // Read from register port A
    uint64_t read_register_a(uint8_t reg_addr) {
        dut->read_addr_a = reg_addr;
        dut->eval();
        return dut->read_data_a;
    }
    
    // Read from register port B
    uint64_t read_register_b(uint8_t reg_addr) {
        dut->read_addr_b = reg_addr;
        dut->eval();
        return dut->read_data_b;
    }
    
    // Convert decimal to pentary
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
        
        for (int i = 0; i < 16; i++) {
            if (((result >> (i * 3)) & 0x7) == 0) {
                result |= (uint64_t(PENT_ZERO) << (i * 3));
            }
        }
        
        if (negative) {
            uint64_t negated = 0;
            for (int i = 0; i < 16; i++) {
                uint64_t digit = (result >> (i * 3)) & 0x7;
                uint64_t neg;
                switch (digit) {
                    case PENT_NEG2: neg = PENT_POS2; break;
                    case PENT_NEG1: neg = PENT_POS1; break;
                    case PENT_ZERO: neg = PENT_ZERO; break;
                    case PENT_POS1: neg = PENT_NEG1; break;
                    case PENT_POS2: neg = PENT_NEG2; break;
                    default: neg = PENT_ZERO;
                }
                negated |= (neg << (i * 3));
            }
            result = negated;
        }
        
        return result;
    }
    
    bool run_test(const std::string& name, bool pass) {
        if (pass) {
            tests_passed++;
            std::cout << "[PASS] " << name << std::endl;
        } else {
            tests_failed++;
            std::cout << "[FAIL] " << name << std::endl;
        }
        return pass;
    }
    
    void test_basic_write_read() {
        std::cout << "\n=== Basic Write/Read Tests ===" << std::endl;
        
        reset();
        
        // Write to register 1
        uint64_t test_val = decimal_to_pentary(42);
        write_register(1, test_val);
        
        // Read back from register 1
        uint64_t read_val = read_register_a(1);
        run_test("Write/Read R1", read_val == test_val);
        
        // Write to register 15
        test_val = decimal_to_pentary(100);
        write_register(15, test_val);
        read_val = read_register_a(15);
        run_test("Write/Read R15", read_val == test_val);
        
        // Write to register 31 (last register)
        test_val = decimal_to_pentary(-55);
        write_register(31, test_val);
        read_val = read_register_a(31);
        run_test("Write/Read R31", read_val == test_val);
    }
    
    void test_register_zero() {
        std::cout << "\n=== Register Zero Tests ===" << std::endl;
        
        reset();
        
        // R0 should always be zero
        uint64_t zero = decimal_to_pentary(0);
        uint64_t read_val = read_register_a(0);
        run_test("R0 is zero initially", read_val == zero);
        
        // Try to write to R0 - should have no effect
        write_register(0, decimal_to_pentary(999));
        read_val = read_register_a(0);
        run_test("R0 cannot be modified", read_val == zero);
    }
    
    void test_dual_read_ports() {
        std::cout << "\n=== Dual Read Port Tests ===" << std::endl;
        
        reset();
        
        // Write different values to R1 and R2
        uint64_t val1 = decimal_to_pentary(123);
        uint64_t val2 = decimal_to_pentary(456);
        write_register(1, val1);
        write_register(2, val2);
        
        // Read both simultaneously
        dut->read_addr_a = 1;
        dut->read_addr_b = 2;
        dut->eval();
        
        bool pass = (dut->read_data_a == val1) && (dut->read_data_b == val2);
        run_test("Simultaneous read from both ports", pass);
    }
    
    void test_all_registers() {
        std::cout << "\n=== All Registers Test ===" << std::endl;
        
        reset();
        
        // Write unique value to each register
        for (int i = 1; i < 32; i++) {
            uint64_t val = decimal_to_pentary(i * 10);
            write_register(i, val);
        }
        
        // Verify all registers
        bool all_pass = true;
        for (int i = 1; i < 32; i++) {
            uint64_t expected = decimal_to_pentary(i * 10);
            uint64_t actual = read_register_a(i);
            if (actual != expected) {
                all_pass = false;
                std::cout << "  R" << i << " mismatch!" << std::endl;
            }
        }
        
        run_test("All registers written and read correctly", all_pass);
    }
    
    void test_reset_clears_registers() {
        std::cout << "\n=== Reset Test ===" << std::endl;
        
        // Write values
        write_register(5, decimal_to_pentary(999));
        write_register(10, decimal_to_pentary(888));
        
        // Reset
        reset();
        
        // Check that registers are cleared
        uint64_t zero = decimal_to_pentary(0);
        bool pass = (read_register_a(5) == zero) && (read_register_a(10) == zero);
        run_test("Reset clears registers", pass);
    }
    
    void run_all_tests() {
        std::cout << "========================================" << std::endl;
        std::cout << "  Register File Verilator Testbench" << std::endl;
        std::cout << "========================================" << std::endl;
        
        test_basic_write_read();
        test_register_zero();
        test_dual_read_ports();
        test_all_registers();
        test_reset_clears_registers();
        
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
    
    RegisterFileTestbench tb;
    tb.run_all_tests();
    
    return tb.get_exit_code();
}
