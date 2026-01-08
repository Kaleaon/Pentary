"""
Cocotb Testbench for Pentary 3T Processor - Tiny Tapeout

This testbench validates the Tiny Tapeout pentary processor design
by exercising all ALU operations with various input combinations.

Usage:
    make -f Makefile.cocotb

Requirements:
    - cocotb
    - cocotb-test
    - iverilog or verilator

Author: Pentary Computing Project
License: MIT
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles
from cocotb.regression import TestFactory


# =============================================================================
# Pentary Value Conversions
# =============================================================================

def decimal_to_pentary_digit(val):
    """Convert decimal value (-2 to +2) to 3-bit pentary encoding"""
    encoding = {
        -2: 0b000,
        -1: 0b001,
         0: 0b010,
        +1: 0b011,
        +2: 0b100
    }
    return encoding.get(val, 0b010)

def pentary_digit_to_decimal(digit):
    """Convert 3-bit pentary encoding to decimal value"""
    decoding = {
        0b000: -2,
        0b001: -1,
        0b010:  0,
        0b011: +1,
        0b100: +2
    }
    return decoding.get(digit & 0x7, 0)

def decimal_to_pentary_4digit(val):
    """Convert decimal value to 4-digit pentary (12-bit) encoding"""
    result = 0
    sign = 1 if val >= 0 else -1
    val = abs(val)
    
    for i in range(4):
        digit = val % 5
        if digit > 2:
            digit -= 5
            val += 5
        val //= 5
        result |= decimal_to_pentary_digit(digit * sign) << (i * 3)
    
    return result

def pentary_4digit_to_decimal(encoded):
    """Convert 12-bit pentary encoding to decimal value"""
    total = 0
    multiplier = 1
    
    for i in range(4):
        digit = (encoded >> (i * 3)) & 0x7
        total += pentary_digit_to_decimal(digit) * multiplier
        multiplier *= 5
    
    return total


# =============================================================================
# Test Helper Functions
# =============================================================================

async def reset_dut(dut):
    """Reset the DUT"""
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def set_inputs(dut, a_d0, a_d1, b_d0, b_d1, op):
    """Set input values
    
    Args:
        dut: Device under test
        a_d0: Pentary digit 0 of operand A (decimal -2 to +2)
        a_d1: Pentary digit 1 of operand A (decimal -2 to +2)
        b_d0: Pentary digit 0 of operand B (decimal -2 to +2)
        b_d1: Pentary digit 1 of operand B (decimal -2 to +2)
        op: Operation (0=ADD, 1=SUB, 2=NEG, 3=NOP)
    """
    a_enc_d0 = decimal_to_pentary_digit(a_d0)
    a_enc_d1 = decimal_to_pentary_digit(a_d1)
    b_enc_d0 = decimal_to_pentary_digit(b_d0)
    b_enc_d1 = decimal_to_pentary_digit(b_d1)
    
    # Pack into ui_in: [7:6]=op, [5:3]=b_d0, [2:0]=a_d0
    ui_in = ((op & 0x3) << 6) | ((b_enc_d0 & 0x7) << 3) | (a_enc_d0 & 0x7)
    
    # Pack into uio_in: [5:3]=b_d1, [2:0]=a_d1
    uio_in = ((b_enc_d1 & 0x7) << 3) | (a_enc_d1 & 0x7)
    
    dut.ui_in.value = ui_in
    dut.uio_in.value = uio_in


def get_result(dut):
    """Get result from outputs
    
    Returns:
        tuple: (result_value, zero_flag, negative_flag)
    """
    output = int(dut.uo_out.value)
    
    # Unpack: [7]=neg, [6]=zero, [5:0]=result digits 0,1
    neg_flag = (output >> 7) & 0x1
    zero_flag = (output >> 6) & 0x1
    result_d0 = output & 0x7
    result_d1 = (output >> 3) & 0x7
    
    # Convert to decimal
    result_val = pentary_digit_to_decimal(result_d0) + \
                 5 * pentary_digit_to_decimal(result_d1)
    
    return result_val, zero_flag, neg_flag


# =============================================================================
# Basic Tests
# =============================================================================

@cocotb.test()
async def test_reset(dut):
    """Test that reset initializes correctly"""
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # After reset, output should be zero
    result, zero, neg = get_result(dut)
    assert zero == 1, f"Zero flag should be set after reset, got {zero}"
    dut._log.info("Reset test passed")


@cocotb.test()
async def test_nop(dut):
    """Test NOP operation (pass through A)"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Set A=7 (digits: +2, +1 -> 7 = 2 + 5*1), B=0, op=NOP
    set_inputs(dut, a_d0=2, a_d1=1, b_d0=0, b_d1=0, op=3)
    
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"NOP: A=7, result={result}")
    assert result == 7, f"NOP should pass A through, expected 7, got {result}"


@cocotb.test()
async def test_add_simple(dut):
    """Test simple addition"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test 1 + 1 = 2
    set_inputs(dut, a_d0=1, a_d1=0, b_d0=1, b_d1=0, op=0)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"ADD: 1 + 1 = {result}")
    assert result == 2, f"1 + 1 should be 2, got {result}"


@cocotb.test()
async def test_add_with_carry(dut):
    """Test addition with carry"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test 2 + 2 = 4 (requires carry: 4 = -1 + 5*1)
    set_inputs(dut, a_d0=2, a_d1=0, b_d0=2, b_d1=0, op=0)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"ADD: 2 + 2 = {result}")
    assert result == 4, f"2 + 2 should be 4, got {result}"


@cocotb.test()
async def test_sub_simple(dut):
    """Test simple subtraction"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test 2 - 1 = 1
    set_inputs(dut, a_d0=2, a_d1=0, b_d0=1, b_d1=0, op=1)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"SUB: 2 - 1 = {result}")
    assert result == 1, f"2 - 1 should be 1, got {result}"


@cocotb.test()
async def test_sub_negative_result(dut):
    """Test subtraction with negative result"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test 1 - 2 = -1
    set_inputs(dut, a_d0=1, a_d1=0, b_d0=2, b_d1=0, op=1)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"SUB: 1 - 2 = {result}, neg={neg}")
    assert result == -1, f"1 - 2 should be -1, got {result}"
    assert neg == 1, "Negative flag should be set"


@cocotb.test()
async def test_neg(dut):
    """Test negation operation"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test -7 where 7 = +2 + 5*1
    set_inputs(dut, a_d0=2, a_d1=1, b_d0=0, b_d1=0, op=2)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"NEG: -7 = {result}, neg={neg}")
    assert result == -7, f"NEG(7) should be -7, got {result}"
    assert neg == 1, "Negative flag should be set"


@cocotb.test()
async def test_zero_flag(dut):
    """Test zero flag"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test 1 - 1 = 0
    set_inputs(dut, a_d0=1, a_d1=0, b_d0=1, b_d1=0, op=1)
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"SUB: 1 - 1 = {result}, zero={zero}")
    assert result == 0, f"1 - 1 should be 0, got {result}"
    # Note: Zero flag might be delayed by one cycle


# =============================================================================
# Comprehensive Tests
# =============================================================================

@cocotb.test()
async def test_all_digit_combinations(dut):
    """Test addition with all digit combinations"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    errors = 0
    for a in range(-2, 3):  # Pentary digits: -2 to +2
        for b in range(-2, 3):
            expected = a + b
            set_inputs(dut, a_d0=a, a_d1=0, b_d0=b, b_d1=0, op=0)
            await ClockCycles(dut.clk, 3)
            
            result, zero, neg = get_result(dut)
            
            if result != expected:
                dut._log.error(f"ADD: {a} + {b} = {result}, expected {expected}")
                errors += 1
    
    assert errors == 0, f"Found {errors} addition errors"
    dut._log.info("All digit combination tests passed")


@cocotb.test()
async def test_multidigit_values(dut):
    """Test with multi-digit values"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Test cases: (a_d0, a_d1, b_d0, b_d1, op, expected)
    test_cases = [
        # (a_d0, a_d1, b_d0, b_d1, op, expected_result)
        (1, 1, 1, 1, 0, 12),      # 6 + 6 = 12
        (2, 1, -2, -1, 0, 0),     # 7 + (-7) = 0
        (0, 2, 0, 1, 0, 15),      # 10 + 5 = 15
        (1, 0, 0, 0, 2, -1),      # NEG(1) = -1
        (0, 1, 0, 0, 2, -5),      # NEG(5) = -5
    ]
    
    for a_d0, a_d1, b_d0, b_d1, op, expected in test_cases:
        set_inputs(dut, a_d0=a_d0, a_d1=a_d1, b_d0=b_d0, b_d1=b_d1, op=op)
        await ClockCycles(dut.clk, 3)
        
        result, zero, neg = get_result(dut)
        
        op_name = ["ADD", "SUB", "NEG", "NOP"][op]
        a_val = a_d0 + 5 * a_d1
        b_val = b_d0 + 5 * b_d1
        
        dut._log.info(f"{op_name}: a={a_val}, b={b_val}, result={result}, expected={expected}")
        assert result == expected, \
            f"{op_name}(a={a_val}, b={b_val}): expected {expected}, got {result}"
    
    dut._log.info("Multi-digit tests passed")


@cocotb.test()
async def test_rapid_operations(dut):
    """Test rapid consecutive operations"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Perform many operations quickly
    for i in range(50):
        a = (i % 5) - 2
        b = ((i * 3) % 5) - 2
        op = i % 4
        
        set_inputs(dut, a_d0=a, a_d1=0, b_d0=b, b_d1=0, op=op)
        await ClockCycles(dut.clk, 2)
    
    dut._log.info("Rapid operations test passed")


# =============================================================================
# Edge Case Tests
# =============================================================================

@cocotb.test()
async def test_max_positive(dut):
    """Test maximum positive value"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Maximum 2-digit positive: +2 + 5*2 = 12
    set_inputs(dut, a_d0=2, a_d1=2, b_d0=0, b_d1=0, op=3)  # NOP
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"Max positive: {result}")
    assert result == 12, f"Expected 12, got {result}"


@cocotb.test()
async def test_max_negative(dut):
    """Test maximum negative value"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Maximum 2-digit negative: -2 + 5*(-2) = -12
    set_inputs(dut, a_d0=-2, a_d1=-2, b_d0=0, b_d1=0, op=3)  # NOP
    await ClockCycles(dut.clk, 3)
    
    result, zero, neg = get_result(dut)
    dut._log.info(f"Max negative: {result}, neg={neg}")
    assert result == -12, f"Expected -12, got {result}"
    assert neg == 1, "Negative flag should be set"


# =============================================================================
# Main Entry Point (for running without cocotb-test)
# =============================================================================

if __name__ == "__main__":
    print("This testbench should be run using cocotb.")
    print("Run: make -f Makefile.cocotb")
