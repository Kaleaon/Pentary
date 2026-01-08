# Pentary Tiny Tapeout - Cocotb Testbench

## Overview

This directory contains the cocotb-based verification testbench for the Pentary 3T processor Tiny Tapeout design.

## Test Coverage

The testbench validates:

- **Reset behavior**: Proper initialization after reset
- **ALU operations**: ADD, SUB, NEG, NOP
- **Carry propagation**: Multi-digit addition/subtraction
- **Flags**: Zero and negative flag generation
- **Edge cases**: Maximum positive/negative values
- **Stress testing**: Rapid consecutive operations

## Prerequisites

```bash
# Install cocotb
pip install cocotb cocotb-test

# Install simulator (one of the following)
# Icarus Verilog
apt install iverilog

# OR Verilator
apt install verilator
```

## Running Tests

### Quick Test

```bash
# Run core tests only
make test_quick
```

### Full Test Suite

```bash
# Run all tests with Icarus Verilog
make

# Run with Verilator
make SIM=verilator
```

### With Waveform Generation

```bash
# Generate VCD waveform
make waves

# View with GTKWave
gtkwave dump.vcd
```

## Test Files

| File | Description |
|------|-------------|
| `test_pentary.py` | Main testbench with all test cases |
| `Makefile` | Cocotb build configuration |

## Test Cases

| Test | Description |
|------|-------------|
| `test_reset` | Verifies reset initializes to zero |
| `test_nop` | Tests pass-through operation |
| `test_add_simple` | Basic addition (1 + 1 = 2) |
| `test_add_with_carry` | Addition with carry (2 + 2 = 4) |
| `test_sub_simple` | Basic subtraction (2 - 1 = 1) |
| `test_sub_negative_result` | Subtraction with negative result |
| `test_neg` | Negation operation |
| `test_zero_flag` | Zero flag verification |
| `test_all_digit_combinations` | Exhaustive digit tests |
| `test_multidigit_values` | Multi-digit operations |
| `test_rapid_operations` | Stress test with rapid operations |
| `test_max_positive` | Maximum positive value test |
| `test_max_negative` | Maximum negative value test |

## Pentary Encoding Reference

| Decimal | Encoding | Symbol |
|---------|----------|--------|
| -2 | 3'b000 | ⊖ |
| -1 | 3'b001 | - |
| 0 | 3'b010 | 0 |
| +1 | 3'b011 | + |
| +2 | 3'b100 | ⊕ |

## Input/Output Mapping

### Inputs (ui_in, uio_in)

```
ui_in[2:0]  = Pentary digit A, position 0
ui_in[5:3]  = Pentary digit B, position 0
ui_in[7:6]  = Operation (00=ADD, 01=SUB, 10=NEG, 11=NOP)

uio_in[2:0] = Pentary digit A, position 1
uio_in[5:3] = Pentary digit B, position 1
```

### Outputs (uo_out)

```
uo_out[2:0] = Result digit 0
uo_out[5:3] = Result digit 1
uo_out[6]   = Zero flag
uo_out[7]   = Negative flag
```

## Results

Test results are written to `results.xml` in JUnit format for CI integration.

View summary:
```bash
make report
```

## Troubleshooting

### "cocotb-config: command not found"

Ensure cocotb is installed and in PATH:
```bash
pip install cocotb
export PATH="$PATH:$HOME/.local/bin"
```

### "iverilog: command not found"

Install Icarus Verilog:
```bash
# Ubuntu/Debian
sudo apt install iverilog

# macOS
brew install icarus-verilog
```

### Tests timing out

Increase timeout in test functions or reduce number of clock cycles.

## Contributing

When adding new tests:
1. Add test function with `@cocotb.test()` decorator
2. Document expected behavior
3. Update this README
4. Run full test suite to verify
