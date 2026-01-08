# Pentary Hardware Verilator Testbenches

This directory contains Verilator-based testbenches for the Pentary hardware modules. Verilator compiles Verilog to C++ for fast simulation.

## Prerequisites

- Verilator 4.0 or later
- GCC or Clang with C++14 support
- Make

### Installing Verilator

**Ubuntu/Debian:**
```bash
sudo apt-get install verilator
```

**macOS (Homebrew):**
```bash
brew install verilator
```

**From source:**
```bash
git clone https://github.com/verilator/verilator
cd verilator
autoconf
./configure
make -j$(nproc)
sudo make install
```

## Directory Structure

```
verilator/
├── Makefile               # Build system
├── README.md              # This file
├── tb_pentary_alu.cpp     # ALU testbench
├── tb_pentary_adder.cpp   # Adder testbench
├── tb_register_file.cpp   # Register file testbench
└── obj_dir/               # Build output (generated)
```

## Building and Running Tests

### Build All Testbenches
```bash
make all
```

### Run All Tests
```bash
make test
```

### Build and Run Individual Testbenches

**ALU Testbench:**
```bash
make tb_alu
make test_alu
```

**Adder Testbench:**
```bash
make tb_adder
make test_adder
```

**Register File Testbench:**
```bash
make tb_register
make test_register
```

### Lint Verilog Files
```bash
make lint
```

### Clean Build Artifacts
```bash
make clean
```

## Testbench Coverage

### ALU Testbench (`tb_pentary_alu.cpp`)

Tests all ALU operations:
- **Addition**: Basic addition, carry propagation, negative numbers
- **Subtraction**: Basic subtraction, borrowing, negative results
- **Negation**: Positive and negative number negation
- **Absolute Value**: Positive and negative inputs
- **Comparison**: Greater than, less than, equal
- **Maximum**: Max of two numbers
- **Shift Operations**: Multiply/divide by 5

### Adder Testbench (`tb_pentary_adder.cpp`)

Tests the 16-digit pentary adder:
- Zero handling
- Single digit addition
- Multi-digit addition with carry
- Negative number handling
- Large number addition

### Register File Testbench (`tb_register_file.cpp`)

Tests the 32-register file:
- Basic write/read operations
- Register 0 (always zero) behavior
- Dual read port simultaneous access
- All registers accessibility
- Reset functionality

## Adding New Testbenches

1. Create a new C++ testbench file:
   ```cpp
   #include <verilated.h>
   #include "VModuleName.h"
   
   int main(int argc, char** argv) {
       Verilated::commandArgs(argc, argv);
       VModuleName* dut = new VModuleName;
       
       // Your test code here
       
       delete dut;
       return 0;
   }
   ```

2. Add build rules to `Makefile`:
   ```makefile
   tb_new: $(OBJ_DIR)/VModuleName
   
   $(OBJ_DIR)/VModuleName: tb_new.cpp ../module.v
       $(VERILATOR) $(VERILATOR_FLAGS) \
           ../module.v \
           --exe tb_new.cpp \
           --top-module ModuleName
       $(MAKE) -C $(OBJ_DIR) -f VModuleName.mk
   ```

## Waveform Generation

Testbenches support VCD waveform generation. To enable:

1. Add `--trace` flag to VERILATOR_FLAGS (already included)
2. In your testbench, add:
   ```cpp
   #include <verilated_vcd_c.h>
   
   VerilatedVcdC* tfp = new VerilatedVcdC;
   dut->trace(tfp, 99);
   tfp->open("waveform.vcd");
   
   // After each eval():
   tfp->dump(time_step);
   
   // At end:
   tfp->close();
   ```

3. View with GTKWave:
   ```bash
   gtkwave waveform.vcd
   ```

## Pentary Number Encoding

The testbenches use 3-bit encoding for pentary digits:

| Digit | Binary | Decimal Equivalent |
|-------|--------|-------------------|
| -2    | 000    | ⊖                 |
| -1    | 001    | -                 |
|  0    | 010    | 0                 |
| +1    | 011    | +                 |
| +2    | 100    | ⊕                 |

A 16-digit pentary number uses 48 bits (16 × 3 bits).

## Troubleshooting

**"Module not found" error:**
- Ensure Verilog files exist in the parent directory
- Check module names match between Verilog and testbench

**Compilation errors:**
- Update Verilator to latest version
- Check C++ syntax in testbench

**Simulation mismatch:**
- Verify pentary encoding is consistent
- Check bit widths in Verilog modules

## Performance

Verilator simulations run significantly faster than interpretive simulators:

| Simulator | Relative Speed |
|-----------|----------------|
| Verilator | 1× (baseline)  |
| Icarus    | 10-100× slower |
| ModelSim  | 20-50× slower  |

This makes Verilator ideal for:
- Large regression suites
- Performance testing
- Coverage analysis
- CI/CD pipelines
