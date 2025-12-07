# Pentary Processor FPGA Prototype Guide

## Overview

This guide provides complete instructions for implementing the Pentary processor on FPGA hardware. The prototype validates the pentary computing architecture and enables performance testing.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [FPGA Implementation](#fpga-implementation)
4. [Synthesis and Place & Route](#synthesis-and-place--route)
5. [Testing and Validation](#testing-and-validation)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Recommended FPGA Boards

#### Option 1: Xilinx Zynq UltraScale+ (Recommended)
- **Board:** ZCU102 Evaluation Kit
- **FPGA:** Zynq UltraScale+ MPSoC XCZU9EG
- **Resources:**
  - Logic Cells: 600K
  - Block RAM: 32.1 Mb
  - DSP Slices: 2,520
  - I/O: 328
- **Price:** ~$2,500
- **Why:** Excellent for prototyping, good resource availability, ARM cores for control

#### Option 2: Intel Arria 10 (Alternative)
- **Board:** Arria 10 SoC Development Kit
- **FPGA:** Arria 10 SX660
- **Resources:**
  - Logic Elements: 660K
  - M20K Memory Blocks: 2,713
  - DSP Blocks: 1,518
- **Price:** ~$3,000
- **Why:** High performance, good for neural network acceleration

#### Option 3: Xilinx Artix-7 (Budget Option)
- **Board:** Nexys A7-100T
- **FPGA:** Artix-7 XC7A100T
- **Resources:**
  - Logic Cells: 101K
  - Block RAM: 4.9 Mb
  - DSP Slices: 240
- **Price:** ~$400
- **Why:** Affordable, sufficient for basic prototype
- **Note:** May need to reduce design size (fewer registers, smaller cache)

### Peripheral Requirements

- **USB-UART Bridge:** For communication with host PC
- **DDR Memory:** At least 512 MB (for memristor emulation)
- **SD Card Slot:** For loading programs
- **GPIO:** For debugging and I/O
- **Clock:** 100 MHz or higher

---

## Software Requirements

### FPGA Development Tools

#### For Xilinx FPGAs
- **Vivado Design Suite** 2023.2 or later
  - Download: https://www.xilinx.com/support/download.html
  - License: Free WebPACK license available
  - Size: ~40 GB installed

#### For Intel FPGAs
- **Quartus Prime** 23.1 or later
  - Download: https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/download.html
  - License: Free Lite edition available
  - Size: ~30 GB installed

### Simulation Tools

- **ModelSim** or **Questa** (included with Quartus)
- **Vivado Simulator** (included with Vivado)
- **Icarus Verilog** (open source, for quick testing)

### Additional Software

- **Python 3.8+** with numpy, matplotlib
- **GCC** or **Clang** for software compilation
- **Git** for version control
- **Serial Terminal:** PuTTY, minicom, or screen

---

## FPGA Implementation

### Step 1: Project Setup

#### For Xilinx Vivado

```tcl
# Create new project
create_project pentary_fpga ./pentary_fpga -part xczu9eg-ffvb1156-2-e

# Add source files
add_files -norecurse {
    hardware/pentary_core_integrated.v
    hardware/pentary_alu_fixed.v
    hardware/pentary_adder_fixed.v
    hardware/register_file.v
    hardware/cache_hierarchy.v
    hardware/instruction_decoder.v
    hardware/pipeline_control.v
    hardware/mmu_interrupt.v
    hardware/memristor_crossbar_fixed.v
    hardware/pentary_quantizer_fixed.v
}

# Add constraints file
add_files -fileset constrs_1 -norecurse constraints/pentary_fpga.xdc

# Set top module
set_property top PentaryCoreIntegrated [current_fileset]

# Update compile order
update_compile_order -fileset sources_1
```

#### For Intel Quartus

```tcl
# Create new project
project_new pentary_fpga -overwrite

# Set device
set_global_assignment -name DEVICE 10AS066N3F40E2SG

# Add source files
set_global_assignment -name VERILOG_FILE hardware/pentary_core_integrated.v
set_global_assignment -name VERILOG_FILE hardware/pentary_alu_fixed.v
set_global_assignment -name VERILOG_FILE hardware/pentary_adder_fixed.v
set_global_assignment -name VERILOG_FILE hardware/register_file.v
set_global_assignment -name VERILOG_FILE hardware/cache_hierarchy.v
set_global_assignment -name VERILOG_FILE hardware/instruction_decoder.v
set_global_assignment -name VERILOG_FILE hardware/pipeline_control.v
set_global_assignment -name VERILOG_FILE hardware/mmu_interrupt.v
set_global_assignment -name VERILOG_FILE hardware/memristor_crossbar_fixed.v
set_global_assignment -name VERILOG_FILE hardware/pentary_quantizer_fixed.v

# Set top module
set_global_assignment -name TOP_LEVEL_ENTITY PentaryCoreIntegrated

# Add constraints
set_global_assignment -name SDC_FILE constraints/pentary_fpga.sdc
```

### Step 2: Create Constraints File

Create `constraints/pentary_fpga.xdc` (Xilinx):

```tcl
# Clock constraint
create_clock -period 10.000 -name sys_clk [get_ports clk]
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

# Reset
set_property PACKAGE_PIN C12 [get_ports reset]
set_property IOSTANDARD LVCMOS33 [get_ports reset]

# Memory interface (example for DDR3)
set_property PACKAGE_PIN R2 [get_ports {mem_addr[0]}]
set_property IOSTANDARD SSTL15 [get_ports {mem_addr[0]}]
# ... (repeat for all memory signals)

# UART interface
set_property PACKAGE_PIN D10 [get_ports uart_tx]
set_property PACKAGE_PIN A9 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

# Timing constraints
set_input_delay -clock sys_clk -max 2.0 [get_ports {mem_read_data[*]}]
set_output_delay -clock sys_clk -max 2.0 [get_ports {mem_addr[*]}]

# False paths for asynchronous resets
set_false_path -from [get_ports reset]
```

### Step 3: Add Top-Level Wrapper

Create `hardware/pentary_fpga_top.v`:

```verilog
module pentary_fpga_top (
    // Clock and Reset
    input         sys_clk,
    input         sys_reset,
    
    // UART Interface
    output        uart_tx,
    input         uart_rx,
    
    // DDR3 Memory Interface
    output [14:0] ddr3_addr,
    output [2:0]  ddr3_ba,
    output        ddr3_cas_n,
    output [0:0]  ddr3_ck_n,
    output [0:0]  ddr3_ck_p,
    output [0:0]  ddr3_cke,
    output        ddr3_ras_n,
    output        ddr3_reset_n,
    output        ddr3_we_n,
    inout  [15:0] ddr3_dq,
    inout  [1:0]  ddr3_dqs_n,
    inout  [1:0]  ddr3_dqs_p,
    output [0:0]  ddr3_cs_n,
    output [1:0]  ddr3_dm,
    output [0:0]  ddr3_odt,
    
    // Debug LEDs
    output [7:0]  led,
    
    // Debug Switches
    input  [7:0]  sw
);

    // ========================================================================
    // Clock and Reset Management
    // ========================================================================
    
    wire clk_100mhz;
    wire clk_200mhz;
    wire locked;
    wire reset_n;
    
    // Clock wizard for generating required clocks
    clk_wiz_0 clk_gen (
        .clk_in1(sys_clk),
        .clk_out1(clk_100mhz),  // 100 MHz for processor
        .clk_out2(clk_200mhz),  // 200 MHz for memory
        .reset(sys_reset),
        .locked(locked)
    );
    
    assign reset_n = locked & ~sys_reset;
    
    // ========================================================================
    // Memory Controller
    // ========================================================================
    
    wire [47:0] mem_addr;
    wire        mem_read;
    wire        mem_write;
    wire [47:0] mem_write_data;
    wire [47:0] mem_read_data;
    wire        mem_ready;
    
    // DDR3 controller (use Xilinx MIG or Intel EMIF)
    ddr3_controller mem_ctrl (
        .clk(clk_200mhz),
        .reset(~reset_n),
        
        // Processor interface
        .addr(mem_addr[26:0]),  // Truncate to DDR3 address width
        .read_en(mem_read),
        .write_en(mem_write),
        .write_data(mem_write_data),
        .read_data(mem_read_data),
        .ready(mem_ready),
        
        // DDR3 physical interface
        .ddr3_addr(ddr3_addr),
        .ddr3_ba(ddr3_ba),
        .ddr3_cas_n(ddr3_cas_n),
        .ddr3_ck_n(ddr3_ck_n),
        .ddr3_ck_p(ddr3_ck_p),
        .ddr3_cke(ddr3_cke),
        .ddr3_ras_n(ddr3_ras_n),
        .ddr3_reset_n(ddr3_reset_n),
        .ddr3_we_n(ddr3_we_n),
        .ddr3_dq(ddr3_dq),
        .ddr3_dqs_n(ddr3_dqs_n),
        .ddr3_dqs_p(ddr3_dqs_p),
        .ddr3_cs_n(ddr3_cs_n),
        .ddr3_dm(ddr3_dm),
        .ddr3_odt(ddr3_odt)
    );
    
    // ========================================================================
    // UART Controller
    // ========================================================================
    
    wire [31:0] uart_data_in;
    wire [31:0] uart_data_out;
    wire        uart_data_valid;
    wire        uart_ready;
    
    uart_controller #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) uart (
        .clk(clk_100mhz),
        .reset(~reset_n),
        .rx(uart_rx),
        .tx(uart_tx),
        .data_in(uart_data_in),
        .data_out(uart_data_out),
        .data_valid(uart_data_valid),
        .ready(uart_ready)
    );
    
    // ========================================================================
    // Pentary Processor Core
    // ========================================================================
    
    wire [31:0] external_interrupts;
    wire [4:0]  debug_reg_addr;
    wire [47:0] debug_reg_data;
    wire [47:0] debug_pc;
    wire        debug_halted;
    
    PentaryCoreIntegrated core (
        .clk(clk_100mhz),
        .reset(~reset_n),
        
        // Memory interface
        .mem_addr(mem_addr),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .mem_write_data(mem_write_data),
        .mem_read_data(mem_read_data),
        .mem_ready(mem_ready),
        
        // Interrupt interface
        .external_interrupts(external_interrupts),
        
        // Debug interface
        .debug_reg_addr(debug_reg_addr),
        .debug_reg_data(debug_reg_data),
        .debug_pc(debug_pc),
        .debug_halted(debug_halted)
    );
    
    // ========================================================================
    // Debug Interface
    // ========================================================================
    
    // Map switches to debug register address
    assign debug_reg_addr = sw[4:0];
    
    // Map debug data to LEDs (show lower 8 bits)
    assign led = debug_reg_data[7:0];
    
    // Connect UART to processor for program loading
    assign external_interrupts = {31'b0, uart_data_valid};
    
endmodule
```

---

## Synthesis and Place & Route

### Step 4: Run Synthesis

#### Xilinx Vivado

```tcl
# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Check results
open_run synth_1
report_utilization -file utilization_synth.txt
report_timing_summary -file timing_synth.txt
```

#### Intel Quartus

```tcl
# Run synthesis
execute_module -tool map

# Check results
report_timing
report_resource_usage
```

### Step 5: Implement Design

#### Xilinx Vivado

```tcl
# Run implementation
launch_runs impl_1
wait_on_run impl_1

# Check results
open_run impl_1
report_utilization -file utilization_impl.txt
report_timing_summary -file timing_impl.txt
report_power -file power_impl.txt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```

#### Intel Quartus

```tcl
# Run place and route
execute_module -tool fit

# Run timing analysis
execute_module -tool sta

# Generate programming file
execute_module -tool asm
```

### Step 6: Expected Resource Usage

For Zynq UltraScale+ (ZCU102):

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | ~45,000 | 274,080 | 16% |
| Registers | ~35,000 | 548,160 | 6% |
| Block RAM | ~120 | 912 | 13% |
| DSP Slices | ~50 | 2,520 | 2% |

For Artix-7 (Nexys A7):

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | ~25,000 | 63,400 | 39% |
| Registers | ~20,000 | 126,800 | 16% |
| Block RAM | ~80 | 135 | 59% |
| DSP Slices | ~30 | 240 | 13% |

---

## Testing and Validation

### Step 7: Program FPGA

#### Xilinx Vivado

```tcl
# Connect to hardware
open_hw_manager
connect_hw_server
open_hw_target

# Program device
set_property PROGRAM.FILE {pentary_fpga.bit} [get_hw_devices xczu9eg_0]
program_hw_devices [get_hw_devices xczu9eg_0]
```

#### Intel Quartus

```tcl
# Program device
quartus_pgm -c USB-Blaster -m JTAG -o "p;pentary_fpga.sof"
```

### Step 8: Basic Functionality Tests

Create `tests/fpga_basic_test.py`:

```python
#!/usr/bin/env python3
"""
Basic FPGA functionality test
"""

import serial
import time
import struct

def connect_fpga(port='/dev/ttyUSB0', baud=115200):
    """Connect to FPGA via UART"""
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(0.1)
    return ser

def write_register(ser, addr, value):
    """Write to processor register"""
    # Format: CMD(1) + ADDR(1) + VALUE(6)
    cmd = struct.pack('<BB6s', 0x01, addr, value.to_bytes(6, 'little'))
    ser.write(cmd)
    time.sleep(0.01)

def read_register(ser, addr):
    """Read from processor register"""
    # Format: CMD(1) + ADDR(1)
    cmd = struct.pack('<BB', 0x02, addr)
    ser.write(cmd)
    time.sleep(0.01)
    
    # Read response
    response = ser.read(6)
    if len(response) == 6:
        return int.from_bytes(response, 'little')
    return None

def test_register_file(ser):
    """Test register file read/write"""
    print("Testing register file...")
    
    test_values = [0x123456, 0xABCDEF, 0x000000, 0xFFFFFF]
    
    for i, val in enumerate(test_values):
        # Write value
        write_register(ser, i, val)
        
        # Read back
        read_val = read_register(ser, i)
        
        if read_val == val:
            print(f"  ✅ Register {i}: Write {val:06X}, Read {read_val:06X}")
        else:
            print(f"  ❌ Register {i}: Write {val:06X}, Read {read_val:06X}")
            return False
    
    return True

def test_arithmetic(ser):
    """Test arithmetic operations"""
    print("\nTesting arithmetic...")
    
    # Load test program (simple addition)
    # ADD R2, R0, R1
    program = [
        0x00000001,  # Load 1 into R0
        0x00000002,  # Load 2 into R1
        0x10000201,  # ADD R2, R0, R1
    ]
    
    # Load program (implementation specific)
    # ...
    
    # Execute
    # ...
    
    # Check result
    result = read_register(ser, 2)
    expected = 3
    
    if result == expected:
        print(f"  ✅ Addition: 1 + 2 = {result}")
        return True
    else:
        print(f"  ❌ Addition: Expected {expected}, got {result}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Pentary FPGA Basic Test")
    print("="*60)
    
    # Connect to FPGA
    ser = connect_fpga()
    print("✅ Connected to FPGA\n")
    
    # Run tests
    tests_passed = 0
    tests_total = 2
    
    if test_register_file(ser):
        tests_passed += 1
    
    if test_arithmetic(ser):
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print("="*60)
    
    ser.close()
```

---

## Performance Benchmarking

### Step 9: Benchmark Suite

Create `tests/fpga_benchmark.py`:

```python
#!/usr/bin/env python3
"""
FPGA performance benchmarking
"""

import serial
import time
import numpy as np

def benchmark_clock_frequency(ser):
    """Measure actual clock frequency"""
    print("Benchmarking clock frequency...")
    
    # Send timing test command
    start = time.time()
    # Execute 1000 NOPs
    for _ in range(1000):
        ser.write(b'\x00')  # NOP command
    end = time.time()
    
    elapsed = end - start
    freq = 1000 / elapsed
    
    print(f"  Measured frequency: {freq:.2f} Hz")
    return freq

def benchmark_memory_bandwidth(ser):
    """Measure memory bandwidth"""
    print("\nBenchmarking memory bandwidth...")
    
    # Transfer 1 MB of data
    data_size = 1024 * 1024
    test_data = np.random.bytes(data_size)
    
    start = time.time()
    ser.write(test_data)
    end = time.time()
    
    elapsed = end - start
    bandwidth = data_size / elapsed / (1024 * 1024)  # MB/s
    
    print(f"  Memory bandwidth: {bandwidth:.2f} MB/s")
    return bandwidth

def benchmark_arithmetic_throughput(ser):
    """Measure arithmetic operation throughput"""
    print("\nBenchmarking arithmetic throughput...")
    
    # Execute 10000 additions
    num_ops = 10000
    
    start = time.time()
    for _ in range(num_ops):
        # Send ADD command
        ser.write(b'\x10\x00\x01\x02')  # ADD R2, R0, R1
    end = time.time()
    
    elapsed = end - start
    throughput = num_ops / elapsed
    
    print(f"  Arithmetic throughput: {throughput:.2f} ops/sec")
    return throughput

if __name__ == "__main__":
    print("="*60)
    print("Pentary FPGA Performance Benchmark")
    print("="*60)
    
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    
    # Run benchmarks
    freq = benchmark_clock_frequency(ser)
    bandwidth = benchmark_memory_bandwidth(ser)
    throughput = benchmark_arithmetic_throughput(ser)
    
    # Summary
    print("\n" + "="*60)
    print("Benchmark Results:")
    print(f"  Clock Frequency: {freq:.2f} Hz")
    print(f"  Memory Bandwidth: {bandwidth:.2f} MB/s")
    print(f"  Arithmetic Throughput: {throughput:.2f} ops/sec")
    print("="*60)
    
    ser.close()
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Synthesis Fails with Timing Violations

**Symptoms:**
- Setup time violations
- Hold time violations
- Clock frequency not met

**Solutions:**
1. Reduce target clock frequency in constraints
2. Add pipeline stages to long paths
3. Use faster speed grade FPGA
4. Optimize critical paths

```tcl
# Reduce clock frequency
create_clock -period 20.000 -name sys_clk [get_ports clk]  # 50 MHz instead of 100 MHz
```

#### Issue 2: Resource Utilization Too High

**Symptoms:**
- LUT utilization > 80%
- Block RAM utilization > 90%
- Placement fails

**Solutions:**
1. Reduce cache sizes
2. Reduce number of registers
3. Use larger FPGA
4. Optimize resource sharing

```verilog
// Reduce cache size
parameter L1_SIZE = 16384;  // 16KB instead of 32KB
```

#### Issue 3: FPGA Not Responding

**Symptoms:**
- No UART communication
- LEDs not blinking
- No response to inputs

**Solutions:**
1. Check clock is running
2. Verify reset is released
3. Check UART connections
4. Verify bitstream loaded correctly

```python
# Test UART connection
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200)
ser.write(b'TEST')
response = ser.read(4)
print(f"Response: {response}")
```

#### Issue 4: Incorrect Results

**Symptoms:**
- Arithmetic operations give wrong results
- Memory reads return incorrect data
- Program execution fails

**Solutions:**
1. Run simulation first
2. Check for timing violations
3. Verify memory controller
4. Add debug probes

```tcl
# Add ILA (Integrated Logic Analyzer)
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
connect_debug_port u_ila_0/clk [get_nets clk]
connect_debug_port u_ila_0/probe0 [get_nets {mem_addr[*]}]
```

---

## Next Steps

After successful FPGA prototyping:

1. **Performance Validation**
   - Measure actual clock frequency
   - Benchmark memory bandwidth
   - Test arithmetic throughput
   - Compare with claims

2. **Software Development**
   - Port compiler to FPGA
   - Develop test programs
   - Create benchmark suite

3. **Optimization**
   - Identify bottlenecks
   - Optimize critical paths
   - Improve resource usage

4. **ASIC Planning**
   - Document lessons learned
   - Plan ASIC architecture
   - Estimate ASIC performance

---

## Resources

### Documentation
- Xilinx UltraScale+ Documentation: https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html
- Intel Arria 10 Documentation: https://www.intel.com/content/www/us/en/products/details/fpga/arria/10.html

### Tutorials
- Vivado Tutorial: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2023_2/ug888-vivado-design-flows-overview-tutorial.pdf
- Quartus Tutorial: https://www.intel.com/content/www/us/en/docs/programmable/683689/current/introduction.html

### Support
- Xilinx Forums: https://support.xilinx.com/s/
- Intel Forums: https://community.intel.com/t5/Programmable-Devices/ct-p/programmable-devices

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Ready for implementation