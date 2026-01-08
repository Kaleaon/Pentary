# Pentary FPGA Implementation Guide

## Overview

This guide covers implementing the Pentary processor on FPGA platforms, specifically targeting:
- **Lattice iCE40** (open-source toolchain)
- **Xilinx Artix-7** (Vivado toolchain)
- **Intel Cyclone V** (Quartus toolchain)

## FPGA Target Specifications

### Lattice iCE40UP5K (Entry Level)

| Resource | Available | Pentary Usage | Utilization |
|----------|-----------|---------------|-------------|
| LUT4     | 5,280     | ~2,000        | 38%         |
| DFF      | 5,280     | ~1,500        | 28%         |
| BRAM     | 30 (4Kb)  | 8             | 27%         |
| DSP      | 8         | 4             | 50%         |

**Achievable:** 4-trit ALU, 8 registers, basic instruction decoder

### Xilinx Artix-7 (Mid-Range)

| Resource | Available (A35T) | Pentary Usage | Utilization |
|----------|------------------|---------------|-------------|
| LUT6     | 33,280           | ~8,000        | 24%         |
| FF       | 41,600           | ~6,000        | 14%         |
| BRAM     | 50 (36Kb)        | 16            | 32%         |
| DSP48E1  | 90               | 16            | 18%         |

**Achievable:** Full 16-trit ALU, 32 registers, pipeline, cache

### Intel Cyclone V (Alternative)

| Resource | Available (5CEBA4) | Pentary Usage | Utilization |
|----------|-------------------|---------------|-------------|
| ALM      | 18,480            | ~4,000        | 22%         |
| FF       | 36,960            | ~6,000        | 16%         |
| M10K     | 308               | 32            | 10%         |
| DSP      | 66                | 16            | 24%         |

## Directory Structure

```
fpga/
├── common/
│   ├── pentary_pkg.sv        # SystemVerilog package with types
│   └── constraints/          # Pin constraint templates
├── lattice/
│   ├── pentary_ice40.v       # iCE40-optimized implementation
│   ├── pentary_ice40.pcf     # Pin constraints
│   └── Makefile              # Yosys/nextpnr build
├── xilinx/
│   ├── pentary_artix.v       # Artix-optimized implementation
│   ├── pentary_artix.xdc     # Timing/pin constraints
│   └── project.tcl           # Vivado project script
└── intel/
    ├── pentary_cyclone.v     # Cyclone-optimized implementation
    └── pentary_cyclone.qsf   # Quartus settings
```

## Synthesizable RTL Guidelines

### 1. Avoid Non-Synthesizable Constructs

```verilog
// BAD - Not synthesizable
initial begin
    for (int i = 0; i < 16; i++) begin
        regs[i] = 0;
    end
end

// GOOD - Synthesizable reset
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        for (int i = 0; i < 16; i++) begin
            regs[i] <= 48'b0;
        end
    end
end
```

### 2. Use Synchronous Design

```verilog
// BAD - Asynchronous logic
assign result = (sel) ? a : b;

// GOOD - Registered output
always_ff @(posedge clk) begin
    result <= (sel) ? a : b;
end
```

### 3. Explicit Reset

```verilog
// GOOD - Explicit synchronous reset
always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        counter <= 0;
    end else begin
        state <= next_state;
        counter <= counter + 1;
    end
end
```

### 4. Parameterized Modules

```verilog
module PentaryALU #(
    parameter DIGITS = 16,
    parameter DATA_WIDTH = DIGITS * 3
)(
    input  logic clk,
    input  logic rst,
    input  logic [DATA_WIDTH-1:0] a,
    input  logic [DATA_WIDTH-1:0] b,
    input  logic [2:0] op,
    output logic [DATA_WIDTH-1:0] result
);
```

## Implementation Files

### 1. Pentary Package (SystemVerilog)

```systemverilog
// File: pentary_pkg.sv
package pentary_pkg;
    
    // Pentary digit encoding (3 bits per digit)
    typedef enum logic [2:0] {
        PENT_NEG2 = 3'b000,  // -2
        PENT_NEG1 = 3'b001,  // -1
        PENT_ZERO = 3'b010,  //  0
        PENT_POS1 = 3'b011,  // +1
        PENT_POS2 = 3'b100   // +2
    } pentary_digit_t;
    
    // ALU operations
    typedef enum logic [2:0] {
        ALU_ADD  = 3'b000,
        ALU_SUB  = 3'b001,
        ALU_MUL2 = 3'b010,
        ALU_DIV2 = 3'b011,
        ALU_NEG  = 3'b100,
        ALU_ABS  = 3'b101,
        ALU_CMP  = 3'b110,
        ALU_MAX  = 3'b111
    } alu_op_t;
    
    // Number of digits in a pentary word
    parameter int PENTARY_DIGITS = 16;
    parameter int PENTARY_WIDTH = PENTARY_DIGITS * 3;  // 48 bits
    
    // Pentary word type
    typedef logic [PENTARY_WIDTH-1:0] pentary_word_t;
    
    // Function to get a single digit from a word
    function automatic pentary_digit_t get_digit(
        pentary_word_t word,
        int index
    );
        return pentary_digit_t'(word[index*3 +: 3]);
    endfunction
    
    // Function to set a single digit in a word
    function automatic pentary_word_t set_digit(
        pentary_word_t word,
        int index,
        pentary_digit_t digit
    );
        pentary_word_t result = word;
        result[index*3 +: 3] = digit;
        return result;
    endfunction
    
    // Function to negate a digit
    function automatic pentary_digit_t negate_digit(pentary_digit_t d);
        case (d)
            PENT_NEG2: return PENT_POS2;
            PENT_NEG1: return PENT_POS1;
            PENT_ZERO: return PENT_ZERO;
            PENT_POS1: return PENT_NEG1;
            PENT_POS2: return PENT_NEG2;
            default:   return PENT_ZERO;
        endcase
    endfunction
    
endpackage
```

### 2. Lattice iCE40 Implementation

See: `lattice/pentary_ice40.v`

### 3. Xilinx Artix-7 Implementation

See: `xilinx/pentary_artix.v`

## Build Instructions

### Lattice iCE40 (Open Source Toolchain)

**Prerequisites:**
```bash
# Install Yosys, nextpnr, icestorm
sudo apt-get install yosys nextpnr-ice40 fpga-icestorm
```

**Build:**
```bash
cd fpga/lattice
make synthesis    # Synthesize with Yosys
make pnr         # Place and route with nextpnr
make bitstream   # Generate bitstream
make program     # Program FPGA
```

### Xilinx Artix-7 (Vivado)

**Prerequisites:**
- Vivado 2020.2 or later (free WebPACK edition)

**Build:**
```bash
cd fpga/xilinx
vivado -mode batch -source project.tcl
```

Or in GUI:
1. Open Vivado
2. Run `source project.tcl` in Tcl console
3. Click "Generate Bitstream"

## Pin Constraints

### iCE40-HX8K Breakout Board

```
# File: pentary_ice40.pcf
# Clock
set_io clk J3

# Reset
set_io rst_n N10

# LEDs (status indicators)
set_io led[0] B5
set_io led[1] B4
set_io led[2] A2
set_io led[3] A1

# UART (for debug)
set_io uart_tx B12
set_io uart_rx B10

# SPI (for programming/data)
set_io spi_clk R11
set_io spi_mosi P11
set_io spi_miso P12
set_io spi_cs R12
```

### Xilinx Artix-7 (Arty Board)

```
# File: pentary_artix.xdc
# Clock (100 MHz)
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# Reset
set_property PACKAGE_PIN C2 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# LEDs
set_property PACKAGE_PIN H5 [get_ports {led[0]}]
set_property PACKAGE_PIN J5 [get_ports {led[1]}]
set_property PACKAGE_PIN T9 [get_ports {led[2]}]
set_property PACKAGE_PIN T10 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

# UART
set_property PACKAGE_PIN D10 [get_ports uart_tx]
set_property PACKAGE_PIN A9 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_*]
```

## Timing Constraints

### Clock Domain Crossing

```verilog
// Use synchronizers for signals crossing clock domains
module sync_ff #(
    parameter STAGES = 2
)(
    input  logic clk,
    input  logic async_in,
    output logic sync_out
);
    logic [STAGES-1:0] sync_reg;
    
    always_ff @(posedge clk) begin
        sync_reg <= {sync_reg[STAGES-2:0], async_in};
    end
    
    assign sync_out = sync_reg[STAGES-1];
endmodule
```

### False Path Constraints

```tcl
# Xilinx XDC
set_false_path -from [get_ports rst_n]
set_false_path -to [get_ports led[*]]
```

## Resource Optimization

### 1. Use Block RAM for Registers

```verilog
// Infer BRAM for register file
(* ram_style = "block" *)
reg [47:0] registers [0:31];
```

### 2. Use DSP Blocks for Multiplication

```verilog
// Hint to use DSP48
(* use_dsp = "yes" *)
wire [47:0] mult_result = a * b;
```

### 3. Pipeline Long Paths

```verilog
// Add pipeline registers to meet timing
always_ff @(posedge clk) begin
    // Stage 1
    operand_a_r1 <= operand_a;
    operand_b_r1 <= operand_b;
    
    // Stage 2
    intermediate_r2 <= operand_a_r1 + operand_b_r1;
    
    // Stage 3
    result <= post_process(intermediate_r2);
end
```

## Verification

### On-Chip Logic Analyzer

```tcl
# Xilinx ILA insertion
create_debug_core u_ila_0 ila
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
set_property ALL_PROBE_SAME_MU_CNT 1 [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets clk]
```

### Self-Test Module

```verilog
module self_test (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic pass,
    output logic fail,
    output logic done
);
    // Built-in self-test for ALU
    // Runs predefined test vectors
endmodule
```

## Performance Targets

| Metric | iCE40UP5K | Artix-7 A35T | Cyclone V |
|--------|-----------|--------------|-----------|
| Fmax   | 48 MHz    | 200 MHz      | 150 MHz   |
| MOPS   | 48        | 200          | 150       |
| Power  | 50 mW     | 500 mW       | 400 mW    |
| Cost   | $5        | $35          | $30       |

## Next Steps

1. **Simulation**: Verify RTL in simulation before synthesis
2. **Synthesis**: Run synthesis to check resource usage
3. **Timing**: Close timing at target frequency
4. **Verification**: Test on actual hardware
5. **Optimization**: Iterate to improve performance

## Resources

- [Project IceStorm](http://www.clifford.at/icestorm/)
- [Yosys Manual](https://yosyshq.readthedocs.io/)
- [Xilinx Vivado User Guide](https://docs.xilinx.com/r/en-US/ug893-vivado-ide)
- [Intel Quartus Handbook](https://www.intel.com/content/www/us/en/programmable/documentation/lit-index.html)
