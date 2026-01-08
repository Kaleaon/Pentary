# Pentary Processor - Xilinx Artix-7 Implementation

## Overview

This directory contains the FPGA implementation of a Pentary processor for Xilinx Artix-7 devices, targeting the Digilent Basys3 development board.

## Features

- **16-digit pentary arithmetic** (48-bit internal representation)
- **Full ALU** with add, subtract, negate, shift operations
- **16 general-purpose registers**
- **7-segment display** output for pentary values
- **UART interface** for debugging (115200 baud)
- **LED status indicators**
- **Button-based control** for demo mode

## Target Hardware

| Board | FPGA Part | Clock | Resources |
|-------|-----------|-------|-----------|
| Basys3 | XC7A35T-1CPG236C | 100 MHz | ~2000 LUTs, ~800 FFs |
| Nexys A7-100T | XC7A100T-1CSG324C | 100 MHz | ~2000 LUTs, ~800 FFs |

## File Structure

```
xilinx/
├── pentary_artix7.v      # Main RTL design
├── basys3.xdc            # Constraints for Basys3 board
├── build_vivado.tcl      # Vivado build script
└── README.md             # This file
```

## Building

### Prerequisites

- Vivado Design Suite 2020.1 or later
- Basys3 or compatible board

### Using Vivado GUI

1. Open Vivado
2. Create new project targeting XC7A35T-1CPG236C
3. Add `pentary_artix7.v` as design source
4. Add `basys3.xdc` as constraints
5. Run Synthesis, Implementation, and Generate Bitstream

### Using TCL Script

```bash
# From this directory
vivado -mode batch -source build_vivado.tcl
```

### Using Command Line

```bash
# Create project directory
mkdir -p build && cd build

# Run Vivado in batch mode
vivado -mode batch -source ../build_vivado.tcl

# Alternatively, use non-project mode:
vivado -mode tcl << 'EOF'
read_verilog ../pentary_artix7.v
read_xdc ../basys3.xdc
synth_design -top pentary_artix7 -part xc7a35tcpg236-1
opt_design
place_design
route_design
write_bitstream -force pentary_artix7.bit
EOF
```

## Programming the FPGA

### Using Vivado Hardware Manager

1. Connect Basys3 via USB
2. Open Hardware Manager
3. Open Target → Auto Connect
4. Program Device → Select bitstream file

### Using Command Line

```bash
vivado -mode tcl << 'EOF'
open_hw_manager
connect_hw_server
open_hw_target
set_property PROGRAM.FILE {pentary_artix7.bit} [get_hw_devices xc7a35t_0]
program_hw_devices [get_hw_devices xc7a35t_0]
close_hw_manager
EOF
```

## Usage

### Demo Mode

The demo mode uses switches and buttons for interactive pentary arithmetic:

| Input | Function |
|-------|----------|
| SW[2:0] | First pentary digit (0-4 maps to pentary values) |
| SW[5:3] | Second pentary digit |
| BTN Center | Add switch value to accumulator |
| BTN Up | Subtract switch value from accumulator |
| BTN Down | Negate accumulator |
| BTN Left | Shift left (×5) |
| BTN Right | Clear accumulator |

### LED Indicators

| LED | Meaning |
|-----|---------|
| LED[2:0] | Current state machine state |
| LED[3] | Positive result |
| LED[4] | Zero result |
| LED[5] | Negative result |
| LED[6] | Overflow |
| LED[7-11] | Digit value indicators |

### 7-Segment Display

Shows the lower 4 pentary digits of the result:
- `-` : Digit = -2
- `n` : Digit = -1
- `0` : Digit = 0
- `+` : Digit = +1
- `P` : Digit = +2

### UART Interface

Connect via serial terminal at 115200 baud, 8N1.
Currently implements simple echo mode for testing.

## Resource Utilization (Estimated)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | ~2,000 | 20,800 | ~10% |
| Flip-Flops | ~800 | 41,600 | ~2% |
| Block RAM | 0 | 50 | 0% |
| DSP48E1 | 0 | 90 | 0% |

## Timing

- Target: 100 MHz (10 ns period)
- Expected WNS: > 2 ns
- Critical path: ALU carry chain

## Customization

### Changing Number of Digits

Modify `PENT_WIDTH` parameter in `pentary_artix7.v`:
- 4 digits: `PENT_WIDTH = 12`
- 8 digits: `PENT_WIDTH = 24`
- 16 digits: `PENT_WIDTH = 48` (default)

### Adding DSP48 Multiplier

The ALU can be extended to use DSP48E1 blocks for efficient multiplication. Uncomment and implement the `ALU_MUL5` operation.

### Increasing Register Count

Modify `NUM_REGS` parameter. Ensure adequate resources for larger register files.

## Troubleshooting

### Synthesis Errors

**"Cannot find port X"**: Ensure constraints file matches top module port names.

**"Timing not met"**: 
- Reduce clock frequency
- Add pipeline stages to ALU
- Check for long combinational paths

### Programming Issues

**"No device found"**: 
- Check USB connection
- Install Digilent USB drivers
- Try different USB port

**"Configuration failed"**:
- Check FPGA jumper settings (JP1 for JTAG mode)
- Power cycle the board

## License

This implementation is provided for educational and research purposes. See main project LICENSE file for details.
