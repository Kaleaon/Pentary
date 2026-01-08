# Pentary Tiny Tapeout Submission Guide

## Overview

This guide covers the complete flow for submitting the Pentary processor to Tiny Tapeout for fabrication using the SkyWater Sky130 open-source PDK.

## Tiny Tapeout Basics

### What is Tiny Tapeout?

Tiny Tapeout is a community project that enables hobbyists and educators to get their own chip designs manufactured. Each shuttle collects multiple designs and fabricates them on a shared die.

### Tile Options

| Size | Dimensions | Area | Cost (approx) |
|------|------------|------|---------------|
| 1×1  | 160×100 µm | 16,000 µm² | €140 |
| 1×2  | 160×225 µm | 36,000 µm² | €280 |
| 2×2  | 334×225 µm | 75,150 µm² | €560 |
| 4×2  | 334×450 µm | 150,300 µm² | €1,120 |

### Analog Pins

| Pins | Cost |
|------|------|
| 2    | €80  |
| 4    | €180 |
| 6    | €300 |

## Project Setup

### 1. Create Repository from Template

```bash
# Clone Tiny Tapeout template
git clone https://github.com/TinyTapeout/tt-verilog-template pentary-tt
cd pentary-tt

# Or use GitHub's template feature
# Go to: https://github.com/TinyTapeout/tt-verilog-template
# Click "Use this template"
```

### 2. Directory Structure

```
pentary-tt/
├── src/
│   ├── project.v              # Top-level module (required name)
│   ├── pentary_alu.v          # ALU module
│   ├── pentary_adder.v        # Adder module
│   └── pentary_registers.v    # Register file
├── test/
│   ├── test.py                # Cocotb testbench
│   └── tb.v                   # Verilog testbench (optional)
├── docs/
│   ├── info.md                # Project documentation (required)
│   └── pinout.md              # Pin descriptions
├── info.yaml                  # Project metadata (required)
├── Makefile                   # Build automation
└── README.md
```

### 3. Required Files

#### `info.yaml`

```yaml
project:
  title: "Pentary 3T Processor"
  author: "Your Name"
  description: "A balanced base-5 processor using 3-transistor analog logic"
  
  # Tile size: 1x1, 1x2, 2x2, or 4x2
  tiles: "2x2"
  
  # Analog pins (0, 2, 4, or 6)
  analog_pins: 6
  
  # Clock speed (for documentation)
  clock_hz: 10000000  # 10 MHz
  
  # External hardware required
  external_hw: "None"
  
  # Discord username for support
  discord: "your_discord_username"
  
  # Documentation language
  language: "en"
  
  # Top module name (must be tt_um_<project_name>)
  top_module: "tt_um_pentary_3t"
  
  # Source files
  source_files:
    - "project.v"
    - "pentary_alu.v"
    - "pentary_adder.v"
```

#### `src/project.v` (Top Module)

```verilog
/*
 * Pentary 3T Processor - Tiny Tapeout Top Module
 */

`default_nettype none

module tt_um_pentary_3t (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (1=output)
    input  wire [5:0] ua,       // Analog pins (directly active)
    input  wire       ena,      // Always 1 when design selected
    input  wire       clk,      // Clock
    input  wire       rst_n     // Active low reset
);

    // Internal signals
    wire rst = ~rst_n;
    
    // Pentary ALU
    wire [11:0] alu_a, alu_b, alu_result;
    wire [2:0] alu_op;
    wire alu_zero, alu_negative;
    
    // Input mapping
    // ui_in[2:0] = Pentary digit A (LSB)
    // ui_in[5:3] = Pentary digit B (LSB)
    // ui_in[7:6] = Operation select
    
    assign alu_a = {9'b010_010_010, ui_in[2:0]};  // Pad with zeros
    assign alu_b = {9'b010_010_010, ui_in[5:3]};
    assign alu_op = {1'b0, ui_in[7:6]};
    
    // ALU instance
    pentary_alu_4digit alu (
        .a(alu_a),
        .b(alu_b),
        .op(alu_op),
        .result(alu_result),
        .zero(alu_zero),
        .negative(alu_negative)
    );
    
    // Output mapping
    assign uo_out[2:0] = alu_result[2:0];  // Result digit 0
    assign uo_out[5:3] = alu_result[5:3];  // Result digit 1
    assign uo_out[6] = alu_zero;
    assign uo_out[7] = alu_negative;
    
    // Bidirectional pins as inputs
    assign uio_oe = 8'b0000_0000;  // All inputs
    assign uio_out = 8'b0;
    
    // Unused analog pins
    // ua[0]: Pentary analog input A
    // ua[1]: Pentary analog input B
    // ua[2]: Pentary analog output
    // ua[3]: Reference voltage
    // ua[4]: Bias current
    // ua[5]: Test output

endmodule
```

#### `docs/info.md`

```markdown
# Pentary 3T Processor

## Overview

This project implements a balanced base-5 (pentary) arithmetic processor
using analog 3-transistor logic for Tiny Tapeout.

## Features

- Pentary ALU (add, subtract, negate)
- 5-level analog signal processing
- UART debug interface

## Pinout

| Pin | Direction | Description |
|-----|-----------|-------------|
| ui_in[2:0] | Input | Pentary operand A |
| ui_in[5:3] | Input | Pentary operand B |
| ui_in[7:6] | Input | Operation select |
| uo_out[5:0] | Output | Pentary result |
| uo_out[6] | Output | Zero flag |
| uo_out[7] | Output | Negative flag |
| ua[0] | Analog In | Analog input A |
| ua[1] | Analog In | Analog input B |
| ua[2] | Analog Out | Analog result |

## Operation Codes

| Code | Operation |
|------|-----------|
| 00   | ADD       |
| 01   | SUB       |
| 10   | NEG       |
| 11   | NOP       |

## Usage

1. Apply reset (rst_n = 0, then 1)
2. Set operands on ui_in[5:0]
3. Set operation on ui_in[7:6]
4. Read result from uo_out

## External Hardware

No external hardware required for basic digital operation.
For analog mode, provide:
- 0.8V reference on ua[3]
- Bias current (10µA) on ua[4]
```

## GDS-II Generation Flow

### Step 1: Install OpenLane

```bash
# Clone OpenLane
git clone https://github.com/The-OpenROAD-Project/OpenLane.git
cd OpenLane

# Install dependencies
make pull-openlane

# Verify installation
make test
```

### Step 2: Configure Design

Create `config.json` in your design directory:

```json
{
    "DESIGN_NAME": "tt_um_pentary_3t",
    "VERILOG_FILES": "dir::src/*.v",
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 100,
    "FP_SIZING": "absolute",
    "DIE_AREA": "0 0 334 225",
    "FP_PDN_MULTILAYER": true,
    "SYNTH_STRATEGY": "AREA 0",
    "PL_TARGET_DENSITY": 0.5,
    "GRT_ALLOW_CONGESTION": true,
    "ROUTING_CORES": 4
}
```

### Step 3: Run OpenLane Flow

```bash
# Navigate to OpenLane
cd OpenLane

# Run flow
./flow.tcl -design /path/to/pentary-tt -tag run1

# Or interactive mode
./flow.tcl -interactive
% package require openlane
% prep -design /path/to/pentary-tt
% run_synthesis
% run_floorplan
% run_placement
% run_cts
% run_routing
% run_magic
% run_klayout
```

### Step 4: Output Files

After successful run, find outputs in:

```
OpenLane/designs/tt_um_pentary_3t/runs/run1/results/
├── synthesis/
│   └── tt_um_pentary_3t.v           # Synthesized netlist
├── placement/
│   └── tt_um_pentary_3t.def         # Placement DEF
├── routing/
│   └── tt_um_pentary_3t.def         # Routed DEF
├── signoff/
│   ├── tt_um_pentary_3t.gds         # Final GDS-II
│   ├── tt_um_pentary_3t.lef         # LEF file
│   └── tt_um_pentary_3t.spef        # Parasitics
└── reports/
    ├── synthesis/
    ├── timing/
    └── drc/
```

## Verification Checklist

### Pre-Submission

- [ ] Verilog lint clean (Verilator)
- [ ] Simulation passes (cocotb/iverilog)
- [ ] Synthesis completes
- [ ] No DRC errors
- [ ] No LVS errors
- [ ] Timing met
- [ ] Power within budget

### Required Checks

```bash
# Run Tiny Tapeout checks
make test          # Run testbench
make lint          # Verilog lint
make synth         # Synthesis
make gds           # Full flow
make verify        # DRC/LVS
```

## Submission Process

### 1. Push to GitHub

```bash
git add .
git commit -m "Ready for TT submission"
git push origin main
```

### 2. Enable GitHub Actions

GitHub Actions will automatically:
1. Run tests
2. Synthesize design
3. Generate GDS-II
4. Run DRC/LVS checks
5. Create artifact for submission

### 3. Submit to Tiny Tapeout

1. Go to [app.tinytapeout.com](https://app.tinytapeout.com)
2. Connect GitHub repository
3. Select shuttle (TT10, TT11, etc.)
4. Configure tile size and analog pins
5. Pay submission fee
6. Submit design

### 4. Post-Submission

- Monitor GitHub Actions for build status
- Check Tiny Tapeout dashboard
- Respond to any DRC/timing issues
- Receive confirmation email

## Timeline

| Phase | Duration |
|-------|----------|
| Design entry deadline | ~2 weeks before shuttle |
| DRC/LVS checks | 1-2 days |
| Tapeout | 1 day |
| Fabrication | 3-4 months |
| Packaging | 1 month |
| Shipping | 2-4 weeks |

## Troubleshooting

### Common Issues

**DRC Errors:**
- Metal density violations → Add fill
- Spacing violations → Reduce density
- Width violations → Check min widths

**LVS Errors:**
- Missing connections → Check power/ground
- Extra devices → Check for shorts
- Wrong pin names → Match netlist

**Timing Violations:**
- Setup violations → Reduce clock speed
- Hold violations → Add buffers
- Long paths → Pipeline design

### Getting Help

- Tiny Tapeout Discord: [discord.gg/tinytapeout](https://discord.gg/tinytapeout)
- GitHub Issues: Open issue on template repo
- Documentation: [tinytapeout.com/docs](https://tinytapeout.com/docs)

## Cost Summary

For Pentary 3T Processor (2×2 tiles + 6 analog pins):

| Item | Cost |
|------|------|
| 2×2 Tiles | €560 |
| 6 Analog Pins | €300 |
| **Total** | **€860** |

## Resources

- [Tiny Tapeout Documentation](https://tinytapeout.com/)
- [OpenLane Documentation](https://openlane.readthedocs.io/)
- [SkyWater PDK](https://skywater-pdk.readthedocs.io/)
- [Magic VLSI](http://opencircuitdesign.com/magic/)
- [KLayout](https://www.klayout.de/)

## License

This project is released under the Apache 2.0 License.
