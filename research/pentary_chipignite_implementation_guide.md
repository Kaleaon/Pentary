# Pentary Processor Implementation Guide for chipIgnite

## Table of Contents
1. [Quick Start](#quick-start)
2. [Development Environment Setup](#development-environment-setup)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [OpenLane Synthesis Flow](#openlane-synthesis-flow)
5. [Testing and Verification](#testing-and-verification)
6. [Tape-out Submission](#tape-out-submission)
7. [Troubleshooting](#troubleshooting)

---

## 1. Quick Start

### 1.1 Prerequisites
- Linux environment (Ubuntu 20.04+ recommended)
- Docker installed
- Git
- Python 3.8+
- 50GB free disk space

### 1.2 One-Command Setup
```bash
# Clone Caravel user project template
git clone https://github.com/efabless/caravel_user_project.git pentary_chipignite
cd pentary_chipignite

# Install dependencies
make setup

# Copy pentary Verilog files
cp /path/to/pentary_chipignite_verilog_templates.v verilog/rtl/pentary_core.v
```

---

## 2. Development Environment Setup

### 2.1 Install Required Tools

**Docker (for OpenLane):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

**Caravel User Project Template:**
```bash
# Clone template
git clone https://github.com/efabless/caravel_user_project.git pentary_chipignite
cd pentary_chipignite

# Initialize submodules
export CARAVEL_ROOT=$(pwd)/caravel
export OPENLANE_ROOT=$(pwd)/dependencies/openlane_src
export PDK_ROOT=$(pwd)/dependencies/pdks
export PDK=sky130A

make setup
```

### 2.2 Directory Structure

After setup, your directory should look like:
```
pentary_chipignite/
├── caravel/                    # Caravel harness
├── dependencies/
│   ├── openlane_src/          # OpenLane flow
│   └── pdks/                  # Skywater PDK
├── verilog/
│   ├── dv/                    # Design verification testbenches
│   ├── rtl/                   # RTL source files
│   │   ├── user_project_wrapper.v
│   │   └── pentary_core.v     # Your pentary processor
│   └── gl/                    # Gate-level netlists (generated)
├── openlane/
│   └── user_project_wrapper/  # OpenLane configuration
├── gds/                       # Final GDS files (generated)
└── signoff/                   # Sign-off reports (generated)
```

---

## 3. Step-by-Step Implementation

### 3.1 Phase 1: RTL Development (Weeks 1-4)

**Step 1: Create Pentary Core Modules**

Create `verilog/rtl/pentary_core.v` with the following modules:
1. `pentary_digit_adder` - Single digit adder
2. `pentary_word_adder` - 20-digit word adder
3. `pentary_alu` - Complete ALU
4. `pentary_register_file` - 25 registers
5. `pentary_wishbone_interface` - Bus interface
6. `pentary_processor_core` - Top-level core

**Step 2: Create User Project Wrapper**

Edit `verilog/rtl/user_project_wrapper.v`:
```verilog
module user_project_wrapper (
    // Power pins
    `ifdef USE_POWER_PINS
        inout vdda1, vdda2, vssa1, vssa2,
        inout vccd1, vccd2, vssd1, vssd2,
    `endif
    
    // Wishbone interface
    input wb_clk_i,
    input wb_rst_i,
    input wbs_stb_i,
    input wbs_cyc_i,
    input wbs_we_i,
    input [3:0] wbs_sel_i,
    input [31:0] wbs_dat_i,
    input [31:0] wbs_adr_i,
    output wbs_ack_o,
    output [31:0] wbs_dat_o,
    
    // Logic Analyzer
    input  [127:0] la_data_in,
    output [127:0] la_data_out,
    input  [127:0] la_oenb,
    
    // IOs
    input  [`MPRJ_IO_PADS-1:0] io_in,
    output [`MPRJ_IO_PADS-1:0] io_out,
    output [`MPRJ_IO_PADS-1:0] io_oeb,
    
    // IRQ
    output [2:0] irq
);

    // Instantiate pentary processor
    pentary_processor_core ppu (
        .clk(wb_clk_i),
        .rst_n(~wb_rst_i),
        .wb_clk_i(wb_clk_i),
        .wb_rst_i(wb_rst_i),
        .wb_adr_i(wbs_adr_i),
        .wb_dat_i(wbs_dat_i),
        .wb_dat_o(wbs_dat_o),
        .wb_we_i(wbs_we_i),
        .wb_sel_i(wbs_sel_i),
        .wb_stb_i(wbs_stb_i),
        .wb_cyc_i(wbs_cyc_i),
        .wb_ack_o(wbs_ack_o),
        .gpio_out(io_out[37:0]),
        .gpio_in(io_in[37:0]),
        .gpio_oe(io_oeb[37:0]),
        .irq(irq[0])
    );
    
    // Tie off unused signals
    assign irq[2:1] = 2'b00;
    assign la_data_out = 128'd0;
    
endmodule
```

**Step 3: RTL Simulation**

Create testbench `verilog/dv/pentary_test/pentary_test.c`:
```c
#include <defs.h>
#include <stub.c>

#define PENTARY_BASE 0x30000000

void main() {
    // Configure IOs
    reg_mprj_io_0  = GPIO_MODE_MGMT_STD_OUTPUT;
    reg_mprj_io_37 = GPIO_MODE_MGMT_STD_OUTPUT;
    
    // Apply configuration
    reg_mprj_xfer = 1;
    while (reg_mprj_xfer == 1);
    
    // Test pentary processor
    // Write to pentary register
    reg_la0_data = 0x12345678;  // Test data
    reg_la0_oenb = 0x00000000;  // Enable output
    
    // Read from pentary register
    uint32_t result = reg_la1_data;
    
    // Signal test completion
    reg_mprj_datal = 0xAB600000;
}
```

Run simulation:
```bash
cd verilog/dv/pentary_test
make
```

### 3.2 Phase 2: Synthesis Configuration (Week 5)

**Step 1: Configure OpenLane**

Edit `openlane/user_project_wrapper/config.tcl`:
```tcl
set ::env(DESIGN_NAME) user_project_wrapper

set ::env(VERILOG_FILES) "\
    $::env(CARAVEL_ROOT)/verilog/rtl/defines.v \
    $script_dir/../../verilog/rtl/user_project_wrapper.v \
    $script_dir/../../verilog/rtl/pentary_core.v"

# Clock configuration
set ::env(CLOCK_PORT) "wb_clk_i"
set ::env(CLOCK_PERIOD) "20"  # 50 MHz

# Area configuration
set ::env(FP_SIZING) absolute
set ::env(DIE_AREA) "0 0 2920 3520"  # 10mm² user area

# Density
set ::env(PL_TARGET_DENSITY) 0.50
set ::env(CELL_PAD) 4

# Routing
set ::env(ROUTING_CORES) 8
set ::env(GLB_RT_MAXLAYER) 5

# Power
set ::env(VDD_NETS) "vccd1 vccd2 vdda1 vdda2"
set ::env(GND_NETS) "vssd1 vssd2 vssa1 vssa2"

# Standard cells
set ::env(SYNTH_STRATEGY) "AREA 0"
set ::env(SYNTH_MAX_FANOUT) 6
```

**Step 2: Pin Configuration**

Edit `openlane/user_project_wrapper/pin_order.cfg`:
```
#N
io_in\[0\]
io_in\[1\]
...
io_in\[37\]

#S
io_out\[0\]
io_out\[1\]
...
io_out\[37\]

#E
wbs_adr_i\[0\]
wbs_adr_i\[1\]
...
wbs_adr_i\[31\]

#W
wbs_dat_o\[0\]
wbs_dat_o\[1\]
...
wbs_dat_o\[31\]
```

### 3.3 Phase 3: Synthesis and Place-and-Route (Weeks 6-8)

**Step 1: Run OpenLane Flow**

```bash
# Start OpenLane
make mount

# Inside Docker container
./flow.tcl -design user_project_wrapper -tag pentary_v1
```

**Step 2: Monitor Progress**

OpenLane will execute these stages:
1. **Synthesis** (~10 minutes)
   - Converts RTL to gate-level netlist
   - Reports gate count and area
   
2. **Floorplanning** (~5 minutes)
   - Places macros and defines power grid
   
3. **Placement** (~20 minutes)
   - Places standard cells
   - Optimizes timing
   
4. **Clock Tree Synthesis** (~15 minutes)
   - Builds clock distribution network
   
5. **Routing** (~30 minutes)
   - Routes all nets
   - Fixes DRC violations
   
6. **Sign-off** (~10 minutes)
   - Final DRC/LVS checks
   - Generates GDS

**Step 3: Check Results**

```bash
# View reports
cat openlane/user_project_wrapper/runs/pentary_v1/reports/synthesis/1-synthesis.stat.rpt
cat openlane/user_project_wrapper/runs/pentary_v1/reports/signoff/sta.rpt

# View layout
klayout openlane/user_project_wrapper/runs/pentary_v1/results/final/gds/user_project_wrapper.gds
```

**Expected Results:**
- Gate count: ~207,000 gates
- Area: ~1.3 mm²
- Max frequency: 50+ MHz
- Setup slack: Positive
- Hold slack: Positive
- DRC violations: 0
- LVS errors: 0

### 3.4 Phase 4: Integration (Week 9)

**Step 1: Harden User Project**

```bash
# Harden the design
make user_project_wrapper

# This generates:
# - gds/user_project_wrapper.gds
# - lef/user_project_wrapper.lef
# - verilog/gl/user_project_wrapper.v
```

**Step 2: Integrate with Caravel**

```bash
# Run full Caravel integration
make ship

# This generates final GDS with:
# - Management core
# - User project (pentary processor)
# - I/O pads
# - Power distribution
```

---

## 4. OpenLane Synthesis Flow

### 4.1 Detailed Flow Steps

**1. Synthesis**
```tcl
# In OpenLane interactive mode
./flow.tcl -interactive

package require openlane 0.9
prep -design user_project_wrapper -tag pentary_v1

# Run synthesis
run_synthesis

# Check results
check_synthesis_failure
```

**2. Floorplanning**
```tcl
# Initialize floorplan
init_floorplan

# Place I/O pins
place_io

# Generate power distribution network
gen_pdn
```

**3. Placement**
```tcl
# Global placement
global_placement_or

# Detailed placement
detailed_placement

# Optimize placement
optimize_placement
```

**4. Clock Tree Synthesis**
```tcl
# Run CTS
run_cts

# Check clock skew
check_clock_skew
```

**5. Routing**
```tcl
# Global routing
global_routing

# Detailed routing
detailed_routing

# Fix antenna violations
run_antenna_check
```

**6. Sign-off**
```tcl
# Run static timing analysis
run_sta

# Run DRC
run_magic_drc

# Run LVS
run_lvs

# Generate final GDS
run_magic
```

### 4.2 Optimization Tips

**If timing fails:**
```tcl
# Increase clock period
set ::env(CLOCK_PERIOD) "25"  # 40 MHz instead of 50 MHz

# Reduce fanout
set ::env(SYNTH_MAX_FANOUT) 4

# Enable buffering
set ::env(SYNTH_BUFFERING) 1
```

**If area exceeds budget:**
```tcl
# Reduce cache size in pentary_core.v
# Change from 4KB to 2KB caches

# Increase density
set ::env(PL_TARGET_DENSITY) 0.60
```

**If DRC violations:**
```tcl
# Increase cell padding
set ::env(CELL_PAD) 6

# Reduce routing layers
set ::env(GLB_RT_MAXLAYER) 4
```

---

## 5. Testing and Verification

### 5.1 RTL Simulation

**Create comprehensive testbench:**

`verilog/dv/pentary_comprehensive/pentary_comprehensive_tb.v`:
```verilog
module pentary_comprehensive_tb;
    reg clk, rst_n;
    reg [31:0] wb_adr, wb_dat_i;
    wire [31:0] wb_dat_o;
    reg wb_we, wb_stb, wb_cyc;
    wire wb_ack;
    
    // Instantiate DUT
    user_project_wrapper uut (
        .wb_clk_i(clk),
        .wb_rst_i(~rst_n),
        .wbs_adr_i(wb_adr),
        .wbs_dat_i(wb_dat_i),
        .wbs_dat_o(wb_dat_o),
        .wbs_we_i(wb_we),
        .wbs_stb_i(wb_stb),
        .wbs_cyc_i(wb_cyc),
        .wbs_ack_o(wb_ack),
        // ... other signals
    );
    
    // Clock generation
    initial clk = 0;
    always #10 clk = ~clk;  // 50 MHz
    
    // Test sequence
    initial begin
        rst_n = 0;
        #100 rst_n = 1;
        
        // Test 1: Write to register
        wb_write(32'h30001000, 32'h12345678);
        
        // Test 2: Read from register
        wb_read(32'h30001000);
        
        // Test 3: Pentary addition
        wb_write(32'h30001004, 32'h00000001);  // R1 = 1
        wb_write(32'h30001008, 32'h00000002);  // R2 = 2
        wb_write(32'h30000000, 32'h00000001);  // Execute ADD R3, R1, R2
        wb_read(32'h3000100C);                 // Read R3 (should be 3)
        
        #1000 $finish;
    end
    
    // Wishbone write task
    task wb_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            wb_adr = addr;
            wb_dat_i = data;
            wb_we = 1;
            wb_stb = 1;
            wb_cyc = 1;
            @(posedge wb_ack);
            @(posedge clk);
            wb_stb = 0;
            wb_cyc = 0;
        end
    endtask
    
    // Wishbone read task
    task wb_read(input [31:0] addr);
        begin
            @(posedge clk);
            wb_adr = addr;
            wb_we = 0;
            wb_stb = 1;
            wb_cyc = 1;
            @(posedge wb_ack);
            $display("Read from 0x%h: 0x%h", addr, wb_dat_o);
            @(posedge clk);
            wb_stb = 0;
            wb_cyc = 0;
        end
    endtask
    
endmodule
```

**Run simulation:**
```bash
cd verilog/dv/pentary_comprehensive
make
```

### 5.2 Gate-Level Simulation

After synthesis, run gate-level simulation:
```bash
# Use gate-level netlist
make SIM=GL pentary_comprehensive
```

### 5.3 Formal Verification

**Install SymbiYosys:**
```bash
sudo apt install symbiyosys
```

**Create formal verification script:**

`formal/pentary_alu_formal.sby`:
```
[tasks]
prove

[options]
prove: mode prove
prove: depth 20

[engines]
prove: smtbmc z3

[script]
read -formal pentary_core.v
prep -top pentary_alu

[files]
../verilog/rtl/pentary_core.v
```

**Run formal verification:**
```bash
cd formal
sby -f pentary_alu_formal.sby prove
```

---

## 6. Tape-out Submission

### 6.1 Pre-submission Checklist

- [ ] All RTL simulations pass
- [ ] Gate-level simulations pass
- [ ] Timing analysis shows positive slack
- [ ] DRC violations = 0
- [ ] LVS clean
- [ ] Power analysis within limits
- [ ] Documentation complete
- [ ] README.md updated

### 6.2 Generate Submission Files

```bash
# Generate final GDS
make ship

# Compress for submission
make compress

# This creates: caravel_pentary.gds.gz
```

### 6.3 Submit to chipIgnite

1. **Create Account:**
   - Go to https://efabless.com/chipignite
   - Create account and project

2. **Upload Design:**
   - Upload `caravel_pentary.gds.gz`
   - Upload documentation
   - Fill out design questionnaire

3. **Design Review:**
   - Efabless reviews design
   - Address any issues
   - Get approval

4. **Fabrication:**
   - Design goes to fabrication
   - Wait 3-6 months
   - Receive chips!

### 6.4 Cost and Timeline

**Costs:**
- chipIgnite shuttle: $300 (subsidized)
- Packaging: $100
- Testing: $200
- **Total: ~$600**

**Timeline:**
- Design: 12 weeks
- Review: 2 weeks
- Fabrication: 12-24 weeks
- **Total: 26-38 weeks**

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue: Synthesis fails with "undefined module"**
```
Solution: Check that all modules are defined in pentary_core.v
Verify: grep "module" verilog/rtl/pentary_core.v
```

**Issue: Timing violations**
```
Solution 1: Increase clock period
set ::env(CLOCK_PERIOD) "25"

Solution 2: Pipeline critical paths
Add registers in long combinational paths

Solution 3: Reduce fanout
set ::env(SYNTH_MAX_FANOUT) 4
```

**Issue: DRC violations**
```
Solution: Check magic DRC report
cat openlane/user_project_wrapper/runs/pentary_v1/reports/magic/magic.drc

Common fixes:
- Increase cell padding
- Reduce routing density
- Fix metal spacing issues
```

**Issue: Area overflow**
```
Solution: Reduce design size
- Decrease cache size
- Remove non-essential features
- Increase density target
```

### 7.2 Debug Commands

**Check synthesis results:**
```bash
cat openlane/user_project_wrapper/runs/pentary_v1/reports/synthesis/1-synthesis.stat.rpt | grep "Number of cells"
```

**Check timing:**
```bash
cat openlane/user_project_wrapper/runs/pentary_v1/reports/signoff/sta.rpt | grep "slack"
```

**View layout:**
```bash
klayout -l sky130A.lyp openlane/user_project_wrapper/runs/pentary_v1/results/final/gds/user_project_wrapper.gds
```

**Check power:**
```bash
cat openlane/user_project_wrapper/runs/pentary_v1/reports/signoff/power.rpt
```

### 7.3 Getting Help

**Resources:**
- Efabless Slack: https://invite.skywater.tools/
- Caravel Docs: https://caravel-harness.readthedocs.io/
- OpenLane Docs: https://openlane.readthedocs.io/
- GitHub Issues: https://github.com/efabless/caravel_user_project/issues

**Community:**
- Join #caravel-users on Slack
- Ask questions on GitHub Discussions
- Check existing issues and solutions

---

## 8. Next Steps After Tape-out

### 8.1 Chip Testing

Once you receive chips:
1. **Basic Functionality:**
   - Power up chip
   - Load test program
   - Verify basic operations

2. **Performance Testing:**
   - Measure clock frequency
   - Test all instructions
   - Benchmark performance

3. **Characterization:**
   - Power consumption
   - Temperature range
   - Voltage margins

### 8.2 Software Development

**Develop toolchain:**
1. Assembler for pentary ISA
2. Simulator for development
3. Debugger with JTAG support
4. Example programs

**Create applications:**
1. Benchmark suite
2. Demo applications
3. Performance comparisons
4. Research papers

### 8.3 Future Improvements

**Version 2.0 features:**
1. Larger caches (16KB)
2. Hardware multiplier optimization
3. Branch prediction
4. Floating-point support
5. Vector extensions

---

## Appendix A: Quick Reference Commands

```bash
# Setup
git clone https://github.com/efabless/caravel_user_project.git pentary_chipignite
cd pentary_chipignite
make setup

# Simulation
cd verilog/dv/pentary_test
make

# Synthesis
make mount
./flow.tcl -design user_project_wrapper

# Integration
make user_project_wrapper
make ship

# Submission
make compress
```

## Appendix B: File Checklist

- [ ] `verilog/rtl/pentary_core.v` - Pentary processor RTL
- [ ] `verilog/rtl/user_project_wrapper.v` - Caravel wrapper
- [ ] `verilog/dv/pentary_test/` - Testbenches
- [ ] `openlane/user_project_wrapper/config.tcl` - Synthesis config
- [ ] `openlane/user_project_wrapper/pin_order.cfg` - Pin configuration
- [ ] `docs/` - Documentation
- [ ] `README.md` - Project description

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-06  
**Status:** Ready for Implementation