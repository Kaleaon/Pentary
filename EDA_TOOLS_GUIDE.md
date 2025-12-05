# EDA Tools Guide for Pentary Chip Design

## Overview

This guide provides detailed information about Electronic Design Automation (EDA) tools and workflows specifically tailored for pentary chip design. It covers tool selection, configuration, best practices, and pentary-specific optimizations.

---

## Table of Contents

1. [Tool Selection & Licensing](#1-tool-selection--licensing)
2. [Design Entry & RTL Development](#2-design-entry--rtl-development)
3. [Simulation & Verification](#3-simulation--verification)
4. [Synthesis & Optimization](#4-synthesis--optimization)
5. [Physical Design](#5-physical-design)
6. [Timing & Power Analysis](#6-timing--power-analysis)
7. [Pentary-Specific Tool Configurations](#7-pentary-specific-tool-configurations)
8. [Automation & Scripting](#8-automation--scripting)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Tool Selection & Licensing

### 1.1 Complete EDA Tool Suite

#### Tier 1: Industry Standard (Recommended)

**Synopsys Suite**
```
Design & Verification:
├─ VCS (Verilog Compiler Simulator)      - $150K/year
├─ Verdi (Debug & Analysis)              - $80K/year
├─ VC Formal (Formal Verification)       - $100K/year
├─ Design Compiler (Synthesis)           - $200K/year
├─ IC Compiler II (Place & Route)        - $250K/year
├─ PrimeTime (Timing Analysis)           - $150K/year
├─ PrimePower (Power Analysis)           - $100K/year
└─ Custom Compiler (Layout)              - $150K/year

Total: ~$1.2M/year (Academic discounts available: 80-90% off)
```

**Cadence Suite**
```
Design & Verification:
├─ Xcelium (Simulator)                   - $150K/year
├─ JasperGold (Formal Verification)      - $120K/year
├─ Genus (Synthesis)                     - $200K/year
├─ Innovus (Place & Route)               - $250K/year
├─ Tempus (Timing Analysis)              - $150K/year
├─ Voltus (Power Analysis)               - $100K/year
└─ Virtuoso (Custom Design)              - $200K/year

Total: ~$1.2M/year (Academic discounts available)
```

#### Tier 2: Open Source & Free Tools

**For Learning & Prototyping**
```
Open Source Tools:
├─ Icarus Verilog (Simulator)            - FREE
├─ Verilator (Fast Simulator)            - FREE
├─ GTKWave (Waveform Viewer)             - FREE
├─ Yosys (Synthesis)                     - FREE
├─ OpenROAD (Place & Route)              - FREE
└─ Magic (Layout)                        - FREE

Limitations:
- Less optimized results
- Limited support
- Fewer features
- Good for learning, not production
```

### 1.2 Licensing Strategies

#### Strategy 1: Academic Partnership

```bash
# Many universities have EDA tool licenses
# Partner with a university to access tools

Benefits:
- 80-90% discount on licenses
- Access to latest tools
- Technical support
- Training resources

Requirements:
- Academic affiliation
- Research collaboration
- Publication agreements
```

#### Strategy 2: Startup Program

```bash
# Most EDA vendors have startup programs

Synopsys Startup Program:
- Free tools for 1-2 years
- Technical support
- Training

Cadence Academic Network:
- Discounted licenses
- Cloud-based access
- Training materials

Requirements:
- Early-stage startup
- Innovative technology
- Growth potential
```

#### Strategy 3: Cloud-Based Access

```bash
# Use cloud-based EDA tools

AWS F1 Instances:
- Pre-installed EDA tools
- Pay-per-use pricing
- Scalable resources

Advantages:
- No upfront license cost
- Access to powerful hardware
- Flexible usage

Disadvantages:
- Ongoing costs
- Data security concerns
- Internet dependency
```

---

## 2. Design Entry & RTL Development

### 2.1 Recommended Editors & IDEs

#### Option 1: Visual Studio Code (Free)

**Installation:**
```bash
# Install VS Code
wget https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64
sudo dpkg -i code_*.deb

# Install Verilog extensions
code --install-extension mshr-h.veriloghdl
code --install-extension leafvmaple.verilog
```

**Configuration for Pentary:**
```json
// .vscode/settings.json
{
    "verilog.linting.linter": "verilator",
    "verilog.linting.verilator.arguments": "-Wall --lint-only",
    "verilog.formatting.verilogHDL.formatter": "verible-verilog-format",
    "files.associations": {
        "*.v": "verilog",
        "*.sv": "systemverilog"
    },
    "editor.tabSize": 4,
    "editor.insertSpaces": true
}
```

**Useful Extensions:**
- Verilog-HDL/SystemVerilog
- Verilog Formatter
- Waveform Viewer
- Git Integration

#### Option 2: Emacs with Verilog Mode

**Configuration:**
```elisp
;; .emacs configuration
(require 'verilog-mode)
(setq verilog-auto-newline nil)
(setq verilog-indent-level 4)
(setq verilog-indent-level-module 4)
(setq verilog-indent-level-declaration 4)
(setq verilog-indent-level-behavioral 4)

;; Pentary-specific snippets
(define-abbrev verilog-mode-abbrev-table "pentary"
  "typedef enum logic [2:0] {
    NEG2 = 3'b000,
    NEG1 = 3'b001,
    ZERO = 3'b010,
    POS1 = 3'b011,
    POS2 = 3'b100
} pentary_t;")
```

#### Option 3: Vim with Verilog Plugins

**Configuration:**
```vim
" .vimrc configuration
Plugin 'vhda/verilog_systemverilog.vim'
Plugin 'vim-scripts/verilog_emacsauto.vim'

" Pentary-specific settings
autocmd FileType verilog setlocal tabstop=4 shiftwidth=4 expandtab
autocmd FileType systemverilog setlocal tabstop=4 shiftwidth=4 expandtab

" Custom snippets for pentary
iabbrev pentary typedef enum logic [2:0] {<CR>NEG2 = 3'b000,<CR>NEG1 = 3'b001,<CR>ZERO = 3'b010,<CR>POS1 = 3'b011,<CR>POS2 = 3'b100<CR>} pentary_t;
```

### 2.2 Code Organization

#### Recommended Directory Structure

```
pentary_chip/
├── rtl/                          # RTL source code
│   ├── core/                     # Core modules
│   │   ├── pentary_alu.v
│   │   ├── register_file.v
│   │   ├── pipeline_control.v
│   │   └── memristor_crossbar.v
│   ├── memory/                   # Memory subsystem
│   │   ├── cache_controller.v
│   │   ├── l1_cache.v
│   │   └── l2_cache.v
│   ├── common/                   # Common modules
│   │   ├── pentary_adder.v
│   │   ├── pentary_types.vh
│   │   └── pentary_functions.vh
│   └── top/                      # Top-level
│       └── pentary_core.v
├── tb/                           # Testbenches
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── system/                   # System tests
├── syn/                          # Synthesis scripts
│   ├── constraints/
│   └── scripts/
├── pnr/                          # Place & Route
│   ├── floorplan/
│   └── scripts/
├── sim/                          # Simulation
│   ├── waves/
│   └── logs/
└── docs/                         # Documentation
    ├── specs/
    └── reports/
```

### 2.3 Coding Standards

#### Pentary-Specific Coding Style

```verilog
// pentary_types.vh - Define pentary types
`ifndef PENTARY_TYPES_VH
`define PENTARY_TYPES_VH

// Pentary digit encoding
typedef enum logic [2:0] {
    PENT_NEG2 = 3'b000,  // -2 (⊖)
    PENT_NEG1 = 3'b001,  // -1 (-)
    PENT_ZERO = 3'b010,  //  0
    PENT_POS1 = 3'b011,  // +1 (+)
    PENT_POS2 = 3'b100   // +2 (⊕)
} pentary_digit_t;

// Pentary word (16 digits = 48 bits)
typedef logic [47:0] pentary_word_t;

// Pentary carry (can be -1, 0, +1)
typedef enum logic [1:0] {
    CARRY_NEG = 2'b00,   // -1
    CARRY_ZERO = 2'b01,  //  0
    CARRY_POS = 2'b10    // +1
} pentary_carry_t;

`endif // PENTARY_TYPES_VH
```

#### Module Template

```verilog
//==============================================================================
// Module: pentary_alu
// Description: Pentary Arithmetic Logic Unit
//              Performs arithmetic and logic operations on pentary numbers
//
// Parameters:
//   - WORD_SIZE: Number of pentary digits (default: 16)
//
// Inputs:
//   - clk: Clock signal
//   - rst: Reset signal (active high)
//   - operand_a: First operand (pentary word)
//   - operand_b: Second operand (pentary word)
//   - opcode: Operation code
//
// Outputs:
//   - result: Operation result (pentary word)
//   - flags: Status flags (zero, negative, overflow)
//
// Author: [Your Name]
// Date: [Date]
// Version: 1.0
//==============================================================================

`include "pentary_types.vh"

module pentary_alu #(
    parameter WORD_SIZE = 16  // Number of pentary digits
) (
    // Clock and reset
    input  logic                    clk,
    input  logic                    rst,
    
    // Operands
    input  pentary_word_t           operand_a,
    input  pentary_word_t           operand_b,
    
    // Control
    input  logic [2:0]              opcode,
    
    // Results
    output pentary_word_t           result,
    output logic                    zero_flag,
    output logic                    negative_flag,
    output logic                    overflow_flag
);

    //--------------------------------------------------------------------------
    // Internal signals
    //--------------------------------------------------------------------------
    
    pentary_word_t add_result;
    pentary_word_t sub_result;
    pentary_word_t mul2_result;
    pentary_word_t neg_result;
    
    //--------------------------------------------------------------------------
    // Operation implementations
    //--------------------------------------------------------------------------
    
    // Addition
    pentary_adder #(
        .WORD_SIZE(WORD_SIZE)
    ) adder_inst (
        .a(operand_a),
        .b(operand_b),
        .sum(add_result),
        .carry_out()
    );
    
    // Subtraction (addition with negation)
    pentary_adder #(
        .WORD_SIZE(WORD_SIZE)
    ) subtractor_inst (
        .a(operand_a),
        .b(negate_pentary(operand_b)),
        .sum(sub_result),
        .carry_out()
    );
    
    // Multiply by 2 (shift left)
    assign mul2_result = shift_left_pentary(operand_a);
    
    // Negation
    assign neg_result = negate_pentary(operand_a);
    
    //--------------------------------------------------------------------------
    // Result multiplexer
    //--------------------------------------------------------------------------
    
    always_comb begin
        case (opcode)
            3'b000: result = add_result;   // ADD
            3'b001: result = sub_result;   // SUB
            3'b010: result = mul2_result;  // MUL2
            3'b011: result = neg_result;   // NEG
            default: result = '0;
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Flag generation
    //--------------------------------------------------------------------------
    
    assign zero_flag = (result == '0);
    assign negative_flag = is_negative_pentary(result);
    assign overflow_flag = detect_overflow(operand_a, operand_b, result, opcode);
    
    //--------------------------------------------------------------------------
    // Helper functions
    //--------------------------------------------------------------------------
    
    function automatic pentary_word_t negate_pentary(input pentary_word_t x);
        integer i;
        pentary_digit_t digit;
        for (i = 0; i < WORD_SIZE; i++) begin
            digit = pentary_digit_t'(x[i*3 +: 3]);
            case (digit)
                PENT_NEG2: negate_pentary[i*3 +: 3] = PENT_POS2;
                PENT_NEG1: negate_pentary[i*3 +: 3] = PENT_POS1;
                PENT_ZERO: negate_pentary[i*3 +: 3] = PENT_ZERO;
                PENT_POS1: negate_pentary[i*3 +: 3] = PENT_NEG1;
                PENT_POS2: negate_pentary[i*3 +: 3] = PENT_NEG2;
            endcase
        end
    endfunction
    
    function automatic pentary_word_t shift_left_pentary(input pentary_word_t x);
        // Shift left = multiply by 5 in pentary
        // Implementation details...
    endfunction
    
    function automatic logic is_negative_pentary(input pentary_word_t x);
        // Check most significant digit
        pentary_digit_t msd = pentary_digit_t'(x[(WORD_SIZE-1)*3 +: 3]);
        return (msd == PENT_NEG2 || msd == PENT_NEG1);
    endfunction
    
    function automatic logic detect_overflow(
        input pentary_word_t a,
        input pentary_word_t b,
        input pentary_word_t result,
        input logic [2:0] op
    );
        // Overflow detection logic
        // Implementation details...
    endfunction

endmodule
```

---

## 3. Simulation & Verification

### 3.1 Synopsys VCS Workflow

#### Basic Simulation Setup

```bash
#!/bin/bash
# simulate.sh - VCS simulation script

# Compile design
vcs -full64 \
    -sverilog \
    +v2k \
    -timescale=1ns/1ps \
    -debug_access+all \
    -lca \
    -kdb \
    +vcs+lic+wait \
    -f filelist.f \
    -top tb_pentary_core \
    -o simv

# Run simulation
./simv \
    +ntb_random_seed=12345 \
    +UVM_TESTNAME=pentary_basic_test \
    +UVM_VERBOSITY=UVM_MEDIUM \
    -l simulation.log

# Generate coverage report
urg -dir simv.vdb -report coverage_report
```

#### File List (filelist.f)

```
# RTL files
+incdir+../rtl/common
../rtl/common/pentary_types.vh
../rtl/common/pentary_functions.vh
../rtl/core/pentary_alu.v
../rtl/core/register_file.v
../rtl/core/pipeline_control.v
../rtl/core/memristor_crossbar.v
../rtl/memory/cache_controller.v
../rtl/top/pentary_core.v

# Testbench files
+incdir+../tb/common
../tb/common/test_utils.sv
../tb/unit/tb_pentary_alu.sv
../tb/integration/tb_pentary_core.sv
```

#### Advanced Simulation with Coverage

```bash
#!/bin/bash
# simulate_with_coverage.sh

# Compile with coverage
vcs -full64 \
    -sverilog \
    -cm line+cond+fsm+tgl+branch+assert \
    -cm_dir coverage.vdb \
    -cm_name pentary_test \
    -cm_hier coverage.cfg \
    -debug_access+all \
    -f filelist.f \
    -top tb_pentary_core \
    -o simv_cov

# Run simulation
./simv_cov \
    -cm line+cond+fsm+tgl+branch+assert \
    -cm_dir coverage.vdb \
    -cm_name pentary_test_run1 \
    -l simulation.log

# Generate coverage report
urg \
    -dir coverage.vdb \
    -format both \
    -report coverage_report \
    -show tests
```

### 3.2 Testbench Development

#### UVM Testbench Structure

```systemverilog
//==============================================================================
// UVM Testbench for Pentary ALU
//==============================================================================

`include "uvm_macros.svh"
import uvm_pkg::*;

//------------------------------------------------------------------------------
// Transaction Class
//------------------------------------------------------------------------------

class pentary_transaction extends uvm_sequence_item;
    `uvm_object_utils(pentary_transaction)
    
    rand pentary_word_t operand_a;
    rand pentary_word_t operand_b;
    rand logic [2:0]    opcode;
    
    pentary_word_t      result;
    logic               zero_flag;
    logic               negative_flag;
    logic               overflow_flag;
    
    // Constraints
    constraint valid_pentary {
        // Ensure valid pentary encoding
        foreach (operand_a[i]) {
            if (i % 3 == 0) {
                operand_a[i +: 3] inside {3'b000, 3'b001, 3'b010, 3'b011, 3'b100};
            }
        }
        foreach (operand_b[i]) {
            if (i % 3 == 0) {
                operand_b[i +: 3] inside {3'b000, 3'b001, 3'b010, 3'b011, 3'b100};
            }
        }
    }
    
    constraint valid_opcode {
        opcode inside {3'b000, 3'b001, 3'b010, 3'b011};
    }
    
    function new(string name = "pentary_transaction");
        super.new(name);
    endfunction
    
    function string convert2string();
        return $sformatf("op=%0d, a=%h, b=%h, result=%h, flags=%b%b%b",
                        opcode, operand_a, operand_b, result,
                        zero_flag, negative_flag, overflow_flag);
    endfunction
endclass

//------------------------------------------------------------------------------
// Sequence Class
//------------------------------------------------------------------------------

class pentary_sequence extends uvm_sequence #(pentary_transaction);
    `uvm_object_utils(pentary_sequence)
    
    function new(string name = "pentary_sequence");
        super.new(name);
    endfunction
    
    task body();
        pentary_transaction tx;
        
        repeat(100) begin
            tx = pentary_transaction::type_id::create("tx");
            start_item(tx);
            assert(tx.randomize());
            finish_item(tx);
        end
    endtask
endclass

//------------------------------------------------------------------------------
// Driver Class
//------------------------------------------------------------------------------

class pentary_driver extends uvm_driver #(pentary_transaction);
    `uvm_component_utils(pentary_driver)
    
    virtual pentary_if vif;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction
    
    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db#(virtual pentary_if)::get(this, "", "vif", vif))
            `uvm_fatal("NOVIF", "Virtual interface not found")
    endfunction
    
    task run_phase(uvm_phase phase);
        pentary_transaction tx;
        
        forever begin
            seq_item_port.get_next_item(tx);
            drive_transaction(tx);
            seq_item_port.item_done();
        end
    endtask
    
    task drive_transaction(pentary_transaction tx);
        @(posedge vif.clk);
        vif.operand_a <= tx.operand_a;
        vif.operand_b <= tx.operand_b;
        vif.opcode <= tx.opcode;
        @(posedge vif.clk);
    endtask
endclass

//------------------------------------------------------------------------------
// Monitor Class
//------------------------------------------------------------------------------

class pentary_monitor extends uvm_monitor;
    `uvm_component_utils(pentary_monitor)
    
    virtual pentary_if vif;
    uvm_analysis_port #(pentary_transaction) ap;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction
    
    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db#(virtual pentary_if)::get(this, "", "vif", vif))
            `uvm_fatal("NOVIF", "Virtual interface not found")
        ap = new("ap", this);
    endfunction
    
    task run_phase(uvm_phase phase);
        pentary_transaction tx;
        
        forever begin
            @(posedge vif.clk);
            tx = pentary_transaction::type_id::create("tx");
            tx.operand_a = vif.operand_a;
            tx.operand_b = vif.operand_b;
            tx.opcode = vif.opcode;
            @(posedge vif.clk);
            tx.result = vif.result;
            tx.zero_flag = vif.zero_flag;
            tx.negative_flag = vif.negative_flag;
            tx.overflow_flag = vif.overflow_flag;
            ap.write(tx);
        end
    endtask
endclass

//------------------------------------------------------------------------------
// Scoreboard Class
//------------------------------------------------------------------------------

class pentary_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(pentary_scoreboard)
    
    uvm_analysis_imp #(pentary_transaction, pentary_scoreboard) ap;
    
    int passed = 0;
    int failed = 0;
    
    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction
    
    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ap = new("ap", this);
    endfunction
    
    function void write(pentary_transaction tx);
        pentary_word_t expected_result;
        
        // Calculate expected result
        case (tx.opcode)
            3'b000: expected_result = pentary_add(tx.operand_a, tx.operand_b);
            3'b001: expected_result = pentary_sub(tx.operand_a, tx.operand_b);
            3'b010: expected_result = pentary_mul2(tx.operand_a);
            3'b011: expected_result = pentary_neg(tx.operand_a);
        endcase
        
        // Compare
        if (tx.result == expected_result) begin
            passed++;
            `uvm_info("PASS", tx.convert2string(), UVM_LOW)
        end else begin
            failed++;
            `uvm_error("FAIL", $sformatf("Expected %h, Got %h", expected_result, tx.result))
        end
    endfunction
    
    function void report_phase(uvm_phase phase);
        `uvm_info("REPORT", $sformatf("Passed: %0d, Failed: %0d", passed, failed), UVM_LOW)
    endfunction
    
    // Helper functions for expected result calculation
    function pentary_word_t pentary_add(pentary_word_t a, pentary_word_t b);
        // Implementation
    endfunction
    
    function pentary_word_t pentary_sub(pentary_word_t a, pentary_word_t b);
        // Implementation
    endfunction
    
    function pentary_word_t pentary_mul2(pentary_word_t a);
        // Implementation
    endfunction
    
    function pentary_word_t pentary_neg(pentary_word_t a);
        // Implementation
    endfunction
endclass
```

### 3.3 Waveform Analysis

#### Verdi Workflow

```bash
# Launch Verdi with waveform
verdi -ssf waves.fsdb \
      -nologo \
      -2001 \
      -f filelist.f &

# Verdi commands (in GUI)
# 1. Load design hierarchy
# 2. Add signals to waveform
# 3. Set up signal groups
# 4. Configure display format for pentary
# 5. Add markers and annotations
```

#### GTKWave (Open Source Alternative)

```bash
# Convert VCD to FST (faster)
vcd2fst waves.vcd waves.fst

# Launch GTKWave
gtkwave waves.fst &

# Save configuration
# File -> Write Save File -> waves.gtkw
```

---

## 4. Synthesis & Optimization

### 4.1 Synopsys Design Compiler

#### Basic Synthesis Script

```tcl
#==============================================================================
# synthesis.tcl - Synopsys Design Compiler Script
#==============================================================================

# Set up library paths
set search_path [list . \
                     /path/to/stdcell/lib \
                     /path/to/memory/lib]

set target_library [list stdcell_7nm.db memory_7nm.db]
set link_library [concat * $target_library]

# Read design
read_verilog -rtl [list \
    ../rtl/common/pentary_types.vh \
    ../rtl/core/pentary_alu.v \
    ../rtl/core/register_file.v \
    ../rtl/top/pentary_core.v]

# Set current design
current_design pentary_core

# Link design
link

# Check design
check_design

#------------------------------------------------------------------------------
# Constraints
#------------------------------------------------------------------------------

# Clock definition
create_clock -name clk -period 0.2 [get_ports clk]  # 5 GHz
set_clock_uncertainty 0.02 [get_clocks clk]          # 20 ps uncertainty
set_clock_transition 0.01 [get_clocks clk]           # 10 ps transition

# Input delays
set_input_delay -clock clk -max 0.05 [all_inputs]
set_input_delay -clock clk -min 0.01 [all_inputs]

# Output delays
set_output_delay -clock clk -max 0.05 [all_outputs]
set_output_delay -clock clk -min 0.01 [all_outputs]

# Area constraint
set_max_area 1250000  # 1.25 mm² in µm²

# Power constraint
set_max_dynamic_power 5.0  # 5W

#------------------------------------------------------------------------------
# Compile
#------------------------------------------------------------------------------

# Initial compile
compile_ultra -gate_clock -no_autoungroup

# Report results
report_area -hierarchy
report_timing -max_paths 10
report_power -hierarchy
report_qor

#------------------------------------------------------------------------------
# Optimization iterations
#------------------------------------------------------------------------------

# If timing not met, try these:

# 1. High-effort timing optimization
# compile_ultra -timing_high_effort_script

# 2. Incremental optimization
# compile_ultra -incremental

# 3. Physical synthesis (if floorplan available)
# compile_ultra -spg

#------------------------------------------------------------------------------
# Write outputs
#------------------------------------------------------------------------------

# Write netlist
write -format verilog -hierarchy -output ../syn/outputs/pentary_core_syn.v

# Write SDC
write_sdc ../syn/outputs/pentary_core.sdc

# Write reports
report_area > ../syn/reports/area.rpt
report_timing > ../syn/reports/timing.rpt
report_power > ../syn/reports/power.rpt
report_qor > ../syn/reports/qor.rpt

exit
```

#### Advanced Optimization Techniques

```tcl
#==============================================================================
# advanced_synthesis.tcl - Advanced Optimization
#==============================================================================

#------------------------------------------------------------------------------
# Pentary-Specific Optimizations
#------------------------------------------------------------------------------

# 1. Group pentary digits together
set_group -name pentary_digit_0 [get_cells *digit_0*]
set_group -name pentary_digit_1 [get_cells *digit_1*]
# ... repeat for all digits

# 2. Optimize carry chain
set_critical_range 0.1 [get_cells *carry*]
compile_ultra -incremental

# 3. Optimize zero-state detection
set_case_analysis 0 [get_pins */zero_state]
compile_ultra -incremental

#------------------------------------------------------------------------------
# Multi-Voltage Optimization
#------------------------------------------------------------------------------

# Define voltage areas
create_voltage_area -name core_area -coordinate {0 0 1000 1000}
create_voltage_area -name memory_area -coordinate {1000 0 2000 1000}

# Set voltages
set_voltage 1.2 -object_list [get_voltage_areas core_area]
set_voltage 0.9 -object_list [get_voltage_areas memory_area]

#------------------------------------------------------------------------------
# Clock Gating
#------------------------------------------------------------------------------

# Enable aggressive clock gating
set_clock_gating_style \
    -sequential_cell latch \
    -minimum_bitwidth 4 \
    -control_point before \
    -control_signal scan_enable

compile_ultra -gate_clock

#------------------------------------------------------------------------------
# Power Gating
#------------------------------------------------------------------------------

# Define power domains
create_power_domain PD_CORE -elements {core/*}
create_power_domain PD_MEMORY -elements {memory/*}

# Set power states
add_port_state VDD -state {ON 1.2} -state {OFF off}
create_pst pentary_pst -supplies {VDD}

# Add power switches
set_isolation iso_core -domain PD_CORE -isolation_power_net VDDG
set_level_shifter ls_core -domain PD_CORE -location parent

#------------------------------------------------------------------------------
# Area Recovery
#------------------------------------------------------------------------------

# After meeting timing, recover area
compile_ultra -area_high_effort_script -incremental

# Remove unnecessary buffers
remove_buffer_tree [all_fanout -flat -only_cells]

# Merge equivalent cells
uniquify -force

#------------------------------------------------------------------------------
# Final Optimization
#------------------------------------------------------------------------------

# One final pass
compile_ultra -incremental -only_design_rule

# Report final results
report_qor
report_area -hierarchy
report_timing -max_paths 100
report_power -hierarchy
```

### 4.2 Cadence Genus

#### Basic Synthesis Script

```tcl
#==============================================================================
# genus_synthesis.tcl - Cadence Genus Script
#==============================================================================

# Set up libraries
set_db init_lib_search_path {. /path/to/libs}
set_db library {stdcell_7nm.lib memory_7nm.lib}

# Read design
read_hdl -sv {
    ../rtl/common/pentary_types.vh
    ../rtl/core/pentary_alu.v
    ../rtl/core/register_file.v
    ../rtl/top/pentary_core.v
}

# Elaborate
elaborate pentary_core

# Read constraints
read_sdc ../syn/constraints/pentary_core.sdc

# Synthesize
syn_generic
syn_map
syn_opt

# Report
report_area
report_timing
report_power
report_gates

# Write outputs
write_hdl > ../syn/outputs/pentary_core_syn.v
write_sdc > ../syn/outputs/pentary_core.sdc

exit
```

---

## 5. Physical Design

### 5.1 Cadence Innovus

#### Complete P&R Flow

```tcl
#==============================================================================
# innovus_pnr.tcl - Place and Route Script
#==============================================================================

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Set design name
set DESIGN pentary_core

# Load libraries
set_db init_lib_search_path {/path/to/libs}
set_db init_lef_file {
    stdcell_7nm.lef
    memory_7nm.lef
}

# Read netlist
read_netlist ../syn/outputs/${DESIGN}_syn.v

# Read constraints
read_sdc ../syn/outputs/${DESIGN}.sdc

# Initialize design
init_design

#------------------------------------------------------------------------------
# Floorplan
#------------------------------------------------------------------------------

# Create floorplan (1.25mm x 1.25mm)
floorPlan -site core -r 1.0 0.7 10 10 10 10

# Add power rings
addRing -nets {VDD VSS} \
        -type core_rings \
        -layer {top M7 bottom M7 left M6 right M6} \
        -width 2.0 \
        -spacing 1.0 \
        -offset 5.0

# Add power stripes
addStripe -nets {VDD VSS} \
          -layer M6 \
          -direction vertical \
          -width 1.0 \
          -spacing 5.0 \
          -set_to_set_distance 50.0

#------------------------------------------------------------------------------
# Placement
#------------------------------------------------------------------------------

# Place standard cells
place_design

# Optimize placement
place_opt_design

# Report placement
report_placement

#------------------------------------------------------------------------------
# Clock Tree Synthesis
#------------------------------------------------------------------------------

# Specify clock tree
create_ccopt_clock_tree_spec -file ${DESIGN}_cts.spec

# Build clock tree
ccopt_design

# Report clock tree
report_ccopt_clock_trees

#------------------------------------------------------------------------------
# Routing
#------------------------------------------------------------------------------

# Set routing options
set_db route_design_with_timing_driven true
set_db route_design_with_si_driven true

# Route design
route_design

# Optimize routing
route_opt_design

# Report routing
report_route

#------------------------------------------------------------------------------
# Optimization
#------------------------------------------------------------------------------

# Post-route optimization
opt_design -post_route

# Fix DRC violations
ecoRoute -fix_drc

# Report final results
report_timing -max_paths 100
report_power
report_area

#------------------------------------------------------------------------------
# Verification
#------------------------------------------------------------------------------

# Verify connectivity
verify_connectivity

# Verify geometry
verify_drc

# Extract parasitics
extractRC

# Report final timing with parasitics
report_timing -max_paths 100 -late

#------------------------------------------------------------------------------
# Output
#------------------------------------------------------------------------------

# Write DEF
write_def ${DESIGN}.def

# Write GDS
streamOut ${DESIGN}.gds \
          -mapFile /path/to/streamout.map \
          -libName ${DESIGN} \
          -units 1000 \
          -mode ALL

# Write netlist
saveNetlist ${DESIGN}_final.v

# Write SDF
write_sdf ${DESIGN}.sdf

exit
```

### 5.2 Floorplanning Best Practices

#### Pentary-Specific Floorplan

```tcl
#==============================================================================
# pentary_floorplan.tcl - Optimized Floorplan for Pentary
#==============================================================================

#------------------------------------------------------------------------------
# Core Placement
#------------------------------------------------------------------------------

# Place ALU in center (most critical)
createInstGroup alu_group -inst {core/alu/*}
placeInstance alu_group 500 500

# Place register file near ALU
createInstGroup rf_group -inst {core/register_file/*}
placeInstance rf_group 300 500

# Place memristor crossbar
createInstGroup memristor_group -inst {core/memristor/*}
placeInstance memristor_group 700 500

#------------------------------------------------------------------------------
# Memory Hierarchy
#------------------------------------------------------------------------------

# L1 cache near core
createInstGroup l1_group -inst {memory/l1_cache/*}
placeInstance l1_group 500 300

# L2 cache
createInstGroup l2_group -inst {memory/l2_cache/*}
placeInstance l2_group 500 100

#------------------------------------------------------------------------------
# Power Planning
#------------------------------------------------------------------------------

# Create power domains
create_power_domain PD_CORE \
    -elements {core/*} \
    -supply {primary VDD_CORE}

create_power_domain PD_MEMORY \
    -elements {memory/*} \
    -supply {primary VDD_MEM}

# Add power switches for zero-state power gating
add_power_switch PS_CORE \
    -domain PD_CORE \
    -output_supply_port {VOUT VDD_CORE} \
    -input_supply_port {VIN VDD} \
    -control_port {CTRL zero_state_ctrl} \
    -on_state {ON VIN {CTRL}}

#------------------------------------------------------------------------------
# Routing Channels
#------------------------------------------------------------------------------

# Reserve routing channels for critical paths
createRouteBlk -box {450 450 550 550} -layer {M1 M2 M3}
createRouteBlk -name alu_channel

# Reserve space for clock tree
createRouteBlk -box {0 0 1000 50} -layer {M4 M5}
createRouteBlk -name clock_channel
```

---

## 6. Timing & Power Analysis

### 6.1 Static Timing Analysis (PrimeTime)

```tcl
#==============================================================================
# timing_analysis.tcl - PrimeTime STA Script
#==============================================================================

# Read libraries
set_app_var search_path {. /path/to/libs}
set_app_var link_library {* stdcell_7nm.db memory_7nm.db}

# Read design
read_verilog ../pnr/outputs/pentary_core_final.v
current_design pentary_core
link_design

# Read parasitics
read_parasitics ../pnr/outputs/pentary_core.spef

# Read constraints
read_sdc ../pnr/outputs/pentary_core.sdc

#------------------------------------------------------------------------------
# Setup Analysis
#------------------------------------------------------------------------------

# Report setup timing
report_timing \
    -delay_type max \
    -max_paths 100 \
    -nworst 10 \
    -sort_by slack \
    -path_type full_clock_expanded \
    > reports/setup_timing.rpt

# Report critical paths
report_timing \
    -delay_type max \
    -max_paths 10 \
    -nworst 1 \
    -path_type full_clock_expanded \
    -transition_time \
    -capacitance \
    -nets \
    > reports/critical_paths.rpt

#------------------------------------------------------------------------------
# Hold Analysis
#------------------------------------------------------------------------------

# Report hold timing
report_timing \
    -delay_type min \
    -max_paths 100 \
    -nworst 10 \
    -sort_by slack \
    > reports/hold_timing.rpt

#------------------------------------------------------------------------------
# Clock Analysis
#------------------------------------------------------------------------------

# Report clock tree
report_clock_timing \
    -type summary \
    > reports/clock_summary.rpt

# Report clock skew
report_clock_timing \
    -type skew \
    > reports/clock_skew.rpt

#------------------------------------------------------------------------------
# Pentary-Specific Analysis
#------------------------------------------------------------------------------

# Analyze carry chain timing
report_timing \
    -from [get_pins */carry_in] \
    -to [get_pins */carry_out] \
    -max_paths 20 \
    > reports/carry_chain_timing.rpt

# Analyze zero-state path timing
report_timing \
    -through [get_pins */zero_state] \
    -max_paths 20 \
    > reports/zero_state_timing.rpt

exit
```

### 6.2 Power Analysis (PrimePower)

```tcl
#==============================================================================
# power_analysis.tcl - PrimePower Script
#==============================================================================

# Read design
read_verilog ../pnr/outputs/pentary_core_final.v
current_design pentary_core
link_design

# Read parasitics
read_parasitics ../pnr/outputs/pentary_core.spef

# Read constraints
read_sdc ../pnr/outputs/pentary_core.sdc

# Read switching activity
read_vcd ../sim/waves/pentary_core.vcd \
    -strip_path tb_pentary_core/dut

#------------------------------------------------------------------------------
# Power Analysis
#------------------------------------------------------------------------------

# Report total power
report_power \
    -hierarchy \
    -verbose \
    > reports/power_total.rpt

# Report power by hierarchy
report_power \
    -hierarchy \
    -levels 3 \
    > reports/power_hierarchy.rpt

# Report power by domain
report_power \
    -domain \
    > reports/power_domain.rpt

#------------------------------------------------------------------------------
# Pentary-Specific Power Analysis
#------------------------------------------------------------------------------

# Analyze zero-state power savings
report_power \
    -instances [get_cells */zero_state*] \
    > reports/zero_state_power.rpt

# Analyze memristor power
report_power \
    -instances [get_cells */memristor*] \
    > reports/memristor_power.rpt

# Analyze clock power
report_power \
    -clock_network \
    > reports/clock_power.rpt

#------------------------------------------------------------------------------
# Power Optimization Recommendations
#------------------------------------------------------------------------------

# Identify high-power cells
report_power \
    -sort_by total_power \
    -max_paths 100 \
    > reports/high_power_cells.rpt

# Suggest clock gating opportunities
report_power \
    -clock_gating \
    > reports/clock_gating_opportunities.rpt

exit
```

---

## 7. Pentary-Specific Tool Configurations

### 7.1 Custom Synthesis Strategies

```tcl
#==============================================================================
# pentary_synthesis_strategy.tcl
#==============================================================================

#------------------------------------------------------------------------------
# Pentary Digit Grouping
#------------------------------------------------------------------------------

# Group pentary digits for better optimization
proc group_pentary_digits {design} {
    set digit_groups {}
    
    for {set i 0} {$i < 16} {incr i} {
        set group_name "pentary_digit_${i}"
        set cells [get_cells -hier -filter "ref_name =~ *digit_${i}*"]
        
        if {[sizeof_collection $cells] > 0} {
            set_group -name $group_name $cells
            lappend digit_groups $group_name
        }
    }
    
    return $digit_groups
}

#------------------------------------------------------------------------------
# Carry Chain Optimization
#------------------------------------------------------------------------------

# Optimize carry propagation paths
proc optimize_carry_chain {design} {
    # Identify carry chain
    set carry_cells [get_cells -hier -filter "ref_name =~ *carry*"]
    
    # Set high priority
    set_critical_range 0.05 $carry_cells
    
    # Use fast cells
    size_cell $carry_cells -library fast_lib
    
    # Minimize routing
    set_dont_touch_network [get_nets -of $carry_cells]
}

#------------------------------------------------------------------------------
# Zero-State Optimization
#------------------------------------------------------------------------------

# Optimize for zero-state power savings
proc optimize_zero_state {design} {
    # Identify zero-state logic
    set zero_cells [get_cells -hier -filter "ref_name =~ *zero_state*"]
    
    # Enable power gating
    set_power_gating_style \
        -control_point before \
        -control_signal zero_state_enable
    
    # Apply to zero-state cells
    compile_ultra -gate_clock -incremental
}

#------------------------------------------------------------------------------
# Memristor Interface Optimization
#------------------------------------------------------------------------------

# Optimize memristor read/write paths
proc optimize_memristor_interface {design} {
    # Identify memristor interface
    set mem_cells [get_cells -hier -filter "ref_name =~ *memristor*"]
    
    # Set timing constraints
    set_multicycle_path 2 -from $mem_cells -to [all_outputs]
    set_multicycle_path 2 -from [all_inputs] -to $mem_cells
    
    # Optimize for low power
    set_max_dynamic_power 0.5 -design $mem_cells
}
```

### 7.2 Custom Timing Constraints

```tcl
#==============================================================================
# pentary_constraints.sdc
#==============================================================================

#------------------------------------------------------------------------------
# Clock Definition
#------------------------------------------------------------------------------

# Main clock (5 GHz)
create_clock -name clk -period 0.2 [get_ports clk]
set_clock_uncertainty 0.02 [get_clocks clk]
set_clock_transition 0.01 [get_clocks clk]

#------------------------------------------------------------------------------
# Pentary-Specific Constraints
#------------------------------------------------------------------------------

# Carry chain timing
set_multicycle_path 2 \
    -from [get_pins */carry_in] \
    -to [get_pins */carry_out]

# Zero-state detection (can be slower)
set_multicycle_path 3 \
    -from [get_pins */data_in*] \
    -to [get_pins */zero_state]

# Memristor access (slower)
set_multicycle_path 5 \
    -from [get_pins */memristor_addr*] \
    -to [get_pins */memristor_data*]

#------------------------------------------------------------------------------
# False Paths
#------------------------------------------------------------------------------

# Asynchronous reset
set_false_path -from [get_ports rst]

# Test mode
set_false_path -from [get_ports test_mode]

# Power gating control
set_false_path -from [get_pins */power_gate_ctrl]

#------------------------------------------------------------------------------
# Input/Output Delays
#------------------------------------------------------------------------------

# Input delays (relative to clock)
set_input_delay -clock clk -max 0.05 [all_inputs]
set_input_delay -clock clk -min 0.01 [all_inputs]

# Output delays
set_output_delay -clock clk -max 0.05 [all_outputs]
set_output_delay -clock clk -min 0.01 [all_outputs]

#------------------------------------------------------------------------------
# Load and Drive
#------------------------------------------------------------------------------

# Input drive strength
set_driving_cell -lib_cell BUFX4 [all_inputs]

# Output load
set_load 0.1 [all_outputs]
```

---

## 8. Automation & Scripting

### 8.1 Makefile for Complete Flow

```makefile
#==============================================================================
# Makefile for Pentary Chip Design Flow
#==============================================================================

# Directories
RTL_DIR = ../rtl
TB_DIR = ../tb
SYN_DIR = ../syn
PNR_DIR = ../pnr
SIM_DIR = ../sim

# Tools
VCS = vcs
VERDI = verdi
DC = dc_shell
PT = pt_shell
INNOVUS = innovus

# Targets
.PHONY: all clean sim syn pnr sta power

all: sim syn pnr sta power

#------------------------------------------------------------------------------
# Simulation
#------------------------------------------------------------------------------

sim:
	@echo "Running simulation..."
	cd $(SIM_DIR) && $(VCS) -full64 -sverilog -f filelist.f
	cd $(SIM_DIR) && ./simv +ntb_random_seed=12345
	@echo "Simulation complete!"

sim_cov:
	@echo "Running simulation with coverage..."
	cd $(SIM_DIR) && $(VCS) -full64 -sverilog -cm line+cond+fsm+tgl -f filelist.f
	cd $(SIM_DIR) && ./simv -cm line+cond+fsm+tgl
	cd $(SIM_DIR) && urg -dir simv.vdb
	@echo "Coverage analysis complete!"

waves:
	@echo "Launching waveform viewer..."
	cd $(SIM_DIR) && $(VERDI) -ssf waves.fsdb -f filelist.f &

#------------------------------------------------------------------------------
# Synthesis
#------------------------------------------------------------------------------

syn:
	@echo "Running synthesis..."
	cd $(SYN_DIR) && $(DC) -f scripts/synthesis.tcl | tee logs/synthesis.log
	@echo "Synthesis complete!"

syn_opt:
	@echo "Running optimized synthesis..."
	cd $(SYN_DIR) && $(DC) -f scripts/advanced_synthesis.tcl | tee logs/synthesis_opt.log
	@echo "Optimized synthesis complete!"

#------------------------------------------------------------------------------
# Place & Route
#------------------------------------------------------------------------------

pnr:
	@echo "Running place and route..."
	cd $(PNR_DIR) && $(INNOVUS) -init scripts/innovus_pnr.tcl | tee logs/pnr.log
	@echo "Place and route complete!"

#------------------------------------------------------------------------------
# Static Timing Analysis
#------------------------------------------------------------------------------

sta:
	@echo "Running static timing analysis..."
	cd $(PNR_DIR) && $(PT) -f scripts/timing_analysis.tcl | tee logs/sta.log
	@echo "STA complete!"

#------------------------------------------------------------------------------
# Power Analysis
#------------------------------------------------------------------------------

power:
	@echo "Running power analysis..."
	cd $(PNR_DIR) && ptpx_shell -f scripts/power_analysis.tcl | tee logs/power.log
	@echo "Power analysis complete!"

#------------------------------------------------------------------------------
# Reports
#------------------------------------------------------------------------------

reports:
	@echo "Generating reports..."
	@echo "=== Area Report ===" > reports/summary.rpt
	@cat $(SYN_DIR)/reports/area.rpt >> reports/summary.rpt
	@echo "\n=== Timing Report ===" >> reports/summary.rpt
	@cat $(PNR_DIR)/reports/timing.rpt >> reports/summary.rpt
	@echo "\n=== Power Report ===" >> reports/summary.rpt
	@cat $(PNR_DIR)/reports/power.rpt >> reports/summary.rpt
	@echo "Reports generated!"

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------

clean:
	@echo "Cleaning build files..."
	rm -rf $(SIM_DIR)/simv* $(SIM_DIR)/*.log $(SIM_DIR)/csrc
	rm -rf $(SYN_DIR)/outputs/* $(SYN_DIR)/logs/*
	rm -rf $(PNR_DIR)/outputs/* $(PNR_DIR)/logs/*
	@echo "Clean complete!"

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo "Pentary Chip Design Flow Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Run complete flow (sim, syn, pnr, sta, power)"
	@echo "  sim        - Run simulation"
	@echo "  sim_cov    - Run simulation with coverage"
	@echo "  waves      - Launch waveform viewer"
	@echo "  syn        - Run synthesis"
	@echo "  syn_opt    - Run optimized synthesis"
	@echo "  pnr        - Run place and route"
	@echo "  sta        - Run static timing analysis"
	@echo "  power      - Run power analysis"
	@echo "  reports    - Generate summary reports"
	@echo "  clean      - Clean build files"
	@echo "  help       - Show this help message"
```

### 8.2 Python Automation Scripts

```python
#!/usr/bin/env python3
"""
pentary_flow_manager.py - Automated Design Flow Manager
"""

import os
import subprocess
import sys
from pathlib import Path
import json
import time

class PentaryFlowManager:
    def __init__(self, config_file="flow_config.json"):
        self.config = self.load_config(config_file)
        self.results = {}
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def run_simulation(self):
        """Run RTL simulation"""
        print("=" * 80)
        print("Running Simulation...")
        print("=" * 80)
        
        start_time = time.time()
        
        cmd = [
            "vcs",
            "-full64",
            "-sverilog",
            "-f", "filelist.f",
            "-o", "simv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Compilation successful")
            
            # Run simulation
            result = subprocess.run(["./simv"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Simulation successful")
                self.results['simulation'] = {
                    'status': 'PASS',
                    'time': time.time() - start_time
                }
            else:
                print("✗ Simulation failed")
                print(result.stderr)
                self.results['simulation'] = {
                    'status': 'FAIL',
                    'error': result.stderr
                }
        else:
            print("✗ Compilation failed")
            print(result.stderr)
            self.results['simulation'] = {
                'status': 'FAIL',
                'error': result.stderr
            }
    
    def run_synthesis(self):
        """Run logic synthesis"""
        print("=" * 80)
        print("Running Synthesis...")
        print("=" * 80)
        
        start_time = time.time()
        
        cmd = [
            "dc_shell",
            "-f", "scripts/synthesis.tcl"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Synthesis successful")
            
            # Parse results
            area, timing, power = self.parse_synthesis_results()
            
            self.results['synthesis'] = {
                'status': 'PASS',
                'time': time.time() - start_time,
                'area': area,
                'timing': timing,
                'power': power
            }
        else:
            print("✗ Synthesis failed")
            print(result.stderr)
            self.results['synthesis'] = {
                'status': 'FAIL',
                'error': result.stderr
            }
    
    def run_pnr(self):
        """Run place and route"""
        print("=" * 80)
        print("Running Place and Route...")
        print("=" * 80)
        
        start_time = time.time()
        
        cmd = [
            "innovus",
            "-init", "scripts/innovus_pnr.tcl"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Place and route successful")
            self.results['pnr'] = {
                'status': 'PASS',
                'time': time.time() - start_time
            }
        else:
            print("✗ Place and route failed")
            print(result.stderr)
            self.results['pnr'] = {
                'status': 'FAIL',
                'error': result.stderr
            }
    
    def parse_synthesis_results(self):
        """Parse synthesis results from reports"""
        area = None
        timing = None
        power = None
        
        # Parse area report
        with open('reports/area.rpt', 'r') as f:
            for line in f:
                if 'Total cell area:' in line:
                    area = float(line.split(':')[1].strip())
        
        # Parse timing report
        with open('reports/timing.rpt', 'r') as f:
            for line in f:
                if 'slack' in line.lower():
                    timing = float(line.split()[-1])
        
        # Parse power report
        with open('reports/power.rpt', 'r') as f:
            for line in f:
                if 'Total Dynamic Power' in line:
                    power = float(line.split()[-2])
        
        return area, timing, power
    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "=" * 80)
        print("FLOW SUMMARY")
        print("=" * 80)
        
        for stage, result in self.results.items():
            print(f"\n{stage.upper()}:")
            print(f"  Status: {result['status']}")
            if 'time' in result:
                print(f"  Time: {result['time']:.2f}s")
            if 'area' in result:
                print(f"  Area: {result['area']:.2f} µm²")
            if 'timing' in result:
                print(f"  Slack: {result['timing']:.3f} ns")
            if 'power' in result:
                print(f"  Power: {result['power']:.3f} W")
        
        # Save to file
        with open('flow_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_complete_flow(self):
        """Run complete design flow"""
        self.run_simulation()
        if self.results['simulation']['status'] == 'PASS':
            self.run_synthesis()
            if self.results['synthesis']['status'] == 'PASS':
                self.run_pnr()
        
        self.generate_report()

if __name__ == "__main__":
    manager = PentaryFlowManager()
    manager.run_complete_flow()
```

---

## 9. Best Practices

### 9.1 Design Best Practices

1. **Use Hierarchical Design**
   - Separate concerns
   - Enable parallel development
   - Improve reusability

2. **Follow Naming Conventions**
   - Use descriptive names
   - Be consistent
   - Document naming scheme

3. **Add Assertions**
   - Catch bugs early
   - Document assumptions
   - Enable formal verification

4. **Use Parameters**
   - Make design configurable
   - Enable reuse
   - Simplify testing

5. **Document Everything**
   - Write clear comments
   - Create block diagrams
   - Maintain specifications

### 9.2 Verification Best Practices

1. **Start Early**
   - Write testbenches with RTL
   - Test incrementally
   - Don't wait for completion

2. **Use Coverage**
   - Define coverage goals
   - Track progress
   - Identify gaps

3. **Automate Testing**
   - Use regression suites
   - Run tests frequently
   - Catch regressions early

4. **Test Corner Cases**
   - Maximum values
   - Minimum values
   - Edge conditions
   - Error conditions

5. **Use Formal Verification**
   - Prove correctness
   - Find corner cases
   - Complement simulation

### 9.3 Synthesis Best Practices

1. **Set Realistic Constraints**
   - Based on requirements
   - Include margins
   - Update as needed

2. **Optimize Iteratively**
   - Start with basic compile
   - Identify bottlenecks
   - Optimize incrementally

3. **Use Hierarchy**
   - Compile bottom-up
   - Preserve hierarchy
   - Enable incremental optimization

4. **Check Results**
   - Review reports
   - Verify functionality
   - Check timing/power/area

5. **Document Decisions**
   - Record constraints
   - Explain optimizations
   - Track changes

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue 1: Timing Violations

**Symptoms:**
- Negative slack
- Setup/hold violations
- Clock skew issues

**Solutions:**
```tcl
# 1. Identify critical paths
report_timing -max_paths 10

# 2. Optimize critical cells
size_cell [get_cells critical_path/*] -library fast_lib

# 3. Add pipeline stages
# (requires RTL changes)

# 4. Reduce clock frequency
# (last resort)
```

#### Issue 2: High Power Consumption

**Symptoms:**
- Power exceeds budget
- Thermal issues
- Battery drain

**Solutions:**
```tcl
# 1. Enable clock gating
compile_ultra -gate_clock

# 2. Use power gating
set_power_gating_style -control_point auto

# 3. Reduce voltage
set_voltage 0.9 -object_list [get_designs]

# 4. Optimize switching activity
# (requires RTL changes)
```

#### Issue 3: Area Overflow

**Symptoms:**
- Design doesn't fit
- Routing congestion
- High cost

**Solutions:**
```tcl
# 1. Optimize for area
compile_ultra -area_high_effort_script

# 2. Share resources
# (requires RTL changes)

# 3. Use smaller cells
set_max_area [expr $current_area * 0.9]

# 4. Reduce features
# (last resort)
```

### 10.2 Debug Techniques

#### Technique 1: Waveform Analysis

```bash
# 1. Generate detailed waveforms
./simv +vcs+dumpvars+all

# 2. View in Verdi
verdi -ssf waves.fsdb &

# 3. Add signals of interest
# 4. Set up triggers
# 5. Analyze behavior
```

#### Technique 2: Assertion-Based Debug

```systemverilog
// Add assertions to catch issues
assert property (@(posedge clk) valid |-> ready)
    else $error("Protocol violation");

assert property (@(posedge clk) !overflow)
    else $error("Overflow detected");
```

#### Technique 3: Print Statements

```verilog
// Add debug prints
always @(posedge clk) begin
    if (debug_enable) begin
        $display("Time=%0t: a=%h, b=%h, result=%h",
                 $time, operand_a, operand_b, result);
    end
end
```

---

## Conclusion

This guide provides comprehensive information about EDA tools and workflows for pentary chip design. Key takeaways:

1. **Tool Selection:** Choose appropriate tools based on budget and requirements
2. **Workflow:** Follow systematic design flow from RTL to GDS
3. **Optimization:** Use pentary-specific optimizations
4. **Automation:** Automate repetitive tasks
5. **Best Practices:** Follow industry standards
6. **Troubleshooting:** Know how to debug issues

**Remember:** The tools are just enablers. Good design practices and understanding of pentary architecture are essential for success.

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Status:** Active Reference Guide

---

## Quick Reference

### Essential Commands

```bash
# Simulation
vcs -full64 -sverilog -f filelist.f
./simv

# Synthesis
dc_shell -f synthesis.tcl

# Place & Route
innovus -init pnr.tcl

# Timing Analysis
pt_shell -f timing.tcl

# Power Analysis
ptpx_shell -f power.tcl

# Waveform Viewing
verdi -ssf waves.fsdb &
```

### Important Files

- `filelist.f` - List of source files
- `synthesis.tcl` - Synthesis script
- `constraints.sdc` - Timing constraints
- `pnr.tcl` - Place and route script
- `Makefile` - Build automation

### Key Reports

- `area.rpt` - Area report
- `timing.rpt` - Timing report
- `power.rpt` - Power report
- `qor.rpt` - Quality of results

---

**End of Document**