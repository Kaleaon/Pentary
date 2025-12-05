# Pentary FPGA Prototyping Guide

## Executive Summary

Complete guide for prototyping pentary processors on FPGA platforms before ASIC fabrication.

**Goal**: Validate design on real hardware  
**Timeline**: 3-6 months  
**Cost**: $50K-$200K  
**Platform**: Xilinx Virtex UltraScale+ or Intel Stratix 10

---

## 1. FPGA Platform Selection

### 1.1 Platform Comparison

#### Xilinx Virtex UltraScale+ VU9P
**Specifications**:
- Logic Cells: 1,182,240
- Block RAM: 75.9 Mb
- DSP Slices: 6,840
- Transceivers: 96 @ 32.75 Gbps
- Cost: ~$30,000

**Pros**:
- ✅ Largest Xilinx FPGA
- ✅ Excellent tool support (Vivado)
- ✅ Good documentation
- ✅ Strong ecosystem

**Cons**:
- ❌ Expensive
- ❌ High power consumption (~100W)

#### Intel Stratix 10 GX 2800
**Specifications**:
- Logic Elements: 2,753,000
- M20K Blocks: 11,721
- DSP Blocks: 5,760
- Transceivers: 96 @ 28.3 Gbps
- Cost: ~$40,000

**Pros**:
- ✅ More logic than Xilinx
- ✅ HyperFlex architecture
- ✅ Good for high-speed designs

**Cons**:
- ❌ More expensive
- ❌ Quartus tools less mature than Vivado
- ❌ Longer compile times

#### AMD Versal AI Core
**Specifications**:
- Logic Cells: 900,000
- AI Engines: 400
- DSP Engines: 1,968
- DDR4 Controllers: 4
- Cost: ~$50,000

**Pros**:
- ✅ Built-in AI acceleration
- ✅ High bandwidth memory
- ✅ Advanced architecture

**Cons**:
- ❌ Most expensive
- ❌ Newer platform (less mature)
- ❌ Steeper learning curve

### 1.2 Recommendation

**Primary**: Xilinx Virtex UltraScale+ VU9P
- Best balance of resources and cost
- Mature tools and ecosystem
- Sufficient resources for pentary core

**Alternative**: Intel Stratix 10 GX
- If need more logic resources
- Good for multi-core implementation

---

## 2. Resource Mapping

### 2.1 Pentary Core Resource Estimate

| Component | LUTs | FFs | BRAM | DSP | Percentage (VU9P) |
|-----------|------|-----|------|-----|-------------------|
| **PentaryALU** | 5,000 | 2,000 | 0 | 0 | 0.4% |
| **Register File** | 3,000 | 1,536 | 2 | 0 | 0.3% |
| **Pipeline Control** | 8,000 | 5,000 | 0 | 0 | 0.7% |
| **L1 I-Cache** | 10,000 | 5,000 | 32 | 0 | 0.8% |
| **L1 D-Cache** | 12,000 | 6,000 | 32 | 0 | 1.0% |
| **L2 Cache** | 20,000 | 10,000 | 256 | 0 | 1.7% |
| **MMU** | 8,000 | 4,000 | 8 | 0 | 0.7% |
| **Interrupt Ctrl** | 3,000 | 2,000 | 0 | 0 | 0.3% |
| **Memristor Emu** | 50,000 | 30,000 | 200 | 100 | 4.2% |
| **Interconnect** | 10,000 | 5,000 | 0 | 0 | 0.8% |
| **Debug** | 5,000 | 3,000 | 4 | 0 | 0.4% |
| **Total (1 core)** | **134,000** | **73,536** | **534** | **100** | **11.3%** |

**Conclusion**: Can fit 8 cores on VU9P with room for debugging infrastructure

### 2.2 Memristor Emulation Strategy

**Challenge**: FPGAs don't have analog memristors

**Solution 1: Digital Emulation**
```verilog
module MemristorEmulator (
    input         clk,
    input  [2:0]  program_value,
    input         program_enable,
    input  [7:0]  read_voltage,
    output [7:0]  read_current
);
    // Store resistance state in BRAM
    reg [7:0] resistance;
    
    // Resistance values for 5 states
    parameter R_NEG2 = 8'd10;
    parameter R_NEG1 = 8'd30;
    parameter R_ZERO = 8'd50;
    parameter R_POS1 = 8'd70;
    parameter R_POS2 = 8'd90;
    
    always @(posedge clk) begin
        if (program_enable) begin
            case (program_value)
                3'b000: resistance <= R_NEG2;
                3'b001: resistance <= R_NEG1;
                3'b010: resistance <= R_ZERO;
                3'b011: resistance <= R_POS1;
                3'b100: resistance <= R_POS2;
            endcase
        end
    end
    
    // Emulate Ohm's law: I = V / R
    // Use DSP block for division
    assign read_current = read_voltage / resistance;
    
endmodule
```

**Solution 2: Lookup Table**
```verilog
// Pre-compute all possible results
// Store in BRAM for fast access
module MemristorLUT (
    input  [767:0] input_vector,   // 256 pents
    input  [7:0]   row_select,
    output [2:0]   output_value
);
    // BRAM stores pre-computed results
    // 256 rows × 256 possible input patterns
    // This is simplified - actual would be huge
    
    (* ram_style = "block" *)
    reg [2:0] lut [0:255][0:255];
    
    assign output_value = lut[row_select][input_hash];
    
endmodule
```

**Recommendation**: Use digital emulation with DSP blocks for multiplication

---

## 3. FPGA Design Flow

### 3.1 Complete Flow

```
RTL (Verilog)
    ↓
┌─────────────────────────────────────┐
│  Synthesis (Vivado Synth)           │
│  - Technology mapping to FPGA       │
│  - Resource optimization            │
│  - Timing optimization              │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Implementation                     │
│  - Placement                        │
│  - Routing                          │
│  - Timing closure                   │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Bitstream Generation               │
│  - Configuration bitstream          │
│  - Programming file (.bit)          │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Programming                        │
│  - Download to FPGA                 │
│  - Verify configuration             │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Testing & Validation               │
│  - Functional tests                 │
│  - Performance measurement          │
│  - Debug and iterate                │
└─────────────────────────────────────┘
```

### 3.2 Vivado Project Setup

```tcl
# Vivado TCL script for pentary project

# Create project
create_project pentary_fpga ./pentary_fpga -part xcvu9p-flga2104-2-i

# Add source files
add_files {
    hardware/pentary_adder_fixed.v
    hardware/pentary_alu_fixed.v
    hardware/pentary_quantizer_fixed.v
    hardware/memristor_crossbar_fixed.v
    hardware/register_file.v
    hardware/pipeline_control.v
    hardware/cache_hierarchy.v
    hardware/mmu_interrupt.v
    hardware/instruction_decoder.v
    hardware/pentary_core_integrated.v
}

# Add constraints
add_files -fileset constrs_1 constraints/pentary_fpga.xdc

# Set top module
set_property top PentaryCoreIntegrated [current_fileset]

# Synthesis settings
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property steps.synth_design.args.retiming true [get_runs synth_1]

# Implementation settings
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# Run synthesis
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -jobs 8
wait_on_run impl_1

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Generate reports
open_run impl_1
report_timing_summary -file timing_summary.rpt
report_utilization -file utilization.rpt
report_power -file power.rpt
```

### 3.3 Constraints File

```tcl
# pentary_fpga.xdc - Timing and placement constraints

# Clock constraints
create_clock -period 5.0 -name clk [get_ports clk]
set_input_delay -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]

# False paths
set_false_path -from [get_ports reset]

# Multi-cycle paths
set_multicycle_path -setup 2 -from [get_pins */memristor_*/reg/C] -to [get_pins */accumulator*/D]

# Clock groups
set_clock_groups -asynchronous -group [get_clocks clk] -group [get_clocks mem_clk]

# Placement constraints
create_pblock pblock_core0
resize_pblock pblock_core0 -add {SLICE_X0Y0:SLICE_X50Y50}
add_cells_to_pblock pblock_core0 [get_cells core0/*]

# I/O standards
set_property IOSTANDARD LVCMOS18 [get_ports *]
set_property PACKAGE_PIN AY24 [get_ports clk]
```

---

## 4. Porting Considerations

### 4.1 ASIC to FPGA Differences

| Aspect | ASIC | FPGA | Adaptation Needed |
|--------|------|------|-------------------|
| **Clock Speed** | 5 GHz | 200-400 MHz | Reduce frequency |
| **Memory** | Custom SRAM | Block RAM | Use BRAM primitives |
| **Arithmetic** | Custom cells | LUTs + DSP | Map to DSP blocks |
| **Routing** | Custom | Fixed | Optimize placement |
| **Power** | 5W | 50-100W | Accept higher power |

### 4.2 Required Modifications

#### Clock Frequency Reduction
```verilog
// ASIC: 5 GHz
parameter ASIC_FREQ = 5_000_000_000;

// FPGA: 200 MHz (25× slower)
parameter FPGA_FREQ = 200_000_000;

// Adjust timing parameters
parameter FPGA_CYCLES_PER_ASIC_CYCLE = ASIC_FREQ / FPGA_FREQ;  // 25
```

#### Memory Mapping
```verilog
// ASIC: Custom SRAM
reg [47:0] registers [0:31];

// FPGA: Use Block RAM
(* ram_style = "block" *)
reg [47:0] registers [0:31];

// Or use BRAM primitives
RAMB36E2 #(
    .READ_WIDTH_A(48),
    .WRITE_WIDTH_A(48)
) register_bram (
    .CLKARDCLK(clk),
    .ADDRARDADDR(read_addr),
    .DOUTADOUT(read_data),
    // ... more ports
);
```

#### Arithmetic Mapping
```verilog
// Use DSP blocks for multiplication
DSP48E2 #(
    .USE_MULT("MULTIPLY")
) dsp_mult (
    .A(operand_a),
    .B(operand_b),
    .P(product),
    // ... more ports
);
```

### 4.3 Memristor Emulation

```verilog
// Simplified memristor crossbar for FPGA
module MemristorCrossbarFPGA (
    input         clk,
    input         reset,
    input  [767:0] input_vector,
    output [767:0] output_vector
);
    // Store weights in Block RAM
    (* ram_style = "block" *)
    reg [2:0] weights [0:255][0:255];
    
    // Use DSP blocks for MAC operations
    reg signed [15:0] accumulator [0:255];
    
    // Parallel MAC units (limited by DSP availability)
    genvar i;
    generate
        for (i = 0; i < 256; i = i + 1) begin : mac_units
            // Each row gets a MAC unit
            PentaryMACUnit mac (
                .clk(clk),
                .weights(weights[i]),
                .inputs(input_vector),
                .result(accumulator[i])
            );
        end
    endgenerate
    
    // Quantize results back to pentary
    generate
        for (i = 0; i < 256; i = i + 1) begin : quantize
            assign output_vector[i*3 +: 3] = quantize_to_pentary(accumulator[i]);
        end
    endgenerate
    
endmodule
```

---

## 5. Development Board Options

### 5.1 Xilinx Boards

#### VCU118 Evaluation Kit
- **FPGA**: Virtex UltraScale+ VU9P
- **Memory**: 4GB DDR4
- **Interfaces**: PCIe Gen3 x16, Ethernet, USB
- **Cost**: $6,995
- **Availability**: In stock

#### Alveo U280
- **FPGA**: Virtex UltraScale+ VU37P
- **Memory**: 8GB HBM2
- **Interfaces**: PCIe Gen3 x16, 100G Ethernet
- **Cost**: $8,995
- **Advantages**: Data center ready, high bandwidth memory

### 5.2 Intel Boards

#### Stratix 10 GX Development Kit
- **FPGA**: Stratix 10 GX 2800
- **Memory**: 4GB DDR4
- **Interfaces**: PCIe Gen3 x16, Ethernet
- **Cost**: $4,995
- **Availability**: In stock

### 5.3 Recommendation

**For Prototyping**: VCU118
- Good balance of features and cost
- Sufficient resources
- Excellent tool support

**For Performance**: Alveo U280
- HBM2 for high bandwidth
- Data center deployment ready
- Better for multi-core testing

---

## 6. Implementation Strategy

### 6.1 Phased Approach

#### Phase 1: Single Module (Week 1-2)
```
Goal: Validate basic pentary arithmetic
Implement:
  - PentaryALU only
  - Simple testbench
  - Verify on FPGA
  
Resources: <10K LUTs
Success: ALU operations work correctly
```

#### Phase 2: Core Components (Week 3-4)
```
Goal: Validate core functionality
Implement:
  - ALU + Register File
  - Basic pipeline (3 stages)
  - Simple memory interface
  
Resources: ~50K LUTs
Success: Simple programs execute
```

#### Phase 3: Complete Core (Week 5-8)
```
Goal: Full single-core implementation
Implement:
  - Complete 5-stage pipeline
  - L1 caches
  - Memristor emulation
  - Debug interface
  
Resources: ~150K LUTs
Success: Neural network inference works
```

#### Phase 4: Multi-Core (Week 9-12)
```
Goal: Multi-core with L2 cache
Implement:
  - 2-4 cores
  - L2 cache
  - Cache coherency
  - Performance measurement
  
Resources: ~500K LUTs
Success: Multi-core benchmarks run
```

### 6.2 Incremental Validation

```
After each phase:
  1. Synthesize and implement
  2. Program FPGA
  3. Run functional tests
  4. Measure performance
  5. Debug issues
  6. Document results
  7. Proceed to next phase
```

---

## 7. Testing on FPGA

### 7.1 Test Infrastructure

```
┌─────────────────────────────────────┐
│  Host PC                            │
│  - Test scripts                     │
│  - Data generation                  │
│  - Result verification              │
└────────────┬────────────────────────┘
             ↓ PCIe
┌─────────────────────────────────────┐
│  FPGA Board                         │
│  - Pentary core(s)                  │
│  - DDR4 memory                      │
│  - Debug logic                      │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Measurement Equipment              │
│  - Oscilloscope                     │
│  - Logic analyzer                   │
│  - Power meter                      │
└─────────────────────────────────────┘
```

### 7.2 Test Software

```python
# FPGA test framework

import pcie_driver
import numpy as np

class PentaryFPGATester:
    def __init__(self, device_id):
        self.device = pcie_driver.open_device(device_id)
        
    def load_program(self, binary_file):
        """Load program into FPGA memory"""
        with open(binary_file, 'rb') as f:
            program = f.read()
        self.device.write_memory(0x0, program)
        
    def run_test(self, test_name):
        """Run a specific test"""
        print(f"Running test: {test_name}")
        
        # Reset processor
        self.device.write_register(0x0, 0x1)  # Assert reset
        time.sleep(0.001)
        self.device.write_register(0x0, 0x0)  # Deassert reset
        
        # Start execution
        self.device.write_register(0x4, 0x1)  # Start
        
        # Wait for completion
        while not self.device.read_register(0x8):  # Done flag
            time.sleep(0.001)
        
        # Read results
        results = self.device.read_memory(0x10000, 1024)
        
        return results
        
    def measure_performance(self, num_iterations):
        """Measure performance metrics"""
        start_time = time.time()
        
        for i in range(num_iterations):
            self.run_test(f"iteration_{i}")
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = num_iterations / total_time
        
        return {
            'throughput': throughput,
            'latency': total_time / num_iterations
        }
```

### 7.3 Debug Infrastructure

```verilog
// Integrated Logic Analyzer (ILA)
module DebugProbe (
    input         clk,
    input  [47:0] probe_pc,
    input  [31:0] probe_instruction,
    input  [47:0] probe_alu_result,
    input         probe_branch_taken
);
    // Xilinx ILA IP
    ila_0 ila_inst (
        .clk(clk),
        .probe0(probe_pc),
        .probe1(probe_instruction),
        .probe2(probe_alu_result),
        .probe3(probe_branch_taken)
    );
endmodule

// Virtual I/O (VIO) for runtime control
module DebugControl (
    input         clk,
    output        reset_out,
    output        start_out,
    input  [31:0] status_in
);
    vio_0 vio_inst (
        .clk(clk),
        .probe_out0(reset_out),
        .probe_out1(start_out),
        .probe_in0(status_in)
    );
endmodule
```

---

## 8. Performance Expectations

### 8.1 FPGA vs ASIC Performance

| Metric | ASIC Target | FPGA Prototype | Ratio |
|--------|-------------|----------------|-------|
| **Clock Freq** | 5 GHz | 200 MHz | 25× slower |
| **Throughput** | 10 TOPS | 0.4 TOPS | 25× slower |
| **Power** | 5W | 50W | 10× higher |
| **Latency** | 1 cycle | 1 cycle | Same |

**Key**: FPGA validates functionality, not final performance

### 8.2 What FPGA Validates

✅ **Functional Correctness**
- All instructions execute correctly
- Pipeline works as designed
- Caches function properly
- Memristor emulation accurate

✅ **Architectural Decisions**
- Pipeline depth appropriate
- Cache sizes sufficient
- Hazard detection works
- Branch prediction effective

✅ **Software Compatibility**
- Programs run correctly
- Compiler generates valid code
- Debugger works
- Libraries function

❌ **Not Validated on FPGA**
- Final clock frequency (5 GHz)
- Actual power consumption (5W)
- True memristor behavior
- Manufacturing yield

---

## 9. Common Issues and Solutions

### 9.1 Timing Closure

**Problem**: Design doesn't meet timing at target frequency

**Solutions**:
1. **Reduce frequency**: Start at 100 MHz, increase gradually
2. **Pipeline deeper**: Add more pipeline stages
3. **Optimize critical paths**: Use pipelining, retiming
4. **Use faster speed grade**: -2 or -3 speed grade

```tcl
# Timing optimization
set_property strategy Performance_ExploreWithRemap [get_runs impl_1]
set_property steps.phys_opt_design.is_enabled true [get_runs impl_1]
set_property steps.post_route_phys_opt_design.is_enabled true [get_runs impl_1]
```

### 9.2 Resource Overflow

**Problem**: Design doesn't fit on FPGA

**Solutions**:
1. **Reduce cores**: Start with 1-2 cores instead of 8
2. **Smaller caches**: Reduce L1/L2 cache sizes
3. **Simplify memristor**: Use smaller crossbar (128×128)
4. **Remove debug logic**: Disable ILA/VIO in final build

```tcl
# Resource optimization
set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property steps.synth_design.args.resource_sharing on [get_runs synth_1]
```

### 9.3 Memory Bandwidth

**Problem**: DDR4 bandwidth insufficient for multi-core

**Solutions**:
1. **Use HBM**: Alveo U280 with HBM2
2. **Optimize access patterns**: Burst transfers
3. **Add more memory controllers**: Use all 4 DDR4 channels
4. **Cache optimization**: Larger caches, better hit rates

---

## 10. Validation Checklist

### 10.1 Functional Validation

- [ ] All instructions execute correctly
- [ ] Pipeline hazards handled properly
- [ ] Caches function correctly
- [ ] Memory operations work
- [ ] Interrupts and exceptions handled
- [ ] Branch prediction works
- [ ] Register file operations correct
- [ ] Memristor emulation accurate

### 10.2 Performance Validation

- [ ] Measure clock frequency achieved
- [ ] Measure throughput (TOPS)
- [ ] Measure latency per operation
- [ ] Measure cache hit rates
- [ ] Measure branch prediction accuracy
- [ ] Measure power consumption
- [ ] Compare with simulation results

### 10.3 Software Validation

- [ ] Assembler generates correct code
- [ ] Programs load and execute
- [ ] Debugger can inspect state
- [ ] Libraries function correctly
- [ ] Neural networks run
- [ ] Benchmarks complete successfully

---

## 11. Cost Analysis

### 11.1 FPGA Prototyping Costs

| Item | Cost | Notes |
|------|------|-------|
| **FPGA Board** | $7,000 | VCU118 |
| **Vivado License** | $5,000/year | Design Edition |
| **Development Time** | $50,000 | 3 months @ $200K/year |
| **Test Equipment** | $10,000 | Oscilloscope, logic analyzer |
| **Misc** | $3,000 | Cables, adapters, etc. |
| **Total** | **$75,000** | First prototype |

### 11.2 Cost Reduction

**Academic Discount**: 80-90% off tools  
**Open Source Tools**: Use Yosys + nextpnr (free)  
**Shared Equipment**: Use university lab  
**Estimated Academic Cost**: $10,000-$15,000

---

## 12. Timeline

### 12.1 Detailed Schedule

```
Week 1-2: Setup and Single Module
  - Set up Vivado project
  - Implement PentaryALU
  - Synthesize and test
  - Validate on FPGA
  
Week 3-4: Core Components
  - Add register file
  - Add basic pipeline
  - Integrate and test
  - Debug issues
  
Week 5-6: Memory System
  - Implement L1 caches
  - Add memory controller
  - Test memory operations
  - Optimize performance
  
Week 7-8: Memristor Emulation
  - Implement crossbar emulation
  - Test matrix-vector multiply
  - Validate accuracy
  - Measure performance
  
Week 9-10: Integration
  - Integrate all components
  - System-level testing
  - Performance benchmarking
  - Bug fixes
  
Week 11-12: Optimization
  - Timing optimization
  - Resource optimization
  - Performance tuning
  - Final validation
```

### 12.2 Milestones

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| **M1: ALU Working** | 2 | Single module validated |
| **M2: Core Running** | 4 | Simple programs execute |
| **M3: Memory Working** | 6 | Cache operations validated |
| **M4: NN Inference** | 8 | Neural networks run |
| **M5: Multi-Core** | 10 | Multiple cores working |
| **M6: Optimized** | 12 | Performance targets met |

---

## 13. Success Criteria

### 13.1 Functional Success

- ✅ All instructions execute correctly
- ✅ Pipeline operates without errors
- ✅ Caches provide correct data
- ✅ Neural networks produce correct results
- ✅ Accuracy within 1% of simulation

### 13.2 Performance Success

- ✅ Achieves >100 MHz clock frequency
- ✅ Demonstrates pentary arithmetic advantages
- ✅ Memristor emulation shows speedup potential
- ✅ Multi-core scaling demonstrated

### 13.3 Validation Success

- ✅ Matches simulation results
- ✅ Software toolchain works
- ✅ Debugger functional
- ✅ Ready for ASIC design

---

## 14. Transition to ASIC

### 14.1 Lessons from FPGA

**What to Keep**:
- Validated architecture
- Proven functionality
- Optimized algorithms
- Tested software

**What to Change**:
- Increase clock frequency (25×)
- Optimize for ASIC (custom cells)
- Add real memristors
- Reduce power (10×)

### 14.2 FPGA to ASIC Checklist

- [ ] Document all FPGA-specific modifications
- [ ] Create ASIC-specific constraints
- [ ] Replace BRAM with SRAM
- [ ] Replace DSP with custom arithmetic
- [ ] Remove FPGA-specific debug logic
- [ ] Add DFT (Design for Test) logic
- [ ] Optimize for target process (7nm)
- [ ] Verify with ASIC tools

---

## 15. Conclusion

### Key Takeaways:
- ✅ **FPGA prototyping essential** for validation before ASIC
- ✅ **3-6 months** to working FPGA prototype
- ✅ **$75K cost** ($10-15K with academic discount)
- ✅ **Xilinx VCU118** recommended platform
- ✅ **Validates functionality**, not final performance
- ✅ **De-risks ASIC** tape-out

### Critical Success Factors:
1. Proper memristor emulation
2. Incremental validation approach
3. Comprehensive testing
4. Performance measurement
5. Software integration

**FPGA prototyping is a critical step that validates the pentary design before committing to expensive ASIC fabrication. It provides confidence that the design works and identifies issues early.**

---

**Document Status**: Complete FPGA Guide  
**Last Updated**: Current Session  
**Next Review**: After FPGA prototype completion