# Pentary Titans FPGA Prototype - Detailed Design

## Overview

This document provides detailed specifications for implementing the Pentary Titans architecture on FPGA hardware for validation and prototyping.

---

## 1. Platform Selection

### Recommended: Xilinx Versal Premium VP1902

**Key Features:**
- 9M logic cells
- 7,968 DSP slices
- 187 Mb Block RAM
- 360 Mb UltraRAM
- 400 AI Engines
- 32 GB HBM2e
- PCIe Gen5 x16
- Price: ~$50,000

**Rationale:**
- Sufficient resources for full Titans implementation
- AI Engines for LTM emulation
- HBM2e for high-bandwidth memory
- PCIe Gen5 for host communication

### Alternative: Xilinx Alveo U280

**Key Features:**
- 2.6M logic cells
- 9,024 DSP slices
- 132 Mb Block RAM
- 8 GB HBM2
- PCIe Gen4 x16
- Price: ~$10,000

**Rationale:**
- Lower cost for initial prototyping
- Sufficient for smaller models (125M-1B)
- Proven platform with good tooling

---

## 2. Architecture Implementation

### 2.1 Resource Allocation

**Xilinx Versal VP1902:**

| Component | Logic Cells | DSP Slices | BRAM (Mb) | UltraRAM (Mb) | AI Engines |
|-----------|-------------|------------|-----------|---------------|------------|
| **Pentary Cores (8)** | 4.0M | 4,000 | 80 | 0 | 0 |
| **Attention Engines (4)** | 1.5M | 2,000 | 40 | 0 | 200 |
| **Surprise Units (8)** | 0.5M | 1,000 | 20 | 0 | 100 |
| **LTM Emulation** | 1.0M | 500 | 10 | 200 | 100 |
| **Memory Controllers** | 0.5M | 200 | 10 | 0 | 0 |
| **PCIe Interface** | 0.3M | 100 | 5 | 0 | 0 |
| **Control Logic** | 0.2M | 168 | 22 | 0 | 0 |
| **Total Used** | 8.0M | 7,968 | 187 | 200 | 400 |
| **Total Available** | 9.0M | 7,968 | 187 | 360 | 400 |
| **Utilization** | 89% | 100% | 100% | 56% | 100% |

### 2.2 Clock Domains

| Domain | Frequency | Components |
|--------|-----------|------------|
| **Core Clock** | 250 MHz | Pentary cores, ALUs |
| **Memory Clock** | 500 MHz | HBM2e interface |
| **PCIe Clock** | 250 MHz | PCIe Gen5 interface |
| **AI Engine Clock** | 1 GHz | AI Engines (LTM) |

**Note:** FPGA clocks are lower than ASIC targets (2.5 GHz) due to FPGA routing delays.

### 2.3 Pentary Core Implementation

**Verilog Module Structure:**

```verilog
module pentary_core (
    input wire clk,
    input wire rst_n,
    
    // Instruction interface
    input wire [47:0] instruction,  // 16 pents × 3 bits/pent
    input wire instruction_valid,
    output wire instruction_ready,
    
    // Data interface
    input wire [47:0] data_in,
    output wire [47:0] data_out,
    output wire data_valid,
    
    // Memory interface
    output wire [47:0] mem_addr,
    output wire [47:0] mem_wdata,
    input wire [47:0] mem_rdata,
    output wire mem_we,
    output wire mem_re
);

// Internal components
pentary_alu alu_inst (
    .clk(clk),
    .rst_n(rst_n),
    .op_a(alu_op_a),
    .op_b(alu_op_b),
    .operation(alu_operation),
    .result(alu_result),
    .flags(alu_flags)
);

pentary_register_file reg_file_inst (
    .clk(clk),
    .rst_n(rst_n),
    .rd_addr1(rd_addr1),
    .rd_addr2(rd_addr2),
    .wr_addr(wr_addr),
    .wr_data(wr_data),
    .wr_en(wr_en),
    .rd_data1(rd_data1),
    .rd_data2(rd_data2)
);

// ... additional components

endmodule
```

### 2.4 Attention Engine Implementation

**Using AI Engines:**

```verilog
module pentary_attention_engine (
    input wire clk,
    input wire rst_n,
    
    // Input: Query, Key, Value matrices
    input wire [47:0] query [0:31],    // 32 heads
    input wire [47:0] key [0:4095],    // 4K context
    input wire [47:0] value [0:4095],
    
    // Output: Attention output
    output wire [47:0] attention_out [0:31],
    output wire valid
);

// Use AI Engines for matrix multiplication
// Q × K^T computation
ai_engine_array qk_mult (
    .clk(clk),
    .matrix_a(query),
    .matrix_b(key_transposed),
    .result(attention_scores)
);

// Pentary softmax approximation
pentary_softmax softmax_inst (
    .scores(attention_scores),
    .weights(attention_weights)
);

// Attention × V computation
ai_engine_array av_mult (
    .clk(clk),
    .matrix_a(attention_weights),
    .matrix_b(value),
    .result(attention_out)
);

endmodule
```

### 2.5 Surprise Metric Unit Implementation

**Using DSP Slices:**

```verilog
module pentary_surprise_unit (
    input wire clk,
    input wire rst_n,
    
    // Inputs
    input wire [47:0] memory_state [0:2047],  // LTM parameters
    input wire [47:0] input_token,
    input wire [47:0] momentum_buffer [0:2047],
    
    // Outputs
    output wire [2:0] surprise_score,  // Pentary level {⊖, -, 0, +, ⊕}
    output wire [47:0] gradient [0:2047],
    output wire valid
);

// Forward pass through LTM
pentary_mlp_forward ltm_forward (
    .weights(memory_state),
    .input(input_token),
    .output(prediction)
);

// Compute error (surprise)
pentary_subtractor error_comp (
    .a(input_token),
    .b(prediction),
    .result(error)
);

// Compute gradient using DSP slices
pentary_backprop gradient_comp (
    .error(error),
    .weights(memory_state),
    .gradient(gradient)
);

// Apply momentum
pentary_momentum momentum_inst (
    .gradient(gradient),
    .momentum_buffer(momentum_buffer),
    .updated_momentum(updated_momentum)
);

// Compute surprise score
pentary_magnitude surprise_comp (
    .vector(updated_momentum),
    .magnitude(surprise_score)
);

endmodule
```

### 2.6 Long-Term Memory Emulation

**Using AI Engines and UltraRAM:**

```verilog
module pentary_ltm_module (
    input wire clk,
    input wire rst_n,
    
    // Input
    input wire [47:0] input_vector [0:2047],
    
    // Output
    output wire [47:0] output_vector [0:2047],
    output wire valid,
    
    // Memory update interface
    input wire update_enable,
    input wire [47:0] gradient [0:33554431],  // 33.5M parameters
    input wire [15:0] learning_rate
);

// Layer 1: 2048 → 4096
ai_engine_array layer1 (
    .clk(clk),
    .input(input_vector),
    .weights(layer1_weights),  // Stored in UltraRAM
    .output(layer1_output)
);

pentary_relu relu1 (
    .input(layer1_output),
    .output(layer1_activated)
);

// Layer 2: 4096 → 4096
ai_engine_array layer2 (
    .clk(clk),
    .input(layer1_activated),
    .weights(layer2_weights),
    .output(layer2_output)
);

pentary_relu relu2 (
    .input(layer2_output),
    .output(layer2_activated)
);

// Layer 3: 4096 → 2048
ai_engine_array layer3 (
    .clk(clk),
    .input(layer2_activated),
    .weights(layer3_weights),
    .output(output_vector)
);

// Memory update logic
pentary_memory_updater updater (
    .enable(update_enable),
    .weights(all_weights),
    .gradient(gradient),
    .learning_rate(learning_rate),
    .updated_weights(updated_weights)
);

endmodule
```

---

## 3. Development Workflow

### 3.1 Tools Required

**Xilinx Vivado:**
- Version: 2024.1 or later
- License: Enterprise (for Versal Premium)
- Components: Synthesis, Implementation, Simulation

**Additional Tools:**
- ModelSim/Questa for simulation
- Python 3.10+ for testbenches
- Git for version control
- Jupyter for analysis

### 3.2 Development Steps

**Step 1: Module Development (Weeks 1-4)**
```bash
# Create project
vivado -mode batch -source create_project.tcl

# Implement pentary ALU
vivado -mode batch -source implement_alu.tcl

# Simulate and verify
vsim -do sim_alu.do

# Synthesize
vivado -mode batch -source synth_alu.tcl
```

**Step 2: Integration (Weeks 5-8)**
```bash
# Integrate all modules
vivado -mode batch -source integrate_system.tcl

# Run system simulation
vsim -do sim_system.do

# Timing analysis
vivado -mode batch -source timing_analysis.tcl
```

**Step 3: Implementation (Weeks 9-12)**
```bash
# Place and route
vivado -mode batch -source implement_design.tcl

# Generate bitstream
vivado -mode batch -source generate_bitstream.tcl

# Program FPGA
vivado -mode batch -source program_fpga.tcl
```

**Step 4: Validation (Weeks 13-16)**
```bash
# Load test firmware
./load_firmware.sh

# Run benchmarks
python run_benchmarks.py

# Collect results
python analyze_results.py
```

### 3.3 Testing Strategy

**Unit Tests:**
- Pentary ALU operations
- Attention engine functionality
- Surprise metric computation
- Memory update logic

**Integration Tests:**
- End-to-end token processing
- Long-context handling
- Memory update pipeline
- PCIe communication

**Performance Tests:**
- Throughput measurement
- Latency profiling
- Power consumption
- Resource utilization

**Validation Tests:**
- BABILong benchmark
- Perplexity measurement
- Accuracy verification
- Comparison with GPU baseline

---

## 4. Expected Results

### 4.1 Performance Targets

| Metric | Target | Baseline (GPU) | Improvement |
|--------|--------|----------------|-------------|
| **Throughput (1B)** | 100K tok/s | 30K tok/s | 3.3× |
| **Latency** | 10 μs | 33 μs | 3.3× |
| **Power** | 100W | 300W | 3× |
| **Context Length** | 2M tokens | 2M tokens | 1× |

**Note:** FPGA performance is lower than ASIC due to lower clock speeds and routing overhead. ASIC will achieve 5-10× better performance.

### 4.2 Validation Criteria

**Success Criteria:**
- ✅ Throughput ≥ 100K tokens/sec (1B model)
- ✅ Context length ≥ 2M tokens
- ✅ Power ≤ 100W
- ✅ Accuracy within 2% of GPU baseline
- ✅ BABILong accuracy ≥ 90%

**Stretch Goals:**
- 🎯 Throughput ≥ 150K tokens/sec
- 🎯 Context length ≥ 5M tokens
- 🎯 Power ≤ 80W
- 🎯 Accuracy matches GPU baseline
- 🎯 BABILong accuracy ≥ 95%

---

## 5. Bill of Materials

### FPGA Development Kit

| Item | Part Number | Quantity | Unit Price | Total |
|------|-------------|----------|------------|-------|
| **Versal VP1902 Board** | VPK180 | 1 | $50,000 | $50,000 |
| **Power Supply** | 12V 50A | 1 | $200 | $200 |
| **Cooling Fan** | 120mm PWM | 2 | $50 | $100 |
| **PCIe Cable** | Gen5 x16 | 1 | $100 | $100 |
| **Debug Probe** | JTAG | 1 | $500 | $500 |
| **Cables & Misc** | Various | - | $1,000 | $1,000 |
| **Total** | - | - | - | **$51,900** |

### Alternative (Alveo U280)

| Item | Part Number | Quantity | Unit Price | Total |
|------|-------------|----------|------------|-------|
| **Alveo U280 Card** | XCU280 | 1 | $10,000 | $10,000 |
| **Host PC** | Workstation | 1 | $3,000 | $3,000 |
| **Cables & Misc** | Various | - | $500 | $500 |
| **Total** | - | - | - | **$13,500** |

---

## 6. Development Timeline

### Month 1-3: Core Implementation
- Week 1-2: Pentary ALU and arithmetic units
- Week 3-4: Register file and control logic
- Week 5-6: Attention engine (basic)
- Week 7-8: Surprise metric unit (basic)
- Week 9-10: Integration and testing
- Week 11-12: Optimization and debugging

**Deliverable:** Functional pentary core with basic Titans components

### Month 4-6: Memory System
- Week 13-14: LTM emulation in AI Engines
- Week 15-16: Memory update controller
- Week 17-18: HBM2e interface
- Week 19-20: Memory update pipeline
- Week 21-22: Integration and testing
- Week 23-24: Optimization and validation

**Deliverable:** Complete memory system with LTM updates

### Month 7-9: Integration & Optimization
- Week 25-26: Full system integration
- Week 27-28: Clock domain crossing
- Week 29-30: Timing optimization
- Week 31-32: Resource optimization
- Week 33-34: Power optimization
- Week 35-36: Final integration testing

**Deliverable:** Fully integrated Titans system

### Month 10-12: Validation & Testing
- Week 37-38: BABILong benchmark
- Week 39-40: Perplexity measurement
- Week 41-42: Throughput testing
- Week 43-44: Power measurement
- Week 45-46: Comparison with GPU
- Week 47-48: Documentation and reporting

**Deliverable:** Validated prototype with benchmark results

---

## 7. Vivado Project Structure

```
pentary_titans_fpga/
├── src/
│   ├── rtl/
│   │   ├── pentary_core.v
│   │   ├── pentary_alu.v
│   │   ├── pentary_attention.v
│   │   ├── pentary_surprise.v
│   │   ├── pentary_ltm.v
│   │   ├── pentary_memory_update.v
│   │   └── top_level.v
│   ├── ip/
│   │   ├── pcie_gen5/
│   │   ├── hbm2e_controller/
│   │   └── ai_engine_config/
│   └── constraints/
│       ├── timing.xdc
│       ├── pinout.xdc
│       └── power.xdc
├── sim/
│   ├── testbenches/
│   │   ├── tb_pentary_core.v
│   │   ├── tb_attention.v
│   │   ├── tb_surprise.v
│   │   └── tb_system.v
│   └── scripts/
│       ├── run_sim.tcl
│       └── wave_config.wcfg
├── scripts/
│   ├── create_project.tcl
│   ├── synthesize.tcl
│   ├── implement.tcl
│   └── generate_bitstream.tcl
├── sw/
│   ├── driver/
│   │   ├── pentary_titans_driver.c
│   │   └── Makefile
│   ├── api/
│   │   ├── pentary_titans_api.py
│   │   └── setup.py
│   └── tests/
│       ├── test_throughput.py
│       ├── test_accuracy.py
│       └── test_power.py
└── docs/
    ├── design_spec.md
    ├── user_guide.md
    └── benchmark_results.md
```

---

## 8. Synthesis and Implementation

### 8.1 Synthesis Settings

```tcl
# synthesis.tcl
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
```

### 8.2 Implementation Settings

```tcl
# implementation.tcl
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreWithRemap [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
```

### 8.3 Timing Constraints

```tcl
# timing.xdc
# Core clock: 250 MHz
create_clock -period 4.0 -name core_clk [get_ports clk_core]

# Memory clock: 500 MHz
create_clock -period 2.0 -name mem_clk [get_ports clk_mem]

# PCIe clock: 250 MHz
create_clock -period 4.0 -name pcie_clk [get_ports clk_pcie]

# AI Engine clock: 1 GHz
create_clock -period 1.0 -name aie_clk [get_ports clk_aie]

# Clock domain crossing
set_clock_groups -asynchronous \
    -group [get_clocks core_clk] \
    -group [get_clocks mem_clk] \
    -group [get_clocks pcie_clk] \
    -group [get_clocks aie_clk]

# Input/output delays
set_input_delay -clock core_clk 0.5 [all_inputs]
set_output_delay -clock core_clk 0.5 [all_outputs]
```

---

## 9. Software Stack

### 9.1 Driver Architecture

**Linux Kernel Module:**

```c
// pentary_titans_driver.c
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/dma-mapping.h>

#define PENTARY_VENDOR_ID 0x1234
#define PENTARY_DEVICE_ID 0x5678

struct pentary_device {
    struct pci_dev *pdev;
    void __iomem *bar0;
    dma_addr_t dma_handle;
    void *dma_buffer;
    size_t buffer_size;
};

static int pentary_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    struct pentary_device *dev;
    int ret;
    
    // Allocate device structure
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;
    
    dev->pdev = pdev;
    pci_set_drvdata(pdev, dev);
    
    // Enable PCI device
    ret = pci_enable_device(pdev);
    if (ret)
        goto err_free;
    
    // Request memory regions
    ret = pci_request_regions(pdev, "pentary_titans");
    if (ret)
        goto err_disable;
    
    // Map BAR0
    dev->bar0 = pci_iomap(pdev, 0, 0);
    if (!dev->bar0) {
        ret = -ENOMEM;
        goto err_release;
    }
    
    // Set up DMA
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if (ret)
        goto err_unmap;
    
    // Allocate DMA buffer
    dev->buffer_size = 1024 * 1024 * 1024;  // 1 GB
    dev->dma_buffer = dma_alloc_coherent(&pdev->dev, dev->buffer_size,
                                          &dev->dma_handle, GFP_KERNEL);
    if (!dev->dma_buffer) {
        ret = -ENOMEM;
        goto err_unmap;
    }
    
    printk(KERN_INFO "Pentary Titans device initialized\n");
    return 0;
    
err_unmap:
    pci_iounmap(pdev, dev->bar0);
err_release:
    pci_release_regions(pdev);
err_disable:
    pci_disable_device(pdev);
err_free:
    kfree(dev);
    return ret;
}

static void pentary_remove(struct pci_dev *pdev) {
    struct pentary_device *dev = pci_get_drvdata(pdev);
    
    dma_free_coherent(&pdev->dev, dev->buffer_size,
                      dev->dma_buffer, dev->dma_handle);
    pci_iounmap(pdev, dev->bar0);
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    kfree(dev);
    
    printk(KERN_INFO "Pentary Titans device removed\n");
}

static struct pci_device_id pentary_ids[] = {
    { PCI_DEVICE(PENTARY_VENDOR_ID, PENTARY_DEVICE_ID) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, pentary_ids);

static struct pci_driver pentary_driver = {
    .name = "pentary_titans",
    .id_table = pentary_ids,
    .probe = pentary_probe,
    .remove = pentary_remove,
};

module_pci_driver(pentary_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Pentary Team");
MODULE_DESCRIPTION("Pentary Titans PCIe Driver");
```

### 9.2 Python API

```python
# pentary_titans_api.py
import ctypes
import numpy as np

class TitansCard:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.lib = ctypes.CDLL("libpentary_titans.so")
        self.handle = self.lib.pentary_init(device_id)
        
    def load_model(self, model_path):
        """Load Titans model onto card"""
        return self.lib.pentary_load_model(self.handle, model_path.encode())
    
    def generate(self, input_ids, max_length=100, surprise_threshold=1.5):
        """Generate tokens with Titans"""
        output = np.zeros(max_length, dtype=np.int32)
        self.lib.pentary_generate(
            self.handle,
            input_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(input_ids),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_length,
            ctypes.c_float(surprise_threshold)
        )
        return output
    
    def get_stats(self):
        """Get memory update statistics"""
        stats = {}
        stats['num_updates'] = self.lib.pentary_get_num_updates(self.handle)
        stats['avg_surprise'] = self.lib.pentary_get_avg_surprise(self.handle)
        return stats
    
    def __del__(self):
        self.lib.pentary_cleanup(self.handle)
```

---

## 10. Budget and Resources

### Development Budget

| Item | Cost |
|------|------|
| **FPGA Hardware** | $52,000 |
| **Software Licenses** | $10,000 |
| **Development Tools** | $5,000 |
| **Test Equipment** | $8,000 |
| **Engineering (5 people × 12 months)** | $600,000 |
| **Facilities** | $25,000 |
| **Total** | **$700,000** |

### Team Requirements

| Role | Count | Duration |
|------|-------|----------|
| **FPGA Engineer** | 2 | 12 months |
| **Software Engineer** | 1 | 12 months |
| **Verification Engineer** | 1 | 12 months |
| **Project Manager** | 1 | 12 months |

---

## Conclusion

This FPGA prototype design provides a comprehensive roadmap for validating the Pentary Titans architecture. With a 12-month timeline and $700K budget, we can demonstrate 3-10× performance improvements over GPU baselines and validate the path to ASIC production.

**Next Steps:**
1. Acquire FPGA hardware
2. Set up development environment
3. Begin module implementation
4. Validate performance claims

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Status:** Complete