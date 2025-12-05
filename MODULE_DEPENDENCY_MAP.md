# Pentary Chip Design - Module Dependency Map

## Visual Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pentary Neural Network Chip                      │
│                              (Top Level)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼──────────┐      ┌────────────▼─────────────┐
        │   PentaryNNCore      │      │  Memory Hierarchy        │
        │   (8 cores)          │      │  (L1/L2/L3 Caches)       │
        └───────────┬──────────┘      └────────────┬─────────────┘
                    │                               │
        ┌───────────┴───────────┐                  │
        │                       │                  │
┌───────▼────────┐    ┌────────▼──────────┐      │
│  Control Unit  │    │  Datapath         │      │
│  (Pipeline)    │    │                   │      │
└───────┬────────┘    └────────┬──────────┘      │
        │                      │                  │
        │         ┌────────────┴──────────┐       │
        │         │                       │       │
┌───────▼─────────▼──────┐    ┌──────────▼───────▼──────┐
│  Instruction Fetch     │    │  Memory Access          │
│  & Decode              │    │  Unit                   │
└────────────────────────┘    └─────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
        ┌───────────▼──────────┐              ┌────────────────▼────────────┐
        │   Register File      │              │  MemristorCrossbarController│
        │   (32 × 16-pent)     │              │  (256×256 array)            │
        └───────────┬──────────┘              └────────────────┬────────────┘
                    │                                          │
        ┌───────────┴───────────┐                             │
        │                       │                             │
┌───────▼────────┐    ┌────────▼──────────┐    ┌─────────────▼────────────┐
│  PentaryALU    │    │  PentaryReLU      │    │  Memristor Array         │
│                │    │                   │    │  (Analog Compute)        │
└───────┬────────┘    └───────────────────┘    └──────────────────────────┘
        │
┌───────▼────────────────────┐
│  PentaryAdder              │
│  (16-digit with carry)     │
└────────────────────────────┘
```

## Module Hierarchy

### Level 0: Top Level
- **PentaryChip** (Not yet implemented)
  - Integrates all cores, memory, and I/O
  - Manages inter-core communication
  - Handles system-level power management

### Level 1: Core Components
1. **PentaryNNCore** (Partially implemented)
   - Main neural network accelerator core
   - Contains pipeline, registers, and execution units
   - **Status**: ⚠️ Basic structure, needs expansion
   - **Dependencies**: All Level 2 modules

2. **MemoryHierarchy** (Not implemented)
   - L1 Instruction Cache (32KB)
   - L1 Data Cache (32KB)
   - L2 Unified Cache (256KB)
   - L3 Shared Cache (8MB)
   - **Status**: ❌ Not implemented
   - **Dependencies**: Cache controller, coherency protocol

### Level 2: Execution Units
1. **PentaryALU** (Partially implemented)
   - Arithmetic and logic operations
   - **Status**: ⚠️ Needs bit width fixes
   - **Dependencies**: PentaryAdder, helper functions
   - **Used by**: PentaryNNCore

2. **MemristorCrossbarController** (Partially implemented)
   - Controls 256×256 memristor array
   - Matrix-vector multiplication
   - **Status**: ⚠️ Basic structure, needs MATVEC implementation
   - **Dependencies**: Memristor array model
   - **Used by**: PentaryNNCore

3. **PentaryReLU** (Implemented)
   - ReLU activation function
   - **Status**: ✓ Simple and correct
   - **Dependencies**: None
   - **Used by**: PentaryNNCore

4. **PentaryQuantizer** (Needs rework)
   - Quantizes values to pentary
   - **Status**: ❌ Uses non-synthesizable floating-point
   - **Dependencies**: Fixed-point arithmetic library
   - **Used by**: PentaryNNCore

5. **RegisterFile** (Not implemented)
   - 32 registers × 16 pentary digits
   - Dual-port read, single-port write
   - **Status**: ❌ Not implemented
   - **Dependencies**: None
   - **Used by**: PentaryNNCore

### Level 3: Arithmetic Primitives
1. **PentaryAdder** (Partially implemented)
   - Single-digit addition with carry
   - **Status**: ⚠️ Needs complete lookup table
   - **Dependencies**: None
   - **Used by**: PentaryALU

2. **PentaryConstantMultiplier** (Implemented)
   - Multiply by {-2, -1, 0, +1, +2}
   - **Status**: ✓ Well-structured
   - **Dependencies**: Helper functions
   - **Used by**: PentaryALU, MemristorCrossbar

3. **Helper Functions** (Partially implemented)
   - negate_pentary
   - shift_left_pentary
   - **Status**: ⚠️ Needs verification
   - **Dependencies**: None
   - **Used by**: Multiple modules

## Dependency Graph

```
PentaryNNCore
    ├── PentaryALU
    │   ├── PentaryAdder
    │   └── Helper Functions
    │       ├── negate_pentary
    │       └── shift_left_pentary
    ├── PentaryReLU
    ├── PentaryQuantizer
    │   └── Fixed-Point Arithmetic (TBD)
    ├── MemristorCrossbarController
    │   ├── Memristor Array Model (TBD)
    │   ├── PentaryConstantMultiplier
    │   └── Matrix-Vector Multiply Logic (TBD)
    ├── RegisterFile (TBD)
    ├── Pipeline Control (TBD)
    │   ├── Hazard Detection (TBD)
    │   ├── Data Forwarding (TBD)
    │   └── Branch Prediction (TBD)
    └── Cache Interface (TBD)
        ├── L1 I-Cache (TBD)
        ├── L1 D-Cache (TBD)
        └── Cache Coherency (TBD)
```

## Critical Path Analysis

### Current Critical Paths (Estimated)

1. **ALU Path**: 
   ```
   Register Read → PentaryAdder (16 stages) → Result Mux → Register Write
   Estimated: ~20 gate delays
   ```

2. **Memristor Path**:
   ```
   Register Read → Crossbar Access → Analog Compute → ADC → Accumulate → Register Write
   Estimated: ~50 gate delays (plus analog delay)
   ```

3. **Memory Path**:
   ```
   Address Gen → Cache Lookup → Tag Compare → Data Read → Register Write
   Estimated: ~15 gate delays
   ```

### Optimization Targets

To achieve 5 GHz operation (200ps cycle time):
- Each gate delay budget: ~10ps
- Maximum gates in critical path: ~20 gates
- **Current status**: Exceeds budget, needs pipelining

## Module Implementation Status

| Module | Status | Priority | Effort | Dependencies |
|--------|--------|----------|--------|--------------|
| PentaryALU | ⚠️ 40% | HIGH | 1 week | PentaryAdder |
| PentaryAdder | ⚠️ 30% | HIGH | 1 week | None |
| PentaryReLU | ✓ 90% | LOW | 1 day | None |
| PentaryQuantizer | ❌ 20% | MEDIUM | 1 week | Fixed-point lib |
| PentaryConstantMultiplier | ✓ 80% | LOW | 2 days | Helper functions |
| MemristorCrossbarController | ⚠️ 40% | HIGH | 2 weeks | Memristor model |
| PentaryNNCore | ⚠️ 30% | HIGH | 2 weeks | All above |
| RegisterFile | ❌ 0% | HIGH | 1 week | None |
| Pipeline Control | ❌ 0% | HIGH | 2 weeks | Hazard detection |
| Cache Hierarchy | ❌ 0% | HIGH | 3 weeks | Coherency protocol |
| MMU | ❌ 0% | MEDIUM | 2 weeks | TLB |
| Interrupt Controller | ❌ 0% | MEDIUM | 1 week | None |
| Debug Interface | ❌ 0% | LOW | 1 week | JTAG |
| Power Management | ❌ 0% | HIGH | 2 weeks | Clock gating |
| Error Correction | ❌ 0% | HIGH | 2 weeks | Pentary ECC |

## Implementation Order (Recommended)

### Phase 1: Core Arithmetic (Weeks 1-2)
1. Fix PentaryAdder (complete lookup table)
2. Fix PentaryALU (bit widths, operations)
3. Verify with comprehensive testbenches
4. **Milestone**: All arithmetic operations working

### Phase 2: Register File & Basic Pipeline (Weeks 3-4)
1. Implement RegisterFile module
2. Create basic 5-stage pipeline
3. Add simple hazard detection
4. **Milestone**: Simple programs can execute

### Phase 3: Memory System (Weeks 5-7)
1. Implement L1 caches
2. Add cache controller
3. Implement basic coherency
4. **Milestone**: Memory operations working

### Phase 4: Memristor Integration (Weeks 8-10)
1. Complete MemristorCrossbarController
2. Implement matrix-vector multiply
3. Add error correction
4. **Milestone**: Neural network inference working

### Phase 5: Optimization (Weeks 11-13)
1. Pipeline optimization
2. Critical path reduction
3. Power optimization
4. **Milestone**: Performance targets met

## Data Flow Diagram

```
Input Data
    │
    ▼
┌─────────────────┐
│  Instruction    │
│  Fetch          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Instruction    │
│  Decode         │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Register│ │  Immediate   │
│  Read  │ │  Generation  │
└───┬────┘ └──────┬───────┘
    │             │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │   Execute    │
    │   (ALU/      │
    │   Memristor) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Memory     │
    │   Access     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Write Back  │
    └──────┬───────┘
           │
           ▼
    Register File
```

## Interface Specifications

### PentaryALU Interface
```systemverilog
module PentaryALU (
    input  logic        clk,
    input  logic        reset,
    input  logic [47:0] operand_a,    // 16 pents
    input  logic [47:0] operand_b,    // 16 pents
    input  logic [2:0]  opcode,       // Operation
    output logic [47:0] result,       // 16 pents
    output logic        zero_flag,
    output logic        negative_flag,
    output logic        overflow_flag,
    output logic        ready
);
```

### MemristorCrossbarController Interface
```systemverilog
module MemristorCrossbarController (
    input  logic         clk,
    input  logic         reset,
    input  logic [7:0]   row_addr,
    input  logic [7:0]   col_addr,
    input  logic [2:0]   write_data,
    input  logic         write_enable,
    input  logic         compute_enable,
    input  logic [767:0] input_vector,   // 256 pents
    output logic [767:0] output_vector,  // 256 pents
    output logic         ready,
    output logic         error
);
```

### RegisterFile Interface
```systemverilog
module RegisterFile (
    input  logic        clk,
    input  logic        reset,
    input  logic [4:0]  read_addr1,
    input  logic [4:0]  read_addr2,
    input  logic [4:0]  write_addr,
    input  logic [47:0] write_data,
    input  logic        write_enable,
    output logic [47:0] read_data1,
    output logic [47:0] read_data2
);
```

## Testing Strategy per Module

### PentaryALU Testing
- Unit tests for each operation
- Corner case testing (overflow, underflow)
- Random testing with golden model
- Timing verification
- Power measurement

### MemristorCrossbar Testing
- Write/read verification
- Matrix-vector multiply accuracy
- Performance benchmarking
- Error injection testing
- Wear-leveling validation

### Integration Testing
- Full neural network inference
- Multi-core synchronization
- Cache coherency validation
- Power management verification
- Stress testing

## Resource Estimates

### Logic Resources (per core, 7nm)
- **PentaryALU**: ~5,000 gates
- **RegisterFile**: ~15,000 gates
- **Pipeline Control**: ~10,000 gates
- **Cache (L1)**: ~50,000 gates
- **MemristorController**: ~20,000 gates
- **Total Logic**: ~100,000 gates

### Memory Resources (per core)
- **Register File**: 32 × 48 bits = 1,536 bits
- **L1 I-Cache**: 32 KB = 262,144 bits
- **L1 D-Cache**: 32 KB = 262,144 bits
- **Memristor Array**: 256×256×3 bits = 196,608 bits
- **Total Memory**: ~720 Kbits

### Area Estimate (7nm)
- **Logic**: ~0.5 mm²
- **Memory**: ~0.75 mm²
- **Total per core**: ~1.25 mm²
- **8 cores + L3**: ~12 mm²

## Conclusion

This dependency map shows:
1. **Clear module hierarchy** with well-defined interfaces
2. **Critical dependencies** that must be resolved first
3. **Implementation order** for efficient development
4. **Resource estimates** for planning

**Next Steps:**
1. Implement modules in dependency order
2. Create testbenches for each module
3. Integrate incrementally
4. Validate at each step

---

**Document Status**: Complete
**Last Updated**: Current Session
**Next Review**: After Phase 1 completion