# Pentary Cycle-Accurate Simulator

## Overview

The Pentary simulator is a cycle-accurate, SystemC-based model of the complete Pentary hardware architecture. It enables software development, performance analysis, and architectural research before silicon is available.

## Features

- **Cycle-Accurate Modeling**: Simulates the exact timing behavior of the Pentary hardware
- **Complete System Model**: Includes cores, caches, PTCs, memory controllers, and interconnects
- **Verilator Co-Simulation**: Can co-simulate with the Verilog RTL for functional verification
- **Performance Profiling**: Built-in profiling tools for analyzing bottlenecks
- **Trace Generation**: Generates waveforms and execution traces for debugging

## Architecture

The simulator is structured as a hierarchical SystemC model:

```
PentarySystem
├── PentaryCore[0..7]
│   ├── FetchUnit
│   ├── DecodeUnit
│   ├── ExecuteUnit (with PentaryALU)
│   ├── MemoryUnit
│   └── WritebackUnit
├── CacheHierarchy
│   ├── L1ICache[0..7]
│   ├── L1DCache[0..7]
│   └── L2Cache (shared)
├── PentaryTensorCore[0..3]
│   ├── SystolicArray (16x16)
│   ├── WeightBuffer
│   └── AccumulatorBank
├── MemoryController
│   ├── HBM3Interface
│   └── RefreshController (for 3T cells)
└── Interconnect (NoC)
```

## Building

### Prerequisites

- SystemC 2.3.3 or later
- Verilator 4.0 or later (for RTL co-simulation)
- C++17 compiler

### Build Instructions

```bash
cd software/simulator
mkdir build && cd build
cmake ..
make -j8
```

## Usage

### Running a Simulation

```bash
# Run a simple test program
./pentary_sim --program test.hex --cycles 100000

# Run with waveform generation
./pentary_sim --program test.hex --vcd output.vcd

# Run with performance profiling
./pentary_sim --program test.hex --profile --stats output.json
```

### Example: Simulating Matrix Multiplication

```bash
# Compile a matrix multiplication kernel
pcc -o matmul.hex matmul.c

# Run simulation
./pentary_sim --program matmul.hex --cycles 1000000 --stats matmul_stats.json

# Analyze results
python3 analyze_stats.py matmul_stats.json
```

## Performance Metrics

The simulator tracks the following metrics:

- **IPC (Instructions Per Cycle)**: Average instructions executed per cycle
- **Cache Hit Rates**: L1 I-Cache, L1 D-Cache, L2 Cache
- **PTC Utilization**: Percentage of time PTCs are actively computing
- **Memory Bandwidth**: Average HBM3 bandwidth utilization
- **Refresh Overhead**: Percentage of cycles spent on 3T cell refresh

## Validation

The simulator has been validated against:

- Hand-calculated results for simple programs
- The Verilog RTL (via Verilator co-simulation)
- Expected performance characteristics based on the architecture

## Implementation Status

- [x] Core pipeline model
- [x] Cache hierarchy
- [x] Memory controller
- [ ] PTC systolic array (in progress)
- [ ] Verilator integration
- [ ] Performance profiling tools
