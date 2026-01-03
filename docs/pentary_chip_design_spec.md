
# Pentary Chip Design Specification

**Version**: 1.0  
**Date**: January 3, 2026  
**Author**: Manus AI

## 1. Overview

This document provides a detailed specification for the Pentary chip, a neural network accelerator based on balanced quinary arithmetic. The design is based on the research and Verilog implementations found in the `Kaleaon/Pentary` repository.

---

## 2. Top-Level Architecture

The chip is an 8-core processor with a shared L3 cache and integrated **Pentary Tensor Cores (PTCs)** for AI acceleration. Each core has its own L1 and L2 caches and a full Pentary ALU. The PTCs are digital systolic arrays built from standard CMOS logic.

---

## 3. Pentary Arithmetic Logic Unit (ALU)

**Source File**: `hardware/pentary_alu_fixed.v`

### 3.1. Overview

The Pentary ALU is a 48-bit (16-pent) arithmetic and logic unit that forms the computational core of the processor. It is a purely combinational block, designed for single-cycle execution of all its supported operations.

### 3.2. Interface

```verilog
module PentaryALU (
    input  [47:0] operand_a,      // 16 pentary digits (48 bits)
    input  [47:0] operand_b,      // 16 pentary digits (48 bits)
    input  [2:0]  opcode,         // Operation code
    output [47:0] result,         // 16 pentary digits (48 bits)
    output        zero_flag,      // Set if result == 0
    output        negative_flag,  // Set if result < 0
    output        overflow_flag,  // Set if overflow occurred
    output        equal_flag,     // Set if operand_a == operand_b
    output        greater_flag    // Set if operand_a > operand_b
);
```

### 3.3. Operations

The ALU supports the following 8 operations, selected by the `opcode` input:

| Opcode | Mnemonic | Description                  |
|--------|----------|------------------------------|
| 3'b000 | ADD      | Addition (a + b)             |
| 3'b001 | SUB      | Subtraction (a - b)          |
| 3'b010 | MUL2     | Multiply by 2 (shift left)   |
| 3'b011 | DIV2     | Divide by 2 (shift right)    |
| 3'b100 | NEG      | Negation (-a)                |
| 3'b101 | ABS      | Absolute Value (|a|)         |
| 3'b110 | CMP      | Compare (sets flags from a-b)|
| 3'b111 | MAX      | Maximum (max(a, b))          |

### 3.4. Sub-Modules

The ALU is constructed from several smaller, specialized helper modules:

- **`PentaryAdder16`**: A 16-pent (48-bit) adder that forms the basis for addition and subtraction.
- **`PentaryNegate`**: Negates a 16-pent number by inverting the sign of each non-zero pent.
- **`PentaryShiftLeft`**: Performs a logical left shift, equivalent to multiplication by 5.
- **`PentaryShiftRight`**: Performs a logical right shift, equivalent to division by 5.
- **`PentaryAbs`**: Computes the absolute value.
- **`PentaryMax`**: Determines the greater of two pentary numbers.
- **`PentaryIsZero`**: Checks if a 16-pent number is equal to zero.
- **`PentaryIsNegative`**: Determines the sign of a pentary number by checking the most significant non-zero pent.

### 3.5. Flag Generation

- **`zero_flag`**: Active high when all 16 pents of the result are zero.
- **`negative_flag`**: Active high when the most significant non-zero pent of the result is negative.
- **`overflow_flag`**: Active high if an addition or subtraction operation results in a carry-out from the most significant pent, indicating the result has exceeded the representable range.
- **`equal_flag`**: Active high when `operand_a` and `operand_b` are identical. Derived from the `zero_flag` of a compare operation.
- **`greater_flag`**: Active high when `operand_a` is greater than `operand_b`. Derived from the sign of a compare operation.

---

## 4. Register File

**Source File**: `hardware/register_file.v`

### 4.1. Overview

The register file is a core component of the processor, providing fast access to operand data. The design implements a 32-entry, 48-bit wide register file with dual read ports and a single write port to support the processor's 5-stage pipeline.

### 4.2. Interface

```verilog
module RegisterFile (
    input         clk,
    input         reset,
    
    // Read ports
    input  [4:0]  read_addr1,       // First read address (0-31)
    input  [4:0]  read_addr2,       // Second read address (0-31)
    output [47:0] read_data1,       // First read data
    output [47:0] read_data2,       // Second read data
    
    // Write port
    input  [4:0]  write_addr,       // Write address (0-31)
    input  [47:0] write_data,       // Write data
    input         write_enable      // Write enable
);
```

### 4.3. Features

- **Capacity**: 32 registers (R0-R31), each holding one 16-pent (48-bit) value.
- **Hardwired Zero**: Register R0 is hardwired to the value zero and cannot be written to. This is a common feature in RISC architectures to simplify instruction encoding.
- **Dual Read Ports**: Two independent read ports allow the ALU to access two source operands simultaneously in a single clock cycle.
- **Single Write Port**: One write port allows the result of an operation to be written back to the register file.
- **Bypass Logic**: The read logic includes a bypass path (also known as forwarding) to mitigate data hazards. If a read operation targets a register that is currently being written in the same cycle, the new data is forwarded directly to the read port, preventing a pipeline stall.
- **Synchronous Write**: All write operations are synchronized to the rising edge of the clock.
- **Asynchronous Reset**: The register file can be reset asynchronously, which initializes all registers to zero.

### 4.4. Extended Versions

The `register_file.v` file also contains more advanced concepts for future development, including:

- **`ExtendedRegisterFile`**: An extension that includes special-purpose registers like the Program Counter (PC), Status Register (SR), and exception handling registers.
- **`RegisterFileWithScoreboard`**: An advanced version that incorporates a scoreboard to enable out-of-order execution by tracking which registers have pending writes.
- **`MultiBankedRegisterFile`**: A parameterized, multi-banked design that can provide higher bandwidth for SIMD or multi-threaded cores.

These extended versions are not part of the baseline chip design but are included as a roadmap for future enhancements.

---
## 5. Pentary Tensor Core (PTC)

### 5.1. Overview

The Pentary Tensor Core is a digital systolic array designed for massively parallel matrix-vector multiplication, which is the cornerstone of neural network inference. It replaces the experimental memristor concept with a robust, standard-CMOS implementation.

### 5.2. Architecture

- **Systolic Array**: The PTC is structured as a 16x16 grid of Processing Elements (PEs).
- **Processing Element (PE)**: Each PE contains a specialized, single-operation Pentary ALU configured for Multiply-Accumulate (MAC) operations. It takes a weight and an activation as inputs, multiplies them, and adds the result to the value from the adjacent PE.
- **Dataflow**: Weights are pre-loaded into the PE grid. Activations are streamed in from one side, and the accumulated results are streamed out from the bottom, achieving high throughput and data reuse.
- **On-Chip Memory**: The PTC is tightly coupled with dedicated SRAM banks that act as weight and activation caches, minimizing latency and reliance on the main memory bus.

### 5.3. Performance

- **Peak Throughput**: A single 16x16 PTC can perform 256 pentary MAC operations per clock cycle.
- **Scalability**: Multiple PTCs can be instantiated on the chip to scale performance. The baseline design includes 4 PTCs, providing a total of 1024 MACs/cycle.

---

## 6. Cache Hierarchy

**Source File**: `hardware/cache_hierarchy.v`

### 5.1. Overview

The Pentary processor implements a three-level cache hierarchy to bridge the speed gap between the fast processor cores and slower main memory. The hierarchy consists of separate L1 instruction and data caches for each core, and a larger, shared L2 unified cache.

### 5.2. L1 Instruction Cache (I-Cache)

- **Size**: 32KB
- **Associativity**: 4-way set associative
- **Line Size**: 64 bytes
- **Policy**: Read-only. On a cache miss, a state machine fetches the required line from the L2 cache.

### 5.3. L1 Data Cache (D-Cache)

- **Size**: 32KB
- **Associativity**: 4-way set associative
- **Line Size**: 64 bytes
- **Policy**: Write-back. A `dirty` bit for each cache line tracks whether the line has been modified. On a miss, if the line to be replaced is dirty, it is written back to the L2 cache before the new line is fetched.

### 5.4. L2 Unified Cache

- **Size**: 256KB
- **Associativity**: 8-way set associative
- **Line Size**: 64 bytes
- **Policy**: Unified, serving both the L1 I-Cache and L1 D-Cache. It includes an arbiter to handle simultaneous requests from the L1 caches. It also uses a write-back policy for dealing with main memory.

### 5.5. Address Decomposition

The 48-bit memory address is decomposed into three parts to navigate the cache structure:

- **Tag**: The most significant bits of the address, used to identify a unique memory line within a cache set.
- **Index**: The middle bits, used to select the set within the cache where the data might reside.
- **Offset**: The least significant bits, used to select the specific byte or word within a cache line.

### 5.6. Cache Coherence

The provided Verilog does not explicitly implement a cache coherence protocol (e.g., MESI). In a multi-core system, this would be a critical addition to ensure that all cores have a consistent view of memory. For the current single-core implementation, the write-back policy is sufficient.

---
