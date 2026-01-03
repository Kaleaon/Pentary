# Pentary Instruction Set Architecture (ISA) Specification

**Version**: 1.0  
**Date**: January 3, 2026

## 1. Overview

The Pentary ISA is a custom, balanced quinary instruction set designed for efficient AI and scientific computing workloads. It features a load-store architecture with 32 general-purpose registers and specialized instructions for tensor operations.

## 2. Data Types

| Type | Size (Pents) | Size (Bits Equivalent) | Description |
|------|--------------|------------------------|-------------|
| PENT | 1 | ~2.32 bits | Single pentary digit (-2, -1, 0, +1, +2) |
| WORD | 16 | 48 bits | Standard word size |
| DWORD | 32 | 96 bits | Double word |

## 3. Registers

### 3.1. General Purpose Registers

- **R0-R31**: 32 registers, each 16 pents (48 bits)
- **R0**: Hardwired to zero (reads always return 0, writes are ignored)
- **R1-R31**: General purpose

### 3.2. Special Registers

- **PC**: Program Counter (16 pents)
- **SP**: Stack Pointer (16 pents, alias for R29)
- **FP**: Frame Pointer (16 pents, alias for R30)
- **LR**: Link Register (16 pents, alias for R31)

## 4. Instruction Format

All instructions are 16 pents (48 bits) wide.

### R-Type (Register)
```
[OPCODE:4][RD:5][RS1:5][RS2:5][FUNCT:11][UNUSED:18]
```

### I-Type (Immediate)
```
[OPCODE:4][RD:5][RS1:5][IMM:34]
```

### M-Type (Memory)
```
[OPCODE:4][RD:5][RS1:5][OFFSET:34]
```

### B-Type (Branch)
```
[OPCODE:4][RS1:5][RS2:5][OFFSET:34]
```

## 5. Instruction Set

### 5.1. Arithmetic Instructions

| Mnemonic | Format | Description |
|----------|--------|-------------|
| ADD | R | RD = RS1 + RS2 |
| SUB | R | RD = RS1 - RS2 |
| MUL | R | RD = RS1 × RS2 |
| DIV | R | RD = RS1 ÷ RS2 |
| MOD | R | RD = RS1 % RS2 |
| ADDI | I | RD = RS1 + IMM |
| SUBI | I | RD = RS1 - IMM |
| MULI | I | RD = RS1 × IMM |

### 5.2. Logical Instructions

| Mnemonic | Format | Description |
|----------|--------|-------------|
| AND | R | RD = RS1 & RS2 (bitwise AND) |
| OR | R | RD = RS1 \| RS2 (bitwise OR) |
| XOR | R | RD = RS1 ^ RS2 (bitwise XOR) |
| NOT | R | RD = ~RS1 (bitwise NOT) |
| SHL | R | RD = RS1 << RS2 (shift left) |
| SHR | R | RD = RS1 >> RS2 (shift right) |

### 5.3. Memory Instructions

| Mnemonic | Format | Description |
|----------|--------|-------------|
| LOAD | M | RD = MEM[RS1 + OFFSET] |
| STORE | M | MEM[RS1 + OFFSET] = RD |
| LOADH | M | Load half-word (8 pents) |
| STOREH | M | Store half-word (8 pents) |

### 5.4. Control Flow Instructions

| Mnemonic | Format | Description |
|----------|--------|-------------|
| BEQ | B | Branch if RS1 == RS2 |
| BNE | B | Branch if RS1 != RS2 |
| BLT | B | Branch if RS1 < RS2 |
| BGE | B | Branch if RS1 >= RS2 |
| JUMP | I | PC = IMM |
| JUMPR | R | PC = RS1 |
| CALL | I | LR = PC + 1; PC = IMM |
| RET | R | PC = LR |

### 5.5. Tensor Instructions

| Mnemonic | Format | Description |
|----------|--------|-------------|
| TLOAD | M | Load tensor data to PTC |
| TSTORE | M | Store tensor result from PTC |
| TGEMM | R | Dispatch GEMM to PTC |
| TCONV | R | Dispatch convolution to PTC |
| TSYNC | R | Synchronize PTC execution |

## 6. Calling Convention

### 6.1. Register Usage

- **R0**: Zero register
- **R1-R8**: Argument registers (first 8 arguments)
- **R9-R15**: Temporary registers (caller-saved)
- **R16-R28**: Saved registers (callee-saved)
- **R29 (SP)**: Stack pointer
- **R30 (FP)**: Frame pointer
- **R31 (LR)**: Link register (return address)

### 6.2. Stack Frame Layout

```
High Address
+------------------+
| Arguments 9+     |
+------------------+
| Return Address   | <- FP
+------------------+
| Saved FP         |
+------------------+
| Local Variables  |
+------------------+
| Saved Registers  |
+------------------+ <- SP
Low Address
```

## 7. Example Code

### Hello World (Conceptual)

```asm
.section .data
msg:    .string "Hello, Pentary!\n"

.section .text
.global _start

_start:
    ADDI    R1, R0, msg      # R1 = address of msg
    ADDI    R2, R0, 16       # R2 = length
    ADDI    R3, R0, 1        # R3 = stdout
    CALL    write            # Call write syscall
    
    ADDI    R1, R0, 0        # Exit code 0
    CALL    exit             # Exit program
```

### Matrix Multiplication (Using PTC)

```asm
matmul:
    # R1 = A matrix address
    # R2 = B matrix address
    # R3 = C matrix address (output)
    # R4 = M (rows of A)
    # R5 = K (cols of A, rows of B)
    # R6 = N (cols of B)
    
    TLOAD   R1, 0            # Load A to PTC
    TLOAD   R2, 1            # Load B to PTC
    TGEMM   R4, R5, R6       # Execute GEMM
    TSYNC                    # Wait for completion
    TSTORE  R3, 0            # Store result to C
    RET
```

## 8. Notes

- All arithmetic operations use balanced quinary representation
- Multiplication by {-2, -1, 0, +1, +2} is optimized to shift-and-add
- The ISA is designed to map efficiently to the Pentary ALU hardware
