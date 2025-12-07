# Pentary Processor Architecture for chipIgnite

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Caravel SoC (Fixed)                      │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │  VexRiscV    │  │   Flash     │  │    GPIO      │      │
│  │     MCU      │──│ Controller  │──│  Controller  │      │
│  │  (1.5KB RAM) │  │             │  │   (38 pins)  │      │
│  └──────┬───────┘  └─────────────┘  └──────────────┘      │
│         │                                                    │
│    Wishbone Bus                                             │
│         │                                                    │
├─────────┼────────────────────────────────────────────────────┤
│         │         User Project Area (10mm²)                 │
│         │                                                    │
│  ┌──────▼──────────────────────────────────────────────┐   │
│  │          Pentary Processing Unit (PPU)              │   │
│  │                                                      │   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  │  Wishbone  │  │ Register │  │   Pentary     │  │   │
│  │  │ Interface  │──│   File   │──│     ALU       │  │   │
│  │  │            │  │ (25×60b) │  │  (Add/Mul/Div)│  │   │
│  │  └────────────┘  └──────────┘  └───────────────┘  │   │
│  │                                                      │   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  │   Control  │  │   L1     │  │   L1 Data     │  │   │
│  │  │    Unit    │──│I-Cache   │  │    Cache      │  │   │
│  │  │  (Decoder) │  │  (4KB)   │  │    (4KB)      │  │   │
│  │  └────────────┘  └──────────┘  └───────────────┘  │   │
│  │                                                      │   │
│  │  ┌────────────────────────────────────────────┐    │   │
│  │  │      Scratchpad RAM (32KB)                 │    │   │
│  │  └────────────────────────────────────────────┘    │   │
│  │                                                      │   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  │   Debug    │  │   IRQ    │  │     GPIO      │  │   │
│  │  │ Interface  │  │Controller│  │   Mux/Demux   │  │   │
│  │  └────────────┘  └──────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## 2. Pentary Encoding Scheme

### 2.1 Binary-Encoded Pentary (BEP)

Each pentary digit (0-4) is encoded using 3 binary bits:

```
Pentary Digit | Binary Encoding | Hex
--------------|-----------------|-----
      0       |      000        | 0x0
      1       |      001        | 0x1
      2       |      010        | 0x2
      3       |      011        | 0x3
      4       |      100        | 0x4
   (invalid)  |      101        | 0x5 (error)
   (invalid)  |      110        | 0x6 (error)
   (invalid)  |      111        | 0x7 (error)
```

### 2.2 Word Format

**20-digit pentary word = 60 binary bits:**

```
Bit Position:  59-57  56-54  53-51  ...  5-3   2-0
Pentary Digit:  D19    D18    D17   ...  D1    D0
Value Range:   0-4    0-4    0-4   ...  0-4   0-4
```

**Example:** Pentary number `12340` (base-5)
- D0 = 0 → `000`
- D1 = 4 → `100`
- D2 = 3 → `011`
- D3 = 2 → `010`
- D4 = 1 → `001`
- Binary: `001_010_011_100_000` (15 bits for 5 digits)

## 3. Instruction Set Architecture (ISA)

### 3.1 Instruction Format

**Type-R (Register-Register):**
```
| Opcode | Rd  | Rs1 | Rs2 | Func |
|  5 bits| 5b  | 5b  | 5b  | 5b   | = 25 bits (9 pentary digits)
```

**Type-I (Immediate):**
```
| Opcode | Rd  | Rs1 | Immediate    |
|  5 bits| 5b  | 5b  | 10 bits      | = 25 bits (9 pentary digits)
```

**Type-M (Memory):**
```
| Opcode | Rd  | Rs1 | Offset       |
|  5 bits| 5b  | 5b  | 10 bits      | = 25 bits (9 pentary digits)
```

### 3.2 Instruction Set (25 Base Instructions)

| Opcode | Mnemonic | Description | Format |
|--------|----------|-------------|--------|
| 00000 | NOP | No operation | - |
| 00001 | ADD | Add Rs1 + Rs2 → Rd | R |
| 00002 | SUB | Subtract Rs1 - Rs2 → Rd | R |
| 00003 | MUL | Multiply Rs1 × Rs2 → Rd | R |
| 00004 | DIV | Divide Rs1 ÷ Rs2 → Rd | R |
| 00010 | AND | Bitwise AND | R |
| 00011 | OR | Bitwise OR | R |
| 00012 | XOR | Bitwise XOR | R |
| 00013 | NOT | Bitwise NOT | R |
| 00014 | SHL | Shift left | R |
| 00020 | SHR | Shift right | R |
| 00021 | ROL | Rotate left | R |
| 00022 | ROR | Rotate right | R |
| 00023 | CMP | Compare | R |
| 00024 | MOV | Move register | R |
| 00030 | ADDI | Add immediate | I |
| 00031 | SUBI | Subtract immediate | I |
| 00032 | MULI | Multiply immediate | I |
| 00033 | ANDI | AND immediate | I |
| 00034 | ORI | OR immediate | I |
| 00040 | LD | Load from memory | M |
| 00041 | ST | Store to memory | M |
| 00042 | JMP | Jump | M |
| 00043 | BEQ | Branch if equal | M |
| 00044 | BNE | Branch if not equal | M |

### 3.3 Register File

**25 General-Purpose Registers (5² addressing):**
- R0-R23: General purpose (60 bits each)
- R24: Stack pointer (SP)
- R25: Program counter (PC) - implicit

**Special Registers:**
- Status Register (SR): Flags (Zero, Carry, Overflow, Negative)
- Control Register (CR): Processor control bits

## 4. Pipeline Architecture

### 4.1 5-Stage Pipeline

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Fetch   │──▶│  Decode  │──▶│ Execute  │──▶│  Memory  │──▶│Writeback │
│   (IF)   │   │   (ID)   │   │   (EX)   │   │   (MEM)  │   │   (WB)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  I-Cache      Register       Pentary         D-Cache       Register
  Access         File           ALU            Access         File
```

**Pipeline Stages:**
1. **IF (Instruction Fetch):** Fetch instruction from I-Cache
2. **ID (Instruction Decode):** Decode instruction, read registers
3. **EX (Execute):** Perform pentary arithmetic/logic operation
4. **MEM (Memory):** Access data cache if needed
5. **WB (Writeback):** Write result back to register file

### 4.2 Hazard Handling

**Data Hazards:**
- Forwarding paths from EX→EX, MEM→EX, WB→EX
- Stall on load-use hazards

**Control Hazards:**
- Branch prediction (static: predict not taken)
- 2-cycle branch penalty

## 5. Memory Subsystem

### 5.1 Cache Organization

**L1 Instruction Cache:**
- Size: 4KB (4096 bytes)
- Line size: 64 bytes (10.67 pentary words)
- Associativity: 2-way set associative
- Sets: 32 sets
- Replacement: LRU (Least Recently Used)

**L1 Data Cache:**
- Size: 4KB (4096 bytes)
- Line size: 64 bytes
- Associativity: 2-way set associative
- Sets: 32 sets
- Write policy: Write-back with write-allocate

**Scratchpad RAM:**
- Size: 32KB (32768 bytes)
- Access: Single-cycle (no cache)
- Use: Stack, local variables, DMA buffers

### 5.2 Memory Map

```
Address Range          | Description
-----------------------|---------------------------
0x00000000-0x00000FFF  | Boot ROM (via Wishbone)
0x00001000-0x00001FFF  | Management SRAM (via WB)
0x30000000-0x30000FFF  | PPU Control Registers
0x30001000-0x30001FFF  | PPU Register File Access
0x30002000-0x30003FFF  | L1 Cache Control
0x30004000-0x3000BFFF  | Scratchpad RAM (32KB)
0x30010000-0x3001FFFF  | Reserved for expansion
0x40000000-0xFFFFFFFF  | External memory (via WB)
```

## 6. Pentary ALU Design

### 6.1 ALU Architecture

```
        Rs1 (60 bits)          Rs2 (60 bits)
             │                      │
             └──────────┬───────────┘
                        │
                   ┌────▼────┐
                   │  Digit  │
                   │Separator│  (Split into 20×3-bit digits)
                   └────┬────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
    │ Pentary │    │ Pentary │   │ Pentary │
    │  Adder  │    │  Mult   │   │ Logical │
    └────┬────┘    └────┬────┘   └────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                   ┌────▼────┐
                   │  Result │
                   │  Mux    │
                   └────┬────┘
                        │
                   Result (60 bits)
```

### 6.2 Pentary Addition Algorithm

**Digit-by-digit addition with carry:**

```
For each digit position i (0 to 19):
    sum = digit_a[i] + digit_b[i] + carry_in
    if sum >= 5:
        result[i] = sum - 5
        carry_out = 1
    else:
        result[i] = sum
        carry_out = 0
    carry_in = carry_out
```

**Hardware Implementation:**
- 20 parallel 3-bit adders
- Carry chain propagation
- 5-state detection logic

### 6.3 Pentary Multiplication

**5×5 Multiplication Lookup Table:**

```
×  | 0 | 1 | 2 | 3 | 4 |
---|---|---|---|---|---|
0  | 0 | 0 | 0 | 0 | 0 |
1  | 0 | 1 | 2 | 3 | 4 |
2  | 0 | 2 | 4 |11 |13 |
3  | 0 | 3 |11 |14 |22 |
4  | 0 | 4 |13 |22 |31 |

Note: Results shown in pentary (e.g., 11₅ = 6₁₀)
```

**Multi-digit multiplication:**
- Booth's algorithm adapted for pentary
- Partial product accumulation
- 20-cycle latency for 20×20 digit multiplication

## 7. Wishbone Bus Interface

### 7.1 Interface Signals

```verilog
// Wishbone slave interface
input         wb_clk_i;      // Clock
input         wb_rst_i;      // Reset
input  [31:0] wb_adr_i;      // Address
input  [31:0] wb_dat_i;      // Data input
output [31:0] wb_dat_o;      // Data output
input         wb_we_i;       // Write enable
input  [3:0]  wb_sel_i;      // Byte select
input         wb_stb_i;      // Strobe
input         wb_cyc_i;      // Cycle
output        wb_ack_o;      // Acknowledge
output        wb_err_o;      // Error
```

### 7.2 Register Access Protocol

**Write Operation:**
1. Management core asserts `wb_cyc_i` and `wb_stb_i`
2. Management core provides address on `wb_adr_i`
3. Management core provides data on `wb_dat_i`
4. Management core asserts `wb_we_i`
5. PPU latches data and asserts `wb_ack_o`
6. Management core deasserts signals

**Read Operation:**
1. Management core asserts `wb_cyc_i` and `wb_stb_i`
2. Management core provides address on `wb_adr_i`
3. Management core deasserts `wb_we_i`
4. PPU provides data on `wb_dat_o` and asserts `wb_ack_o`
5. Management core latches data and deasserts signals

## 8. GPIO Interface

### 8.1 Pin Multiplexing Scheme

**Time-multiplexed data bus (20 pins):**
- Cycle 0: Data[19:0] (bits 2:0 of each digit)
- Cycle 1: Data[39:20] (bits 5:3 of each digit)
- Cycle 2: Data[59:40] (bits 8:6 of each digit)

**Control signals (6 pins):**
- CLK: Clock input
- RST: Reset input
- RD: Read strobe
- WR: Write strobe
- EN: Enable
- IRQ: Interrupt request output

**Address bus (8 pins):**
- ADDR[7:0]: Multiplexed address

**Debug/Status (4 pins):**
- TDI: Test data in (JTAG)
- TDO: Test data out (JTAG)
- RDY: Ready output
- ERR: Error output

### 8.2 External Interface Protocol

**Write Transaction:**
```
CLK:  ___╱‾╲_╱‾╲_╱‾╲_╱‾╲___
WR:   _____╱‾‾‾‾‾‾‾‾‾╲_____
ADDR: ═════X_VALID_X═════
DATA: ═════X_D0-19_X_D20-39_X_D40-59_X═════
```

**Read Transaction:**
```
CLK:  ___╱‾╲_╱‾╲_╱‾╲_╱‾╲___
RD:   _____╱‾‾‾‾‾‾‾‾‾╲_____
ADDR: ═════X_VALID_X═════
DATA: ═════X_D0-19_X_D20-39_X_D40-59_X═════
RDY:  _________╱‾‾‾‾‾╲_____
```

## 9. Debug and Test Infrastructure

### 9.1 JTAG Interface

**Standard JTAG signals:**
- TDI: Test Data In
- TDO: Test Data Out
- TCK: Test Clock
- TMS: Test Mode Select
- TRST: Test Reset (optional)

**Debug Features:**
- Boundary scan for I/O testing
- Register file read/write access
- Single-step execution
- Breakpoint support
- Memory dump capability

### 9.2 Performance Counters

**Hardware counters:**
- Cycle counter (64-bit)
- Instruction counter (64-bit)
- Cache miss counter (32-bit)
- Branch mispredict counter (32-bit)
- Stall cycle counter (32-bit)

## 10. Power Management

### 10.1 Clock Gating

**Gated clock domains:**
- ALU (gated when idle)
- Multiplier (gated when not in use)
- Caches (gated during scratchpad access)
- Debug interface (gated when not debugging)

### 10.2 Power States

**Operating Modes:**
1. **Active:** Full operation, all clocks running
2. **Idle:** Core halted, caches active
3. **Sleep:** Core and caches powered down, registers retained
4. **Deep Sleep:** Everything powered down except wake logic

**Power Consumption by Mode:**
- Active: 49 mW @ 50 MHz
- Idle: 15 mW
- Sleep: 2 mW
- Deep Sleep: 0.1 mW

## 11. Timing Specifications

### 11.1 Critical Paths

**Longest paths:**
1. **ALU Path:** Register → Digit Separator → Adder → Carry Chain → Result Mux → Register
   - Delay: ~1.25 ns @ 1.8V
   - Frequency: 800 MHz (theoretical max)

2. **Multiplier Path:** Register → Booth Encoder → Partial Products → Accumulator → Register
   - Delay: ~2.5 ns @ 1.8V
   - Frequency: 400 MHz (theoretical max)

3. **Cache Path:** Address → Tag Compare → Data Select → Output Register
   - Delay: ~1.0 ns @ 1.8V
   - Frequency: 1 GHz (theoretical max)

**Design Target:** 50 MHz (20 ns period) with 16× timing margin

### 11.2 Setup and Hold Times

**Register file:**
- Setup time: 0.2 ns
- Hold time: 0.1 ns

**Cache SRAM:**
- Setup time: 0.3 ns
- Hold time: 0.15 ns

**I/O pads:**
- Setup time: 1.0 ns
- Hold time: 0.5 ns

## 12. Area Breakdown (Detailed)

### 12.1 Gate Count by Module

| Module | Gates | Area (µm²) | % of Total |
|--------|-------|-----------|-----------|
| **Datapath** | | | |
| Register File (25×60b) | 15,000 | 93,750 | 7.2% |
| Pentary Adder | 12,000 | 75,000 | 5.8% |
| Pentary Multiplier | 28,000 | 175,000 | 13.5% |
| Pentary Divider | 15,000 | 93,750 | 7.2% |
| Logical Unit | 5,000 | 31,250 | 2.4% |
| Shifter/Rotator | 8,000 | 50,000 | 3.9% |
| **Control** | | | |
| Instruction Decoder | 4,000 | 25,000 | 1.9% |
| Pipeline Control | 3,000 | 18,750 | 1.4% |
| Hazard Detection | 2,000 | 12,500 | 1.0% |
| **Memory** | | | |
| I-Cache (4KB) | 40,000 | 250,000 | 19.2% |
| D-Cache (4KB) | 40,000 | 250,000 | 19.2% |
| Cache Controllers | 15,000 | 93,750 | 7.2% |
| **Interface** | | | |
| Wishbone Interface | 5,000 | 31,250 | 2.4% |
| GPIO Controller | 3,000 | 18,750 | 1.4% |
| Interrupt Controller | 2,000 | 12,500 | 1.0% |
| Debug Interface | 5,000 | 31,250 | 2.4% |
| **Support** | | | |
| Clock/Reset Logic | 2,000 | 12,500 | 1.0% |
| **TOTAL** | **207,000** | **1,295,000** | **100%** |

**Total Area:** 1.295 mm² (12.95% of 10 mm² budget)

## 13. Verification Strategy

### 13.1 Testbench Hierarchy

```
tb_top
├── tb_ppu_core (full processor)
│   ├── tb_pipeline (pipeline stages)
│   ├── tb_alu (arithmetic/logic)
│   ├── tb_regfile (register file)
│   └── tb_cache (cache subsystem)
├── tb_wishbone (bus interface)
└── tb_gpio (I/O interface)
```

### 13.2 Test Coverage Goals

**Functional Coverage:**
- All instructions: 100%
- All register combinations: 95%
- All addressing modes: 100%
- Cache hit/miss scenarios: 90%
- Pipeline hazards: 100%

**Code Coverage:**
- Line coverage: >95%
- Branch coverage: >90%
- FSM coverage: 100%

## 14. Software Toolchain

### 14.1 Required Tools

**Assembler:**
- Converts pentary assembly to machine code
- Supports all 25 instructions
- Generates binary or hex output

**Compiler:**
- C-to-pentary compiler (future work)
- Based on LLVM backend
- Optimizes for pentary arithmetic

**Simulator:**
- Cycle-accurate pentary ISA simulator
- Debugging support
- Performance profiling

**Debugger:**
- GDB-compatible interface
- JTAG connection
- Register/memory inspection

### 14.2 Example Assembly Program

```asm
; Pentary Assembly Example: Fibonacci Sequence
; Computes first 10 Fibonacci numbers

.section .text
.global _start

_start:
    ; Initialize
    ADDI R1, R0, 0      ; R1 = 0 (fib[0])
    ADDI R2, R0, 1      ; R2 = 1 (fib[1])
    ADDI R3, R0, 10     ; R3 = 10 (counter)
    ADDI R4, R0, 0      ; R4 = 0 (index)

loop:
    ; Store current Fibonacci number
    ST R1, R4, 0x30004000  ; Store to scratchpad

    ; Compute next Fibonacci
    ADD R5, R1, R2      ; R5 = fib[n] + fib[n+1]
    MOV R1, R2          ; R1 = fib[n+1]
    MOV R2, R5          ; R2 = fib[n+2]

    ; Increment index
    ADDI R4, R4, 1

    ; Check if done
    CMP R4, R3
    BNE loop

    ; Exit
    JMP end

end:
    NOP
```

---

**Document Version:** 1.0  
**Date:** 2025-01-06  
**Status:** Architecture Specification Complete