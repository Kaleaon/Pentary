# Pentary Extended Precision Architecture Specification

## 1. Overview

This document defines extended precision modes for the Pentary architecture, providing equivalents to binary 32-bit, 64-bit, and 128-bit operations.

### 1.1 Precision Mode Summary

| Mode | Pent Width | Physical Bits | Info Bits | Decimal Range | Binary Equivalent |
|------|------------|---------------|-----------|---------------|-------------------|
| P14 | 14 pents | 42 bits | ~32.5 bits | ±1.5 billion | 32-bit |
| P16 | 16 pents | 48 bits | ~37.2 bits | ±76 billion | ~37-bit |
| P28 | 28 pents | 84 bits | ~65 bits | ±2.9×10^19 | 64-bit |
| P32 | 32 pents | 96 bits | ~74.3 bits | ±3.6×10^22 | ~74-bit |
| P55 | 55 pents | 165 bits | ~127.8 bits | ±2.8×10^38 | 128-bit |

### 1.2 Mathematical Foundation

Information content per pent: log₂(5) ≈ 2.322 bits

**Range calculation**:
- N pents can represent values from -(5^N - 1)/2 to +(5^N - 1)/2
- Maximum value = 2×(5^(N-1) + 5^(N-2) + ... + 5^0) = 2×(5^N - 1)/4

## 2. P14 Mode (32-bit Equivalent)

### 2.1 Specifications

```
┌─────────────────────────────────────────────────────────────┐
│                    P14 Word Format                          │
├─────────────────────────────────────────────────────────────┤
│ p13│ p12│ p11│ p10│ p9 │ p8 │ p7 │ p6 │ p5 │ p4 │ p3 │ p2 │ p1 │ p0 │
│ MSP│    │    │    │    │    │    │    │    │    │    │    │    │ LSP│
└─────────────────────────────────────────────────────────────┘
```

- **Width**: 14 pents = 42 physical bits
- **Information**: ~32.5 bits (log₂(5^14) ≈ 32.52)
- **Range**: ±1,525,878,906 (±1.5 billion)
- **Use cases**: General integer arithmetic, loop counters, array indices

### 2.2 P14 Registers

| Register | Width | Purpose |
|----------|-------|---------|
| R0-R31 | 14 pents | General purpose (R0 = 0) |
| PC14 | 14 pents | Program counter |
| SR14 | 4 pents | Status register |

### 2.3 P14 Instructions

```
; P14 Arithmetic
ADD14   Rd, Rs1, Rs2    ; 14-pent addition
SUB14   Rd, Rs1, Rs2    ; 14-pent subtraction
MUL14   Rd, Rs1, Rs2    ; 14-pent multiplication (result in ACC28)
DIV14   Rd, Rs1, Rs2    ; 14-pent division

; P14 to P16 conversion
EXTEND14  Rd16, Rs14    ; Zero-extend P14 to P16
TRUNC16   Rd14, Rs16    ; Truncate P16 to P14
```

## 3. P28 Mode (64-bit Equivalent)

### 3.1 Specifications

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                P28 Word Format                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ p27│ p26│...│ p16│ p15│ p14│ p13│...│ p2 │ p1 │ p0 │
│ MSP│    │   │    │    │    │    │   │    │    │ LSP│
│    │    High Word (14 pents)    │   Low Word (14 pents)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- **Width**: 28 pents = 84 physical bits
- **Information**: ~65 bits (log₂(5^28) ≈ 65.04)
- **Range**: ±2.91×10^19 (comparable to int64)
- **Use cases**: Large integers, file sizes, timestamps, cryptographic operations

### 3.2 P28 Register Pairs

P28 values use register pairs:

| Pair | High Register | Low Register | Combined Width |
|------|---------------|--------------|----------------|
| D0 | P1 | P0 | 28 pents (P0 always 0, so special) |
| D1 | P3 | P2 | 28 pents |
| D2 | P5 | P4 | 28 pents |
| ... | ... | ... | ... |
| D15 | P31 | P30 | 28 pents |

### 3.3 P28 Instructions

```
; P28 Arithmetic (uses register pairs)
ADD28   Dd, Ds1, Ds2    ; 28-pent addition
SUB28   Dd, Ds1, Ds2    ; 28-pent subtraction
MUL28   Dd, Ds1, Ds2    ; 28-pent multiplication (result in ACC56)
DIV28   Dd, Ds1, Ds2    ; 28-pent division
NEG28   Dd, Ds          ; 28-pent negation

; P28 Comparison
CMP28   Ds1, Ds2        ; Compare and set flags
TST28   Ds              ; Test against zero

; P28 Load/Store
LOAD28  Dd, [base+offset]  ; Load 28 pents from memory
STORE28 [base+offset], Ds  ; Store 28 pents to memory

; P28 to P16 conversion
SPLIT28 Rd_hi, Rd_lo, Ds28 ; Split P28 into two P14 values
JOIN28  Dd28, Rs_hi, Rs_lo ; Join two P14 values into P28
```

### 3.4 P28 Accumulator

The 56-pent accumulator supports P28 multiply-accumulate operations:

```
; 56-pent accumulator operations
MACC28  Ds1, Ds2        ; ACC56 += Ds1 × Ds2
MSUB28  Ds1, Ds2        ; ACC56 -= Ds1 × Ds2
GETACC28_HI  Dd         ; Get high 28 pents of ACC56
GETACC28_LO  Dd         ; Get low 28 pents of ACC56
CLRACC                  ; Clear ACC56
```

## 4. P55 Mode (128-bit Equivalent)

### 4.1 Specifications

- **Width**: 55 pents = 165 physical bits
- **Information**: ~127.8 bits (log₂(5^55) ≈ 127.71)
- **Range**: ±2.78×10^38
- **Use cases**: Cryptography, high-precision scientific computing

### 4.2 P55 Register Quads

P55 values use register quads (4 × P14):

| Quad | Registers | Combined Width |
|------|-----------|----------------|
| Q0 | P3:P2:P1:P0 | 55 pents (with 1 pent padding) |
| Q1 | P7:P6:P5:P4 | 55 pents |
| ... | ... | ... |

### 4.3 P55 Instructions

```
; P55 Arithmetic
ADD55   Qd, Qs1, Qs2    ; 55-pent addition
SUB55   Qd, Qs1, Qs2    ; 55-pent subtraction
MUL55   Qd, Qs1, Qs2    ; 55-pent multiplication (partial)
```

## 5. Mixed Precision Operations

### 5.1 Automatic Precision Extension

Operations can mix precisions with automatic extension:

```
; Mixed precision addition
ADD_MIX Rd28, Rs16, Rt14    ; Rs16 extended to P28, Rt14 extended to P28
                             ; Result in 28-pent Rd28
```

### 5.2 Saturation Modes

```
; Saturating arithmetic (clamps to max/min instead of overflow)
ADDS14  Rd, Rs1, Rs2    ; Saturating add (P14)
ADDS28  Dd, Ds1, Ds2    ; Saturating add (P28)

; Flags affected:
; - S (Saturation) flag set if result was clamped
```

### 5.3 Precision Conversion Instructions

```
; Sign extension
SEXT14_28  Dd, Rs       ; Sign-extend P14 to P28
SEXT14_55  Qd, Rs       ; Sign-extend P14 to P55
SEXT28_55  Qd, Ds       ; Sign-extend P28 to P55

; Truncation
TRUNC28_14 Rd, Ds       ; Truncate P28 to P14 (with overflow check)
TRUNC55_28 Dd, Qs       ; Truncate P55 to P28 (with overflow check)
TRUNC55_14 Rd, Qs       ; Truncate P55 to P14 (with overflow check)
```

## 6. Carry and Overflow Handling

### 6.1 Multi-Precision Addition

For arbitrary precision, carry propagation is explicit:

```
; Add two 56-pent numbers using pairs of P28 values
; Result = (A_hi:A_lo) + (B_hi:B_lo)

ADD28C  D0, D2, D4      ; D0 = D2 + D4, Carry flag set if overflow
ADDC28  D1, D3, D5      ; D1 = D3 + D5 + Carry
; Result in D1:D0
```

### 6.2 Overflow Detection

```
; Overflow check instructions
OVFCHK14  Rd, Rs1, Rs2  ; Check if Rs1 + Rs2 would overflow P14
OVFCHK28  Dd, Ds1, Ds2  ; Check if Ds1 + Ds2 would overflow P28
; Sets V (Overflow) flag without performing the operation
```

## 7. Extended Precision in Simulator

### 7.1 Register State for Extended Modes

```python
class ExtendedPentaryProcessor:
    def __init__(self):
        # P14 mode registers (32 × 14 pents)
        self.r14 = ["0" * 14 for _ in range(32)]
        
        # P16 mode registers (existing)
        self.p16 = ["0" * 16 for _ in range(32)]
        
        # P28 mode register pairs (16 pairs)
        self.d28 = [("0" * 14, "0" * 14) for _ in range(16)]
        
        # P55 mode register quads (8 quads)
        self.q55 = [("0" * 14, "0" * 14, "0" * 14, "0" * 14) for _ in range(8)]
        
        # Extended accumulators
        self.acc28 = "0" * 28   # For P14 multiply
        self.acc56 = "0" * 56   # For P28 multiply
        self.acc110 = "0" * 110 # For P55 multiply
```

## 8. Memory Alignment Requirements

### 8.1 Alignment Rules

| Precision | Natural Alignment | Forced Alignment |
|-----------|------------------|------------------|
| P14 | 14 pents | 16 pents (for cache efficiency) |
| P16 | 16 pents | 16 pents |
| P28 | 28 pents | 32 pents (for cache efficiency) |
| P55 | 55 pents | 64 pents (for cache efficiency) |

### 8.2 Misaligned Access

```
; Misaligned load/store (slower but supported)
LOADU28  Dd, [base+offset]   ; Unaligned 28-pent load
STOREU28 [base+offset], Ds   ; Unaligned 28-pent store
```

## 9. Performance Considerations

### 9.1 Latency by Precision

| Operation | P14 Cycles | P16 Cycles | P28 Cycles | P55 Cycles |
|-----------|------------|------------|------------|------------|
| ADD | 1 | 1 | 2 | 4 |
| SUB | 1 | 1 | 2 | 4 |
| MUL | 3 | 4 | 8 | 20 |
| DIV | 14 | 16 | 32 | 80 |
| LOAD | 1 | 1 | 2 | 4 |
| STORE | 1 | 1 | 2 | 4 |

### 9.2 Throughput Recommendations

- Use P14 for loop counters and array indices
- Use P16 for neural network weights
- Use P28 for addresses and large counters
- Use P55 only when cryptographic precision is required

---

**Document Version**: 1.0
**Status**: Specification Complete
