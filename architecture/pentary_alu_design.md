# Pentary ALU Design Specification

## 1. Overview

This document provides detailed circuit-level designs for the Pentary Arithmetic Logic Unit (ALU), the core computational component of the Pentary processor.

## 2. ALU Block Diagram

```
                    ┌─────────────────────────────────────────┐
                    │         Pentary ALU (16 pents)          │
                    ├─────────────────────────────────────────┤
                    │                                         │
    Operand A ──────┤►                                        │
    (16 pents)      │                                         │
                    │   ┌──────────────────────────────┐      │
    Operand B ──────┤►  │   Operand Preprocessing     │      │
    (16 pents)      │   │   - Sign extension          │      │
                    │   │   - Zero detection          │      │
    Operation ──────┤►  └──────────────────────────────┘      │
    (4 bits)        │              ↓                          │
                    │   ┌──────────────────────────────┐      │
    Carry In ───────┤►  │   Execution Units            │      │
    (1 pent)        │   │   - Adder/Subtractor         │      │
                    │   │   - Logic Unit               │      │
                    │   │   - Shifter                  │      │
                    │   │   - Comparator               │      │
                    │   │   - Quantizer                │      │
                    │   └──────────────────────────────┘      │
                    │              ↓                          │
                    │   ┌──────────────────────────────┐      │
                    │   │   Result Selection           │      │
                    │   │   - Multiplexer              │      │
                    │   │   - Flag generation          │      │
                    │   └──────────────────────────────┘      │
                    │              ↓                          │
    Result ◄────────┤  (16 pents)                             │
                    │                                         │
    Flags ◄─────────┤  Z, N, P, V, C, S                       │
                    │                                         │
                    └─────────────────────────────────────────┘
```

## 3. Pentary Full Adder Design

### 3.1 Single-Pent Full Adder

**Inputs**: A, B (pentary digits), C_in (carry in)
**Outputs**: S (sum), C_out (carry out)

**Truth Table** (125 entries total, showing key cases):

```
A    B    C_in  →  S    C_out
⊖    ⊖    ⊖     →  +    -      (-2 + -2 + -2 = -6 = -1 - 5)
⊖    ⊖    -     →  ⊕    -      (-2 + -2 + -1 = -5 = 0 - 5)
⊖    ⊖    0     →  ⊖    -      (-2 + -2 + 0 = -4 = 1 - 5)
...
⊕    ⊕    ⊕     →  -    +      (2 + 2 + 2 = 6 = 1 + 5)
```

**Implementation Options**:

#### Option 1: Lookup Table (LUT)
- 125-entry ROM
- Fast but area-intensive
- Suitable for FPGA implementation

#### Option 2: Logic Gates
- Decompose into sum-of-products
- More gates but synthesizable
- Better for ASIC

**Optimized Logic** (simplified):
```
Sum = f(A, B, C_in)  // Complex 3-input pentary function
Carry_out = g(A, B, C_in)  // Simpler carry logic

Where:
- Carry_out = +1 if (A + B + C_in) ≥ 3
- Carry_out = -1 if (A + B + C_in) ≤ -3
- Carry_out = 0 otherwise
```

### 3.2 Carry-Lookahead Logic

**4-Pent Block Carry-Lookahead**:

For positions i=0,1,2,3 in a 4-pent block:

**Generate (G)**: Block generates a carry
**Propagate (P)**: Block propagates a carry

```
G_i = (A_i + B_i ≥ 3) ? +1 : (A_i + B_i ≤ -3) ? -1 : 0
P_i = (A_i + B_i = ±2) ? 1 : 0
```

**Block Carry Logic**:
```
C_1 = G_0 + (P_0 ∧ C_0)
C_2 = G_1 + (P_1 ∧ C_1) = G_1 + (P_1 ∧ G_0) + (P_1 ∧ P_0 ∧ C_0)
C_3 = G_2 + (P_2 ∧ C_2) = ...
C_4 = G_3 + (P_3 ∧ C_3) = ...
```

**16-Pent Adder Structure**:
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 4-pent   │  │ 4-pent   │  │ 4-pent   │  │ 4-pent   │
│ Block 0  │  │ Block 1  │  │ Block 2  │  │ Block 3  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   │
          ┌────────▼────────┐
          │  Carry Lookahead│
          │  Generator      │
          └─────────────────┘
```

**Estimated Delay**: 
- Ripple-carry: 16 × t_FA ≈ 16 gate delays
- Carry-lookahead: log₄(16) × t_CLA + t_FA ≈ 4 gate delays

## 4. Subtractor Design

**Subtraction**: A - B = A + (-B)

**Implementation**:
1. Negate B (flip all digits)
2. Add A + (-B) using adder
3. Set carry_in = 0

**Pentary Negation Circuit**:
```
Input:  ⊖  -  0  +  ⊕
Output: ⊕  +  0  -  ⊖
```

Simple lookup or logic:
```
NEG(x) = {
    ⊕ if x = ⊖
    + if x = -
    0 if x = 0
    - if x = +
    ⊖ if x = ⊕
}
```

## 5. Logic Unit

### 5.1 MIN Gate

**Function**: Output = min(A, B)

**Truth Table**:
```
A\B  ⊖  -  0  +  ⊕
⊖    ⊖  ⊖  ⊖  ⊖  ⊖
-    ⊖  -  -  -  -
0    ⊖  -  0  0  0
+    ⊖  -  0  +  +
⊕    ⊖  -  0  +  ⊕
```

**Implementation**: Comparator + Multiplexer
```
if (A < B) then output A
else output B
```

### 5.2 MAX Gate

**Function**: Output = max(A, B)

**Truth Table**:
```
A\B  ⊖  -  0  +  ⊕
⊖    ⊖  -  0  +  ⊕
-    -  -  0  +  ⊕
0    0  0  0  +  ⊕
+    +  +  +  +  ⊕
⊕    ⊕  ⊕  ⊕  ⊕  ⊕
```

**Implementation**: Comparator + Multiplexer
```
if (A > B) then output A
else output B
```

### 5.3 CONSENSUS Gate

**Function**: Returns the "middle" value or consensus

**Truth Table**:
```
A\B  ⊖  -  0  +  ⊕
⊖    ⊖  ⊖  ⊖  0  0
-    ⊖  -  -  0  0
0    ⊖  -  0  +  ⊕
+    0  0  +  +  ⊕
⊕    0  0  ⊕  ⊕  ⊕
```

**Implementation**: 
```
CONSENSUS(A, B) = {
    A if |A| > |B|
    B if |B| > |A|
    0 if A = -B
    A if A = B
}
```

## 6. Shifter Design

### 6.1 Left Shift (Multiply by 5)

**Function**: Shift left by n positions = multiply by 5ⁿ

**Implementation**:
- Append n zeros to the right
- Example: +⊕ << 1 = +⊕0 (7 × 5 = 35)

**Circuit**:
```
┌─────────────────────────────────────┐
│  Barrel Shifter (Left)              │
│                                     │
│  Input: [A₁₅ A₁₄ ... A₁ A₀]        │
│  Shift: n (0-15)                    │
│                                     │
│  Output: [A₁₅₋ₙ ... A₀ 0 ... 0]    │
│                    └─n zeros─┘      │
└─────────────────────────────────────┘
```

### 6.2 Right Shift (Divide by 5)

**Function**: Shift right by n positions = divide by 5ⁿ

**Implementation**:
- Remove n digits from the right (truncate)
- Sign-extend from the left
- Example: +⊕0 >> 1 = +⊕ (35 ÷ 5 = 7)

**Circuit**:
```
┌─────────────────────────────────────┐
│  Barrel Shifter (Right)             │
│                                     │
│  Input: [A₁₅ A₁₄ ... A₁ A₀]        │
│  Shift: n (0-15)                    │
│                                     │
│  Output: [S S ... S A₁₅ ... Aₙ]    │
│           └─n sign ext─┘            │
└─────────────────────────────────────┘
```

### 6.3 Multiply by 2 (Special Case)

**Function**: Multiply by 2 (not a pentary shift!)

**Implementation**:
- For each digit d: result = 2×d with carry handling
- More complex than shift, but still simpler than general multiply

**Algorithm**:
```
carry = 0
for i = 0 to 15:
    product = 2 × A[i] + carry
    if product > 2:
        result[i] = product - 5
        carry = 1
    elif product < -2:
        result[i] = product + 5
        carry = -1
    else:
        result[i] = product
        carry = 0
```

## 7. Comparator Design

### 7.1 Single-Pent Comparator

**Inputs**: A, B (pentary digits)
**Outputs**: LT (less than), EQ (equal), GT (greater than)

**Truth Table**:
```
A\B  ⊖  -  0  +  ⊕
⊖    EQ LT LT LT LT
-    GT EQ LT LT LT
0    GT GT EQ LT LT
+    GT GT GT EQ LT
⊕    GT GT GT GT EQ
```

**Implementation**:
```
LT = (A < B)
EQ = (A = B)
GT = (A > B)
```

### 7.2 Multi-Pent Comparator

**16-Pent Magnitude Comparator**:

**Algorithm**:
```
Compare from most significant to least significant:
for i = 15 down to 0:
    if A[i] > B[i]: return GT
    if A[i] < B[i]: return LT
return EQ
```

**Circuit**: Priority encoder with 16 levels

**Optimization**: Tree structure
```
Level 1: Compare pairs (8 comparisons)
Level 2: Combine results (4 comparisons)
Level 3: Combine results (2 comparisons)
Level 4: Final result (1 comparison)
```

**Delay**: log₂(16) = 4 levels

## 8. Quantizer Design

### 8.1 5-Level Quantizer

**Function**: Map continuous/high-precision value to {⊖, -, 0, +, ⊕}

**Thresholds**:
```
Input Range        Output
x < -1.5          ⊖ (-2)
-1.5 ≤ x < -0.5   - (-1)
-0.5 ≤ x < 0.5    0
0.5 ≤ x < 1.5     + (+1)
x ≥ 1.5           ⊕ (+2)
```

**Implementation**: 4 comparators + priority encoder

```
┌─────────────────────────────────────┐
│  Quantizer                          │
│                                     │
│  Input ──┬──[CMP: x < -1.5]──► ⊖   │
│          ├──[CMP: x < -0.5]──► -   │
│          ├──[CMP: x < 0.5]───► 0   │
│          ├──[CMP: x < 1.5]───► +   │
│          └──[else]───────────► ⊕   │
│                                     │
└─────────────────────────────────────┘
```

### 8.2 Saturation Logic

**Function**: Clamp values to valid pentary range

**For 16-pent signed**:
- Max: ⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕
- Min: ⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖⊖

**Implementation**:
```
if (overflow_positive):
    result = MAX_VALUE
elif (overflow_negative):
    result = MIN_VALUE
else:
    result = computed_value
```

## 9. Flag Generation

### 9.1 Zero Flag (Z)

**Condition**: Result = 0

**Implementation**:
```
Z = (result[15] = 0) ∧ (result[14] = 0) ∧ ... ∧ (result[0] = 0)
```

**Circuit**: 16-input NOR gate (tree structure)

### 9.2 Negative Flag (N)

**Condition**: Result < 0

**Implementation**:
```
N = (most_significant_nonzero_digit < 0)
```

**Circuit**: Priority encoder to find MSB, then check sign

### 9.3 Positive Flag (P)

**Condition**: Result > 0

**Implementation**:
```
P = (most_significant_nonzero_digit > 0)
```

### 9.4 Overflow Flag (V)

**Condition**: Signed overflow occurred

**For Addition**:
```
V = (A_sign = B_sign) ∧ (Result_sign ≠ A_sign)
```

**For Subtraction**:
```
V = (A_sign ≠ B_sign) ∧ (Result_sign ≠ A_sign)
```

### 9.5 Carry Flag (C)

**Condition**: Carry out of MSB

**Implementation**: Direct output from adder's carry_out

### 9.6 Saturation Flag (S)

**Condition**: Result was saturated

**Implementation**: Set when overflow occurs and saturation is enabled

## 10. Complete ALU Operation Table

| Operation | Opcode | Function | Flags Updated |
|-----------|--------|----------|---------------|
| ADD | 0000 | R = A + B | Z,N,P,V,C |
| SUB | 0001 | R = A - B | Z,N,P,V,C |
| NEG | 0010 | R = -A | Z,N,P |
| ABS | 0011 | R = \|A\| | Z,N,P |
| MIN | 0100 | R = min(A,B) | Z,N,P |
| MAX | 0101 | R = max(A,B) | Z,N,P |
| CONS | 0110 | R = consensus(A,B) | Z,N,P |
| SHL | 0111 | R = A << B | Z,N,P,C |
| SHR | 1000 | R = A >> B | Z,N,P,C |
| MUL2 | 1001 | R = A × 2 | Z,N,P,V,C |
| DIV2 | 1010 | R = A ÷ 2 | Z,N,P |
| CMP | 1011 | Compare A,B | Z,N,P |
| QUANT | 1100 | R = quantize(A) | Z,N,P,S |
| CLAMP | 1101 | R = clamp(A,B,C) | Z,N,P,S |
| PASS | 1110 | R = A | Z,N,P |
| ZERO | 1111 | R = 0 | Z |

## 11. Timing Analysis

### 11.1 Critical Paths

**Addition** (carry-lookahead):
- Input → CLA logic → Sum output
- Delay: ~4 gate delays
- Frequency: ~2.5 GHz @ 28nm

**Comparison**:
- Input → Comparator tree → Flags
- Delay: ~4 gate delays
- Frequency: ~2.5 GHz @ 28nm

**Shift**:
- Input → Barrel shifter → Output
- Delay: ~3 gate delays
- Frequency: ~3.3 GHz @ 28nm

### 11.2 Power Analysis

**Dynamic Power** (per operation @ 1.0V, 28nm):
- Addition: ~50 pJ
- Subtraction: ~55 pJ (includes negation)
- Logic (MIN/MAX): ~30 pJ
- Shift: ~20 pJ
- Comparison: ~35 pJ

**Static Power** (leakage):
- ~10 mW per ALU @ 28nm

## 12. Area Estimation

**Component Areas** (normalized to binary full adder = 1.0):

| Component | Relative Area | Count | Total |
|-----------|---------------|-------|-------|
| Pentary Full Adder | 4.0 | 16 | 64.0 |
| Carry Lookahead | 2.0 | 4 | 8.0 |
| Comparator | 3.0 | 1 | 3.0 |
| Shifter | 2.5 | 1 | 2.5 |
| Logic Unit | 2.0 | 1 | 2.0 |
| Quantizer | 1.5 | 1 | 1.5 |
| Control Logic | 1.0 | 1 | 1.0 |
| **Total** | | | **82.0** |

**Comparison**: Binary 16-bit ALU ≈ 30.0 units
**Ratio**: Pentary ALU is ~2.7× larger than binary ALU

**BUT**: Pentary ALU handles 16 pents = 37 bits equivalent
**Effective Ratio**: 82.0 / (37/16 × 30.0) ≈ 1.2× (only 20% larger for equivalent precision!)

## 13. Verification Strategy

### 13.1 Test Vectors

**Comprehensive Test Suite**:
1. Boundary cases (MAX, MIN values)
2. Zero cases
3. Carry propagation
4. Overflow conditions
5. Random test vectors

**Example Test Cases**:
```
# Addition
⊕⊕⊕⊕ + 0000 = ⊕⊕⊕⊕  (no carry)
⊕⊕⊕⊕ + 0001 = +⊖⊖⊖⊖ (carry propagation)
⊕⊕⊕⊕ + ⊕⊕⊕⊕ = overflow (saturation)

# Subtraction
0000 - 0001 = ⊖⊖⊖⊖ (borrow)
+000 - ⊕000 = ⊖000 (sign change)

# Logic
MIN(⊕⊕⊕⊕, ⊖⊖⊖⊖) = ⊖⊖⊖⊖
MAX(⊕⊕⊕⊕, ⊖⊖⊖⊖) = ⊕⊕⊕⊕
```

### 13.2 Formal Verification

**Properties to Verify**:
1. Commutativity: A + B = B + A
2. Associativity: (A + B) + C = A + (B + C)
3. Identity: A + 0 = A
4. Inverse: A + (-A) = 0
5. Overflow detection correctness
6. Flag generation correctness

## 14. Implementation Notes

### 14.1 FPGA Implementation

**Recommended**: Xilinx Virtex UltraScale+ or Intel Stratix 10
- Use LUT-based implementation for full adders
- Leverage DSP blocks for multiply-accumulate
- Use block RAM for lookup tables

### 14.2 ASIC Implementation

**Recommended**: 28nm or better process
- Custom standard cells for pentary gates
- Careful layout for matched delays
- Power gating for zero-state optimization

### 14.3 Simulation

**Tools**:
- Verilog/VHDL for RTL simulation
- Python model for functional verification
- SPICE for analog memristor interface

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Status**: Detailed Design Specification