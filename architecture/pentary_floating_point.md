# Pentary Floating-Point Specification

## 1. Overview

This document defines floating-point formats for the Pentary architecture, enabling scientific computing, graphics, and machine learning workloads.

### 1.1 Design Philosophy

Pentary floating-point differs from IEEE 754 binary floating-point:
- Uses base-5 (pentary) representation throughout
- Sign, exponent, and mantissa all in balanced pentary
- Native support for 5-level quantization
- Simplified rounding (natural pentary behavior)

## 2. Floating-Point Formats

### 2.1 PentFloat16 (Half Precision)

```
┌──────┬────────────┬─────────────────────────────┐
│ Sign │  Exponent  │         Mantissa            │
│1 pent│  4 pents   │         11 pents            │
│ ±    │ (-312,+312)│    Fractional precision     │
└──────┴────────────┴─────────────────────────────┘
Total: 16 pents = 48 physical bits
```

**Specifications**:
- Sign: 1 pent (values: -2, -1, 0, +1, +2 → maps to {-, -, 0, +, +})
- Exponent: 4 pents = range ±312 (biased by 312)
- Mantissa: 11 pents = ~25.5 bits precision
- Equivalent to: ~FP32 precision with extended exponent range

### 2.2 PentFloat28 (Single Precision)

```
┌──────┬────────────┬─────────────────────────────────────────────┐
│ Sign │  Exponent  │                 Mantissa                    │
│1 pent│  6 pents   │               21 pents                      │
│ ±    │(±7812)     │       Fractional precision                  │
└──────┴────────────┴─────────────────────────────────────────────┘
Total: 28 pents = 84 physical bits
```

**Specifications**:
- Sign: 1 pent
- Exponent: 6 pents = range ±7,812 (biased by 7,812)
- Mantissa: 21 pents = ~48.8 bits precision
- Equivalent to: ~FP64 precision

### 2.3 PentFloat55 (Double Precision)

```
┌──────┬────────────┬─────────────────────────────────────────────────────────┐
│ Sign │  Exponent  │                      Mantissa                           │
│1 pent│  8 pents   │                    46 pents                             │
│ ±    │(±195312)   │              Fractional precision                       │
└──────┴────────────┴─────────────────────────────────────────────────────────┘
Total: 55 pents = 165 physical bits
```

**Specifications**:
- Sign: 1 pent
- Exponent: 8 pents = range ±195,312 (biased by 195,312)
- Mantissa: 46 pents = ~106.8 bits precision
- Equivalent to: ~FP128 precision

## 3. Special Values

### 3.1 Zero

```
Sign: 0, Exponent: all 0s, Mantissa: all 0s
Positive zero: +0.0
Negative zero: -0.0 (sign = ⊖ or -)
```

### 3.2 Infinity

```
Sign: ±, Exponent: all ⊕s (maximum), Mantissa: all 0s
+∞: Sign=+, Exp=⊕⊕⊕⊕, Mant=0...0
-∞: Sign=-, Exp=⊕⊕⊕⊕, Mant=0...0
```

### 3.3 NaN (Not a Number)

```
Sign: any, Exponent: all ⊕s, Mantissa: non-zero
Quiet NaN (qNaN): Mantissa MSB = ⊕
Signaling NaN (sNaN): Mantissa MSB ≠ ⊕, rest non-zero
```

### 3.4 Subnormal Numbers

```
Sign: ±, Exponent: all 0s, Mantissa: non-zero
Gradual underflow to zero
```

## 4. Value Representation

### 4.1 Normal Numbers

For a normalized PentFloat:

```
Value = Sign × 5^(Exponent - Bias) × (1 + Mantissa × 5^(-MantissaLen))
```

Where:
- `Sign` = +1 if sign pent > 0, -1 if sign pent < 0
- `Exponent` = integer value of exponent pents
- `Bias` = (5^ExpLen - 1) / 2
- `Mantissa` = integer value of mantissa pents (implicit leading 1)

### 4.2 Example: PentFloat16

Value 3.14159... in PentFloat16:

1. Sign: + (1 pent = '+')
2. Find exponent: 3.14159 ≈ 5^0.71, so exp ≈ 1
3. Exponent with bias: 1 + 312 = 313 → pentary representation
4. Mantissa: fractional part in base 5

```
3.14159 = 1 × 5^0 × 3.14159
        = 1 × 5^0 × (1 + 0.11022... in pent)
```

## 5. Arithmetic Operations

### 5.1 Addition/Subtraction

```
FADD16  Rd, Rs1, Rs2    ; PentFloat16 addition
FSUB16  Rd, Rs1, Rs2    ; PentFloat16 subtraction
FADD28  Dd, Ds1, Ds2    ; PentFloat28 addition
FSUB28  Dd, Ds1, Ds2    ; PentFloat28 subtraction
```

**Algorithm**:
1. Align exponents (shift smaller mantissa right)
2. Add/subtract mantissas
3. Normalize result
4. Round and check for overflow/underflow

### 5.2 Multiplication

```
FMUL16  Rd, Rs1, Rs2    ; PentFloat16 multiplication
FMUL28  Dd, Ds1, Ds2    ; PentFloat28 multiplication
```

**Algorithm**:
1. XOR signs
2. Add exponents (subtract bias)
3. Multiply mantissas
4. Normalize and round

### 5.3 Division

```
FDIV16  Rd, Rs1, Rs2    ; PentFloat16 division
FDIV28  Dd, Ds1, Ds2    ; PentFloat28 division
```

**Algorithm**:
1. XOR signs
2. Subtract exponents (add bias)
3. Divide mantissas
4. Normalize and round

### 5.4 Fused Multiply-Add

```
FMADD16  Rd, Rs1, Rs2, Rs3  ; Rd = Rs1 × Rs2 + Rs3 (single rounding)
FMADD28  Dd, Ds1, Ds2, Ds3  ; Dd = Ds1 × Ds2 + Ds3 (single rounding)
```

FMA provides higher precision by avoiding intermediate rounding.

## 6. Comparison Operations

```
FCMP16  Rs1, Rs2        ; Compare PentFloat16, set flags
FCMP28  Ds1, Ds2        ; Compare PentFloat28, set flags

; Comparison results (in status register):
; Z=1: Equal
; N=1: Rs1 < Rs2
; P=1: Rs1 > Rs2
; U=1: Unordered (at least one NaN)
```

### 6.1 Ordered Comparisons

```
FEQ16   Rd, Rs1, Rs2    ; Rd = (Rs1 == Rs2) ? 1 : 0
FLT16   Rd, Rs1, Rs2    ; Rd = (Rs1 < Rs2) ? 1 : 0
FLE16   Rd, Rs1, Rs2    ; Rd = (Rs1 <= Rs2) ? 1 : 0
```

### 6.2 Unordered Comparisons

```
FUNEQ16 Rd, Rs1, Rs2    ; True if equal or unordered
FUNLT16 Rd, Rs1, Rs2    ; True if less than or unordered
```

## 7. Conversion Operations

### 7.1 Integer to Float

```
ITOF16  Rd, Rs          ; Convert P14 integer to PentFloat16
ITOF28  Dd, Ds          ; Convert P28 integer to PentFloat28
```

### 7.2 Float to Integer

```
FTOI16  Rd, Rs          ; Convert PentFloat16 to P14 integer
FTOI28  Dd, Ds          ; Convert PentFloat28 to P28 integer

; Rounding modes:
FTOIZ16 Rd, Rs          ; Truncate toward zero
FTOIN16 Rd, Rs          ; Round to nearest, ties to even
FTOIP16 Rd, Rs          ; Round toward +∞
FTOIM16 Rd, Rs          ; Round toward -∞
```

### 7.3 Float Precision Conversion

```
F16TOF28 Dd, Rs         ; Extend PentFloat16 to PentFloat28
F28TOF16 Rd, Ds         ; Convert PentFloat28 to PentFloat16
F28TOF55 Qd, Ds         ; Extend PentFloat28 to PentFloat55
F55TOF28 Dd, Qs         ; Convert PentFloat55 to PentFloat28
```

## 8. Special Function Instructions

### 8.1 Square Root

```
FSQRT16 Rd, Rs          ; Square root of PentFloat16
FSQRT28 Dd, Ds          ; Square root of PentFloat28
```

### 8.2 Reciprocal

```
FRCP16  Rd, Rs          ; 1/Rs (fast reciprocal approximation)
FRSQRT16 Rd, Rs         ; 1/√Rs (fast inverse square root)
```

### 8.3 Transcendental Functions (Software/LUT)

These are typically implemented via lookup tables or polynomial approximations:

```
; Register-based LUT addresses
FEXP16_APPROX  Rd, Rs   ; e^Rs approximation
FLOG16_APPROX  Rd, Rs   ; ln(Rs) approximation
FSIN16_APPROX  Rd, Rs   ; sin(Rs) approximation
FCOS16_APPROX  Rd, Rs   ; cos(Rs) approximation
```

## 9. Rounding Modes

### 9.1 Supported Modes

| Mode | Name | Description |
|------|------|-------------|
| RNE | Round to Nearest, Even | Default; ties round to even |
| RTZ | Round Toward Zero | Truncation |
| RDN | Round Down | Toward -∞ |
| RUP | Round Up | Toward +∞ |
| RMM | Round to Max Magnitude | Away from zero |

### 9.2 Rounding Mode Control

```
; Set rounding mode in control register
SETFRND mode            ; mode = RNE, RTZ, RDN, RUP, or RMM
GETFRND Rd              ; Get current rounding mode
```

## 10. Exception Handling

### 10.1 Floating-Point Exceptions

| Exception | Flag | Cause |
|-----------|------|-------|
| Invalid | I | Invalid operation (0/0, ∞-∞, etc.) |
| Division by Zero | Z | x/0 where x ≠ 0 |
| Overflow | O | Result too large |
| Underflow | U | Result too small |
| Inexact | X | Rounding occurred |

### 10.2 Exception Control

```
; Enable/disable exception trapping
FEXCEN  mask            ; Enable exceptions in mask
FEXCDIS mask            ; Disable exceptions in mask

; Read/clear exception flags
FEXCRD  Rd              ; Read exception flags
FEXCCLR mask            ; Clear specified exception flags
```

## 11. Floating-Point Register File

### 11.1 Dedicated FP Registers (Optional Extension)

```
F0-F31:  32 × PentFloat16 registers
FD0-FD15: 16 × PentFloat28 register pairs
FQ0-FQ7:  8 × PentFloat55 register quads
```

### 11.2 Shared GP/FP Register Model

Alternatively, floating-point values can use the general-purpose registers:
- P-registers for PentFloat16
- D-register pairs for PentFloat28
- Q-register quads for PentFloat55

## 12. Hardware Implementation

### 12.1 FPU Block Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Pentary Floating-Point Unit                │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  Unpacker     │  │  Aligner      │  │  Adder        │  │
│  │  (Exp, Mant)  │  │  (Shift)      │  │  (Pentary)    │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  Multiplier   │  │  Divider      │  │  Normalizer   │  │
│  │  (Pentary)    │  │  (Iterative)  │  │  (LZD+Shift)  │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  Rounder      │  │  Packer       │  │  Exception    │  │
│  │  (5 modes)    │  │  (Format)     │  │  Handler      │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 Latency Estimates

| Operation | PentFloat16 | PentFloat28 | PentFloat55 |
|-----------|-------------|-------------|-------------|
| FADD | 4 cycles | 6 cycles | 10 cycles |
| FMUL | 5 cycles | 8 cycles | 15 cycles |
| FDIV | 16 cycles | 28 cycles | 55 cycles |
| FSQRT | 16 cycles | 28 cycles | 55 cycles |
| FMADD | 6 cycles | 10 cycles | 18 cycles |

## 13. Neural Network Considerations

### 13.1 Mixed-Precision Training

PentFloat supports efficient mixed-precision:
- Forward pass: PentFloat16 or quantized integers
- Backward pass: PentFloat28 for gradients
- Master weights: PentFloat28 or PentFloat55

### 13.2 Quantization Instructions

```
FQUANT5 Rd, Rs          ; Quantize PentFloat16 to 5 levels {-2,-1,0,+1,+2}
FDEQUANT5 Rd, Rs, scale ; Dequantize 5-level to PentFloat16 with scale
```

---

**Document Version**: 1.0
**Status**: Specification Complete
