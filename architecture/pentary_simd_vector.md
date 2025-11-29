# Pentary SIMD and Vector Extension Specification

## 1. Overview

This document defines SIMD (Single Instruction, Multiple Data) and vector extensions for the Pentary architecture, enabling efficient parallel processing of multiple data elements.

### 1.1 Design Goals

1. **Natural Parallelism**: Process multiple pentary values in single instructions
2. **Neural Network Optimization**: Optimized for matrix operations and activations
3. **Scalable Vector Lengths**: Support for various vector widths
4. **Efficient Memory Access**: Gather/scatter and strided access patterns

## 2. Vector Register Architecture

### 2.1 Vector Registers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Vector Register File                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ V0-V31: 32 vector registers, each 256 pents (768 physical bits)                │
│                                                                                  │
│ Each register can be viewed as:                                                 │
│   - 16 × P16 elements (16-pent integers)                                        │
│   - 18 × P14 elements (14-pent integers, with padding)                          │
│   - 9 × P28 elements (28-pent integers)                                         │
│   - 4 × P55 elements (55-pent integers, with padding)                           │
│   - 16 × PentFloat16 elements                                                   │
│   - 9 × PentFloat28 elements                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Vector Length Register (VL)

```
VL: Controls active elements in vector operations
Range: 1 to VLEN (maximum vector length, architecture-dependent)

; Set vector length
SETVL   Rd, Rs          ; Rd = min(Rs, VLEN); VL set
SETVLI  Rd, imm         ; Rd = min(imm, VLEN); VL set
```

### 2.3 Vector Configuration

```
; Vector type configuration
VSETTYPE  type          ; Set element type
                        ; type: P14, P16, P28, P55, F16, F28, F55

; Get current configuration
VGETCFG  Rd             ; Rd = current vector configuration
```

## 3. Vector Integer Instructions

### 3.1 Arithmetic Operations

```
; Element-wise addition
VADD.P16   Vd, Vs1, Vs2     ; Vd[i] = Vs1[i] + Vs2[i] for i in 0..VL-1
VADD.P28   Vd, Vs1, Vs2     ; 28-pent element addition
VADD.P14   Vd, Vs1, Vs2     ; 14-pent element addition

; Element-wise subtraction
VSUB.P16   Vd, Vs1, Vs2     ; Vd[i] = Vs1[i] - Vs2[i]
VSUB.P28   Vd, Vs1, Vs2

; Negation
VNEG.P16   Vd, Vs           ; Vd[i] = -Vs[i]

; Multiplication (widening)
VMUL.P16   Vd, Vs1, Vs2     ; Vd[i] = Vs1[i] × Vs2[i] (result P32, stored in P16 pairs)
VWMUL.P16  Vd2:Vd1, Vs1, Vs2 ; Full width result in register pair

; Division
VDIV.P16   Vd, Vs1, Vs2     ; Vd[i] = Vs1[i] / Vs2[i]
VMOD.P16   Vd, Vs1, Vs2     ; Vd[i] = Vs1[i] % Vs2[i]
```

### 3.2 Scalar-Vector Operations

```
; Vector-scalar addition
VADD.VX.P16  Vd, Vs, Rs     ; Vd[i] = Vs[i] + Rs for all i
VADD.VI.P16  Vd, Vs, imm    ; Vd[i] = Vs[i] + imm for all i

; Vector-scalar multiplication
VMUL.VX.P16  Vd, Vs, Rs     ; Vd[i] = Vs[i] × Rs
VMUL.VI.P16  Vd, Vs, imm    ; Vd[i] = Vs[i] × imm
```

### 3.3 Reduction Operations

```
; Sum reduction
VREDSUM.P16  Rd, Vs         ; Rd = sum of all Vs[i]

; Max/Min reduction
VREDMAX.P16  Rd, Vs         ; Rd = max of all Vs[i]
VREDMIN.P16  Rd, Vs         ; Rd = min of all Vs[i]

; Dot product
VDOT.P16     Rd, Vs1, Vs2   ; Rd = Σ(Vs1[i] × Vs2[i])
```

### 3.4 Comparison Operations

```
; Element-wise comparison (sets mask register)
VMSEQ.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] == Vs2[i])
VMSNE.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] != Vs2[i])
VMSLT.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] < Vs2[i])
VMSLE.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] <= Vs2[i])
VMSGT.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] > Vs2[i])
VMSGE.P16  Vm, Vs1, Vs2     ; Vm[i] = (Vs1[i] >= Vs2[i])
```

## 4. Vector Floating-Point Instructions

### 4.1 FP Arithmetic

```
; Addition
VFADD.F16  Vd, Vs1, Vs2     ; PentFloat16 element-wise add
VFADD.F28  Vd, Vs1, Vs2     ; PentFloat28 element-wise add

; Subtraction
VFSUB.F16  Vd, Vs1, Vs2
VFSUB.F28  Vd, Vs1, Vs2

; Multiplication
VFMUL.F16  Vd, Vs1, Vs2
VFMUL.F28  Vd, Vs1, Vs2

; Division
VFDIV.F16  Vd, Vs1, Vs2
VFDIV.F28  Vd, Vs1, Vs2

; Fused Multiply-Add
VFMADD.F16 Vd, Vs1, Vs2, Vs3  ; Vd[i] = Vs1[i] × Vs2[i] + Vs3[i]
VFMSUB.F16 Vd, Vs1, Vs2, Vs3  ; Vd[i] = Vs1[i] × Vs2[i] - Vs3[i]
VFNMADD.F16 Vd, Vs1, Vs2, Vs3 ; Vd[i] = -(Vs1[i] × Vs2[i]) + Vs3[i]
```

### 4.2 FP Special Operations

```
; Square root
VFSQRT.F16  Vd, Vs          ; Vd[i] = √Vs[i]

; Reciprocal (fast approximation)
VFRCP.F16   Vd, Vs          ; Vd[i] ≈ 1/Vs[i]

; Reciprocal square root
VFRSQRT.F16 Vd, Vs          ; Vd[i] ≈ 1/√Vs[i]

; Min/Max
VFMIN.F16   Vd, Vs1, Vs2    ; Vd[i] = min(Vs1[i], Vs2[i])
VFMAX.F16   Vd, Vs1, Vs2    ; Vd[i] = max(Vs1[i], Vs2[i])

; Absolute value
VFABS.F16   Vd, Vs          ; Vd[i] = |Vs[i]|

; Negation
VFNEG.F16   Vd, Vs          ; Vd[i] = -Vs[i]
```

## 5. Neural Network Vector Instructions

### 5.1 Activation Functions

```
; ReLU
VRELU.P16   Vd, Vs          ; Vd[i] = max(0, Vs[i])
VRELU.F16   Vd, Vs          ; Floating-point ReLU

; Leaky ReLU
VLRELU.P16  Vd, Vs, alpha   ; Vd[i] = Vs[i] if Vs[i] > 0 else alpha × Vs[i]

; Sigmoid approximation (piecewise linear)
VSIGMOID.F16 Vd, Vs         ; Vd[i] ≈ sigmoid(Vs[i])

; Tanh approximation
VTANH.F16   Vd, Vs          ; Vd[i] ≈ tanh(Vs[i])

; Softmax (multi-instruction sequence)
VEXP.F16    Vd, Vs          ; Vd[i] = exp(Vs[i])
VREDSUM.F16 Rd, Vd          ; Rd = Σ Vd[i]
VFDIV.VX.F16 Vd, Vd, Rd     ; Normalize
```

### 5.2 Quantization

```
; Quantize to 5 levels
VQUANT5.P16  Vd, Vs, scale  ; Vd[i] = round(Vs[i]/scale) clamped to [-2,+2]

; Dequantize from 5 levels
VDEQUANT5.F16 Vd, Vs, scale ; Vd[i] = Vs[i] × scale

; Quantize to 25 levels (2 pents)
VQUANT25.P16 Vd, Vs, scale  ; Vd[i] quantized to [-12,+12]
```

### 5.3 Pooling Operations

```
; Max pooling (2×2)
VMAXPOOL2.P16 Vd, Vs        ; Max pooling with stride 2

; Average pooling (2×2)
VAVGPOOL2.P16 Vd, Vs        ; Average pooling with stride 2

; Global average pooling
VGAVGPOOL.P16 Rd, Vs        ; Rd = mean of all Vs elements
```

## 6. Matrix Operations

### 6.1 Matrix-Vector Multiplication

```
; Matrix-vector multiply (using register groups)
VMATVEC.P16  Vd, Vm[0:3], Vs  ; Vd = M × Vs where M is in Vm0-Vm3

; Outer product
VOUTER.P16   Vd[0:3], Vs1, Vs2 ; Vd[i][j] = Vs1[i] × Vs2[j]
```

### 6.2 Matrix-Matrix Operations (using memristor crossbar)

```
; Trigger memristor computation
VMMCROSS.P16 Vd, Vs, mem_addr  ; Matrix-vector multiply using crossbar at mem_addr
```

## 7. Memory Operations

### 7.1 Contiguous Load/Store

```
; Unit-stride load
VLD.P16   Vd, (Rs)          ; Load VL elements starting at Rs
VLD.P28   Vd, (Rs)          ; Load VL P28 elements
VLD.F16   Vd, (Rs)          ; Load VL PentFloat16 elements

; Unit-stride store
VST.P16   Vs, (Rd)          ; Store VL elements starting at Rd
VST.P28   Vs, (Rd)
VST.F16   Vs, (Rd)
```

### 7.2 Strided Load/Store

```
; Strided load
VLDS.P16  Vd, (Rs), Rt      ; Load with stride Rt: Vd[i] = Mem[Rs + i×Rt]
; Useful for loading matrix columns or non-contiguous data

; Strided store
VSTS.P16  Vs, (Rd), Rt      ; Store with stride Rt
```

### 7.3 Indexed (Gather/Scatter)

```
; Gather (indexed load)
VLDI.P16  Vd, (Rs), Vi      ; Vd[i] = Mem[Rs + Vi[i]]
; Useful for sparse matrix operations

; Scatter (indexed store)
VSTI.P16  Vs, (Rd), Vi      ; Mem[Rd + Vi[i]] = Vs[i]
```

### 7.4 Segment Load/Store (for structs)

```
; Segment load (load N-element structures)
VLSEG2.P16  Vd, (Rs)        ; Load pairs: Vd[0], Vd[1] = struct[0], Vd[2], Vd[3] = struct[1], ...
VLSEG3.P16  Vd, (Rs)        ; Load triples
VLSEG4.P16  Vd, (Rs)        ; Load quads (e.g., RGBA pixels)

; Segment store
VSSEG2.P16  Vs, (Rd)        ; Store pairs
```

## 8. Predicated (Masked) Operations

### 8.1 Mask Registers

```
VM0-VM7: 8 mask registers, each with VLEN bits (one bit per element)
```

### 8.2 Masked Instructions

```
; Masked addition (only update where mask bit is 1)
VADD.P16.M  Vd, Vs1, Vs2, Vm  ; Vd[i] = Vs1[i] + Vs2[i] if Vm[i], else unchanged

; Masked load
VLD.P16.M   Vd, (Rs), Vm      ; Load only elements where Vm[i] = 1

; Masked store
VST.P16.M   Vs, (Rd), Vm      ; Store only elements where Vm[i] = 1
```

### 8.3 Mask Manipulation

```
; Mask population count
VMPOPC  Rd, Vm              ; Rd = number of 1 bits in Vm

; Mask find first
VMFFS   Rd, Vm              ; Rd = index of first set bit

; Mask logical operations
VMAND   Vd, Vs1, Vs2        ; Vd = Vs1 AND Vs2 (element-wise)
VMOR    Vd, Vs1, Vs2        ; Vd = Vs1 OR Vs2
VMXOR   Vd, Vs1, Vs2        ; Vd = Vs1 XOR Vs2
VMNOT   Vd, Vs              ; Vd = NOT Vs
```

## 9. Permutation and Shuffle Operations

### 9.1 Element Permutation

```
; Register-controlled permutation
VPERM.P16  Vd, Vs, Vi       ; Vd[i] = Vs[Vi[i]]

; Fixed permutation patterns
VREV.P16   Vd, Vs           ; Reverse element order
VROT.P16   Vd, Vs, imm      ; Rotate elements by imm positions
```

### 9.2 Interleave/Deinterleave

```
; Zip (interleave two vectors)
VZIP.P16   Vd1, Vd2, Vs1, Vs2  ; Interleave: Vd1[0]=Vs1[0], Vd1[1]=Vs2[0], ...

; Unzip (deinterleave)
VUNZIP.P16 Vd1, Vd2, Vs1, Vs2  ; Deinterleave
```

### 9.3 Broadcast and Splat

```
; Broadcast scalar to all elements
VBROADCAST.P16 Vd, Rs       ; Vd[i] = Rs for all i

; Extract element
VEXTRACT.P16   Rd, Vs, idx  ; Rd = Vs[idx]

; Insert element
VINSERT.P16    Vd, Rs, idx  ; Vd[idx] = Rs
```

## 10. Vector Instruction Encoding

### 10.1 Instruction Format

```
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ VOP    │  Vd    │  Vs1   │  Vs2   │  Vm    │ Width  │  Func  │ Extra  │
│ 2 pent │ 1 pent │ 1 pent │ 1 pent │ 1 pent │ 1 pent │ 1 pent │ 0 pent │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Total: 8 pents (same as scalar instruction width)
```

### 10.2 Width Encoding

| Width Code | Element Type | Elements per Register |
|------------|--------------|----------------------|
| 00 | P14 | 18 |
| 01 | P16 | 16 |
| 10 | P28 | 9 |
| 11 | P55 | 4 |

## 11. Implementation Notes

### 11.1 Vector Execution Units

```
┌─────────────────────────────────────────────────────────────┐
│               Vector Functional Units                       │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ VALU ×4   │  │ VMUL ×2   │  │ VFPU ×2   │               │
│  │ (16-wide) │  │ (16-wide) │  │ (16-wide) │               │
│  └───────────┘  └───────────┘  └───────────┘               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ VLSU      │  │ VPERM     │  │ VREDUCE   │               │
│  │ (load/st) │  │ (shuffle) │  │ (reduce)  │               │
│  └───────────┘  └───────────┘  └───────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Performance Characteristics

| Operation Type | Throughput (elements/cycle) | Latency (cycles) |
|----------------|----------------------------|------------------|
| Integer ALU | 16 | 1 |
| Integer MUL | 16 | 4 |
| FP Add | 16 | 4 |
| FP Mul | 16 | 5 |
| FP FMA | 16 | 6 |
| Load | 16 | 4 (L1 hit) |
| Store | 16 | 1 |
| Reduction | 16→1 | 4 |

### 11.3 Vector Chaining

Vector chaining allows back-to-back dependent operations without stalls:

```
VMUL.P16  V1, V2, V3      ; Start multiplication
VADD.P16  V4, V1, V5      ; Can start immediately, chains from V1
```

---

**Document Version**: 1.0
**Status**: Specification Complete
