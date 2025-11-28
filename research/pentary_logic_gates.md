# Pentary Logic Gates and Truth Tables

![Diagram](../diagrams/pentary_logic_gates.png)


## 1. Introduction to Pentary Logic

Pentary logic extends ternary logic to 5 discrete states: {⊖, -, 0, +, ⊕} representing {-2, -1, 0, +1, +2}.

### 1.1 Logic Value Interpretation

| Value | Numeric | Logical Meaning | Voltage |
|-------|---------|-----------------|---------|
| ⊖ | -2 | Strong False | 0V |
| - | -1 | Weak False | 1.25V |
| 0 | 0 | Unknown/Neutral | 2.5V |
| + | +1 | Weak True | 3.75V |
| ⊕ | +2 | Strong True | 5V |

## 2. Basic Pentary Logic Gates

### 2.1 NOT Gate (Negation)

**Function**: Inverts the input value

| Input | Output |
|-------|--------|
| ⊖ | ⊕ |
| - | + |
| 0 | 0 |
| + | - |
| ⊕ | ⊖ |

**Implementation**: Simple voltage inverter around 2.5V midpoint

### 2.2 MIN Gate (Minimum)

**Function**: Returns the minimum of two inputs

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊖ | ⊖ | ⊖ | ⊖ | ⊖ |
| - | ⊖ | - | - | - | - |
| 0 | ⊖ | - | 0 | 0 | 0 |
| + | ⊖ | - | 0 | + | + |
| ⊕ | ⊖ | - | 0 | + | ⊕ |

**Properties**:
- Commutative: MIN(A,B) = MIN(B,A)
- Associative: MIN(A,MIN(B,C)) = MIN(MIN(A,B),C)
- Identity: MIN(A,⊕) = A

### 2.3 MAX Gate (Maximum)

**Function**: Returns the maximum of two inputs

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊖ | - | 0 | + | ⊕ |
| - | - | - | 0 | + | ⊕ |
| 0 | 0 | 0 | 0 | + | ⊕ |
| + | + | + | + | + | ⊕ |
| ⊕ | ⊕ | ⊕ | ⊕ | ⊕ | ⊕ |

**Properties**:
- Commutative: MAX(A,B) = MAX(B,A)
- Associative: MAX(A,MAX(B,C)) = MAX(MAX(A,B),C)
- Identity: MAX(A,⊖) = A

### 2.4 CONSENSUS Gate

**Function**: Returns the "consensus" or middle value

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊖ | ⊖ | ⊖ | 0 | 0 |
| - | ⊖ | - | - | 0 | 0 |
| 0 | ⊖ | - | 0 | + | ⊕ |
| + | 0 | 0 | + | + | ⊕ |
| ⊕ | 0 | 0 | ⊕ | ⊕ | ⊕ |

**Properties**:
- Used in carry computation
- Symmetric around zero

### 2.5 MODULO-5 SUM Gate

**Function**: Adds two values modulo 5 (no carry)

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊕ | ⊖ | - | 0 | + |
| - | ⊖ | - | 0 | + | ⊕ |
| 0 | - | 0 | + | ⊕ | ⊖ |
| + | 0 | + | ⊕ | ⊖ | - |
| ⊕ | + | ⊕ | ⊖ | - | 0 |

**Properties**:
- Commutative
- Used in half-adder sum computation

## 3. Comparison Gates

### 3.1 EQUAL Gate

**Function**: Returns ⊕ if inputs are equal, ⊖ otherwise

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊕ | ⊖ | ⊖ | ⊖ | ⊖ |
| - | ⊖ | ⊕ | ⊖ | ⊖ | ⊖ |
| 0 | ⊖ | ⊖ | ⊕ | ⊖ | ⊖ |
| + | ⊖ | ⊖ | ⊖ | ⊕ | ⊖ |
| ⊕ | ⊖ | ⊖ | ⊖ | ⊖ | ⊕ |

### 3.2 GREATER THAN Gate

**Function**: Returns ⊕ if A > B, ⊖ if A < B, 0 if A = B

| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | 0 | ⊖ | ⊖ | ⊖ | ⊖ |
| - | ⊕ | 0 | ⊖ | ⊖ | ⊖ |
| 0 | ⊕ | ⊕ | 0 | ⊖ | ⊖ |
| + | ⊕ | ⊕ | ⊕ | 0 | ⊖ |
| ⊕ | ⊕ | ⊕ | ⊕ | ⊕ | 0 |

## 4. Arithmetic Support Gates

### 4.1 SIGN Gate

**Function**: Returns the sign of the input

| Input | Output |
|-------|--------|
| ⊖ | ⊖ |
| - | - |
| 0 | 0 |
| + | + |
| ⊕ | ⊕ |

(Identity function, but conceptually extracts sign)

### 4.2 ABSOLUTE VALUE Gate

**Function**: Returns the absolute value

| Input | Output |
|-------|--------|
| ⊖ | ⊕ |
| - | + |
| 0 | 0 |
| + | + |
| ⊕ | ⊕ |

### 4.3 CLAMP Gate

**Function**: Clamps value to range [min, max]

Example: CLAMP(x, -, +)

| Input | Output |
|-------|--------|
| ⊖ | - |
| - | - |
| 0 | 0 |
| + | + |
| ⊕ | + |

## 5. Decoder and Encoder Gates

### 5.1 One-Hot Decoder

**Function**: Converts pentary value to 5-bit one-hot encoding

| Input | Out₀ | Out₁ | Out₂ | Out₃ | Out₄ |
|-------|------|------|------|------|------|
| ⊖ | 1 | 0 | 0 | 0 | 0 |
| - | 0 | 1 | 0 | 0 | 0 |
| 0 | 0 | 0 | 1 | 0 | 0 |
| + | 0 | 0 | 0 | 1 | 0 |
| ⊕ | 0 | 0 | 0 | 0 | 1 |

### 5.2 Priority Encoder

**Function**: Converts one-hot to pentary value

| In₀ | In₁ | In₂ | In₃ | In₄ | Output |
|-----|-----|-----|-----|-----|--------|
| 1 | X | X | X | X | ⊖ |
| 0 | 1 | X | X | X | - |
| 0 | 0 | 1 | X | X | 0 |
| 0 | 0 | 0 | 1 | X | + |
| 0 | 0 | 0 | 0 | 1 | ⊕ |

## 6. Multiplexer and Demultiplexer

### 6.1 2-to-1 Multiplexer

**Function**: Selects one of two inputs based on select signal

```
OUT = (SEL = ⊖) ? A : B
```

| SEL | A | B | OUT |
|-----|---|---|-----|
| ⊖ | X | Y | X |
| - | X | Y | X |
| 0 | X | Y | ? |
| + | X | Y | Y |
| ⊕ | X | Y | Y |

### 6.2 5-to-1 Multiplexer

**Function**: Selects one of five inputs

| SEL | OUT |
|-----|-----|
| ⊖ | IN₀ |
| - | IN₁ |
| 0 | IN₂ |
| + | IN₃ |
| ⊕ | IN₄ |

## 7. Threshold Gates

### 7.1 Threshold-2 Gate

**Function**: Returns ⊕ if input ≥ threshold, ⊖ otherwise

Example: TH₂(x) with threshold = 0

| Input | Output |
|-------|--------|
| ⊖ | ⊖ |
| - | ⊖ |
| 0 | ⊕ |
| + | ⊕ |
| ⊕ | ⊕ |

### 7.2 Multi-Threshold Gate

**Function**: Quantizes input to pentary levels

| Input Range | Output |
|-------------|--------|
| x < -1.5 | ⊖ |
| -1.5 ≤ x < -0.5 | - |
| -0.5 ≤ x < 0.5 | 0 |
| 0.5 ≤ x < 1.5 | + |
| x ≥ 1.5 | ⊕ |

## 8. Composite Gates for Arithmetic

### 8.1 Half Adder

**Inputs**: A, B
**Outputs**: Sum, Carry

**Sum Truth Table**:
| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | ⊕ | ⊖ | - | 0 | + |
| - | ⊖ | - | 0 | + | ⊕ |
| 0 | - | 0 | + | ⊕ | ⊖ |
| + | 0 | + | ⊕ | ⊖ | - |
| ⊕ | + | ⊕ | ⊖ | - | 0 |

**Carry Truth Table**:
| A\B | ⊖ | - | 0 | + | ⊕ |
|-----|---|---|---|---|---|
| ⊖ | - | - | ⊖ | ⊖ | 0 |
| - | - | ⊖ | ⊖ | 0 | 0 |
| 0 | ⊖ | ⊖ | 0 | 0 | ⊕ |
| + | ⊖ | 0 | 0 | ⊕ | ⊕ |
| ⊕ | 0 | 0 | ⊕ | ⊕ | + |

### 8.2 Full Adder

**Inputs**: A, B, Carry_in
**Outputs**: Sum, Carry_out

Too large to show complete table (125 entries), but follows pattern:
- Sum = (A + B + C_in) mod 5, adjusted for balanced representation
- Carry_out = ⌊(A + B + C_in) / 5⌋, adjusted for balanced representation

## 9. Gate Implementation Complexity

### 9.1 Transistor Count Estimates

| Gate Type | Transistors (est.) | Notes |
|-----------|-------------------|-------|
| NOT | 4-6 | Voltage inverter |
| MIN/MAX | 12-16 | Comparator + mux |
| CONSENSUS | 20-30 | Complex logic |
| MOD-5 SUM | 30-40 | Modular arithmetic |
| EQUAL | 16-20 | 5-way comparison |
| HALF ADDER | 60-80 | Sum + carry logic |
| FULL ADDER | 100-120 | Complex carry logic |

### 9.2 Comparison with Binary

| Operation | Binary | Pentary | Ratio |
|-----------|--------|---------|-------|
| NOT | 2 | 4-6 | 2-3× |
| AND/MIN | 6 | 12-16 | 2-2.7× |
| Half Adder | 10 | 60-80 | 6-8× |
| Full Adder | 28 | 100-120 | 3.6-4.3× |

**Key Insight**: Per-gate complexity is higher, but information density (2.32 bits/pent) partially compensates.

## 10. Optimized Gate Designs

### 10.1 Voltage-Mode Logic

**Advantages**:
- Direct voltage comparison
- Natural threshold detection
- Analog-friendly

**Disadvantages**:
- Noise sensitivity
- Power consumption
- Precise voltage levels required

### 10.2 Current-Mode Logic

**Advantages**:
- Better noise immunity
- Lower voltage swing
- Faster switching

**Disadvantages**:
- Higher power consumption
- More complex design

### 10.3 Memristor-Based Logic

**Advantages**:
- Non-volatile state
- Low power (especially for 0 state)
- Natural multi-level representation

**Disadvantages**:
- Endurance limitations
- Variability
- Immature technology

## 11. Logic Minimization

### 11.1 Karnaugh Map Equivalent

For pentary logic, we need 5-dimensional maps (one per value). Example for 2-input function:

```
     ⊖  -  0  +  ⊕
  ⊖ [·][·][·][·][·]
  -  [·][·][·][·][·]
  0  [·][·][·][·][·]
  +  [·][·][·][·][·]
  ⊕  [·][·][·][·][·]
```

### 11.2 Sum-of-Products Form

Pentary logic can be expressed as:
```
F(A,B) = Σ (minterms)
```

Where each minterm is a conjunction of:
- (A = ⊖), (A = -), (A = 0), (A = +), (A = ⊕)
- (B = ⊖), (B = -), (B = 0), (B = +), (B = ⊕)

### 11.3 Optimization Techniques

1. **Quine-McCluskey Extension**: Adapt for 5-valued logic
2. **Espresso-like Heuristics**: Multi-valued logic minimization
3. **BDD (Binary Decision Diagram) Extension**: Multi-valued decision diagrams

## 12. Standard Cell Library

### 12.1 Proposed Standard Cells

1. **PENT_NOT**: Inverter
2. **PENT_MIN2**: 2-input minimum
3. **PENT_MAX2**: 2-input maximum
4. **PENT_CONS2**: 2-input consensus
5. **PENT_ADD**: Half adder
6. **PENT_FADD**: Full adder
7. **PENT_MUX2**: 2-to-1 multiplexer
8. **PENT_MUX5**: 5-to-1 multiplexer
9. **PENT_DEC**: One-hot decoder
10. **PENT_ENC**: Priority encoder
11. **PENT_REG**: Register (flip-flop)
12. **PENT_COMP**: Comparator

### 12.2 Timing Characteristics

| Cell | Delay (normalized) | Power (normalized) |
|------|-------------------|-------------------|
| PENT_NOT | 1.0 | 1.0 |
| PENT_MIN2 | 1.5 | 1.8 |
| PENT_MAX2 | 1.5 | 1.8 |
| PENT_ADD | 3.0 | 4.0 |
| PENT_FADD | 4.5 | 6.0 |
| PENT_MUX2 | 2.0 | 2.5 |

(Normalized to binary NOT gate = 1.0)

## 13. Future Research

1. **Optimal Gate Synthesis**: Finding minimal gate implementations
2. **Technology Mapping**: Mapping to specific technologies (CMOS, memristor, etc.)
3. **Power Optimization**: Exploiting zero-state power savings
4. **Fault Tolerance**: Error detection and correction in pentary logic
5. **Hybrid Binary-Pentary**: Interfacing with binary control logic

---

**References:**
- Multi-Valued Logic Design (IEEE papers)
- Ternary Logic Gates (adapted to pentary)
- VLSI Design for Multi-Valued Logic