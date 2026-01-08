# Pentary 3-Transistor (3T) Chip Layout Design

## Document Information
**Version**: 1.0  
**Status**: Complete Technical Specification  
**Target Process**: Sky130A (130nm) / GF180MCU (180nm)  
**Last Updated**: 2026

---

## 1. Executive Summary

This document provides the complete physical layout design for a Pentary 3-Transistor (3T) chip implementation. The design targets Tiny Tapeout fabrication using open-source PDKs.

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Process Node | 130nm (Sky130A) |
| Die Size | 2×2 tiles (334µm × 225µm) |
| Core Voltage | 1.8V |
| Analog Voltage | 3.3V (optional) |
| Total PEs | 256 |
| Transistor Count | ~22,000 |
| Power | ~50mW @ 100MHz |

---

## 2. Chip Architecture Overview

### 2.1 Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PENTARY 3T CHIP (2×2 Tile)                       │
│                         334µm × 225µm = 75,150 µm²                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      ANALOG I/O RING                              │   │
│  │  ua[0]: IN_A   ua[1]: IN_B   ua[2]: OUT   ua[3]: VREF            │   │
│  │  ua[4]: BIAS   ua[5]: TEST                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────┐  ┌─────────────────────────────────┐  ┌────────────┐   │
│  │            │  │                                  │  │            │   │
│  │   LEVEL    │  │     16×16 PROCESSING ELEMENT    │  │   OUTPUT   │   │
│  │ GENERATOR  │  │           ARRAY (256 PEs)       │  │  STAGE &   │   │
│  │    AND     │  │                                  │  │   ADC      │   │
│  │ REFERENCE  │  │      ┌───┬───┬───┬───┐         │  │            │   │
│  │  VOLTAGES  │  │      │PE │PE │PE │PE │  ...    │  │            │   │
│  │            │  │      ├───┼───┼───┼───┤         │  │            │   │
│  │  (20×50µm) │  │      │PE │PE │PE │PE │  ...    │  │ (30×50µm)  │   │
│  │            │  │      ├───┼───┼───┼───┤         │  │            │   │
│  │            │  │      │...│...│...│...│  ...    │  │            │   │
│  │            │  │      └───┴───┴───┴───┘         │  │            │   │
│  │            │  │       (240µm × 160µm)          │  │            │   │
│  └────────────┘  └─────────────────────────────────┘  └────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       CONTROL LOGIC                               │   │
│  │  Clock Buffer | FSM | Refresh Controller | Configuration Regs    │   │
│  │                        (334µm × 30µm)                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      DIGITAL I/O RING                             │   │
│  │  ui[0:7]: Control Inputs    uo[0:7]: Status Outputs              │   │
│  │  uio[0:7]: Bidirectional    clk: Clock   rst: Reset              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Area Allocation

| Block | Size (µm²) | % of Die |
|-------|------------|----------|
| PE Array (256 PEs) | 38,400 | 51.1% |
| Level Generator | 1,000 | 1.3% |
| Output Stage + ADC | 1,500 | 2.0% |
| Control Logic | 10,020 | 13.3% |
| I/O Rings | 15,000 | 20.0% |
| Routing/Margins | 9,230 | 12.3% |
| **Total** | **75,150** | **100%** |

---

## 3. Processing Element (PE) Design

### 3.1 PE Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  PENTARY PROCESSING ELEMENT (PE)                 │
│                        15µm × 10µm = 150µm²                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              3T TRIT STORAGE CELL (6µm²)               │    │
│   │                                                         │    │
│   │     VDD ────┬───────────────────────────────           │    │
│   │             │                                           │    │
│   │        ┌────┴────┐                                     │    │
│   │        │   T2    │  PMOS (Storage Transistor)          │    │
│   │        │ W=2µm   │  Gate Capacitance: ~15fF            │    │
│   │        │ L=0.15µm│                                     │    │
│   │        └────┬────┘                                     │    │
│   │             │ Vs (Storage Node)                        │    │
│   │        ┌────┴────┐     ┌────────┐                     │    │
│   │   WL ──┤   T1    ├─BL  │   T3   ├──SL                 │    │
│   │        │ W=0.5µm │     │ W=1µm  │                     │    │
│   │        │ L=0.15µm│     │ L=0.15µm│                    │    │
│   │        └─────────┘     └────┬───┘                     │    │
│   │                             │                          │    │
│   │                    RL ──────┘                          │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                PENTARY ALU (30 Transistors)            │    │
│   │                                                         │    │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐           │    │
│   │   │  ADDER  │    │ MUL BY  │    │COMPARATOR│          │    │
│   │   │ (18T)   │    │ CONST   │    │   (3T)   │          │    │
│   │   │         │    │  (6T)   │    │          │          │    │
│   │   └────┬────┘    └────┬────┘    └────┬────┘          │    │
│   │        │              │              │                │    │
│   │        └──────────────┼──────────────┘                │    │
│   │                       ▼                                │    │
│   │               ┌───────────────┐                       │    │
│   │               │  RESULT MUX   │                       │    │
│   │               │     (4T)      │                       │    │
│   │               └───────┬───────┘                       │    │
│   │                       │                                │    │
│   └───────────────────────┼────────────────────────────────┘    │
│                           ▼                                      │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                  QUANTIZER (15 Transistors)            │    │
│   │                                                         │    │
│   │   VREF[+2] ──┬──╲                                      │    │
│   │   VREF[+1] ──┼───╲                                     │    │
│   │   VREF[0]  ──┼────╲  5-to-1 ──► Quantized Output      │    │
│   │   VREF[-1] ──┼───╱    MUX                              │    │
│   │   VREF[-2] ──┴──╱                                      │    │
│   │                                                         │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              CONTROL & ROUTING (8 Transistors)          │    │
│   │                                                         │    │
│   │   CLK ─────► Enable Logic                              │    │
│   │   OP[1:0] ──► Operation Select                         │    │
│   │   WE ──────► Write Enable                              │    │
│   │                                                         │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   TOTAL: 56 Transistors per PE                                  │
│   Area: 150µm² (15µm × 10µm)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Transistor Sizing (Sky130A)

| Transistor | Type | Width (µm) | Length (µm) | Purpose |
|------------|------|------------|-------------|---------|
| T1 | NMOS | 0.42 | 0.15 | Write access |
| T2 | PMOS | 2.0 | 0.15 | Storage capacitor |
| T3 | NMOS | 1.0 | 0.15 | Read source follower |
| ALU PMOS | PMOS | 0.84 | 0.15 | Pull-up |
| ALU NMOS | NMOS | 0.42 | 0.15 | Pull-down |
| Comparator | NMOS/PMOS | 0.42 | 0.30 | Current source |

### 3.3 PE Layout (Physical)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PE LAYOUT (15µm × 10µm)                      │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Metal 4 (Power):   VDD ═══════════════════════════════│   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌───────────┬───────────────────────────┬─────────────────┐   │
│   │           │                           │                 │   │
│   │   PMOS    │     NMOS (T1, ALU)       │   Capacitors    │   │
│   │   Well    │                           │   & Resistors   │   │
│   │           │                           │                 │   │
│   │  (T2,     │  ┌─────┬─────┬─────┐     │                 │   │
│   │   ALU)    │  │ T1  │ T3  │ ALU │     │                 │   │
│   │           │  │     │     │     │     │                 │   │
│   │   5µm     │  └─────┴─────┴─────┘     │     3µm         │   │
│   │           │         7µm               │                 │   │
│   └───────────┴───────────────────────────┴─────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Metal 4 (Ground): GND ════════════════════════════════│   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Metal Layers Used:                                             │
│   - Metal 1: Local interconnect (horizontal)                    │
│   - Metal 2: Local interconnect (vertical)                      │
│   - Metal 3: Signal routing                                      │
│   - Metal 4: Power grid (VDD/GND)                               │
│   - Metal 5: Reserved (Tiny Tapeout requirement)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Level Generator and Reference Voltages

### 4.1 Resistor Ladder Design

```
┌─────────────────────────────────────────────────────────────────┐
│              PENTARY LEVEL GENERATOR (20µm × 50µm)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   VDD (1.8V) ────┬─────────────────────────────────────────     │
│                  │                                               │
│                 ╔╧╗  R1 = 360Ω                                  │
│                 ║ ║  (Poly resistor)                            │
│                 ╚╤╝                                              │
│                  │                                               │
│     Level 4 ─────┼──►  VREF[+2] = 1.60V ──► [3T Buffer] ──►    │
│     (+2)         │                                               │
│                 ╔╧╗  R2 = 400Ω                                  │
│                 ║ ║                                              │
│                 ╚╤╝                                              │
│                  │                                               │
│     Level 3 ─────┼──►  VREF[+1] = 1.20V ──► [3T Buffer] ──►    │
│     (+1)         │                                               │
│                 ╔╧╗  R3 = 400Ω                                  │
│                 ║ ║                                              │
│                 ╚╤╝                                              │
│                  │                                               │
│     Level 2 ─────┼──►  VREF[0]  = 0.80V ──► [3T Buffer] ──►    │
│     (0)          │                          (Reference)          │
│                 ╔╧╗  R4 = 400Ω                                  │
│                 ║ ║                                              │
│                 ╚╤╝                                              │
│                  │                                               │
│     Level 1 ─────┼──►  VREF[-1] = 0.40V ──► [3T Buffer] ──►    │
│     (-1)         │                                               │
│                 ╔╧╗  R5 = 360Ω                                  │
│                 ║ ║                                              │
│                 ╚╤╝                                              │
│                  │                                               │
│     Level 0 ─────┴──►  VREF[-2] = 0.00V = GND                   │
│     (-2)                                                         │
│                                                                  │
│   Total Resistance: 1.92 kΩ                                     │
│   Current: 0.94 mA                                               │
│   Power: 1.7 mW                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 3T Buffer Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    3T BUFFER (DVL Topology)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│          VDD                                                     │
│           │                                                      │
│      ┌────┴────┐                                                │
│      │   P1    │  PMOS (W=0.84µm, L=0.15µm)                    │
│      │ Pull-up │                                                │
│      └────┬────┘                                                │
│           │                                                      │
│    Vin ───┼──────────────────────────────────────► Vout         │
│           │                                                      │
│      ┌────┴────┐     ┌────────┐                                │
│      │   N1    │     │   N2   │                                │
│      │         │     │        │                                 │
│      └────┬────┘     └────┬───┘                                │
│           │               │                                      │
│          GND             Vin_bar                                 │
│                                                                  │
│   Function: Non-inverting buffer with high drive strength       │
│   Delay: ~100ps                                                  │
│   Drive: 50µA output current                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Pentary ALU Circuits

### 5.1 Pentary Full Adder

```
┌─────────────────────────────────────────────────────────────────┐
│                 PENTARY FULL ADDER (18 Transistors)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Inputs:  A (5-level), B (5-level), Cin (3-level: -1,0,+1)    │
│   Outputs: Sum (5-level), Cout (3-level)                        │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    SUMMING NETWORK                       │   │
│   │                                                          │   │
│   │        R           R           R                         │   │
│   │   A ──╫──┬──── B ──╫──┬──── Cin ╫──┐                    │   │
│   │        ┃  │         ┃  │         ┃  │                    │   │
│   │        ┗══╪═════════╩══╪═════════╩══╡                    │   │
│   │           │            │            │                    │   │
│   │           └────────────┼────────────┘                    │   │
│   │                        │                                 │   │
│   │                        ▼                                 │   │
│   │              ┌─────────────────┐                        │   │
│   │              │   OP-AMP STAGE  │                        │   │
│   │              │   (6 transistors)│                        │   │
│   │              │   Gain = 0.33   │                        │   │
│   │              └────────┬────────┘                        │   │
│   │                       │                                  │   │
│   │                       ▼  Vsum = (VA + VB + VCin) / 3    │   │
│   └───────────────────────┼─────────────────────────────────┘   │
│                           │                                      │
│   ┌───────────────────────┼─────────────────────────────────┐   │
│   │                QUANTIZER + CARRY DETECT                  │   │
│   │                   (12 transistors)                       │   │
│   │                       │                                  │   │
│   │        ┌──────────────┼──────────────┐                  │   │
│   │        │              │              │                  │   │
│   │        ▼              ▼              ▼                  │   │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐              │   │
│   │   │Comp +1.5│   │Comp +0.5│   │Comp -0.5│   ...        │   │
│   │   └────┬────┘   └────┬────┘   └────┬────┘              │   │
│   │        │              │              │                  │   │
│   │        └──────────────┼──────────────┘                  │   │
│   │                       │                                  │   │
│   │                       ▼                                  │   │
│   │               ┌───────────────┐                         │   │
│   │               │ CARRY LOGIC   │                         │   │
│   │               │ If Vsum > +2: Cout = +1, Sum -= 5      │   │
│   │               │ If Vsum < -2: Cout = -1, Sum += 5      │   │
│   │               │ Else: Cout = 0                          │   │
│   │               └───────┬───────┘                         │   │
│   │                       │                                  │   │
│   └───────────────────────┼─────────────────────────────────┘   │
│                           │                                      │
│                    ┌──────┴──────┐                               │
│                    │    Sum      │   Cout                        │
│                    └─────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Pentary Comparator

```
┌─────────────────────────────────────────────────────────────────┐
│               PENTARY COMPARATOR (3 Transistors)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│              VDD                                                 │
│               │                                                  │
│          ┌────┴────┐                                            │
│          │   P1    │  Current Mirror Load                       │
│          │ W=0.84µm│                                            │
│          └────┬────┘                                            │
│               │                                                  │
│        ┌──────┼──────┐                                          │
│        │      │      │                                          │
│   ┌────┴────┐ │ ┌────┴────┐                                    │
│   │   N1    │ │ │   N2    │  Differential Pair                 │
│   │         │ │ │         │                                     │
│   └────┬────┘ │ └────┬────┘                                    │
│        │      │      │                                          │
│       VA     │     VB (Reference)                               │
│              │                                                   │
│         ┌────┴────┐                                             │
│         │   N3    │  Tail Current Source                        │
│         │ W=0.84µm│  Ibias = 10µA                               │
│         │ L=0.30µm│                                             │
│         └────┬────┘                                             │
│              │                                                   │
│             GND                                                  │
│                                                                  │
│   Output:                                                        │
│     Vout = HIGH if VA > VB                                      │
│     Vout = LOW  if VA < VB                                      │
│                                                                  │
│   Specifications:                                                │
│     Gain: 60 dB                                                 │
│     Offset: < 10 mV                                             │
│     Bandwidth: 100 MHz                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Array Architecture

### 6.1 16×16 PE Array Organization

```
┌─────────────────────────────────────────────────────────────────┐
│                    16×16 PE ARRAY LAYOUT                         │
│                     (240µm × 160µm)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Column Address → [0] [1] [2] [3] [4] [5] [6] [7] ... [15]    │
│                     │   │   │   │   │   │   │   │       │      │
│                     ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼       ▼      │
│   ═══════════════════════════════════════════════════════════  │
│   ║                  COLUMN DRIVERS (Write)                   ║  │
│   ║             (Level DACs, 5-level output)                 ║  │
│   ═══════════════════════════════════════════════════════════  │
│                     │   │   │   │   │   │   │   │       │      │
│   Row[0]  ──► WL ──[PE][PE][PE][PE][PE][PE][PE][PE] ... [PE]   │
│               RL    │   │   │   │   │   │   │   │       │      │
│                     │   │   │   │   │   │   │   │       │      │
│   Row[1]  ──► WL ──[PE][PE][PE][PE][PE][PE][PE][PE] ... [PE]   │
│               RL    │   │   │   │   │   │   │   │       │      │
│                     │   │   │   │   │   │   │   │       │      │
│   Row[2]  ──► WL ──[PE][PE][PE][PE][PE][PE][PE][PE] ... [PE]   │
│               RL    │   │   │   │   │   │   │   │       │      │
│                    ...                                          │
│                     │   │   │   │   │   │   │   │       │      │
│   Row[15] ──► WL ──[PE][PE][PE][PE][PE][PE][PE][PE] ... [PE]   │
│               RL    │   │   │   │   │   │   │   │       │      │
│                     │   │   │   │   │   │   │   │       │      │
│   ═══════════════════════════════════════════════════════════  │
│   ║                   SENSE AMPLIFIERS                        ║  │
│   ║             (5-level ADCs, one per column)               ║  │
│   ═══════════════════════════════════════════════════════════  │
│                     │   │   │   │   │   │   │   │       │      │
│                     ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼       ▼      │
│                 Output Data Bus (16 × 3-bit = 48 bits)         │
│                                                                  │
│   Dimensions:                                                    │
│     PE pitch: 15µm × 10µm                                       │
│     Array core: 240µm × 160µm                                   │
│     With drivers: 260µm × 180µm                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Row and Column Addressing

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADDRESS DECODER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │              4-to-16 ROW DECODER                       │    │
│   │                                                         │    │
│   │   Row Address [3:0] ──► ┌───────────────────────────┐  │    │
│   │                         │                           │  │    │
│   │                         │   Binary to One-Hot      │  │    │
│   │                         │   16 AND gates           │  │    │
│   │                         │   (4-input each)         │  │    │
│   │                         │                           │  │    │
│   │                         └──────────┬────────────────┘  │    │
│   │                                    │                    │    │
│   │                                    ▼                    │    │
│   │                           WL[0:15], RL[0:15]           │    │
│   │                                                         │    │
│   │   Transistors: 16 × 8 = 128 transistors               │    │
│   │   Delay: 2 gate delays (~400ps)                        │    │
│   │                                                         │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │            4-to-16 COLUMN MULTIPLEXER                  │    │
│   │                                                         │    │
│   │   Col Address [3:0] ──► ┌───────────────────────────┐  │    │
│   │                         │                           │  │    │
│   │   SL[0:15] ────────────►│   16:1 Analog MUX        │  │    │
│   │                         │   (Pass Transistor)      │  │    │
│   │                         │                           │  │    │
│   │                         └──────────┬────────────────┘  │    │
│   │                                    │                    │    │
│   │                                    ▼                    │    │
│   │                             Selected Output             │    │
│   │                                                         │    │
│   │   Transistors: 15 × 4 = 60 transistors                 │    │
│   │   Delay: 4 gate delays (~800ps)                        │    │
│   │                                                         │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Control Logic

### 7.1 Finite State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL FSM                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   States:                                                        │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                                                        │    │
│   │     ┌───────┐                                         │    │
│   │     │ IDLE  │◄────────────────────────┐              │    │
│   │     └───┬───┘                         │              │    │
│   │         │ start                       │ done          │    │
│   │         ▼                             │              │    │
│   │     ┌───────┐     ┌───────┐     ┌────┴───┐         │    │
│   │     │ LOAD  │────►│ EXEC  │────►│ STORE  │         │    │
│   │     └───────┘     └───┬───┘     └────────┘         │    │
│   │                       │                              │    │
│   │                       │ refresh_needed               │    │
│   │                       ▼                              │    │
│   │                   ┌───────┐                         │    │
│   │                   │REFRESH│                         │    │
│   │                   └───┬───┘                         │    │
│   │                       │                              │    │
│   │                       └──────────────────────────────┘    │
│   │                                                        │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
│   State Encoding:                                                │
│     IDLE    = 3'b000                                            │
│     LOAD    = 3'b001                                            │
│     EXEC    = 3'b010                                            │
│     STORE   = 3'b011                                            │
│     REFRESH = 3'b100                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Refresh Controller

```
┌─────────────────────────────────────────────────────────────────┐
│                   REFRESH CONTROLLER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                                                        │    │
│   │   CLK ──► ┌───────────────┐                           │    │
│   │           │  64ms Timer   │                           │    │
│   │           │  (16-bit)     │──► refresh_trigger        │    │
│   │           └───────────────┘                           │    │
│   │                                                        │    │
│   │   refresh_trigger ──► ┌───────────────┐               │    │
│   │                       │  Row Counter  │               │    │
│   │                       │  (4-bit)      │──► refresh_row│    │
│   │                       └───────────────┘               │    │
│   │                                                        │    │
│   │   ┌───────────────────────────────────────────────┐  │    │
│   │   │  Refresh Sequence (per row):                   │  │    │
│   │   │                                                │  │    │
│   │   │  1. Assert RL[refresh_row]     (5ns)          │  │    │
│   │   │  2. Read all columns to buffer  (10ns)        │  │    │
│   │   │  3. Assert WL[refresh_row]      (5ns)         │  │    │
│   │   │  4. Write buffer back           (10ns)        │  │    │
│   │   │  5. Deassert WL, RL             (5ns)         │  │    │
│   │   │                                                │  │    │
│   │   │  Total per row: 35ns                          │  │    │
│   │   │  Total for array: 35ns × 16 = 560ns           │  │    │
│   │   │  Refresh overhead: 0.00088% (negligible)      │  │    │
│   │   │                                                │  │    │
│   │   └───────────────────────────────────────────────┘  │    │
│   │                                                        │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Power Distribution

### 8.1 Power Grid Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                     POWER GRID (Metal 4)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   VDD Rail (1.8V) ══════════════════════════════════════════   │
│   Width: 2µm, Spacing: 10µm                                      │
│       │       │       │       │       │       │                  │
│       │       │       │       │       │       │                  │
│   ════╪═══════╪═══════╪═══════╪═══════╪═══════╪════ VDD          │
│       │       │       │       │       │       │    strap         │
│   ────┼───────┼───────┼───────┼───────┼───────┼──── (M3)         │
│       │       │       │       │       │       │                  │
│   ════╪═══════╪═══════╪═══════╪═══════╪═══════╪════ GND          │
│       │       │       │       │       │       │    strap         │
│   ────┼───────┼───────┼───────┼───────┼───────┼──── (M3)         │
│       │       │       │       │       │       │                  │
│   ════╪═══════╪═══════╪═══════╪═══════╪═══════╪════ VDD          │
│       │       │       │       │       │       │                  │
│                                                                  │
│   GND Rail (0V) ════════════════════════════════════════════    │
│                                                                  │
│   Decoupling Capacitors:                                         │
│     - MOS capacitors (PMOS in N-well)                           │
│     - 10fF per PE                                                │
│     - Total: 2.56pF                                              │
│                                                                  │
│   Power Consumption:                                              │
│     - Core logic: 30mW                                           │
│     - Level generator: 1.7mW                                     │
│     - Analog circuits: 15mW                                      │
│     - I/O: 5mW                                                   │
│     - Total: ~52mW                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Pin Assignment

### 9.1 Analog Pins (ua[0:5])

| Pin | Name | Direction | Description | Voltage Range |
|-----|------|-----------|-------------|---------------|
| ua[0] | IN_A | Input | Pentary Input A | 0 - 1.6V |
| ua[1] | IN_B | Input | Pentary Input B | 0 - 1.6V |
| ua[2] | OUT | Output | Pentary Output | 0 - 1.6V |
| ua[3] | VREF | Input | Reference (0.8V) | 0.7 - 0.9V |
| ua[4] | IBIAS | Input | Bias Current | 0 - 100µA |
| ua[5] | TEST | I/O | Test/Debug | 0 - 1.8V |

### 9.2 Digital Pins

| Pin | Name | Direction | Description |
|-----|------|-----------|-------------|
| ui[0] | CLK | Input | System clock (100MHz max) |
| ui[1] | RST | Input | Active-low reset |
| ui[2:4] | OP | Input | Operation select (3-bit) |
| ui[5] | START | Input | Start operation |
| ui[6] | WE | Input | Write enable |
| ui[7] | RE | Input | Read enable |
| uo[0:2] | STATUS | Output | Status (ready, busy, error) |
| uo[3:7] | DEBUG | Output | Debug signals |
| uio[0:3] | ADDR | Bidir | Address input |
| uio[4:7] | DATA | Bidir | Data I/O |

---

## 10. Fabrication Checklist

### 10.1 Pre-Submission Verification

- [ ] DRC (Design Rule Check) - 0 errors
- [ ] LVS (Layout vs Schematic) - Match
- [ ] ERC (Electrical Rule Check) - 0 errors
- [ ] Antenna check - Pass
- [ ] Metal density check - 20-80% all layers
- [ ] Power grid IR drop - < 5% VDD
- [ ] Timing analysis - All paths < 10ns

### 10.2 File Deliverables

```
pentary_3t_chip/
├── gds/
│   └── tt_um_pentary_3t.gds          # Final GDS-II
├── lef/
│   └── tt_um_pentary_3t.lef          # LEF (pins only)
├── verilog/
│   └── tt_um_pentary_3t.v            # Verilog stub
├── spice/
│   └── tt_um_pentary_3t.spice        # Extracted netlist
├── docs/
│   ├── info.yaml                      # Tiny Tapeout metadata
│   ├── pinout.md                      # Pin descriptions
│   └── datasheet.pdf                  # Chip datasheet
└── test/
    ├── tb_pentary_3t.v               # Verilog testbench
    └── tb_pentary_3t.sp              # SPICE testbench
```

---

## 11. Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Die Size | 334µm × 225µm | 2×2 Tiny Tapeout tiles |
| Process | Sky130A (130nm) | Open-source PDK |
| Transistors | ~22,000 | Including periphery |
| PEs | 256 (16×16 array) | 56 transistors each |
| Clock | 100 MHz max | Single-phase clock |
| Throughput | 256 MOPS | 1 op/PE/cycle |
| Power | 52 mW | @ 100 MHz, 1.8V |
| Energy | 200 fJ/op | Per pentary operation |
| Latency | 10ns | Single operation |
| Refresh | 64ms | ~0.001% overhead |

---

**Document Complete**

**Next Steps:**
1. Create Xschem schematics for all cells
2. Simulate with ngspice
3. Create Magic layouts
4. Run full DRC/LVS
5. Submit to Tiny Tapeout (TT10+)

**Target Shuttle:** TT11 (Q1 2026)  
**Estimated Cost:** €660 (2×2 tiles + 6 analog pins)
