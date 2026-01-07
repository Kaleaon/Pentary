# Pentary Breadboard-Level PCB Design

## Document Information
**Version**: 1.0  
**Status**: Complete Design Specification  
**Target**: Educational / Prototyping Platform  
**Last Updated**: 2026

---

## 1. Executive Summary

This document provides a complete design for a **breadboard-level PCB** that demonstrates Pentary computing principles using discrete, off-the-shelf components. This design allows experimentation with pentary logic without requiring custom IC fabrication.

### Design Goals

1. **Educational**: Teach pentary computing concepts hands-on
2. **Accessible**: Use only commonly available components
3. **Modular**: Build incrementally, debug easily
4. **Low-Cost**: Under $100 for complete prototype
5. **Breadboard-Compatible**: Can be built on standard breadboards first

### Key Features

- 4-trit pentary register (20 voltage levels)
- Pentary ALU (add, subtract, compare)
- 5-level DAC and ADC
- LED display showing pentary values
- Arduino/Raspberry Pi interface
- Expandable design

---

## 2. System Architecture

### 2.1 Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PENTARY BREADBOARD COMPUTER                               │
│                        PCB: 100mm × 160mm                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        POWER SECTION                                   │  │
│  │                                                                        │  │
│  │   USB-C     ┌─────────┐   ┌─────────┐   ┌─────────┐                  │  │
│  │   5V ──────►│ LM7805  │──►│ LM317   │──►│ TL431   │                  │  │
│  │   Input     │ +5V     │   │ +1.8V   │   │ VREF    │                  │  │
│  │             └─────────┘   └─────────┘   └─────────┘                  │  │
│  │                                                                        │  │
│  │   Outputs: +5V (logic), +3.3V (MCU), +1.8V (analog), VREF (0.4V/div) │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐   │
│  │                   │  │                   │  │                       │   │
│  │   5-LEVEL DAC     │  │   PENTARY ALU     │  │   5-LEVEL ADC        │   │
│  │                   │  │                   │  │                       │   │
│  │   ┌───────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────────┐ │   │
│  │   │ R-2R      │  │  │  │ Op-amp      │  │  │  │ Comparator     │ │   │
│  │   │ Ladder    │  │  │  │ Summer      │  │  │  │ Bank           │ │   │
│  │   │ + MUX     │──┼──┼─►│             │──┼──┼─►│ + Encoder      │ │   │
│  │   └───────────┘  │  │  │ + Quantizer │  │  │  └─────────────────┘ │   │
│  │                   │  │  └─────────────┘  │  │                       │   │
│  │   Digital Input   │  │   A + B = Sum     │  │   Analog to Digital  │   │
│  │   (3-bit/trit)   │  │                   │  │   (3-bit/trit)       │   │
│  │                   │  │                   │  │                       │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘   │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        PENTARY REGISTER FILE                           │  │
│  │                                                                        │  │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │   │ TRIT 0   │  │ TRIT 1   │  │ TRIT 2   │  │ TRIT 3   │             │  │
│  │   │          │  │          │  │          │  │          │             │  │
│  │   │ S/H +    │  │ S/H +    │  │ S/H +    │  │ S/H +    │             │  │
│  │   │ Buffer   │  │ Buffer   │  │ Buffer   │  │ Buffer   │             │  │
│  │   │          │  │          │  │          │  │          │             │  │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘             │  │
│  │                                                                        │  │
│  │   Each trit: 5-level analog storage (0.0V, 0.4V, 0.8V, 1.2V, 1.6V)   │  │
│  │   Total: 4 trits = 5⁴ = 625 possible states                          │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DISPLAY & INTERFACE                             │  │
│  │                                                                        │  │
│  │   ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐ │  │
│  │   │                   │  │                   │  │                  │ │  │
│  │   │   LED BAR GRAPH   │  │   7-SEG DISPLAY   │  │   MCU HEADER     │ │  │
│  │   │   (20 LEDs)       │  │   (4 digits)      │  │   (Arduino)      │ │  │
│  │   │                   │  │                   │  │                  │ │  │
│  │   │   Shows analog    │  │   Shows pentary   │  │   SPI/I2C        │ │  │
│  │   │   voltage levels  │  │   digit values    │  │   interface      │ │  │
│  │   │                   │  │                   │  │                  │ │  │
│  │   └───────────────────┘  └───────────────────┘  └──────────────────┘ │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Signal Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL FLOW DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DIGITAL INPUT (from MCU or switches)                                       │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────┐                                                       │
│   │   3-bit code    │  Value encoding:                                      │
│   │   per trit      │    000 = -2 (0.0V)                                    │
│   │                 │    001 = -1 (0.4V)                                    │
│   │   [D2 D1 D0]    │    010 =  0 (0.8V)                                    │
│   │                 │    011 = +1 (1.2V)                                    │
│   └────────┬────────┘    100 = +2 (1.6V)                                    │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │   5-LEVEL DAC   │  R-2R ladder + analog switch                          │
│   │                 │  Converts 3-bit code to voltage                       │
│   └────────┬────────┘                                                       │
│            │  Analog voltage (0.0V - 1.6V)                                  │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │  PENTARY ALU    │  Op-amp based arithmetic                              │
│   │                 │  Summer: Vout = (Va + Vb) × 0.5                       │
│   │  A op B = Result│  Quantizer: rounds to nearest 0.4V step              │
│   └────────┬────────┘                                                       │
│            │  Quantized analog result                                       │
│            ▼                                                                 │
│   ┌─────────────────┐                                                       │
│   │   5-LEVEL ADC   │  4 comparators + priority encoder                     │
│   │                 │  Converts voltage back to 3-bit code                  │
│   └────────┬────────┘                                                       │
│            │                                                                 │
│            ▼                                                                 │
│   DIGITAL OUTPUT (to MCU or display)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Power Supply Design

### 3.1 Power Supply Schematic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POWER SUPPLY SECTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   USB-C Input (5V, 2A max)                                                   │
│       │                                                                      │
│       │    ┌───────────────────────────────────────────────────────────┐    │
│       │    │  PROTECTION                                                │    │
│       └───►│                                                            │    │
│            │   D1: SS14 (Schottky)  ─── Reverse polarity protection    │    │
│            │   F1: 500mA PTC Fuse   ─── Overcurrent protection         │    │
│            │   C1: 100µF/10V        ─── Input bulk capacitor           │    │
│            │                                                            │    │
│            └───────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│       ┌────────────────────────────┼────────────────────────────────────┐   │
│       │                            │                                     │   │
│       ▼                            ▼                                     │   │
│   ┌───────────────────┐    ┌───────────────────┐                        │   │
│   │   LM7805          │    │   LM1117-3.3      │                        │   │
│   │   +5V Regulator   │    │   +3.3V Regulator │                        │   │
│   │                   │    │                   │                        │   │
│   │  IN─┬─OUT         │    │  IN─┬─OUT         │                        │   │
│   │     │   │         │    │     │   │         │                        │   │
│   │   ┌─┴─┐ │         │    │   ┌─┴─┐ │         │                        │   │
│   │   │C2 │ │         │    │   │C4 │ │         │                        │   │
│   │   │0.1│ │         │    │   │0.1│ │         │                        │   │
│   │   │µF │ │         │    │   │µF │ │         │                        │   │
│   │   └───┘ │         │    │   └───┘ │         │                        │   │
│   │     GND │         │    │     GND │         │                        │   │
│   └─────────┼─────────┘    └─────────┼─────────┘                        │   │
│             │                        │                                   │   │
│         +5V ◄───────────────────     +3.3V ◄─────────────────────       │   │
│         (Logic)                      (MCU)                               │   │
│                                                                          │   │
│   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │  ANALOG REFERENCE GENERATION                                     │   │   │
│   │                                                                  │   │   │
│   │  +5V ────┬──────────────────────────────────────────────────    │   │   │
│   │          │                                                       │   │   │
│   │         ┌┴┐  R1 = 1kΩ                                           │   │   │
│   │         │ │                                                      │   │   │
│   │         └┬┘                                                      │   │   │
│   │          │                                                       │   │   │
│   │          ├──────────────────────────────►  +1.8V (VANALOG)      │   │   │
│   │          │                                                       │   │   │
│   │       ┌──┴──┐                                                   │   │   │
│   │       │ U1  │  LM317 (1.25V + 0.55V = 1.80V)                   │   │   │
│   │       │     │  R_adj = 1kΩ (sets voltage)                       │   │   │
│   │       └──┬──┘                                                   │   │   │
│   │          │                                                       │   │   │
│   │         GND                                                      │   │   │
│   │                                                                  │   │   │
│   │  VREF Generation (0.4V steps):                                  │   │   │
│   │                                                                  │   │   │
│   │  +1.8V ──┬────────────────────────────────────────────────     │   │   │
│   │          │                                                       │   │   │
│   │         ┌┴┐  R_a = 1kΩ                                          │   │   │
│   │         └┬┘                                                      │   │   │
│   │          ├────────────────────────────────►  VREF4 = 1.6V       │   │   │
│   │         ┌┴┐  R_b = 1kΩ                                          │   │   │
│   │         └┬┘                                                      │   │   │
│   │          ├────────────────────────────────►  VREF3 = 1.2V       │   │   │
│   │         ┌┴┐  R_c = 1kΩ                                          │   │   │
│   │         └┬┘                                                      │   │   │
│   │          ├────────────────────────────────►  VREF2 = 0.8V       │   │   │
│   │         ┌┴┐  R_d = 1kΩ                                          │   │   │
│   │         └┬┘                                                      │   │   │
│   │          ├────────────────────────────────►  VREF1 = 0.4V       │   │   │
│   │         ┌┴┐  R_e = 1kΩ                                          │   │   │
│   │         └┬┘                                                      │   │   │
│   │          │                                                       │   │   │
│   │         GND ──────────────────────────────►  VREF0 = 0.0V       │   │   │
│   │                                                                  │   │   │
│   └─────────────────────────────────────────────────────────────────┘   │   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Power Budget

| Rail | Voltage | Max Current | Components |
|------|---------|-------------|------------|
| VIN | 5V | 2A | USB-C input |
| VLOGIC | 5V | 500mA | 74HC logic, LEDs |
| VMCU | 3.3V | 200mA | Arduino/MCU |
| VANALOG | 1.8V | 100mA | Op-amps, analog |
| VREF | 0.4V steps | 10mA | Reference ladder |

---

## 4. Five-Level DAC Design

### 4.1 DAC Schematic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      5-LEVEL DAC (PER TRIT)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Digital Input:  D[2:0] from MCU or switches                               │
│   Analog Output:  VOUT = {0.0V, 0.4V, 0.8V, 1.2V, 1.6V}                     │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  OPTION A: 74HC4051 Analog Multiplexer                               │   │
│   │                                                                      │   │
│   │   VREF4 (1.6V) ───► Y0 ┐                                            │   │
│   │   VREF3 (1.2V) ───► Y1 │                                            │   │
│   │   VREF2 (0.8V) ───► Y2 ├─► 74HC4051 ──► Z ──┬──► VOUT              │   │
│   │   VREF1 (0.4V) ───► Y3 │      MUX           │                       │   │
│   │   VREF0 (0.0V) ───► Y4 ┘                    │                       │   │
│   │                                              │                       │   │
│   │   D[2:0] ─────────────► A B C (select)     │                       │   │
│   │                                              │                       │   │
│   │   Note: Only use 5 of 8 inputs             │                       │   │
│   │                                              │                       │   │
│   │   Output Buffer:                            │                       │   │
│   │                          ┌────────────┐     │                       │   │
│   │                     ┌────┤+           │     │                       │   │
│   │   Z ────────────────┤    │  LM358    ├─────┘                       │   │
│   │                     │ ┌──┤-           │                             │   │
│   │                     │ │  └────────────┘                             │   │
│   │                     │ │       │                                     │   │
│   │                     └─┼───────┘  Unity gain buffer                 │   │
│   │                       │                                             │   │
│   │                      VOUT                                           │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  OPTION B: R-2R Ladder DAC (3-bit) + Scaling                        │   │
│   │                                                                      │   │
│   │         VLOGIC (5V)                                                  │   │
│   │              │                                                       │   │
│   │   D2 ──[SW]──┤                                                      │   │
│   │              │                                                       │   │
│   │             ┌┴┐ 2R                                                  │   │
│   │             └┬┘                                                      │   │
│   │   D1 ──[SW]──┼──┬──►                                                │   │
│   │              │  │                                                    │   │
│   │             ┌┴┐ 2R  ┌┐ R                                            │   │
│   │             └┬┘     └┘                                               │   │
│   │   D0 ──[SW]──┼──┬──►                                                │   │
│   │              │  │                                                    │   │
│   │             ┌┴┐ 2R  ┌┐ R                                            │   │
│   │             └┬┘     └┘                                               │   │
│   │              │                                                       │   │
│   │             GND                                                      │   │
│   │                                                                      │   │
│   │   R = 10kΩ                                                          │   │
│   │   Output: 0 to 5V in 0.625V steps (8 levels)                        │   │
│   │   Scale to 1.6V with resistor divider or op-amp                     │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 DAC Component List

| Ref | Component | Value | Purpose |
|-----|-----------|-------|---------|
| U1 | 74HC4051 | - | 8:1 analog mux |
| U2 | LM358 | - | Op-amp buffer |
| R1-R5 | Resistor | 1kΩ 1% | Voltage divider |
| C1 | Capacitor | 100nF | Decoupling |
| C2 | Capacitor | 10pF | Compensation |

---

## 5. Pentary ALU Design

### 5.1 Analog Summer/Adder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PENTARY ANALOG ADDER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Inputs:  VA (0.0V to 1.6V), VB (0.0V to 1.6V)                            │
│   Output:  VSUM = (VA + VB) / 2, then quantized                             │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SUMMING AMPLIFIER                                                   │   │
│   │                                                                      │   │
│   │                   R1 = 20kΩ                                         │   │
│   │   VA ────────────╫────────┬──────────────────────────────────       │   │
│   │                           │                                          │   │
│   │                   R2 = 20kΩ                                         │   │
│   │   VB ────────────╫────────┤                                         │   │
│   │                           │                                          │   │
│   │                           │   Rf = 10kΩ                             │   │
│   │                           ├────╫────┐                               │   │
│   │                           │         │                               │   │
│   │                       ┌───┴───┐     │                               │   │
│   │   VREF (0.8V) ───────┤-      │     │                               │   │
│   │                       │ LM358 ├─────┴───► VSUM_RAW                  │   │
│   │                   ┌───┤+      │                                     │   │
│   │                   │   └───────┘                                     │   │
│   │                   │                                                  │   │
│   │                  GND (via voltage divider for offset)               │   │
│   │                                                                      │   │
│   │   VSUM_RAW = VREF - (Rf/R1)(VA - VREF) - (Rf/R2)(VB - VREF)        │   │
│   │            = 0.8 - 0.5(VA - 0.8) - 0.5(VB - 0.8)                   │   │
│   │            = 0.8 - 0.5VA + 0.4 - 0.5VB + 0.4                       │   │
│   │            = 1.6 - 0.5(VA + VB)                                     │   │
│   │                                                                      │   │
│   │   For VA=VB=0.8V (both = 0): VSUM_RAW = 0.8V ✓                     │   │
│   │   For VA=0.0V, VB=0.0V (-2+-2=-4→-2+carry): needs adjustment       │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  QUANTIZER (Rounds to nearest 0.4V step)                            │   │
│   │                                                                      │   │
│   │   VSUM_RAW ───┬────────────────────────────────────────────────     │   │
│   │               │                                                      │   │
│   │               │     VREF3=1.4V   VREF2=1.0V   VREF1=0.6V   VREF0=0.2V │
│   │               │         │           │           │           │        │   │
│   │               │         ▼           ▼           ▼           ▼        │   │
│   │               │     ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    │   │
│   │               ├────►│ CMP3 │    │ CMP2 │    │ CMP1 │    │ CMP0 │    │   │
│   │               │     └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘    │   │
│   │               │        │           │           │           │        │   │
│   │               │        ▼           ▼           ▼           ▼        │   │
│   │               │     ┌─────────────────────────────────────────┐     │   │
│   │               │     │         PRIORITY ENCODER                │     │   │
│   │               │     │         (74HC148 or logic)             │     │   │
│   │               │     └────────────────┬────────────────────────┘     │   │
│   │               │                      │                              │   │
│   │               │                      ▼                              │   │
│   │               │               Quantized Output [2:0]                │   │
│   │               │                                                      │   │
│   │   Comparator outputs (thermometer code):                            │   │
│   │     1111 → +2 (1.6V)                                                │   │
│   │     0111 → +1 (1.2V)                                                │   │
│   │     0011 →  0 (0.8V)                                                │   │
│   │     0001 → -1 (0.4V)                                                │   │
│   │     0000 → -2 (0.0V)                                                │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Carry Detection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CARRY DETECTION CIRCUIT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Pentary addition can produce results from -4 to +4:                       │
│     -4 = -2 + carry(-1)  →  output -2, carry -1                            │
│     -3 = +2 + carry(-1)  →  output +2, carry -1                            │
│     -2 to +2             →  output as-is, carry 0                          │
│     +3 = -2 + carry(+1)  →  output -2, carry +1                            │
│     +4 = +2 + carry(+1)  →  output +2, carry +1                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  CARRY COMPARATORS                                                   │   │
│   │                                                                      │   │
│   │   VSUM_RAW ──┬───────────────────────────────────────────────       │   │
│   │              │                                                       │   │
│   │              │   VTHRESH_HIGH = 1.8V (represents +2.5)              │   │
│   │              │        │                                              │   │
│   │              │        ▼                                              │   │
│   │              │    ┌──────┐                                          │   │
│   │              ├───►│CMP_HI│──► CARRY_POS (if VSUM > 1.8V)           │   │
│   │              │    └──────┘                                          │   │
│   │              │                                                       │   │
│   │              │   VTHRESH_LOW = -0.2V (represents -2.5)              │   │
│   │              │        │                                              │   │
│   │              │        ▼                                              │   │
│   │              │    ┌──────┐                                          │   │
│   │              └───►│CMP_LO│──► CARRY_NEG (if VSUM < -0.2V)          │   │
│   │                   └──────┘                                          │   │
│   │                                                                      │   │
│   │   CARRY output encoding:                                            │   │
│   │     CARRY_POS=0, CARRY_NEG=0  →  CARRY = 0  (no overflow)          │   │
│   │     CARRY_POS=1, CARRY_NEG=0  →  CARRY = +1 (positive overflow)    │   │
│   │     CARRY_POS=0, CARRY_NEG=1  →  CARRY = -1 (negative overflow)    │   │
│   │                                                                      │   │
│   │   When carry occurs, subtract/add 5 levels (2.0V) from sum         │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Five-Level ADC Design

### 6.1 Flash ADC Schematic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      5-LEVEL FLASH ADC                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Analog Input:  VIN (0.0V to 1.6V)                                         │
│   Digital Output: D[2:0] = {000, 001, 010, 011, 100}                        │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  COMPARATOR BANK (LM339 - Quad Comparator)                          │   │
│   │                                                                      │   │
│   │                     ┌─────────────────────────────────────────────┐ │   │
│   │   VIN ──────────────┤                                             │ │   │
│   │                     │                                             │ │   │
│   │   VREF3 = 1.4V ─────┤  ┌──────┐                                  │ │   │
│   │                     ├─►│ CMP3 ├──► C3  (VIN > 1.4V?)            │ │   │
│   │                     │  └──────┘                                  │ │   │
│   │                     │                                             │ │   │
│   │   VREF2 = 1.0V ─────┤  ┌──────┐                                  │ │   │
│   │                     ├─►│ CMP2 ├──► C2  (VIN > 1.0V?)            │ │   │
│   │                     │  └──────┘                                  │ │   │
│   │                     │                                             │ │   │
│   │   VREF1 = 0.6V ─────┤  ┌──────┐                                  │ │   │
│   │                     ├─►│ CMP1 ├──► C1  (VIN > 0.6V?)            │ │   │
│   │                     │  └──────┘                                  │ │   │
│   │                     │                                             │ │   │
│   │   VREF0 = 0.2V ─────┤  ┌──────┐                                  │ │   │
│   │                     └─►│ CMP0 ├──► C0  (VIN > 0.2V?)            │ │   │
│   │                        └──────┘                                  │ │   │
│   │                                                                      │   │
│   │   Threshold voltages set at midpoints:                              │   │
│   │     Between -2 (0.0V) and -1 (0.4V): 0.2V                          │   │
│   │     Between -1 (0.4V) and  0 (0.8V): 0.6V                          │   │
│   │     Between  0 (0.8V) and +1 (1.2V): 1.0V                          │   │
│   │     Between +1 (1.2V) and +2 (1.6V): 1.4V                          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  THERMOMETER TO BINARY ENCODER                                      │   │
│   │                                                                      │   │
│   │   Comparator Outputs:      Pentary Value:     Binary Output:        │   │
│   │   [C3 C2 C1 C0]                                [D2 D1 D0]           │   │
│   │   ────────────────         ─────────────      ─────────────         │   │
│   │    0  0  0  0              -2 (0.0V)          0  0  0               │   │
│   │    0  0  0  1              -1 (0.4V)          0  0  1               │   │
│   │    0  0  1  1               0 (0.8V)          0  1  0               │   │
│   │    0  1  1  1              +1 (1.2V)          0  1  1               │   │
│   │    1  1  1  1              +2 (1.6V)          1  0  0               │   │
│   │                                                                      │   │
│   │   Logic equations:                                                  │   │
│   │     D2 = C3                                                         │   │
│   │     D1 = C2 XOR C3   (or C1 AND NOT C3)                            │   │
│   │     D0 = C0 XOR C1   (or C0 AND NOT C2 AND NOT C3) OR (C2 AND C3)  │   │
│   │                                                                      │   │
│   │   Implementation: 74HC86 (XOR gates) + 74HC08 (AND gates)          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Pentary Register Design

### 7.1 Sample-and-Hold Trit Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PENTARY REGISTER (Single Trit)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Function: Store one pentary digit as an analog voltage                    │
│   Retention: Several seconds (requires periodic refresh)                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SAMPLE-AND-HOLD CIRCUIT                                             │   │
│   │                                                                      │   │
│   │                           CD4066 Analog Switch                       │   │
│   │                                │                                     │   │
│   │   VIN ──────────────────────┬──┤◄──── CLK (sample pulse)            │   │
│   │   (from DAC or ALU)         │  │                                     │   │
│   │                             │  │                                     │   │
│   │                             │  ▼                                     │   │
│   │                             └──┬──┐                                  │   │
│   │                                │  │                                  │   │
│   │                               ┌┴┐ │                                  │   │
│   │                               │ │ │  C_HOLD = 100nF                 │   │
│   │                               │C│ │  (low leakage film capacitor)   │   │
│   │                               │ │ │                                  │   │
│   │                               └┬┘ │                                  │   │
│   │                                │  │                                  │   │
│   │                               GND │                                  │   │
│   │                                   │                                  │   │
│   │   BUFFER AMPLIFIER:              │                                  │   │
│   │                                   │                                  │   │
│   │                            ┌──────┴──────┐                          │   │
│   │                        ┌───┤+            │                          │   │
│   │                        │   │   TL071     ├──────────► VOUT          │   │
│   │                        │ ┌─┤-   (JFET    │     (to ADC or next     │   │
│   │                        │ │ │   input)    │      stage)             │   │
│   │                        │ │ └─────────────┘                          │   │
│   │                        │ │       │                                  │   │
│   │                        └─┼───────┘                                  │   │
│   │                          │                                          │   │
│   │                     Feedback (unity gain)                           │   │
│   │                                                                      │   │
│   │   Key specifications:                                               │   │
│   │     - JFET op-amp: Very low input bias current (~10pA)             │   │
│   │     - Film capacitor: Low leakage, stable                           │   │
│   │     - Droop rate: ~1mV/s (need refresh every ~400ms)               │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  4-TRIT REGISTER BANK                                                │   │
│   │                                                                      │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │   │
│   │   │ TRIT 0  │  │ TRIT 1  │  │ TRIT 2  │  │ TRIT 3  │               │   │
│   │   │  (LSD)  │  │         │  │         │  │  (MSD)  │               │   │
│   │   │         │  │         │  │         │  │         │               │   │
│   │   │  S/H    │  │  S/H    │  │  S/H    │  │  S/H    │               │   │
│   │   │  +Buf   │  │  +Buf   │  │  +Buf   │  │  +Buf   │               │   │
│   │   │         │  │         │  │         │  │         │               │   │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │   │
│   │        │            │            │            │                     │   │
│   │        ▼            ▼            ▼            ▼                     │   │
│   │   ┌─────────────────────────────────────────────────┐              │   │
│   │   │            ANALOG MULTIPLEXER (4:1)             │              │   │
│   │   │                    74HC4052                      │              │   │
│   │   └──────────────────────┬──────────────────────────┘              │   │
│   │                          │                                          │   │
│   │                          ▼                                          │   │
│   │                    Selected TRIT Output                             │   │
│   │                                                                      │   │
│   │   Total capacity: 5⁴ = 625 unique values                           │   │
│   │   Equivalent to: log₂(625) = 9.29 bits                             │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Display and LED Indicators

### 8.1 LED Bar Graph Display

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LED BAR GRAPH DISPLAY (Per Trit)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Shows analog voltage level using 5 LEDs (one per pentary value)           │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  VOLTAGE-TO-LED DRIVER                                               │   │
│   │                                                                      │   │
│   │   VIN (0-1.6V) ──┬──────────────────────────────────────────────    │   │
│   │                  │                                                   │   │
│   │                  │                                                   │   │
│   │   VREF = 1.4V ───┼──┬──┐                                            │   │
│   │                  │  │  ├──► CMP4 ──► LED4 (Red)     "+2"            │   │
│   │                  │  │  │                                             │   │
│   │   VREF = 1.0V ───┼──┼──┼──┬──┐                                      │   │
│   │                  │  │  │  │  ├──► CMP3 ──► LED3 (Yellow) "+1"       │   │
│   │                  │  │  │  │  │                                       │   │
│   │   VREF = 0.6V ───┼──┼──┼──┼──┼──┬──┐                                │   │
│   │                  │  │  │  │  │  │  ├──► CMP2 ──► LED2 (Green)  "0"  │   │
│   │                  │  │  │  │  │  │  │                                 │   │
│   │   VREF = 0.2V ───┴──┴──┴──┴──┴──┴──┴──┬──┐                          │   │
│   │                                       │  ├──► CMP1 ──► LED1 (Blue) "-1" │
│   │                                       │  │                           │   │
│   │   (Always on for any valid input) ────┴──┴──► CMP0 ──► LED0 (White)"-2"│
│   │                                                                      │   │
│   │   Window Comparator Logic:                                          │   │
│   │     LED4 lights when 1.4V < VIN < 1.8V (+2)                        │   │
│   │     LED3 lights when 1.0V < VIN < 1.4V (+1)                        │   │
│   │     LED2 lights when 0.6V < VIN < 1.0V (0)                         │   │
│   │     LED1 lights when 0.2V < VIN < 0.6V (-1)                        │   │
│   │     LED0 lights when 0.0V < VIN < 0.2V (-2)                        │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Physical Layout (4 trits):                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                      │   │
│   │     TRIT 3      TRIT 2      TRIT 1      TRIT 0                     │   │
│   │     (MSD)                               (LSD)                       │   │
│   │                                                                      │   │
│   │   ○ +2 (R)    ○ +2 (R)    ○ +2 (R)    ● +2 (R)  ← Currently lit    │   │
│   │   ○ +1 (Y)    ● +1 (Y)    ○ +1 (Y)    ○ +1 (Y)                     │   │
│   │   ○  0 (G)    ○  0 (G)    ● 0  (G)    ○  0 (G)                     │   │
│   │   ● -1 (B)    ○ -1 (B)    ○ -1 (B)    ○ -1 (B)                     │   │
│   │   ○ -2 (W)    ○ -2 (W)    ○ -2 (W)    ○ -2 (W)                     │   │
│   │                                                                      │   │
│   │   Display shows: (-1, +1, 0, +2) = -1×125 + 1×25 + 0×5 + 2×1       │   │
│   │                                  = -125 + 25 + 0 + 2 = -98          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Seven-Segment Display

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    7-SEGMENT DISPLAY DRIVER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Shows pentary digit as character: ⊖, -, 0, +, ⊕                          │
│   (Encoded as: 2̲, 1̲, 0, 1, 2 with underline for negative)                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  CHARACTER ENCODING                                                  │   │
│   │                                                                      │   │
│   │   Pentary   Character    Segments                                    │   │
│   │   ─────────────────────────────────────────────                     │   │
│   │     -2        2̲          a,b,d,e,g + DP (underline)                 │   │
│   │     -1        1̲          b,c + DP                                   │   │
│   │      0        0          a,b,c,d,e,f                                │   │
│   │     +1        1          b,c                                        │   │
│   │     +2        2          a,b,d,e,g                                  │   │
│   │                                                                      │   │
│   │   DP (decimal point) indicates negative value                       │   │
│   │                                                                      │   │
│   │         aaaa                                                        │   │
│   │        f    b                                                       │   │
│   │        f    b                                                       │   │
│   │         gggg                                                        │   │
│   │        e    c                                                       │   │
│   │        e    c                                                       │   │
│   │         dddd   ○DP                                                  │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  DRIVER CIRCUIT                                                      │   │
│   │                                                                      │   │
│   │   Digital Input [2:0] ──► 74HC4511 ──► Common Cathode 7-Seg        │   │
│   │                           BCD to 7-Seg                              │   │
│   │                           + Logic for DP                            │   │
│   │                                                                      │   │
│   │   For 4-digit display, use multiplexing:                           │   │
│   │   - 74HC595 shift registers (one per digit)                        │   │
│   │   - Or: MAX7219 LED driver (single chip solution)                  │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. MCU Interface

### 9.1 Arduino Connection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARDUINO UNO INTERFACE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PIN ASSIGNMENT                                                      │   │
│   │                                                                      │   │
│   │   Arduino Pin    Direction    Signal                                │   │
│   │   ─────────────────────────────────────────────                     │   │
│   │   D2             Output       DAC_D0 (LSB)                          │   │
│   │   D3             Output       DAC_D1                                │   │
│   │   D4             Output       DAC_D2 (MSB)                          │   │
│   │   D5             Output       TRIT_SEL_0                            │   │
│   │   D6             Output       TRIT_SEL_1                            │   │
│   │   D7             Output       SAMPLE_CLK                            │   │
│   │   D8             Output       ALU_OP_0                              │   │
│   │   D9             Output       ALU_OP_1                              │   │
│   │   D10            Input        ADC_D0 (LSB)                          │   │
│   │   D11            Input        ADC_D1                                │   │
│   │   D12            Input        ADC_D2 (MSB)                          │   │
│   │   D13            Output       STATUS_LED                            │   │
│   │   A0             Input        ANALOG_MON (optional)                 │   │
│   │   A1             Input        VREF_MON (optional)                   │   │
│   │   5V             Power        VLOGIC                                │   │
│   │   GND            Power        Ground                                │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  CONNECTOR PINOUT (2×10 header)                                     │   │
│   │                                                                      │   │
│   │     1  2  3  4  5  6  7  8  9  10                                  │   │
│   │   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                                  │   │
│   │   │D2│D3│D4│D5│D6│D7│D8│D9│5V│GND│  (Top row)                      │   │
│   │   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤                                  │   │
│   │   │D10│D11│D12│D13│A0│A1│NC│NC│3V3│GND│  (Bottom row)               │   │
│   │   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                                  │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Example Arduino Code

```cpp
// Pentary Breadboard Computer Controller
// Arduino Sketch

// Pin definitions
#define DAC_D0    2
#define DAC_D1    3
#define DAC_D2    4
#define TRIT_SEL0 5
#define TRIT_SEL1 6
#define SAMPLE_CLK 7
#define ALU_OP0   8
#define ALU_OP1   9
#define ADC_D0    10
#define ADC_D1    11
#define ADC_D2    12
#define STATUS_LED 13

// Pentary value encoding
// -2 = 0b000, -1 = 0b001, 0 = 0b010, +1 = 0b011, +2 = 0b100
const int PENT_NEG2 = 0;
const int PENT_NEG1 = 1;
const int PENT_ZERO = 2;
const int PENT_POS1 = 3;
const int PENT_POS2 = 4;

void setup() {
  // DAC outputs
  pinMode(DAC_D0, OUTPUT);
  pinMode(DAC_D1, OUTPUT);
  pinMode(DAC_D2, OUTPUT);
  
  // Control outputs
  pinMode(TRIT_SEL0, OUTPUT);
  pinMode(TRIT_SEL1, OUTPUT);
  pinMode(SAMPLE_CLK, OUTPUT);
  pinMode(ALU_OP0, OUTPUT);
  pinMode(ALU_OP1, OUTPUT);
  
  // ADC inputs
  pinMode(ADC_D0, INPUT);
  pinMode(ADC_D1, INPUT);
  pinMode(ADC_D2, INPUT);
  
  pinMode(STATUS_LED, OUTPUT);
  
  Serial.begin(9600);
  Serial.println("Pentary Breadboard Computer Ready");
}

// Write a pentary value to DAC
void writePentary(int value) {
  // value: -2 to +2, mapped to 0-4
  int code = value + 2;  // Convert to 0-4
  
  digitalWrite(DAC_D0, code & 0x01);
  digitalWrite(DAC_D1, (code >> 1) & 0x01);
  digitalWrite(DAC_D2, (code >> 2) & 0x01);
}

// Read pentary value from ADC
int readPentary() {
  int code = digitalRead(ADC_D0);
  code |= (digitalRead(ADC_D1) << 1);
  code |= (digitalRead(ADC_D2) << 2);
  
  return code - 2;  // Convert 0-4 to -2 to +2
}

// Select trit register (0-3)
void selectTrit(int trit) {
  digitalWrite(TRIT_SEL0, trit & 0x01);
  digitalWrite(TRIT_SEL1, (trit >> 1) & 0x01);
}

// Store value in selected trit
void storeTrit(int value) {
  writePentary(value);
  
  // Pulse sample clock
  digitalWrite(SAMPLE_CLK, HIGH);
  delayMicroseconds(10);
  digitalWrite(SAMPLE_CLK, LOW);
}

// Pentary addition: a + b
int pentaryAdd(int a, int b) {
  int sum = a + b;
  
  // Handle carry
  if (sum > 2) {
    sum -= 5;
    // Carry = +1 (propagate to next trit)
  } else if (sum < -2) {
    sum += 5;
    // Carry = -1
  }
  
  return sum;
}

void loop() {
  // Demo: Count in pentary
  static int counter = 0;
  
  // Convert counter to 4-trit pentary
  int temp = counter;
  for (int trit = 0; trit < 4; trit++) {
    int digit = (temp % 5) - 2;  // -2 to +2
    temp /= 5;
    
    selectTrit(trit);
    storeTrit(digit);
  }
  
  // Display on serial
  Serial.print("Counter: ");
  Serial.print(counter);
  Serial.print(" = [");
  for (int trit = 3; trit >= 0; trit--) {
    selectTrit(trit);
    int val = readPentary();
    if (val >= 0) Serial.print("+");
    Serial.print(val);
    if (trit > 0) Serial.print(", ");
  }
  Serial.println("]");
  
  // Increment counter
  counter++;
  if (counter >= 625) counter = 0;  // 5^4 = 625
  
  delay(500);
  
  // Toggle status LED
  digitalWrite(STATUS_LED, counter & 1);
}
```

---

## 10. Bill of Materials (BOM)

### 10.1 Core Components

| Ref | Qty | Component | Value/Part | Unit Cost | Total |
|-----|-----|-----------|------------|-----------|-------|
| U1 | 1 | Voltage Regulator | LM7805 | $0.50 | $0.50 |
| U2 | 1 | Voltage Regulator | LM1117-3.3 | $0.60 | $0.60 |
| U3 | 1 | Adjustable Reg | LM317 | $0.50 | $0.50 |
| U4-U7 | 4 | Analog Mux | 74HC4051 | $0.50 | $2.00 |
| U8-U9 | 2 | Quad Op-Amp | LM358 | $0.40 | $0.80 |
| U10-U11 | 2 | JFET Op-Amp | TL072 | $0.60 | $1.20 |
| U12 | 1 | Quad Comparator | LM339 | $0.50 | $0.50 |
| U13 | 1 | Analog Switch | CD4066 | $0.40 | $0.40 |
| U14-U17 | 4 | Sample/Hold | LF398 (or discrete) | $2.00 | $8.00 |
| U18 | 1 | BCD to 7-Seg | 74HC4511 | $0.60 | $0.60 |
| - | - | - | - | **Subtotal** | **$15.10** |

### 10.2 Passive Components

| Ref | Qty | Component | Value | Unit Cost | Total |
|-----|-----|-----------|-------|-----------|-------|
| R | 20 | Resistor 1% | 1kΩ | $0.02 | $0.40 |
| R | 20 | Resistor 1% | 10kΩ | $0.02 | $0.40 |
| R | 10 | Resistor 1% | 20kΩ | $0.02 | $0.20 |
| R | 10 | Various | 100Ω-100kΩ | $0.02 | $0.20 |
| C | 10 | Ceramic | 100nF | $0.05 | $0.50 |
| C | 5 | Electrolytic | 100µF/25V | $0.10 | $0.50 |
| C | 4 | Film (S/H) | 100nF | $0.20 | $0.80 |
| C | 5 | Ceramic | 10pF | $0.05 | $0.25 |
| - | - | - | - | **Subtotal** | **$3.25** |

### 10.3 Display and Interface

| Ref | Qty | Component | Description | Unit Cost | Total |
|-----|-----|-----------|-------------|-----------|-------|
| LED | 20 | 5mm LED | Mixed colors | $0.10 | $2.00 |
| DIS | 4 | 7-Segment | Common cathode | $0.50 | $2.00 |
| SW | 8 | Tactile switch | 6mm | $0.10 | $0.80 |
| J1 | 1 | USB-C connector | Power input | $0.80 | $0.80 |
| J2 | 1 | Header 2×10 | Arduino interface | $0.50 | $0.50 |
| J3 | 4 | Header 1×3 | Test points | $0.10 | $0.40 |
| - | - | - | - | **Subtotal** | **$6.50** |

### 10.4 PCB and Mechanical

| Item | Qty | Description | Unit Cost | Total |
|------|-----|-------------|-----------|-------|
| PCB | 1 | 100×160mm, 2-layer | $5.00 | $5.00 |
| Standoffs | 4 | M3×10mm | $0.20 | $0.80 |
| Screws | 8 | M3×6mm | $0.05 | $0.40 |
| IC Sockets | 15 | DIP-8, DIP-14, DIP-16 | $0.20 | $3.00 |
| Wire | 1 | Hook-up wire set | $3.00 | $3.00 |
| - | - | - | **Subtotal** | **$12.20** |

### 10.5 Total Cost Summary

| Category | Cost |
|----------|------|
| Core ICs | $15.10 |
| Passives | $3.25 |
| Display/Interface | $6.50 |
| PCB/Mechanical | $12.20 |
| **Grand Total** | **$37.05** |

**With 20% contingency: ~$45**

---

## 11. PCB Layout

### 11.1 Board Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PCB LAYOUT (100mm × 160mm)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ● ○ ○ ○    PENTARY BREADBOARD COMPUTER    ○ ○ ○ ●                  │   │
│  │  (mounting holes at corners)                                          │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  ┌────────────────┐  ┌────────────────────────────────────────────┐ │   │
│  │  │                │  │                                             │ │   │
│  │  │  POWER SUPPLY  │  │           LED DISPLAY SECTION              │ │   │
│  │  │                │  │                                             │ │   │
│  │  │  USB-C  [■]    │  │    [●][●][●][●]   [●][●][●][●]            │ │   │
│  │  │                │  │    [●][●][●][●]   [●][●][●][●]            │ │   │
│  │  │  LM7805 [■]    │  │    [●][●][●][●]   [●][●][●][●]            │ │   │
│  │  │  LM1117 [■]    │  │    [●][●][●][●]   [●][●][●][●]            │ │   │
│  │  │  LM317  [■]    │  │    [●][●][●][●]   [●][●][●][●]            │ │   │
│  │  │                │  │                                             │ │   │
│  │  │  Indicator LEDs│  │    TRIT3  TRIT2  TRIT1  TRIT0              │ │   │
│  │  │  [●] [●] [●]   │  │                                             │ │   │
│  │  │                │  │    ┌────┐ ┌────┐ ┌────┐ ┌────┐            │ │   │
│  │  └────────────────┘  │    │ 88 │ │ 88 │ │ 88 │ │ 88 │  7-Seg     │ │   │
│  │                      │    └────┘ └────┘ └────┘ └────┘            │ │   │
│  │                      │                                             │ │   │
│  │                      └────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      ANALOG SECTION                            │  │   │
│  │  │                                                                │  │   │
│  │  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │  │   │
│  │  │   │  DAC 0  │    │  DAC 1  │    │  DAC 2  │    │  DAC 3  │  │  │   │
│  │  │   │ 74HC4051│    │ 74HC4051│    │ 74HC4051│    │ 74HC4051│  │  │   │
│  │  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │  │   │
│  │  │                                                                │  │   │
│  │  │   ┌─────────┐    ┌─────────┐                                  │  │   │
│  │  │   │   ALU   │    │  S & H  │    Reference voltage section    │  │   │
│  │  │   │ LM358×2 │    │ TL072×2 │    (resistor ladder)            │  │   │
│  │  │   └─────────┘    └─────────┘                                  │  │   │
│  │  │                                                                │  │   │
│  │  │   ┌─────────┐    ┌─────────┐                                  │  │   │
│  │  │   │   ADC   │    │ Encoder │    Test points                   │  │   │
│  │  │   │  LM339  │    │ 74HC logic│   [TP1][TP2][TP3][TP4]        │  │   │
│  │  │   └─────────┘    └─────────┘                                  │  │   │
│  │  │                                                                │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      INTERFACE SECTION                         │  │   │
│  │  │                                                                │  │   │
│  │  │   ┌─────────────────────────┐    ┌────────────────────────┐  │  │   │
│  │  │   │    ARDUINO HEADER       │    │    MANUAL CONTROLS     │  │  │   │
│  │  │   │                         │    │                        │  │  │   │
│  │  │   │   [■■■■■■■■■■]          │    │   [SW][SW][SW][SW]    │  │  │   │
│  │  │   │   [■■■■■■■■■■]          │    │   [SW][SW][SW][SW]    │  │  │   │
│  │  │   │                         │    │                        │  │  │   │
│  │  │   └─────────────────────────┘    └────────────────────────┘  │  │   │
│  │  │                                                                │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ● ○ ○ ○                                          ○ ○ ○ ●            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   PCB Specifications:                                                        │
│   - Size: 100mm × 160mm (Eurocard format)                                   │
│   - Layers: 2 (top copper + bottom copper)                                  │
│   - Copper: 1 oz                                                             │
│   - Finish: HASL or ENIG                                                    │
│   - Silkscreen: Top and bottom                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Assembly and Testing

### 12.1 Assembly Order

1. **Power Supply First**
   - Solder voltage regulators
   - Add filter capacitors
   - Test: Verify 5V, 3.3V, 1.8V outputs

2. **Reference Voltages**
   - Install resistor ladder
   - Test: Verify 0.4V, 0.8V, 1.2V, 1.6V levels

3. **DAC Section**
   - Install 74HC4051 multiplexers
   - Add buffer op-amps
   - Test: Verify 5-level output

4. **ALU Section**
   - Install LM358 op-amps
   - Configure summer circuit
   - Test: Verify addition function

5. **ADC Section**
   - Install LM339 comparators
   - Add encoder logic
   - Test: Verify digitization

6. **Registers**
   - Install sample-and-hold circuits
   - Test: Verify storage and readback

7. **Display**
   - Install LEDs and 7-segment displays
   - Test: Verify indication

8. **MCU Interface**
   - Install headers
   - Test: Full system with Arduino

### 12.2 Test Procedure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUNCTIONAL TEST CHECKLIST                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   □ Power Supply                                                            │
│     □ Input voltage: 5V USB ±5%                                            │
│     □ Logic rail: 5.0V ±2%                                                 │
│     □ MCU rail: 3.3V ±2%                                                   │
│     □ Analog rail: 1.8V ±2%                                                │
│     □ Current draw: <200mA quiescent                                       │
│                                                                              │
│   □ Reference Voltages                                                      │
│     □ VREF0: 0.0V (GND)                                                    │
│     □ VREF1: 0.4V ±10mV                                                    │
│     □ VREF2: 0.8V ±10mV                                                    │
│     □ VREF3: 1.2V ±10mV                                                    │
│     □ VREF4: 1.6V ±10mV                                                    │
│                                                                              │
│   □ DAC Function                                                            │
│     □ Input 000 → Output 0.0V ±50mV                                        │
│     □ Input 001 → Output 0.4V ±50mV                                        │
│     □ Input 010 → Output 0.8V ±50mV                                        │
│     □ Input 011 → Output 1.2V ±50mV                                        │
│     □ Input 100 → Output 1.6V ±50mV                                        │
│                                                                              │
│   □ ADC Function                                                            │
│     □ Input 0.0V → Output 000                                              │
│     □ Input 0.4V → Output 001                                              │
│     □ Input 0.8V → Output 010                                              │
│     □ Input 1.2V → Output 011                                              │
│     □ Input 1.6V → Output 100                                              │
│                                                                              │
│   □ ALU Function (Addition)                                                 │
│     □ 0 + 0 = 0                                                            │
│     □ +1 + +1 = +2                                                         │
│     □ +2 + +1 = -2 (with carry +1)                                         │
│     □ -1 + -1 = -2                                                         │
│     □ -2 + -1 = +2 (with carry -1)                                         │
│                                                                              │
│   □ Register Function                                                       │
│     □ Store 0, readback 0                                                  │
│     □ Store +2, readback +2                                                │
│     □ Store -2, readback -2                                                │
│     □ Retention >1 second                                                  │
│                                                                              │
│   □ Display Function                                                        │
│     □ All LEDs light in sequence                                           │
│     □ 7-segment shows all digits                                           │
│                                                                              │
│   □ MCU Interface                                                           │
│     □ Arduino can write all values                                         │
│     □ Arduino can read all values                                          │
│     □ Counter demo runs correctly                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Summary

### What This Board Demonstrates

1. **Pentary Number System**: Physical representation of 5-level values
2. **Analog Computing**: Using voltages for computation
3. **DAC/ADC Design**: Converting between digital and analog
4. **Arithmetic Operations**: Addition in balanced base-5
5. **Register Storage**: Analog sample-and-hold
6. **MCU Integration**: Interfacing with Arduino

### Learning Outcomes

- Understanding of multi-valued logic
- Analog circuit design principles  
- Mixed-signal system design
- Trade-offs between analog and digital

### Next Steps After Building

1. **Expand to 8 trits** (5⁸ = 390,625 values)
2. **Add multiplication** (more complex analog circuit)
3. **Implement full ALU** (all arithmetic operations)
4. **Design pentary CPU** (instruction execution)
5. **Progress to IC design** (3T or memristor chip)

---

**Document Complete**

**Estimated Build Time**: 4-8 hours  
**Skill Level**: Intermediate (soldering, basic electronics)  
**Tools Required**: Soldering iron, multimeter, oscilloscope (helpful)

---

*This breadboard computer provides a hands-on introduction to pentary computing principles before progressing to integrated circuit implementations.*
