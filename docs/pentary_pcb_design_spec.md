
# Pentary Titans PCIe Expansion Card: PCB Design Specification

**Version**: 1.0  
**Date**: January 3, 2026  
**Author**: Manus AI

## 1. Overview

This document provides the detailed Printed Circuit Board (PCB) design specifications for the Pentary Titans AI Accelerator (Model: PT-2000), a full-height, full-length (FHFL) PCIe Gen5 x16 expansion card.

---

## 2. Mechanical Design

### 2.1. Board Dimensions

- **Form Factor**: PCIe Full-Height, Full-Length
- **Length**: 312 mm
- **Height**: 111 mm
- **Width**: 50 mm (Dual-Slot)

### 2.2. PCB Specifications

| Parameter         | Specification                |
|-------------------|------------------------------|
| **Layers**        | 10                           |
| **Material**      | FR-4 High-TG (Tg ≥ 180°C)      |
| **Thickness**     | 2.0 mm                       |
| **Copper Weight** | 2 oz (inner), 3 oz (outer)   |
| **Impedance**     | ±10% controlled impedance    |
| **Via Types**     | Blind, Buried, and Through-hole |

### 2.3. Layer Stack-up

The 10-layer stack-up is designed to support high-speed signals and a complex power delivery network.

| Layer | Type          | Description                                  |
|-------|---------------|----------------------------------------------|
| 1     | Signal (Top)  | Component placement and low-speed signals    |
| 2     | Ground Plane  | Solid ground plane for return paths and shielding |
| 3     | Signal        | High-speed differential pairs (PCIe, HBM)    |
| 4     | Power Plane   | V_CORE (0.9V)                                |
| 5     | Signal        | General routing                              |
| 6     | Signal        | General routing                              |
| 7     | Power Plane   | V_HBM (1.2V), V_MEM (3.3V), V_PCIE (3.3V)      |
| 8     | Signal        | High-speed differential pairs (PCIe, HBM)    |
| 9     | Ground Plane  | Solid ground plane for return paths and shielding |
| 10    | Signal (Bottom)| Low-speed signals and component placement    |

### 2.4. Connector Specifications

- **Primary Interface**: PCIe Gen5 x16 edge connector (164 pins, 1.0mm pitch, 30μ-inch gold plating).
- **Auxiliary Power**: 12VHPWR 16-pin connector (12 power, 4 sense pins) rated for up to 600W, utilizing a Molex Micro-Fit 3.0 or equivalent.

---

## 3. Electrical Design

### 3.1. Power Distribution Network (PDN)

The PDN is designed to deliver stable power to all components with minimal noise and voltage drop.

**Power Rails:**

| Rail    | Voltage | Max Current | Max Power | Tolerance | Max Ripple |
|---------|---------|-------------|-----------|-----------|------------|
| V_CORE  | 0.9V    | 150A        | 135W      | ±2%       | <10mV      |
| V_HBM   | 1.2V    | 15A         | 18W       | ±3%       | <20mV      |
| V_SRAM  | 1.1V    | 18A         | 20W       | ±3%       | <25mV      |
| V_PCIE  | 3.3V    | 3A          | 10W       | ±5%       | <50mV      |

**VRM Design:**

- **Primary Stage**: A high-efficiency buck converter steps down the 12V input from the 12VHPWR connector.
- **Core VRM**: An 8-phase digital VRM provides the 0.9V V_CORE rail for the Pentary ASIC, utilizing DrMOS power stages for high efficiency and thermal performance.
- **Secondary Converters**: Separate buck converters generate the 1.2V, 3.3V, and other necessary voltage rails.

### 3.2. Power Sequencing

A microcontroller-based power sequencer ensures that all components are powered up and down in the correct order to prevent damage.

**Power-Up Sequence:**
1. 12V input detected.
2. Enable 3.3V rails (V_PCIE, V_MEM).
3. Wait 10ms for stabilization.
4. Enable 1.2V rail (V_HBM).
5. Wait 5ms for stabilization.
6. Enable 0.9V rail (V_CORE).
7. Wait 5ms for stabilization.
8. Release system reset.

**Power-Down Sequence:**
1. Assert system reset.
2. Disable 0.9V rail.
3. Wait 5ms.
4. Disable 1.2V rail.
5. Wait 5ms.
6. Disable 3.3V rails.

### 3.3. Signal Integrity

Strict design rules are enforced for all high-speed interfaces.

**PCIe Gen5 Interface:**
- **Impedance**: 85Ω differential.
- **Trace Length Matching**: ±0.5mm for all differential pairs.
- **Max Vias**: No more than 4 vias per trace.
- **Loss Budget**: Total insertion loss must be less than 30 dB at 16 GHz.

**HBM3 Interface:**
- Routing is kept on dedicated high-speed signal layers with solid ground plane references.
- Trace lengths are minimized and matched to ensure timing margins are met.

---

## 4. Thermal Design

### 4.1. Thermal Budget

The total thermal design power (TDP) of the card is approximately 196W.

| Component        | Power (W) | Area (mm²) | Power Density (W/mm²) |
|------------------|-----------|------------|-------------------------|
| Pentary ASIC     | 135       | 450        | 0.30                    |
| HBM3 (4 stacks)  | 18        | 200        | 0.09                    |
| On-Chip SRAM     | 20        | 150        | 0.13                    |
| **Total**        | **173**   | **800**    | **0.22**                |

### 4.2. Cooling Solution

A dual-slot active cooling solution is employed to manage the thermal load.

- **Primary Heat Spreader**: A large copper vapor chamber makes direct contact with the Pentary ASIC and the HBM3 stacks via a high-performance thermal interface material (TIM). The on-chip SRAM is cooled as part of the ASIC die.
- **Heat Sink**: An aluminum fin stack is bonded to the vapor chamber to maximize the surface area for heat dissipation.
- **Airflow**: Two 80mm dual ball-bearing fans provide forced convection, pulling cool air from outside the chassis and exhausting it through the rear bracket.
- **Thermal Performance Target**: Maintain a maximum junction temperature of 85°C for the Pentary ASIC under full load in a 50°C ambient environment.

---

## 5. Component Layout

The component placement is optimized for signal integrity and thermal performance.

- The **Pentary ASIC** is placed centrally on the board.
- The four **HBM3 memory stacks** are placed in a quad-formation around the ASIC to minimize trace lengths.
- The **Pentary Tensor Cores (PTCs)** and their associated SRAM are integrated directly into the Pentary ASIC, so there are no separate memory modules for the accelerator on the PCB.
- The **power delivery circuitry** is located near the 12VHPWR connector to minimize power transmission losses.

---

## 6. Manufacturing & Assembly

- **PCB Fabrication**: The PCB shall be fabricated by a qualified manufacturer with experience in high-layer-count, impedance-controlled boards.
- **Assembly**: Automated pick-and-place and reflow soldering processes are to be used. The BGA package of the Pentary ASIC requires X-ray inspection to ensure solder joint integrity.
- **Testing**: All boards must undergo automated optical inspection (AOI), in-circuit testing (ICT), and a full functional test.
