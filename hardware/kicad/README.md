# Pentary Breadboard Computer - KiCad Project

This directory contains KiCad project files for the Pentary Breadboard Computer PCB.

## Project Files

```
kicad/
├── pentary_breadboard.kicad_pro   # KiCad project file
├── pentary_breadboard.kicad_sch   # Schematic (to be created in KiCad)
├── pentary_breadboard.kicad_pcb   # PCB layout (to be created in KiCad)
├── libraries/                      # Custom symbol and footprint libraries
│   ├── pentary.kicad_sym          # Custom symbols
│   └── pentary.pretty/            # Custom footprints
├── gerbers/                        # Gerber files for manufacturing
├── bom/                            # Bill of Materials
│   └── pentary_bom.csv
└── README.md                       # This file
```

## PCB Specifications

| Parameter | Value |
|-----------|-------|
| Board Size | 100mm × 160mm |
| Layers | 2 (top + bottom copper) |
| Copper Weight | 1 oz |
| Board Thickness | 1.6mm |
| Surface Finish | HASL (lead-free) |
| Min Track Width | 0.25mm |
| Min Spacing | 0.25mm |
| Min Drill | 0.3mm |

## Schematic Hierarchy

### Sheet 1: Power Supply
- USB-C input connector
- LM7805 (5V regulator)
- LM1117-3.3 (3.3V regulator)
- LM317 (1.8V adjustable regulator)
- Reference voltage ladder (0.4V steps)

### Sheet 2: DAC Section (4× channels)
- 74HC4051 analog multiplexer (×4)
- LM358 op-amp buffer (×2)
- Reference voltage distribution

### Sheet 3: ALU Section
- LM358 summing amplifier
- LM339 comparator bank
- Quantizer logic

### Sheet 4: Register Section
- CD4066 analog switches (×4)
- TL072 JFET op-amps (×2)
- Sample-and-hold capacitors

### Sheet 5: ADC Section
- LM339 comparator bank
- 74HC148 priority encoder
- Output buffers

### Sheet 6: Display Section
- 20× LEDs (4× 5-level indicators)
- 4× 7-segment displays
- 74HC595 shift registers

### Sheet 7: Interface Section
- Arduino header (2×10)
- Manual switches (8×)
- Test points

## Component List

### ICs
| Reference | Qty | Part Number | Description | Package |
|-----------|-----|-------------|-------------|---------|
| U1 | 1 | LM7805CT | 5V regulator | TO-220 |
| U2 | 1 | LM1117-3.3 | 3.3V regulator | SOT-223 |
| U3 | 1 | LM317T | Adj. regulator | TO-220 |
| U4-U7 | 4 | 74HC4051 | 8:1 analog mux | DIP-16 |
| U8-U9 | 2 | LM358N | Dual op-amp | DIP-8 |
| U10-U11 | 2 | TL072CP | JFET op-amp | DIP-8 |
| U12 | 1 | LM339N | Quad comparator | DIP-14 |
| U13-U16 | 4 | CD4066BE | Quad analog switch | DIP-14 |
| U17 | 1 | 74HC148 | Priority encoder | DIP-16 |
| U18-U21 | 4 | 74HC595 | Shift register | DIP-16 |

### Connectors
| Reference | Qty | Description |
|-----------|-----|-------------|
| J1 | 1 | USB-C power connector |
| J2 | 1 | 2×10 pin header (Arduino) |
| J3-J6 | 4 | 1×3 pin header (test points) |

### Passives
| Reference | Qty | Value | Description |
|-----------|-----|-------|-------------|
| R1-R5 | 5 | 1kΩ | Voltage divider |
| R6-R10 | 5 | 10kΩ | Pull-up/down |
| R11-R30 | 20 | Various | LED current limit |
| C1-C5 | 5 | 100nF | Decoupling |
| C6-C10 | 5 | 100µF | Bulk capacitor |
| C11-C14 | 4 | 100nF | S/H capacitor |

### Display
| Reference | Qty | Description |
|-----------|-----|-------------|
| LED1-LED20 | 20 | 5mm LED (mixed colors) |
| DIS1-DIS4 | 4 | 7-segment, common cathode |

### Switches
| Reference | Qty | Description |
|-----------|-----|-------------|
| SW1 | 1 | Reset button |
| SW2-SW9 | 8 | Tactile switch |

## KiCad Setup Instructions

### 1. Install KiCad
Download from https://www.kicad.org/download/

### 2. Open Project
```
File -> Open Project -> pentary_breadboard.kicad_pro
```

### 3. Create Schematic
- Open Schematic Editor
- Add symbols from library
- Wire components according to design

### 4. Annotate Schematic
```
Tools -> Annotate Schematic
```

### 5. Run ERC
```
Inspect -> Electrical Rules Checker
```

### 6. Assign Footprints
```
Tools -> Assign Footprints
```

### 7. Generate Netlist
```
Tools -> Generate Netlist
```

### 8. Create PCB Layout
- Open PCB Editor
- Import netlist
- Place components
- Route traces
- Add copper pours

### 9. Run DRC
```
Inspect -> Design Rules Checker
```

### 10. Generate Gerbers
```
File -> Fabrication Outputs -> Gerbers
```

## Manufacturing

### Recommended Fabricators
- **JLCPCB** (China) - $2 for 5 boards
- **PCBWay** (China) - $5 for 10 boards
- **OSH Park** (USA) - $5/sq inch, purple boards
- **Elecrow** (China) - $4.90 for 5 boards

### Gerber Files Required
- Top Copper (F.Cu)
- Bottom Copper (B.Cu)
- Top Silkscreen (F.SilkS)
- Bottom Silkscreen (B.SilkS)
- Top Solder Mask (F.Mask)
- Bottom Solder Mask (B.Mask)
- Edge Cuts (outline)
- Drill file (PTH + NPTH)

### Design Rules for JLCPCB
```
Min Track Width: 0.127mm (5 mil)
Min Spacing: 0.127mm (5 mil)
Min Via Drill: 0.3mm
Min Via Diameter: 0.5mm
Min Hole Size: 0.3mm
Min Annular Ring: 0.13mm
```

## Assembly Notes

### Soldering Order
1. **SMD components** (if any)
2. **Low-profile** - resistors, diodes
3. **IC sockets** - DIP sockets
4. **Capacitors** - ceramic then electrolytic
5. **Connectors** - headers
6. **Tall components** - electrolytic caps, voltage regulators
7. **Through-hole** - switches, LEDs
8. **ICs** - insert into sockets last

### Testing Sequence
1. **Power supply** - verify voltages before inserting ICs
2. **Reference voltages** - check 0.4V, 0.8V, 1.2V, 1.6V
3. **Insert ICs** - one section at a time
4. **DAC** - verify analog outputs
5. **ADC** - verify digital outputs
6. **ALU** - test arithmetic operations
7. **Full system** - test with Arduino

## Schematic Symbol Reference

### Pentary Signal Naming
```
PENT_A[11:0]  - Pentary input A (4 trits × 3 bits)
PENT_B[11:0]  - Pentary input B
PENT_OUT[11:0] - Pentary output
VREF[4:0]     - Reference voltages (0.0V to 1.6V)
```

### Power Rails
```
+5V    - Digital logic power
+3V3   - MCU power
+1V8   - Analog reference maximum
GND    - Ground reference
```

## Revision History

| Rev | Date | Description |
|-----|------|-------------|
| 1.0 | 2026 | Initial design |

## License

This design is released under the CERN Open Hardware License v2 (CERN-OHL-W).
