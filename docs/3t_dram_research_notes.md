# 3T Gain Cell DRAM Research Notes

**Source:** IEEE Journal of Solid-State Circuits, Vol. 46, No. 6, June 2011
**Authors:** Ki Chul Chun, Pulkit Jain, Jung Hwa Lee, and Chris H. Kim
**Title:** A 3T Gain Cell Embedded DRAM Utilizing Preferential Boosting for High Density and Low Power On-Die Caches

---

## Key Findings from Academic Research

### 3T Gain Cell Architecture

The research describes a **three-transistor (3T) gain cell** for embedded DRAM with the following structure:

**Transistor Configuration:**

1. **PW (Pass Write) - NMOS transistor**
   - Controlled by Write Word Line (WWL)
   - Connects Write Bit Line (WBL) to storage node
   - Function: Write access device

2. **PS (Pass Storage) - NMOS transistor**
   - Connected to storage node
   - Function: Storage device (gate capacitance stores charge)
   - The storage node voltage is held on the gate of this transistor

3. **PR (Pass Read) - PMOS transistor**
   - Controlled by Read Bit Line (RBL)
   - Connected to Read Word Line (RWL)
   - Function: Read access device

### Operating Modes

**Signal Conditions for Each Mode:**

| Mode | RWL | RBL | WWL | WBL |
|------|-----|-----|-----|-----|
| Data hold | VDD | 0V | VDD | 0V |
| Read 0/1 | ~0V | ~VDD | VDD | 0V |
| Write 0/1 | VDD | VDD | -500mV | 0V/VDD |

### Key Innovations

**1. Preferential Boosting**
- The storage node voltage is preferentially boosted using RWL signal coupling
- When RWL switches from high to low, capacitive coupling amplifies the storage node voltage
- Data '1' voltage coupled up by 0.16V
- Data '0' voltage coupled down by 0.3V
- This amplification extends data retention time significantly

**2. Hybrid Current/Voltage Sense Amplifier**
- Keeps RBL level close to VDD during read operation
- Uses cross-coupled PMOS and NMOS pairs
- Amplifies differential current to voltage signals
- Enables fast, low-power read operations

**3. Regulated Bit-Line Write Scheme**
- Uses steady-state storage node voltage monitor
- Eliminates data '1' disturbance problem
- Pre-charges WBL to VWR (regulated voltage)
- Prevents signal loss in unselected cells

**4. PVT-Tracking Read Reference Bias**
- Adjustable VDUM (reference voltage) generator
- Tracks process-voltage-temperature variations
- Uses replica cells to match actual cell characteristics
- Adaptively adjusts bias for optimal performance

### Performance Characteristics

**Measured Results (65nm CMOS, 0.9V, 85°C):**
- Data retention time: **1.25 ms** (without refresh)
- Cell size: Approximately **2× smaller than 6T SRAM**
- Static power: **91.3 μW/Mb** at 1.0V, 85°C
- Access time: Faster than conventional 3T cells due to boosting
- Array size tested: 32 kb

**Retention Time Variation:**
- Fast corner cells: 58 μs
- Typical cells: ~345 μs
- The cell-to-cell variation is significant (1 kb array simulation)
- Preferential boosting helps extend retention for both data '0' and '1'

### Advantages Over Other Memory Types

**vs. 1T1C DRAM:**
- Logic-compatible (no capacitor process needed)
- Faster access time
- Can be integrated on-die with logic

**vs. 6T SRAM:**
- ~2× higher bit cell density
- Lower leakage current in sleep mode
- Lower static power dissipation

**vs. Conventional 3T Gain Cell:**
- Longer retention time (preferential boosting)
- Better read performance (hybrid sense amplifier)
- More robust to PVT variations

### Design Challenges Addressed

1. **Short retention time** - Solved by preferential RWL coupling
2. **Read speed degradation** - Solved by hybrid current/voltage sense amplifier
3. **Write disturbance** - Solved by regulated bit-line write scheme
4. **PVT variation sensitivity** - Solved by adaptive VDUM tracking

### Array Architecture

**32 kb Test Array Structure:**
- 128 cells per word line
- 128 cells per bit line
- Common BL sense amplifier at array center
- VWR bias connected to write-back circuitry
- RWL pull-down keepers at top of array
- Dummy cells at array edges

**Operating Sequence:**
1. RWL selected, preferential coupling amplifies node voltage
2. Current sense amplifier (SA) enabled
3. SA amplifies signals to analog voltage with RBL near VDD
4. Voltage SA control signal (SAEN) enabled
5. Read-out and write-back operations complete

---

## Relevance to Pentary Project

This research validates the feasibility of using **3T dynamic cells** for compact, logic-compatible memory. Key insights:

1. **Proven Technology**: 3T cells have been successfully fabricated and tested in 65nm CMOS
2. **Capacitive Storage**: Gate capacitance can reliably store analog voltage levels
3. **Voltage Amplification**: Capacitive coupling can boost stored voltages (relevant for pentary level discrimination)
4. **Retention Time**: ~1ms retention is achievable, requiring periodic refresh
5. **Sense Amplifiers**: Hybrid current/voltage sensing enables reliable level detection
6. **PVT Robustness**: Adaptive biasing and replica cells can compensate for variations

### Differences for Pentary Implementation

**Standard 3T DRAM (binary):**
- 2 voltage levels (0V and VDD)
- Simple threshold detection
- Single reference voltage

**Pentary 3T Cell (5-level):**
- 5 voltage levels (-2V, -1V, 0V, +1V, +2V)
- Multi-level analog storage
- 4 reference voltages needed
- More complex sense amplifiers
- Tighter noise margins (±0.4V vs ±0.5V in binary)
- May require more sophisticated refresh schemes

---

## Technical Specifications from Paper

**Process Technology:** 65nm low-leakage CMOS
**Operating Voltage:** 0.9V (nominal 1.2V process)
**Temperature:** 85°C
**Cell Area:** ~2× smaller than 6T SRAM
**Retention Time:** 1.25ms at 0.9V, 85°C
**Random Cycle Time:** 2.0ns
**Power Dissipation:** 91.3 μW/Mb static power

**Voltage Levels:**
- VDD: 0.9V (test) / 1.0V (typical)
- WWL under-drive: -500mV
- VDUM range: 0.15V to 0.3V (varies with PVT)

---

## Conclusions

The 3T gain cell represents a **proven, manufacturable approach** to compact dynamic memory using only standard CMOS transistors. The preferential boosting technique and sophisticated peripheral circuits enable reliable operation despite the challenges of dynamic storage. 

For the Pentary project, this research demonstrates that:
- Multi-level analog storage on gate capacitance is feasible
- Careful circuit design can overcome retention and sensing challenges
- Standard CMOS processes can support advanced memory architectures
- PVT variation can be managed with adaptive biasing

The main additional challenge for pentary is extending from 2 levels to 5 levels, which requires:
- Tighter voltage control
- More sophisticated multi-level sense amplifiers
- Dual-rail power supplies (±2.5V instead of 0-1V)
- More complex reference voltage generation
