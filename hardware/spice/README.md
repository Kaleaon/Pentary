# Pentary Memristor SPICE Models

This directory contains SPICE models for simulating pentary memristor devices and crossbar arrays.

## Files

```
spice/
├── pentary_memristor.sp    # Main memristor model and test benches
├── pentary_crossbar.sp     # Crossbar array models
├── analog_circuits.sp      # Supporting analog circuits (DAC, ADC, op-amps)
└── README.md               # This file
```

## Model Overview

### Pentary Memristor Model

The memristor model implements a 5-level (pentary) resistive memory device based on the Yakopcic threshold-based switching model.

**Resistance States:**

| Pentary Level | Resistance | State Variable (x) |
|---------------|------------|-------------------|
| -2            | 1 MΩ       | 0.0 - 0.2         |
| -1            | 316 kΩ     | 0.2 - 0.4         |
|  0            | 100 kΩ     | 0.4 - 0.6         |
| +1            | 31.6 kΩ    | 0.6 - 0.8         |
| +2            | 10 kΩ      | 0.8 - 1.0         |

**Switching Characteristics:**

- Threshold voltage (SET): +0.8V
- Threshold voltage (RESET): -0.8V
- Read voltage: ±0.3V (non-destructive)
- Programming pulse: 100ns typical

### Crossbar Array

The crossbar model implements a passive crossbar array with:
- 1M1R (1 Memristor 1 Resistor) cell architecture
- Series selector resistor to mitigate sneak paths
- Configurable initial weight matrix

## SPICE Simulators

These models are compatible with:

| Simulator | Notes |
|-----------|-------|
| ngspice   | Recommended, open-source |
| LTspice   | Free, Windows/Mac |
| HSPICE    | Commercial, industry standard |
| Spectre   | Commercial, Cadence |
| Xyce      | Open-source, Sandia |

## Usage

### ngspice

```bash
# Interactive mode
ngspice pentary_memristor.sp

# Batch mode
ngspice -b pentary_memristor.sp -o results.log

# With plotting
ngspice
> source pentary_memristor.sp
> run
> plot V(in) vs I(Xdut.Gmem)
```

### LTspice

1. Open LTspice
2. File → Open → pentary_memristor.sp
3. Run simulation
4. View waveforms

## Simulation Examples

### 1. Single Device Characterization

Test the I-V characteristics of a single memristor:

```spice
* Sweep voltage and measure current
.dc Vtest -1.5 1.5 0.01
.print dc V(in) I(Xdut.Gmem)
```

Expected output: Pinched hysteresis loop showing multi-level resistance states.

### 2. Switching Dynamics

Program the memristor through all 5 states:

```spice
* Pulse sequence to program different levels
Vtest in 0 PWL(
+ 0     0
+ 10u   1.5     ; SET to +2
+ 20u   0
+ 30u  -1.5     ; RESET to -2
+ 40u   0
+ 50u   1.0     ; Partial SET to 0
+ 60u   0)

.tran 0.1u 60u
.print tran V(level)
```

### 3. Matrix-Vector Multiplication

Perform analog MVM using crossbar array:

```spice
* Input vector as voltages
Vin0 in0 0 DC 0.3    ; +1
Vin1 in1 0 DC 0.0    ;  0
Vin2 in2 0 DC -0.3   ; -1
Vin3 in3 0 DC 0.6    ; +2

* Weight matrix encoded in memristors
* Output currents represent Y = W × X

.op
.print dc V(out0) V(out1) V(out2) V(out3)
```

## Model Parameters

### Memristor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Ron       | 10kΩ    | Minimum resistance |
| Roff      | 1MΩ     | Maximum resistance |
| Rinit     | 100kΩ   | Initial resistance |
| Vt_pos    | 0.8V    | Positive threshold |
| Vt_neg    | -0.8V   | Negative threshold |
| alpha_p   | 1       | SET rate parameter |
| alpha_n   | 1       | RESET rate parameter |
| kp        | 5e-5    | SET velocity |
| kn        | 5e-5    | RESET velocity |
| ap        | 0.9     | SET nonlinearity |
| an        | 0.9     | RESET nonlinearity |

### Modifying Parameters

```spice
* Change to different HfOx variant
.param Ron = 5k
.param Roff = 2Meg
.param Vt_pos = 0.6

* Instantiate with custom parameters
Xmem plus minus level pentary_memristor init_level=0
```

## Validation

### Experimental Data Comparison

The model has been tuned to match experimental data from:
- Yu J, et al. "Multi-Level HfOx Memristors" (2024)
- Chen W-H, et al. "CMOS-Integrated Memristors" (2019)

### Key Metrics

| Metric | Model | Experimental |
|--------|-------|--------------|
| On/Off Ratio | 100:1 | 50-200:1 |
| SET Time | 100ns | 50-500ns |
| RESET Time | 100ns | 50-500ns |
| Endurance | N/A | >10^6 cycles |
| Retention | N/A | >10 years |

## Limitations

1. **Static model**: Does not capture cycle-to-cycle variability
2. **No noise**: Deterministic behavior (real devices have noise)
3. **Temperature**: Fixed at 27°C (no temperature dependence)
4. **Aging**: No degradation over time

## Extending the Model

### Add Variability

```spice
* Add Gaussian noise to resistance
.param Ron_var = {Ron * (1 + 0.1*gauss())}
```

### Temperature Dependence

```spice
* Temperature coefficient
.param TC1 = 0.001
.param R_T = {R0 * (1 + TC1*(TEMP-27))}
```

### Device-to-Device Variation

```spice
* Different devices with variation
.param Ron1 = {Ron * (1 + 0.05*rand())}
.param Ron2 = {Ron * (1 + 0.05*rand())}
```

## References

1. Yakopcic, C. et al. "A memristor device model." IEEE Electron Device Letters, 2011.
2. Chen, W-H. et al. "A 65nm 1Mb nonvolatile computing-in-memory macro." Nature Electronics, 2019.
3. Yu, J. et al. "3D HfO2-based memristor arrays." Advanced Electronic Materials, 2024.

## License

These models are released under the Apache 2.0 License.
