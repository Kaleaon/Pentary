* =============================================================================
* Pentary Memristor SPICE Model
* =============================================================================
*
* This SPICE model simulates a 5-level (pentary) memristor device suitable
* for in-memory computing applications.
*
* Based on:
*   - Yakopcic memristor model (threshold-based switching)
*   - Stanford memristor model extensions
*   - HfOx material parameters
*
* Pentary Levels:
*   -2 (1MΩ), -1 (316kΩ), 0 (100kΩ), +1 (31.6kΩ), +2 (10kΩ)
*
* =============================================================================

* =============================================================================
* Parameters
* =============================================================================

.param Ron = 10k         ; Minimum resistance (state +2)
.param Roff = 1Meg       ; Maximum resistance (state -2)
.param Rinit = 100k      ; Initial resistance (state 0)
.param Vt_pos = 0.8      ; Positive threshold voltage
.param Vt_neg = -0.8     ; Negative threshold voltage
.param alpha_p = 1       ; SET switching rate
.param alpha_n = 1       ; RESET switching rate
.param kp = 5e-5         ; SET velocity coefficient
.param kn = 5e-5         ; RESET velocity coefficient
.param ap = 0.9          ; SET nonlinearity
.param an = 0.9          ; RESET nonlinearity
.param xp = 0.3          ; SET switching point
.param xn = 0.5          ; RESET switching point

* =============================================================================
* Memristor Subcircuit (Yakopcic Model)
* =============================================================================

.subckt memristor plus minus x_out
+ params: R0=100k

* Internal state variable (normalized 0-1)
* 0 = Roff (high resistance, -2)
* 1 = Ron (low resistance, +2)

* State equation: dx/dt = f(V) * g(x)
* f(V) = velocity function
* g(x) = window function

* Approximation using behavioral sources

* Applied voltage
Esense vsense 0 plus minus 1

* Current (Ohm's law with variable R)
Gmem plus minus value={V(plus,minus)/(Ron + (Roff-Ron)*(1-V(x)))}

* State variable dynamics
* Positive switching (SET: decrease resistance, increase x)
.func fp(v) = {kp * (exp(alpha_p*(v-Vt_pos)) - 1) * (v > Vt_pos ? 1 : 0)}
* Negative switching (RESET: increase resistance, decrease x)
.func fn(v) = {-kn * (exp(-alpha_n*(v-Vt_neg)) - 1) * (v < Vt_neg ? 1 : 0)}

* Window function (prevents x from going outside [0,1])
.func wp(x) = {(xp - x) / (1 - xp) + 1}
.func wn(x) = {x / (1 - xn)}
.func w(x,v) = {(v > 0 ? wp(x)**ap : wn(x)**an) * (x > 0 ? 1 : 0) * (x < 1 ? 1 : 0) + (x <= 0 ? (v > 0 ? 1 : 0) : 0) + (x >= 1 ? (v < 0 ? 1 : 0) : 0)}

* State variable capacitor
Cx x 0 1 IC={1 - (R0 - Ron)/(Roff - Ron)}

* Charging current for state variable
Gx 0 x value={fp(V(vsense))*w(V(x),V(vsense)) + fn(V(vsense))*w(V(x),V(vsense))}

* Output for monitoring
Ex_out x_out 0 x 0 1

.ends memristor


* =============================================================================
* 5-Level Quantized Memristor
* =============================================================================

.subckt pentary_memristor plus minus level_out
+ params: init_level=2

* Map init_level (-2,-1,0,+1,+2) to resistance
* Level -2: R = 1MΩ
* Level -1: R = 316kΩ
* Level  0: R = 100kΩ
* Level +1: R = 31.6kΩ
* Level +2: R = 10kΩ

.param R_init = {(init_level == -2) ? 1Meg :
+                (init_level == -1) ? 316k :
+                (init_level == 0)  ? 100k :
+                (init_level == 1)  ? 31.6k :
+                                     10k}

* Instantiate memristor
Xmem plus minus x_state memristor R0=R_init

* Quantize state to 5 levels
* x = 0.0-0.2 -> Level -2
* x = 0.2-0.4 -> Level -1
* x = 0.4-0.6 -> Level 0
* x = 0.6-0.8 -> Level +1
* x = 0.8-1.0 -> Level +2

Elevel level_out 0 value={
+ (V(x_state) < 0.2) ? -2 :
+ (V(x_state) < 0.4) ? -1 :
+ (V(x_state) < 0.6) ? 0 :
+ (V(x_state) < 0.8) ? 1 : 2}

.ends pentary_memristor


* =============================================================================
* Crossbar Array Cell (1M1R - 1 Memristor 1 Resistor selector)
* =============================================================================

.subckt crossbar_cell row col level_out
+ params: init_level=0

* Series resistor (selector, prevents sneak paths)
Rsel row cell_int 1k

* Pentary memristor
Xmem cell_int col level_out pentary_memristor init_level=init_level

.ends crossbar_cell


* =============================================================================
* 4x4 Crossbar Array Example
* =============================================================================

.subckt crossbar_4x4 row0 row1 row2 row3 col0 col1 col2 col3
+ params: W00=0 W01=0 W02=0 W03=0
+         W10=0 W11=0 W12=0 W13=0
+         W20=0 W21=0 W22=0 W23=0
+         W30=0 W31=0 W32=0 W33=0

* Row 0
Xcell00 row0 col0 lev00 crossbar_cell init_level=W00
Xcell01 row0 col1 lev01 crossbar_cell init_level=W01
Xcell02 row0 col2 lev02 crossbar_cell init_level=W02
Xcell03 row0 col3 lev03 crossbar_cell init_level=W03

* Row 1
Xcell10 row1 col0 lev10 crossbar_cell init_level=W10
Xcell11 row1 col1 lev11 crossbar_cell init_level=W11
Xcell12 row1 col2 lev12 crossbar_cell init_level=W12
Xcell13 row1 col3 lev13 crossbar_cell init_level=W13

* Row 2
Xcell20 row2 col0 lev20 crossbar_cell init_level=W20
Xcell21 row2 col1 lev21 crossbar_cell init_level=W21
Xcell22 row2 col2 lev22 crossbar_cell init_level=W22
Xcell23 row2 col3 lev23 crossbar_cell init_level=W23

* Row 3
Xcell30 row3 col0 lev30 crossbar_cell init_level=W30
Xcell31 row3 col1 lev31 crossbar_cell init_level=W31
Xcell32 row3 col2 lev32 crossbar_cell init_level=W32
Xcell33 row3 col3 lev33 crossbar_cell init_level=W33

.ends crossbar_4x4


* =============================================================================
* Test Bench: Single Memristor Characterization
* =============================================================================

.subckt tb_single_memristor

* Supply
Vdd vdd 0 DC 1.0

* Test voltage source (sweep or pulse)
Vtest in 0 PWL(0 0 10u 1.5 20u 0 30u -1.5 40u 0)

* Memristor under test
Xdut in 0 level pentary_memristor init_level=0

* Current sense
Rsense in in_sense 1
Esense i_sense 0 in in_sense 1

.ends tb_single_memristor


* =============================================================================
* Test Bench: Matrix-Vector Multiplication
* =============================================================================

.subckt tb_mvm

* Weight matrix (pentary values)
* W = [[+2, -1, 0, +1],
*      [-1, +2, +1, 0],
*      [0, +1, +2, -1],
*      [+1, 0, -1, +2]]

* Input voltages (represent pentary input vector)
* V = [+1, 0, -1, +2] -> scaled to voltages

Vin0 in0 0 DC 0.3    ; +1 -> 0.3V
Vin1 in1 0 DC 0.0    ;  0 -> 0.0V
Vin2 in2 0 DC -0.3   ; -1 -> -0.3V
Vin3 in3 0 DC 0.6    ; +2 -> 0.6V

* Crossbar array with weight matrix
Xarray in0 in1 in2 in3 out0 out1 out2 out3 crossbar_4x4
+ W00=2 W01=-1 W02=0 W03=1
+ W10=-1 W11=2 W12=1 W13=0
+ W20=0 W21=1 W22=2 W23=-1
+ W30=1 W31=0 W32=-1 W33=2

* Output load resistors (for current-to-voltage conversion)
Rout0 out0 0 1k
Rout1 out1 0 1k
Rout2 out2 0 1k
Rout3 out3 0 1k

.ends tb_mvm


* =============================================================================
* Simulation Commands
* =============================================================================

* Instantiate single memristor test
Xtb_single tb_single_memristor

* DC sweep
.dc Vtest -1.5 1.5 0.01

* Transient for switching behavior
.tran 0.1u 50u

* Operating point
.op

* Measure resistance at different states
.meas tran R_initial = '1/I(Xdut.Gmem)' at=0
.meas tran R_final = '1/I(Xdut.Gmem)' at=40u

* Print results
.print dc V(in) I(Xdut.Gmem) V(Xtb_single.Xdut.x_state)
.print tran V(in) I(Xdut.Gmem) V(level)

.end
