
# High-Level Project Plan: FPGA Prototyping & ASIC Fabrication

**Version**: 1.0  
**Date**: January 3, 2026  
**Author**: Manus AI

## 1. Introduction

This document outlines a high-level project plan and timeline for the next two critical phases of the Pentary project: **FPGA Prototyping** and **ASIC Fabrication**. By focusing on a standard CMOS implementation and replacing experimental components with digital systolic arrays (Pentary Tensor Cores), this plan represents a lower-risk, more achievable path to silicon.

---

## 2. Phase 1: FPGA Prototyping

**Objective**: To functionally validate the complete Pentary processor design on real hardware, benchmark its performance on target workloads, and refine the architecture before committing to the expensive and irreversible ASIC fabrication process.

**Estimated Duration**: 6 Months

### 2.1. Key Activities & Timeline

| Month | Key Activities                                                                                                                              | Deliverables                                                                                                   |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **1**   | **FPGA Platform Selection & Setup**<br>- Select a high-end FPGA board (e.g., Xilinx Versal or Intel Stratix) with sufficient logic, BRAM, and PCIe Gen4/5 support.<br>- Procure development boards and set up the design environment.                               | - FPGA hardware selection report.<br>- Fully configured development environment.                                |
| **2-3** | **RTL Adaptation & Integration**<br>- Adapt the Verilog RTL for the FPGA target. This includes replacing the 3T dynamic trit cells with FPGA block RAMs (BRAMs) and instantiating FPGA-specific primitives for components like PLLs and PCIe controllers.<br>- Integrate the Pentary core with a host interface. | - FPGA-synthesizable Verilog codebase.<br>- Host communication drivers.                                                |
| **4**   | **Synthesis, Place & Route (P&R)**<br>- Run the adapted RTL through the FPGA synthesis and P&R tools.<br>- Perform timing analysis and iterate on the design to meet the target clock frequency on the FPGA.                                           | - Final FPGA bitstream.<br>- Timing closure and resource utilization reports.                               |
| **5**   | **Hardware Bring-up & Validation**<br>- Program the FPGA and perform initial hardware bring-up.<br>- Run a comprehensive suite of functional tests and diagnostics to validate all aspects of the processor architecture, from the ALU to the cache hierarchy. | - Functional validation report.<br>- List of identified and fixed bugs.                                           |
| **6**   | **Benchmarking & Final Report**<br>- Run key neural network kernels (e.g., matrix multiplication, convolution) on the FPGA prototype to gather real-world performance and power data.<br>- Analyze the results and prepare a final report with recommendations for the ASIC design. | - Performance and power benchmark results.<br>- Final FPGA prototyping report with go/no-go recommendation for ASIC tape-out. |

---

## 3. Phase 2: ASIC Fabrication (TSMC 7nm)

**Objective**: To take the validated and refined RTL design through the complete ASIC design flow, resulting in the production of the Pentary Titans PT-2000 chip.

**Estimated Duration**: 18 Months

### 3.1. Key Activities & Timeline

| Quarter | Key Activities                                                                                                                                                                                                                                                                                                                            | Major Milestones                                                                                             |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Q1-Q2** | **Logical Design & Synthesis**<br>- Finalize the RTL based on feedback from the FPGA phase.<br>- Perform full-chip logic synthesis using a TSMC 7nm standard cell library.<br>- Insert Design-for-Test (DFT) structures, including scan chains and Memory BIST (Built-In Self-Test).<br>- Generate and verify the final gate-level netlist. | - **RTL Freeze**: The Verilog code is finalized and placed under strict version control.<br>- **Synthesis Complete**: A fully synthesized, gate-level netlist is ready for physical design. |
| **Q3-Q4** | **Physical Design**<br>- **Floorplanning**: Define the overall chip layout, including the placement of major blocks like cores, caches, and I/O pads.<br>- **Place & Route (P&R)**: Automatically place all standard cells and route the connections between them.<br>- **Clock Tree Synthesis (CTS)**: Build the clock distribution network to ensure low skew and jitter.<br>- **Static Timing Analysis (STA)**: Perform rigorous timing analysis and iterate on the design to close timing on all paths. | - **Floorplan Lock**: The high-level physical layout of the chip is approved.<br>- **P&R Complete**: The design is fully routed and meets initial timing targets.                               |
| **Q5**    | **Physical Verification & Tape-Out**<br>- Run final physical verification checks, including Design Rule Checking (DRC) and Layout vs. Schematic (LVS), to ensure manufacturability.<br>- Generate the final GDSII/OASIS layout files.<br>- **Tape-Out**: Deliver the final design files to TSMC for mask generation and fabrication. | - **Physical Verification Sign-off**: The design is certified as DRC and LVS clean.<br>- **Tape-Out**: The design is officially sent for manufacturing. This is a major point of no return. |
| **Q6**    | **Fabrication & Packaging**<br>- TSMC fabricates the wafers using their 7nm process (approx. 12-16 weeks).<br>- Wafers are tested, diced, and packaged into the final BGA package.<br>- Packaged chips undergo final testing.                                                                                                                                                                                                                                                        | - **Wafers Out**: The first wafers are completed by the foundry.<br>- **Packaged Samples Ready**: The first engineering samples are packaged and ready for testing. |
| **Q7**    | **Post-Silicon Validation (Bring-Up)**<br>- Develop a custom bring-up board for the new ASIC.<br>- Test the first silicon samples, starting with basic power-on checks and gradually moving to full functional validation.<br>- Characterize the chip's performance, power, and yield across different process, voltage, and temperature (PVT) corners. | - **First Silicon Powered On**: The first chip is successfully powered on in the lab.<br>- **Full Functional Validation**: The chip is confirmed to be fully functional and bug-free.               |
| **Q8**    | **Production Ramp & Launch**<br>- Based on the validation results, begin mass production.<br>- Finalize the Pentary Titans PCIe card design and prepare for market launch.                                                                                                                                                                                                                                                        | - **Production Release**: The chip is approved for mass production.<br>- **Product Launch**: The Pentary Titans AI Accelerator is officially launched.                               |

---

## 4. Conclusion

This project plan provides a realistic timeline for bringing the Pentary architecture to market. The pivot to a fully standard-CMOS design significantly de-risks the project, removing dependencies on novel materials and complex analog-in-memory compute schemes. The 6-month FPGA phase remains a critical validation step, but the subsequent 18-month ASIC process is now more straightforward, relying on well-understood digital design flows. This makes the overall project more predictable and increases the likelihood of first-pass silicon success.
