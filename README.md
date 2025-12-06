# Pentary
5 point LLM quantization
THE PENTARY MANIFESTO

Breaking the Binary Stranglehold on Artificial IntelligenceVersion 1.0 | Status: Open Hardware Initiative

## 🚀 Quick Start

### New to Pentary?

1. **Start Here**: [GETTING_STARTED.md](GETTING_STARTED.md) - Complete getting started guide
2. **Beginner Tutorial**: [BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md) - Learn the basics
3. **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet

### Setup

```bash
# Run setup script (checks dependencies, installs packages)
./setup.sh

# Or manually install
pip install numpy
```

### Try It Out

```bash
# Interactive CLI (recommended for beginners)
python3 tools/pentary_cli.py

# Test basic tools
python3 tools/pentary_converter.py
python3 tools/pentary_arithmetic.py
python3 tools/pentary_simulator.py

# Generate examples
python3 tools/example_generator.py

# Neural network tools
python3 tools/pentary_nn.py
python3 tools/pentary_quantizer.py
python3 tools/pentary_iteration.py
```

## 📦 Tools Overview

### Core Tools
- **pentary_converter.py**: Decimal ↔ Pentary conversion, arithmetic operations
- **pentary_arithmetic.py**: Low-level digit operations, carry handling
- **pentary_simulator.py**: Full processor simulator with ISA support

### Neural Network Tools
- **pentary_nn.py**: Neural network layers (Linear, Conv2D, ReLU, Pooling)
- **pentary_quantizer.py**: Model quantization, calibration, accuracy analysis
- **pentary_iteration.py**: Profiling, optimization, benchmarking, rapid iteration

### Hardware Design
- **pentary_chip_design.v**: Verilog implementation of key components
- **pentary_chip_synthesis.tcl**: Synthesis scripts for ASIC design
- **pentary_chip_layout.md**: Layout guidelines and floorplan

## 📚 Documentation

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Start here! Complete guide
- **[BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md)** - Step-by-step tutorial
- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick start
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet

### Visual Documentation
- **[VISUAL_INDEX.md](VISUAL_INDEX.md)** - 🎨 **NEW!** Complete diagram gallery with 16 high-quality visualizations
- **[ENHANCEMENT_SUMMARY.md](ENHANCEMENT_SUMMARY.md)** - 📊 **NEW!** Summary of all enhancements

### Complete Guides
- [Complete Guide](PENTARY_COMPLETE_GUIDE.md) - Comprehensive reference
- [Project Summary](PROJECT_SUMMARY.md) - Project overview

### Architecture & Design
- [Processor Architecture](architecture/pentary_processor_architecture.md) - ISA specification
- [Chip Design Explained](hardware/CHIP_DESIGN_EXPLAINED.md) - Complete chip explanation
- [Chip Layout Guidelines](hardware/pentary_chip_layout.md) - Physical design
- [Memristor Implementation](hardware/memristor_implementation.md) - Hardware details
- [Verilog Design](hardware/pentary_chip_design.v) - RTL implementation

### Research (Expanded)
- [SOTA AI Systems Comparison](research/pentary_sota_comparison.md) - 🚀 **NEW!** Pentary vs Gemini 3, GPT-5.1, H200/B200 (12,000 words)
- [SOTA Comparison Summary](research/SOTA_COMPARISON_SUMMARY.md) - 📊 **NEW!** Executive summary with benchmarks
- [AI Architectures Analysis](research/pentary_ai_architectures_analysis.md) - 🤖 Comprehensive AI implementation guide (15,000 words)
- [AI Architectures Summary](research/AI_ARCHITECTURES_SUMMARY.md) - 📊 Executive summary of AI findings
- [Foundations - Expanded](research/pentary_foundations_expanded.md) - 📖 Deep dive with proofs
- [Logic Gates - Expanded](research/pentary_logic_gates_expanded.md) - 📖 Complete gate library
- [Memristor Drift Analysis](research/memristor_drift_analysis.md) - 🔬 Feature vs. flaw analysis

### Tools & Language
- [Tools README](tools/README.md) - All available tools
- [Language README](language/README.md) - Pent programming language

1. The Crisis of ComputeWe have hit a wall. The current trajectory of Artificial Intelligence is unsustainable.To build "Intelligence," we are currently relying on Binary Logic (Base-2)—a system designed in the 1940s for vacuum tubes, not for neural networks. We are forcing silicon to perform billions of wasteful floating-point multiplications (FP16) just to approximate what a biological neuron does with a simple pulse of voltage.The result is an Energy Crisis and a Privacy Crisis.Training a single model consumes a city's worth of electricity.Running a "Smart" model requires a $30,000 GPU cluster.True AI is becoming the exclusive domain of trillion-dollar monopolies, locked behind cloud APIs, renting our own intelligence back to us.We believe the future of AI is not bigger data centers. It is smarter physics.
2.
3. 2. The Solution: Balanced Quinary ArchitectureWe propose a fundamental shift from Binary Logic to Balanced Pentary (Base-5) Logic.Nature does not think in "On/Off." Neurons function via Excitation and Inhibition. Our hardware must reflect this. By adopting a Signed-Digit System {-2, -1, 0, +1, +2}, we align the silicon with the math of the neural network.The Three Laws of Pentary Logic:Zero is Absolute: In our architecture, the "0" state is a physical disconnect. It consumes zero power. This unlocks massive sparsity natively.Symmetry is Efficiency: Positive and Negative weights are symmetric voltages (+V and -V). Subtraction is simply addition.Multiplication is Obsolete: By locking weights to integers {-2...2}, we replace massive Floating Point Units (3,000 transistors) with simple Shift-Add Circuits (150 transistors).Result: A chip that is 20x smaller, 7x more memory-dense, and 10x more energy-efficient than the industry standard.
   3.
   4. 3. The Hardware Ecosystem
      4. We are not just building a chip; we are building a distributed nervous system for the next generation of machines.I. The Pentary Deck (Personal Inference)A smartphone-sized card capable of running 24-Billion Parameter models locally.Power: < 25 Watts.Latency: Near-zero (In-Memory Compute).Goal: To put a GPT-4 class mind in the pocket of every doctor, engineer, and student, completely offline and uncensored.II. The Monolith (The Enterprise Vault)A stack of 50 Pentary Decks submerged in dielectric fluid, fitting inside a standard file cabinet.Capacity: 1 Trillion Parameters.Power: Runs on a standard wall outlet (110V/15A).Goal: To allow small businesses and universities to own and run "Frontier Models" without building a data center.III. The Reflex (Robotic Autonomy)Dispersed Pentary Chiplets embedded directly into robotic joints and skins.Function: Local processing of physics and pressure.Goal: To give robots "Spinal Cords"—autonomic reflexes that prevent falling and crushing, processing data in microseconds at the edge, rather than milliseconds in the cloud.4. The "Physics-First" PhilosophyWe reject the brute-force approach of modern GPU architecture.We do not move data to compute; we move compute to the data (In-Memory Processing).We do not simulate math; we use voltage physics to perform the accumulation (Analog-Digital Hybrid).We do not wait for the cloud; the intelligence lives in the device.5. The Call to ArmsWe are building this Open Source.The definitions, the Verilog cores, and the quantization compilers are free. We are democratizing the hardware layer of AI because if the hardware remains closed, the intelligence will remain centralized.To the Hackers: Help us build the FPGA prototypes.To the Researchers: Help us refine the 5-state quantization algorithms.To the Investors: Help us print the first 7nm wafer.The future is not Binary. It is Balanced.[The Pentary Project]Est. 2025
