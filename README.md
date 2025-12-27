# Pentary Computing

**Balanced Quinary (Base-5) Architecture for AI Acceleration**

A research project exploring 5-level quantization for neural networks and alternative computing architectures.

[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)]()
[![License](https://img.shields.io/badge/License-Open%20Source-green)]()
[![Validation](https://img.shields.io/badge/Claims%20Validated-50%2B-blue)]()

---

## What is Pentary?

Pentary computing uses a **balanced base-5 number system** with digits {-2, -1, 0, +1, +2} instead of binary {0, 1}. This approach offers potential advantages for neural network inference through:

- **Compact representation** - Each digit carries 2.32 bits of information ([proof](research/pentary_foundations.md#13-information-density-analysis))
- **Zero-state efficiency** - Zero values can be implemented as physical disconnects
- **Simplified multiplication** - Weights in {-2..+2} require only shift-add operations

---

## Quick Start

### Prerequisites

```bash
# Python 3.7+ required
python3 --version
pip install numpy
```

### Try It Out

```bash
# Interactive CLI
python3 tools/pentary_cli.py

# Run examples
python3 tools/pentary_converter.py
python3 tools/pentary_simulator.py
```

### First Steps

| Goal | Start Here |
|------|------------|
| New to Pentary? | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Learn the basics | [BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md) |
| Quick reference | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |

---

## Project Status

**Phase:** Research & Simulation  
**Confidence Level:** 75-85% on core claims ([detailed assessment](CLAIMS_EVIDENCE_MATRIX.md))

| Area | Status | Notes |
|------|--------|-------|
| Mathematical Theory | ✅ Complete | Proofs verified |
| Software Tools | ✅ Working | Tested and documented |
| Hardware Design | 📝 Designed | Awaiting implementation |
| FPGA Prototype | ⏳ Planned | Not yet built |
| ASIC Fabrication | ⏳ Future | Requires funding |

See [RESEARCH_GAP_ANALYSIS.md](RESEARCH_GAP_ANALYSIS.md) for honest assessment of what's proven vs theoretical.

---

## Key Claims & Evidence

All major claims are documented with evidence. See [CLAIMS_EVIDENCE_MATRIX.md](CLAIMS_EVIDENCE_MATRIX.md) for full details.

### Verified Claims (85-100% confidence)

| Claim | Evidence | Status |
|-------|----------|--------|
| 2.32× information density vs binary | [Mathematical proof](research/pentary_foundations.md) | ✅ Proven |
| 10× memory reduction for NNs | [Benchmark results](validation/nn_benchmark_report.md) | ✅ Measured |
| 2.43× multiplication speedup | [Complexity analysis](VALIDATION_MASTER_REPORT.md) | ✅ Calculated |
| 3-8× fewer cycles | [Hardware simulation](validation/hardware_benchmark_report.md) | ✅ Simulated |

### Plausible Claims (Pending Hardware Validation)

| Claim | Evidence | Status |
|-------|----------|--------|
| 20× smaller multipliers | [Design analysis](architecture/pentary_alu_design.md) | ⚠️ Designed |
| 45% power reduction | [Power modeling](VALIDATION_SUMMARY.md) | ⚠️ Modeled |
| Zero-state power savings | [Theoretical analysis](research/pentary_foundations.md) | ⚠️ Theoretical |

---

## Documentation Directory

### Getting Started

- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup and first steps
- [BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md) - Learn pentary concepts
- [QUICK_START.md](QUICK_START.md) - 5-minute quickstart
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet

### Core Documentation

- [INDEX.md](INDEX.md) - Complete project index
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Executive summary
- [PENTARY_COMPLETE_GUIDE.md](PENTARY_COMPLETE_GUIDE.md) - Comprehensive guide
- [USER_GUIDE.md](USER_GUIDE.md) - User documentation

### Research & Theory

| Document | Description |
|----------|-------------|
| [research/pentary_foundations.md](research/pentary_foundations.md) | Mathematical foundations with proofs |
| [research/pentary_logic_gates.md](research/pentary_logic_gates.md) | Logic gate designs and truth tables |
| [research/pentary_sota_comparison.md](research/pentary_sota_comparison.md) | Comparison with TPU, GPU, etc. |
| [research/pentary_ai_architectures_analysis.md](research/pentary_ai_architectures_analysis.md) | AI architecture analysis |
| [research/RESEARCH_ROADMAP.md](research/RESEARCH_ROADMAP.md) | Research priorities |

Full research index: [RESEARCH_INDEX.md](RESEARCH_INDEX.md) (48 documents)

### Architecture & Design

| Document | Description |
|----------|-------------|
| [architecture/pentary_processor_architecture.md](architecture/pentary_processor_architecture.md) | ISA specification (50+ instructions) |
| [architecture/pentary_alu_design.md](architecture/pentary_alu_design.md) | ALU circuit designs |
| [architecture/pentary_memory_model.md](architecture/pentary_memory_model.md) | Memory hierarchy design |
| [hardware/memristor_implementation.md](hardware/memristor_implementation.md) | Memristor hardware design |
| [hardware/CHIP_DESIGN_EXPLAINED.md](hardware/CHIP_DESIGN_EXPLAINED.md) | Complete chip explanation |

### Hardware Implementation

| Document | Description |
|----------|-------------|
| [hardware/pentary_chip_design.v](hardware/pentary_chip_design.v) | Verilog implementation |
| [hardware/pentary_chip_layout.md](hardware/pentary_chip_layout.md) | Physical layout guidelines |
| [FPGA_PROTOTYPE_GUIDE.md](FPGA_PROTOTYPE_GUIDE.md) | FPGA prototyping guide |
| [CHIP_DESIGN_ROADMAP.md](CHIP_DESIGN_ROADMAP.md) | Implementation roadmap |

### Validation & Evidence

| Document | Description |
|----------|-------------|
| [CLAIMS_EVIDENCE_MATRIX.md](CLAIMS_EVIDENCE_MATRIX.md) | All claims with evidence links |
| [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) | Validation overview |
| [VALIDATION_MASTER_REPORT.md](VALIDATION_MASTER_REPORT.md) | Complete validation (50+ pages) |
| [RESEARCH_GAP_ANALYSIS.md](RESEARCH_GAP_ANALYSIS.md) | Honest gap assessment |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | Reproducible benchmark instructions |
| [validation/README.md](validation/README.md) | Test framework documentation |

### Planning & Implementation

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | Step-by-step development guide |
| [QUANTIZATION_COMPARISON.md](QUANTIZATION_COMPARISON.md) | Comparison with INT8/INT4/FP8 |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Failure modes and limitations |
| [FAQ.md](FAQ.md) | Common questions answered |
| [ACADEMIC_PAPER_OUTLINE.md](ACADEMIC_PAPER_OUTLINE.md) | Publication-ready paper outline |
| [FUTURE_RESEARCH_DIRECTIONS.md](FUTURE_RESEARCH_DIRECTIONS.md) | Research gaps and next steps |
| [research/RECENT_ADVANCES_INTEGRATION.md](research/RECENT_ADVANCES_INTEGRATION.md) | Latest research integration |

### Tools & Software

| Tool | Description |
|------|-------------|
| [tools/pentary_cli.py](tools/pentary_cli.py) | Interactive command-line interface |
| [tools/pentary_converter.py](tools/pentary_converter.py) | Number conversion utilities |
| [tools/pentary_arithmetic.py](tools/pentary_arithmetic.py) | Arithmetic operations |
| [tools/pentary_simulator.py](tools/pentary_simulator.py) | Processor simulator |
| [tools/pentary_nn.py](tools/pentary_nn.py) | Neural network layers |
| [tools/pentary_quantizer.py](tools/pentary_quantizer.py) | Model quantization |

Full tools documentation: [tools/README.md](tools/README.md)

### Language

- [language/README.md](language/README.md) - Pent programming language
- [language/pent_language_spec.md](language/pent_language_spec.md) - Language specification
- [language/examples/](language/examples/) - Example programs

---

## Repository Structure

```
pentary/
├── README.md                 # This file
├── CLAIMS_EVIDENCE_MATRIX.md # Claims with evidence links
├── RESEARCH_GAP_ANALYSIS.md  # Honest gap assessment
├── VALIDATION_SUMMARY.md     # Validation overview
│
├── research/                 # 48 research documents (250K+ words)
│   ├── pentary_foundations.md
│   ├── pentary_logic_gates.md
│   └── ...
│
├── architecture/             # Processor architecture specs
│   ├── pentary_processor_architecture.md
│   ├── pentary_alu_design.md
│   └── ...
│
├── hardware/                 # Hardware implementation
│   ├── pentary_chip_design.v
│   ├── memristor_implementation.md
│   └── ...
│
├── tools/                    # Working software (20+ tools)
│   ├── pentary_cli.py
│   ├── pentary_converter.py
│   └── ...
│
├── validation/               # Test suite and results
│   ├── claims_extracted.json    # 12,084 claims extracted
│   ├── validation_framework.py
│   └── ...
│
├── language/                 # Pent programming language
├── pdk/                      # Process design kit integration
├── references/               # Academic references
└── diagrams/                 # Visual documentation
```

---

## Technical Overview

### Pentary Number System

| Digit | Value | Symbol |
|-------|-------|--------|
| -2 | Strong negative | ⊖ |
| -1 | Weak negative | − |
| 0 | Zero | 0 |
| +1 | Weak positive | + |
| +2 | Strong positive | ⊕ |

**Example:** 42 decimal = ⊕⊖⊕ pentary (2×25 - 2×5 + 2×1)

### Information Density

```
Binary:  log₂(2) = 1.00 bits per digit
Pentary: log₂(5) = 2.32 bits per digit

Ratio: 2.32× information density advantage
```

See [mathematical proof](research/pentary_foundations.md#13-information-density-analysis).

### Neural Network Application

Pentary maps naturally to 5-level neural network quantization:

| Weight Range | Pentary Value |
|--------------|---------------|
| [-1.0, -0.6] | -2 (⊖) |
| [-0.6, -0.2] | -1 (−) |
| [-0.2, +0.2] | 0 |
| [+0.2, +0.6] | +1 (+) |
| [+0.6, +1.0] | +2 (⊕) |

---

## Validation Methodology

All claims are validated using:

1. **Mathematical Proofs** - Formal derivations from first principles
2. **Software Benchmarks** - Tested implementations in Python
3. **Hardware Simulations** - ALU models at multiple bit widths
4. **Literature Review** - Cross-referenced with published research

### Run Validation

```bash
# Run benchmarks
python3 validation/pentary_hardware_tests.py
python3 validation/pentary_nn_benchmarks.py

# View results
cat validation/hardware_benchmark_report.md
cat validation/nn_benchmark_report.md
```

---

## References

This project builds on established research in:

- Multi-valued logic (Setun computer, 1958)
- Neural network quantization (Han et al., 2016)
- Memristor technology (Chua, 1971; Strukov et al., 2008)

See [REFERENCES_SUMMARY.md](REFERENCES_SUMMARY.md) for complete bibliography.

---

## Contributing

Contributions welcome in these areas:

| Area | Skills Needed |
|------|---------------|
| FPGA Prototyping | Verilog/VHDL, FPGA tools |
| Hardware Validation | Test engineering |
| Neural Network Research | PyTorch/TensorFlow, ML |
| Documentation | Technical writing |

See [todo.md](todo.md) for current priorities.

---

## Limitations & Caveats

**This is a research project, not production-ready technology.**

- No physical hardware has been fabricated
- Many performance claims are simulation-based
- Manufacturing costs are estimates
- Competitive comparisons are modeled, not measured

See [RESEARCH_GAP_ANALYSIS.md](RESEARCH_GAP_ANALYSIS.md) for detailed limitations.

---

## License

Open Source Hardware Initiative

---

## Contact

- Issues: Use GitHub Issues
- Documentation updates: Submit PRs
- Research collaboration: See [research/RESEARCH_ROADMAP.md](research/RESEARCH_ROADMAP.md)

---

**Project Status:** Research Prototype  
**Validation Coverage:** 50+ critical claims validated  
**Documentation:** 250,000+ words across 150+ files  
**Last Updated:** December 2024
