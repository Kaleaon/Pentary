# Getting Started with Pentary Computing

Welcome to Pentary! This guide will help you get up and running quickly.

## 🚀 Quick Start (5 Minutes)

### Step 1: Setup

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd pentary

# Run setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Check Python version (needs 3.7+)
- ✅ Install dependencies (numpy)
- ✅ Verify tools are present
- ✅ Make scripts executable

### Step 2: Try the Interactive CLI

```bash
python3 tools/pentary_cli.py
```

You'll see:
```
============================================================
  PENTARY INTERACTIVE CLI - Balanced Quinary Computing
============================================================

pentary> 
```

Try these commands:
```
pentary> convert 42
pentary> add 7 9
pentary> examples
pentary> help
```

### Step 3: Run Example Tools

```bash
# See number conversions
python3 tools/pentary_converter.py

# See detailed arithmetic
python3 tools/pentary_arithmetic.py

# Run a simple program
python3 tools/pentary_simulator.py
```

**Congratulations!** You're now using pentary computing! 🎉

---

## 📚 Learning Path

### For Complete Beginners

1. **Start Here**: [BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md)
   - What is pentary?
   - Why use it?
   - Basic operations
   - Your first program

2. **Next**: [QUICK_START.md](QUICK_START.md)
   - More examples
   - Common patterns
   - Troubleshooting

3. **Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
   - Command cheat sheet
   - Common conversions
   - Instruction set

### For Developers

1. **Tools**: [tools/README.md](tools/README.md)
   - All available tools
   - API documentation
   - Examples

2. **Architecture**: [architecture/pentary_processor_architecture.md](architecture/pentary_processor_architecture.md)
   - Complete ISA specification
   - Register conventions
   - Memory hierarchy

3. **Language**: [language/README.md](language/README.md)
   - Pent programming language
   - Compiler usage
   - Example programs

### For Hardware Engineers

1. **Chip Design**: [hardware/CHIP_DESIGN_EXPLAINED.md](hardware/CHIP_DESIGN_EXPLAINED.md)
   - Complete chip explanation
   - Circuit details
   - Design rationale

2. **Verilog**: [hardware/pentary_chip_design.v](hardware/pentary_chip_design.v)
   - RTL implementation
   - Component designs
   - Comments and explanations

3. **Memristors**: [hardware/memristor_implementation.md](hardware/memristor_implementation.md)
   - Physical implementation
   - 5-level resistance states
   - Programming guide

4. **Layout**: [hardware/pentary_chip_layout.md](hardware/pentary_chip_layout.md)
   - Floorplan guidelines
   - Design rules
   - Manufacturing considerations

### For Researchers

1. **Foundations**: [research/pentary_foundations.md](research/pentary_foundations.md)
   - Mathematical theory
   - Arithmetic algorithms
   - Proofs and analysis

2. **Logic Gates**: [research/pentary_logic_gates.md](research/pentary_logic_gates.md)
   - Gate designs
   - Truth tables
   - Circuit implementations

---

## 🛠️ Available Tools

### Interactive Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **CLI** | `python3 tools/pentary_cli.py` | Interactive command-line interface |
| **Example Generator** | `python3 tools/example_generator.py` | Generate example code |

### Core Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **Converter** | `python3 tools/pentary_converter.py` | Convert between number systems |
| **Arithmetic** | `python3 tools/pentary_arithmetic.py` | Detailed arithmetic operations |
| **Simulator** | `python3 tools/pentary_simulator.py` | Run pentary programs |

### Neural Network Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **NN Layers** | `python3 tools/pentary_nn.py` | Neural network layers |
| **Quantizer** | `python3 tools/pentary_quantizer.py` | Model quantization |
| **Trainer** | `python3 tools/pentary_trainer.py` | Training utilities |

See [tools/README.md](tools/README.md) for complete documentation.

---

## 💡 Common Tasks

### Convert a Number

**Using CLI:**
```bash
python3 tools/pentary_cli.py
pentary> convert 42
```

**Using Python:**
```python
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()
pent = c.decimal_to_pentary(42)
print(pent)  # Output: ⊕⊖⊕
```

### Add Two Numbers

**Using CLI:**
```bash
pentary> add 7 9
```

**Using Python:**
```python
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()
result = c.add_pentary("+⊕", "⊕-")
print(result)  # Output: ⊕+0 (16 in decimal)
```

### Run a Program

**Using CLI:**
```bash
pentary> run fibonacci
```

**Using Python:**
```python
from tools.pentary_simulator import PentaryProcessor

proc = PentaryProcessor()
program = [
    "MOVI P1, 5",
    "MOVI P2, 3",
    "ADD P3, P1, P2",
    "HALT"
]
proc.load_program(program)
proc.run(verbose=True)
proc.print_state()
```

### Generate Examples

```bash
# Interactive mode
python3 tools/example_generator.py

# Generate all examples
python3 tools/example_generator.py --all

# Specific examples
python3 tools/example_generator.py --conversions
python3 tools/example_generator.py --arithmetic
python3 tools/example_generator.py --programs
python3 tools/example_generator.py --python
```

---

## 📖 Documentation Structure

```
pentary/
├── GETTING_STARTED.md          ← You are here!
├── BEGINNER_TUTORIAL.md         ← Start here if new
├── QUICK_START.md               ← 5-minute guide
├── QUICK_REFERENCE.md            ← Cheat sheet
├── PENTARY_COMPLETE_GUIDE.md    ← Comprehensive guide
│
├── tools/                       ← Software tools
│   ├── pentary_cli.py          ← Interactive CLI
│   ├── pentary_converter.py    ← Number conversion
│   ├── pentary_simulator.py     ← Processor simulator
│   └── README.md                ← Tools documentation
│
├── architecture/               ← Processor design
│   └── pentary_processor_architecture.md
│
├── hardware/                   ← Chip design
│   ├── CHIP_DESIGN_EXPLAINED.md ← Complete explanation
│   ├── pentary_chip_design.v    ← Verilog RTL
│   ├── memristor_implementation.md
│   └── pentary_chip_layout.md
│
├── research/                   ← Theory
│   ├── pentary_foundations.md
│   └── pentary_logic_gates.md
│
└── language/                    ← Pent language
    └── README.md
```

---

## 🎯 Next Steps

### Immediate Next Steps

1. ✅ **You've completed setup** - Great!
2. 📖 **Read BEGINNER_TUTORIAL.md** - Learn the basics
3. 🛠️ **Try the CLI** - Practice with `pentary_cli.py`
4. 💻 **Write a program** - Create your own program

### Short Term (This Week)

- [ ] Complete the beginner tutorial
- [ ] Try all example tools
- [ ] Write 3-5 simple programs
- [ ] Read the architecture documentation

### Medium Term (This Month)

- [ ] Understand the complete ISA
- [ ] Study the chip design
- [ ] Learn about memristors
- [ ] Contribute improvements

### Long Term

- [ ] Build your own tools
- [ ] Design custom circuits
- [ ] Contribute to the project
- [ ] Help others learn

---

## 🐛 Troubleshooting

### Problem: "Module not found"

**Solution:**
```bash
# Make sure you're in project root
cd /path/to/pentary

# Install dependencies
pip install numpy

# Or use setup script
./setup.sh
```

### Problem: "Command not found"

**Solution:**
```bash
# Use python3 (not python)
python3 tools/pentary_cli.py

# Make scripts executable
chmod +x tools/*.py
```

### Problem: Unicode symbols don't display

**Solution:**
- Your terminal may not support Unicode
- Code still works, just looks different
- Try a different terminal (iTerm2, Windows Terminal)

### Problem: Import errors

**Solution:**
```bash
# Make sure you're in project root
cd /path/to/pentary

# Check Python path
python3 -c "import sys; print(sys.path)"

# Install dependencies
pip install -r requirements.txt
```

---

## 📞 Getting Help

### Documentation

- **Beginner**: [BEGINNER_TUTORIAL.md](BEGINNER_TUTORIAL.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Complete Guide**: [PENTARY_COMPLETE_GUIDE.md](PENTARY_COMPLETE_GUIDE.md)

### Examples

- **CLI Examples**: Run `pentary_cli.py` and type `examples`
- **Code Examples**: Run `example_generator.py`
- **Program Examples**: See `language/examples/`

### Community

- **GitHub Issues**: Report bugs, ask questions
- **Documentation**: Read the guides
- **Code**: Study the tools source code

---

## 🎉 You're Ready!

You now have:
- ✅ Working tools installed
- ✅ Understanding of where to start
- ✅ Resources to learn more
- ✅ Examples to study

**Start exploring!** The future is not Binary. It is Balanced. 🚀

---

## 📝 Quick Checklist

- [ ] Ran `./setup.sh` successfully
- [ ] Tried `pentary_cli.py`
- [ ] Ran example tools
- [ ] Read BEGINNER_TUTORIAL.md
- [ ] Bookmarked QUICK_REFERENCE.md
- [ ] Ready to learn more!

---

*Getting Started Guide v1.0*  
*Last Updated: January 2025*
