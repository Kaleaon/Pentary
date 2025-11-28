# Pentary Processor - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the Pentary Processor tools immediately.

## Prerequisites

```bash
# Python 3.8 or higher
python --version

# Clone the repository
git clone https://github.com/Kaleaon/Pentary.git
cd Pentary
```

## Step 1: Understanding Pentary Numbers (2 minutes)

Pentary uses 5 digits: **⊖ (−2), − (−1), 0, + (+1), ⊕ (+2)**

```python
# Quick examples
0 in pentary = 0
5 in pentary = +0
10 in pentary = ⊕0
42 in pentary = ⊕⊖⊕
```

## Step 2: Run the Converter (1 minute)

```bash
cd tools
python pentary_converter.py
```

**Output**: You'll see conversions, arithmetic, and shifts working!

```
Decimal  Pentary    Back to Decimal
0        0          0
1        +          1
2        ⊕          2
5        +0         5
10       ⊕0         10
42       ⊕⊖⊕        42
```

## Step 3: Try Arithmetic (1 minute)

```bash
python pentary_arithmetic.py
```

**Output**: Detailed addition with step-by-step traces!

```
⊕+ + +⊖ = +⊖-
(11 + 3 = 14)

Step-by-step:
Pos   A     B     C_in    Sum   C_out
0     +     ⊖     0       ⊖     0
1     ⊕     +     0       -     +
2     0     0     +       +     0
```

## Step 4: Run the Simulator (1 minute)

```bash
python pentary_simulator.py
```

**Output**: Three example programs execute!

```
Example 1: Simple Arithmetic
MOVI P1, 5      # P1 = 5
MOVI P2, 3      # P2 = 3
ADD P3, P1, P2  # P3 = 8
```

## 🎯 Your First Program

Create a file `my_program.py`:

```python
from tools.pentary_simulator import PentaryProcessor

# Create processor
proc = PentaryProcessor()

# Write your program
program = [
    "# Calculate 7 * 2 using shift",
    "MOVI P1, 7",      # P1 = 7
    "MUL2 P2, P1",     # P2 = 7 * 2 = 14
    "HALT"
]

# Run it
proc.load_program(program)
proc.run(verbose=True)
proc.print_state()
```

Run it:
```bash
python my_program.py
```

## 📚 What to Read Next

### For Beginners
1. **PENTARY_COMPLETE_GUIDE.md** - Start here for overview
2. **docs/visual_guide.md** - See diagrams and illustrations

### For Developers
1. **tools/pentary_converter.py** - Study the code
2. **tools/pentary_simulator.py** - Understand the ISA
3. **architecture/pentary_processor_architecture.md** - Full ISA spec

### For Hardware Engineers
1. **architecture/pentary_alu_design.md** - Circuit designs
2. **hardware/memristor_implementation.md** - Physical implementation

### For Researchers
1. **research/pentary_foundations.md** - Mathematical theory
2. **research/pentary_logic_gates.md** - Logic gate designs

## 🛠️ Common Tasks

### Convert a Number

```python
from tools.pentary_converter import PentaryConverter

# Decimal to pentary
pentary = PentaryConverter.decimal_to_pentary(42)
print(f"42 = {pentary}")  # Output: ⊕⊖⊕

# Pentary to decimal
decimal = PentaryConverter.pentary_to_decimal("⊕⊖⊕")
print(f"⊕⊖⊕ = {decimal}")  # Output: 42
```

### Add Two Numbers

```python
from tools.pentary_converter import PentaryConverter

result = PentaryConverter.add_pentary("+⊕", "⊕-")
print(f"+⊕ + ⊕- = {result}")  # 7 + 9 = 16
```

### Multiply by 2 (Efficient!)

```python
from tools.pentary_converter import PentaryConverter

result = PentaryConverter.multiply_pentary_by_constant("+⊕", 2)
print(f"+⊕ × 2 = {result}")  # 7 × 2 = 14
```

### Shift Left (Multiply by 5)

```python
from tools.pentary_converter import PentaryConverter

result = PentaryConverter.shift_left_pentary("+⊕", 1)
print(f"+⊕ << 1 = {result}")  # 7 × 5 = 35
```

## 🎓 Learning Path

### Week 1: Basics
- [ ] Understand pentary number system
- [ ] Run all example tools
- [ ] Write simple programs
- [ ] Read PENTARY_COMPLETE_GUIDE.md

### Week 2: Architecture
- [ ] Study the ISA
- [ ] Understand the ALU design
- [ ] Learn about the pipeline
- [ ] Read architecture documentation

### Week 3: Implementation
- [ ] Study circuit designs
- [ ] Understand memristor technology
- [ ] Learn about in-memory computing
- [ ] Read hardware documentation

### Week 4: Advanced
- [ ] Write complex programs
- [ ] Optimize for performance
- [ ] Contribute to the project
- [ ] Start your own experiments

## 💡 Pro Tips

### 1. Use the Simulator for Learning
The simulator has verbose mode that shows exactly what's happening:
```python
proc.run(verbose=True)  # See every instruction execute
```

### 2. Check the State Anytime
```python
proc.print_state()  # See all registers, flags, and memory
```

### 3. Start Simple
Begin with arithmetic, then move to loops, then memory operations.

### 4. Read the Examples
The simulator includes three working examples - study them!

### 5. Experiment Freely
The tools are safe to experiment with - try different operations!

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the tools directory
cd tools
python pentary_converter.py
```

### Symbol Display Issues
If you see weird characters instead of ⊖ and ⊕:
- Your terminal might not support Unicode
- The code still works, just looks different

### Python Version
```bash
# Need Python 3.8+
python --version
# or
python3 --version
```

## 📖 Instruction Reference

### Arithmetic
- `ADD Rd, Rs1, Rs2` - Add two registers
- `SUB Rd, Rs1, Rs2` - Subtract
- `ADDI Rd, Rs1, Imm` - Add immediate
- `NEG Rd, Rs1` - Negate
- `MUL2 Rd, Rs1` - Multiply by 2

### Memory
- `LOAD Rd, [Rs + offset]` - Load from memory
- `STORE Rs, [Rd + offset]` - Store to memory
- `PUSH Rs` - Push to stack
- `POP Rd` - Pop from stack

### Control Flow
- `BEQ Rs, target` - Branch if equal to zero
- `BNE Rs, target` - Branch if not equal
- `BLT Rs, target` - Branch if less than zero
- `BGT Rs, target` - Branch if greater than zero
- `JUMP target` - Unconditional jump
- `HALT` - Stop execution

### Pseudo-Instructions
- `MOVI Rd, Imm` - Move immediate to register
- `NOP` - No operation

## 🎯 Example Programs

### 1. Simple Addition
```assembly
MOVI P1, 5
MOVI P2, 3
ADD P3, P1, P2
HALT
```

### 2. Loop (Sum 1 to 10)
```assembly
MOVI P1, 0      # sum = 0
MOVI P2, 1      # i = 1
MOVI P3, 10     # limit = 10
ADD P1, P1, P2  # sum += i
ADDI P2, P2, 1  # i++
SUB P4, P2, P3  # temp = i - limit
BLT P4, 3       # if temp < 0, goto line 3
HALT
```

### 3. Memory Operations
```assembly
MOVI P1, 10         # base address
MOVI P2, 42         # value
STORE P2, [P1 + 0]  # mem[10] = 42
LOAD P3, [P1 + 0]   # P3 = mem[10]
HALT
```

## 🚀 Next Steps

1. **Run all examples** - Get familiar with the tools
2. **Write your own programs** - Start experimenting
3. **Read the documentation** - Understand the architecture
4. **Join the community** - Contribute and collaborate

## 📞 Getting Help

- **Documentation**: Read PENTARY_COMPLETE_GUIDE.md
- **Examples**: Study the working programs
- **Code**: Read the tool source code
- **Community**: Join GitHub discussions

## 🎉 You're Ready!

You now have:
- ✅ Working tools installed
- ✅ Basic understanding of pentary
- ✅ Example programs to study
- ✅ Resources to learn more

**Start coding and have fun! 🚀**

---

**The future is not Binary. It is Balanced.**

---

*Quick Start Guide v1.0*  
*Last Updated: January 2025*