# Pentary Computing - Quick Reference Guide

A quick reference card for common pentary operations and commands.

## Pentary Digits

| Symbol | Name | Value | Encoding |
|--------|------|-------|----------|
| ⊖ | Strong Negative | -2 | 000 |
| - | Weak Negative | -1 | 001 |
| 0 | Zero | 0 | 010 |
| + | Weak Positive | +1 | 011 |
| ⊕ | Strong Positive | +2 | 100 |

## Common Conversions

| Decimal | Pentary | Notes |
|---------|---------|-------|
| 0 | 0 | Zero |
| 1 | + | One positive |
| 2 | ⊕ | Two positive |
| 5 | +0 | One positive, zero |
| 10 | ⊕0 | Two positive, zero |
| 25 | +00 | One positive, two zeros |
| 42 | ⊕⊖⊕ | Two, negative two, two |

## CLI Commands

### Interactive CLI
```bash
python3 tools/pentary_cli.py
```

### Commands
```
convert <n>        Convert decimal to pentary
add <a> <b>        Add two numbers
sub <a> <b>        Subtract two numbers
mul <a> <b>        Multiply two numbers
run <program>      Run example program
examples           Show examples
help               Show help
quit/exit          Exit
```

## Python API

### Converter
```python
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()

# Convert
pent = c.decimal_to_pentary(42)        # "⊕⊖⊕"
dec = c.pentary_to_decimal("⊕⊖⊕")     # 42

# Arithmetic
result = c.add_pentary("+⊕", "⊕-")     # Addition
result = c.subtract_pentary("⊕0", "+-") # Subtraction
result = c.multiply_pentary_by_constant("+⊕", 2)  # Multiply
```

### Simulator
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

## Instruction Set (Common)

### Arithmetic
```
ADD  Rd, Rs1, Rs2    Rd = Rs1 + Rs2
SUB  Rd, Rs1, Rs2    Rd = Rs1 - Rs2
ADDI Rd, Rs1, Imm    Rd = Rs1 + Imm
MUL2 Rd, Rs1         Rd = Rs1 × 2
NEG  Rd, Rs1         Rd = -Rs1
```

### Memory
```
LOAD  Rd, [Rs + off]  Rd = mem[Rs + off]
STORE Rs, [Rd + off]  mem[Rd + off] = Rs
PUSH  Rs              Push Rs to stack
POP   Rd              Pop from stack to Rd
```

### Control Flow
```
JUMP  target          Unconditional jump
BEQ   Rs, target      Branch if Rs == 0
BNE   Rs, target      Branch if Rs != 0
BLT   Rs, target      Branch if Rs < 0
BGT   Rs, target      Branch if Rs > 0
HALT                  Stop execution
```

### Pseudo-Instructions
```
MOVI  Rd, Imm         Rd = Imm (move immediate)
NOP                   No operation
```

## Register Conventions

| Register | Name | Purpose |
|----------|------|---------|
| P0 | Zero | Always zero (hardwired) |
| P1-P28 | General | General purpose |
| P29 | FP | Frame pointer |
| P30 | SP | Stack pointer |
| P31 | LR | Link register |

## Status Flags

| Flag | Meaning | Set When |
|------|---------|----------|
| Z | Zero | Result == 0 |
| N | Negative | Result < 0 |
| P | Positive | Result > 0 |
| V | Overflow | Result overflowed |
| C | Carry | Carry occurred |

## Common Patterns

### Loop
```python
"MOVI P1, 0",      # counter = 0
"MOVI P2, 10",     # limit = 10
"ADDI P1, P1, 1",  # counter++
"SUB P3, P1, P2",  # temp = counter - limit
"BLT P3, 2",       # if temp < 0, loop
```

### Conditional
```python
"SUB P3, P1, P2",  # temp = a - b
"BEQ P3, label",   # if temp == 0
"BLT P3, label",   # if temp < 0
"BGT P3, label",   # if temp > 0
```

### Function Call
```python
"MOVI P31, return", # Save return address
"JUMP function",    # Call function
"return:",          # Return point
```

## File Locations

| File | Purpose |
|------|---------|
| `tools/pentary_cli.py` | Interactive CLI |
| `tools/pentary_converter.py` | Number conversion |
| `tools/pentary_arithmetic.py` | Arithmetic operations |
| `tools/pentary_simulator.py` | Processor simulator |
| `BEGINNER_TUTORIAL.md` | Beginner guide |
| `QUICK_START.md` | Quick start guide |
| `PENTARY_COMPLETE_GUIDE.md` | Complete reference |
| `hardware/CHIP_DESIGN_EXPLAINED.md` | Chip design guide |

## Quick Examples

### Convert 42
```bash
python3 tools/pentary_cli.py
pentary> convert 42
# Output: ⊕⊖⊕
```

### Add 7 + 9
```bash
pentary> add 7 9
# Output: 16
```

### Run Fibonacci
```bash
pentary> run fibonacci
```

### Python Script
```python
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()
print(c.decimal_to_pentary(42))  # ⊕⊖⊕
```

## Troubleshooting

### Import Error
```bash
# Make sure you're in project root
cd /path/to/pentary
python3 tools/pentary_cli.py
```

### Unicode Symbols
If ⊖ and ⊕ don't display:
- Terminal may not support Unicode
- Code still works, just looks different
- Try different terminal

### Python Version
```bash
# Need Python 3.7+
python3 --version
```

## Performance Tips

1. **Use pentary weights**: Set `use_pentary=True` in layers
2. **Exploit sparsity**: Prune to 70-80% sparsity
3. **Profile first**: Always profile before optimizing
4. **Batch processing**: Process multiple samples together
5. **In-memory compute**: Use memristor crossbars when available

## Resources

- **Documentation**: See `docs/` directory
- **Examples**: See `language/examples/`
- **Architecture**: See `architecture/`
- **Hardware**: See `hardware/`
- **Research**: See `research/`

---

**The future is not Binary. It is Balanced.** 🚀

*Quick Reference v1.0 - January 2025*
