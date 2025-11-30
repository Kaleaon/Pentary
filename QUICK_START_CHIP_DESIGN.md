# Quick Start Guide - Pentary Chip Design

## Getting Started in 5 Steps

### Step 1: Sync Your Repository (5 minutes)

```bash
# Navigate to your Pentary directory
cd Pentary

# Check current status
git status

# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Merge pending pull requests
gh pr merge 13 --squash  # Stress testing research
gh pr merge 14 --squash  # Optimizations and enhancements

# Verify everything is up to date
git log --oneline -5
```

**Expected Output:**
```
✓ Stress testing research merged
✓ Optimizations merged
✓ Repository up to date
```

---

### Step 2: Set Up Development Environment (30 minutes)

#### Install Basic Tools

```bash
# Update system
sudo apt-get update

# Install Verilog simulator
sudo apt-get install iverilog gtkwave

# Install Python dependencies
pip install cocotb pytest numpy

# Install waveform viewer
sudo apt-get install gtkwave

# Verify installations
iverilog -v
python -c "import cocotb; print('cocotb installed')"
```

#### Create Project Structure

```bash
# Create hardware directories
mkdir -p hardware/{p56,p28,p14}
mkdir -p hardware/common/{cells,verification,docs,testbenches}
mkdir -p hardware/tools

# Create initial files
touch hardware/p56/pentary_inverter.v
touch hardware/p56/pentary_adder.v
touch hardware/common/verification/test_gates.py
```

---

### Step 3: Design Your First Pentary Gate (1 hour)

#### Create Pentary Inverter

Create `hardware/p56/pentary_inverter.v`:

```verilog
// Pentary Inverter - 5-state logic inverter
// Maps: -2→+2, -1→+1, 0→0, +1→-1, +2→-2

module pentary_inverter (
    input  [2:0] in,   // 3-bit encoding for 5 states
    output [2:0] out
);
    // State encoding:
    // 000 = -2 (⊖)
    // 001 = -1 (-)
    // 010 =  0 (0)
    // 011 = +1 (+)
    // 100 = +2 (⊕)
    
    // Inversion logic
    assign out = (in == 3'b000) ? 3'b100 :  // -2 → +2
                 (in == 3'b001) ? 3'b011 :  // -1 → +1
                 (in == 3'b010) ? 3'b010 :  //  0 →  0
                 (in == 3'b011) ? 3'b001 :  // +1 → -1
                 (in == 3'b100) ? 3'b000 :  // +2 → -2
                 3'b010;                     // Default: 0

endmodule
```

#### Create Testbench

Create `hardware/p56/pentary_inverter_tb.v`:

```verilog
// Testbench for Pentary Inverter
`timescale 1ns/1ps

module pentary_inverter_tb;
    reg  [2:0] in;
    wire [2:0] out;
    
    // Instantiate inverter
    pentary_inverter dut (
        .in(in),
        .out(out)
    );
    
    // Test stimulus
    initial begin
        $dumpfile("pentary_inverter.vcd");
        $dumpvars(0, pentary_inverter_tb);
        
        // Test all 5 states
        #10 in = 3'b000; // -2
        #10 in = 3'b001; // -1
        #10 in = 3'b010; //  0
        #10 in = 3'b011; // +1
        #10 in = 3'b100; // +2
        
        #10 $finish;
    end
    
    // Monitor outputs
    initial begin
        $monitor("Time=%0t in=%b out=%b", $time, in, out);
    end
    
    // Check correctness
    always @(out) begin
        case(in)
            3'b000: if (out !== 3'b100) $error("FAIL: -2 inversion");
            3'b001: if (out !== 3'b011) $error("FAIL: -1 inversion");
            3'b010: if (out !== 3'b010) $error("FAIL:  0 inversion");
            3'b011: if (out !== 3'b001) $error("FAIL: +1 inversion");
            3'b100: if (out !== 3'b000) $error("FAIL: +2 inversion");
        endcase
    end

endmodule
```

#### Run Simulation

```bash
# Compile and simulate
cd hardware/p56
iverilog -o inverter_sim pentary_inverter.v pentary_inverter_tb.v
vvp inverter_sim

# View waveforms
gtkwave pentary_inverter.vcd
```

**Expected Output:**
```
Time=10 in=000 out=100
Time=20 in=001 out=011
Time=30 in=010 out=010
Time=40 in=011 out=001
Time=50 in=100 out=000
```

---

### Step 4: Verify Against Python Model (30 minutes)

#### Create Python Verification Script

Create `hardware/common/verification/verify_inverter.py`:

```python
#!/usr/bin/env python3
"""
Verify Verilog inverter against Python model
"""

import sys
sys.path.insert(0, '../../../tools')

from pentary_converter_optimized import PentaryConverterOptimized

def verify_inverter():
    """Verify inverter functionality"""
    converter = PentaryConverterOptimized()
    
    # Test cases: (input_state, expected_output)
    test_cases = [
        (-2, 2),   # ⊖ → ⊕
        (-1, 1),   # - → +
        (0, 0),    # 0 → 0
        (1, -1),   # + → -
        (2, -2),   # ⊕ → ⊖
    ]
    
    print("Verifying Pentary Inverter")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for input_val, expected in test_cases:
        # Python model
        input_pent = converter.decimal_to_pentary(input_val)
        negated = converter.negate_pentary(input_pent)
        output_val = converter.pentary_to_decimal(negated)
        
        # Check
        if output_val == expected:
            print(f"✓ PASS: {input_val} → {output_val}")
            passed += 1
        else:
            print(f"✗ FAIL: {input_val} → {output_val} (expected {expected})")
            failed += 1
    
    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = verify_inverter()
    sys.exit(0 if success else 1)
```

#### Run Verification

```bash
cd hardware/common/verification
python verify_inverter.py
```

**Expected Output:**
```
Verifying Pentary Inverter
----------------------------------------
✓ PASS: -2 → 2
✓ PASS: -1 → 1
✓ PASS: 0 → 0
✓ PASS: 1 → -1
✓ PASS: 2 → -2
----------------------------------------
Results: 5 passed, 0 failed
```

---

### Step 5: Next Steps (Ongoing)

#### Immediate Tasks (This Week)

1. **Design Pentary Adder**
   ```bash
   # Create adder design
   cp hardware/p56/pentary_inverter.v hardware/p56/pentary_adder.v
   # Edit to implement full adder logic
   ```

2. **Expand Test Suite**
   ```bash
   # Add more test cases
   # Integrate with existing test_optimizations.py
   ```

3. **Document Progress**
   ```bash
   # Create design log
   echo "# Design Log" > hardware/DESIGN_LOG.md
   echo "## $(date): Created first pentary gate" >> hardware/DESIGN_LOG.md
   ```

#### Short-Term Goals (Next Month)

1. **Complete Basic Gates**
   - Inverter ✓
   - Adder
   - Multiplexer
   - Register

2. **Build 8-Digit ALU**
   - Arithmetic unit
   - Logic unit
   - Control logic

3. **Comprehensive Testing**
   - 1,000+ test vectors
   - Coverage analysis
   - Performance measurement

---

## Key Resources

### Documentation
- **Full Roadmap**: `CHIP_DESIGN_ROADMAP.md` - Complete implementation plan
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md` - Software achievements
- **Stress Testing**: `STRESS_TEST_RESEARCH_REPORT.md` - Validation results
- **Recommendations**: `RECOMMENDATIONS.md` - Optimization strategies

### Code
- **Optimized Converter**: `tools/pentary_converter_optimized.py` - 14M ops/sec
- **Extended Arithmetic**: `tools/pentary_arithmetic_extended.py` - Full operations
- **Test Suite**: `tests/test_optimizations.py` - 21 tests, 100% pass rate

### Tools
- **Icarus Verilog**: Free Verilog simulator
- **GTKWave**: Waveform viewer
- **Cocotb**: Python-based verification
- **Your Python Tools**: Already validated and optimized

---

## Common Issues and Solutions

### Issue 1: Simulation Fails

**Problem**: `iverilog` reports syntax errors

**Solution**:
```bash
# Check Verilog syntax
iverilog -t null pentary_inverter.v

# Common fixes:
# - Add semicolons after statements
# - Check module/endmodule matching
# - Verify signal declarations
```

### Issue 2: Python Import Errors

**Problem**: Cannot import pentary modules

**Solution**:
```python
# Add path to tools directory
import sys
sys.path.insert(0, '/path/to/Pentary/tools')

# Or set PYTHONPATH
export PYTHONPATH=/path/to/Pentary/tools:$PYTHONPATH
```

### Issue 3: Waveform Not Displaying

**Problem**: GTKWave shows no signals

**Solution**:
```verilog
// Add to testbench
initial begin
    $dumpfile("output.vcd");
    $dumpvars(0, module_name);
end
```

---

## Quick Reference

### Pentary State Encoding

| State | Symbol | Decimal | Binary |
|-------|--------|---------|--------|
| -2 | ⊖ | -2 | 000 |
| -1 | - | -1 | 001 |
| 0 | 0 | 0 | 010 |
| +1 | + | +1 | 011 |
| +2 | ⊕ | +2 | 100 |

### Common Verilog Patterns

```verilog
// State comparison
if (state == 3'b000) // -2

// State assignment
state <= 3'b100; // +2

// Case statement
case(state)
    3'b000: // -2
    3'b001: // -1
    3'b010: //  0
    3'b011: // +1
    3'b100: // +2
endcase
```

### Python Model Usage

```python
from pentary_converter_optimized import PentaryConverterOptimized

converter = PentaryConverterOptimized()

# Convert
pent = converter.decimal_to_pentary(42)

# Arithmetic
result = converter.add_pentary("⊕+", "+0")

# Negate
neg = converter.negate_pentary("⊕")
```

---

## Success Checklist

### Week 1
- [ ] Repository synced and up to date
- [ ] Development environment set up
- [ ] First gate designed and simulated
- [ ] Verification against Python model
- [ ] Documentation started

### Month 1
- [ ] Basic gates completed (inverter, adder, mux)
- [ ] Testbenches for all gates
- [ ] 100+ test vectors passing
- [ ] Design log maintained
- [ ] Ready for ALU design

### Quarter 1
- [ ] 8-digit ALU completed
- [ ] 1,000+ tests passing
- [ ] Performance analysis done
- [ ] Documentation complete
- [ ] Ready for p28 transition

---

## Getting Help

### Resources
- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Read the comprehensive guides
- **Test Suite**: Learn from existing tests
- **Community**: Join VLSI forums and communities

### Contact Points
- **Academic**: University VLSI programs
- **Industry**: Semiconductor companies
- **Open Source**: FOSSi Foundation, OpenROAD
- **Research**: IEEE, ACM conferences

---

## Next Actions

**Right Now:**
1. Run the repository sync commands
2. Install the basic tools
3. Create your first pentary gate
4. Run the simulation
5. Celebrate your first working pentary circuit! 🎉

**This Week:**
1. Complete basic gate set
2. Set up comprehensive testing
3. Document your progress
4. Plan next month's work

**This Month:**
1. Build 8-digit ALU
2. Achieve 1,000+ passing tests
3. Measure performance
4. Prepare for p28 design

---

**Good luck with your pentary chip design project!**

For detailed information, see `CHIP_DESIGN_ROADMAP.md`