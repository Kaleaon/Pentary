#!/bin/bash
# Comprehensive Verilog Synthesis Validation Script
# Uses Yosys (open-source synthesis tool)

set -e  # Exit on error

echo "=========================================="
echo "Pentary Processor - Synthesis Validation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Check if Yosys is installed
if ! command -v yosys &> /dev/null; then
    echo -e "${RED}ERROR: Yosys not found. Please install Yosys.${NC}"
    echo "Install with: sudo apt-get install yosys"
    exit 1
fi

echo "Yosys version:"
yosys -V
echo ""

# Create output directory
mkdir -p validation/synthesis_results
cd hardware

# Function to synthesize a Verilog file
synthesize_module() {
    local file=$1
    local module_name=$(basename "$file" .v)
    
    echo -e "${YELLOW}Synthesizing: $module_name${NC}"
    TOTAL=$((TOTAL + 1))
    
    # Create Yosys script
    cat > ../validation/synthesis_results/${module_name}_synth.ys << EOF
# Read Verilog file
read_verilog $file

# Hierarchy check
hierarchy -check -top $module_name

# Translate processes
proc

# Optimize
opt

# Technology mapping (generic)
techmap

# Optimize again
opt

# Clean up
clean

# Statistics
stat

# Write output
write_verilog ../validation/synthesis_results/${module_name}_synth.v

# Check for errors
EOF

    # Run synthesis
    if yosys -s ../validation/synthesis_results/${module_name}_synth.ys > ../validation/synthesis_results/${module_name}_synth.log 2>&1; then
        echo -e "${GREEN}✓ PASSED: $module_name${NC}"
        PASSED=$((PASSED + 1))
        
        # Extract statistics
        grep -A 10 "Printing statistics" ../validation/synthesis_results/${module_name}_synth.log || true
        echo ""
    else
        echo -e "${RED}✗ FAILED: $module_name${NC}"
        FAILED=$((FAILED + 1))
        echo "Error log:"
        tail -20 ../validation/synthesis_results/${module_name}_synth.log
        echo ""
    fi
}

# Function to check syntax only
check_syntax() {
    local file=$1
    local module_name=$(basename "$file" .v)
    
    echo -e "${YELLOW}Checking syntax: $module_name${NC}"
    TOTAL=$((TOTAL + 1))
    
    # Create Yosys script for syntax check
    cat > ../validation/synthesis_results/${module_name}_check.ys << EOF
read_verilog $file
hierarchy -check
EOF

    # Run syntax check
    if yosys -s ../validation/synthesis_results/${module_name}_check.ys > ../validation/synthesis_results/${module_name}_check.log 2>&1; then
        echo -e "${GREEN}✓ PASSED: $module_name (syntax)${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAILED: $module_name (syntax)${NC}"
        FAILED=$((FAILED + 1))
        echo "Error log:"
        tail -20 ../validation/synthesis_results/${module_name}_check.log
    fi
    echo ""
}

echo "=========================================="
echo "Phase 1: Syntax Checking"
echo "=========================================="
echo ""

# Check all Verilog files for syntax
for file in *.v; do
    if [ -f "$file" ]; then
        check_syntax "$file"
    fi
done

echo ""
echo "=========================================="
echo "Phase 2: Module Synthesis"
echo "=========================================="
echo ""

# Synthesize core modules
echo "--- Core Arithmetic Modules ---"
[ -f "pentary_adder_fixed.v" ] && synthesize_module "pentary_adder_fixed.v"
[ -f "pentary_alu_fixed.v" ] && synthesize_module "pentary_alu_fixed.v"
[ -f "pentary_quantizer_fixed.v" ] && synthesize_module "pentary_quantizer_fixed.v"

echo ""
echo "--- Memory Modules ---"
[ -f "register_file.v" ] && synthesize_module "register_file.v"
[ -f "cache_hierarchy.v" ] && synthesize_module "cache_hierarchy.v"
[ -f "memristor_crossbar_fixed.v" ] && synthesize_module "memristor_crossbar_fixed.v"

echo ""
echo "--- Control Modules ---"
[ -f "pipeline_control.v" ] && synthesize_module "pipeline_control.v"
[ -f "instruction_decoder.v" ] && synthesize_module "instruction_decoder.v"
[ -f "mmu_interrupt.v" ] && synthesize_module "mmu_interrupt.v"

echo ""
echo "--- Integration ---"
[ -f "pentary_core_integrated.v" ] && synthesize_module "pentary_core_integrated.v"

echo ""
echo "=========================================="
echo "Phase 3: Testbench Validation"
echo "=========================================="
echo ""

# Check testbenches (syntax only, don't synthesize)
echo "--- Testbenches ---"
for file in testbench_*.v; do
    if [ -f "$file" ]; then
        check_syntax "$file"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Total modules checked: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All synthesis checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some synthesis checks failed. See logs in validation/synthesis_results/${NC}"
    exit 1
fi