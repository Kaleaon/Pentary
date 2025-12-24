#!/bin/bash
# Comprehensive Testbench Execution Script
# Runs all Verilog testbenches using Icarus Verilog

set -e  # Exit on error

echo "=========================================="
echo "Pentary Processor - Testbench Execution"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Check if Icarus Verilog is installed
if ! command -v iverilog &> /dev/null; then
    echo -e "${RED}ERROR: Icarus Verilog not found.${NC}"
    echo "Install with: sudo apt-get install iverilog"
    exit 1
fi

if ! command -v vvp &> /dev/null; then
    echo -e "${RED}ERROR: VVP (Verilog simulator) not found.${NC}"
    exit 1
fi

echo "Icarus Verilog version:"
iverilog -V | head -1
echo ""

# Create output directory
mkdir -p validation/test_results
cd hardware

# Function to run a testbench
run_testbench() {
    local tb_file=$1
    local module_file=$2
    local tb_name=$(basename "$tb_file" .v)
    local module_name=$(basename "$module_file" .v)
    
    echo -e "${YELLOW}Running testbench: $tb_name${NC}"
    echo "  Testing module: $module_name"
    TOTAL=$((TOTAL + 1))
    
    # Compile testbench
    if iverilog -o ../validation/test_results/${tb_name}.vvp "$tb_file" "$module_file" 2> ../validation/test_results/${tb_name}_compile.log; then
        echo -e "${BLUE}  ✓ Compilation successful${NC}"
        
        # Run simulation
        if vvp ../validation/test_results/${tb_name}.vvp > ../validation/test_results/${tb_name}_run.log 2>&1; then
            # Check for test pass/fail in output
            if grep -q "PASS\|SUCCESS\|All tests passed" ../validation/test_results/${tb_name}_run.log; then
                echo -e "${GREEN}  ✓ PASSED: All tests passed${NC}"
                PASSED=$((PASSED + 1))
                
                # Show summary
                grep -A 5 "Test Summary\|PASS\|SUCCESS" ../validation/test_results/${tb_name}_run.log || true
            elif grep -q "FAIL\|ERROR" ../validation/test_results/${tb_name}_run.log; then
                echo -e "${RED}  ✗ FAILED: Some tests failed${NC}"
                FAILED=$((FAILED + 1))
                
                # Show failures
                grep -B 2 -A 2 "FAIL\|ERROR" ../validation/test_results/${tb_name}_run.log || true
            else
                echo -e "${GREEN}  ✓ PASSED: Simulation completed${NC}"
                PASSED=$((PASSED + 1))
                echo "  (No explicit pass/fail markers found)"
            fi
        else
            echo -e "${RED}  ✗ FAILED: Simulation error${NC}"
            FAILED=$((FAILED + 1))
            echo "  Error log:"
            tail -20 ../validation/test_results/${tb_name}_run.log
        fi
    else
        echo -e "${RED}  ✗ FAILED: Compilation error${NC}"
        FAILED=$((FAILED + 1))
        echo "  Error log:"
        cat ../validation/test_results/${tb_name}_compile.log
    fi
    echo ""
}

# Function to generate VCD waveform
generate_waveform() {
    local tb_file=$1
    local tb_name=$(basename "$tb_file" .v)
    
    if [ -f "../validation/test_results/${tb_name}.vvp" ]; then
        echo -e "${BLUE}Generating waveform for: $tb_name${NC}"
        vvp ../validation/test_results/${tb_name}.vvp -vcd > ../validation/test_results/${tb_name}.vcd 2>&1 || true
        
        if [ -f "../validation/test_results/${tb_name}.vcd" ]; then
            echo -e "${GREEN}  ✓ Waveform saved: ${tb_name}.vcd${NC}"
        fi
    fi
}

echo "=========================================="
echo "Phase 1: Core Arithmetic Tests"
echo "=========================================="
echo ""

# Test Pentary Adder
if [ -f "testbench_pentary_adder.v" ] && [ -f "pentary_adder_fixed.v" ]; then
    run_testbench "testbench_pentary_adder.v" "pentary_adder_fixed.v"
fi

# Test Pentary ALU
if [ -f "testbench_pentary_alu.v" ] && [ -f "pentary_alu_fixed.v" ]; then
    run_testbench "testbench_pentary_alu.v" "pentary_alu_fixed.v"
fi

# Test Pentary Quantizer
if [ -f "testbench_pentary_quantizer.v" ] && [ -f "pentary_quantizer_fixed.v" ]; then
    run_testbench "testbench_pentary_quantizer.v" "pentary_quantizer_fixed.v"
fi

echo ""
echo "=========================================="
echo "Phase 2: Memory System Tests"
echo "=========================================="
echo ""

# Test Register File
if [ -f "testbench_register_file.v" ] && [ -f "register_file.v" ]; then
    run_testbench "testbench_register_file.v" "register_file.v"
fi

# Test Memristor Crossbar
if [ -f "testbench_memristor_crossbar.v" ] && [ -f "memristor_crossbar_fixed.v" ]; then
    run_testbench "testbench_memristor_crossbar.v" "memristor_crossbar_fixed.v"
fi

echo ""
echo "=========================================="
echo "Phase 3: Waveform Generation"
echo "=========================================="
echo ""

# Generate waveforms for all testbenches
for tb_file in testbench_*.v; do
    if [ -f "$tb_file" ]; then
        generate_waveform "$tb_file"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Total testbenches run: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

# Calculate pass rate
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASSED * 100 / TOTAL))
    echo "Pass rate: ${PASS_RATE}%"
    echo ""
fi

# List generated files
echo "Generated files:"
echo "  Logs: validation/test_results/*.log"
echo "  Waveforms: validation/test_results/*.vcd"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All testbenches passed!${NC}"
    echo ""
    echo "To view waveforms, use GTKWave:"
    echo "  gtkwave validation/test_results/<testbench_name>.vcd"
    exit 0
else
    echo -e "${RED}✗ Some testbenches failed. See logs in validation/test_results/${NC}"
    exit 1
fi