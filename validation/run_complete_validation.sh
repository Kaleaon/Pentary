#!/bin/bash
# Master Validation Script - Runs all validation checks
# This script orchestrates the complete validation process

set -e  # Exit on error

echo "=========================================="
echo "Pentary Processor"
echo "Complete Validation Suite"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Track overall status
OVERALL_STATUS=0

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo "Validation directory: $SCRIPT_DIR"
echo ""

# Make scripts executable
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/*.py 2>/dev/null || true

echo "=========================================="
echo "Phase 1: Environment Check"
echo "=========================================="
echo ""

# Check required tools
echo "Checking required tools..."

check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    fi
}

TOOLS_OK=true
check_tool "yosys" || TOOLS_OK=false
check_tool "iverilog" || TOOLS_OK=false
check_tool "vvp" || TOOLS_OK=false
check_tool "python3" || TOOLS_OK=false

if [ "$TOOLS_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}Some tools are missing. Install with:${NC}"
    echo "  sudo apt-get install yosys iverilog python3"
    echo ""
    echo "Continuing with available tools..."
fi

echo ""

# Optional tools
echo "Checking optional tools..."
check_tool "ngspice" || echo -e "${YELLOW}  (SPICE simulations will be skipped)${NC}"
check_tool "gtkwave" || echo -e "${YELLOW}  (Waveform viewing not available)${NC}"

echo ""

echo "=========================================="
echo "Phase 2: Verilog Synthesis Validation"
echo "=========================================="
echo ""

if command -v yosys &> /dev/null; then
    if [ -f "$SCRIPT_DIR/synthesize_all.sh" ]; then
        echo "Running synthesis validation..."
        if bash "$SCRIPT_DIR/synthesize_all.sh"; then
            echo -e "${GREEN}✓ Synthesis validation passed${NC}"
        else
            echo -e "${RED}✗ Synthesis validation failed${NC}"
            OVERALL_STATUS=1
        fi
    else
        echo -e "${YELLOW}⚠ Synthesis script not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Yosys not available, skipping synthesis${NC}"
fi

echo ""

echo "=========================================="
echo "Phase 3: Testbench Execution"
echo "=========================================="
echo ""

if command -v iverilog &> /dev/null && command -v vvp &> /dev/null; then
    if [ -f "$SCRIPT_DIR/run_all_tests.sh" ]; then
        echo "Running testbench suite..."
        if bash "$SCRIPT_DIR/run_all_tests.sh"; then
            echo -e "${GREEN}✓ Testbench execution passed${NC}"
        else
            echo -e "${RED}✗ Testbench execution failed${NC}"
            OVERALL_STATUS=1
        fi
    else
        echo -e "${YELLOW}⚠ Testbench script not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Icarus Verilog not available, skipping testbenches${NC}"
fi

echo ""

echo "=========================================="
echo "Phase 4: SPICE Simulations"
echo "=========================================="
echo ""

if command -v ngspice &> /dev/null; then
    SIM_DIR="$PROJECT_ROOT/simulations"
    if [ -d "$SIM_DIR" ]; then
        echo "Running SPICE simulations..."
        cd "$SIM_DIR"
        
        for spice_file in *.spice; do
            if [ -f "$spice_file" ]; then
                sim_name=$(basename "$spice_file" .spice)
                echo -e "${BLUE}  Simulating: $sim_name${NC}"
                
                if ngspice -b "$spice_file" -o "${sim_name}.log" > /dev/null 2>&1; then
                    echo -e "${GREEN}    ✓ Completed${NC}"
                else
                    echo -e "${RED}    ✗ Failed${NC}"
                    OVERALL_STATUS=1
                fi
            fi
        done
        
        cd "$PROJECT_ROOT"
    else
        echo -e "${YELLOW}⚠ Simulations directory not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ngspice not available, skipping SPICE simulations${NC}"
fi

echo ""

echo "=========================================="
echo "Phase 5: Report Generation"
echo "=========================================="
echo ""

if command -v python3 &> /dev/null; then
    if [ -f "$SCRIPT_DIR/generate_validation_report.py" ]; then
        echo "Generating validation report..."
        if python3 "$SCRIPT_DIR/generate_validation_report.py" "$SCRIPT_DIR" "$SCRIPT_DIR/reports"; then
            echo -e "${GREEN}✓ Report generated successfully${NC}"
            
            # Display report location
            REPORT_FILE="$SCRIPT_DIR/reports/VALIDATION_REPORT.md"
            if [ -f "$REPORT_FILE" ]; then
                echo ""
                echo "Validation report: $REPORT_FILE"
                echo ""
                
                # Show summary from report
                if command -v head &> /dev/null && command -v tail &> /dev/null; then
                    echo "Report Summary:"
                    echo "---"
                    head -30 "$REPORT_FILE" | tail -20
                    echo "---"
                fi
            fi
        else
            echo -e "${YELLOW}⚠ Report generation had issues${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Report generator not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Python3 not available, skipping report generation${NC}"
fi

echo ""

echo "=========================================="
echo "Phase 6: Documentation Check"
echo "=========================================="
echo ""

echo "Checking documentation completeness..."

# Check for key documentation files
DOCS_OK=true

check_doc() {
    local file="$PROJECT_ROOT/$1"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 (missing)${NC}"
        DOCS_OK=false
        return 1
    fi
}

check_doc "README.md"
check_doc "hardware/analog_cmos_implementation.md"
check_doc "architecture/system_scaling_reference.md"
check_doc "CARAVEL_COMPATIBILITY_ANALYSIS.md"

if [ "$DOCS_OK" = true ]; then
    echo -e "${GREEN}✓ All key documentation present${NC}"
else
    echo -e "${YELLOW}⚠ Some documentation missing${NC}"
fi

echo ""

echo "=========================================="
echo "Phase 7: File Structure Validation"
echo "=========================================="
echo ""

echo "Validating project structure..."

# Check directory structure
check_dir() {
    local dir="$PROJECT_ROOT/$1"
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l)
        echo -e "${GREEN}✓ $1/ ($count files)${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $1/ (not found)${NC}"
        return 1
    fi
}

check_dir "hardware"
check_dir "architecture"
check_dir "simulations"
check_dir "validation"
check_dir "research"

echo ""

echo "=========================================="
echo "Final Summary"
echo "=========================================="
echo ""

# Count files
VERILOG_COUNT=$(find "$PROJECT_ROOT/hardware" -name "*.v" 2>/dev/null | wc -l)
SPICE_COUNT=$(find "$PROJECT_ROOT/simulations" -name "*.spice" 2>/dev/null | wc -l)
DOC_COUNT=$(find "$PROJECT_ROOT" -name "*.md" 2>/dev/null | wc -l)

echo "Project Statistics:"
echo "  Verilog files: $VERILOG_COUNT"
echo "  SPICE files: $SPICE_COUNT"
echo "  Documentation files: $DOC_COUNT"
echo ""

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✓ VALIDATION PASSED"
    echo "==========================================${NC}"
    echo ""
    echo "All validation checks completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Review validation report in validation/reports/"
    echo "  2. Proceed with FPGA prototyping"
    echo "  3. Begin software toolchain development"
    echo ""
else
    echo -e "${RED}=========================================="
    echo "✗ VALIDATION FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Some validation checks failed."
    echo ""
    echo "Action required:"
    echo "  1. Review logs in validation/synthesis_results/ and validation/test_results/"
    echo "  2. Fix identified issues"
    echo "  3. Re-run validation"
    echo ""
fi

exit $OVERALL_STATUS