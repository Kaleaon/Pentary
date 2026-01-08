# =============================================================================
# Cocotb Makefile for Pentary Tiny Tapeout Testbench
# =============================================================================
#
# Usage:
#   make                  - Run all tests with Icarus Verilog
#   make SIM=verilator    - Run with Verilator
#   make waves            - Run and generate waveform
#   make clean            - Clean build artifacts
#
# Prerequisites:
#   - Python 3.8+
#   - cocotb (pip install cocotb)
#   - Icarus Verilog or Verilator
#
# =============================================================================

# Simulator selection (icarus or verilator)
SIM ?= icarus

# Simulator-specific configuration
ifeq ($(SIM), verilator)
	EXTRA_ARGS += --trace --trace-structs
	COMPILE_ARGS += -Wno-fatal
endif

# Language (verilog or vhdl)
TOPLEVEL_LANG ?= verilog

# Verilog source files
VERILOG_SOURCES += $(PWD)/../src/tt_um_pentary_3t.v

# Top-level module
TOPLEVEL = tt_um_pentary_3t

# Python test module
MODULE = test_pentary

# Cocotb test configuration
export COCOTB_REDUCED_LOG_FMT = 1
export COCOTB_RESOLVE_X = ZEROS

# Waveform generation
WAVES ?= 0
ifeq ($(WAVES), 1)
	ifeq ($(SIM), icarus)
		COMPILE_ARGS += -DWAVES
		PLUSARGS += +vcd
	endif
	ifeq ($(SIM), verilator)
		EXTRA_ARGS += --trace-fst
	endif
endif

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Convenience targets
.PHONY: waves clean_all lint

waves:
	$(MAKE) WAVES=1

clean_all: clean
	rm -rf __pycache__ sim_build results.xml
	rm -f *.vcd *.fst *.log

lint:
	verilator --lint-only $(VERILOG_SOURCES)

# Test-specific targets
.PHONY: test_quick test_all

test_quick:
	TESTCASE=test_reset,test_add_simple,test_sub_simple $(MAKE)

test_all:
	$(MAKE)

# Report generation
.PHONY: report

report: 
	@echo "============================================="
	@echo "        COCOTB TEST REPORT"
	@echo "============================================="
	@if [ -f results.xml ]; then \
		python3 -c "import xml.etree.ElementTree as ET; \
		tree = ET.parse('results.xml'); \
		root = tree.getroot(); \
		tests = root.findall('.//testcase'); \
		passed = len([t for t in tests if not t.find('failure')]); \
		failed = len([t for t in tests if t.find('failure')]); \
		print(f'Tests Run: {len(tests)}'); \
		print(f'Passed:    {passed}'); \
		print(f'Failed:    {failed}')"; \
	else \
		echo "No results.xml found. Run tests first."; \
	fi
	@echo "============================================="
