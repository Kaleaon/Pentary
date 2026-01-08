# =============================================================================
# Verilator Makefile for Pentary Hardware Testbenches
# =============================================================================
#
# Usage:
#   make all         - Build all testbenches
#   make test        - Run all tests
#   make clean       - Remove build artifacts
#   make tb_alu      - Build ALU testbench
#   make tb_adder    - Build Adder testbench
#   make tb_register - Build Register testbench
#
# Requirements:
#   - Verilator (>= 4.0)
#   - GCC or Clang with C++14 support
#
# =============================================================================

# Compiler settings
VERILATOR = verilator
VERILATOR_FLAGS = -Wall -Wno-fatal --trace -cc

CXX = g++
CXXFLAGS = -std=c++14 -O2 -Wall

# Directory structure
VERILOG_DIR = ..
OBJ_DIR = obj_dir

# Verilog source files
VERILOG_SOURCES = \
    $(VERILOG_DIR)/pentary_alu_fixed.v \
    $(VERILOG_DIR)/pentary_adder_fixed.v \
    $(VERILOG_DIR)/register_file.v \
    $(VERILOG_DIR)/memristor_crossbar_fixed.v \
    $(VERILOG_DIR)/pentary_quantizer_fixed.v

# Testbench executables
TB_ALU = $(OBJ_DIR)/VPentaryALU
TB_ADDER = $(OBJ_DIR)/VPentaryAdder16
TB_REGISTER = $(OBJ_DIR)/VRegisterFile

# Default target
all: tb_alu tb_adder tb_register

# =============================================================================
# ALU Testbench
# =============================================================================
tb_alu: $(TB_ALU)

$(TB_ALU): tb_pentary_alu.cpp $(VERILOG_DIR)/pentary_alu_fixed.v $(VERILOG_DIR)/pentary_adder_fixed.v
	$(VERILATOR) $(VERILATOR_FLAGS) \
		$(VERILOG_DIR)/pentary_alu_fixed.v \
		$(VERILOG_DIR)/pentary_adder_fixed.v \
		--exe tb_pentary_alu.cpp \
		--top-module PentaryALU \
		-o VPentaryALU
	$(MAKE) -C $(OBJ_DIR) -f VPentaryALU.mk

# =============================================================================
# Adder Testbench
# =============================================================================
tb_adder: $(TB_ADDER)

$(TB_ADDER): tb_pentary_adder.cpp $(VERILOG_DIR)/pentary_adder_fixed.v
	$(VERILATOR) $(VERILATOR_FLAGS) \
		$(VERILOG_DIR)/pentary_adder_fixed.v \
		--exe tb_pentary_adder.cpp \
		--top-module PentaryAdder16 \
		-o VPentaryAdder16
	$(MAKE) -C $(OBJ_DIR) -f VPentaryAdder16.mk

# =============================================================================
# Register File Testbench
# =============================================================================
tb_register: $(TB_REGISTER)

$(TB_REGISTER): tb_register_file.cpp $(VERILOG_DIR)/register_file.v
	$(VERILATOR) $(VERILATOR_FLAGS) \
		$(VERILOG_DIR)/register_file.v \
		--exe tb_register_file.cpp \
		--top-module RegisterFile \
		-o VRegisterFile
	$(MAKE) -C $(OBJ_DIR) -f VRegisterFile.mk

# =============================================================================
# Test Targets
# =============================================================================
test: test_alu test_adder test_register

test_alu: tb_alu
	@echo "Running ALU tests..."
	@$(TB_ALU)

test_adder: tb_adder
	@echo "Running Adder tests..."
	@$(TB_ADDER)

test_register: tb_register
	@echo "Running Register File tests..."
	@$(TB_REGISTER)

# =============================================================================
# Lint Target (check Verilog syntax without building)
# =============================================================================
lint:
	$(VERILATOR) --lint-only -Wall $(VERILOG_SOURCES)

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -rf $(OBJ_DIR)
	rm -f *.vcd

.PHONY: all clean test lint tb_alu tb_adder tb_register test_alu test_adder test_register
