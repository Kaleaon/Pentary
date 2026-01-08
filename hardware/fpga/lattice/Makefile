# =============================================================================
# Pentary iCE40 Build Makefile
# =============================================================================
#
# Toolchain: Yosys + nextpnr-ice40 + icestorm
#
# Targets:
#   make all       - Build everything
#   make synthesis - Run Yosys synthesis
#   make pnr       - Run nextpnr place and route
#   make bitstream - Generate bitstream
#   make program   - Program the FPGA
#   make clean     - Remove build artifacts
#
# =============================================================================

# Project settings
PROJECT = pentary_ice40
TOP_MODULE = pentary_ice40

# Target FPGA
# Options: up5k, hx8k, lp8k
DEVICE = up5k
PACKAGE = sg48

# Clock constraint (MHz)
FREQ = 12

# Source files
SOURCES = pentary_ice40.v

# Constraints
PCF = pentary_ice40.pcf

# Build directory
BUILD_DIR = build

# Tools
YOSYS = yosys
NEXTPNR = nextpnr-ice40
ICEPACK = icepack
ICEPROG = iceprog
ICETIME = icetime

# Yosys flags
YOSYS_FLAGS = -p "synth_ice40 -top $(TOP_MODULE) -json $(BUILD_DIR)/$(PROJECT).json"

# nextpnr flags
NEXTPNR_FLAGS = --$(DEVICE) --package $(PACKAGE) --freq $(FREQ)
NEXTPNR_FLAGS += --json $(BUILD_DIR)/$(PROJECT).json
NEXTPNR_FLAGS += --pcf $(PCF)
NEXTPNR_FLAGS += --asc $(BUILD_DIR)/$(PROJECT).asc

# =============================================================================
# Targets
# =============================================================================

.PHONY: all synthesis pnr bitstream program timing clean

all: bitstream timing

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Synthesis with Yosys
synthesis: $(BUILD_DIR)/$(PROJECT).json

$(BUILD_DIR)/$(PROJECT).json: $(SOURCES) | $(BUILD_DIR)
	$(YOSYS) $(YOSYS_FLAGS) $(SOURCES)
	@echo "Synthesis complete: $@"

# Place and Route with nextpnr
pnr: $(BUILD_DIR)/$(PROJECT).asc

$(BUILD_DIR)/$(PROJECT).asc: $(BUILD_DIR)/$(PROJECT).json $(PCF)
	$(NEXTPNR) $(NEXTPNR_FLAGS)
	@echo "Place and route complete: $@"

# Generate bitstream
bitstream: $(BUILD_DIR)/$(PROJECT).bin

$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).asc
	$(ICEPACK) $(BUILD_DIR)/$(PROJECT).asc $(BUILD_DIR)/$(PROJECT).bin
	@echo "Bitstream generated: $@"

# Timing analysis
timing: $(BUILD_DIR)/$(PROJECT).asc
	$(ICETIME) -d $(DEVICE) -mtr $(BUILD_DIR)/$(PROJECT).rpt $(BUILD_DIR)/$(PROJECT).asc
	@echo "Timing report: $(BUILD_DIR)/$(PROJECT).rpt"
	@cat $(BUILD_DIR)/$(PROJECT).rpt

# Program FPGA
program: $(BUILD_DIR)/$(PROJECT).bin
	$(ICEPROG) $(BUILD_DIR)/$(PROJECT).bin

# Program to flash (persistent)
program-flash: $(BUILD_DIR)/$(PROJECT).bin
	$(ICEPROG) -S $(BUILD_DIR)/$(PROJECT).bin

# Clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.blif *.json *.asc *.bin *.rpt

# =============================================================================
# Debug targets
# =============================================================================

# Show synthesis statistics
stats: $(BUILD_DIR)/$(PROJECT).json
	$(YOSYS) -p "read_json $(BUILD_DIR)/$(PROJECT).json; stat"

# Generate Verilog from synthesized design (for debugging)
synth-v: $(BUILD_DIR)/$(PROJECT).json
	$(YOSYS) -p "read_json $(BUILD_DIR)/$(PROJECT).json; write_verilog $(BUILD_DIR)/$(PROJECT)_synth.v"

# View design in GTKWave (requires test bench)
wave:
	gtkwave $(BUILD_DIR)/$(PROJECT).vcd &

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Pentary iCE40 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build everything (default)"
	@echo "  synthesis  - Run Yosys synthesis"
	@echo "  pnr        - Run nextpnr place and route"
	@echo "  bitstream  - Generate programming bitstream"
	@echo "  program    - Program FPGA (SRAM, volatile)"
	@echo "  program-flash - Program FPGA flash (persistent)"
	@echo "  timing     - Generate timing report"
	@echo "  stats      - Show resource usage statistics"
	@echo "  clean      - Remove build artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  DEVICE  = $(DEVICE)"
	@echo "  PACKAGE = $(PACKAGE)"
	@echo "  FREQ    = $(FREQ) MHz"
