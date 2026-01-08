# =============================================================================
# Pentary Processor - Vivado Build Script
# =============================================================================
# Usage: vivado -mode batch -source build_vivado.tcl
# =============================================================================

# Project settings
set project_name "pentary_basys3"
set part_number "xc7a35tcpg236-1"
set board_part "digilentinc.com:basys3:part0:1.2"

# Get script directory
set script_dir [file dirname [info script]]

# Create project
create_project ${project_name} ${script_dir}/${project_name} -part ${part_number} -force

# Set board part if available
catch {set_property board_part ${board_part} [current_project]}

# Add source files
add_files -norecurse ${script_dir}/pentary_artix7.v

# Add constraints
add_files -fileset constrs_1 -norecurse ${script_dir}/basys3.xdc

# Set top module
set_property top pentary_artix7 [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# Run synthesis
puts "Running synthesis..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis results
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

puts "Synthesis complete!"

# Report synthesis utilization
open_run synth_1
report_utilization -file ${script_dir}/${project_name}/utilization_synth.rpt
report_timing_summary -file ${script_dir}/${project_name}/timing_synth.rpt

# Run implementation
puts "Running implementation..."
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation results
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

puts "Implementation complete!"

# Generate reports
open_run impl_1
report_utilization -file ${script_dir}/${project_name}/utilization_impl.rpt
report_timing_summary -file ${script_dir}/${project_name}/timing_impl.rpt
report_power -file ${script_dir}/${project_name}/power_impl.rpt

# Generate bitstream
puts "Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Check for bitstream
set bitstream_file ${script_dir}/${project_name}/${project_name}.runs/impl_1/pentary_artix7.bit
if {[file exists ${bitstream_file}]} {
    puts "Bitstream generated: ${bitstream_file}"
    # Copy to project root
    file copy -force ${bitstream_file} ${script_dir}/pentary_artix7.bit
} else {
    puts "ERROR: Bitstream generation failed!"
    exit 1
}

puts ""
puts "============================================="
puts " BUILD COMPLETE"
puts "============================================="
puts "Bitstream: ${script_dir}/pentary_artix7.bit"
puts ""

# Print utilization summary
puts "Utilization Summary:"
puts "-------------------------------------------"
report_utilization -hierarchical -hierarchical_depth 2

exit 0
