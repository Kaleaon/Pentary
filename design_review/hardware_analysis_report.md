# Hardware Design Analysis Report

## Summary

- **Total Files Analyzed:** 16
- **Total Issues:** 299
- **Total Warnings:** 397
- **Total Info:** 0

## Critical Issues ❌

### pentary_core_integrated.v

- **Line 0**: Unmatched begin/end: 0 begin, 1 end
- **Line 17**: Possible missing semicolon
  ```verilog
  *   - 32 general-purpose registers
  ```
- **Line 45**: Possible missing semicolon
  ```verilog
  output        debug_halted
  ```
- **Line 184**: Possible missing semicolon
  ```verilog
  .write_enable(wb_reg_write)
  ```
- **Line 237**: Possible missing semicolon
  ```verilog
  assign forwarded_a = (ex_forward_a == 2'b01) ? mem_alu_result :
  ```
- **Line 241**: Possible missing semicolon
  ```verilog
  assign forwarded_b = (ex_forward_b == 2'b01) ? mem_alu_result :
  ```
- **Line 342**: Possible missing semicolon
  ```verilog
  .wb_mem_to_reg(wb_mem_to_reg)
  ```
- **Line 467**: Possible missing semicolon
  ```verilog
  assign debug_reg_data = (debug_reg_addr == 5'b0) ? 48'b0 :
  ```

### testbench_pentary_quantizer.v

- **Line 0**: Unmatched begin/end: 19 begin, 32 end

### mmu_interrupt.v

- **Line 0**: Unmatched begin/end: 41 begin, 55 end
- **Line 42**: Possible missing semicolon
  ```verilog
  input         mem_ready
  ```
- **Line 58**: Possible missing semicolon
  ```verilog
  reg [35:0] tlb_vpn [0:TLB_ENTRIES-1];      // Virtual page number
  ```
- **Line 59**: Possible missing semicolon
  ```verilog
  reg [35:0] tlb_ppn [0:TLB_ENTRIES-1];      // Physical page number
  ```
- **Line 60**: Possible missing semicolon
  ```verilog
  reg        tlb_valid [0:TLB_ENTRIES-1];    // Valid bit
  ```
- **Line 61**: Possible missing semicolon
  ```verilog
  reg        tlb_dirty [0:TLB_ENTRIES-1];    // Dirty bit
  ```
- **Line 62**: Possible missing semicolon
  ```verilog
  reg        tlb_read [0:TLB_ENTRIES-1];     // Read permission
  ```
- **Line 63**: Possible missing semicolon
  ```verilog
  reg        tlb_write [0:TLB_ENTRIES-1];    // Write permission
  ```
- **Line 64**: Possible missing semicolon
  ```verilog
  reg        tlb_execute [0:TLB_ENTRIES-1];  // Execute permission
  ```
- **Line 65**: Possible missing semicolon
  ```verilog
  reg [5:0]  tlb_lru [0:TLB_ENTRIES-1];      // LRU counter
  ```
- **Line 229**: Possible missing semicolon
  ```verilog
  input  [31:0] interrupt_enable,    // Interrupt enable mask
  ```
- **Line 230**: Possible missing semicolon
  ```verilog
  input  [31:0] interrupt_priority,  // Priority configuration
  ```
- **Line 231**: Possible missing semicolon
  ```verilog
  input         global_enable,       // Global interrupt enable
  ```
- **Line 368**: Possible missing semicolon
  ```verilog
  input  [47:0] faulting_addr
  ```
- **Line 438**: Possible missing semicolon
  ```verilog
  input         count_miss
  ```

### testbench_pentary_alu.v

- **Line 0**: Unmatched begin/end: 48 begin, 60 end

### cache_hierarchy.v

- **Line 0**: Unmatched begin/end: 56 begin, 66 end
- **Line 32**: Possible missing semicolon
  ```verilog
  input         l2_ready
  ```
- **Line 41**: Possible missing semicolon
  ```verilog
  wire [5:0]  offset;     // Byte offset within line (6 bits for 64 bytes)
  ```
- **Line 42**: Possible missing semicolon
  ```verilog
  wire [6:0]  index;      // Set index (7 bits for 128 sets)
  ```
- **Line 43**: Possible missing semicolon
  ```verilog
  wire [34:0] tag;        // Tag (remaining bits)
  ```
- **Line 50**: Possible missing semicolon
  ```verilog
  reg [511:0] data [0:NUM_SETS-1][0:WAYS-1];     // Data array
  ```
- **Line 51**: Possible missing semicolon
  ```verilog
  reg [34:0]  tags [0:NUM_SETS-1][0:WAYS-1];     // Tag array
  ```
- **Line 52**: Possible missing semicolon
  ```verilog
  reg         valid [0:NUM_SETS-1][0:WAYS-1];    // Valid bits
  ```
- **Line 53**: Possible missing semicolon
  ```verilog
  reg [1:0]   lru [0:NUM_SETS-1][0:WAYS-1];      // LRU bits
  ```
- **Line 79**: Possible missing semicolon
  ```verilog
  assign word_offset = offset[5:2];  // Word offset within line
  ```
- **Line 153**: Possible missing semicolon
  ```verilog
  input         l2_ready
  ```
- **Line 310**: Possible missing semicolon
  ```verilog
  input         mem_ready
  ```
- **Line 462**: Possible missing semicolon
  ```verilog
  output [47:0] snoop_addr
  ```
- **Line 513**: Possible missing semicolon
  ```verilog
  assign snoop_invalidate = (core0_write && core1_cache_state != INVALID) ||
  ```

### instruction_decoder.v

- **Line 0**: Unmatched begin/end: 37 begin, 49 end
- **Line 10**: Possible missing semicolon
  ```verilog
  *   [27:23] - Destination register (5 bits)
  ```
- **Line 11**: Possible missing semicolon
  ```verilog
  *   [22:18] - Source register 1 (5 bits)
  ```
- **Line 12**: Possible missing semicolon
  ```verilog
  *   [17:13] - Source register 2 (5 bits)
  ```
- **Line 16**: Possible missing semicolon
  ```verilog
  *   - R-type: Register-register operations
  ```
- **Line 30**: Possible missing semicolon
  ```verilog
  output [4:0]  rs1,              // Source register 1
  ```
- **Line 31**: Possible missing semicolon
  ```verilog
  output [4:0]  rs2,              // Source register 2
  ```
- **Line 32**: Possible missing semicolon
  ```verilog
  output [4:0]  rd,               // Destination register
  ```
- **Line 38**: Possible missing semicolon
  ```verilog
  output [2:0]  alu_op,           // ALU operation
  ```
- **Line 39**: Possible missing semicolon
  ```verilog
  output        alu_src,          // ALU source (0=reg, 1=imm)
  ```
- **Line 40**: Possible missing semicolon
  ```verilog
  output        reg_write,        // Write to register file
  ```
- **Line 41**: Possible missing semicolon
  ```verilog
  output        mem_read,         // Read from memory
  ```
- **Line 42**: Possible missing semicolon
  ```verilog
  output        mem_write,        // Write to memory
  ```
- **Line 43**: Possible missing semicolon
  ```verilog
  output        mem_to_reg,       // Write memory data to register
  ```
- **Line 44**: Possible missing semicolon
  ```verilog
  output        branch,           // Branch instruction
  ```
- **Line 45**: Possible missing semicolon
  ```verilog
  output        jump,             // Jump instruction
  ```
- **Line 46**: Possible missing semicolon
  ```verilog
  output        memristor_op,     // Memristor operation
  ```
- **Line 52**: Possible missing semicolon
  ```verilog
  output        illegal_instruction
  ```
- **Line 197**: Possible missing semicolon
  ```verilog
  alu_op_reg = 3'b001;  // Subtract for comparison
  ```
- **Line 204**: Possible missing semicolon
  ```verilog
  alu_op_reg = 3'b001;  // Subtract for comparison
  ```
- **Line 250**: Possible missing semicolon
  ```verilog
  case (inst_type_reg)
  ```
- **Line 314**: Possible missing semicolon
  ```verilog
  output [47:0] branch_target
  ```
- **Line 404**: Possible missing semicolon
  ```verilog
  output        valid
  ```

### pentary_quantizer_fixed.v

- **Line 0**: Unmatched begin/end: 5 begin, 22 end
- **Line 15**: Possible missing semicolon
  ```verilog
  *   1. Subtract zero_point from input
  ```
- **Line 25**: Possible missing semicolon
  ```verilog
  input  [31:0] input_value,   // Q16.16 fixed-point input
  ```
- **Line 26**: Possible missing semicolon
  ```verilog
  input  [31:0] scale,         // Q16.16 scale factor
  ```
- **Line 27**: Possible missing semicolon
  ```verilog
  input  [31:0] zero_point,    // Q16.16 zero point
  ```
- **Line 28**: Possible missing semicolon
  ```verilog
  output [2:0]  quantized      // Single pentary digit output
  ```
- **Line 38**: Possible missing semicolon
  ```verilog
  assign numerator = {centered, 16'b0};  // Shift left by 16 bits
  ```
- **Line 46**: Possible missing semicolon
  ```verilog
  assign rounded = (scaled + 32'h00008000) >>> 16;  // Arithmetic right shift
  ```
- **Line 50**: Possible missing semicolon
  ```verilog
  assign clipped = (rounded < -2) ? -2 :
  ```
- **Line 58**: Possible missing semicolon
  ```verilog
  -2: quantized_reg = 3'b000;  // -2 (⊖)
  ```
- **Line 59**: Possible missing semicolon
  ```verilog
  -1: quantized_reg = 3'b001;  // -1 (-)
  ```
- **Line 60**: Possible missing semicolon
  ```verilog
  0:  quantized_reg = 3'b010;  //  0 (0)
  ```
- **Line 61**: Possible missing semicolon
  ```verilog
  1:  quantized_reg = 3'b011;  // +1 (+)
  ```
- **Line 62**: Possible missing semicolon
  ```verilog
  2:  quantized_reg = 3'b100;  // +2 (⊕)
  ```
- **Line 63**: Possible missing semicolon
  ```verilog
  default: quantized_reg = 3'b010;  // Default to 0
  ```
- **Line 84**: Possible missing semicolon
  ```verilog
  input  [511:0] input_values,  // 16 × 32-bit Q16.16 values
  ```
- **Line 85**: Possible missing semicolon
  ```verilog
  input  [31:0]  scale,         // Shared scale factor
  ```
- **Line 86**: Possible missing semicolon
  ```verilog
  input  [31:0]  zero_point,    // Shared zero point
  ```
- **Line 87**: Possible missing semicolon
  ```verilog
  output [47:0]  quantized      // 16 pentary digits (48 bits)
  ```
- **Line 122**: Possible missing semicolon
  ```verilog
  input  [2:0]  quantized,     // Single pentary digit
  ```
- **Line 123**: Possible missing semicolon
  ```verilog
  input  [31:0] scale,         // Q16.16 scale factor
  ```
- **Line 124**: Possible missing semicolon
  ```verilog
  input  [31:0] zero_point,    // Q16.16 zero point
  ```
- **Line 125**: Possible missing semicolon
  ```verilog
  output [31:0] output_value   // Q16.16 fixed-point output
  ```
- **Line 150**: Possible missing semicolon
  ```verilog
  assign scaled = product[47:16];  // Extract Q16.16 result
  ```
- **Line 170**: Possible missing semicolon
  ```verilog
  input  [31:0] input_value,      // Q16.16 fixed-point input
  ```
- **Line 171**: Possible missing semicolon
  ```verilog
  input  [31:0] scale,            // Q16.16 scale factor
  ```
- **Line 172**: Possible missing semicolon
  ```verilog
  input  [31:0] zero_point,       // Q16.16 zero point
  ```
- **Line 173**: Possible missing semicolon
  ```verilog
  input  [1:0]  rounding_mode,    // 00=nearest, 01=floor, 10=ceil, 11=truncate
  ```
- **Line 174**: Possible missing semicolon
  ```verilog
  input         symmetric,        // 1=symmetric quantization (no zero_point)
  ```
- **Line 175**: Possible missing semicolon
  ```verilog
  output [2:0]  quantized,        // Pentary digit output
  ```
- **Line 176**: Possible missing semicolon
  ```verilog
  output        clipped           // 1 if value was clipped
  ```
- **Line 207**: Possible missing semicolon
  ```verilog
  assign clipped_val = (rounded < -2) ? -2 :
  ```
- **Line 240**: Possible missing semicolon
  ```verilog
  output [31:0] fixed_val
  ```
- **Line 248**: Possible missing semicolon
  ```verilog
  output signed [15:0] int_val
  ```
- **Line 251**: Possible missing semicolon
  ```verilog
  assign rounded = fixed_val + 32'h00008000;  // Add 0.5
  ```
- **Line 252**: Possible missing semicolon
  ```verilog
  assign int_val = rounded[31:16];            // Extract integer part
  ```
- **Line 259**: Possible missing semicolon
  ```verilog
  output [31:0] result
  ```
- **Line 263**: Possible missing semicolon
  ```verilog
  assign result = product[47:16];  // Extract Q16.16 result
  ```
- **Line 270**: Possible missing semicolon
  ```verilog
  output [31:0] result
  ```
- **Line 273**: Possible missing semicolon
  ```verilog
  assign numerator = {$signed(a), 16'b0};  // Shift left by 16
  ```

### testbench_memristor_crossbar.v

- **Line 0**: Unmatched begin/end: 23 begin, 31 end
- **Line 167**: Possible missing semicolon
  ```verilog
  input_vector[i*3 +: 3] = 3'b011;  // +1
  ```
- **Line 217**: Possible missing semicolon
  ```verilog
  input_vector[i*3 +: 3] = 3'b011;  // All +1
  ```

### pentary_chip_design.v

- **Line 0**: Unmatched begin/end: 28 begin, 44 end
- **Line 18**: Possible missing semicolon
  ```verilog
  * - 8 cores, each with 32 registers and 16-pent word size
  ```
- **Line 63**: Possible missing semicolon
  ```verilog
  input  [15:0] operand_a,  // 16 pents (48 bits: 16 × 3 bits)
  ```
- **Line 64**: Possible missing semicolon
  ```verilog
  input  [15:0] operand_b,  // 16 pents (48 bits: 16 × 3 bits)
  ```
- **Line 65**: Possible missing semicolon
  ```verilog
  input  [2:0]  opcode,     // Operation code (3 bits)
  ```
- **Line 66**: Possible missing semicolon
  ```verilog
  output [15:0] result,     // 16 pents (48 bits: 16 × 3 bits)
  ```
- **Line 67**: Possible missing semicolon
  ```verilog
  output        zero_flag,  // Set if result == 0
  ```
- **Line 68**: Possible missing semicolon
  ```verilog
  output        negative_flag,  // Set if result < 0
  ```
- **Line 69**: Possible missing semicolon
  ```verilog
  output        overflow_flag   // Set if overflow occurred
  ```
- **Line 106**: Possible missing semicolon
  ```verilog
  assign result = (opcode == 3'b000) ? sum_result :
  ```
- **Line 114**: Possible missing semicolon
  ```verilog
  assign negative_flag = (result[47] == 1'b1);  // MSB indicates sign
  ```
- **Line 115**: Possible missing semicolon
  ```verilog
  assign overflow_flag = 1'b0;  // Simplified
  ```
- **Line 145**: Possible missing semicolon
  ```verilog
  input  [2:0] a,         // Single pentary digit (3-bit encoded)
  ```
- **Line 146**: Possible missing semicolon
  ```verilog
  input  [2:0] b,         // Single pentary digit (3-bit encoded)
  ```
- **Line 147**: Possible missing semicolon
  ```verilog
  input  [2:0] carry_in,  // Carry from previous position (-1, 0, or +1)
  ```
- **Line 148**: Possible missing semicolon
  ```verilog
  output [2:0] sum,       // Sum digit (result)
  ```
- **Line 149**: Possible missing semicolon
  ```verilog
  output [2:0] carry_out  // Carry to next position (-1, 0, or +1)
  ```
- **Line 193**: Possible missing semicolon
  ```verilog
  input  [2:0]  constant,  // -2, -1, 0, 1, or 2
  ```
- **Line 194**: Possible missing semicolon
  ```verilog
  output [15:0] result
  ```
- **Line 209**: Possible missing semicolon
  ```verilog
  assign result = (constant == 3'b010) ? 16'b0 :           // ×0
  ```
- **Line 224**: Possible missing semicolon
  ```verilog
  output [15:0] output_value
  ```
- **Line 238**: Possible missing semicolon
  ```verilog
  input  [31:0] input_value,  // 32-bit float input
  ```
- **Line 239**: Possible missing semicolon
  ```verilog
  input  [31:0] scale,        // Scale factor
  ```
- **Line 240**: Possible missing semicolon
  ```verilog
  input  [31:0] zero_point,   // Zero point
  ```
- **Line 241**: Possible missing semicolon
  ```verilog
  output [2:0]  quantized     // Single pentary digit output
  ```
- **Line 253**: Possible missing semicolon
  ```verilog
  assign rounded = scaled + 32'h8000;  // Add 0.5 for rounding
  ```
- **Line 254**: Possible missing semicolon
  ```verilog
  assign rounded = rounded & 32'hFFFF0000;  // Truncate
  ```
- **Line 257**: Possible missing semicolon
  ```verilog
  assign quantized = (rounded < -2) ? 3'b000 :  // -2
  ```
- **Line 270**: Possible missing semicolon
  ```verilog
  input  [7:0]  row_select,      // 256 rows
  ```
- **Line 271**: Possible missing semicolon
  ```verilog
  input  [7:0]  col_select,      // 256 columns
  ```
- **Line 272**: Possible missing semicolon
  ```verilog
  input  [2:0]  write_value,      // Pentary value to write
  ```
- **Line 274**: Possible missing semicolon
  ```verilog
  input         compute_enable,   // Enable matrix-vector multiply
  ```
- **Line 275**: Possible missing semicolon
  ```verilog
  input  [255:0] input_vector,    // Input vector (256 pents)
  ```
- **Line 276**: Possible missing semicolon
  ```verilog
  output [255:0] output_vector,   // Output vector (256 pents)
  ```
- **Line 277**: Possible missing semicolon
  ```verilog
  output        ready
  ```
- **Line 282**: Possible missing semicolon
  ```verilog
  reg [2:0] crossbar [0:255][0:255];  // 256×256 array of 3-bit values
  ```
- **Line 331**: Possible missing semicolon
  ```verilog
  assign accumulator = 0;  // Simplified
  ```
- **Line 338**: Possible missing semicolon
  ```verilog
  assign output_vector[i*3 +: 3] = accumulator[2:0];  // Quantize to pentary
  ```
- **Line 353**: Possible missing semicolon
  ```verilog
  input  [31:0] instruction,      // Instruction word
  ```
- **Line 354**: Possible missing semicolon
  ```verilog
  input  [15:0] data_in,         // Data input
  ```
- **Line 355**: Possible missing semicolon
  ```verilog
  output [15:0] data_out,         // Data output
  ```
- **Line 357**: Possible missing semicolon
  ```verilog
  output        stall
  ```
- **Line 362**: Possible missing semicolon
  ```verilog
  reg [15:0] registers [0:31];   // 32 registers
  ```
- **Line 382**: Possible missing semicolon
  ```verilog
  .output_value(relu_result)
  ```
- **Line 420**: Possible missing semicolon
  ```verilog
  assign stall = 1'b0;  // Simplified
  ```

### testbench_pentary_adder.v

- **Line 0**: Unmatched begin/end: 32 begin, 44 end

### pentary_adder_fixed.v

- **Line 0**: Unmatched begin/end: 6 begin, 13 end
- **Line 31**: Possible missing semicolon
  ```verilog
  input  [2:0] a,         // Single pentary digit (3-bit encoded)
  ```
- **Line 32**: Possible missing semicolon
  ```verilog
  input  [2:0] b,         // Single pentary digit (3-bit encoded)
  ```
- **Line 33**: Possible missing semicolon
  ```verilog
  input  [2:0] carry_in,  // Carry from previous position (-1, 0, or +1)
  ```
- **Line 34**: Possible missing semicolon
  ```verilog
  output [2:0] sum,       // Sum digit (result)
  ```
- **Line 35**: Possible missing semicolon
  ```verilog
  output [2:0] carry_out  // Carry to next position (-1, 0, or +1)
  ```
- **Line 80**: Possible missing semicolon
  ```verilog
  carry_reg = 3'b011;  // carry = +1
  ```
- **Line 83**: Possible missing semicolon
  ```verilog
  carry_reg = 3'b001;  // carry = -1
  ```
- **Line 85**: Possible missing semicolon
  ```verilog
  carry_reg = 3'b010;  // carry = 0
  ```
- **Line 113**: Possible missing semicolon
  ```verilog
  *   - Each input is 48 bits (16 digits × 3 bits)
  ```
- **Line 121**: Possible missing semicolon
  ```verilog
  input  [47:0] a,         // 16 pentary digits
  ```
- **Line 122**: Possible missing semicolon
  ```verilog
  input  [47:0] b,         // 16 pentary digits
  ```
- **Line 123**: Possible missing semicolon
  ```verilog
  input  [2:0]  carry_in,  // Initial carry (usually 0)
  ```
- **Line 124**: Possible missing semicolon
  ```verilog
  output [47:0] sum,       // 16 pentary digits result
  ```
- **Line 125**: Possible missing semicolon
  ```verilog
  output [2:0]  carry_out  // Final carry out
  ```

### testbench_register_file.v

- **Line 0**: Unmatched begin/end: 28 begin, 35 end

### memristor_crossbar_fixed.v

- **Line 0**: Unmatched begin/end: 40 begin, 52 end
- **Line 29**: Possible missing semicolon
  ```verilog
  input  [7:0]  write_row,        // Row address (0-255)
  ```
- **Line 30**: Possible missing semicolon
  ```verilog
  input  [7:0]  write_col,        // Column address (0-255)
  ```
- **Line 31**: Possible missing semicolon
  ```verilog
  input  [2:0]  write_data,       // Pentary value to write
  ```
- **Line 35**: Possible missing semicolon
  ```verilog
  input         compute_enable,   // Start matrix-vector multiply
  ```
- **Line 36**: Possible missing semicolon
  ```verilog
  input  [767:0] input_vector,    // 256 pentary digits (256 × 3 bits)
  ```
- **Line 37**: Possible missing semicolon
  ```verilog
  output [767:0] output_vector,   // 256 pentary digits (256 × 3 bits)
  ```
- **Line 47**: Possible missing semicolon
  ```verilog
  output [7:0]  error_count
  ```
- **Line 72**: Possible missing semicolon
  ```verilog
  reg [2:0] reference_cells [0:255][0:4];  // 5 reference cells per row
  ```
- **Line 195**: Possible missing semicolon
  ```verilog
  assign clipped = (acc_val < -2) ? -2 :
  ```
- **Line 321**: Possible missing semicolon
  ```verilog
  input  [2:0]  program_value,    // Pentary value to program
  ```
- **Line 323**: Possible missing semicolon
  ```verilog
  input  [7:0]  read_voltage,     // Applied voltage for reading
  ```
- **Line 324**: Possible missing semicolon
  ```verilog
  output [7:0]  read_current,     // Measured current
  ```
- **Line 325**: Possible missing semicolon
  ```verilog
  output [2:0]  stored_value      // Current pentary value
  ```
- **Line 375**: Possible missing semicolon
  ```verilog
  input  [7:0]  threshold_1,      // Threshold between -2 and -1
  ```
- **Line 376**: Possible missing semicolon
  ```verilog
  input  [7:0]  threshold_2,      // Threshold between -1 and 0
  ```
- **Line 377**: Possible missing semicolon
  ```verilog
  input  [7:0]  threshold_3,      // Threshold between 0 and +1
  ```
- **Line 378**: Possible missing semicolon
  ```verilog
  input  [7:0]  threshold_4,      // Threshold between +1 and +2
  ```
- **Line 379**: Possible missing semicolon
  ```verilog
  output [2:0]  digital_value
  ```
- **Line 141**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  if (state == WRITE) begin
  ```
- **Line 162**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  if (state == COMPUTE) begin
  ```
- **Line 164**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  row = compute_counter;
  ```
- **Line 167**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  accumulator[row] = 0;
  ```
- **Line 170**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (col = 0; col < 256; col = col + 1) begin
  ```
- **Line 172**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  weight_val = decode_pentary(crossbar[row][col]);
  ```
- **Line 173**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  input_val = decode_pentary(input_vector[col*3 +: 3]);
  ```
- **Line 176**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  accumulator[row] = accumulator[row] + (weight_val * input_val);
  ```
- **Line 212**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  if (state == CALIBRATE) begin
  ```
- **Line 232**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  if (state == ERROR_CORR) begin
  ```
- **Line 233**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  error_detected = 1'b0;
  ```
- **Line 236**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (row = 0; row < 256; row = row + 1) begin
  ```
- **Line 237**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  if (ecc_data[row] != compute_ecc(row)) begin
  ```
- **Line 238**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  error_detected = 1'b1;
  ```

### register_file.v

- **Line 0**: Unmatched begin/end: 19 begin, 30 end
- **Line 6**: Possible missing semicolon
  ```verilog
  * Register file with 32 registers, each storing 16 pentary digits (48 bits).
  ```
- **Line 10**: Possible missing semicolon
  ```verilog
  *   - 32 registers (R0-R31)
  ```
- **Line 11**: Possible missing semicolon
  ```verilog
  *   - R0 is hardwired to zero
  ```
- **Line 29**: Possible missing semicolon
  ```verilog
  input  [4:0]  read_addr1,       // First read address (0-31)
  ```
- **Line 30**: Possible missing semicolon
  ```verilog
  input  [4:0]  read_addr2,       // Second read address (0-31)
  ```
- **Line 31**: Possible missing semicolon
  ```verilog
  output [47:0] read_data1,       // First read data
  ```
- **Line 32**: Possible missing semicolon
  ```verilog
  output [47:0] read_data2,       // Second read data
  ```
- **Line 35**: Possible missing semicolon
  ```verilog
  input  [4:0]  write_addr,       // Write address (0-31)
  ```
- **Line 36**: Possible missing semicolon
  ```verilog
  input  [47:0] write_data,       // Write data
  ```
- **Line 37**: Possible missing semicolon
  ```verilog
  input         write_enable      // Write enable
  ```
- **Line 69**: Possible missing semicolon
  ```verilog
  assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :  // R0 is always zero
  ```
- **Line 74**: Possible missing semicolon
  ```verilog
  assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :  // R0 is always zero
  ```
- **Line 89**: Possible missing semicolon
  ```verilog
  *   R0:  Zero (hardwired to 0)
  ```
- **Line 98**: Possible missing semicolon
  ```verilog
  *   - Status register (SR)
  ```
- **Line 99**: Possible missing semicolon
  ```verilog
  *   - Exception registers
  ```
- **Line 120**: Possible missing semicolon
  ```verilog
  input  [47:0] pc_in,            // Program counter input
  ```
- **Line 122**: Possible missing semicolon
  ```verilog
  output [47:0] pc_out,           // Program counter output
  ```
- **Line 124**: Possible missing semicolon
  ```verilog
  input  [31:0] status_in,        // Status register input
  ```
- **Line 126**: Possible missing semicolon
  ```verilog
  output [31:0] status_out,       // Status register output
  ```
- **Line 130**: Possible missing semicolon
  ```verilog
  output [47:0] debug_data
  ```
- **Line 143**: Possible missing semicolon
  ```verilog
  reg [47:0] pc;          // Program counter
  ```
- **Line 144**: Possible missing semicolon
  ```verilog
  reg [31:0] status;      // Status register
  ```
- **Line 145**: Possible missing semicolon
  ```verilog
  reg [47:0] epc;         // Exception program counter
  ```
- **Line 146**: Possible missing semicolon
  ```verilog
  reg [31:0] cause;       // Exception cause
  ```
- **Line 185**: Possible missing semicolon
  ```verilog
  assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :
  ```
- **Line 189**: Possible missing semicolon
  ```verilog
  assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :
  ```
- **Line 207**: Possible missing semicolon
  ```verilog
  * Advanced register file with scoreboarding for out-of-order execution.
  ```
- **Line 208**: Possible missing semicolon
  ```verilog
  * Tracks which registers are currently being written.
  ```
- **Line 231**: Possible missing semicolon
  ```verilog
  input  [4:0]  reserve_addr,     // Reserve register for future write
  ```
- **Line 233**: Possible missing semicolon
  ```verilog
  input  [4:0]  release_addr,     // Release register (write complete)
  ```
- **Line 234**: Possible missing semicolon
  ```verilog
  input         release_enable
  ```
- **Line 276**: Possible missing semicolon
  ```verilog
  assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :
  ```
- **Line 280**: Possible missing semicolon
  ```verilog
  assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :
  ```
- **Line 285**: Possible missing semicolon
  ```verilog
  assign read_valid1 = (read_addr1 == 5'b0) ? 1'b1 :
  ```
- **Line 289**: Possible missing semicolon
  ```verilog
  assign read_valid2 = (read_addr2 == 5'b0) ? 1'b1 :
  ```
- **Line 320**: Possible missing semicolon
  ```verilog
  input         write_enable [0:NUM_BANKS-1]
  ```
- **Line 55**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (i = 0; i < 32; i = i + 1) begin
  ```
- **Line 156**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (i = 0; i < 32; i = i + 1) begin
  ```
- **Line 251**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (i = 0; i < 32; i = i + 1) begin
  ```

### pipeline_control.v

- **Line 0**: Unmatched begin/end: 18 begin, 24 end
- **Line 37**: Possible missing semicolon
  ```verilog
  input  [4:0]  id_rs1,           // Source register 1
  ```
- **Line 38**: Possible missing semicolon
  ```verilog
  input  [4:0]  id_rs2,           // Source register 2
  ```
- **Line 39**: Possible missing semicolon
  ```verilog
  input  [4:0]  id_rd,            // Destination register
  ```
- **Line 40**: Possible missing semicolon
  ```verilog
  input         id_reg_write,     // Will write to register
  ```
- **Line 41**: Possible missing semicolon
  ```verilog
  input         id_mem_read,      // Will read from memory
  ```
- **Line 42**: Possible missing semicolon
  ```verilog
  input         id_branch,        // Is branch instruction
  ```
- **Line 52**: Possible missing semicolon
  ```verilog
  output [1:0]  ex_forward_a,     // Forward control for operand A
  ```
- **Line 53**: Possible missing semicolon
  ```verilog
  output [1:0]  ex_forward_b,     // Forward control for operand B
  ```
- **Line 71**: Possible missing semicolon
  ```verilog
  output [47:0] predict_target
  ```
- **Line 83**: Possible missing semicolon
  ```verilog
  assign load_use_hazard = ex_mem_read &&
  ```
- **Line 96**: Possible missing semicolon
  ```verilog
  assign forward_mem_a = mem_reg_write &&
  ```
- **Line 100**: Possible missing semicolon
  ```verilog
  assign forward_mem_b = mem_reg_write &&
  ```
- **Line 106**: Possible missing semicolon
  ```verilog
  assign forward_wb_a = wb_reg_write &&
  ```
- **Line 111**: Possible missing semicolon
  ```verilog
  assign forward_wb_b = wb_reg_write &&
  ```
- **Line 120**: Possible missing semicolon
  ```verilog
  assign ex_forward_a = forward_mem_a ? 2'b01 :
  ```
- **Line 124**: Possible missing semicolon
  ```verilog
  assign ex_forward_b = forward_mem_b ? 2'b01 :
  ```
- **Line 155**: Possible missing semicolon
  ```verilog
  assign predict_target = if_pc + 48'd4;  // Next sequential instruction
  ```
- **Line 181**: Possible missing semicolon
  ```verilog
  output reg [31:0] id_instruction
  ```
- **Line 227**: Possible missing semicolon
  ```verilog
  output reg        ex_branch
  ```
- **Line 281**: Possible missing semicolon
  ```verilog
  output reg        mem_mem_write
  ```
- **Line 319**: Possible missing semicolon
  ```verilog
  output reg        wb_mem_to_reg
  ```
- **Line 353**: Possible missing semicolon
  ```verilog
  output        predict_taken
  ```
- **Line 359**: Possible missing semicolon
  ```verilog
  assign index = pc[9:2];  // Use bits [9:2] as index
  ```
- **Line 368**: Blocking assignment (=) in sequential always block, should use non-blocking (<=)
  ```verilog
  for (i = 0; i < 256; i = i + 1) begin
  ```

### pentary_alu_fixed.v

- **Line 0**: Unmatched begin/end: 10 begin, 23 end
- **Line 30**: Possible missing semicolon
  ```verilog
  input  [47:0] operand_a,      // 16 pentary digits (48 bits)
  ```
- **Line 31**: Possible missing semicolon
  ```verilog
  input  [47:0] operand_b,      // 16 pentary digits (48 bits)
  ```
- **Line 32**: Possible missing semicolon
  ```verilog
  input  [2:0]  opcode,         // Operation code
  ```
- **Line 33**: Possible missing semicolon
  ```verilog
  output [47:0] result,         // 16 pentary digits (48 bits)
  ```
- **Line 34**: Possible missing semicolon
  ```verilog
  output        zero_flag,      // Set if result == 0
  ```
- **Line 35**: Possible missing semicolon
  ```verilog
  output        negative_flag,  // Set if result < 0
  ```
- **Line 36**: Possible missing semicolon
  ```verilog
  output        overflow_flag,  // Set if overflow occurred
  ```
- **Line 37**: Possible missing semicolon
  ```verilog
  output        equal_flag,     // Set if operand_a == operand_b
  ```
- **Line 38**: Possible missing semicolon
  ```verilog
  output        greater_flag    // Set if operand_a > operand_b
  ```
- **Line 67**: Possible missing semicolon
  ```verilog
  .output_val(neg_b)
  ```
- **Line 82**: Possible missing semicolon
  ```verilog
  .output_val(mul2_result)
  ```
- **Line 89**: Possible missing semicolon
  ```verilog
  .output_val(div2_result)
  ```
- **Line 95**: Possible missing semicolon
  ```verilog
  .output_val(neg_result)
  ```
- **Line 101**: Possible missing semicolon
  ```verilog
  .output_val(abs_result)
  ```
- **Line 121**: Possible missing semicolon
  ```verilog
  3'b110: result_reg = sub_result;  // CMP uses subtraction
  ```
- **Line 146**: Possible missing semicolon
  ```verilog
  assign overflow_flag = (opcode == 3'b000 && add_carry != 3'b010) ||
  ```
- **Line 179**: Possible missing semicolon
  ```verilog
  output [47:0] output_val
  ```
- **Line 184**: Possible missing semicolon
  ```verilog
  assign output_val[i*3 +: 3] =
  ```
- **Line 185**: Possible missing semicolon
  ```verilog
  (input_val[i*3 +: 3] == 3'b000) ? 3'b100 :  // -2 -> +2
  ```
- **Line 186**: Possible missing semicolon
  ```verilog
  (input_val[i*3 +: 3] == 3'b001) ? 3'b011 :  // -1 -> +1
  ```
- **Line 187**: Possible missing semicolon
  ```verilog
  (input_val[i*3 +: 3] == 3'b010) ? 3'b010 :  //  0 ->  0
  ```
- **Line 188**: Possible missing semicolon
  ```verilog
  (input_val[i*3 +: 3] == 3'b011) ? 3'b001 :  // +1 -> -1
  ```
- **Line 189**: Possible missing semicolon
  ```verilog
  (input_val[i*3 +: 3] == 3'b100) ? 3'b000 :  // +2 -> -2
  ```
- **Line 198**: Possible missing semicolon
  ```verilog
  output        is_zero
  ```
- **Line 209**: Possible missing semicolon
  ```verilog
  assign is_zero = &digit_is_zero;  // All digits must be zero
  ```
- **Line 215**: Possible missing semicolon
  ```verilog
  output        is_negative
  ```
- **Line 225**: Possible missing semicolon
  ```verilog
  is_neg = (input_val[i*3 +: 3] == 3'b000) ||
  ```
- **Line 238**: Possible missing semicolon
  ```verilog
  output [47:0] output_val
  ```
- **Line 250**: Possible missing semicolon
  ```verilog
  .output_val(negated)
  ```
- **Line 260**: Possible missing semicolon
  ```verilog
  output [47:0] result
  ```
- **Line 268**: Possible missing semicolon
  ```verilog
  .output_val(neg_b)
  ```
- **Line 292**: Possible missing semicolon
  ```verilog
  output [47:0] output_val
  ```
- **Line 316**: Possible missing semicolon
  ```verilog
  output [47:0] output_val
  ```

## Warnings ⚠️

### testbench_pentary_quantizer.v

- **Line 0**: Register "scale" may not be initialized
- **Line 0**: Register "input_value" may not be initialized
- **Line 0**: Register "zero_point" may not be initialized
- **Line 0**: Register "input_values" may not be initialized
- **Line 39**: Signal "to_fixed" has multiple drivers at lines: [39, 285]
- **Line 48**: Signal "decode_pent" has multiple drivers at lines: [48, 49, 50, 51, 52, 53]
- **Line 64**: Signal "test_count" has multiple drivers at lines: [64, 245, 294, 333]
- **Line 65**: Signal "pass_count" has multiple drivers at lines: [65, 248, 295, 334]
- **Line 66**: Signal "fail_count" has multiple drivers at lines: [66, 91, 251, 296, 309]
- **Line 84**: Signal "n" has multiple drivers at lines: [84, 302]
  ... and 10 more warnings

### mmu_interrupt.v

- **Line 135**: Width mismatch: pte_addr is 48-bit but assigned 3-bit literal
  ```verilog
  pte_addr <= page_table_base + {virtual_page_number[35:27], 3'b0};
  ```
- **Line 143**: Width mismatch: pte_addr is 48-bit but assigned 3-bit literal
  ```verilog
  pte_addr <= mem_data + {virtual_page_number[26:18], 3'b0};
  ```
- **Line 151**: Width mismatch: pte_addr is 48-bit but assigned 3-bit literal
  ```verilog
  pte_addr <= mem_data + {virtual_page_number[17:9], 3'b0};
  ```
- **Line 0**: Register "tlb_lru" may not be initialized
- **Line 0**: Register "walked_ppn" may not be initialized
- **Line 0**: Register "tlb_read" may not be initialized
- **Line 0**: Register "tlb_write" may not be initialized
- **Line 0**: Register "pte_addr" may not be initialized
- **Line 0**: Register "tlb_vpn" may not be initialized
- **Line 0**: Register "pte_data" may not be initialized
  ... and 26 more warnings

### testbench_pentary_alu.v

- **Line 117**: Width mismatch: operand_a is 48-bit but assigned 3-bit literal
  ```verilog
  operand_a = {16{3'b011}};  // All +1
  ```
- **Line 118**: Width mismatch: operand_b is 48-bit but assigned 3-bit literal
  ```verilog
  operand_b = {16{3'b011}};  // All +1
  ```
- **Line 130**: Width mismatch: operand_a is 48-bit but assigned 3-bit literal
  ```verilog
  operand_a = {16{3'b010}};
  ```
- **Line 131**: Width mismatch: operand_b is 48-bit but assigned 3-bit literal
  ```verilog
  operand_b = {16{3'b010}};
  ```
- **Line 143**: Width mismatch: operand_a is 48-bit but assigned 3-bit literal
  ```verilog
  operand_a = {16{3'b100}};  // All +2
  ```
- **Line 144**: Width mismatch: operand_b is 48-bit but assigned 3-bit literal
  ```verilog
  operand_b = {16{3'b000}};  // All -2
  ```
- **Line 164**: Width mismatch: operand_a is 48-bit but assigned 3-bit literal
  ```verilog
  operand_a = {16{3'b100}};  // All +2
  ```
- **Line 165**: Width mismatch: operand_b is 48-bit but assigned 3-bit literal
  ```verilog
  operand_b = {16{3'b011}};  // All +1
  ```
- **Line 177**: Width mismatch: operand_a is 48-bit but assigned 3-bit literal
  ```verilog
  operand_a = {16{3'b010}};
  ```
- **Line 178**: Width mismatch: operand_b is 48-bit but assigned 3-bit literal
  ```verilog
  operand_b = {16{3'b010}};
  ```
  ... and 42 more warnings

### cache_hierarchy.v

- **Line 0**: Register "data" may not be initialized
- **Line 0**: Register "dirty" may not be initialized
- **Line 0**: Register "lru" may not be initialized
- **Line 0**: Register "valid" may not be initialized
- **Line 0**: Register "tags" may not be initialized
- **Line 35**: Signal "CACHE_SIZE" has multiple drivers at lines: [35, 156, 313]
- **Line 36**: Signal "LINE_SIZE" has multiple drivers at lines: [36, 157, 314]
- **Line 37**: Signal "WAYS" has multiple drivers at lines: [37, 158, 315]
- **Line 38**: Signal "NUM_SETS" has multiple drivers at lines: [38, 159, 316]
- **Line 45**: Signal "offset" has multiple drivers at lines: [45, 166, 334]
  ... and 23 more warnings

### instruction_decoder.v

- **Line 263**: Width mismatch: immediate_reg is 48-bit but assigned 1-bit literal
  ```verilog
  immediate_reg = {{34{imm_field[12]}}, imm_field, 1'b0};
  ```
- **Line 268**: Width mismatch: immediate_reg is 48-bit but assigned 1-bit literal
  ```verilog
  immediate_reg = {{34{imm_field[12]}}, imm_field, 1'b0};
  ```
- **Line 0**: Register "immediate_reg" may not be initialized
- **Line 114**: Signal "alu_op_reg" has multiple drivers at lines: [114, 129, 136, 143, 150, 157, 164, 171, 179, 189, 197, 204, 224, 231]
- **Line 115**: Signal "alu_src_reg" has multiple drivers at lines: [115, 172, 180, 190]
- **Line 116**: Signal "reg_write_reg" has multiple drivers at lines: [116, 130, 137, 144, 151, 158, 165, 173, 183, 218, 225, 232]
- **Line 117**: Signal "mem_read_reg" has multiple drivers at lines: [117, 181]
- **Line 118**: Signal "mem_write_reg" has multiple drivers at lines: [118, 191]
- **Line 119**: Signal "mem_to_reg_reg" has multiple drivers at lines: [119, 182]
- **Line 120**: Signal "branch_reg" has multiple drivers at lines: [120, 198, 205]
  ... and 10 more warnings

### pentary_quantizer_fixed.v

- **Line 0**: Register "signed" may not be initialized
- **Line 33**: Signal "centered" has multiple drivers at lines: [33, 181]
- **Line 38**: Signal "numerator" has multiple drivers at lines: [38, 185, 273]
- **Line 41**: Signal "scaled" has multiple drivers at lines: [41, 150, 188]
- **Line 46**: Signal "rounded" has multiple drivers at lines: [46, 194, 195, 196, 197, 198, 251]
- **Line 50**: Signal "clipped" has multiple drivers at lines: [50, 211]
- **Line 58**: Signal "quantized_reg" has multiple drivers at lines: [58, 59, 60, 61, 62, 63, 217, 218, 219, 220, 221, 222]
- **Line 67**: Signal "quantized" has multiple drivers at lines: [67, 226]
- **Line 132**: Signal "decoded" has multiple drivers at lines: [132, 133, 134, 135, 136, 137]
- **Line 147**: Signal "product" has multiple drivers at lines: [147, 262]
  ... and 4 more warnings

### testbench_memristor_crossbar.v

- **Line 167**: Width mismatch: input_vector is 768-bit but assigned 3-bit literal
  ```verilog
  input_vector[i*3 +: 3] = 3'b011;  // +1
  ```
- **Line 217**: Width mismatch: input_vector is 768-bit but assigned 3-bit literal
  ```verilog
  input_vector[i*3 +: 3] = 3'b011;  // All +1
  ```
- **Line 0**: Register "program_value" may not be initialized
- **Line 0**: Register "write_data" may not be initialized
- **Line 0**: Register "write_row" may not be initialized
- **Line 0**: Register "calibrate_enable" may not be initialized
- **Line 0**: Register "read_voltage" may not be initialized
- **Line 0**: Register "write_col" may not be initialized
- **Line 0**: Register "calibrate_row" may not be initialized
- **Line 0**: Register "clk" may not be initialized
  ... and 31 more warnings

### pentary_chip_design.v

- **Line 0**: Register "registers" may not be initialized
- **Line 0**: Register "crossbar" may not be initialized
- **Line 13**: Signal "state" has multiple drivers at lines: [13, 292, 298, 301, 307, 315, 400]
- **Line 25**: Signal "000" has multiple drivers at lines: [25, 72]
- **Line 26**: Signal "001" has multiple drivers at lines: [26, 73]
- **Line 27**: Signal "010" has multiple drivers at lines: [27, 74]
- **Line 28**: Signal "011" has multiple drivers at lines: [28, 75]
- **Line 29**: Signal "100" has multiple drivers at lines: [29, 76]
- **Line 50**: Signal "opcode" has multiple drivers at lines: [50, 51, 52, 53, 107, 108, 109]
- **Line 67**: Signal "result" has multiple drivers at lines: [67, 106, 209]
  ... and 18 more warnings

### testbench_pentary_adder.v

- **Line 298**: Width mismatch: a is 48-bit but assigned 3-bit literal
  ```verilog
  a = {16{3'b010}};
  ```
- **Line 0**: Register "a" may not be initialized
- **Line 0**: Register "carry_in" may not be initialized
- **Line 39**: Signal "decode_pent" has multiple drivers at lines: [39, 40, 41, 42, 43, 44]
- **Line 54**: Signal "encode_pent" has multiple drivers at lines: [54, 55, 56, 57, 58, 59]
- **Line 70**: Signal "test_count" has multiple drivers at lines: [70, 136, 205, 249, 288, 302, 322, 340]
- **Line 71**: Signal "pass_count" has multiple drivers at lines: [71, 139, 208, 250, 290, 304, 324, 341]
- **Line 72**: Signal "fail_count" has multiple drivers at lines: [72, 88, 141, 211, 251, 270, 293, 307]
- **Line 81**: Signal "n" has multiple drivers at lines: [81, 263]
- **Line 111**: Signal "a" has multiple drivers at lines: [111, 183, 284, 298, 318, 336]
  ... and 19 more warnings

### pentary_adder_fixed.v

- **Line 0**: Register "signed" may not be initialized
- **Line 0**: Register "sum_reg" may not be initialized
- **Line 25**: Signal "sum" has multiple drivers at lines: [25, 99]
- **Line 46**: Signal "a_val" has multiple drivers at lines: [46, 47, 48, 49, 50, 51]
- **Line 55**: Signal "b_val" has multiple drivers at lines: [55, 56, 57, 58, 59, 60]
- **Line 64**: Signal "carry_val" has multiple drivers at lines: [64, 65, 66, 67, 68, 69]
- **Line 75**: Signal "temp_sum" has multiple drivers at lines: [75, 79, 82]
- **Line 80**: Signal "carry_reg" has multiple drivers at lines: [80, 83, 85]
- **Line 90**: Signal "sum_reg" has multiple drivers at lines: [90, 91, 92, 93, 94, 95]
- **Line 100**: Signal "carry_out" has multiple drivers at lines: [100, 149]
  ... and 3 more warnings

### testbench_register_file.v

- **Line 0**: Register "read_addr1" may not be initialized
- **Line 0**: Register "write_data" may not be initialized
- **Line 0**: Register "expected" may not be initialized
- **Line 0**: Register "clk" may not be initialized
- **Line 0**: Register "reset" may not be initialized
- **Line 0**: Register "write_addr" may not be initialized
- **Line 0**: Register "write_enable" may not be initialized
- **Line 0**: Register "read_addr2" may not be initialized
- **Line 49**: Signal "clk" has multiple drivers at lines: [49, 50]
- **Line 59**: Signal "test_count" has multiple drivers at lines: [59, 127, 157, 178, 206, 239, 263, 300]
  ... and 15 more warnings

### memristor_crossbar_fixed.v

- **Line 0**: Register "crossbar" may not be initialized
- **Line 0**: Register "reference_cells" may not be initialized
- **Line 0**: Register "calibration_state" may not be initialized
- **Line 0**: Register "parity" may not be initialized
- **Line 0**: Register "ecc_data" may not be initialized
- **Line 0**: Register "signed" may not be initialized
- **Line 0**: Register "write_count" may not be initialized
- **Line 86**: Signal "state" has multiple drivers at lines: [86, 90, 92, 141, 162, 212, 232]
- **Line 87**: Signal "compute_counter" has multiple drivers at lines: [87, 94, 96]
- **Line 88**: Signal "error_counter" has multiple drivers at lines: [88, 239]
  ... and 14 more warnings

### register_file.v

- **Line 263**: Width mismatch: scoreboard is 32-bit but assigned 1-bit literal
  ```verilog
  scoreboard[reserve_addr] <= 1'b1;
  ```
- **Line 267**: Width mismatch: scoreboard is 32-bit but assigned 1-bit literal
  ```verilog
  scoreboard[release_addr] <= 1'b0;
  ```
- **Line 0**: Register "pc" may not be initialized
- **Line 0**: Register "epc" may not be initialized
- **Line 0**: Register "cause" may not be initialized
- **Line 0**: Register "registers" may not be initialized
- **Line 0**: Register "status" may not be initialized
- **Line 0**: Register "scoreboard" may not be initialized
- **Line 55**: Signal "i" has multiple drivers at lines: [55, 156, 251, 326]
- **Line 69**: Signal "read_data1" has multiple drivers at lines: [69, 185, 276]
  ... and 8 more warnings

### pipeline_control.v

- **Line 0**: Register "bht" may not be initialized
- **Line 84**: Signal "ex_rd" has multiple drivers at lines: [84, 237, 251]
- **Line 98**: Signal "mem_rd" has multiple drivers at lines: [98, 102, 109, 114, 287, 294]
- **Line 108**: Signal "wb_rd" has multiple drivers at lines: [108, 113, 325, 331]
- **Line 154**: Signal "predict_taken" has multiple drivers at lines: [154, 362]
- **Line 185**: Signal "id_pc" has multiple drivers at lines: [185, 188]
- **Line 186**: Signal "id_instruction" has multiple drivers at lines: [186, 189]
- **Line 231**: Signal "ex_pc" has multiple drivers at lines: [231, 245]
- **Line 232**: Signal "ex_read_data1" has multiple drivers at lines: [232, 246]
- **Line 233**: Signal "ex_read_data2" has multiple drivers at lines: [233, 247]
  ... and 19 more warnings

### pentary_alu_fixed.v

- **Line 305**: Width mismatch: shifted is 48-bit but assigned 3-bit literal
  ```verilog
  shifted = {shifted[44:0], 3'b010};  // Append zero at LSB
  ```
- **Line 328**: Width mismatch: shifted is 48-bit but assigned 3-bit literal
  ```verilog
  shifted = {3'b010, shifted[47:3]};  // Prepend zero at MSB
  ```
- **Line 9**: Signal "opcode" has multiple drivers at lines: [9, 10, 11, 12, 13, 14, 15, 16, 147]
- **Line 34**: Signal "result" has multiple drivers at lines: [34, 127, 285]
- **Line 58**: Signal "carry" has multiple drivers at lines: [58, 73]
- **Line 115**: Signal "result_reg" has multiple drivers at lines: [115, 116, 117, 118, 119, 120, 121, 122, 123]
- **Line 183**: Signal "i" has multiple drivers at lines: [183, 204, 223, 227, 303, 326]
- **Line 222**: Signal "is_neg" has multiple drivers at lines: [222, 225]
- **Line 253**: Signal "output_val" has multiple drivers at lines: [253, 309, 332]
- **Line 294**: Signal "positions" has multiple drivers at lines: [294, 318]
  ... and 5 more warnings

## Design Metrics

| File | Lines | Modules | Always Blocks | Registers | Wires |
|------|-------|---------|---------------|-----------|-------|
| pentary_core_integrated.v | 365 | 1 | 0 | 0 | 80 |
| testbench_pentary_quantizer.v | 235 | 2 | 0 | 6 | 2 |
| mmu_interrupt.v | 339 | 4 | 6 | 24 | 5 |
| testbench_pentary_alu.v | 330 | 1 | 0 | 4 | 6 |
| cache_hierarchy.v | 400 | 4 | 8 | 28 | 18 |
| instruction_decoder.v | 340 | 3 | 4 | 17 | 7 |
| pentary_quantizer_fixed.v | 213 | 8 | 4 | 4 | 16 |
| testbench_memristor_crossbar.v | 273 | 2 | 0 | 16 | 7 |
| pentary_chip_design.v | 288 | 7 | 3 | 7 | 12 |
| testbench_pentary_adder.v | 258 | 2 | 0 | 3 | 3 |
| pentary_adder_fixed.v | 124 | 2 | 2 | 3 | 1 |
| testbench_register_file.v | 238 | 1 | 0 | 8 | 2 |
| memristor_crossbar_fixed.v | 270 | 3 | 8 | 16 | 2 |
| register_file.v | 228 | 4 | 5 | 8 | 0 |
| pipeline_control.v | 295 | 6 | 5 | 27 | 5 |
| pentary_alu_fixed.v | 256 | 8 | 4 | 4 | 20 |
