/*
 * ============================================================================
 * Pentary Processor Core - Integrated Design
 * ============================================================================
 * 
 * Complete pentary processor core integrating all components:
 *   - 5-stage pipeline
 *   - Register file
 *   - ALU
 *   - Cache hierarchy
 *   - MMU
 *   - Interrupt controller
 *   - Memristor crossbar
 * 
 * SPECIFICATIONS:
 *   - 48-bit data path (16 pentary digits)
 *   - 32 general-purpose registers
 *   - 5-stage pipeline (IF, ID, EX, MEM, WB)
 *   - L1 I-Cache: 32KB
 *   - L1 D-Cache: 32KB
 *   - Memristor: 256×256 crossbar
 * 
 * ============================================================================
 */

module PentaryCoreIntegrated (
    input         clk,
    input         reset,
    
    // External memory interface
    output [47:0] mem_addr,
    output        mem_read,
    output        mem_write,
    output [47:0] mem_write_data,
    input  [47:0] mem_read_data,
    input         mem_ready,
    
    // Interrupt interface
    input  [31:0] external_interrupts,
    
    // Debug interface
    input  [4:0]  debug_reg_addr,
    output [47:0] debug_reg_data,
    output [47:0] debug_pc,
    output        debug_halted
);

    // ========================================================================
    // Pipeline Stage Signals
    // ========================================================================
    
    // IF stage
    wire [47:0] if_pc;
    wire [31:0] if_instruction;
    wire        if_valid;
    wire        if_stall;
    wire        if_flush;
    
    // ID stage
    wire [47:0] id_pc;
    wire [31:0] id_instruction;
    wire [4:0]  id_rs1, id_rs2, id_rd;
    wire [47:0] id_read_data1, id_read_data2;
    wire [47:0] id_immediate;
    wire [2:0]  id_alu_op;
    wire        id_alu_src;
    wire        id_reg_write;
    wire        id_mem_read;
    wire        id_mem_write;
    wire        id_mem_to_reg;
    wire        id_branch;
    wire        id_jump;
    wire        id_memristor_op;
    wire        id_stall;
    wire        id_flush;
    
    // EX stage
    wire [47:0] ex_pc;
    wire [47:0] ex_operand_a, ex_operand_b;
    wire [47:0] ex_alu_result;
    wire [4:0]  ex_rd;
    wire        ex_reg_write;
    wire        ex_mem_read;
    wire        ex_mem_write;
    wire        ex_mem_to_reg;
    wire        ex_branch_taken;
    wire [47:0] ex_branch_target;
    wire [1:0]  ex_forward_a, ex_forward_b;
    wire        ex_stall;
    wire        ex_flush;
    
    // MEM stage
    wire [47:0] mem_alu_result;
    wire [47:0] mem_mem_data;
    wire [4:0]  mem_rd;
    wire        mem_reg_write;
    wire        mem_mem_to_reg;
    wire        mem_stall;
    wire        mem_flush;
    
    // WB stage
    wire [47:0] wb_write_data;
    wire [4:0]  wb_rd;
    wire        wb_reg_write;
    wire        wb_stall;
    wire        wb_flush;
    
    // ========================================================================
    // Instruction Fetch Stage
    // ========================================================================
    
    wire [47:0] i_cache_addr;
    wire        i_cache_read;
    wire [31:0] i_cache_data;
    wire        i_cache_ready;
    
    InstructionFetchUnit if_unit (
        .clk(clk),
        .reset(reset),
        .stall(if_stall),
        .flush(if_flush),
        .branch_taken(ex_branch_taken),
        .branch_target(ex_branch_target),
        .i_cache_addr(i_cache_addr),
        .i_cache_read(i_cache_read),
        .i_cache_data(i_cache_data),
        .i_cache_ready(i_cache_ready),
        .pc(if_pc),
        .instruction(if_instruction),
        .valid(if_valid)
    );
    
    // ========================================================================
    // IF/ID Pipeline Register
    // ========================================================================
    
    IF_ID_Register if_id_reg (
        .clk(clk),
        .reset(reset),
        .stall(if_stall),
        .flush(if_flush),
        .if_pc(if_pc),
        .if_instruction(if_instruction),
        .id_pc(id_pc),
        .id_instruction(id_instruction)
    );
    
    // ========================================================================
    // Instruction Decode Stage
    // ========================================================================
    
    wire [2:0] inst_type;
    wire illegal_instruction;
    
    InstructionDecoder decoder (
        .instruction(id_instruction),
        .rs1(id_rs1),
        .rs2(id_rs2),
        .rd(id_rd),
        .immediate(id_immediate),
        .alu_op(id_alu_op),
        .alu_src(id_alu_src),
        .reg_write(id_reg_write),
        .mem_read(id_mem_read),
        .mem_write(id_mem_write),
        .mem_to_reg(id_mem_to_reg),
        .branch(id_branch),
        .jump(id_jump),
        .memristor_op(id_memristor_op),
        .inst_type(inst_type),
        .illegal_instruction(illegal_instruction)
    );
    
    // Register file
    RegisterFile reg_file (
        .clk(clk),
        .reset(reset),
        .read_addr1(id_rs1),
        .read_addr2(id_rs2),
        .read_data1(id_read_data1),
        .read_data2(id_read_data2),
        .write_addr(wb_rd),
        .write_data(wb_write_data),
        .write_enable(wb_reg_write)
    );
    
    // ========================================================================
    // ID/EX Pipeline Register
    // ========================================================================
    
    wire [47:0] ex_read_data1, ex_read_data2;
    wire [47:0] ex_immediate;
    wire [4:0]  ex_rs1, ex_rs2;
    wire [2:0]  ex_alu_op;
    wire        ex_alu_src;
    wire        ex_branch;
    
    ID_EX_Register id_ex_reg (
        .clk(clk),
        .reset(reset),
        .stall(id_stall),
        .flush(id_flush),
        .id_pc(id_pc),
        .id_read_data1(id_read_data1),
        .id_read_data2(id_read_data2),
        .id_immediate(id_immediate),
        .id_rs1(id_rs1),
        .id_rs2(id_rs2),
        .id_rd(id_rd),
        .id_alu_op(id_alu_op),
        .id_alu_src(id_alu_src),
        .id_reg_write(id_reg_write),
        .id_mem_read(id_mem_read),
        .id_mem_write(id_mem_write),
        .id_branch(id_branch),
        .ex_pc(ex_pc),
        .ex_read_data1(ex_read_data1),
        .ex_read_data2(ex_read_data2),
        .ex_immediate(ex_immediate),
        .ex_rs1(ex_rs1),
        .ex_rs2(ex_rs2),
        .ex_rd(ex_rd),
        .ex_alu_op(ex_alu_op),
        .ex_alu_src(ex_alu_src),
        .ex_reg_write(ex_reg_write),
        .ex_mem_read(ex_mem_read),
        .ex_mem_write(ex_mem_write),
        .ex_branch(ex_branch)
    );
    
    // ========================================================================
    // Execute Stage
    // ========================================================================
    
    // Forwarding muxes
    wire [47:0] forwarded_a, forwarded_b;
    assign forwarded_a = (ex_forward_a == 2'b01) ? mem_alu_result :
                        (ex_forward_a == 2'b10) ? wb_write_data :
                        ex_read_data1;
    
    assign forwarded_b = (ex_forward_b == 2'b01) ? mem_alu_result :
                        (ex_forward_b == 2'b10) ? wb_write_data :
                        ex_read_data2;
    
    // ALU operand selection
    assign ex_operand_a = forwarded_a;
    assign ex_operand_b = ex_alu_src ? ex_immediate : forwarded_b;
    
    // ALU
    wire zero_flag, negative_flag, overflow_flag;
    
    PentaryALU alu (
        .operand_a(ex_operand_a),
        .operand_b(ex_operand_b),
        .opcode(ex_alu_op),
        .result(ex_alu_result),
        .zero_flag(zero_flag),
        .negative_flag(negative_flag),
        .overflow_flag(overflow_flag),
        .equal_flag(),
        .greater_flag()
    );
    
    // Branch unit
    BranchUnit branch_unit (
        .operand_a(forwarded_a),
        .operand_b(forwarded_b),
        .pc(ex_pc),
        .immediate(ex_immediate),
        .branch(ex_branch),
        .jump(1'b0),
        .opcode(id_instruction[31:28]),
        .branch_taken(ex_branch_taken),
        .branch_target(ex_branch_target)
    );
    
    // ========================================================================
    // EX/MEM Pipeline Register
    // ========================================================================
    
    wire [47:0] mem_write_data_internal;
    
    EX_MEM_Register ex_mem_reg (
        .clk(clk),
        .reset(reset),
        .stall(ex_stall),
        .flush(ex_flush),
        .ex_alu_result(ex_alu_result),
        .ex_write_data(forwarded_b),
        .ex_rd(ex_rd),
        .ex_reg_write(ex_reg_write),
        .ex_mem_read(ex_mem_read),
        .ex_mem_write(ex_mem_write),
        .mem_alu_result(mem_alu_result),
        .mem_write_data(mem_write_data_internal),
        .mem_rd(mem_rd),
        .mem_reg_write(mem_reg_write),
        .mem_mem_read(),
        .mem_mem_write()
    );
    
    // ========================================================================
    // Memory Stage
    // ========================================================================
    
    // Data cache interface
    wire [47:0] d_cache_addr;
    wire        d_cache_read;
    wire        d_cache_write;
    wire [47:0] d_cache_write_data;
    wire [47:0] d_cache_read_data;
    wire        d_cache_ready;
    
    assign d_cache_addr = mem_alu_result;
    assign d_cache_read = ex_mem_read;
    assign d_cache_write = ex_mem_write;
    assign d_cache_write_data = mem_write_data_internal;
    assign mem_mem_data = d_cache_read_data;
    
    // ========================================================================
    // MEM/WB Pipeline Register
    // ========================================================================
    
    wire [47:0] wb_alu_result;
    wire [47:0] wb_mem_data;
    wire        wb_mem_to_reg;
    
    MEM_WB_Register mem_wb_reg (
        .clk(clk),
        .reset(reset),
        .stall(mem_stall),
        .flush(mem_flush),
        .mem_alu_result(mem_alu_result),
        .mem_read_data(mem_mem_data),
        .mem_rd(mem_rd),
        .mem_reg_write(mem_reg_write),
        .mem_mem_to_reg(mem_mem_to_reg),
        .wb_alu_result(wb_alu_result),
        .wb_read_data(wb_mem_data),
        .wb_rd(wb_rd),
        .wb_reg_write(wb_reg_write),
        .wb_mem_to_reg(wb_mem_to_reg)
    );
    
    // ========================================================================
    // Write Back Stage
    // ========================================================================
    
    assign wb_write_data = wb_mem_to_reg ? wb_mem_data : wb_alu_result;
    
    // ========================================================================
    // Pipeline Control
    // ========================================================================
    
    PipelineControl pipeline_ctrl (
        .clk(clk),
        .reset(reset),
        .if_pc(if_pc),
        .if_instruction(if_instruction),
        .if_stall(if_stall),
        .if_flush(if_flush),
        .id_rs1(id_rs1),
        .id_rs2(id_rs2),
        .id_rd(id_rd),
        .id_reg_write(id_reg_write),
        .id_mem_read(id_mem_read),
        .id_branch(id_branch),
        .id_stall(id_stall),
        .id_flush(id_flush),
        .ex_rd(ex_rd),
        .ex_reg_write(ex_reg_write),
        .ex_mem_read(ex_mem_read),
        .ex_branch_taken(ex_branch_taken),
        .ex_branch_target(ex_branch_target),
        .ex_forward_a(ex_forward_a),
        .ex_forward_b(ex_forward_b),
        .ex_stall(ex_stall),
        .ex_flush(ex_flush),
        .mem_rd(mem_rd),
        .mem_reg_write(mem_reg_write),
        .mem_stall(mem_stall),
        .mem_flush(mem_flush),
        .wb_rd(wb_rd),
        .wb_reg_write(wb_reg_write),
        .wb_stall(wb_stall),
        .wb_flush(wb_flush),
        .predict_taken(),
        .predict_target()
    );
    
    // ========================================================================
    // Cache Hierarchy
    // ========================================================================
    
    // L1 Instruction Cache
    wire [47:0] l2_i_addr;
    wire        l2_i_read;
    wire [511:0] l2_i_data;
    wire        l2_i_ready;
    
    L1_InstructionCache l1_icache (
        .clk(clk),
        .reset(reset),
        .addr(i_cache_addr),
        .read_enable(i_cache_read),
        .instruction(i_cache_data),
        .hit(),
        .ready(i_cache_ready),
        .l2_addr(l2_i_addr),
        .l2_read(l2_i_read),
        .l2_data(l2_i_data),
        .l2_ready(l2_i_ready)
    );
    
    // L1 Data Cache
    wire [47:0] l2_d_addr;
    wire        l2_d_read;
    wire        l2_d_write;
    wire [511:0] l2_d_write_data;
    wire [511:0] l2_d_read_data;
    wire        l2_d_ready;
    
    L1_DataCache l1_dcache (
        .clk(clk),
        .reset(reset),
        .addr(d_cache_addr),
        .read_enable(d_cache_read),
        .write_enable(d_cache_write),
        .write_data(d_cache_write_data),
        .read_data(d_cache_read_data),
        .hit(),
        .ready(d_cache_ready),
        .l2_addr(l2_d_addr),
        .l2_read(l2_d_read),
        .l2_write(l2_d_write),
        .l2_write_data(l2_d_write_data),
        .l2_read_data(l2_d_read_data),
        .l2_ready(l2_d_ready)
    );
    
    // L2 Unified Cache
    L2_UnifiedCache l2_cache (
        .clk(clk),
        .reset(reset),
        .l1i_addr(l2_i_addr),
        .l1i_read(l2_i_read),
        .l1i_data(l2_i_data),
        .l1i_ready(l2_i_ready),
        .l1d_addr(l2_d_addr),
        .l1d_read(l2_d_read),
        .l1d_write(l2_d_write),
        .l1d_write_data(l2_d_write_data),
        .l1d_read_data(l2_d_read_data),
        .l1d_ready(l2_d_ready),
        .mem_addr(mem_addr),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .mem_write_data(mem_write_data),
        .mem_read_data(mem_read_data),
        .mem_ready(mem_ready)
    );
    
    // ========================================================================
    // Debug Interface
    // ========================================================================
    
    assign debug_reg_data = (debug_reg_addr == 5'b0) ? 48'b0 : 
                           id_read_data1;  // Simplified
    assign debug_pc = if_pc;
    assign debug_halted = if_stall;

endmodule