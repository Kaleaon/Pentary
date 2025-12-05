/*
 * ============================================================================
 * Pentary Pipeline Control
 * ============================================================================
 * 
 * 5-stage pipeline control for pentary processor.
 * Handles hazard detection, data forwarding, and stall logic.
 * 
 * PIPELINE STAGES:
 *   1. IF  - Instruction Fetch
 *   2. ID  - Instruction Decode
 *   3. EX  - Execute
 *   4. MEM - Memory Access
 *   5. WB  - Write Back
 * 
 * FEATURES:
 *   - Data hazard detection
 *   - Control hazard detection
 *   - Data forwarding
 *   - Pipeline stalling
 *   - Branch prediction
 * 
 * ============================================================================
 */

module PipelineControl (
    input         clk,
    input         reset,
    
    // Instruction Fetch stage
    input  [47:0] if_pc,
    input  [31:0] if_instruction,
    output        if_stall,
    output        if_flush,
    
    // Instruction Decode stage
    input  [4:0]  id_rs1,           // Source register 1
    input  [4:0]  id_rs2,           // Source register 2
    input  [4:0]  id_rd,            // Destination register
    input         id_reg_write,     // Will write to register
    input         id_mem_read,      // Will read from memory
    input         id_branch,        // Is branch instruction
    output        id_stall,
    output        id_flush,
    
    // Execute stage
    input  [4:0]  ex_rd,
    input         ex_reg_write,
    input         ex_mem_read,
    input         ex_branch_taken,
    input  [47:0] ex_branch_target,
    output [1:0]  ex_forward_a,     // Forward control for operand A
    output [1:0]  ex_forward_b,     // Forward control for operand B
    output        ex_stall,
    output        ex_flush,
    
    // Memory stage
    input  [4:0]  mem_rd,
    input         mem_reg_write,
    output        mem_stall,
    output        mem_flush,
    
    // Write Back stage
    input  [4:0]  wb_rd,
    input         wb_reg_write,
    output        wb_stall,
    output        wb_flush,
    
    // Branch prediction
    output        predict_taken,
    output [47:0] predict_target
);

    // ========================================================================
    // Hazard Detection Unit
    // ========================================================================
    
    wire load_use_hazard;
    wire control_hazard;
    
    // Load-use hazard: instruction in EX stage is loading from memory
    // and instruction in ID stage needs that data
    assign load_use_hazard = ex_mem_read && 
                             ((ex_rd == id_rs1) || (ex_rd == id_rs2)) &&
                             (ex_rd != 5'b0);
    
    // Control hazard: branch instruction in EX stage
    assign control_hazard = ex_branch_taken;
    
    // ========================================================================
    // Forwarding Unit
    // ========================================================================
    
    // Forward from MEM stage to EX stage
    wire forward_mem_a, forward_mem_b;
    assign forward_mem_a = mem_reg_write && 
                          (mem_rd != 5'b0) && 
                          (mem_rd == id_rs1);
    
    assign forward_mem_b = mem_reg_write && 
                          (mem_rd != 5'b0) && 
                          (mem_rd == id_rs2);
    
    // Forward from WB stage to EX stage
    wire forward_wb_a, forward_wb_b;
    assign forward_wb_a = wb_reg_write && 
                         (wb_rd != 5'b0) && 
                         (wb_rd == id_rs1) &&
                         !(mem_reg_write && (mem_rd != 5'b0) && (mem_rd == id_rs1));
    
    assign forward_wb_b = wb_reg_write && 
                         (wb_rd != 5'b0) && 
                         (wb_rd == id_rs2) &&
                         !(mem_reg_write && (mem_rd != 5'b0) && (mem_rd == id_rs2));
    
    // Forwarding control signals
    // 00 = No forwarding
    // 01 = Forward from MEM stage
    // 10 = Forward from WB stage
    assign ex_forward_a = forward_mem_a ? 2'b01 :
                         forward_wb_a  ? 2'b10 :
                         2'b00;
    
    assign ex_forward_b = forward_mem_b ? 2'b01 :
                         forward_wb_b  ? 2'b10 :
                         2'b00;
    
    // ========================================================================
    // Stall Control
    // ========================================================================
    
    // Stall IF and ID stages on load-use hazard
    assign if_stall = load_use_hazard;
    assign id_stall = load_use_hazard;
    assign ex_stall = 1'b0;
    assign mem_stall = 1'b0;
    assign wb_stall = 1'b0;
    
    // ========================================================================
    // Flush Control
    // ========================================================================
    
    // Flush IF and ID stages on control hazard (branch taken)
    assign if_flush = control_hazard;
    assign id_flush = control_hazard;
    assign ex_flush = control_hazard;
    assign mem_flush = 1'b0;
    assign wb_flush = 1'b0;
    
    // ========================================================================
    // Branch Prediction (Simple: Always Not Taken)
    // ========================================================================
    
    assign predict_taken = 1'b0;
    assign predict_target = if_pc + 48'd4;  // Next sequential instruction

endmodule


/*
 * ============================================================================
 * Pipeline Registers
 * ============================================================================
 * 
 * Registers between pipeline stages to hold intermediate values.
 * 
 * ============================================================================
 */

// IF/ID Pipeline Register
module IF_ID_Register (
    input         clk,
    input         reset,
    input         stall,
    input         flush,
    
    input  [47:0] if_pc,
    input  [31:0] if_instruction,
    
    output reg [47:0] id_pc,
    output reg [31:0] id_instruction
);
    always @(posedge clk or posedge reset) begin
        if (reset || flush) begin
            id_pc <= 48'b0;
            id_instruction <= 32'b0;
        end else if (!stall) begin
            id_pc <= if_pc;
            id_instruction <= if_instruction;
        end
    end
endmodule

// ID/EX Pipeline Register
module ID_EX_Register (
    input         clk,
    input         reset,
    input         stall,
    input         flush,
    
    input  [47:0] id_pc,
    input  [47:0] id_read_data1,
    input  [47:0] id_read_data2,
    input  [47:0] id_immediate,
    input  [4:0]  id_rs1,
    input  [4:0]  id_rs2,
    input  [4:0]  id_rd,
    input  [2:0]  id_alu_op,
    input         id_alu_src,
    input         id_reg_write,
    input         id_mem_read,
    input         id_mem_write,
    input         id_branch,
    
    output reg [47:0] ex_pc,
    output reg [47:0] ex_read_data1,
    output reg [47:0] ex_read_data2,
    output reg [47:0] ex_immediate,
    output reg [4:0]  ex_rs1,
    output reg [4:0]  ex_rs2,
    output reg [4:0]  ex_rd,
    output reg [2:0]  ex_alu_op,
    output reg        ex_alu_src,
    output reg        ex_reg_write,
    output reg        ex_mem_read,
    output reg        ex_mem_write,
    output reg        ex_branch
);
    always @(posedge clk or posedge reset) begin
        if (reset || flush) begin
            ex_pc <= 48'b0;
            ex_read_data1 <= 48'b0;
            ex_read_data2 <= 48'b0;
            ex_immediate <= 48'b0;
            ex_rs1 <= 5'b0;
            ex_rs2 <= 5'b0;
            ex_rd <= 5'b0;
            ex_alu_op <= 3'b0;
            ex_alu_src <= 1'b0;
            ex_reg_write <= 1'b0;
            ex_mem_read <= 1'b0;
            ex_mem_write <= 1'b0;
            ex_branch <= 1'b0;
        end else if (!stall) begin
            ex_pc <= id_pc;
            ex_read_data1 <= id_read_data1;
            ex_read_data2 <= id_read_data2;
            ex_immediate <= id_immediate;
            ex_rs1 <= id_rs1;
            ex_rs2 <= id_rs2;
            ex_rd <= id_rd;
            ex_alu_op <= id_alu_op;
            ex_alu_src <= id_alu_src;
            ex_reg_write <= id_reg_write;
            ex_mem_read <= id_mem_read;
            ex_mem_write <= id_mem_write;
            ex_branch <= id_branch;
        end
    end
endmodule

// EX/MEM Pipeline Register
module EX_MEM_Register (
    input         clk,
    input         reset,
    input         stall,
    input         flush,
    
    input  [47:0] ex_alu_result,
    input  [47:0] ex_write_data,
    input  [4:0]  ex_rd,
    input         ex_reg_write,
    input         ex_mem_read,
    input         ex_mem_write,
    
    output reg [47:0] mem_alu_result,
    output reg [47:0] mem_write_data,
    output reg [4:0]  mem_rd,
    output reg        mem_reg_write,
    output reg        mem_mem_read,
    output reg        mem_mem_write
);
    always @(posedge clk or posedge reset) begin
        if (reset || flush) begin
            mem_alu_result <= 48'b0;
            mem_write_data <= 48'b0;
            mem_rd <= 5'b0;
            mem_reg_write <= 1'b0;
            mem_mem_read <= 1'b0;
            mem_mem_write <= 1'b0;
        end else if (!stall) begin
            mem_alu_result <= ex_alu_result;
            mem_write_data <= ex_write_data;
            mem_rd <= ex_rd;
            mem_reg_write <= ex_reg_write;
            mem_mem_read <= ex_mem_read;
            mem_mem_write <= ex_mem_write;
        end
    end
endmodule

// MEM/WB Pipeline Register
module MEM_WB_Register (
    input         clk,
    input         reset,
    input         stall,
    input         flush,
    
    input  [47:0] mem_alu_result,
    input  [47:0] mem_read_data,
    input  [4:0]  mem_rd,
    input         mem_reg_write,
    input         mem_mem_to_reg,
    
    output reg [47:0] wb_alu_result,
    output reg [47:0] wb_read_data,
    output reg [4:0]  wb_rd,
    output reg        wb_reg_write,
    output reg        wb_mem_to_reg
);
    always @(posedge clk or posedge reset) begin
        if (reset || flush) begin
            wb_alu_result <= 48'b0;
            wb_read_data <= 48'b0;
            wb_rd <= 5'b0;
            wb_reg_write <= 1'b0;
            wb_mem_to_reg <= 1'b0;
        end else if (!stall) begin
            wb_alu_result <= mem_alu_result;
            wb_read_data <= mem_read_data;
            wb_rd <= mem_rd;
            wb_reg_write <= mem_reg_write;
            wb_mem_to_reg <= mem_mem_to_reg;
        end
    end
endmodule


/*
 * ============================================================================
 * Branch Predictor (2-bit saturating counter)
 * ============================================================================
 */

module BranchPredictor (
    input         clk,
    input         reset,
    
    input  [47:0] pc,
    input         update,
    input         actual_taken,
    
    output        predict_taken
);
    // Branch History Table (BHT) with 256 entries
    reg [1:0] bht [0:255];
    
    wire [7:0] index;
    assign index = pc[9:2];  // Use bits [9:2] as index
    
    // Prediction: taken if counter >= 2
    assign predict_taken = bht[index][1];
    
    // Update on branch resolution
    integer i;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < 256; i = i + 1) begin
                bht[i] <= 2'b01;  // Weakly not taken
            end
        end else if (update) begin
            if (actual_taken) begin
                if (bht[index] != 2'b11)
                    bht[index] <= bht[index] + 1;
            end else begin
                if (bht[index] != 2'b00)
                    bht[index] <= bht[index] - 1;
            end
        end
    end
endmodule