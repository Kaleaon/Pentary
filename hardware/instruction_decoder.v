/*
 * ============================================================================
 * Pentary Instruction Decoder
 * ============================================================================
 * 
 * Decodes pentary instructions and generates control signals.
 * 
 * INSTRUCTION FORMAT (32 bits):
 *   [31:28] - Opcode (4 bits)
 *   [27:23] - Destination register (5 bits)
 *   [22:18] - Source register 1 (5 bits)
 *   [17:13] - Source register 2 (5 bits)
 *   [12:0]  - Immediate/Function (13 bits)
 * 
 * INSTRUCTION TYPES:
 *   - R-type: Register-register operations
 *   - I-type: Immediate operations
 *   - S-type: Store operations
 *   - B-type: Branch operations
 *   - J-type: Jump operations
 *   - M-type: Memristor operations
 * 
 * ============================================================================
 */

module InstructionDecoder (
    input  [31:0] instruction,
    
    // Register addresses
    output [4:0]  rs1,              // Source register 1
    output [4:0]  rs2,              // Source register 2
    output [4:0]  rd,               // Destination register
    
    // Immediate value
    output [47:0] immediate,
    
    // Control signals
    output [2:0]  alu_op,           // ALU operation
    output        alu_src,          // ALU source (0=reg, 1=imm)
    output        reg_write,        // Write to register file
    output        mem_read,         // Read from memory
    output        mem_write,        // Write to memory
    output        mem_to_reg,       // Write memory data to register
    output        branch,           // Branch instruction
    output        jump,             // Jump instruction
    output        memristor_op,     // Memristor operation
    
    // Instruction type
    output [2:0]  inst_type,
    
    // Exception signals
    output        illegal_instruction
);

    // Instruction fields
    wire [3:0]  opcode;
    wire [4:0]  rd_field;
    wire [4:0]  rs1_field;
    wire [4:0]  rs2_field;
    wire [12:0] imm_field;
    
    assign opcode = instruction[31:28];
    assign rd_field = instruction[27:23];
    assign rs1_field = instruction[22:18];
    assign rs2_field = instruction[17:13];
    assign imm_field = instruction[12:0];
    
    // Register addresses
    assign rs1 = rs1_field;
    assign rs2 = rs2_field;
    assign rd = rd_field;
    
    // Instruction types
    localparam R_TYPE = 3'b000;
    localparam I_TYPE = 3'b001;
    localparam S_TYPE = 3'b010;
    localparam B_TYPE = 3'b011;
    localparam J_TYPE = 3'b100;
    localparam M_TYPE = 3'b101;
    
    // Opcodes
    localparam OP_ADD    = 4'b0000;
    localparam OP_SUB    = 4'b0001;
    localparam OP_MUL2   = 4'b0010;
    localparam OP_DIV2   = 4'b0011;
    localparam OP_NEG    = 4'b0100;
    localparam OP_ABS    = 4'b0101;
    localparam OP_ADDI   = 4'b0110;
    localparam OP_LOAD   = 4'b0111;
    localparam OP_STORE  = 4'b1000;
    localparam OP_BEQ    = 4'b1001;
    localparam OP_BNE    = 4'b1010;
    localparam OP_JUMP   = 4'b1011;
    localparam OP_MATVEC = 4'b1100;
    localparam OP_RELU   = 4'b1101;
    localparam OP_QUANT  = 4'b1110;
    localparam OP_NOP    = 4'b1111;
    
    // Decode control signals
    reg [2:0]  alu_op_reg;
    reg        alu_src_reg;
    reg        reg_write_reg;
    reg        mem_read_reg;
    reg        mem_write_reg;
    reg        mem_to_reg_reg;
    reg        branch_reg;
    reg        jump_reg;
    reg        memristor_op_reg;
    reg [2:0]  inst_type_reg;
    reg        illegal_reg;
    
    always @(*) begin
        // Default values
        alu_op_reg = 3'b000;
        alu_src_reg = 1'b0;
        reg_write_reg = 1'b0;
        mem_read_reg = 1'b0;
        mem_write_reg = 1'b0;
        mem_to_reg_reg = 1'b0;
        branch_reg = 1'b0;
        jump_reg = 1'b0;
        memristor_op_reg = 1'b0;
        inst_type_reg = R_TYPE;
        illegal_reg = 1'b0;
        
        case (opcode)
            OP_ADD: begin
                // ADD rd, rs1, rs2
                alu_op_reg = 3'b000;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_SUB: begin
                // SUB rd, rs1, rs2
                alu_op_reg = 3'b001;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_MUL2: begin
                // MUL2 rd, rs1
                alu_op_reg = 3'b010;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_DIV2: begin
                // DIV2 rd, rs1
                alu_op_reg = 3'b011;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_NEG: begin
                // NEG rd, rs1
                alu_op_reg = 3'b100;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_ABS: begin
                // ABS rd, rs1
                alu_op_reg = 3'b101;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_ADDI: begin
                // ADDI rd, rs1, imm
                alu_op_reg = 3'b000;
                alu_src_reg = 1'b1;
                reg_write_reg = 1'b1;
                inst_type_reg = I_TYPE;
            end
            
            OP_LOAD: begin
                // LOAD rd, offset(rs1)
                alu_op_reg = 3'b000;
                alu_src_reg = 1'b1;
                mem_read_reg = 1'b1;
                mem_to_reg_reg = 1'b1;
                reg_write_reg = 1'b1;
                inst_type_reg = I_TYPE;
            end
            
            OP_STORE: begin
                // STORE rs2, offset(rs1)
                alu_op_reg = 3'b000;
                alu_src_reg = 1'b1;
                mem_write_reg = 1'b1;
                inst_type_reg = S_TYPE;
            end
            
            OP_BEQ: begin
                // BEQ rs1, rs2, offset
                alu_op_reg = 3'b001;  // Subtract for comparison
                branch_reg = 1'b1;
                inst_type_reg = B_TYPE;
            end
            
            OP_BNE: begin
                // BNE rs1, rs2, offset
                alu_op_reg = 3'b001;  // Subtract for comparison
                branch_reg = 1'b1;
                inst_type_reg = B_TYPE;
            end
            
            OP_JUMP: begin
                // JUMP offset
                jump_reg = 1'b1;
                inst_type_reg = J_TYPE;
            end
            
            OP_MATVEC: begin
                // MATVEC rd, rs1 (matrix-vector multiply)
                memristor_op_reg = 1'b1;
                reg_write_reg = 1'b1;
                inst_type_reg = M_TYPE;
            end
            
            OP_RELU: begin
                // RELU rd, rs1
                alu_op_reg = 3'b110;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_QUANT: begin
                // QUANT rd, rs1
                alu_op_reg = 3'b111;
                reg_write_reg = 1'b1;
                inst_type_reg = R_TYPE;
            end
            
            OP_NOP: begin
                // NOP - no operation
                inst_type_reg = R_TYPE;
            end
            
            default: begin
                illegal_reg = 1'b1;
            end
        endcase
    end
    
    // Immediate generation based on instruction type
    reg [47:0] immediate_reg;
    always @(*) begin
        case (inst_type_reg)
            I_TYPE: begin
                // Sign-extend 13-bit immediate to 48 bits
                immediate_reg = {{35{imm_field[12]}}, imm_field};
            end
            
            S_TYPE: begin
                // Sign-extend 13-bit immediate to 48 bits
                immediate_reg = {{35{imm_field[12]}}, imm_field};
            end
            
            B_TYPE: begin
                // Sign-extend and shift left by 1 (word-aligned)
                immediate_reg = {{34{imm_field[12]}}, imm_field, 1'b0};
            end
            
            J_TYPE: begin
                // Sign-extend and shift left by 1 (word-aligned)
                immediate_reg = {{34{imm_field[12]}}, imm_field, 1'b0};
            end
            
            default: begin
                immediate_reg = 48'b0;
            end
        endcase
    end
    
    // Output assignments
    assign alu_op = alu_op_reg;
    assign alu_src = alu_src_reg;
    assign reg_write = reg_write_reg;
    assign mem_read = mem_read_reg;
    assign mem_write = mem_write_reg;
    assign mem_to_reg = mem_to_reg_reg;
    assign branch = branch_reg;
    assign jump = jump_reg;
    assign memristor_op = memristor_op_reg;
    assign inst_type = inst_type_reg;
    assign immediate = immediate_reg;
    assign illegal_instruction = illegal_reg;

endmodule


/*
 * ============================================================================
 * Branch Unit
 * ============================================================================
 * 
 * Evaluates branch conditions and calculates branch targets.
 * 
 * ============================================================================
 */

module BranchUnit (
    input  [47:0] operand_a,
    input  [47:0] operand_b,
    input  [47:0] pc,
    input  [47:0] immediate,
    input         branch,
    input         jump,
    input  [3:0]  opcode,
    
    output        branch_taken,
    output [47:0] branch_target
);

    // Branch condition evaluation
    wire zero, negative;
    wire [47:0] diff;
    
    // Subtract for comparison
    PentaryAdder16 comparator (
        .a(operand_a),
        .b(negate_pentary(operand_b)),
        .carry_in(3'b010),
        .sum(diff),
        .carry_out()
    );
    
    PentaryIsZero zero_check (
        .input_val(diff),
        .is_zero(zero)
    );
    
    PentaryIsNegative neg_check (
        .input_val(diff),
        .is_negative(negative)
    );
    
    // Branch decision
    reg taken;
    always @(*) begin
        taken = 1'b0;
        if (jump) begin
            taken = 1'b1;
        end else if (branch) begin
            case (opcode)
                4'b1001: taken = zero;           // BEQ
                4'b1010: taken = !zero;          // BNE
                default: taken = 1'b0;
            endcase
        end
    end
    
    assign branch_taken = taken;
    assign branch_target = pc + immediate;
    
    // Helper function to negate pentary
    function [47:0] negate_pentary;
        input [47:0] value;
        integer i;
        begin
            for (i = 0; i < 16; i = i + 1) begin
                case (value[i*3 +: 3])
                    3'b000: negate_pentary[i*3 +: 3] = 3'b100;
                    3'b001: negate_pentary[i*3 +: 3] = 3'b011;
                    3'b010: negate_pentary[i*3 +: 3] = 3'b010;
                    3'b011: negate_pentary[i*3 +: 3] = 3'b001;
                    3'b100: negate_pentary[i*3 +: 3] = 3'b000;
                    default: negate_pentary[i*3 +: 3] = 3'b010;
                endcase
            end
        end
    endfunction

endmodule


/*
 * ============================================================================
 * Instruction Fetch Unit
 * ============================================================================
 */

module InstructionFetchUnit (
    input         clk,
    input         reset,
    
    // Control
    input         stall,
    input         flush,
    input         branch_taken,
    input  [47:0] branch_target,
    
    // Instruction cache interface
    output [47:0] i_cache_addr,
    output        i_cache_read,
    input  [31:0] i_cache_data,
    input         i_cache_ready,
    
    // Output
    output [47:0] pc,
    output [31:0] instruction,
    output        valid
);

    reg [47:0] pc_reg;
    reg [31:0] instruction_reg;
    reg valid_reg;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pc_reg <= 48'b0;
            instruction_reg <= 32'b0;
            valid_reg <= 1'b0;
        end else if (flush) begin
            valid_reg <= 1'b0;
        end else if (!stall) begin
            if (branch_taken) begin
                pc_reg <= branch_target;
            end else begin
                pc_reg <= pc_reg + 48'd4;
            end
            
            if (i_cache_ready) begin
                instruction_reg <= i_cache_data;
                valid_reg <= 1'b1;
            end else begin
                valid_reg <= 1'b0;
            end
        end
    end
    
    assign pc = pc_reg;
    assign instruction = instruction_reg;
    assign valid = valid_reg;
    assign i_cache_addr = pc_reg;
    assign i_cache_read = !stall && !flush;

endmodule