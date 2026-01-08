/*
 * =============================================================================
 * Pentary 3T Processor - Tiny Tapeout Top Module
 * =============================================================================
 *
 * A balanced base-5 (pentary) processor using 3-transistor analog logic.
 *
 * Target: Tiny Tapeout (2x2 tiles + 6 analog pins)
 * Process: SkyWater Sky130A
 *
 * Features:
 *   - 4-digit pentary ALU (12-bit)
 *   - Operations: ADD, SUB, NEG, NOP
 *   - Zero and negative flags
 *   - Analog I/O for direct pentary signals
 *
 * =============================================================================
 */

`default_nettype none

module tt_um_pentary_3t (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (1=output)
    `ifdef USE_POWER_PINS
    inout  wire       VPWR,     // Power
    inout  wire       VGND,     // Ground
    `endif
    input  wire [5:0] ua,       // Analog pins (directly active)
    input  wire       ena,      // Always 1 when design selected
    input  wire       clk,      // Clock
    input  wire       rst_n     // Active low reset
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    localparam DIGITS = 4;
    localparam DATA_WIDTH = DIGITS * 3;  // 12 bits
    
    // Pentary digit encoding
    localparam [2:0] PENT_ZERO = 3'b010;
    
    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Synchronous reset
    wire rst = ~rst_n;
    
    // ALU signals
    reg  [DATA_WIDTH-1:0] alu_a;
    reg  [DATA_WIDTH-1:0] alu_b;
    reg  [2:0] alu_op;
    wire [DATA_WIDTH-1:0] alu_result;
    wire alu_zero;
    wire alu_negative;
    
    // Register for output stability
    reg [7:0] result_reg;
    
    // =========================================================================
    // Input Processing
    // =========================================================================
    
    // Digital input mapping:
    //   ui_in[2:0] = Pentary digit A (3 bits)
    //   ui_in[5:3] = Pentary digit B (3 bits)
    //   ui_in[7:6] = Operation select (2 bits)
    //     00 = ADD
    //     01 = SUB
    //     10 = NEG
    //     11 = NOP
    
    // Extended input mapping via bidirectional pins:
    //   uio_in[2:0] = Pentary digit A (digit 1)
    //   uio_in[5:3] = Pentary digit B (digit 1)
    //   uio_in[7:6] = Reserved
    
    always @(posedge clk) begin
        if (rst) begin
            alu_a <= {DATA_WIDTH{1'b0}};
            alu_b <= {DATA_WIDTH{1'b0}};
            alu_op <= 3'b111;  // NOP
        end else if (ena) begin
            // Assemble 4-digit pentary numbers from inputs
            // Digits 2,3 are padded with zero (3'b010)
            alu_a <= {PENT_ZERO, PENT_ZERO, uio_in[2:0], ui_in[2:0]};
            alu_b <= {PENT_ZERO, PENT_ZERO, uio_in[5:3], ui_in[5:3]};
            alu_op <= {1'b0, ui_in[7:6]};
        end
    end
    
    // =========================================================================
    // ALU Instance
    // =========================================================================
    
    pentary_alu_tt alu_inst (
        .clk(clk),
        .rst(rst),
        .a(alu_a),
        .b(alu_b),
        .op(alu_op),
        .result(alu_result),
        .zero(alu_zero),
        .negative(alu_negative)
    );
    
    // =========================================================================
    // Output Processing
    // =========================================================================
    
    // Register outputs for clean signals
    always @(posedge clk) begin
        if (rst) begin
            result_reg <= 8'b0;
        end else begin
            result_reg <= {
                alu_negative,      // bit 7: negative flag
                alu_zero,          // bit 6: zero flag
                alu_result[5:0]    // bits 5:0: result digits 0 and 1
            };
        end
    end
    
    assign uo_out = result_reg;
    
    // =========================================================================
    // Bidirectional I/O Configuration
    // =========================================================================
    
    // Configure as inputs (no output enable)
    assign uio_oe = 8'b0000_0000;
    assign uio_out = 8'b0;
    
    // =========================================================================
    // Analog Pin Usage (active in analog-enabled designs)
    // =========================================================================
    
    // ua[0]: Analog pentary input A (0.0V to 1.6V, 5 levels)
    // ua[1]: Analog pentary input B (0.0V to 1.6V, 5 levels)
    // ua[2]: Analog pentary output (0.0V to 1.6V, 5 levels)
    // ua[3]: Reference voltage input (0.8V nominal)
    // ua[4]: Bias current input (10µA nominal)
    // ua[5]: Test/debug output
    
    // Note: Analog signals are directly connected to external pins.
    // The digital logic above operates independently.
    // For analog operation, external circuitry converts between
    // analog voltage levels and digital pentary encoding.

endmodule


// =============================================================================
// Pentary ALU (Tiny Tapeout optimized)
// =============================================================================

module pentary_alu_tt (
    input  wire        clk,
    input  wire        rst,
    input  wire [11:0] a,
    input  wire [11:0] b,
    input  wire [2:0]  op,
    output reg  [11:0] result,
    output reg         zero,
    output reg         negative
);

    // Internal wires
    wire [11:0] add_result;
    wire [11:0] sub_result;
    wire [11:0] neg_result;
    
    // =========================================================================
    // Combinational ALU Logic
    // =========================================================================
    
    // Digit extraction
    wire [2:0] a_d0 = a[2:0];
    wire [2:0] a_d1 = a[5:3];
    wire [2:0] a_d2 = a[8:6];
    wire [2:0] a_d3 = a[11:9];
    
    wire [2:0] b_d0 = b[2:0];
    wire [2:0] b_d1 = b[5:3];
    wire [2:0] b_d2 = b[8:6];
    wire [2:0] b_d3 = b[11:9];
    
    // Negate function
    function [2:0] neg_digit;
        input [2:0] d;
        case (d)
            3'b000: neg_digit = 3'b100;  // -2 -> +2
            3'b001: neg_digit = 3'b011;  // -1 -> +1
            3'b010: neg_digit = 3'b010;  //  0 ->  0
            3'b011: neg_digit = 3'b001;  // +1 -> -1
            3'b100: neg_digit = 3'b000;  // +2 -> -2
            default: neg_digit = 3'b010;
        endcase
    endfunction
    
    // Negated B digits
    wire [2:0] neg_b_d0 = neg_digit(b_d0);
    wire [2:0] neg_b_d1 = neg_digit(b_d1);
    wire [2:0] neg_b_d2 = neg_digit(b_d2);
    wire [2:0] neg_b_d3 = neg_digit(b_d3);
    
    // Negated A digits (for NEG operation)
    assign neg_result = {neg_digit(a_d3), neg_digit(a_d2),
                        neg_digit(a_d1), neg_digit(a_d0)};
    
    // Ripple-carry adder chain for ADD
    wire [2:0] add_c0, add_c1, add_c2, add_c3;
    wire [2:0] add_s0, add_s1, add_s2, add_s3;
    
    pentary_fa fa_add0 (.a(a_d0), .b(b_d0), .cin(3'b010), .sum(add_s0), .cout(add_c0));
    pentary_fa fa_add1 (.a(a_d1), .b(b_d1), .cin(add_c0), .sum(add_s1), .cout(add_c1));
    pentary_fa fa_add2 (.a(a_d2), .b(b_d2), .cin(add_c1), .sum(add_s2), .cout(add_c2));
    pentary_fa fa_add3 (.a(a_d3), .b(b_d3), .cin(add_c2), .sum(add_s3), .cout(add_c3));
    
    assign add_result = {add_s3, add_s2, add_s1, add_s0};
    
    // Ripple-carry adder chain for SUB (a + (-b))
    wire [2:0] sub_c0, sub_c1, sub_c2, sub_c3;
    wire [2:0] sub_s0, sub_s1, sub_s2, sub_s3;
    
    pentary_fa fa_sub0 (.a(a_d0), .b(neg_b_d0), .cin(3'b010), .sum(sub_s0), .cout(sub_c0));
    pentary_fa fa_sub1 (.a(a_d1), .b(neg_b_d1), .cin(sub_c0), .sum(sub_s1), .cout(sub_c1));
    pentary_fa fa_sub2 (.a(a_d2), .b(neg_b_d2), .cin(sub_c1), .sum(sub_s2), .cout(sub_c2));
    pentary_fa fa_sub3 (.a(a_d3), .b(neg_b_d3), .cin(sub_c2), .sum(sub_s3), .cout(sub_c3));
    
    assign sub_result = {sub_s3, sub_s2, sub_s1, sub_s0};
    
    // =========================================================================
    // Registered Output
    // =========================================================================
    
    always @(posedge clk) begin
        if (rst) begin
            result <= 12'b010_010_010_010;  // Zero
            zero <= 1'b1;
            negative <= 1'b0;
        end else begin
            case (op[1:0])
                2'b00: result <= add_result;  // ADD
                2'b01: result <= sub_result;  // SUB
                2'b10: result <= neg_result;  // NEG
                2'b11: result <= a;           // NOP (pass through)
            endcase
            
            // Zero flag
            zero <= (result == 12'b010_010_010_010);
            
            // Negative flag (check MSB digit)
            negative <= (result[11:9] == 3'b000) || (result[11:9] == 3'b001);
        end
    end

endmodule


// =============================================================================
// Pentary Full Adder
// =============================================================================

module pentary_fa (
    input  wire [2:0] a,
    input  wire [2:0] b,
    input  wire [2:0] cin,
    output reg  [2:0] sum,
    output reg  [2:0] cout
);

    // Convert to signed for arithmetic
    function signed [2:0] to_value;
        input [2:0] d;
        case (d)
            3'b000: to_value = -2;
            3'b001: to_value = -1;
            3'b010: to_value = 0;
            3'b011: to_value = 1;
            3'b100: to_value = 2;
            default: to_value = 0;
        endcase
    endfunction
    
    function [2:0] to_digit;
        input signed [3:0] v;
        case (v)
            -2: to_digit = 3'b000;
            -1: to_digit = 3'b001;
            0:  to_digit = 3'b010;
            1:  to_digit = 3'b011;
            2:  to_digit = 3'b100;
            default: to_digit = 3'b010;
        endcase
    endfunction
    
    wire signed [3:0] total = to_value(a) + to_value(b) + to_value(cin);
    
    always @(*) begin
        if (total >= 3) begin
            sum = to_digit(total - 5);
            cout = 3'b011;  // +1 carry
        end else if (total <= -3) begin
            sum = to_digit(total + 5);
            cout = 3'b001;  // -1 carry
        end else begin
            sum = to_digit(total);
            cout = 3'b010;  // 0 carry
        end
    end

endmodule

`default_nettype wire
