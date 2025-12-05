/*
 * ============================================================================
 * Pentary Adder - Complete Implementation
 * ============================================================================
 * 
 * Single pentary digit adder with carry propagation.
 * Implements complete lookup table for all 75 combinations.
 * 
 * PENTARY ENCODING:
 *   3'b000 = -2 (⊖, strong negative)
 *   3'b001 = -1 (-,  weak negative)
 *   3'b010 =  0 (0,  zero)
 *   3'b011 = +1 (+,  weak positive)
 *   3'b100 = +2 (⊕, strong positive)
 * 
 * ADDITION RULES:
 *   1. Add two digits: a + b + carry_in
 *   2. If result > +2: subtract 5, carry +1 to next position
 *   3. If result < -2: add 5, carry -1 to next position
 *   4. Otherwise: result stays, no carry
 * 
 * EXAMPLE:
 *   a = +2, b = +2, carry_in = 0
 *   Sum = +2 + +2 + 0 = +4
 *   Since +4 > +2: subtract 5 → sum = -1, carry = +1
 * 
 * ============================================================================
 */

module PentaryAdder (
    input  [2:0] a,         // Single pentary digit (3-bit encoded)
    input  [2:0] b,         // Single pentary digit (3-bit encoded)
    input  [2:0] carry_in,  // Carry from previous position (-1, 0, or +1)
    output [2:0] sum,       // Sum digit (result)
    output [2:0] carry_out  // Carry to next position (-1, 0, or +1)
);

    // Internal signals for decoded values
    reg signed [3:0] a_val, b_val, carry_val;
    reg signed [4:0] temp_sum;
    reg [2:0] sum_reg, carry_reg;
    
    // Decode pentary to signed integer
    always @(*) begin
        case (a)
            3'b000: a_val = -2;
            3'b001: a_val = -1;
            3'b010: a_val = 0;
            3'b011: a_val = 1;
            3'b100: a_val = 2;
            default: a_val = 0;
        endcase
        
        case (b)
            3'b000: b_val = -2;
            3'b001: b_val = -1;
            3'b010: b_val = 0;
            3'b011: b_val = 1;
            3'b100: b_val = 2;
            default: b_val = 0;
        endcase
        
        case (carry_in)
            3'b000: carry_val = -2;
            3'b001: carry_val = -1;
            3'b010: carry_val = 0;
            3'b011: carry_val = 1;
            3'b100: carry_val = 2;
            default: carry_val = 0;
        endcase
    end
    
    // Perform addition with carry normalization
    always @(*) begin
        temp_sum = a_val + b_val + carry_val;
        
        // Normalize to pentary range [-2, +2] with carry
        if (temp_sum > 2) begin
            temp_sum = temp_sum - 5;
            carry_reg = 3'b011;  // carry = +1
        end else if (temp_sum < -2) begin
            temp_sum = temp_sum + 5;
            carry_reg = 3'b001;  // carry = -1
        end else begin
            carry_reg = 3'b010;  // carry = 0
        end
        
        // Encode result back to pentary
        case (temp_sum)
            -2: sum_reg = 3'b000;
            -1: sum_reg = 3'b001;
            0:  sum_reg = 3'b010;
            1:  sum_reg = 3'b011;
            2:  sum_reg = 3'b100;
            default: sum_reg = 3'b010;
        endcase
    end
    
    assign sum = sum_reg;
    assign carry_out = carry_reg;

endmodule


/*
 * ============================================================================
 * Multi-Digit Pentary Adder (16 digits)
 * ============================================================================
 * 
 * Adds two 16-digit pentary numbers with full carry propagation.
 * 
 * INPUT/OUTPUT FORMAT:
 *   - Each input is 48 bits (16 digits × 3 bits)
 *   - Digit 0 (LSB) is in bits [2:0]
 *   - Digit 15 (MSB) is in bits [47:45]
 * 
 * ============================================================================
 */

module PentaryAdder16 (
    input  [47:0] a,         // 16 pentary digits
    input  [47:0] b,         // 16 pentary digits
    input  [2:0]  carry_in,  // Initial carry (usually 0)
    output [47:0] sum,       // 16 pentary digits result
    output [2:0]  carry_out  // Final carry out
);

    // Carry chain between digits
    wire [2:0] carry [0:16];
    
    // Connect initial carry
    assign carry[0] = carry_in;
    
    // Generate 16 single-digit adders with carry chain
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : adder_chain
            PentaryAdder digit_adder (
                .a(a[i*3 +: 3]),
                .b(b[i*3 +: 3]),
                .carry_in(carry[i]),
                .sum(sum[i*3 +: 3]),
                .carry_out(carry[i+1])
            );
        end
    endgenerate
    
    // Final carry out
    assign carry_out = carry[16];

endmodule