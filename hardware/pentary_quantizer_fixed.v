/*
 * ============================================================================
 * Pentary Quantizer - Fixed-Point Implementation
 * ============================================================================
 * 
 * Quantizes fixed-point values to pentary digits {-2, -1, 0, +1, +2}.
 * Uses fixed-point arithmetic instead of floating-point for synthesizability.
 * 
 * FIXED-POINT FORMAT:
 *   - 32-bit signed fixed-point
 *   - 16 integer bits, 16 fractional bits (Q16.16)
 *   - Range: -32768.0 to +32767.99998
 * 
 * QUANTIZATION ALGORITHM:
 *   1. Subtract zero_point from input
 *   2. Divide by scale (using fixed-point division)
 *   3. Round to nearest integer
 *   4. Clip to [-2, +2] range
 *   5. Encode as pentary digit
 * 
 * ============================================================================
 */

module PentaryQuantizer (
    input  [31:0] input_value,   // Q16.16 fixed-point input
    input  [31:0] scale,         // Q16.16 scale factor
    input  [31:0] zero_point,    // Q16.16 zero point
    output [2:0]  quantized      // Single pentary digit output
);

    // Step 1: Subtract zero_point
    wire signed [31:0] centered;
    assign centered = input_value - zero_point;
    
    // Step 2: Divide by scale (fixed-point division)
    // For Q16.16 format: (a / b) = (a << 16) / b
    wire signed [47:0] numerator;
    assign numerator = {centered, 16'b0};  // Shift left by 16 bits
    
    wire signed [31:0] scaled;
    assign scaled = numerator / scale;
    
    // Step 3: Round to nearest integer
    // Add 0.5 (0x8000 in Q16.16) and truncate
    wire signed [31:0] rounded;
    assign rounded = (scaled + 32'h00008000) >>> 16;  // Arithmetic right shift
    
    // Step 4: Clip to [-2, +2] range
    wire signed [31:0] clipped;
    assign clipped = (rounded < -2) ? -2 :
                     (rounded > 2)  ? 2  :
                     rounded;
    
    // Step 5: Encode as pentary digit
    reg [2:0] quantized_reg;
    always @(*) begin
        case (clipped)
            -2: quantized_reg = 3'b000;  // -2 (⊖)
            -1: quantized_reg = 3'b001;  // -1 (-)
            0:  quantized_reg = 3'b010;  //  0 (0)
            1:  quantized_reg = 3'b011;  // +1 (+)
            2:  quantized_reg = 3'b100;  // +2 (⊕)
            default: quantized_reg = 3'b010;  // Default to 0
        endcase
    end
    
    assign quantized = quantized_reg;

endmodule


/*
 * ============================================================================
 * Multi-Value Pentary Quantizer (16 values)
 * ============================================================================
 * 
 * Quantizes 16 fixed-point values to 16 pentary digits simultaneously.
 * Useful for quantizing weight vectors or activation vectors.
 * 
 * ============================================================================
 */

module PentaryQuantizer16 (
    input  [511:0] input_values,  // 16 × 32-bit Q16.16 values
    input  [31:0]  scale,         // Shared scale factor
    input  [31:0]  zero_point,    // Shared zero point
    output [47:0]  quantized      // 16 pentary digits (48 bits)
);

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : quantize_array
            PentaryQuantizer quant (
                .input_value(input_values[i*32 +: 32]),
                .scale(scale),
                .zero_point(zero_point),
                .quantized(quantized[i*3 +: 3])
            );
        end
    endgenerate

endmodule


/*
 * ============================================================================
 * Pentary Dequantizer
 * ============================================================================
 * 
 * Converts pentary digits back to fixed-point values.
 * Inverse operation of quantization.
 * 
 * DEQUANTIZATION ALGORITHM:
 *   1. Decode pentary digit to integer
 *   2. Multiply by scale
 *   3. Add zero_point
 * 
 * ============================================================================
 */

module PentaryDequantizer (
    input  [2:0]  quantized,     // Single pentary digit
    input  [31:0] scale,         // Q16.16 scale factor
    input  [31:0] zero_point,    // Q16.16 zero point
    output [31:0] output_value   // Q16.16 fixed-point output
);

    // Step 1: Decode pentary digit to signed integer
    reg signed [2:0] decoded;
    always @(*) begin
        case (quantized)
            3'b000: decoded = -2;
            3'b001: decoded = -1;
            3'b010: decoded = 0;
            3'b011: decoded = 1;
            3'b100: decoded = 2;
            default: decoded = 0;
        endcase
    end
    
    // Step 2: Multiply by scale (fixed-point multiplication)
    // For Q16.16: (a * b) >> 16
    wire signed [31:0] decoded_extended;
    assign decoded_extended = {{29{decoded[2]}}, decoded};  // Sign extend
    
    wire signed [63:0] product;
    assign product = decoded_extended * scale;
    
    wire signed [31:0] scaled;
    assign scaled = product[47:16];  // Extract Q16.16 result
    
    // Step 3: Add zero_point
    assign output_value = scaled + zero_point;

endmodule


/*
 * ============================================================================
 * Adaptive Quantizer with Per-Channel Scaling
 * ============================================================================
 * 
 * Advanced quantizer that supports per-channel scale and zero-point.
 * Used for quantizing neural network weights and activations.
 * 
 * ============================================================================
 */

module PentaryAdaptiveQuantizer (
    input  [31:0] input_value,      // Q16.16 fixed-point input
    input  [31:0] scale,            // Q16.16 scale factor
    input  [31:0] zero_point,       // Q16.16 zero point
    input  [1:0]  rounding_mode,    // 00=nearest, 01=floor, 10=ceil, 11=truncate
    input         symmetric,        // 1=symmetric quantization (no zero_point)
    output [2:0]  quantized,        // Pentary digit output
    output        clipped           // 1 if value was clipped
);

    // Step 1: Subtract zero_point (if not symmetric)
    wire signed [31:0] centered;
    assign centered = symmetric ? input_value : (input_value - zero_point);
    
    // Step 2: Divide by scale
    wire signed [47:0] numerator;
    assign numerator = {centered, 16'b0};
    
    wire signed [31:0] scaled;
    assign scaled = numerator / scale;
    
    // Step 3: Round based on mode
    reg signed [31:0] rounded;
    always @(*) begin
        case (rounding_mode)
            2'b00: rounded = (scaled + 32'h00008000) >>> 16;  // Nearest
            2'b01: rounded = scaled >>> 16;                    // Floor
            2'b10: rounded = (scaled + 32'h0000FFFF) >>> 16;  // Ceil
            2'b11: rounded = scaled >>> 16;                    // Truncate
            default: rounded = scaled >>> 16;
        endcase
    end
    
    // Step 4: Clip to [-2, +2] range and detect clipping
    wire signed [31:0] clipped_val;
    wire was_clipped;
    
    assign was_clipped = (rounded < -2) || (rounded > 2);
    assign clipped_val = (rounded < -2) ? -2 :
                        (rounded > 2)  ? 2  :
                        rounded;
    
    assign clipped = was_clipped;
    
    // Step 5: Encode as pentary digit
    reg [2:0] quantized_reg;
    always @(*) begin
        case (clipped_val)
            -2: quantized_reg = 3'b000;
            -1: quantized_reg = 3'b001;
            0:  quantized_reg = 3'b010;
            1:  quantized_reg = 3'b011;
            2:  quantized_reg = 3'b100;
            default: quantized_reg = 3'b010;
        endcase
    end
    
    assign quantized = quantized_reg;

endmodule


/*
 * ============================================================================
 * Fixed-Point Helper Functions
 * ============================================================================
 */

// Convert integer to Q16.16 fixed-point
module IntToFixed (
    input  signed [15:0] int_val,
    output [31:0] fixed_val
);
    assign fixed_val = {int_val, 16'b0};
endmodule

// Convert Q16.16 fixed-point to integer (with rounding)
module FixedToInt (
    input  [31:0] fixed_val,
    output signed [15:0] int_val
);
    wire [31:0] rounded;
    assign rounded = fixed_val + 32'h00008000;  // Add 0.5
    assign int_val = rounded[31:16];            // Extract integer part
endmodule

// Fixed-point multiplication (Q16.16 × Q16.16 = Q16.16)
module FixedMul (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] result
);
    wire signed [63:0] product;
    assign product = $signed(a) * $signed(b);
    assign result = product[47:16];  // Extract Q16.16 result
endmodule

// Fixed-point division (Q16.16 / Q16.16 = Q16.16)
module FixedDiv (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] result
);
    wire signed [47:0] numerator;
    assign numerator = {$signed(a), 16'b0};  // Shift left by 16
    assign result = numerator / $signed(b);
endmodule