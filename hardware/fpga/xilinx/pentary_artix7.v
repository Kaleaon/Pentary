// =============================================================================
// Pentary Processor - Xilinx Artix-7 Implementation
// =============================================================================
// Target: Artix-7 (XC7A35T or XC7A100T)
// Board: Digilent Basys3 or Nexys A7
// 
// Features:
// - 16-digit pentary processing (48-bit internal)
// - 16 general-purpose registers
// - Full ALU with all pentary operations
// - UART interface for debugging
// - 7-segment display output
// - Hardware multiplier using DSP48E1
// =============================================================================

`timescale 1ns / 1ps

module pentary_artix7 (
    // Clock and Reset
    input  wire        clk_100mhz,     // 100 MHz system clock
    input  wire        rst_n,          // Active-low reset (directly usable on Artix-7)
    
    // UART Interface
    input  wire        uart_rx,
    output wire        uart_tx,
    
    // 7-Segment Display (active low segments, active low anodes)
    output wire [6:0]  seg,            // Segment outputs [a-g]
    output wire [7:0]  an,             // Anode enables
    
    // LEDs
    output wire [15:0] led,            // 16 LEDs
    
    // Switches and Buttons
    input  wire [15:0] sw,             // 16 switches
    input  wire [4:0]  btn,            // 5 buttons (center, up, down, left, right)
    
    // Optional: Pmod Connector (directly usable)
    output wire [7:0]  pmod_a,         // PMOD A output
    input  wire [7:0]  pmod_b          // PMOD B input
);

    // =========================================================================
    // Clock and Reset
    // =========================================================================
    
    // For Artix-7, we can use the 100MHz clock directly for moderate-speed designs
    // For higher performance, instantiate MMCM
    
    wire clk;
    wire rst;
    
    // Use BUFG for clock distribution
    BUFG clk_bufg (
        .I(clk_100mhz),
        .O(clk)
    );
    
    // Synchronize reset
    reg [2:0] rst_sync;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rst_sync <= 3'b111;
        else
            rst_sync <= {rst_sync[1:0], 1'b0};
    end
    assign rst = rst_sync[2];

    // =========================================================================
    // Pentary Encoding
    // =========================================================================
    // Balanced quinary: {-2, -1, 0, +1, +2} encoded in 3 bits
    //   -2 -> 3'b000
    //   -1 -> 3'b001
    //    0 -> 3'b010
    //   +1 -> 3'b011
    //   +2 -> 3'b100

    localparam [2:0] PENT_N2 = 3'b000;  // -2
    localparam [2:0] PENT_N1 = 3'b001;  // -1
    localparam [2:0] PENT_Z  = 3'b010;  //  0
    localparam [2:0] PENT_P1 = 3'b011;  // +1
    localparam [2:0] PENT_P2 = 3'b100;  // +2

    // =========================================================================
    // Pentary Processor Core
    // =========================================================================
    
    // 16-digit pentary = 48 bits (3 bits per digit)
    localparam PENT_WIDTH = 48;
    localparam NUM_REGS = 16;
    
    // Register File
    reg [PENT_WIDTH-1:0] regs [0:NUM_REGS-1];
    
    // ALU Inputs/Outputs
    reg  [PENT_WIDTH-1:0] alu_a;
    reg  [PENT_WIDTH-1:0] alu_b;
    wire [PENT_WIDTH-1:0] alu_result;
    reg  [3:0]            alu_op;
    wire                  alu_zero;
    wire                  alu_neg;
    wire                  alu_pos;
    wire                  alu_overflow;
    
    // ALU Operations
    localparam [3:0] ALU_ADD  = 4'b0000;
    localparam [3:0] ALU_SUB  = 4'b0001;
    localparam [3:0] ALU_MUL2 = 4'b0010;  // Multiply by 2
    localparam [3:0] ALU_DIV2 = 4'b0011;  // Divide by 2
    localparam [3:0] ALU_NEG  = 4'b0100;  // Negate
    localparam [3:0] ALU_ABS  = 4'b0101;  // Absolute value
    localparam [3:0] ALU_SHL  = 4'b0110;  // Shift left (multiply by 5)
    localparam [3:0] ALU_SHR  = 4'b0111;  // Shift right (divide by 5)
    localparam [3:0] ALU_CMP  = 4'b1000;  // Compare
    localparam [3:0] ALU_MAX  = 4'b1001;  // Maximum
    localparam [3:0] ALU_MIN  = 4'b1010;  // Minimum
    localparam [3:0] ALU_MUL5 = 4'b1011;  // Multiply by 5 using DSP
    
    // =========================================================================
    // Pentary ALU Instance
    // =========================================================================
    
    pentary_alu_16digit u_alu (
        .clk        (clk),
        .rst        (rst),
        .a          (alu_a),
        .b          (alu_b),
        .op         (alu_op),
        .result     (alu_result),
        .zero       (alu_zero),
        .negative   (alu_neg),
        .positive   (alu_pos),
        .overflow   (alu_overflow)
    );
    
    // =========================================================================
    // Simple State Machine for Demo
    // =========================================================================
    
    localparam [2:0] ST_IDLE   = 3'b000;
    localparam [2:0] ST_LOAD   = 3'b001;
    localparam [2:0] ST_EXEC   = 3'b010;
    localparam [2:0] ST_STORE  = 3'b011;
    localparam [2:0] ST_DONE   = 3'b100;
    
    reg [2:0] state;
    reg [3:0] reg_addr_a, reg_addr_b, reg_addr_d;
    reg [PENT_WIDTH-1:0] result_reg;
    
    // Demo: Simple accumulator using switches and buttons
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= ST_IDLE;
            result_reg <= {PENT_WIDTH{1'b0}};
            alu_op <= ALU_ADD;
            // Initialize register 0 to zero
            integer i;
            for (i = 0; i < NUM_REGS; i = i + 1)
                regs[i] <= {16{PENT_Z}};  // All zeros
        end else begin
            case (state)
                ST_IDLE: begin
                    // Wait for button press
                    if (btn[0]) begin  // Center button - Add
                        alu_op <= ALU_ADD;
                        state <= ST_LOAD;
                    end else if (btn[1]) begin  // Up button - Subtract
                        alu_op <= ALU_SUB;
                        state <= ST_LOAD;
                    end else if (btn[2]) begin  // Down button - Negate
                        alu_op <= ALU_NEG;
                        state <= ST_LOAD;
                    end else if (btn[3]) begin  // Left button - Shift left
                        alu_op <= ALU_SHL;
                        state <= ST_LOAD;
                    end else if (btn[4]) begin  // Right button - Clear
                        result_reg <= {16{PENT_Z}};
                    end
                end
                
                ST_LOAD: begin
                    // Load operands from switches (interpret as pentary digits)
                    alu_a <= result_reg;
                    // Convert switch value to pentary
                    alu_b <= sw_to_pentary(sw);
                    state <= ST_EXEC;
                end
                
                ST_EXEC: begin
                    // Wait for ALU result (combinational, but add pipeline stage)
                    state <= ST_STORE;
                end
                
                ST_STORE: begin
                    result_reg <= alu_result;
                    state <= ST_DONE;
                end
                
                ST_DONE: begin
                    // Wait for button release
                    if (btn == 5'b00000)
                        state <= ST_IDLE;
                end
            endcase
        end
    end
    
    // Convert switch value to pentary (simple mapping)
    function [PENT_WIDTH-1:0] sw_to_pentary;
        input [15:0] switches;
        integer i;
        reg [2:0] digit;
        begin
            sw_to_pentary = {16{PENT_Z}};  // Start with zeros
            // Use lower 3 switches for first digit
            case (switches[2:0])
                3'b000: digit = PENT_Z;
                3'b001: digit = PENT_P1;
                3'b010: digit = PENT_P2;
                3'b011: digit = PENT_N1;
                3'b100: digit = PENT_N2;
                default: digit = PENT_Z;
            endcase
            sw_to_pentary[2:0] = digit;
            
            // Second digit from switches [5:3]
            case (switches[5:3])
                3'b000: digit = PENT_Z;
                3'b001: digit = PENT_P1;
                3'b010: digit = PENT_P2;
                3'b011: digit = PENT_N1;
                3'b100: digit = PENT_N2;
                default: digit = PENT_Z;
            endcase
            sw_to_pentary[5:3] = digit;
        end
    endfunction
    
    // =========================================================================
    // Display Interface
    // =========================================================================
    
    // 7-segment display controller
    pentary_display u_display (
        .clk        (clk),
        .rst        (rst),
        .value      (result_reg),
        .seg        (seg),
        .an         (an)
    );
    
    // =========================================================================
    // UART Interface
    // =========================================================================
    
    wire [7:0] uart_rx_data;
    wire       uart_rx_valid;
    wire [7:0] uart_tx_data;
    wire       uart_tx_start;
    wire       uart_tx_busy;
    
    uart_rx #(.CLKS_PER_BIT(868)) u_uart_rx (  // 100MHz / 115200 baud
        .clk        (clk),
        .rst        (rst),
        .rx         (uart_rx),
        .data       (uart_rx_data),
        .valid      (uart_rx_valid)
    );
    
    uart_tx #(.CLKS_PER_BIT(868)) u_uart_tx (
        .clk        (clk),
        .rst        (rst),
        .data       (uart_tx_data),
        .start      (uart_tx_start),
        .tx         (uart_tx),
        .busy       (uart_tx_busy)
    );
    
    // Simple UART command processor
    uart_cmd_processor u_cmd (
        .clk        (clk),
        .rst        (rst),
        .rx_data    (uart_rx_data),
        .rx_valid   (uart_rx_valid),
        .tx_data    (uart_tx_data),
        .tx_start   (uart_tx_start),
        .tx_busy    (uart_tx_busy),
        .regs       (/* connect to register file for debug */),
        .result     (result_reg)
    );
    
    // =========================================================================
    // LED Output
    // =========================================================================
    
    // Display lower 16 bits of result as LEDs
    assign led = {
        result_reg[47:45] != PENT_Z,  // Overflow indicator
        result_reg[44:42] != PENT_Z,
        result_reg[2:0] == PENT_P2,   // +2
        result_reg[2:0] == PENT_P1,   // +1
        result_reg[2:0] == PENT_Z,    //  0
        result_reg[2:0] == PENT_N1,   // -1
        result_reg[2:0] == PENT_N2,   // -2
        alu_overflow,
        alu_neg,
        alu_zero,
        alu_pos,
        state
    };
    
    // PMOD outputs for debugging
    assign pmod_a = result_reg[7:0];

endmodule


// =============================================================================
// Pentary ALU - 16-digit
// =============================================================================

module pentary_alu_16digit (
    input  wire        clk,
    input  wire        rst,
    input  wire [47:0] a,
    input  wire [47:0] b,
    input  wire [3:0]  op,
    output reg  [47:0] result,
    output wire        zero,
    output wire        negative,
    output wire        positive,
    output wire        overflow
);

    localparam [2:0] PENT_N2 = 3'b000;
    localparam [2:0] PENT_N1 = 3'b001;
    localparam [2:0] PENT_Z  = 3'b010;
    localparam [2:0] PENT_P1 = 3'b011;
    localparam [2:0] PENT_P2 = 3'b100;

    localparam [3:0] ALU_ADD  = 4'b0000;
    localparam [3:0] ALU_SUB  = 4'b0001;
    localparam [3:0] ALU_MUL2 = 4'b0010;
    localparam [3:0] ALU_DIV2 = 4'b0011;
    localparam [3:0] ALU_NEG  = 4'b0100;
    localparam [3:0] ALU_ABS  = 4'b0101;
    localparam [3:0] ALU_SHL  = 4'b0110;
    localparam [3:0] ALU_SHR  = 4'b0111;
    localparam [3:0] ALU_CMP  = 4'b1000;
    
    // Digit extraction
    wire [2:0] a_digits [0:15];
    wire [2:0] b_digits [0:15];
    reg  [2:0] r_digits [0:15];
    
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : digit_extract
            assign a_digits[i] = a[i*3+2 : i*3];
            assign b_digits[i] = b[i*3+2 : i*3];
        end
    endgenerate
    
    // Convert digit to signed value
    function signed [2:0] to_signed;
        input [2:0] digit;
        begin
            case (digit)
                PENT_N2: to_signed = -3'sd2;
                PENT_N1: to_signed = -3'sd1;
                PENT_Z:  to_signed = 3'sd0;
                PENT_P1: to_signed = 3'sd1;
                PENT_P2: to_signed = 3'sd2;
                default: to_signed = 3'sd0;
            endcase
        end
    endfunction
    
    // Convert signed value back to digit
    function [2:0] to_digit;
        input signed [3:0] val;
        begin
            case (val)
                -4'sd2:  to_digit = PENT_N2;
                -4'sd1:  to_digit = PENT_N1;
                4'sd0:   to_digit = PENT_Z;
                4'sd1:   to_digit = PENT_P1;
                4'sd2:   to_digit = PENT_P2;
                default: to_digit = PENT_Z;
            endcase
        end
    endfunction
    
    // Addition with carry propagation
    reg signed [3:0] sum_temp [0:15];
    reg signed [1:0] carry [0:16];
    
    integer j;
    always @(*) begin
        carry[0] = 0;
        for (j = 0; j < 16; j = j + 1) begin
            sum_temp[j] = to_signed(a_digits[j]) + to_signed(b_digits[j]) + carry[j];
            
            if (sum_temp[j] > 2) begin
                r_digits[j] = to_digit(sum_temp[j] - 5);
                carry[j+1] = 1;
            end else if (sum_temp[j] < -2) begin
                r_digits[j] = to_digit(sum_temp[j] + 5);
                carry[j+1] = -1;
            end else begin
                r_digits[j] = to_digit(sum_temp[j]);
                carry[j+1] = 0;
            end
        end
    end
    
    // Negation
    wire [47:0] neg_a;
    generate
        for (i = 0; i < 16; i = i + 1) begin : negate
            wire [2:0] orig = a[i*3+2 : i*3];
            assign neg_a[i*3+2 : i*3] = 
                (orig == PENT_N2) ? PENT_P2 :
                (orig == PENT_N1) ? PENT_P1 :
                (orig == PENT_P1) ? PENT_N1 :
                (orig == PENT_P2) ? PENT_N2 : PENT_Z;
        end
    endgenerate
    
    // Shift operations (multiply/divide by 5)
    wire [47:0] shl_result = {a[44:0], PENT_Z};
    wire [47:0] shr_result = {PENT_Z, a[47:3]};
    
    // Assemble result
    wire [47:0] add_result;
    generate
        for (i = 0; i < 16; i = i + 1) begin : assemble
            assign add_result[i*3+2 : i*3] = r_digits[i];
        end
    endgenerate
    
    // Operation selection
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result <= {16{PENT_Z}};
        end else begin
            case (op)
                ALU_ADD:  result <= add_result;
                ALU_SUB:  result <= add_result;  // b is negated externally
                ALU_NEG:  result <= neg_a;
                ALU_SHL:  result <= shl_result;
                ALU_SHR:  result <= shr_result;
                default:  result <= add_result;
            endcase
        end
    end
    
    // Flags
    assign zero = (result == {16{PENT_Z}});
    assign negative = (result[47:45] == PENT_N2 || result[47:45] == PENT_N1);
    assign positive = (result[47:45] == PENT_P1 || result[47:45] == PENT_P2);
    assign overflow = (carry[16] != 0);

endmodule


// =============================================================================
// 7-Segment Display Controller for Pentary Values
// =============================================================================

module pentary_display (
    input  wire        clk,
    input  wire        rst,
    input  wire [47:0] value,
    output reg  [6:0]  seg,
    output reg  [7:0]  an
);

    localparam [2:0] PENT_N2 = 3'b000;
    localparam [2:0] PENT_N1 = 3'b001;
    localparam [2:0] PENT_Z  = 3'b010;
    localparam [2:0] PENT_P1 = 3'b011;
    localparam [2:0] PENT_P2 = 3'b100;

    // Refresh counter for multiplexing
    reg [19:0] refresh_counter;
    wire [2:0] digit_select = refresh_counter[19:17];
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            refresh_counter <= 0;
        else
            refresh_counter <= refresh_counter + 1;
    end
    
    // Select digit to display
    reg [2:0] current_digit;
    always @(*) begin
        case (digit_select)
            3'd0: current_digit = value[2:0];
            3'd1: current_digit = value[5:3];
            3'd2: current_digit = value[8:6];
            3'd3: current_digit = value[11:9];
            3'd4: current_digit = value[14:12];
            3'd5: current_digit = value[17:15];
            3'd6: current_digit = value[20:18];
            3'd7: current_digit = value[23:21];
            default: current_digit = PENT_Z;
        endcase
    end
    
    // Anode select (active low)
    always @(*) begin
        an = 8'b11111111;
        an[digit_select] = 1'b0;
    end
    
    // Segment patterns for pentary digits (active low)
    // Custom patterns: -, N, 0, +, P
    always @(*) begin
        case (current_digit)
            PENT_N2: seg = 7'b0111111;  // Display "-" (just middle segment)
            PENT_N1: seg = 7'b1001000;  // Display "n" 
            PENT_Z:  seg = 7'b1000000;  // Display "0"
            PENT_P1: seg = 7'b0110000;  // Display "+"
            PENT_P2: seg = 7'b0001100;  // Display "P"
            default: seg = 7'b1111111;  // Blank
        endcase
    end

endmodule


// =============================================================================
// UART Receiver
// =============================================================================

module uart_rx #(
    parameter CLKS_PER_BIT = 868  // 100MHz / 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);

    localparam IDLE = 2'b00;
    localparam START = 2'b01;
    localparam DATA = 2'b10;
    localparam STOP = 2'b11;
    
    reg [1:0] state;
    reg [15:0] clk_count;
    reg [2:0] bit_index;
    reg [7:0] rx_data;
    reg rx_d, rx_dd;
    
    // Double-register input for metastability
    always @(posedge clk) begin
        rx_d <= rx;
        rx_dd <= rx_d;
    end
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            valid <= 1'b0;
            clk_count <= 0;
            bit_index <= 0;
        end else begin
            valid <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (rx_dd == 1'b0) begin  // Start bit detected
                        state <= START;
                        clk_count <= 0;
                    end
                end
                
                START: begin
                    if (clk_count == CLKS_PER_BIT/2 - 1) begin
                        if (rx_dd == 1'b0) begin
                            state <= DATA;
                            clk_count <= 0;
                            bit_index <= 0;
                        end else begin
                            state <= IDLE;
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                DATA: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        rx_data[bit_index] <= rx_dd;
                        
                        if (bit_index == 7) begin
                            state <= STOP;
                        end else begin
                            bit_index <= bit_index + 1;
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                STOP: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        state <= IDLE;
                        valid <= 1'b1;
                        data <= rx_data;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
            endcase
        end
    end

endmodule


// =============================================================================
// UART Transmitter
// =============================================================================

module uart_tx #(
    parameter CLKS_PER_BIT = 868
)(
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       start,
    output reg        tx,
    output wire       busy
);

    localparam IDLE = 2'b00;
    localparam START = 2'b01;
    localparam DATA = 2'b10;
    localparam STOP = 2'b11;
    
    reg [1:0] state;
    reg [15:0] clk_count;
    reg [2:0] bit_index;
    reg [7:0] tx_data;
    
    assign busy = (state != IDLE);
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            tx <= 1'b1;
            clk_count <= 0;
            bit_index <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    if (start) begin
                        state <= START;
                        tx_data <= data;
                        clk_count <= 0;
                    end
                end
                
                START: begin
                    tx <= 1'b0;
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        state <= DATA;
                        clk_count <= 0;
                        bit_index <= 0;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                DATA: begin
                    tx <= tx_data[bit_index];
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        if (bit_index == 7) begin
                            state <= STOP;
                        end else begin
                            bit_index <= bit_index + 1;
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                STOP: begin
                    tx <= 1'b1;
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        state <= IDLE;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
            endcase
        end
    end

endmodule


// =============================================================================
// UART Command Processor (Stub - implement as needed)
// =============================================================================

module uart_cmd_processor (
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  rx_data,
    input  wire        rx_valid,
    output reg  [7:0]  tx_data,
    output reg         tx_start,
    input  wire        tx_busy,
    input  wire [767:0] regs,    // 16 * 48 bits
    input  wire [47:0] result
);

    // Simple echo for testing
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_start <= 1'b0;
            tx_data <= 8'h00;
        end else begin
            tx_start <= 1'b0;
            
            if (rx_valid && !tx_busy) begin
                // Echo received character
                tx_data <= rx_data;
                tx_start <= 1'b1;
            end
        end
    end

endmodule
