/**
 * =============================================================================
 * Pentary Processor - Lattice iCE40 Optimized Implementation
 * =============================================================================
 * 
 * Target: iCE40UP5K or iCE40HX8K
 * Toolchain: Yosys + nextpnr-ice40
 * 
 * Features:
 *   - 4-trit (12-bit) data path (reduced for iCE40 resources)
 *   - 8 pentary registers
 *   - Basic ALU (add, sub, neg)
 *   - UART debug interface
 * 
 * Resources (iCE40UP5K):
 *   - LUT4: ~1,500
 *   - DFF: ~1,000
 *   - BRAM: 4 blocks
 * 
 * =============================================================================
 */

module pentary_ice40 (
    input  wire clk,           // 12 MHz from oscillator
    input  wire rst_n,         // Active-low reset
    
    // UART Debug Interface
    input  wire uart_rx,
    output wire uart_tx,
    
    // Status LEDs
    output wire [3:0] led,
    
    // SPI Interface (optional, for external memory)
    output wire spi_clk,
    output wire spi_mosi,
    input  wire spi_miso,
    output wire spi_cs_n
);

    // =========================================================================
    // Parameters
    // =========================================================================
    
    localparam DIGITS = 4;              // 4 pentary digits
    localparam DATA_WIDTH = DIGITS * 3; // 12 bits
    localparam NUM_REGS = 8;            // 8 registers
    localparam REG_ADDR_WIDTH = 3;      // log2(8)
    
    // Pentary digit encoding
    localparam [2:0] PENT_NEG2 = 3'b000;
    localparam [2:0] PENT_NEG1 = 3'b001;
    localparam [2:0] PENT_ZERO = 3'b010;
    localparam [2:0] PENT_POS1 = 3'b011;
    localparam [2:0] PENT_POS2 = 3'b100;
    
    // ALU opcodes
    localparam [2:0] OP_ADD = 3'b000;
    localparam [2:0] OP_SUB = 3'b001;
    localparam [2:0] OP_NEG = 3'b100;
    localparam [2:0] OP_NOP = 3'b111;
    
    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Reset synchronizer
    reg [1:0] rst_sync;
    wire rst = rst_sync[1];
    
    always @(posedge clk) begin
        rst_sync <= {rst_sync[0], ~rst_n};
    end
    
    // Register file
    reg [DATA_WIDTH-1:0] regs [0:NUM_REGS-1];
    
    // ALU signals
    reg [DATA_WIDTH-1:0] alu_a, alu_b;
    reg [2:0] alu_op;
    wire [DATA_WIDTH-1:0] alu_result;
    wire alu_zero, alu_negative;
    
    // Control signals
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam FETCH = 3'd1;
    localparam DECODE = 3'd2;
    localparam EXECUTE = 3'd3;
    localparam WRITEBACK = 3'd4;
    
    // Debug counter
    reg [23:0] heartbeat_counter;
    
    // =========================================================================
    // ALU Instance
    // =========================================================================
    
    pentary_alu_4digit alu_inst (
        .a(alu_a),
        .b(alu_b),
        .op(alu_op),
        .result(alu_result),
        .zero(alu_zero),
        .negative(alu_negative)
    );
    
    // =========================================================================
    // UART Interface (Simple Debug)
    // =========================================================================
    
    wire [7:0] uart_rx_data;
    wire uart_rx_valid;
    wire uart_tx_ready;
    reg [7:0] uart_tx_data;
    reg uart_tx_valid;
    
    uart_rx #(.CLK_FREQ(12_000_000), .BAUD_RATE(115200)) uart_rx_inst (
        .clk(clk),
        .rst(rst),
        .rx(uart_rx),
        .data(uart_rx_data),
        .valid(uart_rx_valid)
    );
    
    uart_tx #(.CLK_FREQ(12_000_000), .BAUD_RATE(115200)) uart_tx_inst (
        .clk(clk),
        .rst(rst),
        .data(uart_tx_data),
        .valid(uart_tx_valid),
        .tx(uart_tx),
        .ready(uart_tx_ready)
    );
    
    // =========================================================================
    // Debug Command Processor
    // =========================================================================
    
    // Simple command interface:
    // 'R' + addr -> Read register
    // 'W' + addr + data -> Write register
    // 'A' + op + a + b -> ALU operation
    
    reg [2:0] cmd_state;
    reg [7:0] cmd_buffer [0:3];
    reg [1:0] cmd_idx;
    
    always @(posedge clk) begin
        if (rst) begin
            cmd_state <= 0;
            cmd_idx <= 0;
            uart_tx_valid <= 0;
        end else begin
            uart_tx_valid <= 0;
            
            if (uart_rx_valid) begin
                cmd_buffer[cmd_idx] <= uart_rx_data;
                cmd_idx <= cmd_idx + 1;
                
                case (cmd_buffer[0])
                    "R": begin // Read register
                        if (cmd_idx == 1) begin
                            uart_tx_data <= regs[uart_rx_data[2:0]][7:0];
                            uart_tx_valid <= 1;
                            cmd_idx <= 0;
                        end
                    end
                    "W": begin // Write register
                        if (cmd_idx == 2) begin
                            regs[cmd_buffer[1][2:0]] <= {4'b0, uart_rx_data};
                            cmd_idx <= 0;
                        end
                    end
                    "A": begin // ALU operation
                        if (cmd_idx == 3) begin
                            alu_a <= {4'b0, cmd_buffer[2]};
                            alu_b <= {4'b0, uart_rx_data};
                            alu_op <= cmd_buffer[1][2:0];
                            // Result available next cycle
                            cmd_idx <= 0;
                        end
                    end
                    default: begin
                        cmd_idx <= 0;
                    end
                endcase
            end
        end
    end
    
    // =========================================================================
    // Heartbeat LED
    // =========================================================================
    
    always @(posedge clk) begin
        if (rst) begin
            heartbeat_counter <= 0;
        end else begin
            heartbeat_counter <= heartbeat_counter + 1;
        end
    end
    
    assign led[0] = heartbeat_counter[23];  // ~0.7 Hz blink
    assign led[1] = ~alu_zero;
    assign led[2] = alu_negative;
    assign led[3] = uart_tx_valid | uart_rx_valid;
    
    // =========================================================================
    // SPI Interface (unused, directly drive outputs)
    // =========================================================================
    
    assign spi_clk = 1'b0;
    assign spi_mosi = 1'b0;
    assign spi_cs_n = 1'b1;

endmodule


/**
 * =============================================================================
 * 4-Digit Pentary ALU
 * =============================================================================
 */
module pentary_alu_4digit (
    input  wire [11:0] a,
    input  wire [11:0] b,
    input  wire [2:0]  op,
    output reg  [11:0] result,
    output wire        zero,
    output wire        negative
);

    // Digit extraction
    wire [2:0] a_d0 = a[2:0];
    wire [2:0] a_d1 = a[5:3];
    wire [2:0] a_d2 = a[8:6];
    wire [2:0] a_d3 = a[11:9];
    
    wire [2:0] b_d0 = b[2:0];
    wire [2:0] b_d1 = b[5:3];
    wire [2:0] b_d2 = b[8:6];
    wire [2:0] b_d3 = b[11:9];
    
    // Addition results
    wire [2:0] add_d0, add_d1, add_d2, add_d3;
    wire [2:0] add_c0, add_c1, add_c2, add_c3;
    
    // Subtraction results (negate b then add)
    wire [2:0] neg_b_d0 = negate_digit(b_d0);
    wire [2:0] neg_b_d1 = negate_digit(b_d1);
    wire [2:0] neg_b_d2 = negate_digit(b_d2);
    wire [2:0] neg_b_d3 = negate_digit(b_d3);
    
    wire [2:0] sub_d0, sub_d1, sub_d2, sub_d3;
    wire [2:0] sub_c0, sub_c1, sub_c2, sub_c3;
    
    // Addition chain
    pentary_full_adder fa_add0 (.a(a_d0), .b(b_d0), .cin(3'b010), .sum(add_d0), .cout(add_c0));
    pentary_full_adder fa_add1 (.a(a_d1), .b(b_d1), .cin(add_c0), .sum(add_d1), .cout(add_c1));
    pentary_full_adder fa_add2 (.a(a_d2), .b(b_d2), .cin(add_c1), .sum(add_d2), .cout(add_c2));
    pentary_full_adder fa_add3 (.a(a_d3), .b(b_d3), .cin(add_c2), .sum(add_d3), .cout(add_c3));
    
    // Subtraction chain
    pentary_full_adder fa_sub0 (.a(a_d0), .b(neg_b_d0), .cin(3'b010), .sum(sub_d0), .cout(sub_c0));
    pentary_full_adder fa_sub1 (.a(a_d1), .b(neg_b_d1), .cin(sub_c0), .sum(sub_d1), .cout(sub_c1));
    pentary_full_adder fa_sub2 (.a(a_d2), .b(neg_b_d2), .cin(sub_c1), .sum(sub_d2), .cout(sub_c2));
    pentary_full_adder fa_sub3 (.a(a_d3), .b(neg_b_d3), .cin(sub_c2), .sum(sub_d3), .cout(sub_c3));
    
    // Result selection
    always @(*) begin
        case (op)
            3'b000: result = {add_d3, add_d2, add_d1, add_d0}; // ADD
            3'b001: result = {sub_d3, sub_d2, sub_d1, sub_d0}; // SUB
            3'b100: result = {negate_digit(a_d3), negate_digit(a_d2), 
                             negate_digit(a_d1), negate_digit(a_d0)}; // NEG
            default: result = a; // NOP
        endcase
    end
    
    // Flag generation
    assign zero = (result == {3'b010, 3'b010, 3'b010, 3'b010});
    assign negative = (result[11:9] == 3'b000) || (result[11:9] == 3'b001);
    
    // Negate digit function
    function [2:0] negate_digit;
        input [2:0] d;
        case (d)
            3'b000: negate_digit = 3'b100;  // -2 -> +2
            3'b001: negate_digit = 3'b011;  // -1 -> +1
            3'b010: negate_digit = 3'b010;  //  0 ->  0
            3'b011: negate_digit = 3'b001;  // +1 -> -1
            3'b100: negate_digit = 3'b000;  // +2 -> -2
            default: negate_digit = 3'b010;
        endcase
    endfunction

endmodule


/**
 * =============================================================================
 * Pentary Full Adder
 * =============================================================================
 */
module pentary_full_adder (
    input  wire [2:0] a,
    input  wire [2:0] b,
    input  wire [2:0] cin,
    output reg  [2:0] sum,
    output reg  [2:0] cout
);

    // Convert to signed values for arithmetic
    wire signed [2:0] a_val = digit_to_value(a);
    wire signed [2:0] b_val = digit_to_value(b);
    wire signed [2:0] c_val = digit_to_value(cin);
    
    wire signed [3:0] total = a_val + b_val + c_val;
    
    always @(*) begin
        if (total >= 3) begin
            sum = value_to_digit(total - 5);
            cout = 3'b011; // +1
        end else if (total <= -3) begin
            sum = value_to_digit(total + 5);
            cout = 3'b001; // -1
        end else begin
            sum = value_to_digit(total);
            cout = 3'b010; // 0
        end
    end
    
    function signed [2:0] digit_to_value;
        input [2:0] d;
        case (d)
            3'b000: digit_to_value = -2;
            3'b001: digit_to_value = -1;
            3'b010: digit_to_value = 0;
            3'b011: digit_to_value = 1;
            3'b100: digit_to_value = 2;
            default: digit_to_value = 0;
        endcase
    endfunction
    
    function [2:0] value_to_digit;
        input signed [3:0] v;
        case (v)
            -2: value_to_digit = 3'b000;
            -1: value_to_digit = 3'b001;
            0:  value_to_digit = 3'b010;
            1:  value_to_digit = 3'b011;
            2:  value_to_digit = 3'b100;
            default: value_to_digit = 3'b010;
        endcase
    endfunction

endmodule


/**
 * =============================================================================
 * UART Receiver (Simple, 8N1)
 * =============================================================================
 */
module uart_rx #(
    parameter CLK_FREQ = 12_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire clk,
    input  wire rst,
    input  wire rx,
    output reg  [7:0] data,
    output reg  valid
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    reg [2:0] state;
    reg [15:0] clk_count;
    reg [2:0] bit_idx;
    reg [7:0] rx_data;
    reg rx_sync_0, rx_sync_1;
    
    localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;
    
    // Synchronizer
    always @(posedge clk) begin
        rx_sync_0 <= rx;
        rx_sync_1 <= rx_sync_0;
    end
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            valid <= 0;
        end else begin
            valid <= 0;
            
            case (state)
                IDLE: begin
                    if (rx_sync_1 == 0) begin  // Start bit
                        state <= START;
                        clk_count <= CLKS_PER_BIT / 2;
                    end
                end
                
                START: begin
                    if (clk_count == 0) begin
                        if (rx_sync_1 == 0) begin
                            state <= DATA;
                            clk_count <= CLKS_PER_BIT;
                            bit_idx <= 0;
                        end else begin
                            state <= IDLE;
                        end
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
                
                DATA: begin
                    if (clk_count == 0) begin
                        rx_data[bit_idx] <= rx_sync_1;
                        clk_count <= CLKS_PER_BIT;
                        if (bit_idx == 7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
                
                STOP: begin
                    if (clk_count == 0) begin
                        if (rx_sync_1 == 1) begin
                            data <= rx_data;
                            valid <= 1;
                        end
                        state <= IDLE;
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
            endcase
        end
    end

endmodule


/**
 * =============================================================================
 * UART Transmitter (Simple, 8N1)
 * =============================================================================
 */
module uart_tx #(
    parameter CLK_FREQ = 12_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire clk,
    input  wire rst,
    input  wire [7:0] data,
    input  wire valid,
    output reg  tx,
    output wire ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    reg [2:0] state;
    reg [15:0] clk_count;
    reg [2:0] bit_idx;
    reg [7:0] tx_data;
    
    localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;
    
    assign ready = (state == IDLE);
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tx <= 1;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1;
                    if (valid) begin
                        tx_data <= data;
                        state <= START;
                        clk_count <= CLKS_PER_BIT;
                    end
                end
                
                START: begin
                    tx <= 0;
                    if (clk_count == 0) begin
                        state <= DATA;
                        clk_count <= CLKS_PER_BIT;
                        bit_idx <= 0;
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
                
                DATA: begin
                    tx <= tx_data[bit_idx];
                    if (clk_count == 0) begin
                        clk_count <= CLKS_PER_BIT;
                        if (bit_idx == 7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
                
                STOP: begin
                    tx <= 1;
                    if (clk_count == 0) begin
                        state <= IDLE;
                    end else begin
                        clk_count <= clk_count - 1;
                    end
                end
            endcase
        end
    end

endmodule
