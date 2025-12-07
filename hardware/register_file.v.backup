/*
 * ============================================================================
 * Pentary Register File
 * ============================================================================
 * 
 * Register file with 32 registers, each storing 16 pentary digits (48 bits).
 * Supports dual-port read and single-port write for pipeline efficiency.
 * 
 * FEATURES:
 *   - 32 registers (R0-R31)
 *   - R0 is hardwired to zero
 *   - Dual-port read (2 simultaneous reads)
 *   - Single-port write
 *   - Bypass logic for data hazards
 *   - Reset capability
 * 
 * TIMING:
 *   - Read: Combinational (0 cycle latency)
 *   - Write: Registered (1 cycle latency)
 * 
 * ============================================================================
 */

module RegisterFile (
    input         clk,
    input         reset,
    
    // Read ports
    input  [4:0]  read_addr1,       // First read address (0-31)
    input  [4:0]  read_addr2,       // Second read address (0-31)
    output [47:0] read_data1,       // First read data
    output [47:0] read_data2,       // Second read data
    
    // Write port
    input  [4:0]  write_addr,       // Write address (0-31)
    input  [47:0] write_data,       // Write data
    input         write_enable      // Write enable
);

    // ========================================================================
    // Register Array
    // ========================================================================
    
    // 32 registers × 48 bits each
    reg [47:0] registers [0:31];
    
    // ========================================================================
    // Write Logic
    // ========================================================================
    
    integer i;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize all registers to zero
            for (i <= 0; i < 32; i <= i + 1) begin
                registers[i] <= 48'b0;
            end
        end else if (write_enable && write_addr != 5'b0) begin
            // Write to register (except R0 which is hardwired to zero)
            registers[write_addr] <= write_data;
        end
    end
    
    // ========================================================================
    // Read Logic with Bypass
    // ========================================================================
    
    // Read port 1 with bypass
    assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :  // R0 is always zero
                       (write_enable && read_addr1 == write_addr) ? write_data :  // Bypass
                       registers[read_addr1];
    
    // Read port 2 with bypass
    assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :  // R0 is always zero
                       (write_enable && read_addr2 == write_addr) ? write_data :  // Bypass
                       registers[read_addr2];

endmodule


/*
 * ============================================================================
 * Extended Register File with Special Registers
 * ============================================================================
 * 
 * Extended version with special-purpose registers for system state.
 * 
 * SPECIAL REGISTERS:
 *   R0:  Zero (hardwired to 0)
 *   R1:  Return address
 *   R2:  Stack pointer
 *   R3:  Global pointer
 *   R4:  Thread pointer
 *   R5-R31: General purpose
 * 
 * ADDITIONAL FEATURES:
 *   - Program counter (PC)
 *   - Status register (SR)
 *   - Exception registers
 * 
 * ============================================================================
 */

module ExtendedRegisterFile (
    input         clk,
    input         reset,
    
    // Read ports
    input  [4:0]  read_addr1,
    input  [4:0]  read_addr2,
    output [47:0] read_data1,
    output [47:0] read_data2,
    
    // Write port
    input  [4:0]  write_addr,
    input  [47:0] write_data,
    input         write_enable,
    
    // Special registers
    input  [47:0] pc_in,            // Program counter input
    input         pc_write,
    output [47:0] pc_out,           // Program counter output
    
    input  [31:0] status_in,        // Status register input
    input         status_write,
    output [31:0] status_out,       // Status register output
    
    // Debug interface
    input  [4:0]  debug_addr,
    output [47:0] debug_data
);

    // ========================================================================
    // Main Register File
    // ========================================================================
    
    reg [47:0] registers [0:31];
    
    // ========================================================================
    // Special Registers
    // ========================================================================
    
    reg [47:0] pc;          // Program counter
    reg [31:0] status;      // Status register
    reg [47:0] epc;         // Exception program counter
    reg [31:0] cause;       // Exception cause
    
    // ========================================================================
    // Write Logic
    // ========================================================================
    
    integer i;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Initialize all registers
            for (i <= 0; i < 32; i <= i + 1) begin
                registers[i] <= 48'b0;
            end
            pc <= 48'b0;
            status <= 32'b0;
            epc <= 48'b0;
            cause <= 32'b0;
        end else begin
            // Write to general-purpose registers
            if (write_enable && write_addr != 5'b0) begin
                registers[write_addr] <= write_data;
            end
            
            // Write to program counter
            if (pc_write) begin
                pc <= pc_in;
            end
            
            // Write to status register
            if (status_write) begin
                status <= status_in;
            end
        end
    end
    
    // ========================================================================
    // Read Logic
    // ========================================================================
    
    assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :
                       (write_enable && read_addr1 == write_addr) ? write_data :
                       registers[read_addr1];
    
    assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :
                       (write_enable && read_addr2 == write_addr) ? write_data :
                       registers[read_addr2];
    
    assign pc_out = pc;
    assign status_out = status;
    
    // Debug read
    assign debug_data = (debug_addr == 5'b0) ? 48'b0 : registers[debug_addr];

endmodule


/*
 * ============================================================================
 * Register File with Scoreboarding
 * ============================================================================
 * 
 * Advanced register file with scoreboarding for out-of-order execution.
 * Tracks which registers are currently being written.
 * 
 * ============================================================================
 */

module RegisterFileWithScoreboard (
    input         clk,
    input         reset,
    
    // Read ports
    input  [4:0]  read_addr1,
    input  [4:0]  read_addr2,
    output [47:0] read_data1,
    output [47:0] read_data2,
    output        read_valid1,      // 1 if data is valid (not pending write)
    output        read_valid2,
    
    // Write port
    input  [4:0]  write_addr,
    input  [47:0] write_data,
    input         write_enable,
    
    // Scoreboard interface
    input  [4:0]  reserve_addr,     // Reserve register for future write
    input         reserve_enable,
    input  [4:0]  release_addr,     // Release register (write complete)
    input         release_enable
);

    // ========================================================================
    // Register Array and Scoreboard
    // ========================================================================
    
    reg [47:0] registers [0:31];
    reg [31:0] scoreboard;          // 1 bit per register (1 = pending write)
    
    // ========================================================================
    // Write Logic
    // ========================================================================
    
    integer i;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i <= 0; i < 32; i <= i + 1) begin
                registers[i] <= 48'b0;
            end
            scoreboard <= 32'b0;
        end else begin
            // Write to register
            if (write_enable && write_addr != 5'b0) begin
                registers[write_addr] <= write_data;
            end
            
            // Update scoreboard
            if (reserve_enable && reserve_addr != 5'b0) begin
                scoreboard[reserve_addr] <= 1'b1;
            end
            
            if (release_enable && release_addr != 5'b0) begin
                scoreboard[release_addr] <= 1'b0;
            end
        end
    end
    
    // ========================================================================
    // Read Logic with Validity Check
    // ========================================================================
    
    assign read_data1 = (read_addr1 == 5'b0) ? 48'b0 :
                       (write_enable && read_addr1 == write_addr) ? write_data :
                       registers[read_addr1];
    
    assign read_data2 = (read_addr2 == 5'b0) ? 48'b0 :
                       (write_enable && read_addr2 == write_addr) ? write_data :
                       registers[read_addr2];
    
    // Check if data is valid (not waiting for pending write)
    assign read_valid1 = (read_addr1 == 5'b0) ? 1'b1 :
                        (write_enable && read_addr1 == write_addr) ? 1'b1 :
                        ~scoreboard[read_addr1];
    
    assign read_valid2 = (read_addr2 == 5'b0) ? 1'b1 :
                        (write_enable && read_addr2 == write_addr) ? 1'b1 :
                        ~scoreboard[read_addr2];

endmodule


/*
 * ============================================================================
 * Multi-Banked Register File
 * ============================================================================
 * 
 * Register file with multiple banks for higher bandwidth.
 * Useful for SIMD or multi-threaded execution.
 * 
 * ============================================================================
 */

module MultiBankedRegisterFile #(
    parameter NUM_BANKS = 4
) (
    input         clk,
    input         reset,
    
    // Read ports (one per bank)
    input  [4:0]  read_addr [0:NUM_BANKS-1],
    output [47:0] read_data [0:NUM_BANKS-1],
    
    // Write ports (one per bank)
    input  [4:0]  write_addr [0:NUM_BANKS-1],
    input  [47:0] write_data [0:NUM_BANKS-1],
    input         write_enable [0:NUM_BANKS-1]
);

    // Generate one register file per bank
    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : bank_gen
            RegisterFile bank (
                .clk(clk),
                .reset(reset),
                .read_addr1(read_addr[i]),
                .read_addr2(5'b0),
                .read_data1(read_data[i]),
                .read_data2(),
                .write_addr(write_addr[i]),
                .write_data(write_data[i]),
                .write_enable(write_enable[i])
            );
        end
    endgenerate

endmodule