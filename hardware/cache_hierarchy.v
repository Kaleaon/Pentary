/*
 * ============================================================================
 * Pentary Cache Hierarchy
 * ============================================================================
 * 
 * Complete cache hierarchy for pentary processor.
 * Includes L1 instruction cache, L1 data cache, and L2 unified cache.
 * 
 * SPECIFICATIONS:
 *   L1 I-Cache: 32KB, 4-way set associative, 64-byte lines
 *   L1 D-Cache: 32KB, 4-way set associative, 64-byte lines
 *   L2 Cache:   256KB, 8-way set associative, 64-byte lines
 * 
 * ============================================================================
 */

module L1_InstructionCache (
    input         clk,
    input         reset,
    
    // CPU interface
    input  [47:0] addr,
    input         read_enable,
    output [31:0] instruction,
    output        hit,
    output        ready,
    
    // L2 interface
    output [47:0] l2_addr,
    output        l2_read,
    input  [511:0] l2_data,
    input         l2_ready
);
    // Cache parameters
    parameter CACHE_SIZE = 32768;      // 32KB
    parameter LINE_SIZE = 64;          // 64 bytes per line
    parameter WAYS = 4;                // 4-way set associative
    parameter NUM_SETS = CACHE_SIZE / (LINE_SIZE * WAYS);  // 128 sets
    
    // Address breakdown
    wire [5:0]  offset;     // Byte offset within line (6 bits for 64 bytes)
    wire [6:0]  index;      // Set index (7 bits for 128 sets)
    wire [34:0] tag;        // Tag (remaining bits)
    
    assign offset = addr[5:0];
    assign index = addr[12:6];
    assign tag = addr[47:13];
    
    // Cache storage
    reg [511:0] data [0:NUM_SETS-1][0:WAYS-1];     // Data array
    reg [34:0]  tags [0:NUM_SETS-1][0:WAYS-1];     // Tag array
    reg         valid [0:NUM_SETS-1][0:WAYS-1];    // Valid bits
    reg [1:0]   lru [0:NUM_SETS-1][0:WAYS-1];      // LRU bits
    
    // Hit detection
    wire [WAYS-1:0] way_hit;
    genvar i;
    generate
        for (i = 0; i < WAYS; i = i + 1) begin : hit_detect
            assign way_hit[i] = valid[index][i] && (tags[index][i] == tag);
        end
    endgenerate
    
    assign hit = |way_hit;
    
    // Select data from hitting way
    reg [511:0] selected_line;
    integer j;
    always @(*) begin
        selected_line = 512'b0;
        for (j = 0; j < WAYS; j = j + 1) begin
            if (way_hit[j])
                selected_line = data[index][j];
        end
    end
    
    // Extract instruction from cache line
    wire [3:0] word_offset;
    assign word_offset = offset[5:2];  // Word offset within line
    assign instruction = selected_line[word_offset*32 +: 32];
    
    // State machine for cache misses
    reg [1:0] state;
    localparam IDLE = 2'b00;
    localparam FETCH = 2'b01;
    localparam FILL = 2'b10;
    
    reg [1:0] replace_way;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            replace_way <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (read_enable && !hit) begin
                        state <= FETCH;
                        // Select LRU way for replacement
                        replace_way <= lru[index][0];
                    end
                end
                
                FETCH: begin
                    if (l2_ready) begin
                        state <= FILL;
                    end
                end
                
                FILL: begin
                    // Fill cache line
                    data[index][replace_way] <= l2_data;
                    tags[index][replace_way] <= tag;
                    valid[index][replace_way] <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign ready = (state == IDLE) && (hit || !read_enable);
    assign l2_addr = addr;
    assign l2_read = (state == FETCH);

endmodule


/*
 * ============================================================================
 * L1 Data Cache
 * ============================================================================
 */

module L1_DataCache (
    input         clk,
    input         reset,
    
    // CPU interface
    input  [47:0] addr,
    input         read_enable,
    input         write_enable,
    input  [47:0] write_data,
    output [47:0] read_data,
    output        hit,
    output        ready,
    
    // L2 interface
    output [47:0] l2_addr,
    output        l2_read,
    output        l2_write,
    output [511:0] l2_write_data,
    input  [511:0] l2_read_data,
    input         l2_ready
);
    // Cache parameters
    parameter CACHE_SIZE = 32768;
    parameter LINE_SIZE = 64;
    parameter WAYS = 4;
    parameter NUM_SETS = CACHE_SIZE / (LINE_SIZE * WAYS);
    
    // Address breakdown
    wire [5:0]  offset;
    wire [6:0]  index;
    wire [34:0] tag;
    
    assign offset = addr[5:0];
    assign index = addr[12:6];
    assign tag = addr[47:13];
    
    // Cache storage
    reg [511:0] data [0:NUM_SETS-1][0:WAYS-1];
    reg [34:0]  tags [0:NUM_SETS-1][0:WAYS-1];
    reg         valid [0:NUM_SETS-1][0:WAYS-1];
    reg         dirty [0:NUM_SETS-1][0:WAYS-1];
    reg [1:0]   lru [0:NUM_SETS-1][0:WAYS-1];
    
    // Hit detection
    wire [WAYS-1:0] way_hit;
    genvar i;
    generate
        for (i = 0; i < WAYS; i = i + 1) begin : hit_detect
            assign way_hit[i] = valid[index][i] && (tags[index][i] == tag);
        end
    endgenerate
    
    assign hit = |way_hit;
    
    // Select data from hitting way
    reg [511:0] selected_line;
    reg [1:0] hit_way;
    integer j;
    always @(*) begin
        selected_line = 512'b0;
        hit_way = 0;
        for (j = 0; j < WAYS; j = j + 1) begin
            if (way_hit[j]) begin
                selected_line = data[index][j];
                hit_way = j;
            end
        end
    end
    
    // Extract word from cache line
    wire [3:0] word_offset;
    assign word_offset = offset[5:2];
    assign read_data = selected_line[word_offset*48 +: 48];
    
    // Write logic
    reg [511:0] updated_line;
    always @(*) begin
        updated_line = selected_line;
        if (write_enable && hit) begin
            updated_line[word_offset*48 +: 48] = write_data;
        end
    end
    
    // State machine
    reg [2:0] state;
    localparam IDLE = 3'b000;
    localparam WRITEBACK = 3'b001;
    localparam FETCH = 3'b010;
    localparam FILL = 3'b011;
    
    reg [1:0] replace_way;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            replace_way <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (write_enable && hit) begin
                        // Write hit
                        data[index][hit_way] <= updated_line;
                        dirty[index][hit_way] <= 1'b1;
                    end else if ((read_enable || write_enable) && !hit) begin
                        // Miss - check if need writeback
                        replace_way <= lru[index][0];
                        if (dirty[index][lru[index][0]]) begin
                            state <= WRITEBACK;
                        end else begin
                            state <= FETCH;
                        end
                    end
                end
                
                WRITEBACK: begin
                    if (l2_ready) begin
                        state <= FETCH;
                    end
                end
                
                FETCH: begin
                    if (l2_ready) begin
                        state <= FILL;
                    end
                end
                
                FILL: begin
                    data[index][replace_way] <= l2_read_data;
                    tags[index][replace_way] <= tag;
                    valid[index][replace_way] <= 1'b1;
                    dirty[index][replace_way] <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign ready = (state == IDLE);
    assign l2_addr = (state == WRITEBACK) ? {tags[index][replace_way], index, 6'b0} : addr;
    assign l2_read = (state == FETCH);
    assign l2_write = (state == WRITEBACK);
    assign l2_write_data = data[index][replace_way];

endmodule


/*
 * ============================================================================
 * L2 Unified Cache
 * ============================================================================
 */

module L2_UnifiedCache (
    input         clk,
    input         reset,
    
    // L1 I-Cache interface
    input  [47:0] l1i_addr,
    input         l1i_read,
    output [511:0] l1i_data,
    output        l1i_ready,
    
    // L1 D-Cache interface
    input  [47:0] l1d_addr,
    input         l1d_read,
    input         l1d_write,
    input  [511:0] l1d_write_data,
    output [511:0] l1d_read_data,
    output        l1d_ready,
    
    // Main memory interface
    output [47:0] mem_addr,
    output        mem_read,
    output        mem_write,
    output [511:0] mem_write_data,
    input  [511:0] mem_read_data,
    input         mem_ready
);
    // Cache parameters
    parameter CACHE_SIZE = 262144;     // 256KB
    parameter LINE_SIZE = 64;
    parameter WAYS = 8;
    parameter NUM_SETS = CACHE_SIZE / (LINE_SIZE * WAYS);  // 512 sets
    
    // Arbitration between L1 I-Cache and L1 D-Cache
    reg l1d_priority;
    wire [47:0] selected_addr;
    wire selected_read, selected_write;
    wire [511:0] selected_write_data;
    
    assign selected_addr = l1d_priority ? l1d_addr : l1i_addr;
    assign selected_read = l1d_priority ? l1d_read : l1i_read;
    assign selected_write = l1d_priority ? l1d_write : 1'b0;
    assign selected_write_data = l1d_write_data;
    
    // Address breakdown
    wire [5:0]  offset;
    wire [8:0]  index;
    wire [32:0] tag;
    
    assign offset = selected_addr[5:0];
    assign index = selected_addr[14:6];
    assign tag = selected_addr[47:15];
    
    // Cache storage
    reg [511:0] data [0:NUM_SETS-1][0:WAYS-1];
    reg [32:0]  tags [0:NUM_SETS-1][0:WAYS-1];
    reg         valid [0:NUM_SETS-1][0:WAYS-1];
    reg         dirty [0:NUM_SETS-1][0:WAYS-1];
    reg [2:0]   lru [0:NUM_SETS-1][0:WAYS-1];
    
    // Hit detection
    wire [WAYS-1:0] way_hit;
    genvar i;
    generate
        for (i = 0; i < WAYS; i = i + 1) begin : hit_detect
            assign way_hit[i] = valid[index][i] && (tags[index][i] == tag);
        end
    endgenerate
    
    wire hit;
    assign hit = |way_hit;
    
    // Select data from hitting way
    reg [511:0] selected_line;
    integer j;
    always @(*) begin
        selected_line = 512'b0;
        for (j = 0; j < WAYS; j = j + 1) begin
            if (way_hit[j])
                selected_line = data[index][j];
        end
    end
    
    assign l1i_data = selected_line;
    assign l1d_read_data = selected_line;
    
    // State machine
    reg [2:0] state;
    localparam IDLE = 3'b000;
    localparam WRITEBACK = 3'b001;
    localparam FETCH = 3'b010;
    localparam FILL = 3'b011;
    
    reg [2:0] replace_way;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            l1d_priority <= 0;
        end else begin
            case (state)
                IDLE: begin
                    // Arbitrate between L1 caches
                    if (l1d_read || l1d_write) begin
                        l1d_priority <= 1'b1;
                    end else if (l1i_read) begin
                        l1d_priority <= 1'b0;
                    end
                    
                    if ((selected_read || selected_write) && !hit) begin
                        replace_way <= lru[index][0];
                        if (dirty[index][lru[index][0]]) begin
                            state <= WRITEBACK;
                        end else begin
                            state <= FETCH;
                        end
                    end
                end
                
                WRITEBACK: begin
                    if (mem_ready) begin
                        state <= FETCH;
                    end
                end
                
                FETCH: begin
                    if (mem_ready) begin
                        state <= FILL;
                    end
                end
                
                FILL: begin
                    data[index][replace_way] <= mem_read_data;
                    tags[index][replace_way] <= tag;
                    valid[index][replace_way] <= 1'b1;
                    dirty[index][replace_way] <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign l1i_ready = (state == IDLE) && !l1d_priority && (hit || !l1i_read);
    assign l1d_ready = (state == IDLE) && l1d_priority && (hit || (!l1d_read && !l1d_write));
    
    assign mem_addr = (state == WRITEBACK) ? {tags[index][replace_way], index, 6'b0} : selected_addr;
    assign mem_read = (state == FETCH);
    assign mem_write = (state == WRITEBACK);
    assign mem_write_data = data[index][replace_way];

endmodule


/*
 * ============================================================================
 * Cache Coherency Controller (MESI Protocol)
 * ============================================================================
 */

module CacheCoherencyController (
    input         clk,
    input         reset,
    
    // Core 0 interface
    input  [47:0] core0_addr,
    input         core0_read,
    input         core0_write,
    output [1:0]  core0_state,
    
    // Core 1 interface
    input  [47:0] core1_addr,
    input         core1_read,
    input         core1_write,
    output [1:0]  core1_state,
    
    // Snoop interface
    output        snoop_invalidate,
    output [47:0] snoop_addr
);
    // MESI states
    localparam INVALID = 2'b00;
    localparam SHARED = 2'b01;
    localparam EXCLUSIVE = 2'b10;
    localparam MODIFIED = 2'b11;
    
    // State tracking (simplified - would need full cache line tracking)
    reg [1:0] core0_cache_state;
    reg [1:0] core1_cache_state;
    
    assign core0_state = core0_cache_state;
    assign core1_state = core1_cache_state;
    
    // Coherency logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            core0_cache_state <= INVALID;
            core1_cache_state <= INVALID;
        end else begin
            // Handle core 0 operations
            if (core0_write) begin
                core0_cache_state <= MODIFIED;
                if (core1_cache_state != INVALID)
                    core1_cache_state <= INVALID;
            end else if (core0_read) begin
                if (core0_cache_state == INVALID) begin
                    if (core1_cache_state == MODIFIED || core1_cache_state == EXCLUSIVE)
                        core0_cache_state <= SHARED;
                    else
                        core0_cache_state <= EXCLUSIVE;
                end
            end
            
            // Handle core 1 operations
            if (core1_write) begin
                core1_cache_state <= MODIFIED;
                if (core0_cache_state != INVALID)
                    core0_cache_state <= INVALID;
            end else if (core1_read) begin
                if (core1_cache_state == INVALID) begin
                    if (core0_cache_state == MODIFIED || core0_cache_state == EXCLUSIVE)
                        core1_cache_state <= SHARED;
                    else
                        core1_cache_state <= EXCLUSIVE;
                end
            end
        end
    end
    
    assign snoop_invalidate = (core0_write && core1_cache_state != INVALID) ||
                             (core1_write && core0_cache_state != INVALID);
    assign snoop_addr = core0_write ? core0_addr : core1_addr;

endmodule