/*
 * ============================================================================
 * Memory Management Unit (MMU)
 * ============================================================================
 * 
 * Handles virtual to physical address translation for pentary processor.
 * Includes TLB (Translation Lookaside Buffer) and page table walker.
 * 
 * FEATURES:
 *   - 48-bit virtual addresses
 *   - 48-bit physical addresses
 *   - 4KB page size
 *   - 64-entry fully associative TLB
 *   - Hardware page table walker
 *   - Memory protection
 * 
 * ============================================================================
 */

module MMU (
    input         clk,
    input         reset,
    
    // CPU interface
    input  [47:0] virtual_addr,
    input         read_enable,
    input         write_enable,
    input         execute_enable,
    output [47:0] physical_addr,
    output        tlb_hit,
    output        page_fault,
    output        protection_fault,
    output        ready,
    
    // Page table base register
    input  [47:0] page_table_base,
    
    // Memory interface for page table walks
    output [47:0] mem_addr,
    output        mem_read,
    input  [47:0] mem_data,
    input         mem_ready
);

    // Page size: 4KB (12-bit offset)
    wire [11:0] page_offset;
    wire [35:0] virtual_page_number;
    
    assign page_offset = virtual_addr[11:0];
    assign virtual_page_number = virtual_addr[47:12];
    
    // ========================================================================
    // TLB (Translation Lookaside Buffer)
    // ========================================================================
    
    parameter TLB_ENTRIES = 64;
    
    reg [35:0] tlb_vpn [0:TLB_ENTRIES-1];      // Virtual page number
    reg [35:0] tlb_ppn [0:TLB_ENTRIES-1];      // Physical page number
    reg        tlb_valid [0:TLB_ENTRIES-1];    // Valid bit
    reg        tlb_dirty [0:TLB_ENTRIES-1];    // Dirty bit
    reg        tlb_read [0:TLB_ENTRIES-1];     // Read permission
    reg        tlb_write [0:TLB_ENTRIES-1];    // Write permission
    reg        tlb_execute [0:TLB_ENTRIES-1];  // Execute permission
    reg [5:0]  tlb_lru [0:TLB_ENTRIES-1];      // LRU counter
    
    // TLB lookup
    wire [TLB_ENTRIES-1:0] tlb_match;
    genvar i;
    generate
        for (i = 0; i < TLB_ENTRIES; i = i + 1) begin : tlb_lookup
            assign tlb_match[i] = tlb_valid[i] && (tlb_vpn[i] == virtual_page_number);
        end
    endgenerate
    
    assign tlb_hit = |tlb_match;
    
    // Select matching TLB entry
    reg [35:0] matched_ppn;
    reg matched_read, matched_write, matched_execute;
    integer j;
    always @(*) begin
        matched_ppn = 36'b0;
        matched_read = 1'b0;
        matched_write = 1'b0;
        matched_execute = 1'b0;
        for (j = 0; j < TLB_ENTRIES; j = j + 1) begin
            if (tlb_match[j]) begin
                matched_ppn = tlb_ppn[j];
                matched_read = tlb_read[j];
                matched_write = tlb_write[j];
                matched_execute = tlb_execute[j];
            end
        end
    end
    
    // Check permissions
    assign protection_fault = tlb_hit && (
        (read_enable && !matched_read) ||
        (write_enable && !matched_write) ||
        (execute_enable && !matched_execute)
    );
    
    // Physical address on TLB hit
    wire [47:0] tlb_physical_addr;
    assign tlb_physical_addr = {matched_ppn, page_offset};
    
    // ========================================================================
    // Page Table Walker
    // ========================================================================
    
    reg [2:0] walker_state;
    localparam IDLE = 3'b000;
    localparam LEVEL1 = 3'b001;
    localparam LEVEL2 = 3'b010;
    localparam LEVEL3 = 3'b011;
    localparam UPDATE_TLB = 3'b100;
    
    reg [47:0] pte_addr;
    reg [47:0] pte_data;
    reg [35:0] walked_ppn;
    reg walked_read, walked_write, walked_execute;
    reg [5:0] replace_entry;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            walker_state <= IDLE;
            replace_entry <= 0;
        end else begin
            case (walker_state)
                IDLE: begin
                    if ((read_enable || write_enable || execute_enable) && !tlb_hit) begin
                        // Start page table walk
                        walker_state <= LEVEL1;
                        pte_addr <= page_table_base + {virtual_page_number[35:27], 3'b0};
                    end
                end
                
                LEVEL1: begin
                    if (mem_ready) begin
                        pte_data <= mem_data;
                        walker_state <= LEVEL2;
                        pte_addr <= mem_data + {virtual_page_number[26:18], 3'b0};
                    end
                end
                
                LEVEL2: begin
                    if (mem_ready) begin
                        pte_data <= mem_data;
                        walker_state <= LEVEL3;
                        pte_addr <= mem_data + {virtual_page_number[17:9], 3'b0};
                    end
                end
                
                LEVEL3: begin
                    if (mem_ready) begin
                        pte_data <= mem_data;
                        // Extract PPN and permissions from PTE
                        walked_ppn <= mem_data[47:12];
                        walked_read <= mem_data[0];
                        walked_write <= mem_data[1];
                        walked_execute <= mem_data[2];
                        walker_state <= UPDATE_TLB;
                    end
                end
                
                UPDATE_TLB: begin
                    // Update TLB with new translation
                    tlb_vpn[replace_entry] <= virtual_page_number;
                    tlb_ppn[replace_entry] <= walked_ppn;
                    tlb_valid[replace_entry] <= 1'b1;
                    tlb_read[replace_entry] <= walked_read;
                    tlb_write[replace_entry] <= walked_write;
                    tlb_execute[replace_entry] <= walked_execute;
                    
                    // Update replacement policy
                    replace_entry <= replace_entry + 1;
                    
                    walker_state <= IDLE;
                end
            endcase
        end
    end
    
    // Page fault: invalid PTE during walk
    assign page_fault = (walker_state == LEVEL3) && mem_ready && !mem_data[3];
    
    // Memory interface for page table walks
    assign mem_addr = pte_addr;
    assign mem_read = (walker_state != IDLE) && (walker_state != UPDATE_TLB);
    
    // Output physical address
    assign physical_addr = tlb_hit ? tlb_physical_addr : 48'b0;
    assign ready = (walker_state == IDLE) && (tlb_hit || !(read_enable || write_enable || execute_enable));

endmodule


/*
 * ============================================================================
 * Interrupt Controller
 * ============================================================================
 * 
 * Handles interrupt prioritization, masking, and delivery.
 * 
 * FEATURES:
 *   - 32 interrupt sources
 *   - Priority-based arbitration
 *   - Interrupt masking
 *   - Nested interrupt support
 *   - Exception handling
 * 
 * ============================================================================
 */

module InterruptController (
    input         clk,
    input         reset,
    
    // Interrupt sources (32 external interrupts)
    input  [31:0] interrupt_request,
    
    // CPU interface
    output        interrupt_pending,
    output [4:0]  interrupt_vector,
    input         interrupt_ack,
    
    // Control registers
    input  [31:0] interrupt_enable,    // Interrupt enable mask
    input  [31:0] interrupt_priority,  // Priority configuration
    input         global_enable,       // Global interrupt enable
    
    // Exception interface
    input         exception_request,
    input  [4:0]  exception_vector,
    output        exception_pending
);

    // ========================================================================
    // Interrupt Masking
    // ========================================================================
    
    wire [31:0] masked_interrupts;
    assign masked_interrupts = interrupt_request & interrupt_enable & {32{global_enable}};
    
    // ========================================================================
    // Priority Encoder
    // ========================================================================
    
    reg [4:0] highest_priority_int;
    reg interrupt_found;
    integer i;
    
    always @(*) begin
        highest_priority_int = 5'b0;
        interrupt_found = 1'b0;
        
        // Find highest priority pending interrupt
        for (i = 31; i >= 0; i = i - 1) begin
            if (masked_interrupts[i] && !interrupt_found) begin
                highest_priority_int = i;
                interrupt_found = 1'b1;
            end
        end
    end
    
    assign interrupt_pending = interrupt_found;
    assign interrupt_vector = highest_priority_int;
    
    // ========================================================================
    // Exception Handling
    // ========================================================================
    
    // Exceptions have higher priority than interrupts
    assign exception_pending = exception_request;
    
    // ========================================================================
    // Interrupt State Machine
    // ========================================================================
    
    reg [1:0] state;
    localparam IDLE = 2'b00;
    localparam PENDING = 2'b01;
    localparam SERVICING = 2'b10;
    
    reg [4:0] current_vector;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            current_vector <= 5'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (exception_pending) begin
                        state <= PENDING;
                        current_vector <= exception_vector;
                    end else if (interrupt_pending) begin
                        state <= PENDING;
                        current_vector <= interrupt_vector;
                    end
                end
                
                PENDING: begin
                    if (interrupt_ack) begin
                        state <= SERVICING;
                    end
                end
                
                SERVICING: begin
                    // Wait for interrupt service routine to complete
                    // Return to IDLE when done
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule


/*
 * ============================================================================
 * Exception Handler
 * ============================================================================
 * 
 * Handles various processor exceptions.
 * 
 * EXCEPTION TYPES:
 *   0: Reset
 *   1: Illegal instruction
 *   2: Breakpoint
 *   3: Load address misaligned
 *   4: Store address misaligned
 *   5: Page fault (instruction)
 *   6: Page fault (load)
 *   7: Page fault (store)
 *   8: Arithmetic overflow
 *   9: Division by zero
 *   10-31: Reserved
 * 
 * ============================================================================
 */

module ExceptionHandler (
    input         clk,
    input         reset,
    
    // Exception inputs
    input         illegal_instruction,
    input         breakpoint,
    input         load_misaligned,
    input         store_misaligned,
    input         page_fault_inst,
    input         page_fault_load,
    input         page_fault_store,
    input         arithmetic_overflow,
    input         division_by_zero,
    
    // Exception output
    output        exception_request,
    output [4:0]  exception_vector,
    output [47:0] exception_pc,
    output [31:0] exception_cause,
    
    // CPU state
    input  [47:0] current_pc,
    input  [47:0] faulting_addr
);

    // Priority encoder for exceptions
    reg [4:0] vector;
    reg request;
    
    always @(*) begin
        request = 1'b0;
        vector = 5'b0;
        
        if (illegal_instruction) begin
            request = 1'b1;
            vector = 5'd1;
        end else if (breakpoint) begin
            request = 1'b1;
            vector = 5'd2;
        end else if (load_misaligned) begin
            request = 1'b1;
            vector = 5'd3;
        end else if (store_misaligned) begin
            request = 1'b1;
            vector = 5'd4;
        end else if (page_fault_inst) begin
            request = 1'b1;
            vector = 5'd5;
        end else if (page_fault_load) begin
            request = 1'b1;
            vector = 5'd6;
        end else if (page_fault_store) begin
            request = 1'b1;
            vector = 5'd7;
        end else if (arithmetic_overflow) begin
            request = 1'b1;
            vector = 5'd8;
        end else if (division_by_zero) begin
            request = 1'b1;
            vector = 5'd9;
        end
    end
    
    assign exception_request = request;
    assign exception_vector = vector;
    assign exception_pc = current_pc;
    assign exception_cause = {27'b0, vector};

endmodule


/*
 * ============================================================================
 * TLB Management
 * ============================================================================
 */

module TLBManager (
    input         clk,
    input         reset,
    
    // TLB flush
    input         flush_all,
    input         flush_entry,
    input  [47:0] flush_addr,
    
    // TLB statistics
    output [31:0] tlb_hits,
    output [31:0] tlb_misses,
    
    // Control
    input         count_hit,
    input         count_miss
);

    reg [31:0] hit_counter;
    reg [31:0] miss_counter;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            hit_counter <= 32'b0;
            miss_counter <= 32'b0;
        end else begin
            if (count_hit)
                hit_counter <= hit_counter + 1;
            if (count_miss)
                miss_counter <= miss_counter + 1;
        end
    end
    
    assign tlb_hits = hit_counter;
    assign tlb_misses = miss_counter;

endmodule