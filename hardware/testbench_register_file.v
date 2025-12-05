/*
 * ============================================================================
 * Testbench for Pentary Register File
 * ============================================================================
 * 
 * Comprehensive testbench for RegisterFile module.
 * Tests read/write operations, bypass logic, and R0 hardwiring.
 * 
 * ============================================================================
 */

`timescale 1ns / 1ps

module tb_register_file;

    // Clock and reset
    reg clk;
    reg reset;
    
    // Test signals
    reg [4:0] read_addr1;
    reg [4:0] read_addr2;
    wire [47:0] read_data1;
    wire [47:0] read_data2;
    reg [4:0] write_addr;
    reg [47:0] write_data;
    reg write_enable;
    
    // Test counters
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Instantiate the Unit Under Test (UUT)
    RegisterFile uut (
        .clk(clk),
        .reset(reset),
        .read_addr1(read_addr1),
        .read_addr2(read_addr2),
        .read_data1(read_data1),
        .read_data2(read_data2),
        .write_addr(write_addr),
        .write_data(write_data),
        .write_enable(write_enable)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test procedure
    initial begin
        $display("========================================");
        $display("Register File Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Initialize
        reset = 1;
        write_enable = 0;
        read_addr1 = 0;
        read_addr2 = 0;
        write_addr = 0;
        write_data = 0;
        
        #20;
        reset = 0;
        #10;
        
        // Run tests
        test_reset();
        test_write_read();
        test_r0_hardwired();
        test_dual_port_read();
        test_bypass_logic();
        test_all_registers();
        
        // Display results
        #100;
        $display("\n========================================");
        $display("Test Results:");
        $display("  Total tests: %0d", test_count);
        $display("  Passed:      %0d", pass_count);
        $display("  Failed:      %0d", fail_count);
        $display("========================================");
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Test reset functionality
    task test_reset;
        integer i;
        begin
            $display("\nTesting reset...");
            
            // Write some data
            write_enable = 1;
            for (i = 1; i < 32; i = i + 1) begin
                write_addr = i;
                write_data = i * 48'h1111;
                @(posedge clk);
            end
            write_enable = 0;
            
            // Reset
            reset = 1;
            @(posedge clk);
            @(posedge clk);
            reset = 0;
            @(posedge clk);
            
            // Check all registers are zero
            for (i = 0; i < 32; i = i + 1) begin
                read_addr1 = i;
                #1;
                test_count = test_count + 1;
                if (read_data1 == 48'b0) begin
                    pass_count = pass_count + 1;
                end else begin
                    fail_count = fail_count + 1;
                    $display("FAIL: R%0d not reset to zero", i);
                end
            end
            
            $display("PASS: Reset test completed");
        end
    endtask
    
    // Test basic write and read
    task test_write_read;
        begin
            $display("\nTesting write and read...");
            
            // Write to R1
            write_enable = 1;
            write_addr = 5'd1;
            write_data = 48'hAAAAAAAAAAAA;
            @(posedge clk);
            write_enable = 0;
            @(posedge clk);
            
            // Read from R1
            read_addr1 = 5'd1;
            #1;
            
            test_count = test_count + 1;
            if (read_data1 == 48'hAAAAAAAAAAAA) begin
                pass_count = pass_count + 1;
                $display("PASS: Write/Read R1");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Write/Read R1 - got %h, expected %h", read_data1, 48'hAAAAAAAAAAAA);
            end
            
            // Write to R31
            write_enable = 1;
            write_addr = 5'd31;
            write_data = 48'h555555555555;
            @(posedge clk);
            write_enable = 0;
            @(posedge clk);
            
            // Read from R31
            read_addr1 = 5'd31;
            #1;
            
            test_count = test_count + 1;
            if (read_data1 == 48'h555555555555) begin
                pass_count = pass_count + 1;
                $display("PASS: Write/Read R31");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Write/Read R31");
            end
        end
    endtask
    
    // Test R0 hardwired to zero
    task test_r0_hardwired;
        begin
            $display("\nTesting R0 hardwired to zero...");
            
            // Try to write to R0
            write_enable = 1;
            write_addr = 5'd0;
            write_data = 48'hFFFFFFFFFFFF;
            @(posedge clk);
            write_enable = 0;
            @(posedge clk);
            
            // Read from R0
            read_addr1 = 5'd0;
            #1;
            
            test_count = test_count + 1;
            if (read_data1 == 48'b0) begin
                pass_count = pass_count + 1;
                $display("PASS: R0 remains zero");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: R0 not hardwired to zero");
            end
        end
    endtask
    
    // Test dual-port read
    task test_dual_port_read;
        begin
            $display("\nTesting dual-port read...");
            
            // Write to R2 and R3
            write_enable = 1;
            write_addr = 5'd2;
            write_data = 48'h222222222222;
            @(posedge clk);
            
            write_addr = 5'd3;
            write_data = 48'h333333333333;
            @(posedge clk);
            write_enable = 0;
            @(posedge clk);
            
            // Read from both ports simultaneously
            read_addr1 = 5'd2;
            read_addr2 = 5'd3;
            #1;
            
            test_count = test_count + 1;
            if (read_data1 == 48'h222222222222 && read_data2 == 48'h333333333333) begin
                pass_count = pass_count + 1;
                $display("PASS: Dual-port read");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Dual-port read");
            end
        end
    endtask
    
    // Test bypass logic
    task test_bypass_logic;
        begin
            $display("\nTesting bypass logic...");
            
            // Write to R4 and read simultaneously
            write_enable = 1;
            write_addr = 5'd4;
            write_data = 48'h444444444444;
            read_addr1 = 5'd4;
            @(posedge clk);
            #1;
            
            test_count = test_count + 1;
            if (read_data1 == 48'h444444444444) begin
                pass_count = pass_count + 1;
                $display("PASS: Bypass logic (read during write)");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Bypass logic - got %h, expected %h", read_data1, 48'h444444444444);
            end
            
            write_enable = 0;
            @(posedge clk);
        end
    endtask
    
    // Test all registers
    task test_all_registers;
        integer i;
        reg [47:0] expected;
        begin
            $display("\nTesting all registers...");
            
            // Write unique pattern to each register
            write_enable = 1;
            for (i = 1; i < 32; i = i + 1) begin
                write_addr = i;
                write_data = {i[7:0], i[7:0], i[7:0], i[7:0], i[7:0], i[7:0]};
                @(posedge clk);
            end
            write_enable = 0;
            @(posedge clk);
            
            // Read and verify each register
            for (i = 1; i < 32; i = i + 1) begin
                read_addr1 = i;
                expected = {i[7:0], i[7:0], i[7:0], i[7:0], i[7:0], i[7:0]};
                #1;
                
                test_count = test_count + 1;
                if (read_data1 == expected) begin
                    pass_count = pass_count + 1;
                end else begin
                    fail_count = fail_count + 1;
                    $display("FAIL: R%0d - got %h, expected %h", i, read_data1, expected);
                end
            end
            
            $display("PASS: All registers test completed");
        end
    endtask

endmodule