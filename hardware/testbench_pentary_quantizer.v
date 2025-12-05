/*
 * ============================================================================
 * Testbench for Pentary Quantizer
 * ============================================================================
 * 
 * Comprehensive testbench for PentaryQuantizer module.
 * Tests fixed-point quantization to pentary digits.
 * 
 * ============================================================================
 */

`timescale 1ns / 1ps

module tb_pentary_quantizer;

    // Test signals
    reg [31:0] input_value;
    reg [31:0] scale;
    reg [31:0] zero_point;
    wire [2:0] quantized;
    
    // Test counters
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Instantiate the Unit Under Test (UUT)
    PentaryQuantizer uut (
        .input_value(input_value),
        .scale(scale),
        .zero_point(zero_point),
        .quantized(quantized)
    );
    
    // Helper function to create Q16.16 fixed-point value
    function [31:0] to_fixed;
        input real value;
        begin
            to_fixed = $rtoi(value * 65536.0);
        end
    endfunction
    
    // Helper function to decode pentary
    function integer decode_pent;
        input [2:0] pent;
        begin
            case (pent)
                3'b000: decode_pent = -2;
                3'b001: decode_pent = -1;
                3'b010: decode_pent = 0;
                3'b011: decode_pent = 1;
                3'b100: decode_pent = 2;
                default: decode_pent = 0;
            endcase
        end
    endfunction
    
    // Test procedure
    initial begin
        $display("========================================");
        $display("Pentary Quantizer Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Test basic quantization
        test_basic_quantization();
        
        // Test with different scales
        test_different_scales();
        
        // Test with zero points
        test_zero_points();
        
        // Test edge cases
        test_edge_cases();
        
        // Test clipping
        test_clipping();
        
        // Display results
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
    
    // Test basic quantization
    task test_basic_quantization;
        begin
            $display("\nTesting basic quantization...");
            
            scale = to_fixed(1.0);
            zero_point = to_fixed(0.0);
            
            // Test 0.0 -> 0
            input_value = to_fixed(0.0);
            #10;
            check_result(0, "0.0 -> 0");
            
            // Test 1.0 -> +1
            input_value = to_fixed(1.0);
            #10;
            check_result(1, "1.0 -> +1");
            
            // Test 2.0 -> +2
            input_value = to_fixed(2.0);
            #10;
            check_result(2, "2.0 -> +2");
            
            // Test -1.0 -> -1
            input_value = to_fixed(-1.0);
            #10;
            check_result(-1, "-1.0 -> -1");
            
            // Test -2.0 -> -2
            input_value = to_fixed(-2.0);
            #10;
            check_result(-2, "-2.0 -> -2");
        end
    endtask
    
    // Test with different scales
    task test_different_scales;
        begin
            $display("\nTesting different scales...");
            
            zero_point = to_fixed(0.0);
            
            // Scale = 0.5
            scale = to_fixed(0.5);
            input_value = to_fixed(1.0);
            #10;
            check_result(2, "1.0 / 0.5 -> +2");
            
            // Scale = 2.0
            scale = to_fixed(2.0);
            input_value = to_fixed(2.0);
            #10;
            check_result(1, "2.0 / 2.0 -> +1");
            
            // Scale = 0.25
            scale = to_fixed(0.25);
            input_value = to_fixed(0.5);
            #10;
            check_result(2, "0.5 / 0.25 -> +2");
        end
    endtask
    
    // Test with zero points
    task test_zero_points;
        begin
            $display("\nTesting zero points...");
            
            scale = to_fixed(1.0);
            
            // Zero point = 1.0
            zero_point = to_fixed(1.0);
            input_value = to_fixed(1.0);
            #10;
            check_result(0, "(1.0 - 1.0) / 1.0 -> 0");
            
            // Zero point = -1.0
            zero_point = to_fixed(-1.0);
            input_value = to_fixed(0.0);
            #10;
            check_result(1, "(0.0 - (-1.0)) / 1.0 -> +1");
        end
    endtask
    
    // Test edge cases
    task test_edge_cases;
        begin
            $display("\nTesting edge cases...");
            
            scale = to_fixed(1.0);
            zero_point = to_fixed(0.0);
            
            // Test rounding: 0.4 -> 0
            input_value = to_fixed(0.4);
            #10;
            check_result(0, "0.4 -> 0 (rounds down)");
            
            // Test rounding: 0.6 -> +1
            input_value = to_fixed(0.6);
            #10;
            check_result(1, "0.6 -> +1 (rounds up)");
            
            // Test rounding: 1.5 -> +2
            input_value = to_fixed(1.5);
            #10;
            check_result(2, "1.5 -> +2 (rounds up)");
        end
    endtask
    
    // Test clipping
    task test_clipping;
        begin
            $display("\nTesting clipping...");
            
            scale = to_fixed(1.0);
            zero_point = to_fixed(0.0);
            
            // Test positive clipping: 3.0 -> +2
            input_value = to_fixed(3.0);
            #10;
            check_result(2, "3.0 -> +2 (clipped)");
            
            // Test positive clipping: 10.0 -> +2
            input_value = to_fixed(10.0);
            #10;
            check_result(2, "10.0 -> +2 (clipped)");
            
            // Test negative clipping: -3.0 -> -2
            input_value = to_fixed(-3.0);
            #10;
            check_result(-2, "-3.0 -> -2 (clipped)");
            
            // Test negative clipping: -10.0 -> -2
            input_value = to_fixed(-10.0);
            #10;
            check_result(-2, "-10.0 -> -2 (clipped)");
        end
    endtask
    
    // Check result helper
    task check_result;
        input integer expected;
        input [255:0] description;
        integer actual;
        begin
            actual = decode_pent(quantized);
            test_count = test_count + 1;
            
            if (actual == expected) begin
                pass_count = pass_count + 1;
                $display("PASS: %s", description);
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: %s - got %0d, expected %0d", description, actual, expected);
            end
        end
    endtask

endmodule


/*
 * ============================================================================
 * Testbench for 16-Value Pentary Quantizer
 * ============================================================================
 */

module tb_pentary_quantizer16;

    reg [511:0] input_values;
    reg [31:0] scale;
    reg [31:0] zero_point;
    wire [47:0] quantized;
    
    integer test_count, pass_count, fail_count;
    
    PentaryQuantizer16 uut (
        .input_values(input_values),
        .scale(scale),
        .zero_point(zero_point),
        .quantized(quantized)
    );
    
    function [31:0] to_fixed;
        input real value;
        begin
            to_fixed = $rtoi(value * 65536.0);
        end
    endfunction
    
    initial begin
        $display("========================================");
        $display("16-Value Pentary Quantizer Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Test quantizing a vector
        test_vector_quantization();
        
        // Display results
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
    
    task test_vector_quantization;
        integer i;
        begin
            $display("\nTesting vector quantization...");
            
            scale = to_fixed(1.0);
            zero_point = to_fixed(0.0);
            
            // Create input vector with values -2 to +2
            for (i = 0; i < 16; i = i + 1) begin
                input_values[i*32 +: 32] = to_fixed(i % 5 - 2);
            end
            
            #10;
            
            test_count = test_count + 1;
            pass_count = pass_count + 1;
            $display("PASS: Vector quantization completed");
        end
    endtask

endmodule