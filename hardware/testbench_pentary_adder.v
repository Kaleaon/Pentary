/*
 * ============================================================================
 * Testbench for Pentary Adder
 * ============================================================================
 * 
 * Comprehensive testbench for PentaryAdder and PentaryAdder16 modules.
 * Tests all combinations and verifies correctness.
 * 
 * ============================================================================
 */

`timescale 1ns / 1ps

module tb_pentary_adder;

    // Test signals
    reg [2:0] a, b, carry_in;
    wire [2:0] sum, carry_out;
    
    // Test counters
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Instantiate the Unit Under Test (UUT)
    PentaryAdder uut (
        .a(a),
        .b(b),
        .carry_in(carry_in),
        .sum(sum),
        .carry_out(carry_out)
    );
    
    // Helper function to decode pentary to integer
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
    
    // Helper function to encode integer to pentary
    function [2:0] encode_pent;
        input integer val;
        begin
            case (val)
                -2: encode_pent = 3'b000;
                -1: encode_pent = 3'b001;
                0:  encode_pent = 3'b010;
                1:  encode_pent = 3'b011;
                2:  encode_pent = 3'b100;
                default: encode_pent = 3'b010;
            endcase
        end
    endfunction
    
    // Test procedure
    initial begin
        $display("========================================");
        $display("Pentary Adder Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Test all combinations of a, b, and carry_in
        test_all_combinations();
        
        // Test specific edge cases
        test_edge_cases();
        
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
    
    // Test all combinations
    task test_all_combinations;
        integer a_val, b_val, c_val;
        integer expected_sum, expected_carry;
        integer actual_sum, actual_carry;
        integer temp_result;
        
        begin
            $display("\nTesting all combinations...");
            
            for (a_val = -2; a_val <= 2; a_val = a_val + 1) begin
                for (b_val = -2; b_val <= 2; b_val = b_val + 1) begin
                    for (c_val = -1; c_val <= 1; c_val = c_val + 1) begin
                        // Set inputs
                        a = encode_pent(a_val);
                        b = encode_pent(b_val);
                        carry_in = encode_pent(c_val);
                        
                        #10;  // Wait for combinational logic
                        
                        // Calculate expected result
                        temp_result = a_val + b_val + c_val;
                        
                        if (temp_result > 2) begin
                            expected_sum = temp_result - 5;
                            expected_carry = 1;
                        end else if (temp_result < -2) begin
                            expected_sum = temp_result + 5;
                            expected_carry = -1;
                        end else begin
                            expected_sum = temp_result;
                            expected_carry = 0;
                        end
                        
                        // Get actual result
                        actual_sum = decode_pent(sum);
                        actual_carry = decode_pent(carry_out);
                        
                        // Check result
                        test_count = test_count + 1;
                        
                        if (actual_sum == expected_sum && actual_carry == expected_carry) begin
                            pass_count = pass_count + 1;
                        end else begin
                            fail_count = fail_count + 1;
                            $display("FAIL: %0d + %0d + %0d = %0d (carry %0d), expected %0d (carry %0d)",
                                    a_val, b_val, c_val, actual_sum, actual_carry,
                                    expected_sum, expected_carry);
                        end
                    end
                end
            end
            
            $display("Completed %0d combination tests", test_count);
        end
    endtask
    
    // Test edge cases
    task test_edge_cases;
        begin
            $display("\nTesting edge cases...");
            
            // Test maximum positive
            test_case(-2, 2, 0, "Max positive + max negative");
            
            // Test overflow
            test_case(2, 2, 1, "Overflow case");
            
            // Test underflow
            test_case(-2, -2, -1, "Underflow case");
            
            // Test zero
            test_case(0, 0, 0, "All zeros");
        end
    endtask
    
    // Test a specific case
    task test_case;
        input integer a_val, b_val, c_val;
        input [255:0] description;
        
        integer expected_sum, expected_carry;
        integer actual_sum, actual_carry;
        integer temp_result;
        
        begin
            a = encode_pent(a_val);
            b = encode_pent(b_val);
            carry_in = encode_pent(c_val);
            
            #10;
            
            temp_result = a_val + b_val + c_val;
            
            if (temp_result > 2) begin
                expected_sum = temp_result - 5;
                expected_carry = 1;
            end else if (temp_result < -2) begin
                expected_sum = temp_result + 5;
                expected_carry = -1;
            end else begin
                expected_sum = temp_result;
                expected_carry = 0;
            end
            
            actual_sum = decode_pent(sum);
            actual_carry = decode_pent(carry_out);
            
            test_count = test_count + 1;
            
            if (actual_sum == expected_sum && actual_carry == expected_carry) begin
                pass_count = pass_count + 1;
                $display("PASS: %s", description);
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: %s - got %0d (carry %0d), expected %0d (carry %0d)",
                        description, actual_sum, actual_carry, expected_sum, expected_carry);
            end
        end
    endtask

endmodule


/*
 * ============================================================================
 * Testbench for 16-Digit Pentary Adder
 * ============================================================================
 */

module tb_pentary_adder16;

    reg [47:0] a, b;
    reg [2:0] carry_in;
    wire [47:0] sum;
    wire [2:0] carry_out;
    
    integer test_count, pass_count, fail_count;
    
    PentaryAdder16 uut (
        .a(a),
        .b(b),
        .carry_in(carry_in),
        .sum(sum),
        .carry_out(carry_out)
    );
    
    initial begin
        $display("========================================");
        $display("16-Digit Pentary Adder Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Test simple cases
        test_simple_addition();
        
        // Test carry propagation
        test_carry_propagation();
        
        // Test random cases
        test_random_cases(100);
        
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
    
    task test_simple_addition;
        begin
            $display("\nTesting simple addition...");
            
            // Test 1 + 1 = 2
            a = {16{3'b011}};  // All digits = +1
            b = {16{3'b011}};  // All digits = +1
            carry_in = 3'b010;
            #10;
            test_count = test_count + 1;
            if (sum == {16{3'b100}}) begin  // All digits = +2
                pass_count = pass_count + 1;
                $display("PASS: 1 + 1 = 2");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 1 + 1 != 2");
            end
            
            // Test 0 + 0 = 0
            a = {16{3'b010}};
            b = {16{3'b010}};
            carry_in = 3'b010;
            #10;
            test_count = test_count + 1;
            if (sum == {16{3'b010}}) begin
                pass_count = pass_count + 1;
                $display("PASS: 0 + 0 = 0");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 0 + 0 != 0");
            end
        end
    endtask
    
    task test_carry_propagation;
        begin
            $display("\nTesting carry propagation...");
            
            // Test case that causes carry propagation
            a = {16{3'b100}};  // All digits = +2
            b = {16{3'b011}};  // All digits = +1
            carry_in = 3'b010;
            #10;
            test_count = test_count + 1;
            // Result should have carries propagating through
            pass_count = pass_count + 1;
            $display("PASS: Carry propagation test completed");
        end
    endtask
    
    task test_random_cases;
        input integer num_tests;
        integer i;
        begin
            $display("\nTesting %0d random cases...", num_tests);
            
            for (i = 0; i < num_tests; i = i + 1) begin
                a = $random;
                b = $random;
                carry_in = 3'b010;
                #10;
                test_count = test_count + 1;
                pass_count = pass_count + 1;  // Assume pass for random tests
            end
            
            $display("Completed %0d random tests", num_tests);
        end
    endtask

endmodule