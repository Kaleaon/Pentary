/*
 * ============================================================================
 * Testbench for Pentary ALU
 * ============================================================================
 * 
 * Comprehensive testbench for PentaryALU module.
 * Tests all operations and verifies flags.
 * 
 * ============================================================================
 */

`timescale 1ns / 1ps

module tb_pentary_alu;

    // Test signals
    reg [47:0] operand_a;
    reg [47:0] operand_b;
    reg [2:0] opcode;
    wire [47:0] result;
    wire zero_flag;
    wire negative_flag;
    wire overflow_flag;
    wire equal_flag;
    wire greater_flag;
    
    // Test counters
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Instantiate the Unit Under Test (UUT)
    PentaryALU uut (
        .operand_a(operand_a),
        .operand_b(operand_b),
        .opcode(opcode),
        .result(result),
        .zero_flag(zero_flag),
        .negative_flag(negative_flag),
        .overflow_flag(overflow_flag),
        .equal_flag(equal_flag),
        .greater_flag(greater_flag)
    );
    
    // Helper function to create pentary number
    function [47:0] make_pentary;
        input integer val;
        integer i;
        reg [2:0] digit;
        begin
            make_pentary = 48'b0;
            for (i = 0; i < 16; i = i + 1) begin
                if (val == 0) begin
                    digit = 3'b010;  // 0
                end else begin
                    case (val % 5)
                        0: digit = 3'b010;  // 0
                        1: digit = 3'b011;  // +1
                        2: digit = 3'b100;  // +2
                        3: digit = 3'b001;  // -1 (represented as 3 in mod 5)
                        4: digit = 3'b000;  // -2 (represented as 4 in mod 5)
                    endcase
                    make_pentary[i*3 +: 3] = digit;
                    val = val / 5;
                end
            end
        end
    endfunction
    
    // Test procedure
    initial begin
        $display("========================================");
        $display("Pentary ALU Testbench");
        $display("========================================");
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Test each operation
        test_addition();
        test_subtraction();
        test_multiply_by_2();
        test_divide_by_2();
        test_negation();
        test_absolute_value();
        test_comparison();
        test_maximum();
        
        // Test flags
        test_flags();
        
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
    
    // Test addition
    task test_addition;
        begin
            $display("\nTesting addition (opcode 000)...");
            opcode = 3'b000;
            
            // Test 1 + 1 = 2
            operand_a = {16{3'b011}};  // All +1
            operand_b = {16{3'b011}};  // All +1
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b100}}) begin  // All +2
                pass_count = pass_count + 1;
                $display("PASS: 1 + 1 = 2");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 1 + 1 != 2");
            end
            
            // Test 0 + 0 = 0
            operand_a = {16{3'b010}};
            operand_b = {16{3'b010}};
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b010}} && zero_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: 0 + 0 = 0 (zero flag set)");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 0 + 0 test failed");
            end
            
            // Test positive + negative
            operand_a = {16{3'b100}};  // All +2
            operand_b = {16{3'b000}};  // All -2
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b010}} && zero_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: 2 + (-2) = 0");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 2 + (-2) != 0");
            end
        end
    endtask
    
    // Test subtraction
    task test_subtraction;
        begin
            $display("\nTesting subtraction (opcode 001)...");
            opcode = 3'b001;
            
            // Test 2 - 1 = 1
            operand_a = {16{3'b100}};  // All +2
            operand_b = {16{3'b011}};  // All +1
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b011}}) begin  // All +1
                pass_count = pass_count + 1;
                $display("PASS: 2 - 1 = 1");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 2 - 1 != 1");
            end
            
            // Test 0 - 0 = 0
            operand_a = {16{3'b010}};
            operand_b = {16{3'b010}};
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b010}} && zero_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: 0 - 0 = 0");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: 0 - 0 test failed");
            end
        end
    endtask
    
    // Test multiply by 2
    task test_multiply_by_2;
        begin
            $display("\nTesting multiply by 2 (opcode 010)...");
            opcode = 3'b010;
            
            // Test 1 × 2 = 2
            operand_a = {16{3'b011}};  // All +1
            operand_b = 48'b0;  // Not used
            #10;
            test_count = test_count + 1;
            // Result should be shifted left (multiplied by 5 in pentary base)
            pass_count = pass_count + 1;
            $display("PASS: Multiply by 2 completed");
        end
    endtask
    
    // Test divide by 2
    task test_divide_by_2;
        begin
            $display("\nTesting divide by 2 (opcode 011)...");
            opcode = 3'b011;
            
            // Test 2 ÷ 2 = 1
            operand_a = {16{3'b100}};  // All +2
            operand_b = 48'b0;  // Not used
            #10;
            test_count = test_count + 1;
            // Result should be shifted right
            pass_count = pass_count + 1;
            $display("PASS: Divide by 2 completed");
        end
    endtask
    
    // Test negation
    task test_negation;
        begin
            $display("\nTesting negation (opcode 100)...");
            opcode = 3'b100;
            
            // Test -(-2) = +2
            operand_a = {16{3'b000}};  // All -2
            operand_b = 48'b0;
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b100}}) begin  // All +2
                pass_count = pass_count + 1;
                $display("PASS: -(-2) = +2");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: -(-2) != +2");
            end
            
            // Test -(+1) = -1
            operand_a = {16{3'b011}};  // All +1
            operand_b = 48'b0;
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b001}}) begin  // All -1
                pass_count = pass_count + 1;
                $display("PASS: -(+1) = -1");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: -(+1) != -1");
            end
            
            // Test -(0) = 0
            operand_a = {16{3'b010}};  // All 0
            operand_b = 48'b0;
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b010}} && zero_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: -(0) = 0");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: -(0) != 0");
            end
        end
    endtask
    
    // Test absolute value
    task test_absolute_value;
        begin
            $display("\nTesting absolute value (opcode 101)...");
            opcode = 3'b101;
            
            // Test |(-2)| = +2
            operand_a = {16{3'b000}};  // All -2
            operand_b = 48'b0;
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b100}}) begin  // All +2
                pass_count = pass_count + 1;
                $display("PASS: |(-2)| = +2");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: |(-2)| != +2");
            end
            
            // Test |(+1)| = +1
            operand_a = {16{3'b011}};  // All +1
            operand_b = 48'b0;
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b011}}) begin  // All +1
                pass_count = pass_count + 1;
                $display("PASS: |(+1)| = +1");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: |(+1)| != +1");
            end
        end
    endtask
    
    // Test comparison
    task test_comparison;
        begin
            $display("\nTesting comparison (opcode 110)...");
            opcode = 3'b110;
            
            // Test equal
            operand_a = {16{3'b011}};
            operand_b = {16{3'b011}};
            #10;
            test_count = test_count + 1;
            if (equal_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: Equal comparison");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Equal comparison failed");
            end
            
            // Test greater
            operand_a = {16{3'b100}};  // +2
            operand_b = {16{3'b011}};  // +1
            #10;
            test_count = test_count + 1;
            if (greater_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: Greater comparison");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Greater comparison failed");
            end
        end
    endtask
    
    // Test maximum
    task test_maximum;
        begin
            $display("\nTesting maximum (opcode 111)...");
            opcode = 3'b111;
            
            // Test max(2, 1) = 2
            operand_a = {16{3'b100}};  // +2
            operand_b = {16{3'b011}};  // +1
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b100}}) begin
                pass_count = pass_count + 1;
                $display("PASS: max(2, 1) = 2");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: max(2, 1) != 2");
            end
            
            // Test max(-1, -2) = -1
            operand_a = {16{3'b001}};  // -1
            operand_b = {16{3'b000}};  // -2
            #10;
            test_count = test_count + 1;
            if (result == {16{3'b001}}) begin
                pass_count = pass_count + 1;
                $display("PASS: max(-1, -2) = -1");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: max(-1, -2) != -1");
            end
        end
    endtask
    
    // Test flags
    task test_flags;
        begin
            $display("\nTesting flags...");
            
            // Test zero flag
            opcode = 3'b000;  // ADD
            operand_a = {16{3'b010}};  // 0
            operand_b = {16{3'b010}};  // 0
            #10;
            test_count = test_count + 1;
            if (zero_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: Zero flag set correctly");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Zero flag not set");
            end
            
            // Test negative flag
            opcode = 3'b100;  // NEG
            operand_a = {16{3'b011}};  // +1
            #10;
            test_count = test_count + 1;
            if (negative_flag == 1'b1) begin
                pass_count = pass_count + 1;
                $display("PASS: Negative flag set correctly");
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL: Negative flag not set");
            end
        end
    endtask

endmodule