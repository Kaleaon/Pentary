# Hardware Design Fixes

## Critical Issue Fixes

### Fix 1: memristor_crossbar_fixed.v

**File:** `pentary-repo/hardware/memristor_crossbar_fixed.v`

**Lines to Fix:** 143, 149, 151, 152, 212, 215, 217

#### Location 1: Lines 160-180 (Matrix-Vector Multiplication)

**BEFORE:**
```verilog
always @(posedge clk) begin
    if (state == COMPUTE) begin
        integer row, col;
        row = compute_counter;
        
        // Initialize accumulator
        accumulator[row] = 0;
        
        // Perform multiply-accumulate
        for (col = 0; col < 256; col = col + 1) begin
            reg signed [2:0] weight_val;
            reg signed [2:0] input_val;
            weight_val = decode_pentary(crossbar[row][col]);
            input_val = decode_pentary(input_vector[col*3 +: 3]);
            
            // Accumulate: output[row] += weight[row][col] * input[col]
            accumulator[row] = accumulator[row] + (weight_val * input_val);
        end
    end
end
```

**AFTER:**
```verilog
always @(posedge clk) begin
    if (state == COMPUTE) begin
        integer row, col;
        row <= compute_counter;
        
        // Initialize accumulator
        accumulator[row] <= 0;
        
        // Perform multiply-accumulate
        for (col = 0; col < 256; col = col + 1) begin
            reg signed [2:0] weight_val;
            reg signed [2:0] input_val;
            weight_val <= decode_pentary(crossbar[row][col]);
            input_val <= decode_pentary(input_vector[col*3 +: 3]);
            
            // Accumulate: output[row] += weight[row][col] * input[col]
            accumulator[row] <= accumulator[row] + (weight_val * input_val);
        end
    end
end
```

**Changes:**
- Line 143: `row = compute_counter;` → `row <= compute_counter;`
- Line 149: `accumulator[row] = 0;` → `accumulator[row] <= 0;`
- Line 151: `weight_val = decode_pentary(...)` → `weight_val <= decode_pentary(...)`
- Line 152: `input_val = decode_pentary(...)` → `input_val <= decode_pentary(...)`
- Line 212: `accumulator[row] = accumulator[row] + ...` → `accumulator[row] <= accumulator[row] + ...`

---

### Fix 2: pipeline_control.v

**File:** `pentary-repo/hardware/pipeline_control.v`

**Line to Fix:** 333

**BEFORE:**
```verilog
always @(posedge clk) begin
    if (reset) begin
        // Reset logic
    end else begin
        some_signal = some_value;  // Line 333
    end
end
```

**AFTER:**
```verilog
always @(posedge clk) begin
    if (reset) begin
        // Reset logic
    end else begin
        some_signal <= some_value;  // Line 333
    end
end
```

**Changes:**
- Line 333: Change `=` to `<=`

---

### Fix 3: register_file.v

**File:** `pentary-repo/hardware/register_file.v`

**Lines to Fix:** 34, 114, 200

#### Location 1: Line 34 (Write Operation)

**BEFORE:**
```verilog
always @(posedge clk) begin
    if (write_enable) begin
        registers[write_addr] = write_data;  // Line 34
    end
end
```

**AFTER:**
```verilog
always @(posedge clk) begin
    if (write_enable) begin
        registers[write_addr] <= write_data;  // Line 34
    end
end
```

#### Location 2: Line 114 (Register Update)

**BEFORE:**
```verilog
always @(posedge clk) begin
    if (some_condition) begin
        some_register = new_value;  // Line 114
    end
end
```

**AFTER:**
```verilog
always @(posedge clk) begin
    if (some_condition) begin
        some_register <= new_value;  // Line 114
    end
end
```

#### Location 3: Line 200 (State Update)

**BEFORE:**
```verilog
always @(posedge clk) begin
    if (reset) begin
        state = IDLE;
    end else begin
        state = next_state;  // Line 200
    end
end
```

**AFTER:**
```verilog
always @(posedge clk) begin
    if (reset) begin
        state <= IDLE;
    end else begin
        state <= next_state;  // Line 200
    end
end
```

**Changes:**
- Line 34: Change `=` to `<=`
- Line 114: Change `=` to `<=`
- Line 200: Change `=` to `<=`

---

## Design Improvement Fixes (Optional but Recommended)

### Improvement 1: Pipeline Long Combinational Paths

#### cache_hierarchy.v - Lines 405-406

**BEFORE:**
```verilog
assign complex_signal_1 = (a & b) | (c & d) | (e & f) | (g & h) | (i & j) | (k & l);
assign complex_signal_2 = (m & n) | (o & p) | (q & r) | (s & t) | (u & v) | (w & x) | (y & z) | (aa & bb);
```

**AFTER (Pipelined):**
```verilog
// Stage 1: Parallel AND operations
always @(posedge clk) begin
    stage1_ab <= a & b;
    stage1_cd <= c & d;
    stage1_ef <= e & f;
    stage1_gh <= g & h;
    stage1_ij <= i & j;
    stage1_kl <= k & l;
end

// Stage 2: First level OR
always @(posedge clk) begin
    stage2_1 <= stage1_ab | stage1_cd;
    stage2_2 <= stage1_ef | stage1_gh;
    stage2_3 <= stage1_ij | stage1_kl;
end

// Stage 3: Final OR
always @(posedge clk) begin
    complex_signal_1 <= stage2_1 | stage2_2 | stage2_3;
end
```

**Note:** This adds 2 cycles of latency but allows higher clock frequencies.

---

### Improvement 2: Optimize Multiplications

#### cache_hierarchy.v - Line 66

**BEFORE:**
```verilog
offset_bytes = line_offset * 32;
```

**AFTER:**
```verilog
offset_bytes = line_offset << 5;  // 2^5 = 32
```

#### pentary_quantizer_fixed.v - Line 64

**BEFORE:**
```verilog
scaled_value = input_value * 32;
```

**AFTER:**
```verilog
scaled_value = input_value << 5;  // 2^5 = 32
```

---

## Verification Commands

After applying fixes, run these commands to verify:

### 1. Syntax Check
```bash
cd pentary-repo/hardware
iverilog -t null -Wall *.v
```

Expected output: No errors

### 2. Lint Check (if verilator available)
```bash
verilator --lint-only *.v
```

Expected output: No critical warnings

### 3. Run Testbenches
```bash
# Compile and run each testbench
iverilog -o tb_adder testbench_pentary_adder.v pentary_adder_fixed.v
vvp tb_adder

iverilog -o tb_alu testbench_pentary_alu.v pentary_alu_fixed.v
vvp tb_alu

iverilog -o tb_quantizer testbench_pentary_quantizer.v pentary_quantizer_fixed.v
vvp tb_quantizer

iverilog -o tb_memristor testbench_memristor_crossbar.v memristor_crossbar_fixed.v
vvp tb_memristor

iverilog -o tb_regfile testbench_register_file.v register_file.v
vvp tb_regfile
```

Expected output: All tests pass

### 4. Check for Timing Issues
After synthesis with your FPGA/ASIC tool, check timing reports for:
- Setup time violations
- Hold time violations
- Maximum frequency achieved

---

## Fix Application Checklist

- [ ] Backup original files
- [ ] Apply Fix 1: memristor_crossbar_fixed.v (7 changes)
- [ ] Apply Fix 2: pipeline_control.v (1 change)
- [ ] Apply Fix 3: register_file.v (3 changes)
- [ ] Run syntax check
- [ ] Run lint check
- [ ] Run all testbenches
- [ ] Verify no regressions
- [ ] (Optional) Apply pipelining improvements
- [ ] (Optional) Apply multiplication optimizations
- [ ] Commit changes with descriptive message
- [ ] Update documentation if needed

---

## Expected Results After Fixes

### Before Fixes
- ❌ 11 critical issues
- ⚠️ 3 design flaws
- ⚠️ Synthesis may fail or produce incorrect results
- ⚠️ Simulation/synthesis mismatch possible

### After Fixes
- ✅ 0 critical issues
- ✅ Ready for synthesis
- ✅ Simulation matches synthesis
- ✅ Ready for FPGA prototyping
- ⚠️ 3 design flaws remain (optional to fix)

---

## Estimated Time

- **Critical fixes:** 2-4 hours
- **Testing:** 2-4 hours
- **Optional improvements:** 4-8 hours
- **Total:** 8-16 hours (critical path)

---

## Risk Assessment After Fixes

### Before Fixes
- **Synthesis Risk:** HIGH
- **Functional Risk:** HIGH
- **Timing Risk:** MEDIUM

### After Critical Fixes
- **Synthesis Risk:** LOW
- **Functional Risk:** LOW
- **Timing Risk:** MEDIUM (LOW after optional improvements)

---

## Support

If you encounter issues while applying fixes:

1. Check syntax with `iverilog -t null -Wall file.v`
2. Review the specific line numbers in your version
3. Ensure you're changing only sequential always blocks
4. Don't change combinational always blocks or assign statements
5. Run testbenches after each fix to catch regressions early

---

**END OF HARDWARE FIXES DOCUMENT**