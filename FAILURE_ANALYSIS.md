# Pentary Architecture Failure Analysis

This document summarizes the critical flaws discovered in the Pentary architecture and its current implementation.

## 1. Fundamental Arithmetic Logic Errors

The core arithmetic lookup table (`ADD_TABLE` in `tools/pentary_arithmetic.py`) contains mathematical errors. This means the hardware design (which relies on this truth table) is fundamentally broken and will produce incorrect results for basic addition.

### Proof
Run: `python3 tools/prove_arithmetic_bug.py`

**Findings:**
- 16 entries in the addition table are incorrect.
- Example: `(-2) + (-2) + (-1)` (inputs)
  - Mathematical Truth: `-5`
  - Table Entry: `(1, -1)` -> `1 + 5*(-1) = -4`
  - Error: The table yields `-4` instead of `-5`.

This proves that the "Balanced Pentary" logic gates as defined are inconsistent.

## 2. Simulator "Cheating" & Infinite Precision

The simulator (`tools/pentary_simulator.py`) does not accurately reflect the proposed hardware constraints.

### Proof
Run: `python3 tests/test_pentary_failures.py`

**Findings:**
- The simulator uses Python's arbitrary-precision strings to store registers.
- It allows values to grow indefinitely (e.g., 20+ pents), whereas the hardware is defined to have 16-pent words.
- Overflow conditions that would crash or corrupt data on the chip are silently handled correctly by the simulator because it ignores the word limit.
- This creates a false sense of correctness; the hardware would fail where the simulator succeeds.

## 3. Missing Multiplication & Performance

The manifesto claims "Multiplication is Obsolete" and replaces it with shift-add. However, the instruction set lacks a general purpose `MUL` instruction, and the shift-add approach is not exposed as a hardware macro.

### Proof
Run: `python3 tests/test_pentary_failures.py`

**Findings:**
- No `MUL` instruction exists for general register-to-register multiplication.
- Performing `A * B` (where both are variables) requires O(N) software emulation loop.
- Compared to binary hardware multipliers (single cycle), this is a massive performance regression.

## 4. Simulator Inefficiency

The string-based simulation is extremely inefficient.

### Proof
Run: `python3 tools/prove_memory_inefficiency.py`

**Findings:**
- Storing a simple integer requires ~21x more memory than a binary representation.
- This makes large-scale simulation (e.g., full LLM inference) practically impossible due to RAM constraints.

## Conclusion

The Pentary architecture, in its current state, "doesn't work" because:
1.  **It does math wrong**: The basic addition logic table has errors.
2.  **The simulator lies**: It succeeds on tasks the hardware would fail (overflow).
3.  **It is incomplete**: Critical operations for AI (multiplication) are missing or prohibitively slow.
