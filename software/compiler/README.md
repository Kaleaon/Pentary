# Pentary Compiler (PCC) - LLVM Backend

## Overview

The Pentary Compiler (PCC) is an LLVM-based compiler that translates high-level C/C++ code into the Pentary Instruction Set Architecture (ISA). This backend enables developers to write performant code for the Pentary accelerator without needing to write assembly by hand.

## Architecture

The compiler consists of three main components:

1. **Frontend (Clang)**: Standard Clang frontend for C/C++ parsing and semantic analysis.
2. **Middle-end (LLVM IR)**: LLVM's optimization passes operate on the intermediate representation.
3. **Backend (Pentary)**: Custom LLVM backend that generates Pentary assembly and machine code.

## Pentary ISA Summary

- **Architecture**: Load-Store RISC-like architecture
- **Word Size**: 48 bits (16 pentary digits)
- **Registers**: 32 general-purpose registers (R0-R31), R0 is hardwired to 0
- **Addressing**: Pentary-based addressing with 48-bit pointers

### Instruction Categories

1. **Arithmetic**: ADD, SUB, MUL, DIV, MOD
2. **Logical**: AND, OR, XOR, NOT, SHL, SHR
3. **Memory**: LOAD, STORE (with various addressing modes)
4. **Control Flow**: BRANCH, JUMP, CALL, RETURN
5. **Special**: TENSOR (dispatch to PTC), SYNC (synchronization)

## Implementation Status

- [x] ISA Specification
- [ ] LLVM TableGen definitions
- [ ] Instruction selection patterns
- [ ] Register allocation
- [ ] Assembly printer
- [ ] Object file generation

## Building

```bash
cd software/compiler
mkdir build && cd build
cmake -G Ninja -DLLVM_TARGETS_TO_BUILD="Pentary" ../llvm
ninja
```

## Usage

```bash
# Compile C code to Pentary assembly
pcc -S -o output.s input.c

# Compile to object file
pcc -c -o output.o input.c

# Link and generate executable
pcc -o program input1.o input2.o
```
