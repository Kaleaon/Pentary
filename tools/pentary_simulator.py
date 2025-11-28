#!/usr/bin/env python3
"""
Pentary Processor Simulator
Simulates the execution of pentary assembly code
"""

from typing import Dict, List, Tuple, Optional
import os
import sys

# Add directory to path to allow importing assembler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic
from pentary_assembler import PentaryAssembler


class PentaryProcessor:
    """Simulates a pentary processor with registers, memory, and ALU"""
    
    def __init__(self, memory_size: int = 1024):
        """
        Initialize the pentary processor.
        
        Args:
            memory_size: Size of memory in words (16 pents each)
        """
        # Registers (32 general purpose, each 16 pents)
        self.registers = {f"P{i}": "0" for i in range(32)}
        self.registers["P0"] = "0"  # Always zero
        
        # Special registers
        self.pc = 0  # Program counter
        self.sr = {  # Status register
            'Z': False,  # Zero
            'N': False,  # Negative
            'P': False,  # Positive
            'V': False,  # Overflow
            'C': False,  # Carry
            'S': False   # Saturation
        }
        
        # Memory (address -> 16-pent word)
        self.memory = ["0"] * memory_size
        self.memory_size = memory_size
        
        # Stack
        self.stack = []
        
        # Execution state
        self.running = False
        self.cycle_count = 0
        self.instruction_count = 0
        
        # Instruction memory (separate from data memory)
        self.instructions = []
        
    def load_program(self, instructions: List[str]):
        """
        Load a program into instruction memory.
        If the program contains labels, it will be automatically assembled.
        """
        # Check if program needs assembly (has labels)
        needs_assembly = False
        for line in instructions:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(':'):
                needs_assembly = True
                break
            # Check if any instruction uses a label (this is harder to detect robustly without full parsing)
            # But existence of a label definition is a strong indicator.

        if needs_assembly:
            print("Detected labels in program. Assembling...")
            assembler = PentaryAssembler()
            try:
                self.instructions = assembler.assemble(instructions)
            except Exception as e:
                print(f"Assembly failed: {e}")
                print("Loading raw instructions (might fail execution)...")
                self.instructions = instructions
        else:
            self.instructions = instructions

        self.pc = 0
        
    def reset(self):
        """Reset processor state"""
        self.registers = {f"P{i}": "0" for i in range(32)}
        self.pc = 0
        self.sr = {k: False for k in self.sr}
        self.stack = []
        self.cycle_count = 0
        self.instruction_count = 0
        self.running = False
        
    def set_register(self, reg: str, value: str):
        """Set a register value"""
        if reg == "P0":
            return  # P0 is always zero
        if reg in self.registers:
            self.registers[reg] = value
            
    def get_register(self, reg: str) -> str:
        """Get a register value"""
        return self.registers.get(reg, "0")
    
    def update_flags(self, result: str):
        """Update status flags based on result"""
        result_dec = PentaryConverter.pentary_to_decimal(result)
        
        self.sr['Z'] = (result_dec == 0)
        self.sr['N'] = (result_dec < 0)
        self.sr['P'] = (result_dec > 0)
        
    def execute_instruction(self, instruction: str) -> bool:
        """
        Execute a single instruction.
        
        Returns:
            True if execution should continue, False if halted
        """
        parts = instruction.strip().split()
        if not parts or parts[0].startswith('#'):
            return True  # Comment or empty line
        
        opcode = parts[0].upper()
        
        try:
            if opcode == "NOP":
                pass
            
            elif opcode == "HALT":
                return False
            
            elif opcode == "ADD":
                # ADD Rd, Rs1, Rs2
                rd, rs1, rs2 = parts[1].rstrip(','), parts[2].rstrip(','), parts[3]
                a = self.get_register(rs1)
                b = self.get_register(rs2)
                result = PentaryConverter.add_pentary(a, b)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "SUB":
                # SUB Rd, Rs1, Rs2
                rd, rs1, rs2 = parts[1].rstrip(','), parts[2].rstrip(','), parts[3]
                a = self.get_register(rs1)
                b = self.get_register(rs2)
                result = PentaryConverter.subtract_pentary(a, b)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "ADDI":
                # ADDI Rd, Rs1, Imm
                rd, rs1, imm = parts[1].rstrip(','), parts[2].rstrip(','), parts[3]
                a = self.get_register(rs1)
                # Convert immediate to pentary if it's decimal
                if imm.startswith(('⊖', '-', '0', '+', '⊕')):
                    b = imm
                else:
                    b = PentaryConverter.decimal_to_pentary(int(imm))
                result = PentaryConverter.add_pentary(a, b)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "NEG":
                # NEG Rd, Rs1
                rd, rs1 = parts[1].rstrip(','), parts[2]
                a = self.get_register(rs1)
                result = PentaryConverter.negate_pentary(a)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "MUL2":
                # MUL2 Rd, Rs1 (multiply by 2)
                rd, rs1 = parts[1].rstrip(','), parts[2]
                a = self.get_register(rs1)
                result = PentaryConverter.multiply_pentary_by_constant(a, 2)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "SHL":
                # SHL Rd, Rs1, n (shift left by n positions)
                rd, rs1, n = parts[1].rstrip(','), parts[2].rstrip(','), int(parts[3])
                a = self.get_register(rs1)
                result = PentaryConverter.shift_left_pentary(a, n)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "SHR":
                # SHR Rd, Rs1, n (shift right by n positions)
                rd, rs1, n = parts[1].rstrip(','), parts[2].rstrip(','), int(parts[3])
                a = self.get_register(rs1)
                result = PentaryConverter.shift_right_pentary(a, n)
                self.set_register(rd, result)
                self.update_flags(result)
            
            elif opcode == "LOAD":
                # LOAD Rd, [Rs1 + offset]
                rd = parts[1].rstrip(',')
                # Parse [Rs1 + offset] or [Rs1]
                addr_str = ' '.join(parts[2:])
                addr_str = addr_str.strip('[]')
                if '+' in addr_str:
                    base_reg, offset = addr_str.split('+')
                    base_reg = base_reg.strip()
                    offset = int(offset.strip())
                else:
                    base_reg = addr_str.strip()
                    offset = 0
                
                base_addr = PentaryConverter.pentary_to_decimal(self.get_register(base_reg))
                addr = base_addr + offset
                
                if 0 <= addr < self.memory_size:
                    value = self.memory[addr]
                    self.set_register(rd, value)
                else:
                    raise ValueError(f"Memory address out of bounds: {addr}")
            
            elif opcode == "STORE":
                # STORE Rs, [Rd + offset]
                rs = parts[1].rstrip(',')
                addr_str = ' '.join(parts[2:])
                addr_str = addr_str.strip('[]')
                if '+' in addr_str:
                    base_reg, offset = addr_str.split('+')
                    base_reg = base_reg.strip()
                    offset = int(offset.strip())
                else:
                    base_reg = addr_str.strip()
                    offset = 0
                
                base_addr = PentaryConverter.pentary_to_decimal(self.get_register(base_reg))
                addr = base_addr + offset
                
                if 0 <= addr < self.memory_size:
                    value = self.get_register(rs)
                    self.memory[addr] = value
                else:
                    raise ValueError(f"Memory address out of bounds: {addr}")
            
            elif opcode == "PUSH":
                # PUSH Rs
                rs = parts[1]
                value = self.get_register(rs)
                self.stack.append(value)
            
            elif opcode == "POP":
                # POP Rd
                rd = parts[1]
                if self.stack:
                    value = self.stack.pop()
                    self.set_register(rd, value)
                else:
                    raise ValueError("Stack underflow")
            
            elif opcode == "BEQ":
                # BEQ Rs, target (branch if Rs == 0)
                rs, target = parts[1].rstrip(','), int(parts[2])
                value = self.get_register(rs)
                if PentaryConverter.pentary_to_decimal(value) == 0:
                    self.pc = target - 1  # -1 because PC will be incremented
            
            elif opcode == "BNE":
                # BNE Rs, target (branch if Rs != 0)
                rs, target = parts[1].rstrip(','), int(parts[2])
                value = self.get_register(rs)
                if PentaryConverter.pentary_to_decimal(value) != 0:
                    self.pc = target - 1
            
            elif opcode == "BLT":
                # BLT Rs, target (branch if Rs < 0)
                rs, target = parts[1].rstrip(','), int(parts[2])
                value = self.get_register(rs)
                if PentaryConverter.pentary_to_decimal(value) < 0:
                    self.pc = target - 1
            
            elif opcode == "BGT":
                # BGT Rs, target (branch if Rs > 0)
                rs, target = parts[1].rstrip(','), int(parts[2])
                value = self.get_register(rs)
                if PentaryConverter.pentary_to_decimal(value) > 0:
                    self.pc = target - 1
            
            elif opcode == "JUMP":
                # JUMP target
                target = int(parts[1])
                self.pc = target - 1
            
            elif opcode == "CALL":
                # CALL target
                target = int(parts[1])
                # Push return address
                self.stack.append(PentaryConverter.decimal_to_pentary(self.pc + 1))
                self.pc = target - 1
            
            elif opcode == "RET":
                # RET
                if self.stack:
                    return_addr = self.stack.pop()
                    self.pc = PentaryConverter.pentary_to_decimal(return_addr) - 1
                else:
                    return False  # No return address, halt
            
            elif opcode == "MOVI":
                # MOVI Rd, Imm (pseudo-instruction: move immediate)
                rd, imm = parts[1].rstrip(','), parts[2]
                if imm.startswith(('⊖', '-', '0', '+', '⊕')):
                    value = imm
                else:
                    value = PentaryConverter.decimal_to_pentary(int(imm))
                self.set_register(rd, value)
                self.update_flags(value)
            
            else:
                print(f"Unknown instruction: {opcode}")
        
        except Exception as e:
            print(f"Error executing instruction '{instruction}': {e}")
            return False
        
        self.instruction_count += 1
        return True
    
    def run(self, max_cycles: int = 10000, verbose: bool = False):
        """
        Run the loaded program.
        
        Args:
            max_cycles: Maximum number of cycles to execute
            verbose: Print execution trace
        """
        self.running = True
        self.cycle_count = 0
        
        while self.running and self.cycle_count < max_cycles:
            if self.pc >= len(self.instructions):
                break
            
            instruction = self.instructions[self.pc]
            
            if verbose:
                print(f"[{self.cycle_count:4d}] PC={self.pc:3d}: {instruction}")
            
            # Execute instruction
            continue_execution = self.execute_instruction(instruction)
            
            if not continue_execution:
                self.running = False
                break
            
            # Increment PC
            self.pc += 1
            self.cycle_count += 1
        
        if verbose:
            print(f"\nExecution completed in {self.cycle_count} cycles")
            print(f"Instructions executed: {self.instruction_count}")
    
    def print_state(self):
        """Print current processor state"""
        print("\n" + "=" * 70)
        print("Processor State")
        print("=" * 70)
        
        print("\nRegisters:")
        for i in range(0, 32, 4):
            regs = [f"P{j}" for j in range(i, min(i+4, 32))]
            values = [self.get_register(r) for r in regs]
            decimals = [PentaryConverter.pentary_to_decimal(v) for v in values]
            
            print(f"  {regs[0]:<4} = {values[0]:<10} ({decimals[0]:>6})", end="")
            if len(regs) > 1:
                print(f"  {regs[1]:<4} = {values[1]:<10} ({decimals[1]:>6})", end="")
            if len(regs) > 2:
                print(f"  {regs[2]:<4} = {values[2]:<10} ({decimals[2]:>6})", end="")
            if len(regs) > 3:
                print(f"  {regs[3]:<4} = {values[3]:<10} ({decimals[3]:>6})")
            else:
                print()
        
        print(f"\nProgram Counter: {self.pc}")
        print(f"Cycle Count: {self.cycle_count}")
        print(f"Instruction Count: {self.instruction_count}")
        
        print("\nStatus Flags:")
        print(f"  Z={self.sr['Z']}  N={self.sr['N']}  P={self.sr['P']}", end="")
        print(f"  V={self.sr['V']}  C={self.sr['C']}  S={self.sr['S']}")
        
        if self.stack:
            print(f"\nStack (top {min(5, len(self.stack))} items):")
            for i, value in enumerate(self.stack[-5:]):
                decimal = PentaryConverter.pentary_to_decimal(value)
                print(f"  [{i}] {value} ({decimal})")
        
        print("=" * 70)


def main():
    """Demo and testing of pentary processor simulator"""
    print("=" * 70)
    print("Pentary Processor Simulator")
    print("=" * 70)
    print()
    
    # Example program 1: Simple arithmetic
    print("Example 1: Simple Arithmetic")
    print("-" * 70)
    
    program1 = [
        "# Simple arithmetic example",
        "MOVI P1, 5",      # P1 = 5
        "MOVI P2, 3",      # P2 = 3
        "ADD P3, P1, P2",  # P3 = P1 + P2 = 8
        "SUB P4, P1, P2",  # P4 = P1 - P2 = 2
        "MUL2 P5, P3",     # P5 = P3 * 2 = 16
        "HALT"
    ]
    
    proc = PentaryProcessor()
    proc.load_program(program1)
    proc.run(verbose=True)
    proc.print_state()
    
    print("\n" + "=" * 70)
    print("Example 2: Loop and Accumulation")
    print("-" * 70)
    
    # Example program 2: Sum from 1 to 10
    program2 = [
        "# Sum from 1 to 10",
        "MOVI P1, 0",       # sum = 0
        "MOVI P2, 1",       # i = 1
        "MOVI P3, 10",      # limit = 10
        "# Loop start (line 3)",
        "ADD P1, P1, P2",   # sum += i
        "ADDI P2, P2, 1",   # i++
        "SUB P4, P2, P3",   # temp = i - limit
        "BLT P4, 3",        # if temp < 0, goto line 3
        "HALT"
    ]
    
    proc2 = PentaryProcessor()
    proc2.load_program(program2)
    proc2.run(verbose=True)
    proc2.print_state()
    
    print("\n" + "=" * 70)
    print("Example 3: Memory Operations")
    print("-" * 70)
    
    # Example program 3: Array operations
    program3 = [
        "# Store and load from memory",
        "MOVI P1, 10",          # base address
        "MOVI P2, 42",          # value to store
        "STORE P2, [P1 + 0]",   # mem[10] = 42
        "MOVI P3, 99",          # another value
        "STORE P3, [P1 + 1]",   # mem[11] = 99
        "LOAD P4, [P1 + 0]",    # P4 = mem[10]
        "LOAD P5, [P1 + 1]",    # P5 = mem[11]
        "ADD P6, P4, P5",       # P6 = P4 + P5
        "HALT"
    ]
    
    proc3 = PentaryProcessor()
    proc3.load_program(program3)
    proc3.run(verbose=True)
    proc3.print_state()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()