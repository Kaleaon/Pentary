#!/usr/bin/env python3
"""
Pentary Assembler
Converts Pentary Assembly (with labels) to executable machine code (simulator instructions)
"""

import sys
import re
from typing import List, Dict, Tuple, Optional

class PentaryAssembler:
    """Assembler for Pentary processor"""

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[str] = []
        self.source_map: List[int] = []  # Maps output line index to original source line number

    def assemble(self, source_lines: List[str]) -> List[str]:
        """
        Assemble source code into executable instructions.

        Args:
            source_lines: List of source code lines

        Returns:
            List of executable instructions (with labels resolved)
        """
        # Reset state
        self.labels = {}
        self.instructions = []
        self.source_map = []

        cleaned_lines = []

        # Pass 1: Clean code and find labels
        current_address = 1  # 1-based address for simulator

        for i, line in enumerate(source_lines):
            # Strip comments and whitespace
            line = line.split('#')[0].strip()

            if not line:
                continue

            # Check for label definition
            if line.endswith(':'):
                label_name = line[:-1].strip()
                if label_name in self.labels:
                    raise ValueError(f"Duplicate label definition: {label_name}")
                self.labels[label_name] = current_address
                continue

            # Record instruction
            cleaned_lines.append((line, i + 1))  # (instruction, original_line_num)
            current_address += 1

        # Pass 2: Resolve labels
        final_instructions = []

        for line, line_num in cleaned_lines:
            try:
                resolved_line = self._resolve_labels(line)
                final_instructions.append(resolved_line)
                self.source_map.append(line_num)
            except ValueError as e:
                raise ValueError(f"Error at line {line_num}: {e}")

        return final_instructions

    def _resolve_labels(self, instruction: str) -> str:
        """Replace labels with addresses in an instruction"""
        parts = instruction.split(maxsplit=1)
        if len(parts) == 1:
            return instruction

        opcode = parts[0]
        args_str = parts[1]

        # Instructions that use labels/targets
        # JUMP target
        # CALL target
        # BEQ Rs, target
        # BNE Rs, target
        # BLT Rs, target
        # BGT Rs, target

        if opcode in ["JUMP", "CALL"]:
            target = args_str.strip()
            if target in self.labels:
                return f"{opcode} {self.labels[target]}"
            elif self._is_int(target):
                return instruction
            else:
                raise ValueError(f"Undefined label: {target}")

        elif opcode in ["BEQ", "BNE", "BLT", "BGT"]:
            # Format: OPCODE Rs, Target
            args = [arg.strip() for arg in args_str.split(',')]
            if len(args) != 2:
                # Maybe no space after comma?
                return instruction

            rs, target = args

            if target in self.labels:
                return f"{opcode} {rs}, {self.labels[target]}"
            elif self._is_int(target):
                return instruction
            else:
                raise ValueError(f"Undefined label: {target}")

        return instruction

    def _is_int(self, s: str) -> bool:
        """Check if string is an integer"""
        try:
            int(s)
            return True
        except ValueError:
            return False

    def assemble_file(self, input_file: str, output_file: str = None) -> List[str]:
        """Assemble a file and optionally write to output file"""
        with open(input_file, 'r') as f:
            lines = f.readlines()

        instructions = self.assemble(lines)

        if output_file:
            with open(output_file, 'w') as f:
                for line in instructions:
                    f.write(line + '\n')

        return instructions

def main():
    if len(sys.argv) < 2:
        print("Usage: python pentary_assembler.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    assembler = PentaryAssembler()
    try:
        instructions = assembler.assemble_file(input_file, output_file)

        if not output_file:
            print("Assembled Code:")
            print("-" * 20)
            for i, line in enumerate(instructions):
                print(f"{i+1:3d}: {line}")
        else:
            print(f"Successfully assembled {input_file} to {output_file}")

    except Exception as e:
        print(f"Assembly failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
