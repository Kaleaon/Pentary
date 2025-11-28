#!/usr/bin/env python3
"""
Pentary Interactive CLI Tool
Easy-to-use command-line interface for common pentary operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic
from pentary_simulator import PentaryProcessor

class PentaryCLI:
    """Interactive command-line interface for pentary operations"""
    
    def __init__(self):
        self.converter = PentaryConverter()
        self.arithmetic = PentaryArithmetic()
        self.processor = None
        
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*60)
        print("  PENTARY INTERACTIVE CLI - Balanced Quinary Computing")
        print("="*60)
        print("\nCommands:")
        print("  convert <number>     - Convert decimal to pentary")
        print("  add <a> <b>          - Add two decimal numbers in pentary")
        print("  sub <a> <b>          - Subtract two decimal numbers")
        print("  mul <a> <b>          - Multiply two decimal numbers")
        print("  run <program>        - Run a simple program")
        print("  examples             - Show example operations")
        print("  help                 - Show this help")
        print("  quit/exit            - Exit the CLI")
        print("\n" + "-"*60 + "\n")
    
    def convert(self, args):
        """Convert decimal to pentary"""
        if len(args) < 1:
            print("Usage: convert <decimal_number>")
            return
        
        try:
            num = int(args[0])
            pentary = self.converter.decimal_to_pentary(num)
            back = self.converter.pentary_to_decimal(pentary)
            print(f"\nDecimal: {num}")
            print(f"Pentary: {pentary}")
            print(f"Verify:  {back} (should be {num})")
            print()
        except ValueError:
            print(f"Error: '{args[0]}' is not a valid integer")
    
    def add(self, args):
        """Add two numbers in pentary"""
        if len(args) < 2:
            print("Usage: add <number1> <number2>")
            return
        
        try:
            a, b = int(args[0]), int(args[1])
            pent_a = self.converter.decimal_to_pentary(a)
            pent_b = self.converter.decimal_to_pentary(b)
            result = self.converter.add_pentary(pent_a, pent_b)
            result_dec = self.converter.pentary_to_decimal(result)
            
            print(f"\n{a} + {b} = {result_dec}")
            print(f"  {pent_a} + {pent_b} = {result}")
            print()
        except ValueError as e:
            print(f"Error: Invalid numbers - {e}")
    
    def sub(self, args):
        """Subtract two numbers in pentary"""
        if len(args) < 2:
            print("Usage: sub <number1> <number2>")
            return
        
        try:
            a, b = int(args[0]), int(args[1])
            pent_a = self.converter.decimal_to_pentary(a)
            pent_b = self.converter.decimal_to_pentary(b)
            result = self.converter.subtract_pentary(pent_a, pent_b)
            result_dec = self.converter.pentary_to_decimal(result)
            
            print(f"\n{a} - {b} = {result_dec}")
            print(f"  {pent_a} - {pent_b} = {result}")
            print()
        except ValueError as e:
            print(f"Error: Invalid numbers - {e}")
    
    def mul(self, args):
        """Multiply two numbers in pentary"""
        if len(args) < 2:
            print("Usage: mul <number1> <number2>")
            return
        
        try:
            a, b = int(args[0]), int(args[1])
            pent_a = self.converter.decimal_to_pentary(a)
            result = self.converter.multiply_pentary_by_constant(pent_a, b)
            result_dec = self.converter.pentary_to_decimal(result)
            
            print(f"\n{a} × {b} = {result_dec}")
            print(f"  {pent_a} × {b} = {result}")
            print()
        except ValueError as e:
            print(f"Error: Invalid numbers - {e}")
    
    def run(self, args):
        """Run a simple program"""
        if len(args) < 1:
            print("Usage: run <program_name>")
            print("Available programs: fibonacci, factorial, sum")
            return
        
        program_name = args[0].lower()
        
        if program_name == "fibonacci":
            program = [
                "MOVI P1, 0",      # a = 0
                "MOVI P2, 1",      # b = 1
                "MOVI P3, 10",     # count = 10
                "MOVI P4, 0",      # i = 0
                "ADD P5, P1, P2",  # c = a + b
                "MOVI P1, P2",     # a = b
                "MOVI P2, P5",     # b = c
                "ADDI P4, P4, 1",  # i++
                "SUB P6, P4, P3",  # temp = i - count
                "BLT P6, 4",       # if temp < 0, loop
                "HALT"
            ]
        elif program_name == "sum":
            program = [
                "MOVI P1, 0",      # sum = 0
                "MOVI P2, 1",       # i = 1
                "MOVI P3, 10",     # limit = 10
                "ADD P1, P1, P2",  # sum += i
                "ADDI P2, P2, 1",  # i++
                "SUB P4, P2, P3",  # temp = i - limit
                "BLT P4, 3",       # if temp < 0, loop
                "HALT"
            ]
        else:
            print(f"Unknown program: {program_name}")
            return
        
        if self.processor is None:
            self.processor = PentaryProcessor()
        
        print(f"\nRunning program: {program_name}")
        print("-" * 40)
        self.processor.load_program(program)
        self.processor.run(verbose=True)
        print("-" * 40)
        self.processor.print_state()
        print()
    
    def examples(self, args):
        """Show example operations"""
        print("\n" + "="*60)
        print("EXAMPLE OPERATIONS")
        print("="*60 + "\n")
        
        examples = [
            (5, "Convert 5 to pentary"),
            (42, "Convert 42 to pentary"),
            (100, "Convert 100 to pentary"),
        ]
        
        for num, desc in examples:
            pent = self.converter.decimal_to_pentary(num)
            print(f"{desc}:")
            print(f"  {num} = {pent}")
        
        print("\nArithmetic Examples:")
        print("  7 + 9 =", self.converter.pentary_to_decimal(
            self.converter.add_pentary(
                self.converter.decimal_to_pentary(7),
                self.converter.decimal_to_pentary(9)
            )
        ))
        print("  15 - 8 =", self.converter.pentary_to_decimal(
            self.converter.subtract_pentary(
                self.converter.decimal_to_pentary(15),
                self.converter.decimal_to_pentary(8)
            )
        ))
        print("  6 × 2 =", self.converter.pentary_to_decimal(
            self.converter.multiply_pentary_by_constant(
                self.converter.decimal_to_pentary(6), 2
            )
        ))
        print()
    
    def help(self, args):
        """Show help"""
        self.print_banner()
    
    def run_interactive(self):
        """Run interactive CLI loop"""
        self.print_banner()
        
        while True:
            try:
                line = input("pentary> ").strip()
                if not line:
                    continue
                
                parts = line.split()
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in ['quit', 'exit', 'q']:
                    print("\nGoodbye! The future is not Binary. It is Balanced.\n")
                    break
                elif cmd == 'convert':
                    self.convert(args)
                elif cmd == 'add':
                    self.add(args)
                elif cmd == 'sub':
                    self.sub(args)
                elif cmd == 'mul':
                    self.mul(args)
                elif cmd == 'run':
                    self.run(args)
                elif cmd == 'examples':
                    self.examples(args)
                elif cmd == 'help':
                    self.help(args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            except KeyboardInterrupt:
                print("\n\nGoodbye! The future is not Binary. It is Balanced.\n")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main entry point"""
    cli = PentaryCLI()
    cli.run_interactive()

if __name__ == "__main__":
    main()
