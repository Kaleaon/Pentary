#!/usr/bin/env python3
"""
Pentary Example Generator
Generates example programs and code snippets for common use cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pentary_converter import PentaryConverter
from pentary_simulator import PentaryProcessor

class ExampleGenerator:
    """Generate example code and programs"""
    
    def __init__(self):
        self.converter = PentaryConverter()
    
    def generate_conversion_examples(self):
        """Generate number conversion examples"""
        examples = [
            (0, "Zero"),
            (1, "One"),
            (5, "Five (base)"),
            (10, "Ten"),
            (25, "Twenty-five (5²)"),
            (42, "Forty-two (answer to everything)"),
            (100, "One hundred"),
            (125, "One hundred twenty-five (5³)"),
        ]
        
        print("="*60)
        print("NUMBER CONVERSION EXAMPLES")
        print("="*60 + "\n")
        
        for num, desc in examples:
            pent = self.converter.decimal_to_pentary(num)
            back = self.converter.pentary_to_decimal(pent)
            print(f"{desc:30} {num:4} → {pent:10} → {back:4}")
        
        print()
    
    def generate_arithmetic_examples(self):
        """Generate arithmetic operation examples"""
        operations = [
            (7, 9, "add"),
            (15, 8, "sub"),
            (6, 2, "mul"),
            (10, 3, "sub"),
            (12, 5, "add"),
        ]
        
        print("="*60)
        print("ARITHMETIC OPERATION EXAMPLES")
        print("="*60 + "\n")
        
        for a, b, op in operations:
            pent_a = self.converter.decimal_to_pentary(a)
            pent_b = self.converter.decimal_to_pentary(b)
            
            if op == "add":
                result = self.converter.add_pentary(pent_a, pent_b)
                result_dec = self.converter.pentary_to_decimal(result)
                print(f"{a:3} + {b:3} = {result_dec:3}  ({pent_a:8} + {pent_b:8} = {result:8})")
            elif op == "sub":
                result = self.converter.subtract_pentary(pent_a, pent_b)
                result_dec = self.converter.pentary_to_decimal(result)
                print(f"{a:3} - {b:3} = {result_dec:3}  ({pent_a:8} - {pent_b:8} = {result:8})")
            elif op == "mul":
                result = self.converter.multiply_pentary_by_constant(pent_a, b)
                result_dec = self.converter.pentary_to_decimal(result)
                print(f"{a:3} × {b:3} = {result_dec:3}  ({pent_a:8} × {b:3} = {result:8})")
        
        print()
    
    def generate_program_examples(self):
        """Generate example programs"""
        print("="*60)
        print("EXAMPLE PROGRAMS")
        print("="*60 + "\n")
        
        # Example 1: Simple addition
        print("Example 1: Simple Addition")
        print("-" * 40)
        program1 = [
            "MOVI P1, 5",      # P1 = 5
            "MOVI P2, 3",      # P2 = 3
            "ADD P3, P1, P2",  # P3 = P1 + P2 = 8
            "HALT"
        ]
        for i, line in enumerate(program1, 1):
            print(f"{i:2}. {line}")
        print()
        
        # Example 2: Sum 1 to 10
        print("Example 2: Sum 1 to 10")
        print("-" * 40)
        program2 = [
            "MOVI P1, 0",      # sum = 0
            "MOVI P2, 1",      # i = 1
            "MOVI P3, 10",     # limit = 10
            "loop:",
            "ADD P1, P1, P2",  # sum += i
            "ADDI P2, P2, 1",  # i++
            "SUB P4, P2, P3",  # temp = i - limit
            "BLT P4, loop",    # if temp < 0, loop
            "HALT"             # Result in P1 = 55
        ]
        for i, line in enumerate(program2, 1):
            print(f"{i:2}. {line}")
        print()
        
        # Example 3: Multiply by 2
        print("Example 3: Multiply by 2")
        print("-" * 40)
        program3 = [
            "MOVI P1, 7",      # P1 = 7
            "MUL2 P2, P1",     # P2 = P1 × 2 = 14
            "HALT"
        ]
        for i, line in enumerate(program3, 1):
            print(f"{i:2}. {line}")
        print()
        
        # Example 4: Conditional
        print("Example 4: Conditional (if a > b)")
        print("-" * 40)
        program4 = [
            "MOVI P1, 10",     # a = 10
            "MOVI P2, 7",      # b = 7
            "SUB P3, P1, P2",  # temp = a - b
            "BGT P3, greater", # if temp > 0, jump
            "MOVI P4, 0",      # else: result = 0
            "JUMP end",
            "greater:",
            "MOVI P4, 1",      # result = 1
            "end:",
            "HALT"
        ]
        for i, line in enumerate(program4, 1):
            print(f"{i:2}. {line}")
        print()
    
    def generate_python_examples(self):
        """Generate Python code examples"""
        print("="*60)
        print("PYTHON CODE EXAMPLES")
        print("="*60 + "\n")
        
        # Example 1: Basic conversion
        print("Example 1: Basic Conversion")
        print("-" * 40)
        print("""
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()
pent = c.decimal_to_pentary(42)
print(pent)  # Output: ⊕⊖⊕

dec = c.pentary_to_decimal("⊕⊖⊕")
print(dec)   # Output: 42
""")
        
        # Example 2: Arithmetic
        print("\nExample 2: Arithmetic Operations")
        print("-" * 40)
        print("""
from tools.pentary_converter import PentaryConverter

c = PentaryConverter()

# Addition
result = c.add_pentary("+⊕", "⊕-")
print(result)  # Output: ⊕+0 (16 in decimal)

# Subtraction
result = c.subtract_pentary("⊕0", "+-")
print(result)  # Output: +⊕ (7 in decimal)

# Multiplication by constant
result = c.multiply_pentary_by_constant("+⊕", 2)
print(result)  # Output: ⊕+ (14 in decimal)
""")
        
        # Example 3: Simulator
        print("\nExample 3: Running a Program")
        print("-" * 40)
        print("""
from tools.pentary_simulator import PentaryProcessor

proc = PentaryProcessor()
program = [
    "MOVI P1, 5",
    "MOVI P2, 3",
    "ADD P3, P1, P2",
    "HALT"
]

proc.load_program(program)
proc.run(verbose=True)
proc.print_state()
""")
    
    def generate_all(self):
        """Generate all examples"""
        self.generate_conversion_examples()
        self.generate_arithmetic_examples()
        self.generate_program_examples()
        self.generate_python_examples()
    
    def interactive(self):
        """Interactive example generator"""
        print("\n" + "="*60)
        print("  PENTARY EXAMPLE GENERATOR")
        print("="*60 + "\n")
        print("What would you like to see?")
        print("  1. Number conversions")
        print("  2. Arithmetic operations")
        print("  3. Example programs")
        print("  4. Python code examples")
        print("  5. All examples")
        print("  6. Quit")
        print()
        
        while True:
            try:
                choice = input("Choice (1-6): ").strip()
                
                if choice == "1":
                    self.generate_conversion_examples()
                elif choice == "2":
                    self.generate_arithmetic_examples()
                elif choice == "3":
                    self.generate_program_examples()
                elif choice == "4":
                    self.generate_python_examples()
                elif choice == "5":
                    self.generate_all()
                elif choice == "6" or choice.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-6.")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main entry point"""
    import sys
    
    generator = ExampleGenerator()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            generator.generate_all()
        elif sys.argv[1] == "--conversions":
            generator.generate_conversion_examples()
        elif sys.argv[1] == "--arithmetic":
            generator.generate_arithmetic_examples()
        elif sys.argv[1] == "--programs":
            generator.generate_program_examples()
        elif sys.argv[1] == "--python":
            generator.generate_python_examples()
        else:
            print("Usage: python3 example_generator.py [--all|--conversions|--arithmetic|--programs|--python]")
            print("       Or run without arguments for interactive mode")
    else:
        generator.interactive()

if __name__ == "__main__":
    main()
