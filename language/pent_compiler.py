#!/usr/bin/env python3
"""
Pent Language Compiler
Compiles Pent source code to Pentary Assembly
"""

from typing import List, Dict, Optional
import sys
import os

# Ensure consistent imports by always importing from the language directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from pent_lexer import Lexer
from pent_parser import Parser, ASTNode, Function, LetStatement, ReturnStatement, \
    IfStatement, WhileStatement, ForStatement, BinaryExpression, UnaryExpression, \
    CallExpression, Literal, Identifier, Block, Program

from tools.pentary_converter import PentaryConverter


class CodeGenerator:
    """Generates pentary assembly code from AST"""
    
    def __init__(self):
        self.assembly: List[str] = []
        self.label_counter = 0
        self.register_counter = 1  # Start from P1 (P0 is always zero)
        self.variables: Dict[str, int] = {}  # Variable name -> register mapping
        self.functions: Dict[str, Function] = {}
        self.current_function: Optional[Function] = None
    
    def generate(self, ast: ASTNode) -> List[str]:
        """Generate assembly code from AST"""
        self.assembly = []
        
        # First pass: collect functions
        if isinstance(ast, Program):
            for stmt in ast.statements:
                if isinstance(stmt, Function):
                    self.functions[stmt.name] = stmt
        
        # Generate code for each function
        if isinstance(ast, Program):
            for stmt in ast.statements:
                if isinstance(stmt, Function):
                    self.generate_function(stmt)
        
        return self.assembly
    
    def generate_function(self, func: Function):
        """Generate code for a function"""
        self.current_function = func
        self.variables = {}
        self.register_counter = 1
        
        # Function label
        self.emit(f"{func.name}:")
        
        # Allocate registers for parameters
        for i, param in enumerate(func.params):
            reg = f"P{i + 1}"
            self.variables[param.name] = i + 1
        
        # Generate function body
        self.generate_block(func.body)
        
        # If no explicit return, add one
        if not any(isinstance(stmt, ReturnStatement) for stmt in func.body.statements):
            self.emit("RET")
        
        self.emit("")  # Blank line
    
    def generate_block(self, block: Block):
        """Generate code for a block"""
        for stmt in block.statements:
            self.generate_statement(stmt)
    
    def generate_statement(self, stmt: ASTNode):
        """Generate code for a statement"""
        if isinstance(stmt, LetStatement):
            self.generate_let(stmt)
        elif isinstance(stmt, ReturnStatement):
            self.generate_return(stmt)
        elif isinstance(stmt, IfStatement):
            self.generate_if(stmt)
        elif isinstance(stmt, WhileStatement):
            self.generate_while(stmt)
        elif isinstance(stmt, ForStatement):
            self.generate_for(stmt)
        elif isinstance(stmt, Block):
            self.generate_block(stmt)
        elif isinstance(stmt, BinaryExpression):
            # Expression statement (e.g., function call)
            self.generate_expression(stmt)
        elif isinstance(stmt, CallExpression):
            self.generate_call(stmt)
        else:
            # Other expression statements
            self.generate_expression(stmt)
    
    def generate_let(self, stmt: LetStatement):
        """Generate code for let statement"""
        # Allocate register for variable
        reg = self.allocate_register()
        self.variables[stmt.name] = reg
        
        # Generate expression and store in register
        result_reg = self.generate_expression(stmt.value)
        if result_reg != reg:
            self.emit(f"MOV P{reg}, P{result_reg}")
    
    def generate_return(self, stmt: ReturnStatement):
        """Generate code for return statement"""
        if stmt.value:
            result_reg = self.generate_expression(stmt.value)
            # Return value in P1 (convention)
            if result_reg != 1:
                self.emit(f"MOV P1, P{result_reg}")
        self.emit("RET")
    
    def generate_if(self, stmt: IfStatement):
        """Generate code for if statement"""
        else_label = self.new_label("else")
        end_label = self.new_label("endif")
        
        # Generate condition
        cond_reg = self.generate_expression(stmt.condition)
        
        # Branch to else if false
        if stmt.else_block:
            self.emit(f"BEQ P{cond_reg}, {else_label}")
        else:
            self.emit(f"BEQ P{cond_reg}, {end_label}")
        
        # Then block
        self.generate_block(stmt.then_block)
        
        if stmt.else_block:
            self.emit(f"JUMP {end_label}")
            self.emit(f"{else_label}:")
            self.generate_block(stmt.else_block)
        
        self.emit(f"{end_label}:")
    
    def generate_while(self, stmt: WhileStatement):
        """Generate code for while loop"""
        loop_label = self.new_label("loop")
        end_label = self.new_label("endloop")
        
        self.emit(f"{loop_label}:")
        
        # Generate condition
        cond_reg = self.generate_expression(stmt.condition)
        
        # Branch to end if condition is false
        self.emit(f"BEQ P{cond_reg}, {end_label}")
        
        # Generate body
        self.generate_block(stmt.body)
        
        # Jump back to loop
        self.emit(f"JUMP {loop_label}")
        self.emit(f"{end_label}:")
    
    def generate_for(self, stmt: ForStatement):
        """Generate code for for loop"""
        # For now, treat as while loop
        # TODO: Implement proper for loop generation
        self.generate_while(WhileStatement(stmt.iterable, stmt.body))
    
    def generate_expression(self, expr: ASTNode) -> int:
        """Generate code for expression, return register containing result"""
        if isinstance(expr, BinaryExpression):
            return self.generate_binary(expr)
        elif isinstance(expr, UnaryExpression):
            return self.generate_unary(expr)
        elif isinstance(expr, CallExpression):
            return self.generate_call(expr)
        elif isinstance(expr, Literal):
            return self.generate_literal(expr)
        elif isinstance(expr, Identifier):
            return self.generate_identifier(expr)
        else:
            raise Exception(f"Unknown expression type: {type(expr)}")
    
    def generate_binary(self, expr: BinaryExpression) -> int:
        """Generate code for binary expression"""
        left_reg = self.generate_expression(expr.left)
        right_reg = self.generate_expression(expr.right)
        result_reg = self.allocate_register()
        
        if expr.operator == "+":
            self.emit(f"ADD P{result_reg}, P{left_reg}, P{right_reg}")
        elif expr.operator == "-":
            self.emit(f"SUB P{result_reg}, P{left_reg}, P{right_reg}")
        elif expr.operator == "*":
            # General multiplication now supported
            if isinstance(expr.right, Literal) and expr.right.type == "int" and expr.right.value == 2:
                # Optimize multiplication by 2
                self.emit(f"MUL2 P{result_reg}, P{left_reg}")
            else:
                # General multiplication
                self.emit(f"MUL P{result_reg}, P{left_reg}, P{right_reg}")
        elif expr.operator == "/":
            # Integer division
            self.emit(f"DIV P{result_reg}, P{left_reg}, P{right_reg}")
        elif expr.operator == "%":
            # Modulo
            self.emit(f"MOD P{result_reg}, P{left_reg}, P{right_reg}")
        elif expr.operator == "<<":
            if isinstance(expr.right, Literal) and expr.right.type == "int":
                self.emit(f"SHL P{result_reg}, P{left_reg}, {expr.right.value}")
            else:
                raise Exception("Shift amount must be constant")
        elif expr.operator == ">>":
            if isinstance(expr.right, Literal) and expr.right.type == "int":
                self.emit(f"SHR P{result_reg}, P{left_reg}, {expr.right.value}")
            else:
                raise Exception("Shift amount must be constant")
        elif expr.operator == "==":
            # Compare and set result to 0 or 1
            temp_reg = self.allocate_register()
            eq_label = self.new_label('eq')
            done_label = self.new_label('eq_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BEQ P{temp_reg}, {eq_label}")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{eq_label}:")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"{done_label}:")
        elif expr.operator == "!=":
            # Not equal comparison
            temp_reg = self.allocate_register()
            ne_label = self.new_label('ne')
            done_label = self.new_label('ne_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BNE P{temp_reg}, {ne_label}")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{ne_label}:")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"{done_label}:")
        elif expr.operator == "<":
            # Less than comparison
            temp_reg = self.allocate_register()
            lt_label = self.new_label('lt')
            done_label = self.new_label('lt_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BLT P{temp_reg}, {lt_label}")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{lt_label}:")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"{done_label}:")
        elif expr.operator == ">":
            # Greater than comparison
            temp_reg = self.allocate_register()
            gt_label = self.new_label('gt')
            done_label = self.new_label('gt_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BGT P{temp_reg}, {gt_label}")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{gt_label}:")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"{done_label}:")
        elif expr.operator == "<=":
            # Less than or equal - swap and use > then negate
            temp_reg = self.allocate_register()
            le_label = self.new_label('le')
            done_label = self.new_label('le_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BGT P{temp_reg}, {le_label}")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{le_label}:")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"{done_label}:")
        elif expr.operator == ">=":
            # Greater than or equal
            temp_reg = self.allocate_register()
            ge_label = self.new_label('ge')
            done_label = self.new_label('ge_done')
            self.emit(f"SUB P{temp_reg}, P{left_reg}, P{right_reg}")
            self.emit(f"BLT P{temp_reg}, {ge_label}")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{ge_label}:")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"{done_label}:")
        else:
            raise Exception(f"Unsupported binary operator: {expr.operator}")
        
        return result_reg
    
    def generate_unary(self, expr: UnaryExpression) -> int:
        """Generate code for unary expression"""
        operand_reg = self.generate_expression(expr.operand)
        result_reg = self.allocate_register()
        
        if expr.operator == "-":
            self.emit(f"NEG P{result_reg}, P{operand_reg}")
        elif expr.operator == "!":
            # Logical NOT
            not_label = self.new_label('not')
            done_label = self.new_label('not_done')
            self.emit(f"BEQ P{operand_reg}, {not_label}")
            self.emit(f"MOVI P{result_reg}, 0")
            self.emit(f"JUMP {done_label}")
            self.emit(f"{not_label}:")
            self.emit(f"MOVI P{result_reg}, 1")
            self.emit(f"{done_label}:")
        else:
            raise Exception(f"Unsupported unary operator: {expr.operator}")
        
        return result_reg
    
    def generate_call(self, expr: CallExpression) -> int:
        """Generate code for function call"""
        # Push arguments to stack (in reverse order for simplicity)
        arg_regs = []
        for arg in reversed(expr.arguments):
            arg_reg = self.generate_expression(arg)
            arg_regs.append(arg_reg)
            self.emit(f"PUSH P{arg_reg}")
        
        # Call function
        if expr.name in self.functions:
            self.emit(f"CALL {expr.name}")
        else:
            # Assume it's a built-in function
            self.emit(f"# Call to {expr.name}")
        
        # Pop arguments (for now, just clear stack)
        for _ in arg_regs:
            self.emit("# POP (cleaned up by callee)")
        
        # Return value is in P1 (convention)
        result_reg = self.allocate_register()
        self.emit(f"MOV P{result_reg}, P1")
        return result_reg
    
    def generate_literal(self, expr: Literal) -> int:
        """Generate code for literal"""
        result_reg = self.allocate_register()
        
        if expr.type == "pent":
            # Pentary literal
            self.emit(f"MOVI P{result_reg}, {expr.value}")
        elif expr.type == "int":
            # Convert to pentary
            pentary = PentaryConverter.decimal_to_pentary(expr.value)
            self.emit(f"MOVI P{result_reg}, {pentary}")
        elif expr.type == "bool":
            # Boolean: true = 1, false = 0
            value = 1 if expr.value else 0
            pentary = PentaryConverter.decimal_to_pentary(value)
            self.emit(f"MOVI P{result_reg}, {pentary}")
        else:
            raise Exception(f"Unsupported literal type: {expr.type}")
        
        return result_reg
    
    def generate_identifier(self, expr: Identifier) -> int:
        """Generate code for identifier"""
        if expr.name in self.variables:
            return self.variables[expr.name]
        else:
            raise Exception(f"Undefined variable: {expr.name}")
    
    def allocate_register(self) -> int:
        """Allocate a new register"""
        reg = self.register_counter
        self.register_counter += 1
        if reg > 28:  # P1-P28 are available
            raise Exception("Out of registers")
        return reg
    
    def new_label(self, prefix: str) -> str:
        """Generate a new unique label"""
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label
    
    def emit(self, line: str):
        """Emit an assembly line"""
        self.assembly.append(line)


class Compiler:
    """Main compiler class"""
    
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.generator = CodeGenerator()
    
    def compile(self, source: str) -> List[str]:
        """Compile Pent source to assembly"""
        # Lex
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Generate
        generator = CodeGenerator()
        assembly = generator.generate(ast)
        
        return assembly
    
    def compile_file(self, filename: str) -> List[str]:
        """Compile a Pent source file"""
        with open(filename, 'r') as f:
            source = f.read()
        return self.compile(source)


def main():
    """Test the compiler"""
    source = """
    fn add(a: pent, b: pent) -> pent {
        return a + b;
    }
    
    fn main() {
        let x = ⊕+;
        let y = +⊕;
        let z = add(x, y);
    }
    """
    
    compiler = Compiler()
    assembly = compiler.compile(source)
    
    print("Generated Assembly:")
    print("=" * 60)
    for line in assembly:
        print(line)


if __name__ == "__main__":
    main()
