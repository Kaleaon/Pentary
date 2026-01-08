#!/usr/bin/env python3
"""
Pent Language Interpreter
Runtime interpreter for the Pent programming language
"""

import sys
import os

# Ensure consistent imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from typing import Dict, Any, List, Optional, Callable
from pent_lexer import Lexer
from pent_parser import (
    Parser, ASTNode, Program, Function, LetStatement, ReturnStatement,
    IfStatement, WhileStatement, ForStatement, Block,
    BinaryExpression, UnaryExpression, CallExpression, Literal, Identifier,
    ArrayLiteral, IndexExpression, StructDefinition, StructInstantiation,
    FieldAccess, Assignment, RangeExpression
)
from pent_stdlib import PentStdLib


class PentStruct:
    """Runtime representation of a struct instance"""
    def __init__(self, struct_name: str, fields: Dict[str, Any]):
        self.struct_name = struct_name
        self.fields = fields
    
    def __repr__(self):
        field_str = ", ".join(f"{k}: {v}" for k, v in self.fields.items())
        return f"{self.struct_name} {{ {field_str} }}"
    
    def get_field(self, name: str) -> Any:
        if name not in self.fields:
            raise AttributeError(f"Struct '{self.struct_name}' has no field '{name}'")
        return self.fields[name]
    
    def set_field(self, name: str, value: Any):
        if name not in self.fields:
            raise AttributeError(f"Struct '{self.struct_name}' has no field '{name}'")
        self.fields[name] = value

# Import pentary tools
try:
    from tools.pentary_converter import PentaryConverter
except ImportError:
    sys.path.insert(0, os.path.join(_parent_dir, 'tools'))
    from pentary_converter import PentaryConverter


class ReturnValue(Exception):
    """Exception to handle return statements"""
    def __init__(self, value):
        self.value = value


class BreakException(Exception):
    """Exception to handle break statements"""
    pass


class ContinueException(Exception):
    """Exception to handle continue statements"""
    pass


class Environment:
    """Variable environment with scope management"""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.variables: Dict[str, Any] = {}
        self.parent = parent
    
    def define(self, name: str, value: Any):
        """Define a variable in current scope"""
        self.variables[name] = value
    
    def get(self, name: str) -> Any:
        """Get variable value, checking parent scopes"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")
    
    def set(self, name: str, value: Any):
        """Set variable value, checking parent scopes"""
        if name in self.variables:
            self.variables[name] = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise NameError(f"Undefined variable: {name}")
    
    def exists(self, name: str) -> bool:
        """Check if variable exists"""
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.exists(name)
        return False


class PentInterpreter:
    """Interpreter for Pent programming language"""
    
    def __init__(self):
        self.global_env = Environment()
        self.functions: Dict[str, Function] = {}
        self.structs: Dict[str, StructDefinition] = {}  # Store struct definitions
        self.current_env = self.global_env
    
    def interpret(self, source: str) -> Any:
        """
        Interpret Pent source code.
        
        Args:
            source: Pent source code
            
        Returns:
            Return value of main() if exists, else None
        """
        # Parse source code
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Execute program
        return self.execute(ast)
    
    def execute(self, node: ASTNode) -> Any:
        """Execute an AST node"""
        if isinstance(node, Program):
            return self._execute_program(node)
        elif isinstance(node, Function):
            return self._execute_function_def(node)
        elif isinstance(node, StructDefinition):
            return self._execute_struct_def(node)
        elif isinstance(node, LetStatement):
            return self._execute_let(node)
        elif isinstance(node, ReturnStatement):
            return self._execute_return(node)
        elif isinstance(node, IfStatement):
            return self._execute_if(node)
        elif isinstance(node, WhileStatement):
            return self._execute_while(node)
        elif isinstance(node, ForStatement):
            return self._execute_for(node)
        elif isinstance(node, Block):
            return self._execute_block(node)
        elif isinstance(node, BinaryExpression):
            return self._execute_binary(node)
        elif isinstance(node, UnaryExpression):
            return self._execute_unary(node)
        elif isinstance(node, CallExpression):
            return self._execute_call(node)
        elif isinstance(node, Literal):
            return self._execute_literal(node)
        elif isinstance(node, Identifier):
            return self._execute_identifier(node)
        elif isinstance(node, ArrayLiteral):
            return self._execute_array_literal(node)
        elif isinstance(node, IndexExpression):
            return self._execute_index(node)
        elif isinstance(node, StructInstantiation):
            return self._execute_struct_instantiation(node)
        elif isinstance(node, FieldAccess):
            return self._execute_field_access(node)
        elif isinstance(node, Assignment):
            return self._execute_assignment(node)
        elif isinstance(node, RangeExpression):
            return self._execute_range(node)
        else:
            raise RuntimeError(f"Unknown node type: {type(node)}")
    
    def _execute_program(self, program: Program) -> Any:
        """Execute a program"""
        # First pass: collect function and struct definitions
        for stmt in program.statements:
            if isinstance(stmt, Function):
                self.functions[stmt.name] = stmt
            elif isinstance(stmt, StructDefinition):
                self.structs[stmt.name] = stmt
        
        # Second pass: execute top-level statements (if any non-functions/structs)
        for stmt in program.statements:
            if not isinstance(stmt, (Function, StructDefinition)):
                self.execute(stmt)
        
        # Call main() if it exists
        if 'main' in self.functions:
            return self._call_function('main', [])
        
        return None
    
    def _execute_function_def(self, func: Function) -> None:
        """Store function definition"""
        self.functions[func.name] = func
    
    def _execute_struct_def(self, struct: StructDefinition) -> None:
        """Store struct definition"""
        self.structs[struct.name] = struct
    
    def _execute_let(self, stmt: LetStatement) -> None:
        """Execute let statement"""
        value = self.execute(stmt.value)
        self.current_env.define(stmt.name, value)
    
    def _execute_return(self, stmt: ReturnStatement) -> None:
        """Execute return statement"""
        value = self.execute(stmt.value) if stmt.value else None
        raise ReturnValue(value)
    
    def _execute_if(self, stmt: IfStatement) -> Any:
        """Execute if statement"""
        condition = self.execute(stmt.condition)
        
        if self._is_truthy(condition):
            return self._execute_block(stmt.then_block)
        elif stmt.else_block:
            return self._execute_block(stmt.else_block)
        
        return None
    
    def _execute_while(self, stmt: WhileStatement) -> None:
        """Execute while statement"""
        while self._is_truthy(self.execute(stmt.condition)):
            try:
                self._execute_block(stmt.body)
            except BreakException:
                break
            except ContinueException:
                continue
    
    def _execute_for(self, stmt: ForStatement) -> None:
        """Execute for statement"""
        iterable = self.execute(stmt.iterable)
        
        # Handle range-like iterables
        if isinstance(iterable, range):
            iterable = list(iterable)
        elif isinstance(iterable, int):
            iterable = range(iterable)
        
        # Create new scope for loop variable
        old_env = self.current_env
        self.current_env = Environment(old_env)
        
        try:
            for value in iterable:
                self.current_env.define(stmt.variable, value)
                try:
                    self._execute_block(stmt.body)
                except BreakException:
                    break
                except ContinueException:
                    continue
        finally:
            self.current_env = old_env
    
    def _execute_block(self, block: Block) -> Any:
        """Execute a block of statements"""
        # Create new scope
        old_env = self.current_env
        self.current_env = Environment(old_env)
        
        result = None
        try:
            for stmt in block.statements:
                result = self.execute(stmt)
        finally:
            self.current_env = old_env
        
        return result
    
    def _execute_binary(self, expr: BinaryExpression) -> Any:
        """Execute binary expression"""
        left = self.execute(expr.left)
        
        # Short-circuit evaluation for logical operators
        if expr.operator == '&&':
            if not self._is_truthy(left):
                return False
            return self._is_truthy(self.execute(expr.right))
        
        if expr.operator == '||':
            if self._is_truthy(left):
                return True
            return self._is_truthy(self.execute(expr.right))
        
        right = self.execute(expr.right)
        
        # Arithmetic operators
        if expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return int(left / right)  # Integer division
        elif expr.operator == '%':
            if right == 0:
                raise ZeroDivisionError("Modulo by zero")
            return left % right
        
        # Shift operators (pentary: multiply/divide by 5)
        elif expr.operator == '<<':
            return left * (5 ** right)
        elif expr.operator == '>>':
            return int(left / (5 ** right))
        
        # Comparison operators
        elif expr.operator == '==':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>=':
            return left >= right
        
        else:
            raise RuntimeError(f"Unknown operator: {expr.operator}")
    
    def _execute_unary(self, expr: UnaryExpression) -> Any:
        """Execute unary expression"""
        operand = self.execute(expr.operand)
        
        if expr.operator == '-':
            return -operand
        elif expr.operator == '!':
            return not self._is_truthy(operand)
        else:
            raise RuntimeError(f"Unknown unary operator: {expr.operator}")
    
    def _execute_call(self, expr: CallExpression) -> Any:
        """Execute function call"""
        args = [self.execute(arg) for arg in expr.arguments]
        
        # Check standard library first
        if expr.name in PentStdLib.FUNCTIONS:
            return PentStdLib.call(expr.name, *args)
        
        # Check user-defined functions
        if expr.name in self.functions:
            return self._call_function(expr.name, args)
        
        raise NameError(f"Undefined function: {expr.name}")
    
    def _call_function(self, name: str, args: List[Any]) -> Any:
        """Call a user-defined function"""
        func = self.functions[name]
        
        # Check argument count
        if len(args) != len(func.params):
            raise TypeError(
                f"Function '{name}' expects {len(func.params)} arguments, got {len(args)}"
            )
        
        # Create new environment for function
        old_env = self.current_env
        self.current_env = Environment(self.global_env)  # Functions use global scope
        
        try:
            # Bind parameters
            for param, arg in zip(func.params, args):
                self.current_env.define(param.name, arg)
            
            # Execute function body
            self._execute_block(func.body)
            
            return None  # No explicit return
        except ReturnValue as ret:
            return ret.value
        finally:
            self.current_env = old_env
    
    def _execute_literal(self, expr: Literal) -> Any:
        """Execute literal expression"""
        if expr.type == 'pent':
            # Pentary literal - convert to decimal
            # The value should be a pentary string like "⊕+" or "+⊖"
            try:
                return PentaryConverter.pentary_to_decimal(str(expr.value))
            except (ValueError, AttributeError):
                # If conversion fails, it might already be a number
                return int(expr.value)
        elif expr.type == 'int':
            return int(expr.value)
        elif expr.type == 'float':
            return float(expr.value)
        elif expr.type == 'bool':
            if isinstance(expr.value, bool):
                return expr.value
            return str(expr.value).lower() == 'true'
        elif expr.type == 'string':
            return str(expr.value)
        else:
            return expr.value
    
    def _execute_identifier(self, expr: Identifier) -> Any:
        """Execute identifier (variable reference)"""
        return self.current_env.get(expr.name)
    
    def _execute_array_literal(self, expr: ArrayLiteral) -> List[Any]:
        """Execute array literal"""
        return [self.execute(elem) for elem in expr.elements]
    
    def _execute_index(self, expr: IndexExpression) -> Any:
        """Execute array index expression"""
        array = self.execute(expr.array)
        index = self.execute(expr.index)
        
        if not isinstance(array, list):
            raise TypeError(f"Cannot index non-array type: {type(array)}")
        
        if not isinstance(index, int):
            raise TypeError(f"Array index must be integer, got: {type(index)}")
        
        if index < 0 or index >= len(array):
            raise IndexError(f"Array index {index} out of bounds (length {len(array)})")
        
        return array[index]
    
    def _execute_struct_instantiation(self, expr: StructInstantiation) -> PentStruct:
        """Execute struct instantiation"""
        if expr.struct_name not in self.structs:
            raise NameError(f"Undefined struct: {expr.struct_name}")
        
        struct_def = self.structs[expr.struct_name]
        expected_fields = {f.name for f in struct_def.fields}
        provided_fields = set(expr.field_values.keys())
        
        # Check for missing fields
        missing = expected_fields - provided_fields
        if missing:
            raise ValueError(f"Missing fields in struct instantiation: {missing}")
        
        # Check for extra fields
        extra = provided_fields - expected_fields
        if extra:
            raise ValueError(f"Unknown fields in struct instantiation: {extra}")
        
        # Evaluate field values
        field_values = {
            name: self.execute(value)
            for name, value in expr.field_values.items()
        }
        
        return PentStruct(expr.struct_name, field_values)
    
    def _execute_field_access(self, expr: FieldAccess) -> Any:
        """Execute field access (struct.field)"""
        obj = self.execute(expr.object)
        
        if isinstance(obj, PentStruct):
            return obj.get_field(expr.field)
        elif isinstance(obj, list):
            # Support length property for arrays
            if expr.field == "len" or expr.field == "length":
                return len(obj)
            raise AttributeError(f"Arrays don't have field '{expr.field}'")
        else:
            raise TypeError(f"Cannot access field on type: {type(obj)}")
    
    def _execute_assignment(self, expr: Assignment) -> Any:
        """Execute assignment expression"""
        value = self.execute(expr.value)
        self.current_env.set(expr.name, value)
        return value
    
    def _execute_range(self, expr: RangeExpression) -> range:
        """Execute range expression (start..end)"""
        start = self.execute(expr.start)
        end = self.execute(expr.end)
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("Range bounds must be integers")
        
        return range(start, end)
    
    def _is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, PentStruct):
            return True
        return True


def run_file(filename: str):
    """Run a Pent source file"""
    with open(filename, 'r') as f:
        source = f.read()
    
    interpreter = PentInterpreter()
    result = interpreter.interpret(source)
    
    if result is not None:
        print(f"Result: {result}")


def run_repl():
    """Run interactive REPL"""
    print("Pent Language REPL")
    print("Type 'exit' to quit")
    print()
    
    interpreter = PentInterpreter()
    
    while True:
        try:
            line = input("pent> ")
            if line.strip() == 'exit':
                break
            if not line.strip():
                continue
            
            # Try to interpret the line
            result = interpreter.interpret(line)
            if result is not None:
                print(f"=> {result}")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def test_interpreter():
    """Test the interpreter"""
    print("Testing Pent Interpreter...")
    
    interpreter = PentInterpreter()
    
    # Test 1: Simple arithmetic
    result = interpreter.interpret("""
fn main() {
    let x = 5;
    let y = 3;
    return x + y;
}
""")
    assert result == 8, f"Expected 8, got {result}"
    print("  ✓ Simple arithmetic")
    
    # Test 2: Function calls
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn add(a: pent, b: pent) -> pent {
    return a + b;
}

fn main() {
    return add(10, 20);
}
""")
    assert result == 30, f"Expected 30, got {result}"
    print("  ✓ Function calls")
    
    # Test 3: If statement
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn main() {
    let x = 10;
    if x > 5 {
        return 1;
    } else {
        return 0;
    }
}
""")
    assert result == 1, f"Expected 1, got {result}"
    print("  ✓ If statement")
    
    # Test 4: While loop (using function-based accumulation since assignment not supported)
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn sum_to(n: int) -> int {
    if n <= 0 {
        return 0;
    }
    return n + sum_to(n - 1);
}

fn main() {
    return sum_to(5);
}
""")
    assert result == 15, f"Expected 15, got {result}"
    print("  ✓ Recursive sum (while loop alternative)")
    
    # Test 5: Pentary literals
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn main() {
    let x = ⊕+;
    let y = +⊕;
    return x + y;
}
""")
    # ⊕+ = 2*5 + 1 = 11
    # +⊕ = 1*5 + 2 = 7
    # Sum = 18
    assert result == 18, f"Expected 18, got {result}"
    print("  ✓ Pentary literals")
    
    # Test 6: Recursion
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn factorial(n: int) -> int {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() {
    return factorial(5);
}
""")
    assert result == 120, f"Expected 120, got {result}"
    print("  ✓ Recursion")
    
    # Test 7: Nested functions
    interpreter = PentInterpreter()
    result = interpreter.interpret("""
fn square(x: pent) -> pent {
    return x * x;
}

fn sum_of_squares(a: pent, b: pent) -> pent {
    return square(a) + square(b);
}

fn main() {
    return sum_of_squares(3, 4);
}
""")
    assert result == 25, f"Expected 25, got {result}"
    print("  ✓ Nested functions")
    
    print("\nAll interpreter tests passed!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_interpreter()
        else:
            run_file(sys.argv[1])
    else:
        test_interpreter()
