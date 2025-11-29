#!/usr/bin/env python3
"""
Pent Language Type Checker
Performs semantic analysis and type checking on the AST
"""

from typing import Dict, Optional, List, Set
import sys
import os

# Ensure consistent imports by always importing from the language directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from pent_parser import (
    ASTNode, Program, Function, Parameter, Block, LetStatement,
    ReturnStatement, IfStatement, WhileStatement, ForStatement,
    BinaryExpression, UnaryExpression, CallExpression, Literal,
    Identifier, Expression
)


class TypeError(Exception):
    """Type checking error"""
    pass


class SymbolTable:
    """Symbol table for tracking variables and their types"""
    
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, str] = {}
        self.parent = parent
    
    def define(self, name: str, type_: str):
        """Define a symbol in the current scope"""
        if name in self.symbols:
            raise TypeError(f"Variable '{name}' already defined in this scope")
        self.symbols[name] = type_
    
    def lookup(self, name: str) -> Optional[str]:
        """Look up a symbol, checking parent scopes"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def create_child(self) -> 'SymbolTable':
        """Create a child scope"""
        return SymbolTable(self)


class TypeChecker:
    """Type checker for Pent language"""
    
    # Built-in function signatures: name -> (param_types, return_type)
    BUILTINS = {
        'println': (['any'], 'void'),
        'print': (['any'], 'void'),
        'abs': (['pent'], 'pent'),
        'min': (['pent', 'pent'], 'pent'),
        'max': (['pent', 'pent'], 'pent'),
    }
    
    def __init__(self):
        self.global_scope = SymbolTable()
        self.current_scope = self.global_scope
        self.functions: Dict[str, Function] = {}
        self.current_function: Optional[Function] = None
        self.errors: List[str] = []
    
    def check(self, ast: ASTNode) -> List[str]:
        """
        Type check the AST.
        
        Returns:
            List of error messages (empty if type checking passed)
        """
        self.errors = []
        
        if isinstance(ast, Program):
            # First pass: collect function signatures
            for stmt in ast.statements:
                if isinstance(stmt, Function):
                    self.functions[stmt.name] = stmt
            
            # Second pass: check each function
            for stmt in ast.statements:
                self.check_node(stmt)
        else:
            self.check_node(ast)
        
        return self.errors
    
    def error(self, message: str):
        """Record an error"""
        self.errors.append(message)
    
    def check_node(self, node: ASTNode) -> Optional[str]:
        """
        Check a node and return its type.
        
        Returns:
            Type string or None for statements
        """
        if isinstance(node, Program):
            for stmt in node.statements:
                self.check_node(stmt)
            return None
        
        elif isinstance(node, Function):
            return self.check_function(node)
        
        elif isinstance(node, LetStatement):
            return self.check_let(node)
        
        elif isinstance(node, ReturnStatement):
            return self.check_return(node)
        
        elif isinstance(node, IfStatement):
            return self.check_if(node)
        
        elif isinstance(node, WhileStatement):
            return self.check_while(node)
        
        elif isinstance(node, ForStatement):
            return self.check_for(node)
        
        elif isinstance(node, Block):
            return self.check_block(node)
        
        elif isinstance(node, BinaryExpression):
            return self.check_binary(node)
        
        elif isinstance(node, UnaryExpression):
            return self.check_unary(node)
        
        elif isinstance(node, CallExpression):
            return self.check_call(node)
        
        elif isinstance(node, Literal):
            return self.check_literal(node)
        
        elif isinstance(node, Identifier):
            return self.check_identifier(node)
        
        else:
            return None
    
    def check_function(self, func: Function) -> None:
        """Check a function definition"""
        # Save current function for return type checking
        prev_function = self.current_function
        self.current_function = func
        
        # Create new scope for function
        prev_scope = self.current_scope
        self.current_scope = self.current_scope.create_child()
        
        # Add parameters to scope
        for param in func.params:
            self.current_scope.define(param.name, param.type)
        
        # Check function body
        self.check_block(func.body)
        
        # Restore scope and function context
        self.current_scope = prev_scope
        self.current_function = prev_function
    
    def check_let(self, stmt: LetStatement) -> None:
        """Check a let statement"""
        # Check the value expression
        value_type = self.check_node(stmt.value)
        
        # Determine variable type
        if stmt.type:
            var_type = stmt.type
            # Check type compatibility
            if value_type and value_type != 'any' and var_type != value_type:
                # Allow some implicit conversions
                if not self.is_compatible(var_type, value_type):
                    self.error(f"Type mismatch: cannot assign {value_type} to {var_type}")
        else:
            var_type = value_type or 'any'
        
        # Define in current scope
        try:
            self.current_scope.define(stmt.name, var_type)
        except TypeError as e:
            self.error(str(e))
    
    def check_return(self, stmt: ReturnStatement) -> None:
        """Check a return statement"""
        if stmt.value:
            return_type = self.check_node(stmt.value)
            
            if self.current_function and self.current_function.return_type:
                if return_type and not self.is_compatible(
                    self.current_function.return_type, return_type
                ):
                    self.error(
                        f"Return type mismatch: expected {self.current_function.return_type}, "
                        f"got {return_type}"
                    )
    
    def check_if(self, stmt: IfStatement) -> None:
        """Check an if statement"""
        cond_type = self.check_node(stmt.condition)
        
        # Condition should be boolean-compatible
        if cond_type and cond_type not in ['bool', 'pent', 'int', 'any']:
            self.error(f"If condition must be boolean, got {cond_type}")
        
        self.check_block(stmt.then_block)
        
        if stmt.else_block:
            self.check_block(stmt.else_block)
    
    def check_while(self, stmt: WhileStatement) -> None:
        """Check a while statement"""
        cond_type = self.check_node(stmt.condition)
        
        if cond_type and cond_type not in ['bool', 'pent', 'int', 'any']:
            self.error(f"While condition must be boolean, got {cond_type}")
        
        self.check_block(stmt.body)
    
    def check_for(self, stmt: ForStatement) -> None:
        """Check a for statement"""
        # Create scope for loop variable
        prev_scope = self.current_scope
        self.current_scope = self.current_scope.create_child()
        
        # Define loop variable (assume pent type for now)
        self.current_scope.define(stmt.variable, 'pent')
        
        self.check_node(stmt.iterable)
        self.check_block(stmt.body)
        
        self.current_scope = prev_scope
    
    def check_block(self, block: Block) -> None:
        """Check a block of statements"""
        # Create new scope for block
        prev_scope = self.current_scope
        self.current_scope = self.current_scope.create_child()
        
        for stmt in block.statements:
            self.check_node(stmt)
        
        self.current_scope = prev_scope
    
    def check_binary(self, expr: BinaryExpression) -> str:
        """Check a binary expression and return its type"""
        left_type = self.check_node(expr.left)
        right_type = self.check_node(expr.right)
        
        # Arithmetic operators
        if expr.operator in ['+', '-', '*', '/', '%', '<<', '>>']:
            # Both operands should be numeric
            if left_type and left_type not in ['pent', 'int', 'float', 'any']:
                self.error(f"Left operand of '{expr.operator}' must be numeric, got {left_type}")
            if right_type and right_type not in ['pent', 'int', 'float', 'any']:
                self.error(f"Right operand of '{expr.operator}' must be numeric, got {right_type}")
            
            # Result type
            if left_type == 'float' or right_type == 'float':
                return 'float'
            return 'pent'
        
        # Comparison operators
        elif expr.operator in ['==', '!=', '<', '>', '<=', '>=']:
            return 'bool'
        
        # Logical operators
        elif expr.operator in ['&&', '||']:
            return 'bool'
        
        else:
            self.error(f"Unknown binary operator: {expr.operator}")
            return 'any'
    
    def check_unary(self, expr: UnaryExpression) -> str:
        """Check a unary expression and return its type"""
        operand_type = self.check_node(expr.operand)
        
        if expr.operator == '-':
            if operand_type and operand_type not in ['pent', 'int', 'float', 'any']:
                self.error(f"Unary '-' requires numeric operand, got {operand_type}")
            return operand_type or 'pent'
        
        elif expr.operator == '!':
            return 'bool'
        
        else:
            self.error(f"Unknown unary operator: {expr.operator}")
            return 'any'
    
    def check_call(self, expr: CallExpression) -> str:
        """Check a function call and return its type"""
        # Check if function exists
        if expr.name in self.BUILTINS:
            param_types, return_type = self.BUILTINS[expr.name]
            # Check argument count
            if len(expr.arguments) != len(param_types) and param_types[0] != 'any':
                self.error(
                    f"Function '{expr.name}' expects {len(param_types)} arguments, "
                    f"got {len(expr.arguments)}"
                )
            return return_type
        
        elif expr.name in self.functions:
            func = self.functions[expr.name]
            
            # Check argument count
            if len(expr.arguments) != len(func.params):
                self.error(
                    f"Function '{expr.name}' expects {len(func.params)} arguments, "
                    f"got {len(expr.arguments)}"
                )
            
            # Check argument types
            for i, (arg, param) in enumerate(zip(expr.arguments, func.params)):
                arg_type = self.check_node(arg)
                if arg_type and not self.is_compatible(param.type, arg_type):
                    self.error(
                        f"Argument {i+1} of '{expr.name}': expected {param.type}, "
                        f"got {arg_type}"
                    )
            
            return func.return_type or 'void'
        
        else:
            self.error(f"Undefined function: {expr.name}")
            return 'any'
    
    def check_literal(self, expr: Literal) -> str:
        """Check a literal and return its type"""
        return expr.type
    
    def check_identifier(self, expr: Identifier) -> str:
        """Check an identifier and return its type"""
        var_type = self.current_scope.lookup(expr.name)
        
        if var_type is None:
            self.error(f"Undefined variable: {expr.name}")
            return 'any'
        
        return var_type
    
    def is_compatible(self, expected: str, actual: str) -> bool:
        """Check if types are compatible"""
        if expected == actual:
            return True
        if expected == 'any' or actual == 'any':
            return True
        # Allow pent and int to be interchangeable
        if expected in ['pent', 'int'] and actual in ['pent', 'int']:
            return True
        return False


def main():
    """Test the type checker"""
    from pent_lexer import Lexer
    from pent_parser import Parser
    
    source = """fn add(a: pent, b: pent) -> pent {
    return a + b;
}

fn main() {
    let x = 10;
    let y = 5;
    let z = add(x, y);
    let result = x * y / 2;
}
"""
    
    # Parse
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Type check
    checker = TypeChecker()
    errors = checker.check(ast)
    
    if errors:
        print("Type checking errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Type checking passed!")


if __name__ == "__main__":
    main()
