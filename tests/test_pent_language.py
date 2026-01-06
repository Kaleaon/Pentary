#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pent Programming Language

Tests:
1. Lexer - tokenization
2. Parser - AST generation
3. Type Checker - semantic analysis
4. Compiler - assembly generation
5. Interpreter - runtime execution
"""

import sys
import os
import pytest

# Add language directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLexer:
    """Tests for Pent Lexer"""
    
    def test_lexer_import(self):
        """Test that lexer can be imported."""
        from pent_lexer import Lexer, Token, TokenType
        assert Lexer is not None
        assert Token is not None
        assert TokenType is not None
    
    def test_lexer_basic_tokens(self):
        """Test basic token recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = "let x = 5;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        token_types = [t.type for t in tokens]
        assert TokenType.LET in token_types
        assert TokenType.IDENTIFIER in token_types
        assert TokenType.EQUAL in token_types
        assert TokenType.INTEGER in token_types
        assert TokenType.SEMICOLON in token_types
        assert TokenType.EOF in token_types
    
    def test_lexer_pentary_literals(self):
        """Test pentary literal recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = "let x = ⊕+; let y = +⊕;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        pentary_tokens = [t for t in tokens if t.type == TokenType.PENTARY]
        assert len(pentary_tokens) == 2
        assert pentary_tokens[0].value == "⊕+"
        assert pentary_tokens[1].value == "+⊕"
    
    def test_lexer_operators(self):
        """Test operator recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = "a + b - c * d / e % f"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_ops = [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, 
                       TokenType.SLASH, TokenType.PERCENT]
        actual_ops = [t.type for t in tokens if t.type in expected_ops]
        assert actual_ops == expected_ops
    
    def test_lexer_comparison_operators(self):
        """Test comparison operator recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = "a == b != c < d > e <= f >= g"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_ops = [TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL, 
                       TokenType.LESS, TokenType.GREATER,
                       TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL]
        actual_ops = [t.type for t in tokens if t.type in expected_ops]
        assert actual_ops == expected_ops
    
    def test_lexer_keywords(self):
        """Test keyword recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = "fn let mut const if else while for return"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected = [TokenType.FN, TokenType.LET, TokenType.MUT, 
                   TokenType.CONST, TokenType.IF, TokenType.ELSE,
                   TokenType.WHILE, TokenType.FOR, TokenType.RETURN]
        actual = [t.type for t in tokens[:-1]]  # Exclude EOF
        assert actual == expected
    
    def test_lexer_strings(self):
        """Test string literal recognition."""
        from pent_lexer import Lexer, TokenType
        
        source = '"Hello, World!"'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "Hello, World!"
    
    def test_lexer_comments(self):
        """Test comment handling."""
        from pent_lexer import Lexer, TokenType
        
        source = """
        // This is a comment
        let x = 5;  // Inline comment
        /* Multi-line
           comment */
        let y = 10;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        # Comments should be ignored
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 0
        
        # But the code should still be tokenized
        let_tokens = [t for t in tokens if t.type == TokenType.LET]
        assert len(let_tokens) == 2


class TestParser:
    """Tests for Pent Parser"""
    
    def test_parser_import(self):
        """Test that parser can be imported."""
        from pent_parser import Parser, Program, Function
        assert Parser is not None
        assert Program is not None
        assert Function is not None
    
    def test_parser_function_declaration(self):
        """Test function declaration parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, Function
        
        source = """
        fn add(a: pent, b: pent) -> pent {
            return a + b;
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], Function)
        assert ast.statements[0].name == "add"
        assert len(ast.statements[0].params) == 2
        assert ast.statements[0].return_type == "pent"
    
    def test_parser_let_statement(self):
        """Test let statement parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, LetStatement
        
        source = "fn main() { let x = 5; }"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        main_func = ast.statements[0]
        let_stmt = main_func.body.statements[0]
        assert isinstance(let_stmt, LetStatement)
        assert let_stmt.name == "x"
    
    def test_parser_if_statement(self):
        """Test if statement parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, IfStatement
        
        source = """
        fn main() {
            if x > 5 {
                return 1;
            } else {
                return 0;
            }
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        main_func = ast.statements[0]
        if_stmt = main_func.body.statements[0]
        assert isinstance(if_stmt, IfStatement)
        assert if_stmt.else_block is not None
    
    def test_parser_while_statement(self):
        """Test while statement parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, WhileStatement
        
        source = """
        fn main() {
            while x < 10 {
                x = x + 1;
            }
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        main_func = ast.statements[0]
        while_stmt = main_func.body.statements[0]
        assert isinstance(while_stmt, WhileStatement)
    
    def test_parser_binary_expression(self):
        """Test binary expression parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, BinaryExpression
        
        source = "fn main() { let x = 1 + 2 * 3; }"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        main_func = ast.statements[0]
        let_stmt = main_func.body.statements[0]
        # The value should be a binary expression
        assert isinstance(let_stmt.value, BinaryExpression)
    
    def test_parser_function_call(self):
        """Test function call parsing."""
        from pent_lexer import Lexer
        from pent_parser import Parser, CallExpression
        
        source = """
        fn main() {
            let result = add(1, 2);
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        main_func = ast.statements[0]
        let_stmt = main_func.body.statements[0]
        assert isinstance(let_stmt.value, CallExpression)
        assert let_stmt.value.name == "add"
        assert len(let_stmt.value.arguments) == 2


class TestTypeChecker:
    """Tests for Pent Type Checker"""
    
    def test_type_checker_import(self):
        """Test that type checker can be imported."""
        from pent_type_checker import TypeChecker
        assert TypeChecker is not None
    
    def test_type_checker_valid_program(self):
        """Test type checking of valid program."""
        from pent_lexer import Lexer
        from pent_parser import Parser
        from pent_type_checker import TypeChecker
        
        source = """
        fn add(a: pent, b: pent) -> pent {
            return a + b;
        }
        
        fn main() {
            let x = 5;
            let y = 10;
            let z = add(x, y);
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        checker = TypeChecker()
        errors = checker.check(ast)
        assert len(errors) == 0, f"Unexpected errors: {errors}"
    
    def test_type_checker_undefined_variable(self):
        """Test detection of undefined variables."""
        from pent_lexer import Lexer
        from pent_parser import Parser
        from pent_type_checker import TypeChecker
        
        source = """
        fn main() {
            let x = y;
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        checker = TypeChecker()
        errors = checker.check(ast)
        assert any("Undefined variable" in e for e in errors)
    
    def test_type_checker_undefined_function(self):
        """Test detection of undefined functions."""
        from pent_lexer import Lexer
        from pent_parser import Parser
        from pent_type_checker import TypeChecker
        
        source = """
        fn main() {
            let x = unknown_func(5);
        }
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        checker = TypeChecker()
        errors = checker.check(ast)
        assert any("Undefined function" in e for e in errors)


class TestCompiler:
    """Tests for Pent Compiler"""
    
    def test_compiler_import(self):
        """Test that compiler can be imported."""
        from pent_compiler import Compiler, CodeGenerator
        assert Compiler is not None
        assert CodeGenerator is not None
    
    def test_compiler_basic_function(self):
        """Test compilation of basic function."""
        from pent_compiler import Compiler
        
        source = """
        fn add(a: pent, b: pent) -> pent {
            return a + b;
        }
        """
        compiler = Compiler()
        assembly = compiler.compile(source)
        
        # Check that assembly contains expected instructions
        assembly_str = '\n'.join(assembly)
        assert 'add:' in assembly_str
        assert 'ADD' in assembly_str
        assert 'RET' in assembly_str
    
    def test_compiler_if_statement(self):
        """Test compilation of if statement."""
        from pent_compiler import Compiler
        
        source = """
        fn main() {
            let x = 5;
            if x > 0 {
                return 1;
            }
            return 0;
        }
        """
        compiler = Compiler()
        assembly = compiler.compile(source)
        
        assembly_str = '\n'.join(assembly)
        # Should contain branch instructions
        assert 'JUMP' in assembly_str or 'BEQ' in assembly_str or 'BNE' in assembly_str
    
    def test_compiler_arithmetic(self):
        """Test compilation of arithmetic operations."""
        from pent_compiler import Compiler
        
        source = """
        fn main() {
            let a = 10;
            let b = 5;
            let sum = a + b;
            let diff = a - b;
            let prod = a * b;
            let quot = a / b;
        }
        """
        compiler = Compiler()
        assembly = compiler.compile(source)
        
        assembly_str = '\n'.join(assembly)
        assert 'ADD' in assembly_str
        assert 'SUB' in assembly_str
        assert 'MUL' in assembly_str
        assert 'DIV' in assembly_str


class TestInterpreter:
    """Tests for Pent Interpreter"""
    
    def test_interpreter_import(self):
        """Test that interpreter can be imported."""
        from pent_interpreter import PentInterpreter
        assert PentInterpreter is not None
    
    def test_interpreter_arithmetic(self):
        """Test arithmetic operations."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn main() {
            let a = 10;
            let b = 5;
            return a + b;
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 15
    
    def test_interpreter_subtraction(self):
        """Test subtraction."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 10 - 3; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 7
    
    def test_interpreter_multiplication(self):
        """Test multiplication."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 6 * 7; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 42
    
    def test_interpreter_division(self):
        """Test division."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 20 / 4; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 5
    
    def test_interpreter_modulo(self):
        """Test modulo."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 17 % 5; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 2
    
    def test_interpreter_negation(self):
        """Test unary negation."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return -5; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == -5
    
    def test_interpreter_comparison_equal(self):
        """Test equality comparison."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { if 5 == 5 { return 1; } return 0; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 1
    
    def test_interpreter_comparison_less_than(self):
        """Test less than comparison."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { if 3 < 5 { return 1; } return 0; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 1
    
    def test_interpreter_comparison_greater_than(self):
        """Test greater than comparison."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { if 10 > 5 { return 1; } return 0; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 1
    
    def test_interpreter_function_call(self):
        """Test function calls."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn add(a: pent, b: pent) -> pent {
            return a + b;
        }
        
        fn main() {
            return add(3, 4);
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 7
    
    def test_interpreter_recursion(self):
        """Test recursive function calls."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn factorial(n: int) -> int {
            if n <= 1 {
                return 1;
            }
            return n * factorial(n - 1);
        }
        
        fn main() {
            return factorial(5);
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 120
    
    def test_interpreter_fibonacci(self):
        """Test Fibonacci sequence."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn fib(n: int) -> int {
            if n <= 1 {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        
        fn main() {
            return fib(10);
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 55
    
    def test_interpreter_pentary_literals(self):
        """Test pentary literal evaluation."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn main() {
            let x = ⊕+;
            let y = +⊕;
            return x + y;
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        # ⊕+ = 2*5 + 1 = 11
        # +⊕ = 1*5 + 2 = 7
        # Total = 18
        assert result == 18
    
    def test_interpreter_nested_functions(self):
        """Test nested function calls."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn square(x: pent) -> pent {
            return x * x;
        }
        
        fn sum_of_squares(a: pent, b: pent) -> pent {
            return square(a) + square(b);
        }
        
        fn main() {
            return sum_of_squares(3, 4);
        }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 25  # 9 + 16


class TestStdLib:
    """Tests for Pent Standard Library"""
    
    def test_stdlib_import(self):
        """Test that stdlib can be imported."""
        from pent_stdlib import PentStdLib
        assert PentStdLib is not None
    
    def test_stdlib_math_functions(self):
        """Test math functions."""
        from pent_stdlib import PentStdLib
        
        assert PentStdLib.call('abs', -5) == 5
        assert PentStdLib.call('abs', 5) == 5
        assert PentStdLib.call('min', 3, 7) == 3
        assert PentStdLib.call('max', 3, 7) == 7
        assert PentStdLib.call('clamp', 10, 0, 5) == 5
        assert PentStdLib.call('clamp', -5, 0, 10) == 0
        assert PentStdLib.call('sign', -10) == -1
        assert PentStdLib.call('sign', 0) == 0
        assert PentStdLib.call('sign', 10) == 1
        assert PentStdLib.call('pow', 2, 10) == 1024
    
    def test_stdlib_array_functions(self):
        """Test array functions."""
        from pent_stdlib import PentStdLib
        
        arr = PentStdLib.call('array_new', 5, 0)
        assert PentStdLib.call('array_len', arr) == 5
        
        PentStdLib.call('array_set', arr, 2, 42)
        assert PentStdLib.call('array_get', arr, 2) == 42
        
        arr = [1, 2, 3, 4, 5]
        assert PentStdLib.call('array_sum', arr) == 15
        assert PentStdLib.call('array_min', arr) == 1
        assert PentStdLib.call('array_max', arr) == 5
    
    def test_stdlib_matrix_functions(self):
        """Test matrix functions."""
        from pent_stdlib import PentStdLib
        
        mat = PentStdLib.call('matrix_new', 3, 4, 1)
        assert PentStdLib.call('matrix_rows', mat) == 3
        assert PentStdLib.call('matrix_cols', mat) == 4
        
        PentStdLib.call('matrix_set', mat, 1, 2, 99)
        assert PentStdLib.call('matrix_get', mat, 1, 2) == 99
    
    def test_stdlib_neural_functions(self):
        """Test neural network functions."""
        from pent_stdlib import PentStdLib
        
        assert PentStdLib.call('relu', -5) == 0
        assert PentStdLib.call('relu', 5) == 5
        assert PentStdLib.call('relu', 0) == 0
    
    def test_stdlib_memory_functions(self):
        """Test memory management functions."""
        from pent_stdlib import PentStdLib
        
        ptr = PentStdLib.call('alloc', 10)
        PentStdLib.call('store', ptr, 0, 42)
        assert PentStdLib.call('load', ptr, 0) == 42
        PentStdLib.call('free', ptr)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_division_by_zero(self):
        """Test division by zero handling."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 10 / 0; }"
        interpreter = PentInterpreter()
        
        with pytest.raises(ZeroDivisionError):
            interpreter.interpret(source)
    
    def test_modulo_by_zero(self):
        """Test modulo by zero handling."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 10 % 0; }"
        interpreter = PentInterpreter()
        
        with pytest.raises(ZeroDivisionError):
            interpreter.interpret(source)
    
    def test_undefined_variable_runtime(self):
        """Test undefined variable at runtime."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return undefined_var; }"
        interpreter = PentInterpreter()
        
        with pytest.raises(NameError):
            interpreter.interpret(source)
    
    def test_function_argument_count_mismatch(self):
        """Test function argument count mismatch."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn add(a: pent, b: pent) -> pent {
            return a + b;
        }
        
        fn main() {
            return add(1);
        }
        """
        interpreter = PentInterpreter()
        
        with pytest.raises(TypeError):
            interpreter.interpret(source)
    
    def test_empty_program(self):
        """Test empty program handling."""
        from pent_lexer import Lexer
        from pent_parser import Parser
        
        source = ""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 0
    
    def test_large_numbers(self):
        """Test large number handling."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return 1000000 * 1000; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 1000000000
    
    def test_negative_numbers(self):
        """Test negative number handling."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return -10 + -5; }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == -15
    
    def test_deeply_nested_expressions(self):
        """Test deeply nested expressions."""
        from pent_interpreter import PentInterpreter
        
        source = "fn main() { return ((((1 + 2) * 3) - 4) / 5); }"
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 1  # ((((3) * 3) - 4) / 5) = ((9 - 4) / 5) = (5 / 5) = 1
    
    def test_multiple_functions(self):
        """Test multiple function definitions."""
        from pent_interpreter import PentInterpreter
        
        source = """
        fn a() -> int { return 1; }
        fn b() -> int { return 2; }
        fn c() -> int { return 3; }
        fn main() { return a() + b() + c(); }
        """
        interpreter = PentInterpreter()
        result = interpreter.interpret(source)
        assert result == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
