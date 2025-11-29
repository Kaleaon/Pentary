#!/usr/bin/env python3
"""
Pent Language Parser
Parses tokens into an Abstract Syntax Tree (AST)
"""

from typing import List, Optional, Dict, Any
import sys
import os

# Ensure consistent imports by always importing from pent_lexer directly
# when run from the language directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from pent_lexer import Token, TokenType, Lexer


class ASTNode:
    """Base class for AST nodes"""
    def __init__(self, token: Optional[Token] = None):
        self.token = token
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Program(ASTNode):
    """Root node representing a program"""
    def __init__(self, statements: List[ASTNode]):
        super().__init__()
        self.statements = statements
    
    def __repr__(self):
        return f"Program({len(self.statements)} statements)"


class Function(ASTNode):
    """Function definition"""
    def __init__(self, name: str, params: List['Parameter'], return_type: Optional[str], body: 'Block'):
        super().__init__()
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
    
    def __repr__(self):
        return f"Function({self.name}, {len(self.params)} params)"


class Parameter(ASTNode):
    """Function parameter"""
    def __init__(self, name: str, type: str):
        super().__init__()
        self.name = name
        self.type = type
    
    def __repr__(self):
        return f"Parameter({self.name}: {self.type})"


class Block(ASTNode):
    """Block of statements"""
    def __init__(self, statements: List[ASTNode]):
        super().__init__()
        self.statements = statements
    
    def __repr__(self):
        return f"Block({len(self.statements)} statements)"


class LetStatement(ASTNode):
    """Let statement: let [mut] name [: type] = expression"""
    def __init__(self, name: str, mutable: bool, type: Optional[str], value: ASTNode):
        super().__init__()
        self.name = name
        self.mutable = mutable
        self.type = type
        self.value = value
    
    def __repr__(self):
        return f"Let({self.name}, mut={self.mutable})"


class ReturnStatement(ASTNode):
    """Return statement"""
    def __init__(self, value: Optional[ASTNode]):
        super().__init__()
        self.value = value
    
    def __repr__(self):
        return f"Return({self.value})"


class IfStatement(ASTNode):
    """If statement"""
    def __init__(self, condition: ASTNode, then_block: Block, else_block: Optional[Block]):
        super().__init__()
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block
    
    def __repr__(self):
        return f"If(else={self.else_block is not None})"


class WhileStatement(ASTNode):
    """While loop"""
    def __init__(self, condition: ASTNode, body: Block):
        super().__init__()
        self.condition = condition
        self.body = body
    
    def __repr__(self):
        return "While()"


class ForStatement(ASTNode):
    """For loop"""
    def __init__(self, variable: str, iterable: ASTNode, body: Block):
        super().__init__()
        self.variable = variable
        self.iterable = iterable
        self.body = body
    
    def __repr__(self):
        return f"For({self.variable})"


class Expression(ASTNode):
    """Base class for expressions"""
    pass


class BinaryExpression(Expression):
    """Binary expression: left op right"""
    def __init__(self, left: Expression, operator: str, right: Expression):
        super().__init__()
        self.left = left
        self.operator = operator
        self.right = right
    
    def __repr__(self):
        return f"Binary({self.operator})"


class UnaryExpression(Expression):
    """Unary expression: op operand"""
    def __init__(self, operator: str, operand: Expression):
        super().__init__()
        self.operator = operator
        self.operand = operand
    
    def __repr__(self):
        return f"Unary({self.operator})"


class CallExpression(Expression):
    """Function call: name(args...)"""
    def __init__(self, name: str, arguments: List[Expression]):
        super().__init__()
        self.name = name
        self.arguments = arguments
    
    def __repr__(self):
        return f"Call({self.name}, {len(self.arguments)} args)"


class Literal(Expression):
    """Literal value"""
    def __init__(self, value: Any, type: str):
        super().__init__()
        self.value = value
        self.type = type
    
    def __repr__(self):
        return f"Literal({self.value}, {self.type})"


class Identifier(Expression):
    """Identifier reference"""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def __repr__(self):
        return f"Identifier({self.name})"


class Parser:
    """Parser for Pent language"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
    
    def parse(self) -> Program:
        """Parse tokens into AST"""
        statements = []
        while not self.is_at_end():
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
        return Program(statements)
    
    def declaration(self) -> Optional[ASTNode]:
        """Parse a declaration (function, variable, etc.)"""
        if self.check(TokenType.FN):
            return self.function_declaration()
        elif self.check(TokenType.LET):
            return self.let_statement()
        else:
            return self.statement()
    
    def function_declaration(self) -> Function:
        """Parse function declaration: fn name(params) -> type? { body }"""
        self.consume(TokenType.FN, "Expected 'fn'")
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        
        # Parameters
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        params = []
        if not self.check(TokenType.RIGHT_PAREN):
            params = self.parameter_list()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        # Return type
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.type_annotation()
        
        # Body
        body = self.block()
        
        return Function(name, params, return_type, body)
    
    def parameter_list(self) -> List[Parameter]:
        """Parse parameter list"""
        params = []
        while True:
            name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
            self.consume(TokenType.COLON, "Expected ':' after parameter name")
            param_type = self.type_annotation()
            params.append(Parameter(name, param_type))
            
            if not self.match(TokenType.COMMA):
                break
        return params
    
    def type_annotation(self) -> str:
        """Parse type annotation"""
        if self.match(TokenType.TYPE_PENT):
            return "pent"
        elif self.match(TokenType.TYPE_INT):
            return "int"
        elif self.match(TokenType.TYPE_FLOAT):
            return "float"
        elif self.match(TokenType.TYPE_BOOL):
            return "bool"
        elif self.match(TokenType.TYPE_VOID):
            return "void"
        else:
            # Assume identifier type (for custom types)
            return self.consume(TokenType.IDENTIFIER, "Expected type").value
    
    def let_statement(self) -> LetStatement:
        """Parse let statement: let [mut] name [: type] = expression"""
        self.consume(TokenType.LET, "Expected 'let'")
        mutable = self.match(TokenType.MUT)
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            type_annotation = self.type_annotation()
        
        self.consume(TokenType.EQUAL, "Expected '=' after variable declaration")
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        
        return LetStatement(name, mutable, type_annotation, value)
    
    def statement(self) -> ASTNode:
        """Parse a statement"""
        if self.match(TokenType.RETURN):
            return self.return_statement()
        elif self.match(TokenType.IF):
            return self.if_statement()
        elif self.match(TokenType.WHILE):
            return self.while_statement()
        elif self.match(TokenType.FOR):
            return self.for_statement()
        elif self.match(TokenType.LEFT_BRACE):
            return self.block()
        else:
            # Expression statement
            expr = self.expression()
            # Semicolon is optional in some contexts (e.g., last statement in block)
            if not self.check(TokenType.RIGHT_BRACE) and not self.check(TokenType.EOF):
                self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
            return expr
    
    def return_statement(self) -> ReturnStatement:
        """Parse return statement"""
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after return")
        return ReturnStatement(value)
    
    def if_statement(self) -> IfStatement:
        """Parse if statement"""
        condition = self.expression()
        then_block = self.block()
        
        else_block = None
        if self.match(TokenType.ELSE):
            if self.match(TokenType.IF):
                # else if
                else_block = Block([self.if_statement()])
            else:
                else_block = self.block()
        
        return IfStatement(condition, then_block, else_block)
    
    def while_statement(self) -> WhileStatement:
        """Parse while statement"""
        condition = self.expression()
        body = self.block()
        return WhileStatement(condition, body)
    
    def for_statement(self) -> ForStatement:
        """Parse for statement: for var in iterable { body }"""
        variable = self.consume(TokenType.IDENTIFIER, "Expected loop variable").value
        self.consume(TokenType.USE if self.check(TokenType.USE) else TokenType.IDENTIFIER, "Expected 'in'")
        if self.previous().type == TokenType.USE:
            # Handle 'in' keyword (if we add it)
            pass
        iterable = self.expression()
        body = self.block()
        return ForStatement(variable, iterable, body)
    
    def block(self) -> Block:
        """Parse block: { statements }"""
        # Consume the opening brace if not already consumed
        if self.check(TokenType.LEFT_BRACE):
            self.advance()
        
        statements = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            # Try declaration first (function, let)
            if self.check(TokenType.FN):
                stmt = self.declaration()
            elif self.check(TokenType.LET):
                stmt = self.let_statement()
            else:
                # Otherwise, try statement
                stmt = self.statement()
            if stmt:
                statements.append(stmt)
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after block")
        return Block(statements)
    
    def expression(self) -> Expression:
        """Parse expression"""
        return self.assignment()
    
    def assignment(self) -> Expression:
        """Parse assignment (for now, just equality)"""
        expr = self.equality()
        return expr
    
    def equality(self) -> Expression:
        """Parse equality: ==, !="""
        expr = self.comparison()
        while self.match(TokenType.EQUAL_EQUAL) or self.match(TokenType.BANG_EQUAL):
            operator = self.previous().value
            right = self.comparison()
            expr = BinaryExpression(expr, operator, right)
        return expr
    
    def comparison(self) -> Expression:
        """Parse comparison: <, >, <=, >="""
        expr = self.term()
        while self.match(TokenType.LESS) or self.match(TokenType.LESS_EQUAL) or \
              self.match(TokenType.GREATER) or self.match(TokenType.GREATER_EQUAL):
            operator = self.previous().value
            right = self.term()
            expr = BinaryExpression(expr, operator, right)
        return expr
    
    def term(self) -> Expression:
        """Parse term: +, -"""
        expr = self.factor()
        while self.match(TokenType.PLUS) or self.match(TokenType.MINUS):
            operator = self.previous().value
            right = self.factor()
            expr = BinaryExpression(expr, operator, right)
        return expr
    
    def factor(self) -> Expression:
        """Parse factor: *, /, %"""
        expr = self.unary()
        while self.match(TokenType.STAR) or self.match(TokenType.SLASH) or \
              self.match(TokenType.PERCENT) or self.match(TokenType.SHL) or \
              self.match(TokenType.SHR):
            operator = self.previous().value
            right = self.unary()
            expr = BinaryExpression(expr, operator, right)
        return expr
    
    def unary(self) -> Expression:
        """Parse unary: !, -"""
        if self.match(TokenType.BANG) or self.match(TokenType.MINUS):
            operator = self.previous().value
            right = self.unary()
            return UnaryExpression(operator, right)
        return self.call()
    
    def call(self) -> Expression:
        """Parse function call"""
        expr = self.primary()
        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finish_call(expr)
            else:
                break
        return expr
    
    def finish_call(self, callee: Expression) -> CallExpression:
        """Finish parsing function call"""
        arguments = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                arguments.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
        if isinstance(callee, Identifier):
            return CallExpression(callee.name, arguments)
        else:
            # For method calls, etc.
            return CallExpression("", arguments)
    
    def primary(self) -> Expression:
        """Parse primary expression"""
        if self.match(TokenType.TRUE):
            return Literal(True, "bool")
        if self.match(TokenType.FALSE):
            return Literal(False, "bool")
        if self.match(TokenType.NULL):
            return Literal(None, "null")
        
        if self.match(TokenType.PENTARY):
            return Literal(self.previous().value, "pent")
        if self.match(TokenType.INTEGER):
            try:
                value = int(self.previous().value)
            except ValueError:
                # Handle hex/binary
                value = self.previous().value
            return Literal(value, "int")
        if self.match(TokenType.FLOAT):
            return Literal(float(self.previous().value), "float")
        if self.match(TokenType.STRING):
            return Literal(self.previous().value, "string")
        
        if self.match(TokenType.IDENTIFIER):
            return Identifier(self.previous().value)
        
        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        
        raise self.error(self.peek(), "Expected expression")
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the types"""
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
    
    def check(self, type: TokenType) -> bool:
        """Check if current token is of type"""
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def advance(self) -> Token:
        """Advance and return current token"""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        """Check if at end of tokens"""
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        """Peek at current token"""
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        """Get previous token"""
        return self.tokens[self.current - 1]
    
    def consume(self, type: TokenType, message: str) -> Token:
        """Consume token of expected type"""
        if self.check(type):
            return self.advance()
        raise self.error(self.peek(), message)
    
    def error(self, token: Token, message: str) -> Exception:
        """Create parse error"""
        if token.type == TokenType.EOF:
            return Exception(f"Error at end: {message}")
        return Exception(f"Error at {token.value} (line {token.line}): {message}")


def main():
    """Test the parser"""
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
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    print("AST:")
    print(f"  {ast}")
    for stmt in ast.statements:
        print(f"    {stmt}")


if __name__ == "__main__":
    main()
