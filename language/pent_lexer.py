#!/usr/bin/env python3
"""
Pent Language Lexer
Tokenizes Pent source code into tokens
"""

from enum import Enum
from typing import List, Optional, Tuple
import re


class TokenType(Enum):
    # Literals
    PENTARY = "PENTARY"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    
    # Keywords
    FN = "FN"
    LET = "LET"
    MUT = "MUT"
    CONST = "CONST"
    IF = "IF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    FOR = "FOR"
    RETURN = "RETURN"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    STRUCT = "STRUCT"
    ENUM = "ENUM"
    IMPL = "IMPL"
    PUB = "PUB"
    USE = "USE"
    AS = "AS"
    MATCH = "MATCH"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NULL = "NULL"
    
    # Types
    TYPE_PENT = "TYPE_PENT"
    TYPE_INT = "TYPE_INT"
    TYPE_FLOAT = "TYPE_FLOAT"
    TYPE_BOOL = "TYPE_BOOL"
    TYPE_VOID = "TYPE_VOID"
    
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    PERCENT = "PERCENT"
    EQUAL = "EQUAL"
    EQUAL_EQUAL = "EQUAL_EQUAL"
    BANG = "BANG"
    BANG_EQUAL = "BANG_EQUAL"
    LESS = "LESS"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER = "GREATER"
    GREATER_EQUAL = "GREATER_EQUAL"
    AND = "AND"
    OR = "OR"
    SHL = "SHL"  # <<
    SHR = "SHR"  # >>
    
    # Delimiters
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    LEFT_BRACE = "LEFT_BRACE"
    RIGHT_BRACE = "RIGHT_BRACE"
    LEFT_BRACKET = "LEFT_BRACKET"
    RIGHT_BRACKET = "RIGHT_BRACKET"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    COMMA = "COMMA"
    DOT = "DOT"
    ARROW = "ARROW"  # ->
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"
    COMMENT = "COMMENT"


class Token:
    def __init__(self, type: TokenType, value: str, line: int, column: int):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Lexer:
    """Lexer for Pent programming language"""
    
    # Pentary digit symbols
    PENTARY_DIGITS = {'⊖', '-', '0', '+', '⊕'}
    
    # Keywords mapping
    KEYWORDS = {
        'fn': TokenType.FN,
        'let': TokenType.LET,
        'mut': TokenType.MUT,
        'const': TokenType.CONST,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'return': TokenType.RETURN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'struct': TokenType.STRUCT,
        'enum': TokenType.ENUM,
        'impl': TokenType.IMPL,
        'pub': TokenType.PUB,
        'use': TokenType.USE,
        'as': TokenType.AS,
        'match': TokenType.MATCH,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'null': TokenType.NULL,
        'pent': TokenType.TYPE_PENT,
        'int': TokenType.TYPE_INT,
        'float': TokenType.TYPE_FLOAT,
        'bool': TokenType.TYPE_BOOL,
        'void': TokenType.TYPE_VOID,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code"""
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens
    
    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def scan_token(self):
        """Scan a single token"""
        c = self.advance()
        
        # Single character tokens
        if c == '(':
            self.add_token(TokenType.LEFT_PAREN)
        elif c == ')':
            self.add_token(TokenType.RIGHT_PAREN)
        elif c == '{':
            self.add_token(TokenType.LEFT_BRACE)
        elif c == '}':
            self.add_token(TokenType.RIGHT_BRACE)
        elif c == '[':
            self.add_token(TokenType.LEFT_BRACKET)
        elif c == ']':
            self.add_token(TokenType.RIGHT_BRACKET)
        elif c == ';':
            self.add_token(TokenType.SEMICOLON)
        elif c == ',':
            self.add_token(TokenType.COMMA)
        elif c == '.':
            self.add_token(TokenType.DOT)
        elif c == ':':
            self.add_token(TokenType.COLON)
        
        # Operators
        elif c == '+':
            self.add_token(TokenType.PLUS)
        elif c == '-':
            if self.match('>'):
                self.add_token(TokenType.ARROW)
            else:
                self.add_token(TokenType.MINUS)
        elif c == '*':
            self.add_token(TokenType.STAR)
        elif c == '/':
            if self.match('/'):
                # Single-line comment
                while self.peek() != '\n' and not self.is_at_end():
                    self.advance()
            elif self.match('*'):
                # Multi-line comment
                self.scan_multiline_comment()
            else:
                self.add_token(TokenType.SLASH)
        elif c == '%':
            self.add_token(TokenType.PERCENT)
        elif c == '=':
            self.add_token(TokenType.EQUAL_EQUAL if self.match('=') else TokenType.EQUAL)
        elif c == '!':
            self.add_token(TokenType.BANG_EQUAL if self.match('=') else TokenType.BANG)
        elif c == '<':
            if self.match('<'):
                self.add_token(TokenType.SHL)
            else:
                self.add_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
        elif c == '>':
            if self.match('>'):
                self.add_token(TokenType.SHR)
            else:
                self.add_token(TokenType.GREATER_EQUAL if self.match('=') else TokenType.GREATER)
        elif c == '&':
            if self.match('&'):
                self.add_token(TokenType.AND)
            else:
                self.error(f"Unexpected character: {c}")
        elif c == '|':
            if self.match('|'):
                self.add_token(TokenType.OR)
            else:
                self.error(f"Unexpected character: {c}")
        
        # Whitespace
        elif c in ' \t\r':
            pass  # Ignore whitespace
        elif c == '\n':
            self.line += 1
            self.column = 1
            # Optionally add NEWLINE token
            # self.add_token(TokenType.NEWLINE)
        
        # String literals
        elif c == '"' or c == "'":
            self.scan_string(c)
        
        # Numbers
        elif c.isdigit():
            self.scan_number()
        
        # Identifiers and pentary literals
        elif c.isalpha() or c == '_' or c in self.PENTARY_DIGITS:
            self.scan_identifier_or_pentary()
        
        else:
            self.error(f"Unexpected character: {c}")
    
    def scan_multiline_comment(self):
        """Scan multi-line comment /* ... */"""
        depth = 1
        while depth > 0 and not self.is_at_end():
            if self.peek() == '/' and self.peek_next() == '*':
                self.advance()
                self.advance()
                depth += 1
            elif self.peek() == '*' and self.peek_next() == '/':
                self.advance()
                self.advance()
                depth -= 1
            else:
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 1
                self.advance()
    
    def scan_string(self, quote: str):
        """Scan string literal"""
        while self.peek() != quote and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
            self.advance()
        
        if self.is_at_end():
            self.error("Unterminated string")
            return
        
        # Closing quote
        self.advance()
        
        # Extract string value (without quotes)
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokenType.STRING, value)
    
    def scan_number(self):
        """Scan number literal (integer or float)"""
        while self.peek().isdigit():
            self.advance()
        
        # Check for hex or binary
        if self.peek() == 'x' and self.source[self.start] == '0':
            self.advance()  # consume 'x'
            while self.peek().isdigit() or self.peek().lower() in 'abcdef':
                self.advance()
            value = self.source[self.start:self.current]
            self.add_token(TokenType.INTEGER, value)
            return
        elif self.peek() == 'b' and self.source[self.start] == '0':
            self.advance()  # consume 'b'
            while self.peek() in '01':
                self.advance()
            value = self.source[self.start:self.current]
            self.add_token(TokenType.INTEGER, value)
            return
        
        # Check for float
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()  # consume '.'
            while self.peek().isdigit():
                self.advance()
            
            # Check for exponent
            if self.peek().lower() == 'e':
                self.advance()
                if self.peek() in '+-':
                    self.advance()
                while self.peek().isdigit():
                    self.advance()
        
        value = self.source[self.start:self.current]
        if '.' in value or 'e' in value.lower():
            self.add_token(TokenType.FLOAT, value)
        else:
            self.add_token(TokenType.INTEGER, value)
    
    def scan_identifier_or_pentary(self):
        """Scan identifier or pentary literal"""
        # Check if it starts with a pentary digit
        if self.source[self.start] in self.PENTARY_DIGITS:
            # Could be pentary literal or identifier starting with pentary digit
            # For now, scan as pentary if all characters are pentary digits
            while self.peek() in self.PENTARY_DIGITS:
                self.advance()
            
            value = self.source[self.start:self.current]
            # Check if it's a valid pentary literal
            if all(c in self.PENTARY_DIGITS for c in value):
                self.add_token(TokenType.PENTARY, value)
            else:
                # Treat as identifier
                self.add_token(TokenType.IDENTIFIER, value)
        else:
            # Regular identifier
            while self.peek().isalnum() or self.peek() == '_':
                self.advance()
            
            value = self.source[self.start:self.current]
            token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
            self.add_token(token_type, value)
    
    def advance(self) -> str:
        """Advance and return current character"""
        if self.is_at_end():
            return '\0'
        c = self.source[self.current]
        self.current += 1
        self.column += 1
        return c
    
    def peek(self) -> str:
        """Peek at current character without advancing"""
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self) -> str:
        """Peek at next character"""
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def match(self, expected: str) -> bool:
        """Match and consume if expected character"""
        if self.is_at_end():
            return False
        if self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True
    
    def add_token(self, token_type: TokenType, value: str = None):
        """Add a token to the list"""
        if value is None:
            value = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, value, self.line, self.column - len(value)))
    
    def error(self, message: str):
        """Report an error"""
        print(f"Lexer error at line {self.line}, column {self.column}: {message}")


def main():
    """Test the lexer"""
    source = """
    fn add(a: pent, b: pent) -> pent {
        return a + b;
    }
    
    fn main() {
        let x = ⊕+;
        let y = +⊕;
        let z = add(x, y);
        println(z);
    }
    """
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    print("Tokens:")
    for token in tokens:
        print(f"  {token}")


if __name__ == "__main__":
    main()
