# Pent Programming Language Specification

## Overview

**Pent** is a high-level programming language designed specifically for the Pentary Processor architecture. It provides native support for balanced pentary arithmetic, neural network operations, and efficient code generation for the pentary ISA.

## Language Design Principles

1. **Pentary-First**: Native support for pentary literals and operations
2. **Neural Network Optimized**: Built-in types and operations for AI workloads
3. **Type Safety**: Strong typing with pentary-aware type system
4. **Performance**: Compiles to efficient pentary assembly
5. **Readability**: Clean, modern syntax inspired by Rust and Python

## Lexical Structure

### Comments

```pent
// Single-line comment
/* Multi-line
   comment */
```

### Identifiers

- Start with letter or underscore
- Followed by letters, digits, underscores
- Case-sensitive
- Examples: `x`, `myVar`, `neural_net`, `_temp`

### Keywords

```
fn, let, mut, const, if, else, while, for, return, break, continue
struct, enum, impl, pub, use, as, match
pent, int, float, bool, void
true, false, null
```

### Literals

#### Pentary Literals

```pent
// Pentary digit symbols: ⊖ (-2), - (-1), 0, + (+1), ⊕ (+2)
let x = ⊖⊕+;        // -2*25 + 2*5 + 1 = -39
let y = ⊕⊕0;        // 2*25 + 2*5 + 0 = 60
let z = 0;          // zero
let neg = -⊕+;      // negative: -(+2*5 + 1) = -11
```

#### Integer Literals

```pent
let x = 42;         // Decimal
let y = 0x2A;       // Hexadecimal
let z = 0b101010;   // Binary
```

#### Float Literals

```pent
let x = 3.14;
let y = 1.0e-5;
```

#### Boolean Literals

```pent
let x = true;
let y = false;
```

#### String Literals

```pent
let msg = "Hello, Pentary!";
let path = 'single quotes also work';
```

## Types

### Primitive Types

```pent
pent      // Balanced pentary number (16 pents = ~37 bits)
int       // Signed integer (32 bits)
float     // Floating point (32 bits)
bool      // Boolean
void      // No value
```

### Composite Types

```pent
// Arrays
let arr: [pent; 10] = [0; 10];  // Array of 10 pents, initialized to 0

// Tuples
let pair: (pent, int) = (⊕+, 42);

// Structs
struct Point {
    x: pent,
    y: pent,
}

// Enums
enum Result {
    Ok(pent),
    Error(string),
}
```

## Variables and Constants

```pent
// Immutable variable
let x = 42;

// Mutable variable
let mut y = 10;
y = 20;

// Constant (compile-time)
const MAX_SIZE: int = 100;
```

## Expressions

### Arithmetic Operations

```pent
let a = ⊕+;         // 11
let b = +⊕;         // 7
let c = a + b;      // 18
let d = a - b;      // 4
let e = a * 2;      // Multiply by constant (efficient!)
let f = a << 1;     // Shift left (multiply by 5)
let g = a >> 1;     // Shift right (divide by 5)
let h = -a;         // Negation
```

### Comparison Operations

```pent
a == b    // Equal
a != b    // Not equal
a < b     // Less than
a > b     // Greater than
a <= b    // Less than or equal
a >= b    // Greater than or equal
```

### Logical Operations

```pent
x && y    // Logical AND
x || y    // Logical OR
!x        // Logical NOT
```

## Control Flow

### If Statements

```pent
if x > 0 {
    println("positive");
} else if x < 0 {
    println("negative");
} else {
    println("zero");
}
```

### While Loops

```pent
let mut i = 0;
while i < 10 {
    println(i);
    i = i + 1;
}
```

### For Loops

```pent
for i in 0..10 {
    println(i);
}

// Iterate over array
for item in arr {
    process(item);
}
```

### Match Expressions

```pent
match x {
    ⊖ => println("strong negative"),
    - => println("weak negative"),
    0 => println("zero"),
    + => println("weak positive"),
    ⊕ => println("strong positive"),
    _ => println("other"),
}
```

## Functions

```pent
// Function definition
fn add(a: pent, b: pent) -> pent {
    return a + b;
}

// Expression body (no return needed)
fn multiply(x: pent, y: pent) -> pent {
    x * y
}

// Void function
fn print_sum(a: pent, b: pent) {
    println(a + b);
}

// Recursive function
fn factorial(n: int) -> int {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
```

## Neural Network Support

### Matrix Types

```pent
// Matrix type (for neural networks)
let weights: Matrix<pent> = Matrix::new(256, 256);
let input: Vector<pent> = Vector::new(256);
```

### Neural Network Operations

```pent
// Matrix-vector multiplication (in-memory compute)
let output = matvec(weights, input);

// ReLU activation
let activated = relu(output);

// Quantization to 5 levels
let quantized = quantize(activated);

// Convolution
let result = conv(input, kernel, stride, padding);
```

### Built-in Neural Network Functions

```pent
fn matvec(matrix: Matrix<pent>, vector: Vector<pent>) -> Vector<pent>
fn relu(x: pent) -> pent
fn quantize(x: pent) -> pent
fn conv(input: Tensor<pent>, kernel: Tensor<pent>, ...) -> Tensor<pent>
fn pool(input: Tensor<pent>, size: int) -> Tensor<pent>
```

## Memory Management

```pent
// Stack allocation (automatic)
let x = 42;

// Heap allocation (explicit)
let ptr = alloc(100);  // Allocate 100 words
free(ptr);

// Memory operations
let value = load(ptr, offset);
store(ptr, offset, value);
```

## Modules and Imports

```pent
// Import standard library
use std::io;
use std::math;

// Import from file
use mymodule;

// Module definition
pub mod neural {
    pub fn inference(input: Vector<pent>) -> Vector<pent> {
        // ...
    }
}
```

## Example Programs

### Hello World

```pent
fn main() {
    println("Hello, Pentary World!");
}
```

### Fibonacci

```pent
fn fibonacci(n: int) -> int {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() {
    for i in 0..10 {
        println(fibonacci(i));
    }
}
```

### Neural Network Inference

```pent
use std::neural;

struct NeuralNet {
    weights: Matrix<pent>,
    bias: Vector<pent>,
}

impl NeuralNet {
    fn forward(&self, input: Vector<pent>) -> Vector<pent> {
        let output = matvec(self.weights, input);
        let biased = output + self.bias;
        let activated = relu(biased);
        quantize(activated)
    }
}

fn main() {
    let net = NeuralNet {
        weights: load_weights("model.pent"),
        bias: load_bias("bias.pent"),
    };
    
    let input = Vector::from([⊕+, +⊕, ⊕0, ...]);
    let output = net.forward(input);
    println(output);
}
```

### Matrix Operations

```pent
fn matrix_multiply(a: Matrix<pent>, b: Matrix<pent>) -> Matrix<pent> {
    let rows = a.rows();
    let cols = b.cols();
    let mut result = Matrix::new(rows, cols);
    
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0;
            for k in 0..a.cols() {
                sum = sum + a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    result
}
```

## Compiler Targets

The Pent compiler generates pentary assembly code that can be executed on:
1. Pentary Processor Simulator (for development)
2. FPGA prototypes
3. ASIC implementations

## Standard Library

### Core Types

- `Vector<T>`: Dynamic array
- `Matrix<T>`: 2D matrix
- `Tensor<T>`: Multi-dimensional tensor
- `String`: String type
- `Option<T>`: Optional value
- `Result<T, E>`: Result type

### I/O

```pent
println(x);           // Print with newline
print(x);            // Print without newline
let input = read();   // Read input
```

### Math

```pent
abs(x)                // Absolute value
min(a, b)             // Minimum
max(a, b)             // Maximum
clamp(x, min, max)    // Clamp value
```

### Neural Network

```pent
matvec()              // Matrix-vector multiply
relu()                // ReLU activation
quantize()            // Quantization
conv()                // Convolution
pool()                // Pooling
```

## Syntax Summary

```
Program     ::= (Module | Function | Struct | Enum)*
Module      ::= "mod" IDENT "{" (Item)* "}"
Function    ::= "fn" IDENT "(" Params ")" ("->" Type)? Block
Struct      ::= "struct" IDENT "{" (Field)* "}"
Enum        ::= "enum" IDENT "{" (Variant)* "}"
Block       ::= "{" (Statement)* "}"
Statement   ::= LetStmt | ExprStmt | ReturnStmt | IfStmt | WhileStmt | ForStmt
Expr        ::= BinaryExpr | UnaryExpr | CallExpr | Literal | Ident | ...
Type        ::= "pent" | "int" | "float" | "bool" | "void" | ArrayType | ...
```

## Implementation Notes

1. **Lexer**: Tokenizes source code into tokens
2. **Parser**: Builds Abstract Syntax Tree (AST)
3. **Type Checker**: Validates types and resolves symbols
4. **Code Generator**: Emits pentary assembly code
5. **Optimizer**: Performs optimizations (constant folding, dead code elimination)

## Future Extensions

- Generic types
- Traits/interfaces
- Pattern matching
- Async/await for parallel operations
- Foreign function interface (FFI)
- Package manager

---

**Version**: 1.0  
**Status**: Specification Complete  
**Last Updated**: 2025
