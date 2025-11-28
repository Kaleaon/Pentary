# Pent Programming Language

**Pent** is a high-level programming language designed specifically for the Pentary Processor architecture. It provides native support for balanced pentary arithmetic, neural network operations, and efficient code generation.

## Features

- **Native Pentary Support**: Direct use of pentary literals (⊖, -, 0, +, ⊕)
- **Type Safety**: Strong typing with pentary-aware type system
- **Neural Network Optimized**: Built-in types and operations for AI workloads
- **Modern Syntax**: Clean, readable syntax inspired by Rust and Python
- **Efficient Compilation**: Generates optimized pentary assembly code

## Quick Start

### Installation

The Pent compiler is written in Python and requires:
- Python 3.8+
- Pentary tools (from `tools/` directory)

```bash
# Make sure you're in the workspace root
cd /workspace
python language/pent_compiler.py
```

### Hello World

Create a file `hello.pent`:

```pent
fn main() {
    println("Hello, Pentary World!");
}
```

Compile it:

```bash
python language/pent_compiler.py hello.pent
```

### Example Programs

See the `examples/` directory for:
- `hello_world.pent` - Basic hello world
- `fibonacci.pent` - Recursive Fibonacci
- `arithmetic.pent` - Pentary arithmetic examples
- `neural_network.pent` - Neural network inference

## Language Syntax

### Pentary Literals

```pent
let x = ⊕+;        // 2*5 + 1 = 11
let y = +⊕;        // 1*5 + 2 = 7
let z = ⊖⊕0;       // -2*25 + 2*5 + 0 = -40
```

### Variables

```pent
// Immutable
let x = 42;

// Mutable
let mut y = 10;
y = 20;
```

### Functions

```pent
fn add(a: pent, b: pent) -> pent {
    return a + b;
}

// Expression body
fn multiply(x: pent, y: pent) -> pent {
    x * y
}
```

### Control Flow

```pent
// If statement
if x > 0 {
    println("positive");
} else {
    println("negative or zero");
}

// While loop
let mut i = 0;
while i < 10 {
    println(i);
    i = i + 1;
}

// For loop
for i in 0..10 {
    println(i);
}
```

### Neural Network Operations

```pent
// Matrix-vector multiplication
let output = matvec(weights, input);

// ReLU activation
let activated = relu(output);

// Quantization
let quantized = quantize(activated);
```

## Compiler Architecture

The Pent compiler consists of:

1. **Lexer** (`pent_lexer.py`): Tokenizes source code
2. **Parser** (`pent_parser.py`): Builds Abstract Syntax Tree (AST)
3. **Code Generator** (`pent_compiler.py`): Generates pentary assembly

### Compilation Pipeline

```
Source Code → Lexer → Tokens → Parser → AST → Code Generator → Assembly
```

## Type System

### Primitive Types

- `pent`: Balanced pentary number (16 pents ≈ 37 bits)
- `int`: Signed integer (32 bits)
- `float`: Floating point (32 bits)
- `bool`: Boolean
- `void`: No value

### Composite Types

```pent
// Arrays
let arr: [pent; 10] = [0; 10];

// Tuples
let pair: (pent, int) = (⊕+, 42);

// Structs
struct Point {
    x: pent,
    y: pent,
}
```

## Standard Library

### I/O

```pent
println(x);         // Print with newline
print(x);          // Print without newline
let input = read(); // Read input
```

### Math

```pent
abs(x)              // Absolute value
min(a, b)           // Minimum
max(a, b)           // Maximum
clamp(x, min, max)  // Clamp value
```

### Neural Network

```pent
matvec()            // Matrix-vector multiply
relu()              // ReLU activation
quantize()          // Quantization
conv()              // Convolution
pool()              // Pooling
```

## Examples

### Fibonacci

```pent
fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

fn main() {
    for i in 0..10 {
        println(fibonacci(i));
    }
}
```

### Arithmetic

```pent
fn main() {
    let a = ⊕+;      // 11
    let b = +⊕;      // 7
    let sum = a + b; // 18
    let diff = a - b; // 4
    let doubled = a * 2; // 22
    println(sum);
}
```

## Status

**Current Version**: 1.0  
**Status**: Specification Complete, Compiler in Development

### Implemented

- ✅ Language specification
- ✅ Lexer (tokenizer)
- ✅ Parser (AST generation)
- ✅ Basic code generator
- ✅ Example programs

### In Progress

- ⏳ Full code generator
- ⏳ Type checker
- ⏳ Optimizer
- ⏳ Standard library

### Planned

- 🔜 Generic types
- 🔜 Traits/interfaces
- 🔜 Pattern matching
- 🔜 Package manager

## Documentation

- [Language Specification](pent_language_spec.md) - Complete language reference
- [Examples](examples/) - Example programs
- [Architecture](../architecture/) - Processor architecture docs

## Contributing

Contributions are welcome! Areas of interest:

1. **Compiler Improvements**: Better code generation, optimizations
2. **Standard Library**: More built-in functions
3. **Language Features**: New syntax, type system improvements
4. **Documentation**: Tutorials, guides, examples
5. **Tools**: Debugger, profiler, IDE support

## License

This project is part of the Pentary project and follows the same license.

---

**The future is not Binary. It is Balanced.**
