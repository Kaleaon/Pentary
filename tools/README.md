# Pentary Tools Guide

This directory contains all software tools for working with pentary computing systems.

## Core Tools

### pentary_converter.py
**Purpose**: Convert between decimal, binary, and balanced pentary number systems.

**Usage**:
```bash
python tools/pentary_converter.py
```

**Features**:
- Decimal ↔ Pentary conversion
- Binary ↔ Pentary conversion
- Arithmetic operations (add, subtract, multiply by constants)
- Shift operations (left/right)
- Negation

**Example**:
```python
from pentary_converter import PentaryConverter

# Convert decimal to pentary
pentary = PentaryConverter.decimal_to_pentary(42)
print(pentary)  # Output: "⊕+0"

# Convert back
decimal = PentaryConverter.pentary_to_decimal(pentary)
print(decimal)  # Output: 42
```

### pentary_arithmetic.py
**Purpose**: Low-level pentary arithmetic operations with detailed tracing.

**Usage**:
```bash
python tools/pentary_arithmetic.py
```

**Features**:
- Digit-level addition with carry propagation
- Multiplication by constants {-2, -1, 0, 1, 2}
- Step-by-step operation traces
- Comparison operations

**Example**:
```python
from pentary_arithmetic import PentaryArithmetic

# Add two pentary numbers with detailed trace
result, steps = PentaryArithmetic.add_pentary_detailed("⊕+", "+-")
print(f"Result: {result}")
for step in steps:
    print(f"Position {step['position']}: {step['a']} + {step['b']} = {step['sum']}")
```

### pentary_simulator.py
**Purpose**: Simulate pentary processor execution.

**Usage**:
```bash
python tools/pentary_simulator.py
```

**Features**:
- Full ISA implementation
- 32 registers + memory
- Instruction execution
- Debugging support

**Example**:
```python
from pentary_simulator import PentaryProcessor

# Create processor
proc = PentaryProcessor()

# Load program
program = [
    "MOVI P1, 5",
    "MOVI P2, 3",
    "ADD P3, P1, P2",
    "HALT"
]

proc.load_program(program)
proc.run(verbose=True)
proc.print_state()
```

## Neural Network Tools

### pentary_nn.py
**Purpose**: Implement neural network layers optimized for pentary arithmetic.

**Usage**:
```bash
python tools/pentary_nn.py
```

**Features**:
- PentaryLinear: Fully connected layer with pentary weights
- PentaryConv2D: Convolution layer with pentary weights
- PentaryReLU: ReLU activation
- PentaryQuantizedReLU: 5-level quantized ReLU
- PentaryMaxPool2D: Max pooling
- Complete network support

**Example**:
```python
from pentary_nn import PentaryNetwork, PentaryLinear, PentaryReLU

# Create a simple classifier
network = PentaryNetwork([
    PentaryLinear(784, 128, use_pentary=True),
    PentaryReLU(),
    PentaryLinear(128, 10, use_pentary=True)
])

# Forward pass
import numpy as np
x = np.random.randn(32, 784)
output = network.forward(x)

# Get parameter statistics
params = network.count_parameters()
print(f"Sparsity: {params['sparsity']:.1%}")
```

### pentary_quantizer.py
**Purpose**: Quantize neural network models to pentary representation.

**Usage**:
```bash
python tools/pentary_quantizer.py
```

**Features**:
- Post-training quantization
- Quantization-aware training support
- Calibration with representative data
- Accuracy analysis
- Model saving/loading

**Example**:
```python
from pentary_quantizer import PentaryQuantizer
import numpy as np

# Create quantizer
quantizer = PentaryQuantizer(calibration_method='minmax')

# Model weights (example)
model_weights = {
    'layer1.weight': np.random.randn(128, 784) * 0.1,
    'layer1.bias': np.random.randn(128) * 0.01,
}

# Quantize model
quantized_model = quantizer.quantize_model(model_weights)

# Analyze quantization error
original = model_weights['layer1.weight']
quantized = quantized_model['weights']['layer1.weight']
scale = quantized_model['scales']['layer1.weight']
zero_point = quantized_model['zero_points']['layer1.weight']

error_stats = quantizer.analyze_quantization_error(
    original, quantized, scale, zero_point
)
print(f"MSE: {error_stats['mse']:.6f}")
print(f"SNR: {error_stats['snr_db']:.2f} dB")
```

### pentary_iteration.py
**Purpose**: Profile, optimize, and benchmark pentary networks for rapid iteration.

**Usage**:
```bash
python tools/pentary_iteration.py
```

**Features**:
- Profiling: Execution time, memory usage, energy estimates
- Optimization: Weight pruning, quantization
- Benchmarking: Throughput, latency analysis
- Rapid iteration: Automated optimization loop

**Example**:
```python
from pentary_iteration import PentaryProfiler, PentaryOptimizer, PentaryBenchmark
from pentary_nn import create_simple_classifier
import numpy as np

# Create network
network = create_simple_classifier(784, 128, 10)

# Profile
profiler = PentaryProfiler()
x = np.random.randn(32, 784)
profiler.profile_network(network, x)
profiler.print_summary()

# Optimize
optimizer = PentaryOptimizer()
optimized_network = optimizer.optimize_for_inference(network)

# Benchmark
benchmark = PentaryBenchmark()
result = benchmark.benchmark_network(optimized_network, (32, 784))
print(f"Mean latency: {result['mean_time_ms']:.2f} ms")
print(f"Throughput: {result['throughput_fps']:.2f} fps")
```

## Advanced Usage

### Rapid Iteration Workflow

```python
from pentary_iteration import PentaryRapidIteration
from pentary_nn import create_simple_classifier

# Create network
network = create_simple_classifier(784, 128, 10)

# Rapid iteration: profile, optimize, benchmark
rapid_iter = PentaryRapidIteration()
optimized_network = rapid_iter.iterate(
    network,
    input_shape=(32, 784),
    target_latency_ms=10.0,
    target_sparsity=0.7
)
```

### Complete Training Pipeline

```python
from pentary_nn import PentaryNetwork, PentaryLinear, PentaryReLU
from pentary_quantizer import PentaryQuantizer
from pentary_iteration import PentaryProfiler
import numpy as np

# 1. Create network
network = PentaryNetwork([
    PentaryLinear(784, 128, use_pentary=True),
    PentaryReLU(),
    PentaryLinear(128, 10, use_pentary=True)
])

# 2. Train (simplified - use your training loop)
# ... training code ...

# 3. Quantize
quantizer = PentaryQuantizer()
params = network.get_parameters()
quantized_params = {}
for key, value in params.items():
    if 'weights' in value:
        quantized, scale, zp = quantizer.quantize_tensor(value['weights'])
        quantized_params[key] = {'weights': quantized, 'scale': scale, 'zero_point': zp}
network.set_parameters(quantized_params)

# 4. Profile and optimize
profiler = PentaryProfiler()
x = np.random.randn(32, 784)
profiler.profile_network(network, x)
profiler.print_summary()
```

## Tool Dependencies

All tools require:
- Python 3.7+
- NumPy

Install dependencies:
```bash
pip install numpy
```

## Performance Tips

1. **Use Pentary Weights**: Set `use_pentary=True` in layers for maximum efficiency
2. **Exploit Sparsity**: Prune weights to 70-80% sparsity for power savings
3. **Profile First**: Always profile before optimizing
4. **Batch Processing**: Process multiple samples in batches for better throughput
5. **Hardware Mapping**: Map large matrix operations to memristor crossbars when available

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd /path/to/pentary
python tools/pentary_nn.py
```

### Performance Issues
- Use smaller batch sizes if memory is limited
- Profile to identify bottlenecks
- Consider quantization to reduce memory usage

### Accuracy Issues
- Use quantization-aware training instead of post-training quantization
- Try mixed precision (float for first/last layers, pentary for middle)
- Increase model capacity if needed

## Contributing

When adding new tools:
1. Follow the existing code style
2. Add comprehensive docstrings
3. Include example usage in `main()` function
4. Update this README

## License

See LICENSE file in project root.

---

**Last Updated**: 2025  
**Status**: Tools are functional and tested
