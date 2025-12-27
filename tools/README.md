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
- **Automatic Assembly**: Automatically assembles code with labels

**Example**:
```python
from pentary_simulator import PentaryProcessor

# Create processor
proc = PentaryProcessor()

# Load program (can use labels!)
program = [
    "start:",
    "MOVI P1, 5",
    "MOVI P2, 3",
    "ADD P3, P1, P2",
    "JUMP end",
    "MOVI P3, 0",  # skipped
    "end:",
    "HALT"
]

proc.load_program(program)
proc.run(verbose=True)
proc.print_state()
```

### pentary_assembler.py
**Purpose**: Assemble pentary assembly code (with labels) into machine instructions.

**Usage**:
```bash
python tools/pentary_assembler.py <input_file> [output_file]
```

**Features**:
- Label resolution
- Comment stripping
- Syntax checking
- 2-pass assembly process

**Example**:
```bash
python tools/pentary_assembler.py my_program.pentasm output.asm
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

## Advanced Neural Architectures

### pentary_mamba.py
**Purpose**: Selective State Space Model (Mamba) for O(n) sequence modeling.

**Based on**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

**Features**:
- PentarySSMCore: Selective state space with input-dependent parameters
- PentaryMambaBlock: Complete block with gating and convolution
- PentaryMamba: Full model with generation

**Example**:
```python
from pentary_mamba import PentaryMamba

model = PentaryMamba(vocab_size=10000, d_model=256, n_layers=4)
logits = model.forward(input_ids)
generated = model.generate(prompt_ids, max_new_tokens=50)
```

### pentary_rwkv.py
**Purpose**: Linear Attention RNN combining Transformer training with RNN inference.

**Based on**: [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

**Features**:
- PentaryTimeMix: Core WKV attention mechanism
- PentaryChannelMix: Gated feed-forward network
- O(1) inference memory per token

**Example**:
```python
from pentary_rwkv import PentaryRWKV

model = PentaryRWKV(vocab_size=10000, d_model=256, n_layers=6)

# Parallel (training)
logits = model.forward(input_ids)

# Recurrent O(1) (inference)
logits, states = model.forward_recurrent(token_id, states)
```

### pentary_retnet.py
**Purpose**: Retentive Network replacing attention with explicit decay.

**Based on**: [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

**Features**:
- PentaryRetention: Core retention with decay matrix
- Three modes: Parallel, Recurrent, Chunkwise
- Multi-head retention

**Example**:
```python
from pentary_retnet import PentaryRetNet

model = PentaryRetNet(vocab_size=10000, d_model=256, n_layers=6, gamma=0.95)

# Parallel (training)
logits = model.forward(input_ids)

# Recurrent O(1) (inference)
logits, states = model.forward_recurrent(token, states, position)
```

### pentary_world_model.py
**Purpose**: World Model for model-based reinforcement learning.

**Based on**: [World Models](https://arxiv.org/abs/1803.10122), [DreamerV3](https://arxiv.org/abs/2301.04104)

**Features**:
- PentaryEncoder/Decoder: Visual processing
- PentaryRSSM: Recurrent State Space Model with 5-category stochastic states
- 5 categories = perfect pentary alignment!

**Example**:
```python
from pentary_world_model import PentaryWorldModel

model = PentaryWorldModel(obs_shape=(64, 64, 3), action_dim=4)

# Encode observation
z = model.encode(observation)

# Imagine future (planning)
imagined = model.imagine(h, stoch_z, actions, horizon=15)
```

### pentary_pope.py
**Purpose**: Polar Coordinate Position Embeddings (PoPE) for Transformers.

**Based on**: [Decoupling the 'What' and 'Where' With Polar Coordinate Positional Embeddings](https://arxiv.org/abs/2509.10534)

**Features**:
- Magnitude for content, angle for position
- Pentary-quantized angular lookup tables
- Better length extrapolation

---

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

**Last Updated**: December 2025  
**Status**: Tools are functional and tested

## New in December 2025
- Added PentaryMamba (Selective State Space Model)
- Added PentaryRWKV (Linear Attention RNN)
- Added PentaryRetNet (Retentive Network)
- Added PentaryWorldModel (Latent Dynamics)
- Added PentaryPoPE (Polar Position Embeddings)
- Comprehensive test suite: `tests/test_architectures_quick.py`
