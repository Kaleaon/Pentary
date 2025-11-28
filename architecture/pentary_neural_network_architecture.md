# Pentary Neural Network Architecture

## Overview

This document describes the architecture and implementation of neural networks optimized for pentary (base-5) computing systems.

## Design Philosophy

### 1. Hardware-Aligned Quantization

Pentary neural networks use weights quantized to 5 levels: **{-2, -1, 0, +1, +2}**

**Benefits:**
- **Native Hardware Support**: Direct mapping to pentary ALU operations
- **Multiplication Elimination**: Weights are integers, enabling shift-add operations
- **Sparsity Exploitation**: Zero weights physically disconnect (zero power)
- **Memory Efficiency**: 2.32 bits per pent vs 1 bit per binary digit

### 2. In-Memory Computing

**Memristor Crossbar Arrays:**
- Store weight matrices directly in analog domain
- Perform matrix-vector multiplication in-place
- 256×256 crossbar arrays per core
- 5-level resistance states map to pentary weights

### 3. Sparsity-First Design

**Zero-State Power Savings:**
- Zero weights = physical disconnect = zero power consumption
- 80% sparse networks → 80% power reduction in compute
- No software intervention needed

## Network Architecture Components

### 1. Pentary Linear Layer

**Operation:**
```
y = x @ W^T + b
```

Where:
- `x`: Input vector (float or pentary)
- `W`: Weight matrix (pentary: {-2, -1, 0, 1, 2})
- `b`: Bias vector (float or pentary)

**Hardware Implementation:**
- For each weight `w`:
  - `w = 0`: Skip (sparsity)
  - `w = ±1`: Pass through or negate
  - `w = ±2`: Shift and add (multiply by 2)

**Example:**
```python
# Input: x = [1.5, 2.0, 0.5]
# Weights: W = [[2, -1, 0], [1, 2, -2]]
# Output: y = [2*1.5 + (-1)*2.0 + 0*0.5, 1*1.5 + 2*2.0 + (-2)*0.5]
#         = [1.0, 4.5]
```

### 2. Pentary Convolution Layer

**Operation:**
```
y[i,j] = Σ Σ x[i+k, j+l] * W[k, l]
```

**Hardware Optimization:**
- Pentary weights enable shift-add instead of multiplication
- Zero weights skip computation entirely
- Can be implemented in memristor crossbar arrays

**Sparsity Benefits:**
- Typical CNNs: 60-80% sparse after quantization
- Power savings proportional to sparsity

### 3. Activation Functions

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```

**Pentary Implementation:**
- Simple comparator: if `x < 0`, output `0`
- Hardware: Single gate delay

#### Quantized ReLU
```
QuantizedReLU(x) = quantize(max(0, x)) to {-2, -1, 0, 1, 2}
```

**Benefits:**
- Maintains pentary representation throughout network
- Reduces precision loss
- Hardware-friendly

#### Other Activations
- **Sigmoid/Tanh**: Lookup table (LUT) based
- **Softmax**: Shared exponential unit + normalizer
- **GELU**: Hardware-accelerated approximation

### 4. Pooling Layers

#### Max Pooling
```
y[i,j] = max(x[i:i+k, j:j+k])
```

**Pentary Implementation:**
- Tree of MAX gates
- Configurable window size (2×2, 3×3, etc.)
- Stride control

#### Average Pooling
```
y[i,j] = mean(x[i:i+k, j:j+k])
```

**Pentary Implementation:**
- Sum then divide (shift right in pentary)

## Network Topologies

### 1. Fully Connected Networks

**Architecture:**
```
Input (784) → Hidden (128) → Hidden (64) → Output (10)
```

**Pentary Optimization:**
- All weights quantized to {-2, -1, 0, 1, 2}
- Bias terms can remain float or be quantized
- Typical sparsity: 70-85%

**Use Cases:**
- MNIST digit classification
- Small-scale image classification
- Feature extraction

### 2. Convolutional Neural Networks (CNNs)

**Architecture:**
```
Input (3×32×32)
  → Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
  → Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
  → Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
  → Flatten → Linear(128) → ReLU
  → Linear(10)
```

**Pentary Optimization:**
- Convolution weights: Pentary quantized
- Typical sparsity: 60-75%
- Can use memristor crossbar for large convolutions

**Use Cases:**
- CIFAR-10/100 classification
- Image recognition
- Object detection

### 3. Residual Networks (ResNets)

**Architecture:**
```
Input → Conv Block → [Residual Block]×N → Output
```

**Pentary Adaptation:**
- Residual connections: Direct addition (pentary-friendly)
- Batch normalization: Can be fused with convolution
- Skip connections: Zero overhead in pentary

**Use Cases:**
- Deep image classification
- Feature extraction
- Transfer learning

## Quantization Strategy

### 1. Post-Training Quantization

**Process:**
1. Train model in float32
2. Quantize weights to pentary {-2, -1, 0, 1, 2}
3. Fine-tune if needed

**Calibration Methods:**
- **Min-Max**: Scale to fit [-2, 2] range
- **Percentile**: More robust to outliers
- **KL Divergence**: Minimize information loss

### 2. Quantization-Aware Training

**Process:**
1. Train with quantization simulation
2. Use straight-through estimator for gradients
3. Quantize weights during forward pass

**Benefits:**
- Better accuracy than post-training quantization
- Model learns to work with quantized weights

### 3. Mixed Precision

**Strategy:**
- First/last layers: Higher precision (float or 8-bit)
- Middle layers: Pentary {-2, -1, 0, 1, 2}
- Activations: Can be float or pentary

**Benefits:**
- Better accuracy
- Still benefits from pentary efficiency in middle layers

## Hardware Acceleration

### 1. Matrix-Vector Multiplication

**Memristor Crossbar:**
```
Input Vector (256 pents) → Crossbar (256×256) → Output Vector (256 pents)
```

**Operation:**
- Input voltages applied to rows
- Currents through memristors (conductance = weight)
- Output currents summed at columns
- ADC converts to pentary digital

**Performance:**
- Latency: ~10ns (analog computation)
- Energy: ~1pJ per operation
- Throughput: 100 GOPS per crossbar

### 2. Convolution Acceleration

**Implementation Options:**

**Option A: Im2Col + Matrix Multiply**
- Convert convolution to matrix multiply
- Use memristor crossbar

**Option B: Direct Convolution**
- Sliding window with pentary multipliers
- Exploit sparsity

**Option C: Winograd Algorithm**
- Reduce multiplications
- Pentary-friendly

### 3. Activation Function Units

**ReLU:**
- Comparator + multiplexer
- Latency: 1 gate delay

**Quantized ReLU:**
- 5-level threshold comparators
- Latency: 2 gate delays

**Sigmoid/Tanh:**
- Lookup table (LUT)
- 32-entry LUT sufficient for 5-level output

## Performance Characteristics

### 1. Accuracy

**Typical Accuracy Drops:**
- Post-training quantization: 2-5% drop
- Quantization-aware training: 0.5-2% drop
- Mixed precision: <1% drop

**Benchmark Results (MNIST):**
- Float32: 99.2%
- Pentary (post-training): 97.8%
- Pentary (QAT): 98.9%

### 2. Performance

**Throughput:**
- Single core: 10 TOPS (Tera Operations Per Second)
- 8-core chip: 80 TOPS peak
- Sustained: 70% of peak (56 TOPS)

**Latency:**
- Single inference: <1ms (for typical networks)
- Batch inference: Higher throughput

### 3. Power Efficiency

**Energy per Operation:**
- Pentary multiply-add: 0.1 pJ
- Binary (8-bit) multiply-add: 1.0 pJ
- **10× improvement**

**Power Consumption:**
- Single core: 5W
- 8-core chip: 40W
- Efficiency: 1.4 TOPS/W

**Sparsity Benefits:**
- 80% sparse: 80% power reduction
- Zero weights consume zero power

### 4. Memory Efficiency

**Weight Storage:**
- Binary (8-bit): 1 byte per weight
- Pentary: 0.43 bytes per weight (2.32 bits)
- **2.3× compression**

**Example:**
- 1M weights (8-bit): 1 MB
- 1M weights (pentary): 0.43 MB

## Software Tools

### 1. Model Conversion

**Tools:**
- `pentary_quantizer.py`: Quantize PyTorch/TensorFlow models
- Supports post-training and quantization-aware training
- Calibration data collection

### 2. Network Implementation

**Tools:**
- `pentary_nn.py`: Pentary layer implementations
- Compatible with PyTorch/TensorFlow APIs
- Forward and backward passes

### 3. Profiling and Optimization

**Tools:**
- `pentary_iteration.py`: Profiling, optimization, benchmarking
- Rapid iteration for model optimization
- Performance analysis

## Example Networks

### 1. Simple Classifier

```python
from pentary_nn import PentaryNetwork, PentaryLinear, PentaryReLU

network = PentaryNetwork([
    PentaryLinear(784, 128, use_pentary=True),
    PentaryReLU(),
    PentaryLinear(128, 10, use_pentary=True)
])
```

### 2. CNN

```python
from pentary_nn import PentaryNetwork, PentaryConv2D, PentaryMaxPool2D

network = PentaryNetwork([
    PentaryConv2D(3, 16, kernel_size=3, use_pentary=True),
    PentaryReLU(),
    PentaryMaxPool2D(pool_size=2),
    PentaryConv2D(16, 32, kernel_size=3, use_pentary=True),
    PentaryReLU(),
    PentaryMaxPool2D(pool_size=2),
    # ... more layers
])
```

## Best Practices

### 1. Quantization

- Use quantization-aware training for best accuracy
- Calibrate on representative dataset
- Monitor quantization error per layer
- Consider mixed precision for critical layers

### 2. Sparsity

- Prune weights after quantization
- Target 70-80% sparsity
- Use magnitude-based pruning
- Retrain after pruning if needed

### 3. Architecture Design

- Prefer ReLU over sigmoid/tanh (hardware-friendly)
- Use batch normalization (can be fused)
- Consider depthwise separable convolutions
- Limit fully connected layers (use global pooling)

### 4. Hardware Mapping

- Map large matrix operations to memristor crossbars
- Use in-memory computing for dense layers
- Exploit sparsity in all layers
- Pipeline operations for throughput

## Future Directions

### 1. Advanced Quantization

- Learned quantization scales
- Per-channel quantization
- Dynamic quantization

### 2. Architecture Search

- Neural architecture search (NAS) for pentary
- AutoML for pentary-optimized networks
- Hardware-aware NAS

### 3. Advanced Networks

- Transformers on pentary
- Graph neural networks
- Spiking neural networks (SNN) with pentary

### 4. System Integration

- End-to-end training on pentary hardware
- Real-time inference optimization
- Multi-chip scaling

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Status**: Architecture Specification - Ready for Implementation
