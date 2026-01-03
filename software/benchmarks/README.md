# Pentary Benchmarking Suite

## Overview

This benchmarking suite provides a comprehensive set of tests to validate the performance of the Pentary architecture against other AI accelerators (GPUs, TPUs).

## Benchmark Categories

### 1. Micro-Benchmarks

These tests measure the performance of individual operations:

- **GEMM (Matrix Multiplication)**: Various sizes from 128x128 to 8192x8192
- **Convolution**: Standard kernel sizes (3x3, 5x5, 7x7) with various input sizes
- **Activation Functions**: ReLU, GeLU, Sigmoid, Tanh
- **Memory Bandwidth**: Sequential and random access patterns
- **Cache Performance**: Hit rate analysis for different access patterns

### 2. Macro-Benchmarks

These tests measure end-to-end performance on real neural network models:

- **ResNet-50**: Image classification (224x224 input)
- **BERT-Base**: Natural language processing (512 sequence length)
- **GPT-2**: Language generation (1024 context length)
- **Llama-7B**: Large language model inference
- **YOLO-v8**: Object detection
- **Stable Diffusion**: Image generation

### 3. Comparison Benchmarks

Direct comparisons against:

- **NVIDIA H100**: Using CUDA and cuDNN
- **Google TPUv4**: Using JAX and XLA
- **AMD MI300X**: Using ROCm

## Running Benchmarks

### Prerequisites

```bash
pip install torch torchvision transformers numpy pandas matplotlib
```

### Run All Benchmarks

```bash
cd software/benchmarks
python run_all_benchmarks.py --device pentary:0 --output results/
```

### Run Specific Benchmark

```bash
# GEMM benchmark
python micro/gemm_bench.py --sizes 1024,2048,4096 --device pentary:0

# ResNet-50 benchmark
python macro/resnet50_bench.py --batch-size 32 --device pentary:0
```

## Results Format

Benchmark results are saved in JSON format:

```json
{
  "benchmark": "gemm",
  "device": "pentary:0",
  "timestamp": "2026-01-03T15:00:00",
  "results": [
    {
      "size": [1024, 1024, 1024],
      "time_ms": 0.523,
      "throughput_tflops": 4.12,
      "memory_bandwidth_gb_s": 1024.5
    }
  ]
}
```

## Visualization

Generate comparison plots:

```bash
python visualize_results.py --input results/ --output plots/
```

This will generate:

- **Performance comparison bar charts**
- **Speedup vs. baseline graphs**
- **Power efficiency plots (TOPS/Watt)**
- **Cost efficiency plots (TOPS/Dollar)**

## Expected Results

Based on our architecture analysis, we expect:

| Metric | Pentary PT-2000 | NVIDIA H100 | Speedup |
|--------|-----------------|-------------|---------|
| GEMM (4096x4096) | 0.21 ms | 0.65 ms | 3.1x |
| ResNet-50 (batch=32) | 12.3 ms | 35.2 ms | 2.9x |
| BERT-Base (seq=512) | 8.7 ms | 24.1 ms | 2.8x |
| Power Efficiency | 34.7 TOPS/W | 14.2 TOPS/W | 2.4x |

## Benchmark Implementation

### Example: GEMM Benchmark

```python
import pentary
import torch
import time

def benchmark_gemm(M, N, K, device, iterations=100):
    # Create random matrices
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    
    # Warm-up
    for _ in range(10):
        C = torch.matmul(A, B)
    
    # Synchronize
    if device.type == "pentary":
        pentary.synchronize()
    else:
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    
    if device.type == "pentary":
        pentary.synchronize()
    else:
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    # Calculate metrics
    time_ms = (end - start) * 1000 / iterations
    flops = 2 * M * N * K
    tflops = flops / (time_ms * 1e9)
    
    return {
        "time_ms": time_ms,
        "throughput_tflops": tflops
    }
```

## Contributing

To add a new benchmark:

1. Create a new Python file in `micro/` or `macro/`
2. Implement the benchmark following the template
3. Add it to `run_all_benchmarks.py`
4. Update this README with expected results
