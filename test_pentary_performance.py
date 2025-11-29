#!/usr/bin/env python3
"""
Simplified Pentary Performance Test
Tests speed and demonstrates Gemma quantization
"""

import sys
import os

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: numpy not available. Some tests will be skipped.")
    NUMPY_AVAILABLE = False

import time
import json
from pathlib import Path


def test_matrix_multiply_speed():
    """Test matrix multiplication speed comparison"""
    if not NUMPY_AVAILABLE:
        print("Skipping matrix multiply test (numpy not available)")
        return None

    print("\n" + "="*70)
    print("Matrix Multiplication Speed Test")
    print("="*70)

    m, n, k = 32, 784, 128
    sparsity = 0.7

    # Generate test data
    X = np.random.randn(m, n).astype(np.float32)
    W_binary = np.random.randn(k, n).astype(np.float32) * 0.1

    # Create pentary weights
    W_pentary = np.random.randn(k, n).astype(np.float32) * 0.1
    max_abs = np.max(np.abs(W_pentary))
    scale = max_abs / 2.0 if max_abs > 0 else 1.0
    W_pentary_quantized = np.round(np.clip(W_pentary / scale, -2, 2)).astype(np.int32)

    # Introduce sparsity
    zero_mask = np.random.rand(k, n) < sparsity
    W_pentary_quantized[zero_mask] = 0

    # Count operations
    total_ops = m * n * k
    zero_ops = np.sum(W_pentary_quantized == 0) * m
    nonzero_ops = total_ops - zero_ops

    # Benchmark binary
    start = time.perf_counter()
    for _ in range(10):
        Y_binary = X @ W_binary.T
    binary_time = (time.perf_counter() - start) / 10

    # Benchmark pentary (optimized)
    start = time.perf_counter()
    for _ in range(10):
        Y_pentary = np.zeros((m, k), dtype=np.float32)
        for i in range(k):
            for j in range(n):
                w = W_pentary_quantized[i, j]
                if w == 0:
                    continue
                elif w == 1:
                    Y_pentary[:, i] += X[:, j] * scale
                elif w == -1:
                    Y_pentary[:, i] -= X[:, j] * scale
                elif w == 2:
                    Y_pentary[:, i] += 2 * X[:, j] * scale
                elif w == -2:
                    Y_pentary[:, i] -= 2 * X[:, j] * scale
    pentary_time = (time.perf_counter() - start) / 10

    # Memory usage
    memory_binary = W_binary.nbytes
    memory_pentary_theoretical = (k * n * 3) // 8

    speedup = binary_time / pentary_time if pentary_time > 0 else 0
    memory_reduction = memory_binary / memory_pentary_theoretical if memory_pentary_theoretical > 0 else 0

    print(f"Matrix size: {m}×{n} @ {n}×{k}")
    print(f"Binary time:     {binary_time*1000:.3f} ms")
    print(f"Pentary time:    {pentary_time*1000:.3f} ms")
    print(f"Speedup:         {speedup:.2f}×")
    print(f"Memory binary:   {memory_binary / 1024:.2f} KB")
    print(f"Memory pentary:  {memory_pentary_theoretical / 1024:.2f} KB")
    print(f"Memory reduction: {memory_reduction:.2f}×")
    print(f"Sparsity:        {sparsity*100:.1f}%")
    print(f"Operations:      {total_ops:,} → {nonzero_ops:,} ({nonzero_ops/total_ops*100:.1f}% of original)")

    return {
        'binary_time': binary_time,
        'pentary_time': pentary_time,
        'speedup': speedup,
        'memory_reduction': memory_reduction,
        'sparsity': sparsity
    }


def test_gemma_quantization():
    """Test Gemma model quantization"""
    if not NUMPY_AVAILABLE:
        print("Skipping Gemma quantization test (numpy not available)")
        return None

    print("\n" + "="*70)
    print("Gemma 2B Quantization Test")
    print("="*70)

    # Create smaller dummy Gemma weights for testing
    hidden_size = 512  # Reduced from 2048
    num_layers = 4  # Reduced from 18
    intermediate_size = 2048  # Reduced from 8192
    vocab_size = 10000  # Reduced from 256000

    print("Creating dummy Gemma 2B model...")
    weights = {}

    # Embedding
    weights['embed_tokens.weight'] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

    # Transformer layers
    for i in range(num_layers):
        prefix = f'layers.{i}'
        weights[f'{prefix}.self_attn.q_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.self_attn.k_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.self_attn.v_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.self_attn.o_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.mlp.gate_proj.weight'] = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.mlp.up_proj.weight'] = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
        weights[f'{prefix}.mlp.down_proj.weight'] = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02

    # Output
    weights['lm_head.weight'] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

    print(f"Created {len(weights)} weight tensors")

    # Quantize to pentary
    print("\nQuantizing to pentary format...")
    start_time = time.time()

    quantized_weights = {}
    scales = {}
    total_params = 0
    total_zeros = 0

    for name, weight in weights.items():
        # Quantize
        max_abs = np.max(np.abs(weight))
        scale = max_abs / 2.0 if max_abs > 0 else 1.0
        quantized = np.round(np.clip(weight / scale, -2, 2)).astype(np.int32)

        # Introduce sparsity (70% zeros)
        zero_mask = np.random.rand(*quantized.shape) < 0.7
        quantized[zero_mask] = 0

        quantized_weights[name] = quantized
        scales[name] = scale

        # Statistics
        params = quantized.size
        zeros = np.sum(quantized == 0)
        total_params += params
        total_zeros += zeros

    quantization_time = time.time() - start_time

    # Calculate statistics
    sparsity = total_zeros / total_params if total_params > 0 else 0
    original_size = sum(w.nbytes for w in weights.values())
    quantized_size_bits = total_params * 3
    quantized_size_bytes = quantized_size_bits / 8
    size_reduction = original_size / quantized_size_bytes if quantized_size_bytes > 0 else 0

    print(f"\nQuantization Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Zero parameters:     {total_zeros:,}")
    print(f"  Sparsity:            {sparsity*100:.1f}%")
    print(f"  Original size:       {original_size / (1024**2):.2f} MB")
    print(f"  Quantized size:      {quantized_size_bytes / (1024**2):.2f} MB")
    print(f"  Size reduction:      {size_reduction:.2f}×")
    print(f"  Quantization time:   {quantization_time:.2f} seconds")

    # Test inference speed
    print("\nTesting inference speed...")
    test_input = np.random.randint(0, 1000, (1, 32))  # Batch=1, seq_len=32

    # Simulate forward pass through one layer
    layer_name = 'layers.0.self_attn.q_proj.weight'
    w_quantized = quantized_weights[layer_name]
    scale = scales[layer_name]

    # Create dummy input embeddings
    x = np.random.randn(1, 32, hidden_size).astype(np.float32)

    # Benchmark pentary multiplication
    start = time.perf_counter()
    for _ in range(10):
        output = np.zeros((1, 32, hidden_size), dtype=np.float32)
        for b in range(1):
            for s in range(32):
                for i in range(hidden_size):
                    for j in range(hidden_size):
                        w = w_quantized[i, j]
                        if w == 0:
                            continue
                        elif w == 1:
                            output[b, s, i] += x[b, s, j] * scale
                        elif w == -1:
                            output[b, s, i] -= x[b, s, j] * scale
                        elif w == 2:
                            output[b, s, i] += 2 * x[b, s, j] * scale
                        elif w == -2:
                            output[b, s, i] -= 2 * x[b, s, j] * scale
    inference_time = (time.perf_counter() - start) / 10

    print(f"  Inference time (1 layer): {inference_time*1000:.2f} ms")

    return {
        'total_parameters': total_params,
        'sparsity': sparsity,
        'size_reduction': size_reduction,
        'quantization_time': quantization_time,
        'inference_time': inference_time
    }


def main():
    """Run all tests"""
    print("="*70)
    print("PENTARY SYSTEM PERFORMANCE TESTS")
    print("="*70)

    results = {}

    # Test 1: Matrix multiplication speed
    result1 = test_matrix_multiply_speed()
    if result1:
        results['matrix_multiply'] = result1

    # Test 2: Gemma quantization
    result2 = test_gemma_quantization()
    if result2:
        results['gemma_quantization'] = result2

    # Save results
    output_file = Path('pentary_performance_test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    if 'matrix_multiply' in results:
        r = results['matrix_multiply']
        print(f"Matrix Multiply Speedup: {r['speedup']:.2f}×")
        print(f"Memory Reduction: {r['memory_reduction']:.2f}×")

    if 'gemma_quantization' in results:
        r = results['gemma_quantization']
        print(f"Gemma Parameters: {r['total_parameters']:,}")
        print(f"Gemma Sparsity: {r['sparsity']*100:.1f}%")
        print(f"Gemma Size Reduction: {r['size_reduction']:.2f}×")

    print(f"\nResults saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
