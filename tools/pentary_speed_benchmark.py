#!/usr/bin/env python3
"""
Pentary System Speed Benchmark
Comprehensive tests demonstrating speed advantages of pentary computing
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
try:
    from pentary_nn import PentaryLinear, PentaryNetwork
except ImportError:
    # Fallback if pentary_nn not available
    PentaryLinear = None
    PentaryNetwork = None

try:
    from pentary_quantizer import PentaryQuantizer
except ImportError:
    PentaryQuantizer = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""
    name: str
    binary_time: float
    pentary_time: float
    speedup: float
    memory_binary: int
    memory_pentary: int
    memory_reduction: float
    sparsity: float
    operations_binary: int
    operations_pentary: int
    operations_reduction: float


class PentarySpeedBenchmark:
    """Comprehensive speed benchmarking suite"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark_matrix_multiply(self, m: int, n: int, k: int,
                                  sparsity: float = 0.7) -> BenchmarkResult:
        """
        Benchmark matrix multiplication: Y = X @ W^T

        Args:
            m: Batch size
            n: Input features
            k: Output features
            sparsity: Fraction of zero weights
        """
        print(f"\n{'='*70}")
        print(f"Matrix Multiplication Benchmark: {m}×{n} @ {n}×{k}")
        print(f"{'='*70}")

        # Generate test data
        X = np.random.randn(m, n).astype(np.float32)
        W_binary = np.random.randn(k, n).astype(np.float32) * 0.1

        # Create pentary weights with sparsity
        W_pentary = np.random.randn(k, n).astype(np.float32) * 0.1
        # Quantize to pentary
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
                        continue  # Skip zero weights
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
        memory_binary = W_binary.nbytes  # FP32 = 4 bytes
        memory_pentary = W_pentary_quantized.nbytes  # int32 = 4 bytes (but could be 3 bits)
        # Theoretical pentary: 3 bits per weight
        memory_pentary_theoretical = (k * n * 3) // 8

        speedup = binary_time / pentary_time if pentary_time > 0 else 0
        memory_reduction = memory_binary / memory_pentary_theoretical if memory_pentary_theoretical > 0 else 0
        operations_reduction = nonzero_ops / total_ops if total_ops > 0 else 0

        result = BenchmarkResult(
            name=f"Matrix Multiply {m}×{n}×{k}",
            binary_time=binary_time,
            pentary_time=pentary_time,
            speedup=speedup,
            memory_binary=memory_binary,
            memory_pentary=memory_pentary_theoretical,
            memory_reduction=memory_reduction,
            sparsity=sparsity,
            operations_binary=total_ops,
            operations_pentary=nonzero_ops,
            operations_reduction=operations_reduction
        )

        print(f"Binary time:     {binary_time*1000:.3f} ms")
        print(f"Pentary time:    {pentary_time*1000:.3f} ms")
        print(f"Speedup:         {speedup:.2f}×")
        print(f"Memory binary:   {memory_binary / 1024:.2f} KB")
        print(f"Memory pentary:  {memory_pentary_theoretical / 1024:.2f} KB")
        print(f"Memory reduction: {memory_reduction:.2f}×")
        print(f"Sparsity:        {sparsity*100:.1f}%")
        print(f"Operations:      {total_ops:,} → {nonzero_ops:,} ({operations_reduction*100:.1f}% reduction)")

        return result

    def benchmark_linear_layer(self, batch_size: int, in_features: int,
                              out_features: int) -> BenchmarkResult:
        """Benchmark a single linear layer"""
        print(f"\n{'='*70}")
        print(f"Linear Layer Benchmark: {batch_size}×{in_features} → {batch_size}×{out_features}")
        print(f"{'='*70}")

        # Create layers
        layer_binary = PentaryLinear(in_features, out_features, use_pentary=False)
        layer_pentary = PentaryLinear(in_features, out_features, use_pentary=True)

        # Generate input
        X = np.random.randn(batch_size, in_features).astype(np.float32)

        # Count sparsity
        sparsity = np.sum(layer_pentary.weights == 0) / layer_pentary.weights.size

        # Benchmark binary
        start = time.perf_counter()
        for _ in range(10):
            Y_binary = layer_binary.forward(X)
        binary_time = (time.perf_counter() - start) / 10

        # Benchmark pentary
        start = time.perf_counter()
        for _ in range(10):
            Y_pentary = layer_pentary.forward(X)
        pentary_time = (time.perf_counter() - start) / 10

        # Memory
        memory_binary = layer_binary.weights.nbytes
        memory_pentary_theoretical = (in_features * out_features * 3) // 8

        speedup = binary_time / pentary_time if pentary_time > 0 else 0
        memory_reduction = memory_binary / memory_pentary_theoretical if memory_pentary_theoretical > 0 else 0

        result = BenchmarkResult(
            name=f"Linear Layer {in_features}→{out_features}",
            binary_time=binary_time,
            pentary_time=pentary_time,
            speedup=speedup,
            memory_binary=memory_binary,
            memory_pentary=memory_pentary_theoretical,
            memory_reduction=memory_reduction,
            sparsity=sparsity,
            operations_binary=in_features * out_features * batch_size,
            operations_pentary=int((1 - sparsity) * in_features * out_features * batch_size),
            operations_reduction=1 - sparsity
        )

        print(f"Binary time:      {binary_time*1000:.3f} ms")
        print(f"Pentary time:    {pentary_time*1000:.3f} ms")
        print(f"Speedup:          {speedup:.2f}×")
        print(f"Memory reduction: {memory_reduction:.2f}×")
        print(f"Sparsity:         {sparsity*100:.1f}%")

        return result

    def benchmark_network(self, input_size: int, hidden_sizes: List[int],
                         output_size: int, batch_size: int = 32) -> BenchmarkResult:
        """Benchmark a complete network"""
        print(f"\n{'='*70}")
        print(f"Network Benchmark: {input_size} → {hidden_sizes} → {output_size}")
        print(f"{'='*70}")

        # Create networks
        layers_binary = []
        layers_pentary = []

        dims = [input_size] + hidden_sizes + [output_size]
        for i in range(len(dims) - 1):
            layers_binary.append(PentaryLinear(dims[i], dims[i+1], use_pentary=False))
            layers_pentary.append(PentaryLinear(dims[i], dims[i+1], use_pentary=True))

        network_binary = PentaryNetwork(layers_binary)
        network_pentary = PentaryNetwork(layers_pentary)

        # Generate input
        X = np.random.randn(batch_size, input_size).astype(np.float32)

        # Count total sparsity
        total_params = 0
        total_zeros = 0
        for layer in layers_pentary:
            params = layer.weights.size
            zeros = np.sum(layer.weights == 0)
            total_params += params
            total_zeros += zeros
        sparsity = total_zeros / total_params if total_params > 0 else 0

        # Benchmark binary
        start = time.perf_counter()
        for _ in range(10):
            Y_binary = network_binary.forward(X)
        binary_time = (time.perf_counter() - start) / 10

        # Benchmark pentary
        start = time.perf_counter()
        for _ in range(10):
            Y_pentary = network_pentary.forward(X)
        pentary_time = (time.perf_counter() - start) / 10

        # Memory
        memory_binary = sum(l.weights.nbytes for l in layers_binary)
        memory_pentary_theoretical = (total_params * 3) // 8

        speedup = binary_time / pentary_time if pentary_time > 0 else 0
        memory_reduction = memory_binary / memory_pentary_theoretical if memory_pentary_theoretical > 0 else 0

        result = BenchmarkResult(
            name=f"Network {input_size}→{hidden_sizes}→{output_size}",
            binary_time=binary_time,
            pentary_time=pentary_time,
            speedup=speedup,
            memory_binary=memory_binary,
            memory_pentary=memory_pentary_theoretical,
            memory_reduction=memory_reduction,
            sparsity=sparsity,
            operations_binary=total_params * batch_size,
            operations_pentary=int((1 - sparsity) * total_params * batch_size),
            operations_reduction=1 - sparsity
        )

        print(f"Binary time:      {binary_time*1000:.3f} ms")
        print(f"Pentary time:     {pentary_time*1000:.3f} ms")
        print(f"Speedup:          {speedup:.2f}×")
        print(f"Memory reduction: {memory_reduction:.2f}×")
        print(f"Total parameters: {total_params:,}")
        print(f"Sparsity:         {sparsity*100:.1f}%")

        return result

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks and return summary"""
        print("="*70)
        print("PENTARY SYSTEM SPEED BENCHMARK SUITE")
        print("="*70)

        # Matrix multiplication benchmarks
        self.results.append(self.benchmark_matrix_multiply(32, 784, 128, sparsity=0.7))
        self.results.append(self.benchmark_matrix_multiply(64, 512, 256, sparsity=0.7))
        self.results.append(self.benchmark_matrix_multiply(128, 1024, 512, sparsity=0.7))

        # Linear layer benchmarks
        self.results.append(self.benchmark_linear_layer(32, 784, 128))
        self.results.append(self.benchmark_linear_layer(64, 512, 256))
        self.results.append(self.benchmark_linear_layer(128, 1024, 512))

        # Network benchmarks
        self.results.append(self.benchmark_network(784, [128, 64], 10, batch_size=32))
        self.results.append(self.benchmark_network(512, [256, 128, 64], 10, batch_size=64))

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")

        avg_speedup = np.mean([r.speedup for r in self.results])
        avg_memory_reduction = np.mean([r.memory_reduction for r in self.results])
        avg_sparsity = np.mean([r.sparsity for r in self.results])
        avg_ops_reduction = np.mean([r.operations_reduction for r in self.results])

        print(f"\nAverage Speedup:           {avg_speedup:.2f}×")
        print(f"Average Memory Reduction:  {avg_memory_reduction:.2f}×")
        print(f"Average Sparsity:          {avg_sparsity*100:.1f}%")
        print(f"Average Operations Reduction: {avg_ops_reduction*100:.1f}%")

        print(f"\n{'Benchmark':<40} {'Speedup':<10} {'Mem Red':<10} {'Sparsity':<10}")
        print("-" * 70)
        for r in self.results:
            print(f"{r.name:<40} {r.speedup:>8.2f}×  {r.memory_reduction:>8.2f}×  {r.sparsity*100:>7.1f}%")

        summary = {
            'results': [
                {
                    'name': r.name,
                    'binary_time_ms': r.binary_time * 1000,
                    'pentary_time_ms': r.pentary_time * 1000,
                    'speedup': r.speedup,
                    'memory_reduction': r.memory_reduction,
                    'sparsity': r.sparsity,
                    'operations_reduction': r.operations_reduction
                }
                for r in self.results
            ],
            'summary': {
                'avg_speedup': float(avg_speedup),
                'avg_memory_reduction': float(avg_memory_reduction),
                'avg_sparsity': float(avg_sparsity),
                'avg_operations_reduction': float(avg_ops_reduction)
            }
        }

        return summary

    def save_results(self, filepath: str):
        """Save benchmark results to JSON"""
        summary = self.generate_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {filepath}")


def main():
    """Run speed benchmarks"""
    benchmark = PentarySpeedBenchmark()
    summary = benchmark.run_all_benchmarks()
    benchmark.save_results('pentary_speed_benchmark_results.json')

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
