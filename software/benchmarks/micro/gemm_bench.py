#!/usr/bin/env python3
"""
GEMM (General Matrix Multiplication) Benchmark

This benchmark measures the performance of matrix multiplication operations
across different sizes and compares Pentary against baseline implementations.
"""

import argparse
import json
import time
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some benchmarks will be skipped.")

try:
    import pentary
    import torch_pentary
    PENTARY_AVAILABLE = True
except ImportError:
    PENTARY_AVAILABLE = False
    print("Warning: Pentary not available. Pentary benchmarks will be skipped.")


def benchmark_gemm(M: int, N: int, K: int, device: str, iterations: int = 100) -> Dict:
    """
    Benchmark GEMM operation: C = A @ B
    
    Args:
        M: Number of rows in A
        N: Number of columns in B
        K: Number of columns in A (and rows in B)
        device: Device to run on ("cpu", "cuda:0", "pentary:0")
        iterations: Number of iterations for timing
    
    Returns:
        Dictionary with timing and performance metrics
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Create random matrices
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    
    # Warm-up
    for _ in range(10):
        C = torch.matmul(A, B)
    
    # Synchronize
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("pentary"):
        if PENTARY_AVAILABLE:
            pentary.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    
    # Synchronize
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("pentary"):
        if PENTARY_AVAILABLE:
            pentary.synchronize()
    
    end = time.perf_counter()
    
    # Calculate metrics
    elapsed_s = (end - start) / iterations
    elapsed_ms = elapsed_s * 1000
    
    # FLOPS calculation: 2*M*N*K (multiply-add counts as 2 ops)
    flops = 2 * M * N * K
    tflops = flops / (elapsed_s * 1e12)
    
    # Memory traffic (bytes)
    # Read A (M*K), Read B (K*N), Write C (M*N)
    bytes_transferred = (M * K + K * N + M * N) * 4  # 4 bytes per float32
    bandwidth_gb_s = bytes_transferred / (elapsed_s * 1e9)
    
    return {
        "M": M,
        "N": N,
        "K": K,
        "device": device,
        "time_ms": elapsed_ms,
        "throughput_tflops": tflops,
        "memory_bandwidth_gb_s": bandwidth_gb_s,
        "flops": flops
    }


def run_benchmark_suite(devices: List[str], sizes: List[Tuple[int, int, int]]) -> List[Dict]:
    """Run GEMM benchmark across multiple devices and sizes"""
    results = []
    
    for device in devices:
        print(f"\n{'='*60}")
        print(f"Benchmarking on device: {device}")
        print(f"{'='*60}")
        
        for M, N, K in sizes:
            print(f"  Size: {M}x{N}x{K}...", end=" ", flush=True)
            
            try:
                result = benchmark_gemm(M, N, K, device)
                results.append(result)
                print(f"{result['time_ms']:.3f} ms ({result['throughput_tflops']:.2f} TFLOPS)")
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "M": M, "N": N, "K": K,
                    "device": device,
                    "error": str(e)
                })
    
    return results


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of results"""
    print(f"\n{'='*80}")
    print("GEMM Benchmark Results")
    print(f"{'='*80}")
    print(f"{'Size':<20} {'Device':<15} {'Time (ms)':<12} {'TFLOPS':<12} {'BW (GB/s)':<12}")
    print(f"{'-'*80}")
    
    for result in results:
        if "error" in result:
            continue
        
        size_str = f"{result['M']}x{result['N']}x{result['K']}"
        print(f"{size_str:<20} {result['device']:<15} "
              f"{result['time_ms']:<12.3f} {result['throughput_tflops']:<12.2f} "
              f"{result['memory_bandwidth_gb_s']:<12.1f}")
    
    print(f"{'='*80}\n")


def calculate_speedup(results: List[Dict], baseline_device: str = "cpu") -> Dict:
    """Calculate speedup relative to baseline device"""
    speedup_data = {}
    
    # Group results by size
    by_size = {}
    for result in results:
        if "error" in result:
            continue
        size_key = (result['M'], result['N'], result['K'])
        if size_key not in by_size:
            by_size[size_key] = {}
        by_size[size_key][result['device']] = result
    
    # Calculate speedup
    for size_key, device_results in by_size.items():
        if baseline_device not in device_results:
            continue
        
        baseline_time = device_results[baseline_device]['time_ms']
        speedup_data[size_key] = {}
        
        for device, result in device_results.items():
            speedup = baseline_time / result['time_ms']
            speedup_data[size_key][device] = speedup
    
    return speedup_data


def main():
    parser = argparse.ArgumentParser(description="GEMM Benchmark for Pentary")
    parser.add_argument("--devices", type=str, default="cpu,pentary:0",
                        help="Comma-separated list of devices to benchmark")
    parser.add_argument("--sizes", type=str, default="128,256,512,1024,2048,4096",
                        help="Comma-separated list of matrix sizes (square matrices)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for timing")
    parser.add_argument("--output", type=str, default="gemm_results.json",
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Parse devices
    devices = [d.strip() for d in args.devices.split(",")]
    
    # Parse sizes (assume square matrices for simplicity)
    size_list = [int(s.strip()) for s in args.sizes.split(",")]
    sizes = [(s, s, s) for s in size_list]
    
    # Run benchmarks
    results = run_benchmark_suite(devices, sizes)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Calculate and print speedup
    if "cpu" in devices and len(devices) > 1:
        speedup_data = calculate_speedup(results, baseline_device="cpu")
        print("\nSpeedup vs. CPU:")
        for size_key, device_speedups in speedup_data.items():
            print(f"  Size {size_key[0]}x{size_key[1]}x{size_key[2]}:")
            for device, speedup in device_speedups.items():
                if device != "cpu":
                    print(f"    {device}: {speedup:.2f}x")
    
    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump({
            "benchmark": "gemm",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
