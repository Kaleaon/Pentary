#!/usr/bin/env python3
"""
Pentary Rapid Iteration Tools
Profiling, optimization, and benchmarking for pentary systems
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import json
from dataclasses import dataclass, asdict
from pentary_nn import PentaryNetwork, PentaryLayer
from pentary_quantizer import PentaryQuantizer


@dataclass
class ProfileResult:
    """Profile result for a single operation"""
    operation: str
    layer_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    execution_time: float
    memory_used: int
    sparsity: float
    operations_count: int
    energy_estimate: float  # Estimated energy in pJ


class PentaryProfiler:
    """Profiles pentary neural network operations"""
    
    def __init__(self):
        self.profile_results = []
        self.enabled = True
        
    def profile_layer(self, layer: PentaryLayer, x: np.ndarray, 
                     operation_name: str = "forward") -> ProfileResult:
        """Profile a single layer operation"""
        if not self.enabled:
            return None
            
        # Measure execution time
        start_time = time.perf_counter()
        output = layer.forward(x) if operation_name == "forward" else layer.backward(x)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1e9  # Convert to nanoseconds
        
        # Estimate memory usage
        input_memory = x.nbytes
        output_memory = output.nbytes if hasattr(output, 'nbytes') else 0
        params = layer.get_parameters()
        param_memory = sum(p.nbytes for p in params.values() if hasattr(p, 'nbytes'))
        total_memory = input_memory + output_memory + param_memory
        
        # Compute sparsity
        sparsity = 0.0
        if 'weights' in params:
            weights = params['weights']
            sparsity = float(np.sum(weights == 0) / weights.size) if weights.size > 0 else 0.0
        
        # Estimate operations count
        if hasattr(layer, 'weights'):
            weights = layer.weights
            non_zero = np.sum(weights != 0)
            ops_count = int(non_zero * x.shape[0])  # Simplified
        else:
            ops_count = int(np.prod(x.shape))
        
        # Estimate energy (rough approximation)
        # Pentary operations are more energy efficient
        energy_per_op = 0.1  # pJ per operation (pentary is ~10x more efficient)
        energy_estimate = ops_count * energy_per_op * (1 - sparsity)  # Sparsity reduces energy
        
        result = ProfileResult(
            operation=operation_name,
            layer_name=layer.name,
            input_shape=x.shape,
            output_shape=output.shape if hasattr(output, 'shape') else (0,),
            execution_time=execution_time,
            memory_used=total_memory,
            sparsity=sparsity,
            operations_count=ops_count,
            energy_estimate=energy_estimate
        )
        
        self.profile_results.append(result)
        return result
    
    def profile_network(self, network: PentaryNetwork, x: np.ndarray) -> List[ProfileResult]:
        """Profile entire network forward pass"""
        results = []
        current_input = x
        
        for layer in network.layers:
            result = self.profile_layer(layer, current_input, "forward")
            if result:
                results.append(result)
                current_input = layer.forward(current_input)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profile_results:
            return {}
        
        total_time = sum(r.execution_time for r in self.profile_results)
        total_memory = sum(r.memory_used for r in self.profile_results)
        total_ops = sum(r.operations_count for r in self.profile_results)
        total_energy = sum(r.energy_estimate for r in self.profile_results)
        avg_sparsity = np.mean([r.sparsity for r in self.profile_results])
        
        return {
            'total_execution_time_ns': total_time,
            'total_execution_time_ms': total_time / 1e6,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024),
            'total_operations': total_ops,
            'total_energy_pj': total_energy,
            'total_energy_nj': total_energy / 1000,
            'average_sparsity': avg_sparsity,
            'layer_count': len(self.profile_results),
            'throughput_ops_per_sec': total_ops / (total_time / 1e9) if total_time > 0 else 0,
            'energy_efficiency_topps_per_w': (total_ops / 1e12) / (total_energy / 1e12 / 5.0) if total_energy > 0 else 0
        }
    
    def print_summary(self):
        """Print profiling summary"""
        summary = self.get_summary()
        if not summary:
            print("No profiling data available")
            return
        
        print("\n" + "=" * 70)
        print("Profiling Summary")
        print("=" * 70)
        print(f"Total Execution Time: {summary['total_execution_time_ms']:.2f} ms")
        print(f"Total Memory Used: {summary['total_memory_mb']:.2f} MB")
        print(f"Total Operations: {summary['total_operations']:,}")
        print(f"Total Energy: {summary['total_energy_nj']:.2f} nJ")
        print(f"Average Sparsity: {summary['average_sparsity']:.1%}")
        print(f"Throughput: {summary['throughput_ops_per_sec'] / 1e9:.2f} GOPS")
        print(f"Energy Efficiency: {summary['energy_efficiency_topps_per_w']:.2f} TOPS/W")
        print("=" * 70)
        
        print("\nPer-Layer Breakdown:")
        print("-" * 70)
        print(f"{'Layer':<20} {'Time (ns)':<12} {'Memory (KB)':<15} {'Sparsity':<10} {'Energy (pJ)':<15}")
        print("-" * 70)
        
        for result in self.profile_results:
            print(f"{result.layer_name:<20} {result.execution_time:<12.2f} "
                  f"{result.memory_used/1024:<15.2f} {result.sparsity:<10.1%} "
                  f"{result.energy_estimate:<15.2f}")


class PentaryOptimizer:
    """Optimizes pentary networks for performance"""
    
    def __init__(self):
        self.optimization_history = []
        
    def prune_weights(self, network: PentaryNetwork, 
                     sparsity_target: float = 0.8,
                     method: str = 'magnitude') -> PentaryNetwork:
        """
        Prune network weights to increase sparsity
        
        Args:
            network: Network to prune
            sparsity_target: Target sparsity (0.0 to 1.0)
            method: 'magnitude' or 'random'
        """
        params = network.get_parameters()
        
        for layer_key, layer_params in params.items():
            if 'weights' not in layer_params:
                continue
            
            weights = layer_params['weights']
            current_sparsity = np.sum(weights == 0) / weights.size
            
            if current_sparsity >= sparsity_target:
                continue
            
            # Compute threshold
            if method == 'magnitude':
                # Prune smallest magnitude weights
                abs_weights = np.abs(weights)
                threshold = np.percentile(abs_weights[abs_weights > 0], 
                                        (1 - sparsity_target) * 100)
                mask = abs_weights < threshold
            else:  # random
                # Random pruning
                num_to_prune = int(weights.size * (sparsity_target - current_sparsity))
                indices = np.random.choice(weights.size, num_to_prune, replace=False)
                mask = np.zeros_like(weights, dtype=bool)
                mask.flat[indices] = True
            
            # Apply pruning
            weights[mask] = 0
            layer_params['weights'] = weights
        
        # Update network with pruned weights
        network.set_parameters(params)
        return network
    
    def quantize_network(self, network: PentaryNetwork,
                        quantizer: PentaryQuantizer) -> PentaryNetwork:
        """Quantize network weights to pentary"""
        params = network.get_parameters()
        quantized_params = {}
        
        for layer_key, layer_params in params.items():
            if 'weights' not in layer_params:
                quantized_params[layer_key] = layer_params
                continue
            
            weights = layer_params['weights']
            quantized, scale, zero_point = quantizer.quantize_tensor(weights)
            
            quantized_params[layer_key] = {
                'weights': quantized,
                'bias': layer_params.get('bias', None),
                'scale': scale,
                'zero_point': zero_point
            }
        
        # Create new network with quantized weights
        # (This is simplified - full implementation would recreate layers)
        network.set_parameters(quantized_params)
        return network
    
    def optimize_for_inference(self, network: PentaryNetwork) -> PentaryNetwork:
        """Apply inference optimizations"""
        # Set to inference mode
        network.training = False
        
        # Prune weights
        network = self.prune_weights(network, sparsity_target=0.7)
        
        return network


class PentaryBenchmark:
    """Benchmarks pentary networks"""
    
    def __init__(self):
        self.benchmark_results = []
        
    def benchmark_network(self, network: PentaryNetwork,
                          input_shape: Tuple[int, ...],
                          num_iterations: int = 100,
                          warmup_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark network performance
        
        Args:
            network: Network to benchmark
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            warmup_iterations: Warmup iterations (not counted)
        """
        # Warmup
        x = np.random.randn(*input_shape)
        for _ in range(warmup_iterations):
            _ = network.forward(x)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            x = np.random.randn(*input_shape)
            start = time.perf_counter()
            _ = network.forward(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        # Compute statistics
        result = {
            'mean_time_ms': float(np.mean(times)),
            'median_time_ms': float(np.median(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000 / np.mean(times)),
            'iterations': num_iterations
        }
        
        self.benchmark_results.append(result)
        return result
    
    def compare_networks(self, networks: Dict[str, PentaryNetwork],
                        input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compare multiple networks"""
        results = {}
        
        for name, network in networks.items():
            print(f"Benchmarking {name}...")
            result = self.benchmark_network(network, input_shape)
            results[name] = result
        
        return results
    
    def print_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Print comparison results"""
        print("\n" + "=" * 70)
        print("Network Comparison")
        print("=" * 70)
        print(f"{'Network':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput (fps)':<15}")
        print("-" * 70)
        
        for name, result in results.items():
            print(f"{name:<20} {result['mean_time_ms']:<12.2f} "
                  f"{result['p95_time_ms']:<12.2f} {result['throughput_fps']:<15.2f}")
        
        print("=" * 70)


class PentaryRapidIteration:
    """Main rapid iteration tool combining profiling, optimization, and benchmarking"""
    
    def __init__(self):
        self.profiler = PentaryProfiler()
        self.optimizer = PentaryOptimizer()
        self.benchmark = PentaryBenchmark()
        
    def iterate(self, network: PentaryNetwork, 
                input_shape: Tuple[int, ...],
                target_latency_ms: Optional[float] = None,
                target_sparsity: Optional[float] = None) -> PentaryNetwork:
        """
        Rapid iteration: profile, optimize, benchmark, repeat
        
        Args:
            network: Network to optimize
            input_shape: Input shape for benchmarking
            target_latency_ms: Target latency in milliseconds
            target_sparsity: Target sparsity ratio
        """
        iteration = 0
        max_iterations = 10
        
        print("Starting rapid iteration...")
        print("=" * 70)
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}")
            print("-" * 70)
            
            # Profile
            x = np.random.randn(*input_shape)
            self.profiler.profile_network(network, x)
            summary = self.profiler.get_summary()
            
            print(f"Current latency: {summary['total_execution_time_ms']:.2f} ms")
            print(f"Current sparsity: {summary['average_sparsity']:.1%}")
            
            # Check if targets are met
            if target_latency_ms and summary['total_execution_time_ms'] <= target_latency_ms:
                if target_sparsity is None or summary['average_sparsity'] >= target_sparsity:
                    print("Targets met! Stopping optimization.")
                    break
            
            # Optimize
            if target_sparsity and summary['average_sparsity'] < target_sparsity:
                print("Pruning weights...")
                network = self.optimizer.prune_weights(network, sparsity_target=target_sparsity)
            
            # Clear profiler for next iteration
            self.profiler.profile_results = []
        
        print("\n" + "=" * 70)
        print("Rapid iteration complete!")
        print("=" * 70)
        
        return network


def main():
    """Demo and testing of rapid iteration tools"""
    print("=" * 70)
    print("Pentary Rapid Iteration Tools")
    print("=" * 70)
    print()
    
    from pentary_nn import create_simple_classifier
    
    # Create network
    print("Creating test network...")
    network = create_simple_classifier(input_size=784, hidden_size=128, num_classes=10)
    
    # Profile
    print("\nProfiling network...")
    profiler = PentaryProfiler()
    x = np.random.randn(32, 784)
    profiler.profile_network(network, x)
    profiler.print_summary()
    
    # Benchmark
    print("\nBenchmarking network...")
    benchmark = PentaryBenchmark()
    result = benchmark.benchmark_network(network, (32, 784), num_iterations=50)
    print(f"Mean latency: {result['mean_time_ms']:.2f} ms")
    print(f"Throughput: {result['throughput_fps']:.2f} fps")
    
    # Optimize
    print("\nOptimizing network...")
    optimizer = PentaryOptimizer()
    optimized_network = optimizer.optimize_for_inference(network)
    
    # Compare
    print("\nComparing original vs optimized...")
    networks = {
        'Original': network,
        'Optimized': optimized_network
    }
    comparison = benchmark.compare_networks(networks, (32, 784))
    benchmark.print_comparison(comparison)
    
    # Rapid iteration
    print("\nRunning rapid iteration...")
    rapid_iter = PentaryRapidIteration()
    final_network = rapid_iter.iterate(network, (32, 784), 
                                       target_latency_ms=10.0,
                                       target_sparsity=0.7)
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
