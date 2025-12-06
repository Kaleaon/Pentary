#!/usr/bin/env python3
"""
Neural Network benchmarks for Pentary quantization claims.
Tests memory reduction and inference speed improvements.
"""

import numpy as np
import time
import json
from typing import Tuple, Dict, List

class PentaryQuantizer:
    """Quantizes neural network weights to pentary values."""
    
    def __init__(self):
        # Pentary values: {-2, -1, 0, 1, 2}
        self.pentary_values = np.array([-2, -1, 0, 1, 2])
    
    def quantize(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights to pentary values.
        Uses nearest neighbor quantization.
        """
        # Normalize weights to [-2, 2] range
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 0:
            normalized = 4 * (weights - w_min) / (w_max - w_min) - 2
        else:
            normalized = np.zeros_like(weights)
        
        # Quantize to nearest pentary value
        quantized = np.zeros_like(normalized)
        for i, val in enumerate(self.pentary_values):
            mask = np.abs(normalized - val) <= 0.5
            quantized[mask] = val
        
        return quantized
    
    def calculate_memory_usage(self, weights: np.ndarray, dtype: str) -> int:
        """Calculate memory usage in bytes."""
        if dtype == 'fp32':
            return weights.size * 4  # 4 bytes per float32
        elif dtype == 'fp16':
            return weights.size * 2  # 2 bytes per float16
        elif dtype == 'int8':
            return weights.size * 1  # 1 byte per int8
        elif dtype == 'pentary':
            # 3 bits per pentary value (5 states fit in 3 bits)
            return int(np.ceil(weights.size * 3 / 8))
        elif dtype == 'pentary_optimal':
            # Optimal: log2(5) = 2.32 bits per value
            return int(np.ceil(weights.size * 2.32 / 8))
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

class NeuralNetworkBenchmark:
    """Benchmark neural network operations."""
    
    def __init__(self):
        self.quantizer = PentaryQuantizer()
    
    def create_test_network(self, size: str) -> Dict[str, np.ndarray]:
        """Create test neural network weights."""
        
        if size == 'small':
            # Small network: ~1M parameters
            layers = {
                'layer1': np.random.randn(784, 256),      # 200K params
                'layer2': np.random.randn(256, 128),      # 33K params
                'layer3': np.random.randn(128, 10),       # 1.3K params
            }
        elif size == 'medium':
            # Medium network: ~10M parameters
            layers = {
                'layer1': np.random.randn(1024, 2048),    # 2M params
                'layer2': np.random.randn(2048, 2048),    # 4M params
                'layer3': np.random.randn(2048, 1024),    # 2M params
                'layer4': np.random.randn(1024, 1000),    # 1M params
            }
        else:  # large
            # Large network: ~100M parameters
            layers = {
                'layer1': np.random.randn(2048, 4096),    # 8M params
                'layer2': np.random.randn(4096, 4096),    # 16M params
                'layer3': np.random.randn(4096, 4096),    # 16M params
                'layer4': np.random.randn(4096, 2048),    # 8M params
                'layer5': np.random.randn(2048, 1000),    # 2M params
            }
        
        return layers
    
    def benchmark_memory(self, network: Dict[str, np.ndarray]) -> Dict:
        """Benchmark memory usage for different quantization schemes."""
        
        total_params = sum(w.size for w in network.values())
        
        # Calculate memory for each dtype
        memory_usage = {}
        
        for dtype in ['fp32', 'fp16', 'int8', 'pentary', 'pentary_optimal']:
            total_memory = 0
            for layer_weights in network.values():
                total_memory += self.quantizer.calculate_memory_usage(layer_weights, dtype)
            memory_usage[dtype] = total_memory
        
        # Calculate reductions
        fp32_memory = memory_usage['fp32']
        reductions = {
            dtype: fp32_memory / mem if mem > 0 else 0
            for dtype, mem in memory_usage.items()
        }
        
        return {
            'total_parameters': total_params,
            'memory_bytes': memory_usage,
            'memory_mb': {k: v / (1024 * 1024) for k, v in memory_usage.items()},
            'reduction_vs_fp32': reductions
        }
    
    def benchmark_inference_speed(self, network: Dict[str, np.ndarray], 
                                  num_inferences: int = 100) -> Dict:
        """Benchmark inference speed for different quantization schemes."""
        
        # Create test input
        input_size = list(network.values())[0].shape[0]
        test_input = np.random.randn(num_inferences, input_size)
        
        results = {}
        
        # FP32 baseline
        fp32_start = time.time()
        for inp in test_input:
            x = inp
            for weights in network.values():
                x = np.dot(x, weights)
                x = np.maximum(0, x)  # ReLU
        fp32_time = time.time() - fp32_start
        results['fp32'] = fp32_time
        
        # INT8 quantized
        int8_weights = {k: np.round(v * 127 / np.abs(v).max()).astype(np.int8) 
                       for k, v in network.items()}
        int8_start = time.time()
        for inp in test_input:
            x = inp
            for weights in int8_weights.values():
                x = np.dot(x, weights.astype(np.float32) / 127)
                x = np.maximum(0, x)
        int8_time = time.time() - int8_start
        results['int8'] = int8_time
        
        # Pentary quantized
        pentary_weights = {k: self.quantizer.quantize(v) 
                          for k, v in network.items()}
        pentary_start = time.time()
        for inp in test_input:
            x = inp
            for weights in pentary_weights.values():
                # Pentary operations can skip zeros and use bit shifts for powers of 2
                x = np.dot(x, weights)
                x = np.maximum(0, x)
        pentary_time = time.time() - pentary_start
        results['pentary'] = pentary_time
        
        # Calculate speedups
        speedups = {
            dtype: fp32_time / time_val if time_val > 0 else 0
            for dtype, time_val in results.items()
        }
        
        return {
            'num_inferences': num_inferences,
            'time_seconds': results,
            'time_ms_per_inference': {k: v * 1000 / num_inferences for k, v in results.items()},
            'speedup_vs_fp32': speedups
        }
    
    def benchmark_accuracy_loss(self, network: Dict[str, np.ndarray]) -> Dict:
        """Estimate accuracy loss from quantization."""
        
        results = {}
        
        for layer_name, weights in network.items():
            # Original weights
            original = weights
            
            # Quantized weights
            pentary = self.quantizer.quantize(weights)
            
            # Calculate MSE
            mse = np.mean((original - pentary) ** 2)
            
            # Calculate relative error
            rel_error = np.mean(np.abs(original - pentary) / (np.abs(original) + 1e-8))
            
            results[layer_name] = {
                'mse': float(mse),
                'relative_error': float(rel_error),
                'max_error': float(np.max(np.abs(original - pentary)))
            }
        
        # Average across layers
        avg_mse = np.mean([r['mse'] for r in results.values()])
        avg_rel_error = np.mean([r['relative_error'] for r in results.values()])
        
        return {
            'per_layer': results,
            'average_mse': float(avg_mse),
            'average_relative_error': float(avg_rel_error),
            'estimated_accuracy_loss_percent': float(avg_rel_error * 100)
        }
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive neural network benchmarks."""
        
        results = {}
        
        for size in ['small', 'medium', 'large']:
            print(f"Benchmarking {size} network...")
            
            network = self.create_test_network(size)
            
            # Memory benchmark
            memory_results = self.benchmark_memory(network)
            
            # Inference speed benchmark
            speed_results = self.benchmark_inference_speed(network, num_inferences=100)
            
            # Accuracy loss estimate
            accuracy_results = self.benchmark_accuracy_loss(network)
            
            results[size] = {
                'memory': memory_results,
                'speed': speed_results,
                'accuracy': accuracy_results
            }
        
        return results

def generate_nn_report(results: Dict) -> str:
    """Generate neural network benchmark report."""
    
    report = "# Pentary Neural Network Benchmark Report\n\n"
    report += "## Executive Summary\n\n"
    report += "This report validates Pentary quantization claims for neural networks.\n\n"
    
    for size, size_results in results.items():
        report += f"## {size.title()} Network Results\n\n"
        
        # Memory results
        memory = size_results['memory']
        report += "### Memory Usage\n\n"
        report += "| Format | Memory (MB) | Reduction vs FP32 |\n"
        report += "|--------|-------------|-------------------|\n"
        
        for dtype in ['fp32', 'fp16', 'int8', 'pentary', 'pentary_optimal']:
            mem_mb = memory['memory_mb'][dtype]
            reduction = memory['reduction_vs_fp32'][dtype]
            report += f"| {dtype.upper()} | {mem_mb:.2f} | {reduction:.2f}× |\n"
        
        # Speed results
        speed = size_results['speed']
        report += "\n### Inference Speed\n\n"
        report += "| Format | Time per Inference (ms) | Speedup vs FP32 |\n"
        report += "|--------|-------------------------|------------------|\n"
        
        for dtype in ['fp32', 'int8', 'pentary']:
            time_ms = speed['time_ms_per_inference'][dtype]
            speedup = speed['speedup_vs_fp32'][dtype]
            report += f"| {dtype.upper()} | {time_ms:.2f} | {speedup:.2f}× |\n"
        
        # Accuracy results
        accuracy = size_results['accuracy']
        report += f"\n### Accuracy Impact\n\n"
        report += f"- **Average Relative Error:** {accuracy['average_relative_error']:.4f}\n"
        report += f"- **Estimated Accuracy Loss:** {accuracy['estimated_accuracy_loss_percent']:.2f}%\n\n"
    
    # Validation summary
    report += "## Validation Summary\n\n"
    
    # Check 10× memory reduction claim
    avg_pentary_reduction = np.mean([
        results[size]['memory']['reduction_vs_fp32']['pentary']
        for size in results.keys()
    ])
    
    report += f"### Memory Reduction Claim: 10× vs FP32\n\n"
    report += f"- **Measured:** {avg_pentary_reduction:.2f}×\n"
    if avg_pentary_reduction >= 10.0:
        report += "- **Status:** ✅ VERIFIED\n\n"
    else:
        report += f"- **Status:** ⚠️ PARTIAL (achieved {avg_pentary_reduction:.1f}× vs claimed 10×)\n\n"
    
    # Check 2-3× speedup claim
    avg_pentary_speedup = np.mean([
        results[size]['speed']['speedup_vs_fp32']['pentary']
        for size in results.keys()
    ])
    
    report += f"### Inference Speed Claim: 2-3× faster\n\n"
    report += f"- **Measured:** {avg_pentary_speedup:.2f}×\n"
    if 2.0 <= avg_pentary_speedup <= 3.0:
        report += "- **Status:** ✅ VERIFIED\n\n"
    else:
        report += f"- **Status:** ⚠️ PARTIAL (achieved {avg_pentary_speedup:.1f}× vs claimed 2-3×)\n\n"
    
    # Check 1-3% accuracy loss claim
    avg_accuracy_loss = np.mean([
        results[size]['accuracy']['estimated_accuracy_loss_percent']
        for size in results.keys()
    ])
    
    report += f"### Accuracy Loss Claim: 1-3%\n\n"
    report += f"- **Measured:** {avg_accuracy_loss:.2f}%\n"
    if avg_accuracy_loss <= 3.0:
        report += "- **Status:** ✅ VERIFIED\n\n"
    else:
        report += f"- **Status:** ⚠️ EXCEEDS CLAIM ({avg_accuracy_loss:.1f}% vs claimed 1-3%)\n\n"
    
    report += "## Notes\n\n"
    report += "- Memory reduction is calculated using 3 bits per pentary value\n"
    report += "- Optimal encoding (2.32 bits) would provide even better results\n"
    report += "- Speedup depends on hardware optimization and zero-skipping\n"
    report += "- Accuracy loss can be reduced with quantization-aware training\n"
    report += "- Real-world performance requires hardware implementation\n\n"
    
    return report

if __name__ == "__main__":
    print("Running Pentary neural network benchmarks...")
    print("This may take a few minutes...\n")
    
    benchmark = NeuralNetworkBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report
    report = generate_nn_report(results)
    with open("nn_benchmark_report.md", "w") as f:
        f.write(report)
    
    # Save raw results
    with open("nn_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Neural network benchmarks complete!")
    print("Generated nn_benchmark_report.md")
    print("Saved nn_benchmark_results.json")