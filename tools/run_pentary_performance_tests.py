#!/usr/bin/env python3
"""
Comprehensive Pentary Performance Test Suite
Runs speed benchmarks and Gemma quantization tests
"""

import sys
import os
import time
import json
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pentary_speed_benchmark import PentarySpeedBenchmark
from pentary_gemma_quantizer import GemmaPentaryQuantizer, GemmaPentaryInference


class PentaryPerformanceTestSuite:
    """Comprehensive performance test suite"""

    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def run_speed_benchmarks(self):
        """Run speed benchmark suite"""
        print("\n" + "="*70)
        print("PHASE 1: SPEED BENCHMARKS")
        print("="*70)

        benchmark = PentarySpeedBenchmark()
        summary = benchmark.run_all_benchmarks()

        # Save results
        results_file = self.output_dir / "speed_benchmark_results.json"
        benchmark.save_results(str(results_file))

        self.results['speed_benchmarks'] = summary

        return summary

    def run_gemma_quantization(self, model_size: str = '2b'):
        """Run Gemma quantization test"""
        print("\n" + "="*70)
        print(f"PHASE 2: GEMMA {model_size.upper()} QUANTIZATION")
        print("="*70)

        quantizer = GemmaPentaryQuantizer(calibration_method='minmax')

        # Create dummy Gemma weights
        print(f"\nCreating dummy Gemma {model_size} model...")
        quantizer.create_dummy_gemma_weights(model_size=model_size)

        # Quantize
        print("\nQuantizing model to pentary format...")
        start_time = time.time()
        quantized_model = quantizer.quantize_gemma()
        quantization_time = time.time() - start_time

        # Analyze
        print("\nAnalyzing quantization impact...")
        analysis = quantizer.analyze_quantization_impact()

        # Save quantized model
        model_file = self.output_dir / f"gemma_{model_size}_pentary.json"
        quantizer.save_quantized_gemma(str(model_file))

        # Test inference speed
        print("\nTesting inference speed...")
        inference = GemmaPentaryInference(quantized_model)

        # Benchmark inference
        batch_sizes = [1, 4, 8, 16]
        seq_lengths = [32, 64, 128, 256]

        inference_results = []
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                test_input = np.random.randint(0, 1000, (batch_size, seq_len))

                # Warmup
                _ = inference.inference_step(test_input, max_length=10)

                # Benchmark
                times = []
                for _ in range(5):
                    start = time.time()
                    _ = inference.inference_step(test_input, max_length=10)
                    times.append(time.time() - start)

                avg_time = np.mean(times)
                inference_results.append({
                    'batch_size': batch_size,
                    'seq_length': seq_len,
                    'time_ms': avg_time * 1000,
                    'tokens_per_second': (batch_size * seq_len) / avg_time if avg_time > 0 else 0
                })

        # Save results
        gemma_results = {
            'model_size': model_size,
            'quantization_time': quantization_time,
            'quantization_stats': quantized_model['metadata']['global'],
            'model_info': quantized_model['metadata']['model_info'],
            'analysis': analysis,
            'inference_benchmarks': inference_results
        }

        results_file = self.output_dir / f"gemma_{model_size}_quantization_results.json"
        with open(results_file, 'w') as f:
            json.dump(gemma_results, f, indent=2)

        self.results[f'gemma_{model_size}'] = gemma_results

        # Print summary
        print(f"\n{'='*70}")
        print("Gemma Quantization Summary")
        print(f"{'='*70}")
        print(f"Model size:              {model_size}")
        print(f"Total parameters:       {quantized_model['metadata']['global']['total_parameters']:,}")
        print(f"Sparsity:               {quantized_model['metadata']['global']['global_sparsity']*100:.1f}%")
        print(f"Size reduction:         {quantized_model['metadata']['model_info']['size_reduction']:.2f}×")
        print(f"Quantization time:      {quantization_time:.2f} seconds")

        print(f"\nInference Performance (sample):")
        sample = inference_results[0]
        print(f"  Batch size:           {sample['batch_size']}")
        print(f"  Sequence length:      {sample['seq_length']}")
        print(f"  Time:                 {sample['time_ms']:.2f} ms")
        print(f"  Tokens/second:        {sample['tokens_per_second']:.0f}")

        return gemma_results

    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("GENERATING PERFORMANCE REPORT")
        print("="*70)

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'summary': {}
        }

        # Extract summary statistics
        if 'speed_benchmarks' in self.results:
            speed_summary = self.results['speed_benchmarks']['summary']
            report['summary']['avg_speedup'] = speed_summary['avg_speedup']
            report['summary']['avg_memory_reduction'] = speed_summary['avg_memory_reduction']
            report['summary']['avg_sparsity'] = speed_summary['avg_sparsity']

        if 'gemma_2b' in self.results:
            gemma = self.results['gemma_2b']
            report['summary']['gemma_parameters'] = gemma['quantization_stats']['total_parameters']
            report['summary']['gemma_sparsity'] = gemma['quantization_stats']['global_sparsity']
            report['summary']['gemma_size_reduction'] = gemma['model_info']['size_reduction']

        # Save report
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        self.generate_markdown_report(report)

        print(f"\nReport saved to {report_file}")
        return report

    def generate_markdown_report(self, report: dict):
        """Generate markdown performance report"""
        report_file = self.output_dir / "PERFORMANCE_REPORT.md"

        with open(report_file, 'w') as f:
            f.write("# Pentary System Performance Report\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")

            f.write("## Executive Summary\n\n")
            if 'summary' in report:
                summary = report['summary']
                if 'avg_speedup' in summary:
                    f.write(f"- **Average Speedup**: {summary['avg_speedup']:.2f}×\n")
                if 'avg_memory_reduction' in summary:
                    f.write(f"- **Average Memory Reduction**: {summary['avg_memory_reduction']:.2f}×\n")
                if 'avg_sparsity' in summary:
                    f.write(f"- **Average Sparsity**: {summary['avg_sparsity']*100:.1f}%\n")
                if 'gemma_size_reduction' in summary:
                    f.write(f"- **Gemma Model Size Reduction**: {summary['gemma_size_reduction']:.2f}×\n")

            f.write("\n## Speed Benchmarks\n\n")
            if 'speed_benchmarks' in report['results']:
                f.write("| Benchmark | Speedup | Memory Reduction | Sparsity |\n")
                f.write("|-----------|---------|------------------|----------|\n")
                for result in report['results']['speed_benchmarks']['results']:
                    f.write(f"| {result['name']} | {result['speedup']:.2f}× | "
                           f"{result['memory_reduction']:.2f}× | {result['sparsity']*100:.1f}% |\n")

            f.write("\n## Gemma Quantization\n\n")
            if 'gemma_2b' in report['results']:
                gemma = report['results']['gemma_2b']
                f.write(f"### Model Statistics\n\n")
                f.write(f"- **Total Parameters**: {gemma['quantization_stats']['total_parameters']:,}\n")
                f.write(f"- **Sparsity**: {gemma['quantization_stats']['global_sparsity']*100:.1f}%\n")
                f.write(f"- **Size Reduction**: {gemma['model_info']['size_reduction']:.2f}×\n")
                f.write(f"- **Quantization Time**: {gemma['quantization_time']:.2f} seconds\n")

                if 'inference_benchmarks' in gemma:
                    f.write(f"\n### Inference Performance\n\n")
                    f.write("| Batch Size | Seq Length | Time (ms) | Tokens/sec |\n")
                    f.write("|------------|------------|-----------|------------|\n")
                    for bench in gemma['inference_benchmarks'][:10]:  # Show first 10
                        f.write(f"| {bench['batch_size']} | {bench['seq_length']} | "
                               f"{bench['time_ms']:.2f} | {bench['tokens_per_second']:.0f} |\n")

        print(f"Markdown report saved to {report_file}")

    def run_all_tests(self):
        """Run all performance tests"""
        print("="*70)
        print("PENTARY SYSTEM PERFORMANCE TEST SUITE")
        print("="*70)
        print(f"Output directory: {self.output_dir}")

        start_time = time.time()

        # Phase 1: Speed benchmarks
        self.run_speed_benchmarks()

        # Phase 2: Gemma quantization
        self.run_gemma_quantization(model_size='2b')

        # Generate report
        self.generate_report()

        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Pentary Performance Tests')
    parser.add_argument('--output-dir', type=str, default='performance_results',
                       help='Output directory for results')
    parser.add_argument('--speed-only', action='store_true',
                       help='Run only speed benchmarks')
    parser.add_argument('--gemma-only', action='store_true',
                       help='Run only Gemma quantization')

    args = parser.parse_args()

    suite = PentaryPerformanceTestSuite(output_dir=args.output_dir)

    if args.speed_only:
        suite.run_speed_benchmarks()
        suite.generate_report()
    elif args.gemma_only:
        suite.run_gemma_quantization()
        suite.generate_report()
    else:
        suite.run_all_tests()


if __name__ == "__main__":
    import numpy as np
    main()
