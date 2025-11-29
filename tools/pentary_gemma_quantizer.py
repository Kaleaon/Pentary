#!/usr/bin/env python3
"""
Pentary Quantization for Gemma Models
Quantizes Google's Gemma models to pentary format for efficient inference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from collections import defaultdict
import time

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features may be limited.")

try:
    from pentary_quantizer import PentaryQuantizer, PentaryCalibrator
except ImportError:
    # Create minimal stubs if not available
    class PentaryQuantizer:
        def __init__(self, *args, **kwargs):
            pass
    class PentaryCalibrator:
        def __init__(self, *args, **kwargs):
            pass


class GemmaPentaryQuantizer:
    """Quantizes Gemma models to pentary format"""

    def __init__(self, calibration_method: str = 'minmax'):
        """
        Initialize Gemma quantizer.

        Args:
            calibration_method: 'minmax', 'percentile', or 'kl_divergence'
        """
        self.quantizer = PentaryQuantizer(calibration_method=calibration_method)
        self.calibrator = PentaryCalibrator(self.quantizer)
        self.model_weights = {}
        self.quantized_model = None

    def load_gemma_weights(self, model_path: Optional[str] = None,
                          model: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Load Gemma model weights.

        Args:
            model_path: Path to model checkpoint or directory
            model: PyTorch model instance (alternative)

        Returns:
            Dictionary mapping layer names to weight tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Gemma model loading")

        weights = {}

        if model is not None:
            # Extract weights from PyTorch model
            for name, param in model.named_parameters():
                if param.requires_grad and 'weight' in name.lower():
                    weights[name] = param.detach().cpu().numpy()
        elif model_path:
            # Try to load from checkpoint
            if os.path.isdir(model_path):
                # HuggingFace format
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_path)
                    for name, param in model.named_parameters():
                        if param.requires_grad and 'weight' in name.lower():
                            weights[name] = param.detach().cpu().numpy()
                except ImportError:
                    raise ImportError("transformers library required for HuggingFace models")
            else:
                # PyTorch checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                for name, tensor in state_dict.items():
                    if 'weight' in name.lower() and len(tensor.shape) >= 2:
                        weights[name] = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model_weights = weights
        return weights

    def create_dummy_gemma_weights(self, model_size: str = '2b') -> Dict[str, np.ndarray]:
        """
        Create dummy Gemma weights for testing (when model not available).

        Args:
            model_size: '2b' or '7b'

        Returns:
            Dictionary with dummy weights matching Gemma architecture
        """
        print(f"Creating dummy Gemma {model_size} weights for testing...")

        # Gemma architecture parameters
        if model_size == '2b':
            hidden_size = 2048
            num_layers = 18
            num_heads = 8
            intermediate_size = 8192
        else:  # 7b
            hidden_size = 3072
            num_layers = 28
            num_heads = 16
            intermediate_size = 12288

        weights = {}

        # Embedding layer
        vocab_size = 256000  # Gemma vocab size
        weights['embed_tokens.weight'] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

        # Transformer layers
        for i in range(num_layers):
            prefix = f'layers.{i}'

            # Self-attention
            weights[f'{prefix}.self_attn.q_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            weights[f'{prefix}.self_attn.k_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            weights[f'{prefix}.self_attn.v_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            weights[f'{prefix}.self_attn.o_proj.weight'] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

            # MLP
            weights[f'{prefix}.mlp.gate_proj.weight'] = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
            weights[f'{prefix}.mlp.up_proj.weight'] = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02
            weights[f'{prefix}.mlp.down_proj.weight'] = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02

            # Layer norms (not quantized, but included for completeness)
            weights[f'{prefix}.input_layernorm.weight'] = np.ones(hidden_size, dtype=np.float32)
            weights[f'{prefix}.post_attention_layernorm.weight'] = np.ones(hidden_size, dtype=np.float32)

        # Output layer
        weights['norm.weight'] = np.ones(hidden_size, dtype=np.float32)
        weights['lm_head.weight'] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

        self.model_weights = weights
        print(f"Created {len(weights)} weight tensors")
        return weights

    def quantize_gemma(self, calibration_data: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Quantize Gemma model to pentary format.

        Args:
            calibration_data: Optional calibration data for per-channel quantization

        Returns:
            Quantized model dictionary
        """
        if not self.model_weights:
            raise ValueError("No model weights loaded. Call load_gemma_weights() first.")

        print("="*70)
        print("Quantizing Gemma Model to Pentary Format")
        print("="*70)

        start_time = time.time()

        # Quantize all weight layers
        quantized_model = self.quantizer.quantize_model(
            self.model_weights,
            calibration_data=calibration_data
        )

        quantization_time = time.time() - start_time

        # Calculate statistics
        total_params = quantized_model['metadata']['global']['total_parameters']
        zero_params = quantized_model['metadata']['global']['zero_parameters']
        sparsity = quantized_model['metadata']['global']['global_sparsity']

        # Calculate model size
        original_size = sum(w.nbytes for w in self.model_weights.values())
        quantized_size_bits = total_params * 3  # 3 bits per pentary value
        quantized_size_bytes = quantized_size_bits / 8
        size_reduction = original_size / quantized_size_bytes if quantized_size_bytes > 0 else 0

        print(f"\nQuantization Statistics:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Zero parameters:     {zero_params:,}")
        print(f"  Sparsity:            {sparsity*100:.1f}%")
        print(f"  Original size:       {original_size / (1024**2):.2f} MB")
        print(f"  Quantized size:      {quantized_size_bytes / (1024**2):.2f} MB")
        print(f"  Size reduction:      {size_reduction:.2f}×")
        print(f"  Quantization time:   {quantization_time:.2f} seconds")

        # Add metadata
        quantized_model['metadata']['model_info'] = {
            'original_size_bytes': int(original_size),
            'quantized_size_bytes': int(quantized_size_bytes),
            'size_reduction': float(size_reduction),
            'quantization_time': float(quantization_time)
        }

        self.quantized_model = quantized_model
        return quantized_model

    def save_quantized_gemma(self, filepath: str):
        """Save quantized Gemma model to file"""
        if self.quantized_model is None:
            raise ValueError("No quantized model. Call quantize_gemma() first.")

        # Convert numpy arrays to lists for JSON serialization
        save_dict = {
            'weights': {k: v.tolist() for k, v in self.quantized_model['weights'].items()},
            'scales': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in self.quantized_model['scales'].items()},
            'zero_points': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.quantized_model['zero_points'].items()},
            'metadata': self.quantized_model['metadata']
        }

        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\nQuantized model saved to {filepath}")
        file_size = os.path.getsize(filepath) / (1024**2)
        print(f"File size: {file_size:.2f} MB")

    def load_quantized_gemma(self, filepath: str) -> Dict[str, Any]:
        """Load quantized Gemma model from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        quantized_model = {
            'weights': {k: np.array(v, dtype=np.int32) for k, v in data['weights'].items()},
            'scales': {k: (np.array(v) if isinstance(v, list) else v)
                      for k, v in data['scales'].items()},
            'zero_points': {k: (np.array(v) if isinstance(v, list) else v)
                          for k, v in data['zero_points'].items()},
            'metadata': data['metadata']
        }

        self.quantized_model = quantized_model
        return quantized_model

    def analyze_quantization_impact(self) -> Dict[str, Any]:
        """Analyze the impact of quantization on model weights"""
        if not self.model_weights or not self.quantized_model:
            raise ValueError("Both original and quantized models required")

        print("\n" + "="*70)
        print("Quantization Impact Analysis")
        print("="*70)

        analysis = {
            'layers': {},
            'global': {}
        }

        total_mse = 0.0
        total_mae = 0.0
        total_params = 0

        for layer_name in self.model_weights.keys():
            if layer_name not in self.quantized_model['weights']:
                continue

            original = self.model_weights[layer_name]
            quantized = self.quantized_model['weights'][layer_name]
            scale = self.quantized_model['scales'][layer_name]
            zero_point = self.quantized_model['zero_points'][layer_name]

            # Dequantize
            dequantized = quantized.astype(np.float32) * scale + zero_point

            # Compute errors
            mse = np.mean((original - dequantized) ** 2)
            mae = np.mean(np.abs(original - dequantized))
            max_error = np.max(np.abs(original - dequantized))

            # Relative error
            relative_error = np.mean(np.abs(original - dequantized) / (np.abs(original) + 1e-8))

            # SNR
            signal_power = np.var(original)
            snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

            layer_params = original.size
            total_mse += mse * layer_params
            total_mae += mae * layer_params
            total_params += layer_params

            analysis['layers'][layer_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'relative_error': float(relative_error),
                'snr_db': float(snr_db),
                'parameters': int(layer_params)
            }

        analysis['global'] = {
            'avg_mse': float(total_mse / total_params) if total_params > 0 else 0.0,
            'avg_mae': float(total_mae / total_params) if total_params > 0 else 0.0,
            'total_parameters': int(total_params)
        }

        print(f"\nGlobal Statistics:")
        print(f"  Average MSE:  {analysis['global']['avg_mse']:.6f}")
        print(f"  Average MAE:  {analysis['global']['avg_mae']:.6f}")

        print(f"\nTop 10 Layers by Parameter Count:")
        sorted_layers = sorted(
            analysis['layers'].items(),
            key=lambda x: x[1]['parameters'],
            reverse=True
        )[:10]

        for layer_name, stats in sorted_layers:
            print(f"  {layer_name:<50} {stats['parameters']:>12,} params  "
                  f"MSE: {stats['mse']:.6f}  SNR: {stats['snr_db']:.1f} dB")

        return analysis


class GemmaPentaryInference:
    """Inference engine for quantized Gemma models"""

    def __init__(self, quantized_model: Dict[str, Any]):
        """
        Initialize inference engine.

        Args:
            quantized_model: Quantized model dictionary
        """
        self.quantized_model = quantized_model
        self.weights = quantized_model['weights']
        self.scales = quantized_model['scales']
        self.zero_points = quantized_model['zero_points']

    def forward_linear(self, x: np.ndarray, weight_name: str) -> np.ndarray:
        """
        Forward pass through a quantized linear layer.

        Args:
            x: Input tensor
            weight_name: Name of weight tensor

        Returns:
            Output tensor
        """
        if weight_name not in self.weights:
            raise ValueError(f"Weight {weight_name} not found")

        w_quantized = self.weights[weight_name]
        scale = self.scales[weight_name]
        zero_point = self.zero_points[weight_name]

        # Optimized pentary matrix multiplication
        if len(w_quantized.shape) == 2:
            # Standard matrix multiplication: x @ w^T
            out_features, in_features = w_quantized.shape
            batch_size = x.shape[0]

            output = np.zeros((batch_size, out_features), dtype=np.float32)

            for i in range(out_features):
                for j in range(in_features):
                    w = w_quantized[i, j]
                    if w == 0:
                        continue  # Skip zero weights
                    elif w == 1:
                        output[:, i] += x[:, j] * scale
                    elif w == -1:
                        output[:, i] -= x[:, j] * scale
                    elif w == 2:
                        output[:, i] += 2 * x[:, j] * scale
                    elif w == -2:
                        output[:, i] -= 2 * x[:, j] * scale

            return output
        else:
            # Fallback to standard multiplication
            w_dequantized = w_quantized.astype(np.float32) * scale + zero_point
            return x @ w_dequantized.T

    def inference_step(self, input_ids: np.ndarray, max_length: int = 100) -> np.ndarray:
        """
        Perform a single inference step (simplified).

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length

        Returns:
            Generated token IDs
        """
        # Simplified inference - just demonstrate the concept
        # Full implementation would include attention, layer norm, etc.

        # Embedding lookup
        if 'embed_tokens.weight' in self.weights:
            embeddings = self.forward_linear(
                input_ids.astype(np.float32),
                'embed_tokens.weight'
            )
        else:
            embeddings = input_ids.astype(np.float32)

        # Process through transformer layers (simplified)
        hidden = embeddings
        for i in range(18):  # Gemma 2B has 18 layers
            layer_prefix = f'layers.{i}'

            # Self-attention (simplified - just use q_proj)
            if f'{layer_prefix}.self_attn.q_proj.weight' in self.weights:
                hidden = self.forward_linear(hidden, f'{layer_prefix}.self_attn.q_proj.weight')

            # MLP (simplified - just use gate_proj)
            if f'{layer_prefix}.mlp.gate_proj.weight' in self.weights:
                hidden = self.forward_linear(hidden, f'{layer_prefix}.mlp.gate_proj.weight')

        # Output projection
        if 'lm_head.weight' in self.weights:
            logits = self.forward_linear(hidden, 'lm_head.weight')
        else:
            logits = hidden

        return logits


def main():
    """Test Gemma quantization"""
    print("="*70)
    print("Gemma Pentary Quantization Test")
    print("="*70)

    # Create quantizer
    quantizer = GemmaPentaryQuantizer(calibration_method='minmax')

    # Create dummy Gemma 2B weights
    print("\n1. Creating dummy Gemma 2B model...")
    quantizer.create_dummy_gemma_weights(model_size='2b')

    # Quantize
    print("\n2. Quantizing model...")
    quantized_model = quantizer.quantize_gemma()

    # Analyze
    print("\n3. Analyzing quantization impact...")
    analysis = quantizer.analyze_quantization_impact()

    # Save
    print("\n4. Saving quantized model...")
    quantizer.save_quantized_gemma('gemma_2b_pentary.json')

    # Test inference
    print("\n5. Testing inference...")
    inference = GemmaPentaryInference(quantized_model)
    test_input = np.random.randint(0, 1000, (1, 10))  # Batch=1, seq_len=10
    start = time.time()
    output = inference.inference_step(test_input)
    inference_time = time.time() - start

    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
