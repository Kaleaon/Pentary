#!/usr/bin/env python3
"""
Pentary Quantization Tools
Quantizes neural network models to pentary representation {-2, -1, 0, 1, 2}
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from collections import defaultdict


class PentaryQuantizer:
    """Quantizes floating-point weights to pentary levels"""
    
    def __init__(self, calibration_method: str = 'minmax'):
        """
        Initialize quantizer
        
        Args:
            calibration_method: 'minmax', 'percentile', or 'kl_divergence'
        """
        self.calibration_method = calibration_method
        self.scale_factors = {}
        self.zero_points = {}
        self.quantization_stats = {}
        
    def quantize_tensor(self, tensor: np.ndarray, 
                       scale: Optional[float] = None,
                       zero_point: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
        """
        Quantize a tensor to pentary levels {-2, -1, 0, 1, 2}
        
        Args:
            tensor: Input tensor (float)
            scale: Optional pre-computed scale factor
            zero_point: Optional pre-computed zero point
            
        Returns:
            (quantized_tensor, scale, zero_point)
        """
        if scale is None or zero_point is None:
            # Compute scale and zero point
            if self.calibration_method == 'minmax':
                scale, zero_point = self._compute_minmax_scale(tensor)
            elif self.calibration_method == 'percentile':
                scale, zero_point = self._compute_percentile_scale(tensor)
            else:
                scale, zero_point = self._compute_minmax_scale(tensor)
        
        # Quantize: q = round((x - zero_point) / scale)
        quantized = np.round((tensor - zero_point) / scale)
        
        # Clip to pentary range
        quantized = np.clip(quantized, -2, 2).astype(np.int32)
        
        return quantized, scale, zero_point
    
    def dequantize_tensor(self, quantized: np.ndarray, 
                         scale: float, zero_point: float) -> np.ndarray:
        """
        Dequantize a pentary tensor back to float
        
        Args:
            quantized: Quantized tensor (int32, values in {-2, -1, 0, 1, 2})
            scale: Scale factor
            zero_point: Zero point
            
        Returns:
            Dequantized tensor (float)
        """
        return quantized.astype(np.float32) * scale + zero_point
    
    def _compute_minmax_scale(self, tensor: np.ndarray) -> Tuple[float, float]:
        """Compute scale using min-max method"""
        t_min = np.min(tensor)
        t_max = np.max(tensor)
        
        # Scale to fit in [-2, 2] range
        scale = (t_max - t_min) / 4.0 if (t_max - t_min) > 0 else 1.0
        zero_point = t_min
        
        return scale, zero_point
    
    def _compute_percentile_scale(self, tensor: np.ndarray, 
                                  percentile: float = 99.9) -> Tuple[float, float]:
        """Compute scale using percentile method (more robust to outliers)"""
        t_min = np.percentile(tensor, 100 - percentile)
        t_max = np.percentile(tensor, percentile)
        
        scale = (t_max - t_min) / 4.0 if (t_max - t_min) > 0 else 1.0
        zero_point = t_min
        
        return scale, zero_point
    
    def quantize_model(self, model_weights: Dict[str, np.ndarray],
                      calibration_data: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Quantize an entire model
        
        Args:
            model_weights: Dictionary mapping layer names to weight tensors
            calibration_data: Optional calibration data for per-channel quantization
            
        Returns:
            Dictionary with quantized weights, scales, and zero points
        """
        quantized_model = {
            'weights': {},
            'scales': {},
            'zero_points': {},
            'metadata': {}
        }
        
        total_params = 0
        total_zero = 0
        total_nonzero = 0
        
        for layer_name, weights in model_weights.items():
            print(f"Quantizing layer: {layer_name}")
            
            # Determine quantization strategy
            if calibration_data and layer_name in calibration_data:
                # Per-channel quantization for better accuracy
                quantized, scales, zero_points = self._quantize_per_channel(
                    weights, calibration_data[layer_name]
                )
                quantized_model['scales'][layer_name] = scales
                quantized_model['zero_points'][layer_name] = zero_points
            else:
                # Per-tensor quantization
                quantized, scale, zero_point = self.quantize_tensor(weights)
                quantized_model['scales'][layer_name] = scale
                quantized_model['zero_points'][layer_name] = zero_point
            
            quantized_model['weights'][layer_name] = quantized
            
            # Statistics
            layer_params = quantized.size
            layer_zero = np.sum(quantized == 0)
            layer_nonzero = layer_params - layer_zero
            
            total_params += layer_params
            total_zero += layer_zero
            total_nonzero += layer_nonzero
            
            quantized_model['metadata'][layer_name] = {
                'shape': list(weights.shape),
                'sparsity': float(layer_zero / layer_params),
                'non_zero_count': int(layer_nonzero),
                'value_distribution': {
                    '-2': int(np.sum(quantized == -2)),
                    '-1': int(np.sum(quantized == -1)),
                    '0': int(np.sum(quantized == 0)),
                    '1': int(np.sum(quantized == 1)),
                    '2': int(np.sum(quantized == 2))
                }
            }
        
        quantized_model['metadata']['global'] = {
            'total_parameters': int(total_params),
            'zero_parameters': int(total_zero),
            'non_zero_parameters': int(total_nonzero),
            'global_sparsity': float(total_zero / total_params) if total_params > 0 else 0.0
        }
        
        return quantized_model
    
    def _quantize_per_channel(self, weights: np.ndarray, 
                              calibration_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize with per-channel scales (for better accuracy)"""
        # For now, use per-tensor (can be extended to per-channel)
        quantized, scale, zero_point = self.quantize_tensor(weights)
        scales = np.full(weights.shape[0] if len(weights.shape) > 1 else 1, scale)
        zero_points = np.full(weights.shape[0] if len(weights.shape) > 1 else 1, zero_point)
        return quantized, scales, zero_points
    
    def analyze_quantization_error(self, original: np.ndarray, 
                                  quantized: np.ndarray,
                                  scale: float, zero_point: float) -> Dict[str, float]:
        """
        Analyze quantization error
        
        Returns:
            Dictionary with error metrics
        """
        dequantized = self.dequantize_tensor(quantized, scale, zero_point)
        
        mse = np.mean((original - dequantized) ** 2)
        mae = np.mean(np.abs(original - dequantized))
        max_error = np.max(np.abs(original - dequantized))
        
        # Relative error
        relative_error = np.mean(np.abs(original - dequantized) / (np.abs(original) + 1e-8))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'relative_error': float(relative_error),
            'snr_db': float(10 * np.log10(np.var(original) / mse)) if mse > 0 else float('inf')
        }


class PentaryCalibrator:
    """Calibrates quantization parameters using representative data"""
    
    def __init__(self, quantizer: PentaryQuantizer):
        self.quantizer = quantizer
        self.activation_stats = defaultdict(list)
        
    def collect_activations(self, layer_name: str, activations: np.ndarray):
        """Collect activation statistics for calibration"""
        self.activation_stats[layer_name].append(activations)
    
    def compute_optimal_scales(self) -> Dict[str, Tuple[float, float]]:
        """Compute optimal scale factors based on collected activations"""
        optimal_scales = {}
        
        for layer_name, activation_list in self.activation_stats.items():
            # Concatenate all activations
            all_activations = np.concatenate(activation_list, axis=0)
            
            # Compute scale using percentile method (more robust)
            scale, zero_point = self.quantizer._compute_percentile_scale(all_activations)
            optimal_scales[layer_name] = (scale, zero_point)
        
        return optimal_scales


class PentaryAccuracyAnalyzer:
    """Analyzes accuracy impact of quantization"""
    
    def __init__(self):
        self.results = {}
        
    def compare_models(self, original_model: Callable, 
                      quantized_model: Callable,
                      test_data: Tuple[np.ndarray, np.ndarray],
                      metric: str = 'accuracy') -> Dict[str, float]:
        """
        Compare original and quantized model accuracy
        
        Args:
            original_model: Function that takes input and returns predictions
            quantized_model: Function that takes input and returns quantized predictions
            test_data: (x_test, y_test) tuple
            metric: 'accuracy', 'top5_accuracy', or 'mse'
            
        Returns:
            Dictionary with accuracy metrics
        """
        x_test, y_test = test_data
        
        # Get predictions
        original_preds = original_model(x_test)
        quantized_preds = quantized_model(x_test)
        
        if metric == 'accuracy':
            orig_acc = self._compute_accuracy(original_preds, y_test)
            quant_acc = self._compute_accuracy(quantized_preds, y_test)
        elif metric == 'top5_accuracy':
            orig_acc = self._compute_top5_accuracy(original_preds, y_test)
            quant_acc = self._compute_top5_accuracy(quantized_preds, y_test)
        else:  # mse
            orig_acc = np.mean((original_preds - y_test) ** 2)
            quant_acc = np.mean((quantized_preds - y_test) ** 2)
        
        results = {
            'original_' + metric: float(orig_acc),
            'quantized_' + metric: float(quant_acc),
            'accuracy_drop': float(orig_acc - quant_acc),
            'relative_drop': float((orig_acc - quant_acc) / orig_acc * 100) if orig_acc > 0 else 0.0
        }
        
        self.results = results
        return results
    
    def _compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy"""
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        return np.mean(pred_classes == labels)
    
    def _compute_top5_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute top-5 accuracy"""
        top5_preds = np.argsort(predictions, axis=1)[:, -5:]
        return np.mean([labels[i] in top5_preds[i] for i in range(len(labels))])


def save_quantized_model(quantized_model: Dict[str, Any], filepath: str):
    """Save quantized model to file"""
    # Convert numpy arrays to lists for JSON serialization
    save_dict = {
        'weights': {k: v.tolist() for k, v in quantized_model['weights'].items()},
        'scales': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                  for k, v in quantized_model['scales'].items()},
        'zero_points': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                       for k, v in quantized_model['zero_points'].items()},
        'metadata': quantized_model['metadata']
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_dict, f, indent=2)


def load_quantized_model(filepath: str) -> Dict[str, Any]:
    """Load quantized model from file"""
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
    
    return quantized_model


def main():
    """Demo and testing of pentary quantizer"""
    print("=" * 70)
    print("Pentary Quantization Tools")
    print("=" * 70)
    print()
    
    # Create sample model weights
    print("Creating sample model weights...")
    model_weights = {
        'layer1.weight': np.random.randn(128, 784) * 0.1,
        'layer1.bias': np.random.randn(128) * 0.01,
        'layer2.weight': np.random.randn(64, 128) * 0.1,
        'layer2.bias': np.random.randn(64) * 0.01,
        'layer3.weight': np.random.randn(10, 64) * 0.1,
        'layer3.bias': np.random.randn(10) * 0.01
    }
    
    # Quantize model
    print("\nQuantizing model...")
    quantizer = PentaryQuantizer(calibration_method='minmax')
    quantized_model = quantizer.quantize_model(model_weights)
    
    # Print statistics
    print("\nQuantization Statistics:")
    print("-" * 70)
    global_stats = quantized_model['metadata']['global']
    print(f"Total parameters: {global_stats['total_parameters']:,}")
    print(f"Non-zero parameters: {global_stats['non_zero_parameters']:,}")
    print(f"Zero parameters: {global_stats['zero_parameters']:,}")
    print(f"Global sparsity: {global_stats['global_sparsity']:.1%}")
    print()
    
    print("Per-layer statistics:")
    for layer_name, stats in quantized_model['metadata'].items():
        if layer_name == 'global':
            continue
        print(f"\n  {layer_name}:")
        print(f"    Shape: {stats['shape']}")
        print(f"    Sparsity: {stats['sparsity']:.1%}")
        print(f"    Value distribution:")
        for val, count in stats['value_distribution'].items():
            print(f"      {val}: {count:,} ({count/stats['non_zero_count']*100:.1f}%)" 
                  if stats['non_zero_count'] > 0 else f"      {val}: {count:,}")
    
    # Analyze quantization error
    print("\n" + "=" * 70)
    print("Quantization Error Analysis:")
    print("-" * 70)
    
    layer_name = 'layer1.weight'
    original = model_weights[layer_name]
    quantized = quantized_model['weights'][layer_name]
    scale = quantized_model['scales'][layer_name]
    zero_point = quantized_model['zero_points'][layer_name]
    
    error_stats = quantizer.analyze_quantization_error(original, quantized, scale, zero_point)
    
    print(f"Layer: {layer_name}")
    print(f"  MSE: {error_stats['mse']:.6f}")
    print(f"  MAE: {error_stats['mae']:.6f}")
    print(f"  Max Error: {error_stats['max_error']:.6f}")
    print(f"  Relative Error: {error_stats['relative_error']:.2%}")
    print(f"  SNR: {error_stats['snr_db']:.2f} dB")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
