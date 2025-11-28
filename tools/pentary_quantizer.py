#!/usr/bin/env python3
"""
Pentary Model Quantization Utilities
Converts standard neural network models to pentary format
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os


class ModelQuantizer:
    """Quantizes neural network models to pentary format"""
    
    def __init__(self, quantization_method: str = 'per_tensor'):
        """
        Initialize quantizer.
        
        Args:
            quantization_method: 'per_tensor' or 'per_channel'
        """
        self.quantization_method = quantization_method
    
    def quantize_layer_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize layer weights to pentary.
        
        Args:
            weights: Weight tensor
            
        Returns:
            (quantized_weights, scale_factor)
        """
        if self.quantization_method == 'per_tensor':
            # Single scale for entire tensor
            max_abs = np.max(np.abs(weights))
            scale = max_abs / 2.0 if max_abs > 0 else 1.0
            
            quantized = np.round(np.clip(weights / scale, -2, 2)).astype(np.int32)
            
            return quantized, scale
        
        elif self.quantization_method == 'per_channel':
            # Per-channel scaling (for conv layers)
            if len(weights.shape) == 4:  # Conv2D: (out_ch, in_ch, h, w)
                scales = []
                quantized = np.zeros_like(weights, dtype=np.int32)
                
                for i in range(weights.shape[0]):
                    channel_weights = weights[i]
                    max_abs = np.max(np.abs(channel_weights))
                    scale = max_abs / 2.0 if max_abs > 0 else 1.0
                    scales.append(scale)
                    
                    quantized[i] = np.round(np.clip(channel_weights / scale, -2, 2)).astype(np.int32)
                
                return quantized, np.array(scales)
            
            elif len(weights.shape) == 2:  # Linear: (out_features, in_features)
                scales = []
                quantized = np.zeros_like(weights, dtype=np.int32)
                
                for i in range(weights.shape[0]):
                    row_weights = weights[i]
                    max_abs = np.max(np.abs(row_weights))
                    scale = max_abs / 2.0 if max_abs > 0 else 1.0
                    scales.append(scale)
                    
                    quantized[i] = np.round(np.clip(row_weights / scale, -2, 2)).astype(np.int32)
                
                return quantized, np.array(scales)
            
            else:
                # Fallback to per-tensor
                return self.quantize_layer_weights(weights)
    
    def quantize_model_from_dict(self, model_dict: Dict) -> Dict:
        """
        Quantize a model from dictionary format.
        
        Args:
            model_dict: Model dictionary with weights
            
        Returns:
            Quantized model dictionary
        """
        quantized_model = {
            'architecture': model_dict.get('architecture', {}),
            'layers': []
        }
        
        for layer in model_dict.get('layers', []):
            quantized_layer = {'type': layer['type']}
            
            if 'weights' in layer:
                weights = np.array(layer['weights'])
                quantized_weights, scale = self.quantize_layer_weights(weights)
                
                quantized_layer['weights_pentary'] = quantized_weights.tolist()
                quantized_layer['weight_scale'] = float(scale) if isinstance(scale, (int, float)) else scale.tolist()
            
            if 'bias' in layer and layer['bias'] is not None:
                bias = np.array(layer['bias'])
                quantized_bias, bias_scale = self.quantize_layer_weights(bias)
                
                quantized_layer['bias_pentary'] = quantized_bias.tolist()
                quantized_layer['bias_scale'] = float(bias_scale) if isinstance(bias_scale, (int, float)) else bias_scale.tolist()
            
            # Copy other layer attributes
            for key in layer:
                if key not in ['weights', 'bias']:
                    quantized_layer[key] = layer[key]
            
            quantized_model['layers'].append(quantized_layer)
        
        return quantized_model
    
    def save_quantized_model(self, quantized_model: Dict, filepath: str):
        """Save quantized model to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(quantized_model, f, indent=2)
    
    def load_quantized_model(self, filepath: str) -> Dict:
        """Load quantized model from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class ONNXToPentaryConverter:
    """Converts ONNX models to pentary format"""
    
    def __init__(self):
        """Initialize converter"""
        self.quantizer = ModelQuantizer()
    
    def convert(self, onnx_model_path: str, output_path: str):
        """
        Convert ONNX model to pentary format.
        
        Args:
            onnx_model_path: Path to ONNX model file
            output_path: Output path for pentary model
        """
        try:
            import onnx
            import onnx.numpy_helper as nph
        except ImportError:
            raise ImportError("ONNX package required. Install with: pip install onnx")
        
        # Load ONNX model
        model = onnx.load(onnx_model_path)
        
        # Extract weights and architecture
        model_dict = {
            'architecture': {
                'input_shape': None,
                'output_shape': None
            },
            'layers': []
        }
        
        # Process graph
        for node in model.graph.node:
            layer_dict = {
                'type': node.op_type.lower(),
                'name': node.name
            }
            
            # Extract weights
            weights = []
            bias = None
            
            for input_name in node.input:
                for initializer in model.graph.initializer:
                    if initializer.name == input_name:
                        weight_array = nph.to_array(initializer)
                        
                        if len(weight_array.shape) >= 2:
                            weights.append(weight_array)
                        elif len(weight_array.shape) == 1:
                            bias = weight_array
            
            if weights:
                layer_dict['weights'] = weights[0].tolist()
                if len(weights) > 1:
                    layer_dict['weights'] = weights[0].tolist()
            
            if bias is not None:
                layer_dict['bias'] = bias.tolist()
            
            # Extract attributes
            for attr in node.attribute:
                layer_dict[attr.name] = self._onnx_attr_to_python(attr)
            
            model_dict['layers'].append(layer_dict)
        
        # Quantize model
        quantized_model = self.quantizer.quantize_model_from_dict(model_dict)
        
        # Save
        self.quantizer.save_quantized_model(quantized_model, output_path)
    
    def _onnx_attr_to_python(self, attr):
        """Convert ONNX attribute to Python value"""
        if attr.type == 1:  # FLOAT
            return attr.f
        elif attr.type == 2:  # INT
            return attr.i
        elif attr.type == 3:  # STRING
            return attr.s.decode('utf-8')
        elif attr.type == 4:  # TENSOR
            return nph.to_array(attr.t).tolist()
        elif attr.type == 5:  # GRAPH
            return None  # Skip graph attributes
        elif attr.type == 6:  # FLOATS
            return list(attr.floats)
        elif attr.type == 7:  # INTS
            return list(attr.ints)
        elif attr.type == 8:  # STRINGS
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return None


class PyTorchToPentaryConverter:
    """Converts PyTorch models to pentary format"""
    
    def __init__(self):
        """Initialize converter"""
        self.quantizer = ModelQuantizer()
    
    def convert(self, pytorch_model, output_path: str, input_shape: Tuple[int, ...] = None):
        """
        Convert PyTorch model to pentary format.
        
        Args:
            pytorch_model: PyTorch model instance
            output_path: Output path for pentary model
            input_shape: Input shape (optional)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        model_dict = {
            'architecture': {
                'input_shape': input_shape,
                'framework': 'pytorch'
            },
            'layers': []
        }
        
        # Extract layers
        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_dict = {
                    'type': 'linear',
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'weights': module.weight.detach().cpu().numpy().tolist()
                }
                
                if module.bias is not None:
                    layer_dict['bias'] = module.bias.detach().cpu().numpy().tolist()
                
                model_dict['layers'].append(layer_dict)
            
            elif isinstance(module, torch.nn.Conv2d):
                layer_dict = {
                    'type': 'conv2d',
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    'stride': module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                    'padding': module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                    'weights': module.weight.detach().cpu().numpy().tolist()
                }
                
                if module.bias is not None:
                    layer_dict['bias'] = module.bias.detach().cpu().numpy().tolist()
                
                model_dict['layers'].append(layer_dict)
        
        # Quantize model
        quantized_model = self.quantizer.quantize_model_from_dict(model_dict)
        
        # Save
        self.quantizer.save_quantized_model(quantized_model, output_path)


def main():
    """Test quantization utilities"""
    print("=" * 70)
    print("Pentary Model Quantization Test")
    print("=" * 70)
    
    # Test quantizer
    print("\n1. Testing Model Quantizer:")
    print("-" * 70)
    quantizer = ModelQuantizer(quantization_method='per_tensor')
    
    # Test linear layer weights
    linear_weights = np.random.randn(64, 128).astype(np.float32) * 0.1
    quantized, scale = quantizer.quantize_layer_weights(linear_weights)
    print(f"Linear weights shape: {linear_weights.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Scale: {scale:.4f}")
    print(f"Original range: [{linear_weights.min():.4f}, {linear_weights.max():.4f}]")
    print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")
    print(f"Unique values: {np.unique(quantized)}")
    
    # Test conv layer weights
    print("\n2. Testing Conv2D Quantization:")
    print("-" * 70)
    conv_weights = np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1
    quantized_conv, scale_conv = quantizer.quantize_layer_weights(conv_weights)
    print(f"Conv weights shape: {conv_weights.shape}")
    print(f"Quantized shape: {quantized_conv.shape}")
    print(f"Scale: {scale_conv:.4f}")
    print(f"Quantized range: [{quantized_conv.min()}, {quantized_conv.max()}]")
    
    # Test model quantization
    print("\n3. Testing Model Quantization:")
    print("-" * 70)
    dummy_model = {
        'architecture': {'input_dim': 128, 'output_dim': 10},
        'layers': [
            {
                'type': 'linear',
                'in_features': 128,
                'out_features': 64,
                'weights': np.random.randn(64, 128).astype(np.float32).tolist(),
                'bias': np.random.randn(64).astype(np.float32).tolist()
            },
            {
                'type': 'linear',
                'in_features': 64,
                'out_features': 10,
                'weights': np.random.randn(10, 64).astype(np.float32).tolist(),
                'bias': np.random.randn(10).astype(np.float32).tolist()
            }
        ]
    }
    
    quantized_model = quantizer.quantize_model_from_dict(dummy_model)
    print(f"Original layers: {len(dummy_model['layers'])}")
    print(f"Quantized layers: {len(quantized_model['layers'])}")
    print(f"First layer weight scale: {quantized_model['layers'][0]['weight_scale']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
