#!/usr/bin/env python3
"""
Pentary Neural Network Implementation
Implements neural network layers and operations optimized for pentary arithmetic
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pentary_converter import PentaryConverter
from pentary_arithmetic import PentaryArithmetic


class PentaryLayer:
    """Base class for pentary neural network layers"""
    
    def __init__(self, name: str = "Layer"):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.trainable = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        raise NotImplementedError
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass (gradient computation)"""
        raise NotImplementedError
        
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get layer parameters"""
        return {}
        
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set layer parameters"""
        pass


class PentaryLinear(PentaryLayer):
    """Fully connected layer with pentary weights"""
    
    def __init__(self, in_features: int, out_features: int, 
                 use_pentary: bool = True, name: str = "Linear"):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.use_pentary = use_pentary
        
        # Initialize weights in pentary range {-2, -1, 0, 1, 2}
        if use_pentary:
            # Xavier-like initialization scaled to pentary
            weights = np.random.randn(out_features, in_features) * 0.5
            self.weights = self._quantize_to_pentary(weights)
        else:
            self.weights = np.random.randn(out_features, in_features) * 0.1
            
        self.bias = np.zeros(out_features)
        self.input_cache = None
        
    def _quantize_to_pentary(self, x: np.ndarray) -> np.ndarray:
        """Quantize values to pentary levels {-2, -1, 0, 1, 2}"""
        # Scale to [-2, 2] range, then round to nearest pentary level
        x_scaled = np.clip(x, -2, 2)
        x_rounded = np.round(x_scaled)
        return np.clip(x_rounded, -2, 2).astype(np.int32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W^T + b
        
        For pentary weights, multiplication is simplified:
        - w = 0: skip (sparsity)
        - w = ±1: pass through or negate
        - w = ±2: shift and add
        """
        self.input_cache = x
        
        if self.use_pentary:
            # Optimized pentary matrix multiplication
            output = np.zeros((x.shape[0], self.out_features))
            
            for i in range(self.out_features):
                for j in range(self.in_features):
                    w = self.weights[i, j]
                    if w == 0:
                        continue  # Sparsity: skip zero weights
                    elif w == 1:
                        output[:, i] += x[:, j]
                    elif w == -1:
                        output[:, i] -= x[:, j]
                    elif w == 2:
                        output[:, i] += 2 * x[:, j]
                    elif w == -2:
                        output[:, i] -= 2 * x[:, j]
                        
            output += self.bias
        else:
            # Standard floating point
            output = x @ self.weights.T + self.bias
            
        return output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Backward pass: compute gradients"""
        if self.input_cache is None:
            raise ValueError("Must call forward before backward")
            
        # Gradient w.r.t. input
        grad_input = grad_output @ self.weights
        
        # Gradient w.r.t. weights
        grad_weights = grad_output.T @ self.input_cache
        
        # Gradient w.r.t. bias
        grad_bias = np.sum(grad_output, axis=0)
        
        return grad_input, grad_weights, grad_bias
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            'weights': self.weights,
            'bias': self.bias
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        if 'weights' in params:
            self.weights = params['weights']
        if 'bias' in params:
            self.bias = params['bias']


class PentaryReLU(PentaryLayer):
    """ReLU activation function"""
    
    def __init__(self, name: str = "ReLU"):
        super().__init__(name)
        self.input_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_cache is None:
            raise ValueError("Must call forward before backward")
        return grad_output * (self.input_cache > 0)


class PentaryQuantizedReLU(PentaryLayer):
    """Quantized ReLU with 5 output levels"""
    
    def __init__(self, name: str = "QuantizedReLU"):
        super().__init__(name)
        self.input_cache = None
        # Thresholds for 5-level quantization
        self.thresholds = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Quantize to pentary levels after ReLU"""
        self.input_cache = x
        relu_output = np.maximum(0, x)
        # Quantize to nearest pentary level
        quantized = np.clip(np.round(relu_output), 0, 2)
        return quantized.astype(np.int32)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_cache is None:
            raise ValueError("Must call forward before backward")
        # Straight-through estimator
        return grad_output * (self.input_cache > 0)


class PentaryConv2D(PentaryLayer):
    """2D Convolution layer with pentary weights"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 use_pentary: bool = True, name: str = "Conv2D"):
        super().__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_pentary = use_pentary
        
        # Initialize weights
        if use_pentary:
            weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.5
            self.weights = self._quantize_to_pentary(weights)
        else:
            self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
            
        self.bias = np.zeros(out_channels)
        self.input_cache = None
        
    def _quantize_to_pentary(self, x: np.ndarray) -> np.ndarray:
        """Quantize to pentary levels"""
        x_scaled = np.clip(x, -2, 2)
        x_rounded = np.round(x_scaled)
        return np.clip(x_rounded, -2, 2).astype(np.int32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Convolution forward pass
        x shape: (batch, in_channels, height, width)
        """
        self.input_cache = x
        batch_size, in_ch, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Add padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
            
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Convolution operation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        x_patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        if self.use_pentary:
                            # Optimized pentary convolution
                            result = 0
                            for ic in range(self.in_channels):
                                for kh in range(self.kernel_size):
                                    for kw in range(self.kernel_size):
                                        w = self.weights[oc, ic, kh, kw]
                                        if w == 0:
                                            continue
                                        elif w == 1:
                                            result += x_patch[ic, kh, kw]
                                        elif w == -1:
                                            result -= x_patch[ic, kh, kw]
                                        elif w == 2:
                                            result += 2 * x_patch[ic, kh, kw]
                                        elif w == -2:
                                            result -= 2 * x_patch[ic, kh, kw]
                            output[b, oc, oh, ow] = result + self.bias[oc]
                        else:
                            # Standard convolution
                            output[b, oc, oh, ow] = np.sum(x_patch * self.weights[oc, :, :, :]) + self.bias[oc]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass (simplified)"""
        # Simplified backward - full implementation would compute weight gradients
        return grad_output
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            'weights': self.weights,
            'bias': self.bias
        }


class PentaryMaxPool2D(PentaryLayer):
    """Max pooling layer"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2, name: str = "MaxPool2D"):
        super().__init__(name)
        self.pool_size = pool_size
        self.stride = stride
        self.input_cache = None
        self.max_indices = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Max pooling forward pass"""
        self.input_cache = x
        batch_size, channels, in_h, in_w = x.shape
        
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(pool_region)
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        
                        output[b, c, oh, ow] = max_val
                        self.max_indices[b, c, oh, ow] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for max pooling"""
        if self.input_cache is None:
            raise ValueError("Must call forward before backward")
            
        grad_input = np.zeros_like(self.input_cache)
        
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                for oh in range(grad_output.shape[2]):
                    for ow in range(grad_output.shape[3]):
                        h_idx, w_idx = self.max_indices[b, c, oh, ow]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, oh, ow]
        
        return grad_input


class PentaryNetwork:
    """Complete pentary neural network"""
    
    def __init__(self, layers: List[PentaryLayer]):
        self.layers = layers
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through all layers"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameters"""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            if layer_params:
                params[f'layer_{i}_{layer.name}'] = layer_params
        return params
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set all parameters"""
        for i, layer in enumerate(self.layers):
            key = f'layer_{i}_{layer.name}'
            if key in params:
                layer.set_parameters(params[key])
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters and sparsity"""
        total_params = 0
        zero_params = 0
        
        for layer in self.layers:
            layer_params = layer.get_parameters()
            if 'weights' in layer_params:
                weights = layer_params['weights']
                total_params += weights.size
                zero_params += np.sum(weights == 0)
        
        return {
            'total': total_params,
            'zero': zero_params,
            'non_zero': total_params - zero_params,
            'sparsity': zero_params / total_params if total_params > 0 else 0.0
        }


def create_simple_classifier(input_size: int, hidden_size: int, num_classes: int) -> PentaryNetwork:
    """Create a simple pentary classifier"""
    layers = [
        PentaryLinear(input_size, hidden_size, use_pentary=True),
        PentaryReLU(),
        PentaryQuantizedReLU(),
        PentaryLinear(hidden_size, num_classes, use_pentary=True)
    ]
    return PentaryNetwork(layers)


def create_simple_cnn(input_channels: int, num_classes: int) -> PentaryNetwork:
    """Create a simple pentary CNN"""
    layers = [
        PentaryConv2D(input_channels, 16, kernel_size=3, padding=1, use_pentary=True),
        PentaryReLU(),
        PentaryMaxPool2D(pool_size=2, stride=2),
        PentaryConv2D(16, 32, kernel_size=3, padding=1, use_pentary=True),
        PentaryReLU(),
        PentaryMaxPool2D(pool_size=2, stride=2),
        # Flatten (would need implementation)
        # PentaryLinear(32 * 7 * 7, num_classes, use_pentary=True)
    ]
    return PentaryNetwork(layers)


def main():
    """Demo and testing of pentary neural network"""
    print("=" * 70)
    print("Pentary Neural Network Implementation")
    print("=" * 70)
    print()
    
    # Test linear layer
    print("Testing PentaryLinear Layer:")
    print("-" * 70)
    
    batch_size = 4
    in_features = 10
    out_features = 5
    
    layer = PentaryLinear(in_features, out_features, use_pentary=True)
    x = np.random.randn(batch_size, in_features)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {layer.weights.shape}")
    print(f"Weight range: [{layer.weights.min()}, {layer.weights.max()}]")
    print(f"Zero weights: {np.sum(layer.weights == 0)} / {layer.weights.size} "
          f"({100 * np.sum(layer.weights == 0) / layer.weights.size:.1f}%)")
    
    output = layer.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    print()
    
    # Test network
    print("Testing Complete Network:")
    print("-" * 70)
    
    network = create_simple_classifier(input_size=784, hidden_size=128, num_classes=10)
    test_input = np.random.randn(32, 784)
    
    output = network.forward(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    params = network.count_parameters()
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Non-zero parameters: {params['non_zero']:,}")
    print(f"  Zero parameters: {params['zero']:,}")
    print(f"  Sparsity: {params['sparsity']:.1%}")
    print()
    
    # Test convolution
    print("Testing PentaryConv2D Layer:")
    print("-" * 70)
    
    conv = PentaryConv2D(in_channels=3, out_channels=16, kernel_size=3, 
                        padding=1, use_pentary=True)
    x_conv = np.random.randn(2, 3, 32, 32)
    
    print(f"Input shape: {x_conv.shape}")
    print(f"Weight shape: {conv.weights.shape}")
    print(f"Zero weights: {np.sum(conv.weights == 0)} / {conv.weights.size} "
          f"({100 * np.sum(conv.weights == 0) / conv.weights.size:.1f}%)")
    
    output_conv = conv.forward(x_conv)
    print(f"Output shape: {output_conv.shape}")
    print()
    
    print("=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
