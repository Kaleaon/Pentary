#!/usr/bin/env python3
"""
Pentary Neural Network Layers
Implements neural network layers using pentary arithmetic
"""

import numpy as np
from typing import Tuple, Optional, List


class PentaryQuantizer:
    """Quantizes floating point values to pentary {-2, -1, 0, +1, +2}"""
    
    @staticmethod
    def quantize_to_pentary(value: float, scale: float = 1.0) -> int:
        """
        Quantize a floating point value to pentary digit.
        
        Args:
            value: Floating point value
            scale: Scaling factor (typically max(abs(weights)) / 2)
            
        Returns:
            Pentary digit in {-2, -1, 0, 1, 2}
        """
        if scale == 0:
            return 0
        
        # Normalize and quantize
        normalized = value / scale
        
        # Clamp to [-2, 2] range
        clamped = np.clip(normalized, -2, 2)
        
        # Round to nearest integer
        quantized = int(np.round(clamped))
        
        return quantized
    
    @staticmethod
    def quantize_tensor(tensor: np.ndarray, scale: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Quantize a tensor to pentary values.
        
        Args:
            tensor: Input tensor (numpy array)
            scale: Optional scale factor. If None, computed from tensor.
            
        Returns:
            (quantized_tensor, scale_factor)
        """
        if scale is None:
            # Compute scale from tensor
            max_abs = np.max(np.abs(tensor))
            scale = max_abs / 2.0 if max_abs > 0 else 1.0
        
        quantized = np.round(np.clip(tensor / scale, -2, 2)).astype(np.int32)
        
        return quantized, scale
    
    @staticmethod
    def dequantize_tensor(quantized: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize a pentary tensor back to floating point.
        
        Args:
            quantized: Pentary tensor
            scale: Scale factor used for quantization
            
        Returns:
            Dequantized floating point tensor
        """
        return quantized.astype(np.float32) * scale


class PentaryLinear:
    """Linear layer with pentary weights"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize pentary linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights (will be quantized)
        self.weight_scale = 1.0
        self.bias_scale = 1.0
        
        # Pentary weights and bias (stored as integers {-2, -1, 0, 1, 2})
        self.weight_pentary = np.zeros((out_features, in_features), dtype=np.int32)
        self.bias_pentary = np.zeros(out_features, dtype=np.int32) if bias else None
        
        # For training: store full precision weights
        self.weight_fp = None
        self.bias_fp = None
        
    def initialize_weights(self, method: str = 'xavier'):
        """Initialize weights using specified method"""
        if method == 'xavier':
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            self.weight_fp = np.random.uniform(-limit, limit, 
                                               (self.out_features, self.in_features))
        elif method == 'he':
            limit = np.sqrt(2.0 / self.in_features)
            self.weight_fp = np.random.uniform(-limit, limit,
                                              (self.out_features, self.in_features))
        else:  # uniform
            self.weight_fp = np.random.uniform(-0.1, 0.1,
                                              (self.out_features, self.in_features))
        
        if self.use_bias:
            self.bias_fp = np.zeros(self.out_features, dtype=np.float32)
        
        # Quantize to pentary
        self.quantize_weights()
    
    def quantize_weights(self):
        """Quantize floating point weights to pentary"""
        if self.weight_fp is not None:
            self.weight_pentary, self.weight_scale = PentaryQuantizer.quantize_tensor(
                self.weight_fp
            )
        
        if self.use_bias and self.bias_fp is not None:
            self.bias_pentary, self.bias_scale = PentaryQuantizer.quantize_tensor(
                self.bias_fp
            )
    
    def forward(self, x: np.ndarray, use_pentary: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_features)
            use_pentary: If True, use pentary arithmetic; else use floating point
            
        Returns:
            Output tensor (batch_size, out_features)
        """
        if use_pentary:
            # Pentary matrix multiplication
            # For efficiency, we'll use numpy but simulate pentary behavior
            # In hardware, this would use memristor crossbar arrays
            
            # Matrix multiply with pentary weights, then dequantize by scaling the result
            output = np.dot(x, self.weight_pentary.T.astype(np.float32))
            # Dequantize by scaling the matmul result, not the input
            output = output * (self.weight_scale if self.weight_scale > 0 else 1.0)
            
            # Add bias
            if self.use_bias and self.bias_pentary is not None:
                bias_scaled = self.bias_pentary.astype(np.float32) * self.bias_scale
                output = output + bias_scaled
            
            return output
        else:
            # Floating point forward pass (for training)
            if self.weight_fp is None:
                raise ValueError("Floating point weights not initialized")
            
            output = np.dot(x, self.weight_fp.T)
            if self.use_bias and self.bias_fp is not None:
                output = output + self.bias_fp
            
            return output
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Backward pass for training.
        
        Args:
            grad_output: Gradient w.r.t. output (batch_size, out_features)
            x: Input tensor (batch_size, in_features)
            
        Returns:
            (grad_input, grad_weight, grad_bias)
        """
        # Gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weight_fp)
        
        # Gradient w.r.t. weights
        grad_weight = np.dot(grad_output.T, x)
        
        # Gradient w.r.t. bias
        grad_bias = np.sum(grad_output, axis=0) if self.use_bias else None
        
        return grad_input, grad_weight, grad_bias
    
    def update_weights(self, grad_weight: np.ndarray, grad_bias: Optional[np.ndarray],
                      learning_rate: float):
        """Update weights using gradients"""
        if self.weight_fp is None:
            self.weight_fp = np.zeros((self.out_features, self.in_features), dtype=np.float32)
        
        self.weight_fp -= learning_rate * grad_weight
        
        if self.use_bias and grad_bias is not None:
            if self.bias_fp is None:
                self.bias_fp = np.zeros(self.out_features, dtype=np.float32)
            self.bias_fp -= learning_rate * grad_bias
        
        # Re-quantize after update
        self.quantize_weights()


class PentaryReLU:
    """ReLU activation function"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """ReLU forward pass"""
        return np.maximum(0, x)
    
    @staticmethod
    def backward(grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ReLU backward pass"""
        return grad_output * (x > 0).astype(np.float32)


class PentaryConv2D:
    """2D Convolution layer with pentary weights"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        """
        Initialize pentary 2D convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding size
            bias: Whether to use bias
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Pentary weights
        self.weight_scale = 1.0
        self.bias_scale = 1.0
        self.weight_pentary = np.zeros((out_channels, in_channels, kernel_size, kernel_size),
                                      dtype=np.int32)
        self.bias_pentary = np.zeros(out_channels, dtype=np.int32) if bias else None
        
        # For training
        self.weight_fp = None
        self.bias_fp = None
    
    def initialize_weights(self, method: str = 'xavier'):
        """Initialize weights"""
        if method == 'xavier':
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            fan_out = self.out_channels * self.kernel_size * self.kernel_size
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight_fp = np.random.uniform(-limit, limit,
                                              (self.out_channels, self.in_channels,
                                               self.kernel_size, self.kernel_size))
        else:
            self.weight_fp = np.random.uniform(-0.1, 0.1,
                                              (self.out_channels, self.in_channels,
                                               self.kernel_size, self.kernel_size))
        
        if self.use_bias:
            self.bias_fp = np.zeros(self.out_channels, dtype=np.float32)
        
        self.quantize_weights()
    
    def quantize_weights(self):
        """Quantize weights to pentary"""
        if self.weight_fp is not None:
            self.weight_pentary, self.weight_scale = PentaryQuantizer.quantize_tensor(
                self.weight_fp
            )
        
        if self.use_bias and self.bias_fp is not None:
            self.bias_pentary, self.bias_scale = PentaryQuantizer.quantize_tensor(
                self.bias_fp
            )
    
    def forward(self, x: np.ndarray, use_pentary: bool = True) -> np.ndarray:
        """
        Forward pass (simplified convolution).
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            use_pentary: Use pentary arithmetic
            
        Returns:
            Output tensor (batch_size, out_channels, out_height, out_width)
        """
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
        
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)
        
        if use_pentary:
            weights = self.weight_pentary.astype(np.float32) * self.weight_scale
        else:
            weights = self.weight_fp
        
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
                        output[b, oc, oh, ow] = np.sum(x_patch * weights[oc, :, :, :])
                        
                        if self.use_bias:
                            if use_pentary:
                                bias_val = self.bias_pentary[oc] * self.bias_scale
                            else:
                                bias_val = self.bias_fp[oc]
                            output[b, oc, oh, ow] += bias_val
        
        return output


class PentaryMaxPool2D:
    """2D Max Pooling layer"""
    
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        """
        Initialize max pooling layer.
        
        Args:
            kernel_size: Size of pooling window
            stride: Stride of pooling
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.indices = None  # Store indices for backward pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output tensor (batch_size, channels, out_height, out_width)
        """
        batch_size, channels, in_h, in_w = x.shape
        
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
        self.indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        output[b, c, oh, ow] = max_val
                        self.indices[b, c, oh, ow] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, grad_output: np.ndarray, x_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output
            x_shape: Shape of input tensor
            
        Returns:
            Gradient w.r.t. input
        """
        batch_size, channels, in_h, in_w = x_shape
        grad_input = np.zeros((batch_size, channels, in_h, in_w), dtype=np.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(grad_output.shape[2]):
                    for ow in range(grad_output.shape[3]):
                        idx_h, idx_w = self.indices[b, c, oh, ow]
                        grad_input[b, c, idx_h, idx_w] += grad_output[b, c, oh, ow]
        
        return grad_input


class PentaryBatchNorm:
    """Batch normalization layer"""
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        """
        Initialize batch normalization.
        
        Args:
            num_features: Number of features
            momentum: Momentum for running statistics
            eps: Small value for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        
        # Running statistics
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, num_features, ...)
            
        Returns:
            Normalized tensor
        """
        if self.training:
            # Compute batch statistics
            axis = tuple(range(len(x.shape) - 1))
            mean = np.mean(x, axis=axis, keepdims=True)
            var = np.var(x, axis=axis, keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * np.mean(mean)
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * np.mean(var)
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output


def main():
    """Test pentary neural network layers"""
    print("=" * 70)
    print("Pentary Neural Network Layers Test")
    print("=" * 70)
    
    # Test quantizer
    print("\n1. Testing Pentary Quantizer:")
    print("-" * 70)
    test_values = np.array([-2.5, -1.2, 0.0, 0.8, 1.9, 2.3])
    quantized, scale = PentaryQuantizer.quantize_tensor(test_values)
    print(f"Original: {test_values}")
    print(f"Scale: {scale:.4f}")
    print(f"Quantized: {quantized}")
    dequantized = PentaryQuantizer.dequantize_tensor(quantized, scale)
    print(f"Dequantized: {dequantized}")
    
    # Test linear layer
    print("\n2. Testing Pentary Linear Layer:")
    print("-" * 70)
    linear = PentaryLinear(10, 5, bias=True)
    linear.initialize_weights('xavier')
    x = np.random.randn(3, 10).astype(np.float32)
    output = linear.forward(x, use_pentary=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight scale: {linear.weight_scale:.4f}")
    print(f"Sample weights (pentary): {linear.weight_pentary[0, :5]}")
    
    # Test ReLU
    print("\n3. Testing Pentary ReLU:")
    print("-" * 70)
    x_relu = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    output_relu = PentaryReLU.forward(x_relu)
    print(f"Input: {x_relu}")
    print(f"Output: {output_relu}")
    
    # Test Conv2D
    print("\n4. Testing Pentary Conv2D:")
    print("-" * 70)
    conv = PentaryConv2D(3, 16, kernel_size=3, padding=1)
    conv.initialize_weights('xavier')
    x_conv = np.random.randn(2, 3, 32, 32).astype(np.float32)
    output_conv = conv.forward(x_conv, use_pentary=True)
    print(f"Input shape: {x_conv.shape}")
    print(f"Output shape: {output_conv.shape}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
