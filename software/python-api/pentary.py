"""
Pentary Python API

This module provides a high-level Python interface to the Pentary accelerator,
enabling seamless integration with PyTorch and other ML frameworks.

Example usage:
    import pentary
    import torch
    
    # Initialize device
    device = pentary.Device(0)
    
    # Create tensors on Pentary device
    a = torch.randn(1024, 1024).to("pentary:0")
    b = torch.randn(1024, 1024).to("pentary:0")
    
    # Perform matrix multiplication on Pentary
    c = torch.matmul(a, b)
    
    # Transfer back to CPU
    c_cpu = c.cpu()
"""

import numpy as np
from typing import Optional, Tuple, Union
import _pentary_core  # C++ extension module (built with pybind11)

__version__ = "1.0.0"

class Device:
    """Pentary device handle"""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a Pentary device
        
        Args:
            device_id: Device ID (0-based)
        """
        self._handle = _pentary_core.device_init(device_id)
        self.device_id = device_id
    
    def __del__(self):
        """Cleanup device resources"""
        if hasattr(self, '_handle'):
            _pentary_core.device_destroy(self._handle)
    
    def synchronize(self):
        """Wait for all operations on this device to complete"""
        _pentary_core.device_synchronize(self._handle)
    
    def __repr__(self):
        return f"pentary.Device(id={self.device_id})"


class Tensor:
    """Pentary tensor (wrapper around device memory)"""
    
    def __init__(self, shape: Tuple[int, ...], dtype=np.float32, device: Optional[Device] = None):
        """
        Create a tensor on the Pentary device
        
        Args:
            shape: Tensor shape
            dtype: Data type (default: float32)
            device: Pentary device (default: Device(0))
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device or Device(0)
        
        # Calculate size in bytes
        self.size = int(np.prod(shape))
        self.nbytes = self.size * np.dtype(dtype).itemsize
        
        # Allocate device memory
        self._ptr = _pentary_core.malloc(self.device._handle, self.nbytes)
    
    def __del__(self):
        """Free device memory"""
        if hasattr(self, '_ptr'):
            _pentary_core.free(self._ptr)
    
    def to_numpy(self) -> np.ndarray:
        """Copy tensor data to NumPy array on CPU"""
        arr = np.empty(self.shape, dtype=self.dtype)
        _pentary_core.memcpy_d2h(arr.ctypes.data, self._ptr, self.nbytes)
        return arr
    
    @staticmethod
    def from_numpy(arr: np.ndarray, device: Optional[Device] = None) -> 'Tensor':
        """Create a Pentary tensor from a NumPy array"""
        tensor = Tensor(arr.shape, arr.dtype, device)
        _pentary_core.memcpy_h2d(tensor._ptr, arr.ctypes.data, tensor.nbytes)
        return tensor
    
    def __repr__(self):
        return f"pentary.Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class Stream:
    """Pentary stream for asynchronous operations"""
    
    def __init__(self, device: Optional[Device] = None):
        """Create a stream on the specified device"""
        self.device = device or Device(0)
        self._handle = _pentary_core.stream_create(self.device._handle)
    
    def __del__(self):
        """Destroy stream"""
        if hasattr(self, '_handle'):
            _pentary_core.stream_destroy(self._handle)
    
    def synchronize(self):
        """Wait for all operations in this stream to complete"""
        _pentary_core.stream_synchronize(self._handle)


# ============================================================================
# Neural Network Operations
# ============================================================================

def gemm(A: Tensor, B: Tensor, alpha: float = 1.0, beta: float = 0.0, 
         C: Optional[Tensor] = None, stream: Optional[Stream] = None) -> Tensor:
    """
    General Matrix Multiplication: C = alpha * A @ B + beta * C
    
    Args:
        A: Input matrix A (M x K)
        B: Input matrix B (K x N)
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for C
        C: Optional output matrix (M x N). If None, a new tensor is created.
        stream: Optional stream for asynchronous execution
    
    Returns:
        Output matrix C (M x N)
    """
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("GEMM requires 2D tensors")
    
    M, K1 = A.shape
    K2, N = B.shape
    
    if K1 != K2:
        raise ValueError(f"Inner dimensions must match: {K1} != {K2}")
    
    if C is None:
        C = Tensor((M, N), A.dtype, A.device)
    elif C.shape != (M, N):
        raise ValueError(f"Output shape mismatch: expected {(M, N)}, got {C.shape}")
    
    stream_handle = stream._handle if stream else None
    
    _pentary_core.nn_gemm(
        A.device._handle,
        M, N, K1,
        alpha,
        A._ptr, K1,
        B._ptr, N,
        beta,
        C._ptr, N,
        stream_handle
    )
    
    return C


def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           stream: Optional[Stream] = None) -> Tensor:
    """
    2D Convolution
    
    Args:
        input: Input tensor (N x C_in x H_in x W_in)
        weight: Filter weights (C_out x C_in x K_h x K_w)
        bias: Optional bias (C_out)
        stride: Convolution stride (h, w)
        padding: Zero padding (h, w)
        stream: Optional stream for asynchronous execution
    
    Returns:
        Output tensor (N x C_out x H_out x W_out)
    """
    if len(input.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("Conv2d requires 4D tensors")
    
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    
    if C_in != C_in_w:
        raise ValueError(f"Channel mismatch: {C_in} != {C_in_w}")
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    
    H_out = (H_in + 2 * padding_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * padding_w - K_w) // stride_w + 1
    
    output = Tensor((N, C_out, H_out, W_out), input.dtype, input.device)
    
    bias_ptr = bias._ptr if bias else None
    stream_handle = stream._handle if stream else None
    
    _pentary_core.nn_conv2d(
        input.device._handle,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        input._ptr,
        weight._ptr,
        bias_ptr,
        output._ptr,
        stream_handle
    )
    
    return output


def relu(input: Tensor, inplace: bool = False, stream: Optional[Stream] = None) -> Tensor:
    """ReLU activation function"""
    output = input if inplace else Tensor(input.shape, input.dtype, input.device)
    stream_handle = stream._handle if stream else None
    
    _pentary_core.nn_activation(
        input.device._handle,
        0,  # PENTARY_NN_ACTIVATION_RELU
        input.size,
        input._ptr,
        output._ptr,
        stream_handle
    )
    
    return output


# ============================================================================
# Utility Functions
# ============================================================================

def get_device_count() -> int:
    """Get the number of available Pentary devices"""
    return _pentary_core.get_device_count()


def synchronize():
    """Synchronize all Pentary devices"""
    for i in range(get_device_count()):
        device = Device(i)
        device.synchronize()
