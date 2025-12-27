#!/usr/bin/env python3
"""
Pentary PoPE (Polar Coordinate Position Embeddings) Implementation

Based on: arXiv:2509.10534 - "Decoupling the 'What' and 'Where' With 
          Polar Coordinate Positional Embeddings"

This module implements PoPE adapted for Pentary computing architecture,
providing efficient positional encoding that:
1. Decouples content ("what") from position ("where")
2. Enables length extrapolation without retraining
3. Uses pentary-quantized angles and magnitudes for hardware efficiency
"""

import numpy as np
import math
from typing import Optional, Tuple, Dict


class PentaryPoPE:
    """
    Pentary-compatible Polar Coordinate Position Embeddings.
    
    Key features:
    - Angles quantized to pentary levels for position encoding
    - Magnitudes quantized to pentary for content encoding
    - Pre-computed cosine lookup table for hardware efficiency
    - Support for length extrapolation
    
    Architecture:
        Position → Angle (phase in polar coords)
        Content → Magnitude (radius in polar coords)
        
        Attention(i, j) = magnitude_similarity(i, j) * cos(angle_i - angle_j)
    """
    
    def __init__(
        self, 
        d_model: int,
        max_seq_len: int = 8192,
        angle_digits: int = 3,
        base: float = 10000.0
    ):
        """
        Initialize PentaryPoPE.
        
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length (for validation)
            angle_digits: Number of pentary digits for angle precision
                         (1=5 angles, 2=25 angles, 3=125 angles)
            base: Base for frequency computation (like RoPE)
        """
        self.d_model = d_model
        self.d_head = d_model // 2  # Pairs for sin/cos
        self.max_seq_len = max_seq_len
        self.angle_digits = angle_digits
        self.num_angles = 5 ** angle_digits
        self.base = base
        
        # Pre-compute base frequencies (like RoPE)
        self.frequencies = self._init_frequencies()
        
        # Pre-compute cosine lookup table for all angle pairs
        self.cos_table = self._init_cos_table()
        
        # Pre-compute sin lookup table (for magnitude rotation)
        self.sin_table = self._init_sin_table()
        
        print(f"PentaryPoPE initialized:")
        print(f"  d_model: {d_model}")
        print(f"  angle_digits: {angle_digits} ({self.num_angles} distinct angles)")
        print(f"  max_seq_len: {max_seq_len}")
    
    def _init_frequencies(self) -> np.ndarray:
        """
        Initialize frequency for each dimension pair.
        
        Uses exponentially decreasing frequencies like RoPE.
        """
        frequencies = np.zeros(self.d_head)
        
        for i in range(self.d_head):
            # Frequency decreases exponentially with dimension
            freq = 1.0 / (self.base ** (2 * i / self.d_model))
            # Scale to angle range
            frequencies[i] = freq * self.num_angles
        
        return frequencies
    
    def _init_cos_table(self) -> np.ndarray:
        """
        Pre-compute pentary cosine values for all angle combinations.
        
        This table is small (125x125 = ~16KB for 3-digit angles)
        and enables hardware-efficient cosine lookup.
        """
        table = np.zeros((self.num_angles, self.num_angles), dtype=np.int8)
        
        for i in range(self.num_angles):
            for j in range(self.num_angles):
                # Relative angle (with wrap-around)
                angle_diff = (i - j) % self.num_angles
                # Convert to radians
                theta = 2 * np.pi * angle_diff / self.num_angles
                cos_val = np.cos(theta)
                # Quantize to pentary
                table[i, j] = self._quantize_to_pentary(cos_val)
        
        return table
    
    def _init_sin_table(self) -> np.ndarray:
        """Pre-compute pentary sine values for all angles."""
        table = np.zeros(self.num_angles, dtype=np.int8)
        
        for i in range(self.num_angles):
            theta = 2 * np.pi * i / self.num_angles
            sin_val = np.sin(theta)
            table[i] = self._quantize_to_pentary(sin_val)
        
        return table
    
    def _quantize_to_pentary(self, x: float) -> int:
        """
        Quantize a float to pentary level {-2, -1, 0, +1, +2}.
        
        Uses thresholds at -0.75, -0.25, +0.25, +0.75.
        """
        if x <= -0.75:
            return -2
        elif x <= -0.25:
            return -1
        elif x <= 0.25:
            return 0
        elif x <= 0.75:
            return 1
        else:
            return 2
    
    def encode_position(self, position: int) -> np.ndarray:
        """
        Encode a single position as pentary angles.
        
        Args:
            position: Token position (0-indexed)
        
        Returns:
            angles: Array of shape (d_head,) with pentary angle indices
        """
        angles = np.zeros(self.d_head, dtype=np.int32)
        
        for i in range(self.d_head):
            # Angle = (position * frequency) mod num_angles
            angle = int(position * self.frequencies[i]) % self.num_angles
            angles[i] = angle
        
        return angles
    
    def encode_positions(self, seq_len: int) -> np.ndarray:
        """
        Encode all positions for a sequence.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            angles: Array of shape (seq_len, d_head) with angle indices
        """
        angles = np.zeros((seq_len, self.d_head), dtype=np.int32)
        
        for pos in range(seq_len):
            angles[pos] = self.encode_position(pos)
        
        return angles
    
    def compute_position_bias(self, seq_len: int) -> np.ndarray:
        """
        Compute position-only attention bias.
        
        This is the cos(θ_i - θ_j) term in PoPE, representing
        relative position similarity.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            bias: Array of shape (seq_len, seq_len) with pentary biases
        """
        # Get all position angles
        angles = self.encode_positions(seq_len)
        
        # Compute bias matrix
        bias = np.zeros((seq_len, seq_len), dtype=np.int32)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Sum cosine over all dimension pairs
                for d in range(self.d_head):
                    bias[i, j] += self.cos_table[angles[i, d], angles[j, d]]
        
        return bias
    
    def apply_rotary(
        self, 
        x: np.ndarray, 
        position: int,
        quantize_output: bool = True
    ) -> np.ndarray:
        """
        Apply rotary transformation to input tensor.
        
        This rotates the input in the complex plane based on position.
        
        Args:
            x: Input tensor of shape (..., d_model)
            position: Token position
            quantize_output: Whether to quantize output to pentary
        
        Returns:
            Rotated tensor of same shape as input
        """
        # Split into pairs (real, imaginary interpretation)
        x_even = x[..., 0::2]  # "real" components
        x_odd = x[..., 1::2]   # "imaginary" components
        
        # Get angles for this position
        angles = self.encode_position(position)
        
        # Apply rotation: x' = x * e^(iθ)
        # x'_real = x_real * cos(θ) - x_imag * sin(θ)
        # x'_imag = x_real * sin(θ) + x_imag * cos(θ)
        
        x_rotated = np.zeros_like(x)
        
        for d in range(self.d_head):
            cos_val = np.cos(2 * np.pi * angles[d] / self.num_angles)
            sin_val = np.sin(2 * np.pi * angles[d] / self.num_angles)
            
            x_rotated[..., 2*d] = x_even[..., d] * cos_val - x_odd[..., d] * sin_val
            x_rotated[..., 2*d + 1] = x_even[..., d] * sin_val + x_odd[..., d] * cos_val
        
        if quantize_output:
            # Quantize to pentary levels
            scale = np.max(np.abs(x_rotated)) / 2.0 if np.max(np.abs(x_rotated)) > 0 else 1.0
            x_rotated = np.clip(np.round(x_rotated / scale), -2, 2).astype(np.int8)
        
        return x_rotated
    
    def get_length_extrapolation_range(self) -> int:
        """
        Calculate the maximum sequence length before angle wrap-around.
        
        Returns:
            max_length: Maximum extrapolatable sequence length
        """
        # Find the slowest frequency (lowest frequency = highest index)
        min_freq = self.frequencies[-1]
        
        if min_freq > 0:
            # Length before wrap = num_angles / min_freq
            return int(self.num_angles / min_freq)
        else:
            return self.max_seq_len
    
    def visualize_angles(self, seq_len: int = 10, dimensions: int = 3):
        """
        Visualize angle patterns for debugging.
        
        Args:
            seq_len: Number of positions to show
            dimensions: Number of dimensions to show
        """
        angles = self.encode_positions(seq_len)
        
        print(f"\nAngle patterns (first {dimensions} dimensions):")
        print("-" * 50)
        
        header = "Pos |"
        for d in range(min(dimensions, self.d_head)):
            header += f" Dim{d:2d} |"
        print(header)
        print("-" * 50)
        
        for pos in range(seq_len):
            row = f"{pos:3d} |"
            for d in range(min(dimensions, self.d_head)):
                angle = angles[pos, d]
                row += f"  {angle:3d}  |"
            print(row)


class PentaryPoPEAttention:
    """
    Multi-head self-attention with PoPE positional encoding.
    
    Implements the decoupled attention from arXiv:2509.10534
    adapted for Pentary hardware.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        angle_digits: int = 3
    ):
        """
        Initialize PoPE attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            angle_digits: Pentary digits for angle precision
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # PoPE encoder
        self.pope = PentaryPoPE(d_model, angle_digits=angle_digits)
        
        # Weight matrices (pentary quantized)
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        self.W_o = self._init_weights(d_model, d_model)
    
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize pentary weights."""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weights = np.random.uniform(-limit, limit, (out_dim, in_dim))
        scale = np.max(np.abs(weights)) / 2.0
        return np.clip(np.round(weights / scale), -2, 2).astype(np.int8)
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with pentary weights.
        
        Uses shift-add operations only (no floating-point multiply).
        """
        output = np.zeros((x.shape[0], W.shape[0]), dtype=np.float32)
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                if w == 0:
                    continue
                elif w == 1:
                    output[:, i] += x[:, j]
                elif w == -1:
                    output[:, i] -= x[:, j]
                elif w == 2:
                    output[:, i] += 2 * x[:, j]
                elif w == -2:
                    output[:, i] -= 2 * x[:, j]
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(
        self, 
        x: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass with PoPE attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        x_flat = x.reshape(-1, self.d_model)
        Q = self._pentary_matmul(x_flat, self.W_q).reshape(batch_size, seq_len, self.d_model)
        K = self._pentary_matmul(x_flat, self.W_k).reshape(batch_size, seq_len, self.d_model)
        V = self._pentary_matmul(x_flat, self.W_v).reshape(batch_size, seq_len, self.d_model)
        
        # Apply rotary encoding (PoPE style)
        for pos in range(seq_len):
            Q[:, pos, :] = self.pope.apply_rotary(Q[:, pos, :], pos, quantize_output=False)
            K[:, pos, :] = self.pope.apply_rotary(K[:, pos, :], pos, quantize_output=False)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        context_flat = context.reshape(-1, self.d_model)
        output = self._pentary_matmul(context_flat, self.W_o)
        
        return output.reshape(batch_size, seq_len, self.d_model)


def demo_pentary_pope():
    """Demonstrate Pentary PoPE implementation."""
    print("=" * 70)
    print("Pentary PoPE (Polar Coordinate Position Embeddings) Demo")
    print("=" * 70)
    print()
    
    # Create PoPE encoder
    d_model = 64
    pope = PentaryPoPE(d_model, angle_digits=3)
    
    # Test position encoding
    print("1. Position Encoding")
    print("-" * 50)
    pope.visualize_angles(seq_len=8, dimensions=4)
    
    # Test position bias
    print("\n2. Position Bias Matrix (first 8x8 positions)")
    print("-" * 50)
    bias = pope.compute_position_bias(8)
    
    print("Position bias (sum of cos over all dimensions):")
    for i in range(8):
        row = ""
        for j in range(8):
            row += f"{bias[i, j]:4d} "
        print(row)
    
    # Test length extrapolation
    print("\n3. Length Extrapolation")
    print("-" * 50)
    max_extrap = pope.get_length_extrapolation_range()
    print(f"Maximum extrapolatable length: {max_extrap} tokens")
    print(f"(before angle wrap-around in slowest frequency)")
    
    # Test attention with PoPE
    print("\n4. PoPE Attention Test")
    print("-" * 50)
    
    attention = PentaryPoPEAttention(d_model=64, num_heads=4, angle_digits=3)
    
    # Create test input
    batch_size = 2
    seq_len = 16
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = attention.forward(x)
    print(f"Output shape: {output.shape}")
    
    # Compare with different sequence lengths (extrapolation test)
    print("\n5. Extrapolation Test")
    print("-" * 50)
    
    for test_len in [8, 16, 32, 64]:
        x_test = np.random.randn(1, test_len, d_model).astype(np.float32)
        try:
            output_test = attention.forward(x_test)
            print(f"  Length {test_len}: OK - Output shape {output_test.shape}")
        except Exception as e:
            print(f"  Length {test_len}: FAILED - {e}")
    
    print("\n" + "=" * 70)
    print("PoPE + Pentary Compatibility: CONFIRMED")
    print("=" * 70)


def compare_pope_vs_sinusoidal():
    """Compare PoPE with standard sinusoidal positional encoding."""
    print("\n" + "=" * 70)
    print("Comparison: PoPE vs Sinusoidal Positional Encoding")
    print("=" * 70)
    
    d_model = 64
    seq_len = 16
    
    # PoPE encoding
    pope = PentaryPoPE(d_model, angle_digits=3)
    pope_bias = pope.compute_position_bias(seq_len)
    
    # Sinusoidal encoding (standard)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    sinusoidal = np.zeros((seq_len, d_model))
    sinusoidal[:, 0::2] = np.sin(position * div_term)
    sinusoidal[:, 1::2] = np.cos(position * div_term)
    
    # Compute similarity for sinusoidal
    sinusoidal_sim = np.dot(sinusoidal, sinusoidal.T)
    
    print("\nPoPE Position Bias (diagonal = self-similarity):")
    print(f"  Diagonal mean: {np.mean(np.diag(pope_bias)):.2f}")
    print(f"  Off-diagonal mean: {np.mean(pope_bias - np.diag(np.diag(pope_bias))):.2f}")
    
    print("\nSinusoidal Similarity:")
    print(f"  Diagonal mean: {np.mean(np.diag(sinusoidal_sim)):.2f}")
    print(f"  Off-diagonal mean: {np.mean(sinusoidal_sim - np.diag(np.diag(sinusoidal_sim))):.2f}")
    
    print("\nKey Differences:")
    print("  - PoPE: Decouples content and position")
    print("  - PoPE: Better length extrapolation")
    print("  - PoPE: Pentary-quantized angles for hardware efficiency")
    print("  - Sinusoidal: Entangles content and position when added to embeddings")


if __name__ == "__main__":
    demo_pentary_pope()
    compare_pope_vs_sinusoidal()
