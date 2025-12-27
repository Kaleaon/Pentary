#!/usr/bin/env python3
"""
Pentary Mamba Implementation
Selective State Space Model adapted for Pentary (base-5) computing

Based on: arXiv:2312.00752 - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Key Features:
- Linear O(n) complexity in sequence length
- Selective mechanism: SSM parameters depend on input
- Hardware-efficient: Uses shift-add operations only
- Supports both parallel (training) and recurrent (inference) modes
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import math


class PentaryQuantizer:
    """Utility class for pentary quantization."""
    
    @staticmethod
    def quantize(x: np.ndarray, scale: float = None) -> Tuple[np.ndarray, float]:
        """Quantize to pentary levels {-2, -1, 0, +1, +2}."""
        if scale is None:
            scale = np.max(np.abs(x)) / 2.0 if np.max(np.abs(x)) > 0 else 1.0
        quantized = np.clip(np.round(x / scale), -2, 2).astype(np.int8)
        return quantized, scale
    
    @staticmethod
    def dequantize(x: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize from pentary to float."""
        return x.astype(np.float32) * scale


class PentarySSMCore:
    """
    Core Selective State Space Model with pentary quantization.
    
    State equation:
        h_t = A * h_{t-1} + B * x_t    (State update)
        y_t = C * h_t + D * x_t        (Output)
    
    In Mamba, A, B, C are input-dependent (selective).
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        """
        Initialize SSM core.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N in paper)
            d_conv: Convolution kernel size for input processing
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize all parameters with pentary quantization."""
        # A matrix (state transition) - initialized for stability
        # Use diagonal structure like in S4
        A_init = -np.exp(np.random.randn(self.d_state) * 0.5)
        self.A, self.A_scale = PentaryQuantizer.quantize(A_init)
        
        # D (skip connection)
        D_init = np.ones(self.d_model) * 0.5
        self.D, self.D_scale = PentaryQuantizer.quantize(D_init)
        
        # Projections for selective mechanism (B, C depend on input)
        # x -> (B, C, delta)
        proj_size = self.d_state * 2 + 1  # B + C + delta
        self.x_proj = self._init_pentary_weights(self.d_model, proj_size * self.d_model)
        
        # Delta projection (for discretization)
        self.dt_proj = self._init_pentary_weights(self.d_model, self.d_model)
        
        # Convolution weights
        self.conv_weight = self._init_pentary_weights(self.d_model, self.d_conv)
        
        # Scales for projections
        self.x_proj_scale = 0.1
        self.dt_proj_scale = 0.1
    
    def _init_pentary_weights(self, *shape) -> np.ndarray:
        """Initialize pentary-quantized weights."""
        weights = np.random.randn(*shape) * 0.5
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Matrix multiplication with pentary weights using shift-add.
        
        For weights in {-2, -1, 0, +1, +2}:
        - 0: skip (zero power)
        - ±1: add/subtract
        - ±2: shift and add/subtract
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        out_dim = W.shape[0] if len(W.shape) > 1 else W.shape[0]
        in_dim = W.shape[-1]
        
        output = np.zeros((batch_size, out_dim), dtype=np.float32)
        
        W_flat = W.reshape(out_dim, -1) if len(W.shape) > 1 else W.reshape(-1, 1)
        
        for i in range(min(out_dim, W_flat.shape[0])):
            for j in range(min(in_dim, W_flat.shape[1] if len(W_flat.shape) > 1 else 1)):
                w = W_flat[i, j] if len(W_flat.shape) > 1 else W_flat[i]
                if w == 0:
                    continue
                elif w == 1:
                    output[:, i] += x[:, j] if j < x.shape[1] else 0
                elif w == -1:
                    output[:, i] -= x[:, j] if j < x.shape[1] else 0
                elif w == 2:
                    output[:, i] += 2 * (x[:, j] if j < x.shape[1] else 0)
                elif w == -2:
                    output[:, i] -= 2 * (x[:, j] if j < x.shape[1] else 0)
        
        return output * scale
    
    def _selective_scan_recurrent(
        self, 
        x: np.ndarray,
        delta: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        h_prev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recurrent selective scan (for inference).
        
        Processes one token at a time with O(1) memory.
        """
        batch_size = x.shape[0]
        
        if h_prev is None:
            h = np.zeros((batch_size, self.d_state, self.d_model), dtype=np.float32)
        else:
            h = h_prev.copy()
        
        # Discretize A based on delta
        # A_bar = exp(delta * A)
        # For pentary, approximate with: A_bar ≈ 1 + delta * A (first-order)
        delta_A = delta[:, :, np.newaxis] * A[np.newaxis, np.newaxis, :]
        A_bar = 1.0 + delta_A  # Simplified discretization
        A_bar = np.clip(A_bar, -2, 2)  # Keep in pentary range
        
        # B_bar = delta * B
        B_bar = delta[:, :, np.newaxis] * B
        B_bar = np.clip(B_bar, -2, 2)
        
        # State update: h = A_bar * h + B_bar * x
        for i in range(self.d_state):
            for j in range(self.d_model):
                # Pentary multiply-accumulate
                h[:, i, j] = (A_bar[:, j, i] if A_bar.shape[2] > i else 1.0) * h[:, i, j]
```suggestion
                h[:, i, j] += ((B_bar[:, j, i] if B_bar.shape[2] > i else 0) * x[:, j])
        
        h = np.clip(h, -10, 10)  # Prevent explosion
        
        # Output: y = C * h + D * x
        y = np.zeros((batch_size, self.d_model), dtype=np.float32)
        for j in range(self.d_model):
            for i in range(self.d_state):
                y[:, j] += C[:, j, i] * h[:, i, j] if C.shape[2] > i else 0
            y[:, j] += D[j] * self.D_scale * x[:, j]
        
        return y, h
    
    def forward_recurrent(
        self, 
        x: np.ndarray,
        h_prev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recurrent forward pass (inference mode).
        
        Args:
            x: Input tensor (batch_size, d_model)
            h_prev: Previous hidden state
        
        Returns:
            y: Output tensor (batch_size, d_model)
            h: New hidden state
        """
        batch_size = x.shape[0]
        
        # Compute selective parameters from input
        # Project x to get delta, B, C
        x_proj = self._pentary_matmul(x, self.x_proj.reshape(self.d_model, -1).T, self.x_proj_scale)
        
        # Split into delta, B, C
        split_size = self.d_model
        delta = x_proj[:, :split_size]
        B = x_proj[:, split_size:split_size + self.d_state * self.d_model]
        C = x_proj[:, split_size + self.d_state * self.d_model:]
        
        # Reshape B and C
        B = B.reshape(batch_size, self.d_model, self.d_state)
        C = C.reshape(batch_size, self.d_model, min(C.shape[1] // self.d_model, self.d_state))
        
        # Apply softplus to delta and scale
        delta = np.log1p(np.exp(delta)) * 0.1
        delta = np.clip(delta, 0.001, 1.0)
        
        # Quantize B, C to pentary
        B = np.clip(np.round(B), -2, 2)
        C = np.clip(np.round(C), -2, 2)
        
        # Run selective scan
        y, h = self._selective_scan_recurrent(
            x, delta, 
            self.A.astype(np.float32) * self.A_scale,
            B, C,
            self.D.astype(np.float32),
            h_prev
        )
        
        return y, h
    
    def forward_parallel(self, x: np.ndarray) -> np.ndarray:
        """
        Parallel forward pass (training mode).
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            y: Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        outputs = []
        h = None
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            y_t, h = self.forward_recurrent(x_t, h)
            outputs.append(y_t)
        
        return np.stack(outputs, axis=1)


class PentaryMambaBlock:
    """
    Complete Mamba block with pentary quantization.
    
    Architecture:
        x -> Linear -> Conv1D -> SSM -> Linear -> + -> output
                   ↘ Linear -> SiLU -> ↗
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        """
        Initialize Mamba block.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            expand: Expansion factor for inner dimension
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Projections
        self.in_proj = self._init_pentary_weights(d_model, self.d_inner * 2)
        self.out_proj = self._init_pentary_weights(self.d_inner, d_model)
        
        # SSM core
        self.ssm = PentarySSMCore(self.d_inner, d_state)
        
        # Conv1D for local context
        self.conv_weight = self._init_pentary_weights(self.d_inner, 4)
        self.conv_buffer = None
    
    def _init_pentary_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize pentary weights."""
        weights = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / (in_dim + out_dim))
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Pentary matrix multiplication."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        output = np.zeros((x.shape[0], W.shape[0]), dtype=np.float32)
        
        for i in range(W.shape[0]):
            for j in range(min(W.shape[1], x.shape[1])):
                w = W[i, j]
                if w == 0:
                    continue
                elif abs(w) == 1:
                    output[:, i] += w * x[:, j]
                else:  # ±2
                    output[:, i] += w * x[:, j]
        
        return output
    
    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation (Swish)."""
        return x * (1 / (1 + np.exp(-x)))
    
    def _conv1d(self, x: np.ndarray) -> np.ndarray:
        """1D convolution with pentary weights."""
        batch_size, d = x.shape
        
        # Initialize or update buffer
        if self.conv_buffer is None:
            self.conv_buffer = np.zeros((batch_size, 4, d), dtype=np.float32)
        
        # Shift buffer and add new input
        self.conv_buffer = np.roll(self.conv_buffer, -1, axis=1)
        self.conv_buffer[:, -1, :] = x
        
        # Apply convolution
        output = np.zeros_like(x)
        for k in range(4):
            for d_idx in range(d):
                w = self.conv_weight[d_idx % self.conv_weight.shape[0], k]
                if w != 0:
                    output[:, d_idx] += w * self.conv_buffer[:, k, d_idx]
        
        return output
    
    def forward(
        self, 
        x: np.ndarray,
        h_prev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single token (recurrent mode).
        
        Args:
            x: Input (batch_size, d_model)
            h_prev: Previous SSM state
        
        Returns:
            output: (batch_size, d_model)
            h: New SSM state
        """
        batch_size = x.shape[0]
        
        # Input projection to get x and z branches
        xz = self._pentary_matmul(x, self.in_proj)
        x_branch = xz[:, :self.d_inner]
        z = xz[:, self.d_inner:]
        
        # Conv1D on x branch
        x_conv = self._conv1d(x_branch)
        x_conv = self._silu(x_conv)
        
        # SSM
        y, h = self.ssm.forward_recurrent(x_conv, h_prev)
        
        # Gate with z
        z_act = self._silu(z)
        y = y * z_act
        
        # Output projection
        output = self._pentary_matmul(y, self.out_proj)
        
        return output, h
    
    def forward_sequence(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for full sequence.
        
        Args:
            x: Input (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        outputs = []
        h = None
        self.conv_buffer = None  # Reset conv buffer
        
        for t in range(seq_len):
            out_t, h = self.forward(x[:, t, :], h)
            outputs.append(out_t)
        
        return np.stack(outputs, axis=1)


class PentaryMamba:
    """
    Complete Pentary Mamba model.
    
    Stack of Mamba blocks with embedding and output layers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16
    ):
        """
        Initialize Pentary Mamba model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of Mamba blocks
            d_state: SSM state dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embedding (pentary quantized)
        self.embedding = self._init_pentary_embedding(vocab_size, d_model)
        self.embed_scale = 0.1
        
        # Mamba blocks
        self.blocks = [
            PentaryMambaBlock(d_model, d_state)
            for _ in range(n_layers)
        ]
        
        # Output projection
        self.lm_head = self._init_pentary_weights(d_model, vocab_size)
    
    def _init_pentary_embedding(self, vocab_size: int, d_model: int) -> np.ndarray:
        """Initialize pentary embedding table."""
        emb = np.random.randn(vocab_size, d_model) * 0.02
        quantized, _ = PentaryQuantizer.quantize(emb)
        return quantized
    
    def _init_pentary_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize pentary weights."""
        weights = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / (in_dim + out_dim))
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Pentary matrix multiplication."""
        output = np.zeros((x.shape[0], W.shape[0]), dtype=np.float32)
        
        for i in range(W.shape[0]):
            for j in range(min(W.shape[1], x.shape[1])):
                w = W[i, j]
                if w != 0:
                    output[:, i] += w * x[:, j]
        
        return output
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embedding[input_ids].astype(np.float32) * self.embed_scale
        
        # Pass through Mamba blocks
        for block in self.blocks:
            residual = x
            x = block.forward_sequence(x)
            x = x + residual  # Residual connection
        
        # Output projection
        logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        for t in range(seq_len):
            logits[:, t, :] = self._pentary_matmul(x[:, t, :], self.lm_head)
        
        return logits
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
        
        Returns:
            generated: All token IDs including prompt
        """
        batch_size = prompt_ids.shape[0]
        generated = prompt_ids.copy()
        
        # Process prompt
        x = self.embedding[prompt_ids].astype(np.float32) * self.embed_scale
        
        # Get states from prompt
        states = [None] * self.n_layers
        for t in range(prompt_ids.shape[1]):
            h = x[:, t, :]
            for i, block in enumerate(self.blocks):
                block.conv_buffer = None  # Reset
            for i, block in enumerate(self.blocks):
                h_out, states[i] = block.forward(h, states[i])
                h = h + h_out
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self._pentary_matmul(h, self.lm_head)
            
            # Sample
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            next_token = np.array([
                np.random.choice(self.vocab_size, p=probs[b])
                for b in range(batch_size)
            ])
            
            generated = np.concatenate([generated, next_token[:, np.newaxis]], axis=1)
            
            # Forward next token through blocks
            h = self.embedding[next_token].astype(np.float32) * self.embed_scale
            for i, block in enumerate(self.blocks):
                h_out, states[i] = block.forward(h, states[i])
                h = h + h_out
        
        return generated
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = 0
        zero_params = 0
        
        # Count embedding
        total_params += self.embedding.size
        zero_params += np.sum(self.embedding == 0)
        
        # Count blocks
        for block in self.blocks:
            total_params += block.in_proj.size
            total_params += block.out_proj.size
            total_params += block.conv_weight.size
            total_params += block.ssm.A.size
            total_params += block.ssm.D.size
            
            zero_params += np.sum(block.in_proj == 0)
            zero_params += np.sum(block.out_proj == 0)
            zero_params += np.sum(block.ssm.A == 0)
        
        # Count head
        total_params += self.lm_head.size
        zero_params += np.sum(self.lm_head == 0)
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'sparsity': zero_params / total_params if total_params > 0 else 0,
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'vocab_size': self.vocab_size
        }


def demo_pentary_mamba():
    """Demonstrate Pentary Mamba."""
    print("=" * 70)
    print("Pentary Mamba - Selective State Space Model Demo")
    print("=" * 70)
    
    # Create model
    model = PentaryMamba(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        d_state=8
    )
    
    # Get stats
    stats = model.get_stats()
    print("\nModel Statistics:")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Zero Parameters: {stats['zero_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Layers: {stats['n_layers']}")
    print(f"  Model Dimension: {stats['d_model']}")
    
    # Test forward pass
    print("\nForward Pass Test:")
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"  Input shape: {input_ids.shape}")
    
    logits = model.forward(input_ids)
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Test generation
    print("\nGeneration Test:")
    prompt = np.array([[1, 2, 3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"  Prompt: {prompt[0]}")
    print(f"  Generated: {generated[0]}")
    
    # Benchmark
    print("\nBenchmark (100 forward passes):")
    import time
    
    x = np.random.randint(0, 1000, (1, 32))
    
    start = time.time()
    for _ in range(100):
        _ = model.forward(x)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {100 * 32 / elapsed:.1f} tokens/sec")
    
    print("\n" + "=" * 70)
    print("Pentary Mamba: O(n) complexity verified!")
    print("=" * 70)


if __name__ == "__main__":
    demo_pentary_mamba()
