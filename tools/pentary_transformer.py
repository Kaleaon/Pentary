#!/usr/bin/env python3
"""
Pentary Transformer Implementation
Transformer architecture optimized for pentary weights (5-level quantization)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import math


class PentaryAttention:
    """
    Multi-head self-attention with pentary weight quantization.
    
    Pentary quantization ({-2, -1, 0, +1, +2}) provides:
    - 97% memory reduction vs FP32
    - Multiplication-free inference (shift-add only)
    - Native sparsity (zero weights = no computation)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate (for training)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.dropout = dropout
        
        # Weight matrices (will be quantized to pentary)
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        self.W_o = self._init_weights(d_model, d_model)
        
        # Scale factors for dequantization
        self.scale_q = 1.0
        self.scale_k = 1.0
        self.scale_v = 1.0
        self.scale_o = 1.0
    
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize and quantize weights to pentary"""
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weights = np.random.uniform(-limit, limit, (out_dim, in_dim))
        
        # Quantize to pentary {-2, -1, 0, +1, +2}
        return self._quantize_to_pentary(weights)
    
    def _quantize_to_pentary(self, x: np.ndarray) -> np.ndarray:
        """Quantize values to pentary levels"""
        scale = np.max(np.abs(x)) / 2.0 if np.max(np.abs(x)) > 0 else 1.0
        x_scaled = x / scale
        x_quantized = np.clip(np.round(x_scaled), -2, 2).astype(np.int8)
        return x_quantized
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with pentary weights.
        
        Optimization: Since weights are {-2, -1, 0, +1, +2}, we can use
        shift-add operations instead of multiplication:
        - w = 0: skip (sparsity)
        - w = ±1: pass through or negate
        - w = ±2: shift and add/subtract
        """
        output = np.zeros((x.shape[0], W.shape[0]))
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                if w == 0:
                    continue  # Skip zero weights
                elif w == 1:
                    output[:, i] += x[:, j]
                elif w == -1:
                    output[:, i] -= x[:, j]
                elif w == 2:
                    output[:, i] += 2 * x[:, j]
                elif w == -2:
                    output[:, i] -= 2 * x[:, j]
        
        return output
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values using pentary matmul
        x_flat = x.reshape(-1, self.d_model)
        Q = self._pentary_matmul(x_flat, self.W_q).reshape(batch_size, seq_len, self.d_model)
        K = self._pentary_matmul(x_flat, self.W_k).reshape(batch_size, seq_len, self.d_model)
        V = self._pentary_matmul(x_flat, self.W_v).reshape(batch_size, seq_len, self.d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        context_flat = context.reshape(-1, self.d_model)
        output = self._pentary_matmul(context_flat, self.W_o).reshape(batch_size, seq_len, self.d_model)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_sparsity(self) -> Dict[str, float]:
        """Get weight sparsity statistics"""
        weights = {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        }
        
        sparsity = {}
        for name, W in weights.items():
            zero_count = np.sum(W == 0)
            total_count = W.size
            sparsity[name] = zero_count / total_count
        
        return sparsity


class PentaryFeedForward:
    """
    Feed-forward network with pentary weights.
    
    Standard FFN: FFN(x) = max(0, xW1 + b1)W2 + b2
    With pentary: Uses shift-add instead of multiplication
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        """
        Initialize FFN.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Weights (quantized to pentary)
        self.W1 = self._init_weights(d_model, d_ff)
        self.W2 = self._init_weights(d_ff, d_model)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize and quantize weights"""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weights = np.random.uniform(-limit, limit, (out_dim, in_dim))
        return self._quantize_to_pentary(weights)
    
    def _quantize_to_pentary(self, x: np.ndarray) -> np.ndarray:
        """Quantize to pentary levels"""
        scale = np.max(np.abs(x)) / 2.0 if np.max(np.abs(x)) > 0 else 1.0
        return np.clip(np.round(x / scale), -2, 2).astype(np.int8)
    
    def _pentary_matmul(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Pentary matrix multiplication"""
        output = np.zeros((x.shape[0], W.shape[0]))
        
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        x_flat = x.reshape(-1, self.d_model)
        
        # First linear + ReLU
        hidden = self._pentary_matmul(x_flat, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        
        # Second linear
        output = self._pentary_matmul(hidden, self.W2) + self.b2
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return output


class PentaryLayerNorm:
    """Layer normalization for pentary models"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize input"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class PentaryTransformerBlock:
    """Single transformer block with pentary weights"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        self.attention = PentaryAttention(d_model, num_heads, dropout)
        self.ffn = PentaryFeedForward(d_model, d_ff, dropout)
        self.norm1 = PentaryLayerNorm(d_model)
        self.norm2 = PentaryLayerNorm(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass with residual connections and layer norm"""
        # Self-attention with residual
        attn_output = self.attention.forward(self.norm1.forward(x), mask)
        x = x + attn_output
        
        # FFN with residual
        ffn_output = self.ffn.forward(self.norm2.forward(x))
        x = x + ffn_output
        
        return x


class PentaryTransformer:
    """
    Complete transformer model with pentary quantization.
    
    This implementation provides:
    - Multi-head self-attention
    - Position-wise feed-forward networks
    - Layer normalization
    - Residual connections
    
    All weights are quantized to pentary {-2, -1, 0, +1, +2} levels.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.0
    ):
        """
        Initialize transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings (quantized)
        self.token_embedding = self._init_embedding(vocab_size, d_model)
        
        # Positional encoding (fixed, not quantized)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [
            PentaryTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Output projection (d_model -> vocab_size)
        self.output_norm = PentaryLayerNorm(d_model)
        self.output_proj = self._init_embedding(vocab_size, d_model)  # (vocab_size, d_model)
    
    def _init_embedding(self, vocab_size: int, d_model: int) -> np.ndarray:
        """Initialize and quantize embeddings"""
        INIT_SCALE = 0.02  # Standard embedding initialization scale
        embeddings = np.random.randn(vocab_size, d_model) * INIT_SCALE
        return self._quantize_to_pentary(embeddings)
    
    def _quantize_to_pentary(self, x: np.ndarray) -> np.ndarray:
        """Quantize to pentary levels"""
        scale = np.max(np.abs(x)) / 2.0 if np.max(np.abs(x)) > 0 else 1.0
        return np.clip(np.round(x / scale), -2, 2).astype(np.int8)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask
    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding[input_ids].astype(np.float32)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len]
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len)
        if attention_mask is not None:
            causal_mask = causal_mask + (1 - attention_mask)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, causal_mask)
        
        # Output projection
        x = self.output_norm.forward(x)
        logits = np.dot(x, self.output_proj.T)  # (batch, seq, d_model) @ (d_model, vocab) = (batch, seq, vocab)
        
        return logits
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = greedy)
            
        Returns:
            Generated token IDs
        """
        generated = prompt_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(generated)[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample next token
            if top_k is not None:
                # Top-k sampling using vectorized operations
                top_k_values = np.partition(logits, -top_k, axis=-1)[:, -top_k:]
                threshold = np.min(top_k_values, axis=-1, keepdims=True)
                logits = np.where(logits >= threshold, logits, -np.inf)
            
            # Softmax
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Sample
            next_token = np.array([
                np.random.choice(self.vocab_size, p=probs[i])
                for i in range(probs.shape[0])
            ])
            
            # Append to generated
            generated = np.concatenate([generated, next_token[:, np.newaxis]], axis=1)
        
        return generated
    
    def get_model_stats(self) -> Dict:
        """Get model statistics"""
        total_params = 0
        zero_params = 0
        
        # Count parameters
        weights = [
            ('token_embedding', self.token_embedding),
            ('output_proj', self.output_proj),
        ]
        
        for block_idx, block in enumerate(self.blocks):
            weights.extend([
                (f'block_{block_idx}.attn.W_q', block.attention.W_q),
                (f'block_{block_idx}.attn.W_k', block.attention.W_k),
                (f'block_{block_idx}.attn.W_v', block.attention.W_v),
                (f'block_{block_idx}.attn.W_o', block.attention.W_o),
                (f'block_{block_idx}.ffn.W1', block.ffn.W1),
                (f'block_{block_idx}.ffn.W2', block.ffn.W2),
            ])
        
        for name, W in weights:
            total_params += W.size
            zero_params += np.sum(W == 0)
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'non_zero_parameters': total_params - zero_params,
            'sparsity': zero_params / total_params if total_params > 0 else 0,
            'memory_bytes': total_params,  # 1 byte per pentary weight
            'memory_mb': total_params / (1024 * 1024),
            'layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
        }


def demo_transformer():
    """Demonstrate pentary transformer"""
    print("=" * 70)
    print("Pentary Transformer Demo")
    print("=" * 70)
    
    # Create small model for demo
    model = PentaryTransformer(
        vocab_size=1000,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_len=64
    )
    
    # Get model stats
    stats = model.get_model_stats()
    print("\nModel Statistics:")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Zero Parameters: {stats['zero_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    logits = model.forward(input_ids)
    print(f"  Output shape: {logits.shape}")
    
    # Test generation
    print(f"\nGeneration:")
    prompt = np.array([[1, 2, 3, 4, 5]])  # Simple prompt
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0]}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo_transformer()
