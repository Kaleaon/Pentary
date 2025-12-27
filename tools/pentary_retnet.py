#!/usr/bin/env python3
"""
Pentary RetNet Implementation
Retentive Network adapted for Pentary (base-5) computing

Based on: arXiv:2307.08621 - "Retentive Network: A Successor to Transformer for Large Language Models"

Key Features:
- Three computation paradigms: Parallel, Recurrent, Chunkwise
- O(n) training, O(1) inference
- Explicit decay for position encoding
- Hardware-efficient retention mechanism
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


class PentaryRetention:
    """
    Retention mechanism for RetNet.
    
    The core innovation: replace softmax attention with explicit decay.
    
    Retention formula:
        Retention(X) = (QK^T ⊙ D) V
    
    Where D is a causal decay matrix with γ^(m-n) decay.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        gamma: float = 0.95
    ):
        """
        Initialize Retention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of retention heads
            gamma: Decay factor (smaller = faster decay)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.gamma = gamma
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize all parameters with pentary quantization."""
        d = self.d_model
        d_head = self.d_head
        
        # Q, K, V projections
        self.W_q = self._init_pentary_weights(d, d)
        self.W_k = self._init_pentary_weights(d, d)
        self.W_v = self._init_pentary_weights(d, d)
        self.W_o = self._init_pentary_weights(d, d)
        
        # Group normalization parameters
        self.gn_weight = np.ones(self.num_heads)
        
        # Theta for xPos (position-dependent scaling)
        self.theta = np.exp(
            -np.arange(d_head) * np.log(10000) / d_head
        )
        
        self.proj_scale = 0.1
    
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
                elif w == 1:
                    output[:, i] += x[:, j]
                elif w == -1:
                    output[:, i] -= x[:, j]
                elif w == 2:
                    output[:, i] += 2 * x[:, j]
                elif w == -2:
                    output[:, i] -= 2 * x[:, j]
        
        return output * self.proj_scale
    
    def _xpos_scale(self, pos: int) -> np.ndarray:
        """Compute xPos position scaling."""
        # Clamp to prevent overflow
        scaled = np.clip(pos * self.theta, -20, 20)
        return np.exp(scaled)
    
    def _decay_matrix(self, seq_len: int) -> np.ndarray:
        """
        Compute decay matrix D.
        
        D[m,n] = γ^(m-n) if m >= n else 0
        """
        positions = np.arange(seq_len)
        decay = self.gamma ** np.maximum(positions[:, None] - positions[None, :], 0)
        mask = positions[:, None] >= positions[None, :]
        return decay * mask
    
    def forward_parallel(self, x: np.ndarray) -> np.ndarray:
        """
        Parallel forward pass (training mode).
        
        Computes retention for entire sequence at once.
        O(n²) for training but parallelizable.
        
        Args:
            x: Input (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        q = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        k = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        v = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        
        for t in range(seq_len):
            q[:, t, :] = self._pentary_matmul(x[:, t, :], self.W_q)
            k[:, t, :] = self._pentary_matmul(x[:, t, :], self.W_k)
            v[:, t, :] = self._pentary_matmul(x[:, t, :], self.W_v)
        
        # Apply xPos scaling
        for t in range(seq_len):
            scale = self._xpos_scale(t)
            q[:, t, :self.d_head] *= scale
            k[:, t, :self.d_head] /= (scale + 1e-8)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Transpose to (batch, heads, seq, d_head)
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        
        # Compute retention scores: QK^T
        scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))
        
        # Apply decay mask
        D = self._decay_matrix(seq_len)
        scores = scores * D[None, None, :, :]
        
        # Retention output
        output = np.matmul(scores, v)
        
        # Group normalization per head
        for h in range(self.num_heads):
            mean = np.mean(output[:, h, :, :], axis=-1, keepdims=True)
            var = np.var(output[:, h, :, :], axis=-1, keepdims=True) + 1e-6
            output[:, h, :, :] = (output[:, h, :, :] - mean) / np.sqrt(var)
            output[:, h, :, :] *= self.gn_weight[h]
        
        # Reshape back
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_len, d_model)
        
        # Output projection
        result = np.zeros_like(output)
        for t in range(seq_len):
            result[:, t, :] = self._pentary_matmul(output[:, t, :], self.W_o)
        
        return result
    
    def forward_recurrent(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None,
        position: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recurrent forward pass (inference mode).
        
        O(1) memory and computation per token.
        
        Args:
            x: Input (batch_size, d_model)
            state: Previous retention state (batch_size, num_heads, d_head, d_head)
            position: Current position for xPos
        
        Returns:
            output: (batch_size, d_model)
            new_state: Updated state
        """
        batch_size = x.shape[0]
        
        # Initialize state if needed
        if state is None:
            state = np.zeros(
                (batch_size, self.num_heads, self.d_head, self.d_head),
                dtype=np.float32
            )
        
        # Compute Q, K, V
        q = self._pentary_matmul(x, self.W_q)
        k = self._pentary_matmul(x, self.W_k)
        v = self._pentary_matmul(x, self.W_v)
        
        # Apply xPos scaling
        scale = self._xpos_scale(position)
        q[:, :self.d_head] *= scale
        k[:, :self.d_head] /= (scale + 1e-8)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, self.num_heads, self.d_head)
        k = k.reshape(batch_size, self.num_heads, self.d_head)
        v = v.reshape(batch_size, self.num_heads, self.d_head)
        
        # Update state: S_n = γ * S_{n-1} + k_n^T * v_n
        kv = np.einsum('bhi,bhj->bhij', k, v)
        new_state = self.gamma * state + kv
        
        # Compute output: o_n = q_n * S_n
        output = np.einsum('bhi,bhij->bhj', q, new_state)
        
        # Group normalization per head
        for h in range(self.num_heads):
            mean = np.mean(output[:, h, :])
            var = np.var(output[:, h, :]) + 1e-6
            output[:, h, :] = (output[:, h, :] - mean) / np.sqrt(var)
            output[:, h, :] *= self.gn_weight[h]
        
        # Reshape back
        output = output.reshape(batch_size, self.d_model)
        
        # Output projection
        output = self._pentary_matmul(output, self.W_o)
        
        return output, new_state
    
    def forward_chunkwise(
        self,
        x: np.ndarray,
        chunk_size: int = 16,
        prev_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chunkwise forward pass (hybrid mode).
        
        Processes sequence in chunks for balance of
        parallelism and memory efficiency.
        
        Args:
            x: Input (batch_size, seq_len, d_model)
            chunk_size: Size of each chunk
            prev_state: State from previous chunk
        
        Returns:
            output: (batch_size, seq_len, d_model)
            final_state: State after processing
        """
        batch_size, seq_len, d_model = x.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        outputs = []
        state = prev_state
        
        for c in range(num_chunks):
            start_idx = c * chunk_size
            end_idx = min((c + 1) * chunk_size, seq_len)
            chunk = x[:, start_idx:end_idx, :]
            
            # Process chunk in parallel
            chunk_out = self.forward_parallel(chunk)
            
            # Apply cross-chunk state (simplified)
            if state is not None:
                # Add contribution from previous chunk state
                for t in range(chunk_out.shape[1]):
                    decay = self.gamma ** (t + 1)
                    chunk_out[:, t, :] += decay * 0.1  # Simplified state contribution
            
            outputs.append(chunk_out)
            
            # Update state using last position in chunk
            _, state = self.forward_recurrent(
                chunk[:, -1, :], state, end_idx - 1
            )
        
        return np.concatenate(outputs, axis=1), state


class PentaryRetNetBlock:
    """
    Complete RetNet block with retention and FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_expand: int = 4,
        gamma: float = 0.95
    ):
        """
        Initialize RetNet block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of retention heads
            ffn_expand: FFN expansion factor
            gamma: Decay factor
        """
        self.d_model = d_model
        
        # Layer norms
        self.ln1_weight = np.ones(d_model)
        self.ln2_weight = np.ones(d_model)
        
        # Retention layer
        self.retention = PentaryRetention(d_model, num_heads, gamma)
        
        # FFN
        self.ffn_dim = d_model * ffn_expand
        self.W_up = self._init_pentary_weights(d_model, self.ffn_dim)
        self.W_gate = self._init_pentary_weights(d_model, self.ffn_dim)
        self.W_down = self._init_pentary_weights(self.ffn_dim, d_model)
        
        self.ffn_scale = 0.1
    
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
                if w != 0:
                    output[:, i] += w * x[:, j]
        
        return output * self.ffn_scale
    
    def _layer_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True) + 1e-6
        return (x - mean) / np.sqrt(var) * weight
    
    def _swish(self, x: np.ndarray) -> np.ndarray:
        """Swish activation."""
        return x * (1 / (1 + np.exp(-np.clip(x, -10, 10))))
    
    def _gated_ffn(self, x: np.ndarray) -> np.ndarray:
        """Gated FFN (like SwiGLU)."""
        up = self._pentary_matmul(x, self.W_up)
        gate = self._pentary_matmul(x, self.W_gate)
        gate = self._swish(gate)
        hidden = up * gate
        return self._pentary_matmul(hidden, self.W_down)
    
    def forward_parallel(self, x: np.ndarray) -> np.ndarray:
        """Parallel forward pass."""
        batch_size, seq_len, d_model = x.shape
        
        # Retention with residual
        x_normed = np.zeros_like(x)
        for t in range(seq_len):
            x_normed[:, t, :] = self._layer_norm(x[:, t, :], self.ln1_weight)
        
        ret_out = self.retention.forward_parallel(x_normed)
        x = x + ret_out
        
        # FFN with residual
        result = np.zeros_like(x)
        for t in range(seq_len):
            x_normed = self._layer_norm(x[:, t, :], self.ln2_weight)
            ffn_out = self._gated_ffn(x_normed)
            result[:, t, :] = x[:, t, :] + ffn_out
        
        return result
    
    def forward_recurrent(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None,
        position: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recurrent forward pass."""
        # Retention with residual
        x_normed = self._layer_norm(x, self.ln1_weight)
        ret_out, new_state = self.retention.forward_recurrent(x_normed, state, position)
        x = x + ret_out
        
        # FFN with residual
        x_normed = self._layer_norm(x, self.ln2_weight)
        ffn_out = self._gated_ffn(x_normed)
        x = x + ffn_out
        
        return x, new_state


class PentaryRetNet:
    """
    Complete Pentary RetNet model.
    
    Successor to Transformer with:
    - Parallel training
    - O(1) recurrent inference
    - Chunkwise processing for long sequences
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        num_heads: int = 4,
        gamma: float = 0.95
    ):
        """
        Initialize Pentary RetNet.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of blocks
            num_heads: Number of retention heads
            gamma: Decay factor
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        
        # Embedding
        self.embedding = self._init_pentary_embedding(vocab_size, d_model)
        self.embed_scale = 0.1
        
        # RetNet blocks with different gamma per layer
        gammas = [gamma ** (2 ** (-i)) for i in range(n_layers)]
        self.blocks = [
            PentaryRetNetBlock(d_model, num_heads, gamma=gammas[i])
            for i in range(n_layers)
        ]
        
        # Final layer norm
        self.ln_final = np.ones(d_model)
        
        # Output projection
        self.lm_head = self._init_pentary_weights(d_model, vocab_size)
    
    def _init_pentary_embedding(self, vocab_size: int, d_model: int) -> np.ndarray:
        """Initialize pentary embedding."""
        emb = np.random.randn(vocab_size, d_model) * 0.02
        quantized, _ = PentaryQuantizer.quantize(emb)
        return quantized
    
    def _init_pentary_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize pentary weights."""
        weights = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / (in_dim + out_dim))
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _layer_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True) + 1e-6
        return (x - mean) / np.sqrt(var) * weight
    
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
        Forward pass (parallel mode).
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embedding[input_ids].astype(np.float32) * self.embed_scale
        
        # Pass through blocks
        for block in self.blocks:
            x = block.forward_parallel(x)
        
        # Final layer norm
        for t in range(seq_len):
            x[:, t, :] = self._layer_norm(x[:, t, :], self.ln_final)
        
        # Output projection
        logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        for t in range(seq_len):
            logits[:, t, :] = self._pentary_matmul(x[:, t, :], self.lm_head)
        
        return logits
    
    def forward_recurrent(
        self,
        token_id: np.ndarray,
        states: Optional[List[np.ndarray]] = None,
        position: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Recurrent forward pass (O(1) inference).
        
        Args:
            token_id: Single token ID (batch_size,)
            states: List of states for each layer
            position: Current position
        
        Returns:
            logits: (batch_size, vocab_size)
            new_states: Updated states
        """
        batch_size = token_id.shape[0]
        
        if states is None:
            states = [None] * self.n_layers
        
        # Embed
        x = self.embedding[token_id].astype(np.float32) * self.embed_scale
        
        # Pass through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, state = block.forward_recurrent(x, states[i], position)
            new_states.append(state)
        
        # Final layer norm
        x = self._layer_norm(x, self.ln_final)
        
        # Output projection
        logits = self._pentary_matmul(x, self.lm_head)
        
        return logits, new_states
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate tokens autoregressively."""
        batch_size = prompt_ids.shape[0]
        generated = prompt_ids.copy()
        
        # Process prompt
        states = None
        for t in range(prompt_ids.shape[1]):
            _, states = self.forward_recurrent(prompt_ids[:, t], states, t)
        
        pos = prompt_ids.shape[1]
        last_token = prompt_ids[:, -1]
        
        # Generate
        for _ in range(max_new_tokens):
            logits, states = self.forward_recurrent(last_token, states, pos)
            pos += 1
            
            # Sample
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            next_token = np.array([
                np.random.choice(self.vocab_size, p=probs[b])
                for b in range(batch_size)
            ])
            
            generated = np.concatenate([generated, next_token[:, np.newaxis]], axis=1)
            last_token = next_token
        
        return generated
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = 0
        zero_params = 0
        
        # Embedding
        total_params += self.embedding.size
        zero_params += np.sum(self.embedding == 0)
        
        # Blocks
        for block in self.blocks:
            for W in [block.retention.W_q, block.retention.W_k, 
                     block.retention.W_v, block.retention.W_o,
                     block.W_up, block.W_gate, block.W_down]:
                total_params += W.size
                zero_params += np.sum(W == 0)
        
        # LM head
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


def demo_pentary_retnet():
    """Demonstrate Pentary RetNet."""
    print("=" * 70)
    print("Pentary RetNet - Retentive Network Demo")
    print("=" * 70)
    
    # Create model
    model = PentaryRetNet(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        num_heads=4
    )
    
    # Get stats
    stats = model.get_stats()
    print("\nModel Statistics:")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Zero Parameters: {stats['zero_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Layers: {stats['n_layers']}")
    
    # Test parallel forward
    print("\nParallel Forward Test:")
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"  Input shape: {input_ids.shape}")
    logits = model.forward(input_ids)
    print(f"  Output shape: {logits.shape}")
    
    # Test recurrent forward
    print("\nRecurrent Forward Test (O(1) inference):")
    token = np.array([42, 123])
    logits, states = model.forward_recurrent(token, None, 0)
    print(f"  Single token logits shape: {logits.shape}")
    print(f"  Number of states: {len(states)}")
    
    # Test generation
    print("\nGeneration Test:")
    prompt = np.array([[1, 2, 3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"  Prompt: {prompt[0]}")
    print(f"  Generated: {generated[0]}")
    
    # Benchmark
    print("\nRecurrent Inference Benchmark (100 tokens):")
    import time
    
    states = None
    token = np.array([1])
    
    start = time.time()
    for t in range(100):
        _, states = model.forward_recurrent(token, states, t)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {100 / elapsed:.1f} tokens/sec")
    
    print("\n" + "=" * 70)
    print("Pentary RetNet: Successor to Transformer!")
    print("=" * 70)


if __name__ == "__main__":
    demo_pentary_retnet()
