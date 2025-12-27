#!/usr/bin/env python3
"""
Pentary World Model Implementation
World Models for Model-Based RL adapted for Pentary (base-5) computing

Based on: 
- "World Models" (Ha & Schmidhuber, 2018) - arXiv:1803.10122
- "DreamerV3" (Hafner et al., 2023) - arXiv:2301.04104
- "IRIS" (Micheli et al., 2022) - arXiv:2209.00588

Key Features:
- Latent dynamics model learning
- Imagination-based planning
- Discrete latent space (natural for pentary)
- Efficient shift-add operations only
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


class PentaryEncoder:
    """
    Encoder for World Model.
    
    Encodes observations into pentary latent representations.
    Uses convolutional architecture for visual inputs.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        num_layers: int = 4
    ):
        """
        Initialize Encoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent representation
            num_layers: Number of conv layers
        """
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Build convolutional layers - adjust for small images
        self.conv_weights = []
        # Use smaller channel counts for efficiency
        base_channels = 16
        channels = [input_channels] + [min(base_channels * (2 ** i), 256) for i in range(num_layers)]
        
        for i in range(num_layers):
            # 3x3 kernels for better compatibility with small images
            W = self._init_pentary_weights(channels[i], channels[i+1], 3, 3)
            self.conv_weights.append(W)
        
        # Final projection to latent
        self.proj_weight = self._init_pentary_weights(channels[-1], latent_dim)
        self.proj_scale = 0.1
    
    def _init_pentary_weights(self, *shape) -> np.ndarray:
        """Initialize pentary weights."""
        weights = np.random.randn(*shape) * 0.1
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _pentary_conv2d(
        self,
        x: np.ndarray,
        W: np.ndarray,
        stride: int = 2
    ) -> np.ndarray:
        """
        2D convolution with pentary weights.
        
        Args:
            x: Input (batch, height, width, channels)
            W: Weights (in_channels, out_channels, kH, kW)
            stride: Stride for convolution
        
        Returns:
            Output (batch, new_height, new_width, out_channels)
        """
        batch_size, H, W_dim, in_channels = x.shape
        in_ch, out_ch, kH, kW = W.shape
        
        out_H = max(1, (H - kH) // stride + 1)
        out_W = max(1, (W_dim - kW) // stride + 1)
        
        output = np.zeros((batch_size, out_H, out_W, out_ch), dtype=np.float32)
        
        for b in range(batch_size):
            for i in range(out_H):
                for j in range(out_W):
                    for oc in range(out_ch):
                        val = 0.0
                        for ic in range(min(in_ch, in_channels)):
                            for ki in range(kH):
                                for kj in range(kW):
                                    h_idx = i * stride + ki
                                    w_idx = j * stride + kj
                                    if h_idx < H and w_idx < W_dim:
                                        w = W[ic, oc, ki, kj]
                                        if w != 0:
                                            val += w * x[b, h_idx, w_idx, ic]
                        output[b, i, j, oc] = val
        
        return output
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(x, 0)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Encode observation to latent.
        
        Args:
            x: Observation (batch_size, H, W, C) normalized to [-1, 1]
        
        Returns:
            z: Latent representation (batch_size, latent_dim)
        """
        h = x
        
        # Convolutional layers with ReLU
        for W in self.conv_weights:
            h = self._pentary_conv2d(h, W, stride=2)
            h = self._relu(h)
        
        # Global average pooling
        batch_size = h.shape[0]
        h = np.mean(h, axis=(1, 2))  # (batch, channels)
        
        # Project to latent
        z = np.zeros((batch_size, self.latent_dim), dtype=np.float32)
        for i in range(self.latent_dim):
            for j in range(min(self.proj_weight.shape[0], h.shape[1])):
                w = self.proj_weight[j, i] if i < self.proj_weight.shape[1] else 0
                if w != 0:
                    z[:, i] += w * h[:, j]
        
        return z * self.proj_scale


class PentaryDecoder:
    """
    Decoder for World Model.
    
    Decodes pentary latent representations back to observations.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        output_channels: int = 3,
        output_size: int = 64,
        num_layers: int = 4
    ):
        """
        Initialize Decoder.
        
        Args:
            latent_dim: Dimension of latent representation
            output_channels: Number of output channels
            output_size: Size of output image
            num_layers: Number of deconv layers
        """
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Adjust num_layers based on output_size to avoid too small initial size
        max_layers = max(1, int(np.log2(output_size)) - 1)
        num_layers = min(num_layers, max_layers)
        
        # Initial projection
        self.initial_size = max(2, output_size // (2 ** num_layers))
        self.initial_channels = min(256, 16 * (2 ** (num_layers - 1)))
        
        self.proj_weight = self._init_pentary_weights(
            latent_dim, 
            self.initial_channels * self.initial_size * self.initial_size
        )
        
        # Transposed conv layers
        self.deconv_weights = []
        channels = [self.initial_channels]
        for i in range(num_layers - 1, -1, -1):
            out_ch = 32 * (2 ** (i - 1)) if i > 0 else output_channels
            channels.append(max(out_ch, output_channels))
        
        for i in range(num_layers):
            W = self._init_pentary_weights(channels[i], channels[i+1], 4, 4)
            self.deconv_weights.append(W)
        
        self.proj_scale = 0.1
    
    def _init_pentary_weights(self, *shape) -> np.ndarray:
        """Initialize pentary weights."""
        weights = np.random.randn(*shape) * 0.1
        quantized, _ = PentaryQuantizer.quantize(weights)
        return quantized
    
    def _pentary_deconv2d(
        self,
        x: np.ndarray,
        W: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Transposed 2D convolution (upsampling).
        
        Simplified: nearest neighbor upsampling + regular conv.
        """
        batch_size = x.shape[0]
        out_H, out_W = output_size
        in_ch, out_ch, kH, kW = W.shape
        
        # Upsample 2x using nearest neighbor
        h = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        
        # Pad for same convolution
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        h_padded = np.pad(h, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), 'constant')
        
        output = np.zeros((batch_size, out_H, out_W, out_ch), dtype=np.float32)
        
        for b in range(batch_size):
            for i in range(min(out_H, h_padded.shape[1] - kH + 1)):
                for j in range(min(out_W, h_padded.shape[2] - kW + 1)):
                    for oc in range(out_ch):
                        val = 0.0
                        for ic in range(min(in_ch, h_padded.shape[3])):
                            for ki in range(kH):
                                for kj in range(kW):
                                    w = W[ic, oc, ki, kj]
                                    if w != 0:
                                        val += w * h_padded[b, i + ki, j + kj, ic]
                        output[b, i, j, oc] = val
        
        return output
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(x, 0)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation for output."""
        return np.tanh(x)
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent to observation.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
        
        Returns:
            x: Reconstructed observation (batch_size, H, W, C)
        """
        batch_size = z.shape[0]
        
        # Project and reshape
        h = np.zeros(
            (batch_size, self.initial_channels * self.initial_size * self.initial_size),
            dtype=np.float32
        )
        for i in range(h.shape[1]):
            for j in range(min(self.proj_weight.shape[0], z.shape[1])):
                w = self.proj_weight[j, i] if i < self.proj_weight.shape[1] else 0
                if w != 0:
                    h[:, i] += w * z[:, j]
        
        h = h * self.proj_scale
        h = h.reshape(batch_size, self.initial_size, self.initial_size, self.initial_channels)
        
        # Transposed convolutions
        current_size = self.initial_size
        for i, W in enumerate(self.deconv_weights):
            current_size *= 2
            out_size = (min(current_size, self.output_size), 
                       min(current_size, self.output_size))
            h = self._pentary_deconv2d(h, W, out_size)
            
            if i < len(self.deconv_weights) - 1:
                h = self._relu(h)
        
        # Tanh for output normalization
        return self._tanh(h)


class PentaryRSSM:
    """
    Recurrent State Space Model for World Model.
    
    The core dynamics model that predicts future latent states.
    Uses a combination of deterministic and stochastic states.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 256,
        stoch_dim: int = 32,
        num_categories: int = 5  # Natural for pentary!
    ):
        """
        Initialize RSSM.
        
        Args:
            latent_dim: Dimension of encoded observations
            action_dim: Dimension of action space
            hidden_dim: Hidden state dimension (deterministic)
            stoch_dim: Stochastic state dimension
            num_categories: Number of categories for discrete stochastic state
        """
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = num_categories  # 5 is perfect for pentary!
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize RSSM parameters."""
        # GRU for deterministic state
        concat_dim = self.stoch_dim * self.num_categories + self.action_dim
        
        self.W_z = self._init_pentary_weights(concat_dim + self.hidden_dim, self.hidden_dim)
        self.W_r = self._init_pentary_weights(concat_dim + self.hidden_dim, self.hidden_dim)
        self.W_h = self._init_pentary_weights(concat_dim + self.hidden_dim, self.hidden_dim)
        
        # Prior (imagination): predict stochastic from deterministic
        self.W_prior = self._init_pentary_weights(
            self.hidden_dim, 
            self.stoch_dim * self.num_categories
        )
        
        # Posterior (observation): predict stochastic from deterministic + observation
        self.W_posterior = self._init_pentary_weights(
            self.hidden_dim + self.latent_dim,
            self.stoch_dim * self.num_categories
        )
        
        self.scale = 0.1
    
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
        
        return output * self.scale
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _gumbel_softmax(
        self, 
        logits: np.ndarray, 
        temperature: float = 1.0,
        hard: bool = True
    ) -> np.ndarray:
        """
        Gumbel-softmax for discrete sampling.
        
        Perfect for pentary with 5 categories!
        """
        # Sample Gumbel noise
        U = np.random.uniform(0.001, 0.999, logits.shape)
        gumbel = -np.log(-np.log(U))
        
        # Softmax with temperature
        y = self._softmax((logits + gumbel) / temperature, axis=-1)
        
        if hard:
            # Straight-through estimator
            indices = np.argmax(y, axis=-1)
            y_hard = np.zeros_like(y)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    y_hard[i, j, indices[i, j]] = 1.0
            y = y_hard - y + y  # Gradient flows through soft
        
        return y
    
    def _gru_cell(
        self,
        x: np.ndarray,
        h: np.ndarray
    ) -> np.ndarray:
        """GRU cell with pentary weights."""
        concat = np.concatenate([x, h], axis=-1)
        
        z = self._sigmoid(self._pentary_matmul(concat, self.W_z))
        r = self._sigmoid(self._pentary_matmul(concat, self.W_r))
        
        concat_reset = np.concatenate([x, r * h], axis=-1)
        h_tilde = self._tanh(self._pentary_matmul(concat_reset, self.W_h))
        
        h_new = (1 - z) * h + z * h_tilde
        return h_new
    
    def imagine_step(
        self,
        h: np.ndarray,
        z: np.ndarray,
        action: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Imagination step (prior only, no observation).
        
        Used for planning and policy learning.
        
        Args:
            h: Deterministic state (batch_size, hidden_dim)
            z: Stochastic state (batch_size, stoch_dim, num_categories)
            action: Action (batch_size, action_dim)
        
        Returns:
            h_new: New deterministic state
            z_new: New stochastic state
            prior_logits: Logits for prior distribution
        """
        batch_size = h.shape[0]
        
        # Flatten z and concat with action
        z_flat = z.reshape(batch_size, -1)
        x = np.concatenate([z_flat, action], axis=-1)
        
        # GRU update
        h_new = self._gru_cell(x, h)
        
        # Prior: p(z|h)
        prior_logits = self._pentary_matmul(h_new, self.W_prior)
        prior_logits = prior_logits.reshape(batch_size, self.stoch_dim, self.num_categories)
        
        # Sample new stochastic state
        z_new = self._gumbel_softmax(prior_logits, temperature)
        
        return h_new, z_new, prior_logits
    
    def observe_step(
        self,
        h: np.ndarray,
        z: np.ndarray,
        action: np.ndarray,
        obs_embed: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Observation step (posterior, with observation).
        
        Used during training.
        
        Args:
            h: Deterministic state
            z: Stochastic state
            action: Action
            obs_embed: Encoded observation
        
        Returns:
            h_new: New deterministic state
            z_new: New stochastic state (from posterior)
            prior_logits: Logits for prior
            posterior_logits: Logits for posterior
        """
        batch_size = h.shape[0]
        
        # Flatten z and concat with action
        z_flat = z.reshape(batch_size, -1)
        x = np.concatenate([z_flat, action], axis=-1)
        
        # GRU update
        h_new = self._gru_cell(x, h)
        
        # Prior: p(z|h)
        prior_logits = self._pentary_matmul(h_new, self.W_prior)
        prior_logits = prior_logits.reshape(batch_size, self.stoch_dim, self.num_categories)
        
        # Posterior: q(z|h, obs)
        h_obs = np.concatenate([h_new, obs_embed], axis=-1)
        posterior_logits = self._pentary_matmul(h_obs, self.W_posterior)
        posterior_logits = posterior_logits.reshape(batch_size, self.stoch_dim, self.num_categories)
        
        # Sample from posterior
        z_new = self._gumbel_softmax(posterior_logits, temperature)
        
        return h_new, z_new, prior_logits, posterior_logits
    
    def initial_state(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial state."""
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        z = np.zeros((batch_size, self.stoch_dim, self.num_categories), dtype=np.float32)
        z[:, :, 0] = 1.0  # Initialize to first category
        return h, z


class PentaryRewardPredictor:
    """Predicts rewards from world model state."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize reward predictor.
        
        Args:
            state_dim: Dimension of combined state
            hidden_dim: Hidden layer dimension
        """
        self.W1 = self._init_pentary_weights(state_dim, hidden_dim)
        self.W2 = self._init_pentary_weights(hidden_dim, 1)
        self.scale = 0.1
    
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
        
        return output * self.scale
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Predict reward.
        
        Args:
            state: Combined state (batch_size, state_dim)
        
        Returns:
            reward: Predicted reward (batch_size, 1)
        """
        h = self._pentary_matmul(state, self.W1)
        h = np.maximum(h, 0)  # ReLU
        return self._pentary_matmul(h, self.W2)


class PentaryWorldModel:
    """
    Complete Pentary World Model.
    
    Combines:
    - Encoder: observations -> latent
    - RSSM: latent dynamics
    - Decoder: latent -> observations
    - Reward predictor: state -> reward
    
    Perfect for Pentary:
    - 5 categories in stochastic state match pentary levels
    - All operations use shift-add only
    - Discrete latent space is natural
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (64, 64, 3),
        action_dim: int = 4,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        stoch_dim: int = 32
    ):
        """
        Initialize World Model.
        
        Args:
            obs_shape: Observation shape (H, W, C)
            action_dim: Action dimension
            latent_dim: Encoded observation dimension
            hidden_dim: RSSM hidden dimension
            stoch_dim: Stochastic state dimension
        """
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = 5  # Pentary!
        
        # Components
        self.encoder = PentaryEncoder(
            input_channels=obs_shape[2],
            latent_dim=latent_dim
        )
        
        self.decoder = PentaryDecoder(
            latent_dim=hidden_dim + stoch_dim * self.num_categories,
            output_channels=obs_shape[2],
            output_size=obs_shape[0]
        )
        
        self.rssm = PentaryRSSM(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stoch_dim=stoch_dim,
            num_categories=self.num_categories
        )
        
        self.reward_predictor = PentaryRewardPredictor(
            state_dim=hidden_dim + stoch_dim * self.num_categories
        )
    
    def _get_combined_state(
        self, 
        h: np.ndarray, 
        z: np.ndarray
    ) -> np.ndarray:
        """Combine deterministic and stochastic states."""
        z_flat = z.reshape(z.shape[0], -1)
        return np.concatenate([h, z_flat], axis=-1)
    
    def encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode observation to latent.
        
        Args:
            obs: Observation (batch_size, H, W, C)
        
        Returns:
            z: Latent encoding (batch_size, latent_dim)
        """
        return self.encoder.forward(obs)
    
    def decode(self, h: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Decode state to observation.
        
        Args:
            h: Deterministic state
            z: Stochastic state
        
        Returns:
            obs: Reconstructed observation
        """
        state = self._get_combined_state(h, z)
        return self.decoder.forward(state)
    
    def imagine(
        self,
        initial_h: np.ndarray,
        initial_z: np.ndarray,
        actions: np.ndarray,
        horizon: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Imagine future trajectories.
        
        Used for planning and policy learning.
        
        Args:
            initial_h: Initial deterministic state
            initial_z: Initial stochastic state
            actions: Actions (batch_size, horizon, action_dim)
            horizon: Override horizon from actions
        
        Returns:
            Dictionary with imagined states and rewards
        """
        batch_size = initial_h.shape[0]
        if horizon is None:
            horizon = actions.shape[1]
        
        h = initial_h
        z = initial_z
        
        h_list = [h]
        z_list = [z]
        reward_list = []
        
        for t in range(horizon):
            action = actions[:, t, :]
            h, z, _ = self.rssm.imagine_step(h, z, action)
            
            state = self._get_combined_state(h, z)
            reward = self.reward_predictor.forward(state)
            
            h_list.append(h)
            z_list.append(z)
            reward_list.append(reward)
        
        return {
            'h': np.stack(h_list, axis=1),  # (batch, horizon+1, hidden_dim)
            'z': np.stack(z_list, axis=1),  # (batch, horizon+1, stoch_dim, num_cat)
            'rewards': np.stack(reward_list, axis=1)  # (batch, horizon, 1)
        }
    
    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Process observed trajectory.
        
        Used during training.
        
        Args:
            observations: Observations (batch_size, seq_len, H, W, C)
            actions: Actions (batch_size, seq_len-1, action_dim)
        
        Returns:
            Dictionary with states, priors, posteriors
        """
        batch_size, seq_len = observations.shape[0], observations.shape[1]
        
        # Encode all observations
        obs_embeds = []
        for t in range(seq_len):
            embed = self.encoder.forward(observations[:, t])
            obs_embeds.append(embed)
        obs_embeds = np.stack(obs_embeds, axis=1)
        
        # Initialize states
        h, z = self.rssm.initial_state(batch_size)
        
        h_list = [h]
        z_list = [z]
        prior_list = []
        posterior_list = []
        
        for t in range(seq_len - 1):
            action = actions[:, t, :]
            obs_embed = obs_embeds[:, t + 1, :]
            
            h, z, prior, posterior = self.rssm.observe_step(
                h, z, action, obs_embed
            )
            
            h_list.append(h)
            z_list.append(z)
            prior_list.append(prior)
            posterior_list.append(posterior)
        
        return {
            'h': np.stack(h_list, axis=1),
            'z': np.stack(z_list, axis=1),
            'priors': np.stack(prior_list, axis=1),
            'posteriors': np.stack(posterior_list, axis=1)
        }
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = 0
        zero_params = 0
        
        # Count encoder
        for W in self.encoder.conv_weights:
            total_params += W.size
            zero_params += np.sum(W == 0)
        total_params += self.encoder.proj_weight.size
        zero_params += np.sum(self.encoder.proj_weight == 0)
        
        # Count RSSM
        for W in [self.rssm.W_z, self.rssm.W_r, self.rssm.W_h, 
                  self.rssm.W_prior, self.rssm.W_posterior]:
            total_params += W.size
            zero_params += np.sum(W == 0)
        
        # Count decoder
        for W in self.decoder.deconv_weights:
            total_params += W.size
            zero_params += np.sum(W == 0)
        total_params += self.decoder.proj_weight.size
        zero_params += np.sum(self.decoder.proj_weight == 0)
        
        # Count reward predictor
        total_params += self.reward_predictor.W1.size
        total_params += self.reward_predictor.W2.size
        zero_params += np.sum(self.reward_predictor.W1 == 0)
        zero_params += np.sum(self.reward_predictor.W2 == 0)
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'sparsity': zero_params / total_params if total_params > 0 else 0,
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'stoch_dim': self.stoch_dim,
            'num_categories': self.num_categories
        }


def demo_pentary_world_model():
    """Demonstrate Pentary World Model."""
    print("=" * 70)
    print("Pentary World Model - Latent Dynamics Demo")
    print("=" * 70)
    
    # Create model
    model = PentaryWorldModel(
        obs_shape=(32, 32, 3),  # Smaller for demo
        action_dim=4,
        latent_dim=64,
        hidden_dim=64,
        stoch_dim=16
    )
    
    # Get stats
    stats = model.get_stats()
    print("\nModel Statistics:")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Zero Parameters: {stats['zero_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Observation Shape: {stats['obs_shape']}")
    print(f"  Action Dimension: {stats['action_dim']}")
    print(f"  Stochastic Categories: {stats['num_categories']} (pentary!)")
    
    # Test encoding
    print("\nEncoder Test:")
    batch_size = 2
    obs = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
    z = model.encode(obs)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Encoded shape: {z.shape}")
    
    # Test RSSM imagination
    print("\nImagination Test:")
    h, stoch_z = model.rssm.initial_state(batch_size)
    actions = np.random.randn(batch_size, 10, 4).astype(np.float32)
    
    imagined = model.imagine(h, stoch_z, actions)
    print(f"  Actions shape: {actions.shape}")
    print(f"  Imagined h shape: {imagined['h'].shape}")
    print(f"  Imagined z shape: {imagined['z'].shape}")
    print(f"  Imagined rewards shape: {imagined['rewards'].shape}")
    
    # Test observation processing
    print("\nObservation Processing Test:")
    observations = np.random.randn(batch_size, 5, 32, 32, 3).astype(np.float32)
    actions = np.random.randn(batch_size, 4, 4).astype(np.float32)
    
    observed = model.observe(observations, actions)
    print(f"  Observations shape: {observations.shape}")
    print(f"  Processed h shape: {observed['h'].shape}")
    print(f"  Prior logits shape: {observed['priors'].shape}")
    print(f"  Posterior logits shape: {observed['posteriors'].shape}")
    
    # Demonstrate pentary categorical distribution
    print("\nPentary Categorical Advantage:")
    print(f"  5 categories = 5 pentary levels {{-2, -1, 0, +1, +2}}")
    print(f"  Each stochastic variable has log_5(5) = 1 pentary digit")
    print(f"  Total stochastic info: {stats['stoch_dim']} pentary digits")
    
    # Benchmark
    print("\nBenchmark (10 imagination steps):")
    import time
    
    h, stoch_z = model.rssm.initial_state(1)
    action = np.random.randn(1, 4).astype(np.float32)
    
    start = time.time()
    for _ in range(10):
        h, stoch_z, _ = model.rssm.imagine_step(h, stoch_z, action)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {10 / elapsed:.1f} steps/sec")
    
    print("\n" + "=" * 70)
    print("Pentary World Model: Perfect alignment with 5-level quantization!")
    print("=" * 70)


if __name__ == "__main__":
    demo_pentary_world_model()
