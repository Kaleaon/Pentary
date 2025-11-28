#!/usr/bin/env python3
"""
Pentary Neural Network Trainer
Training framework with pentary arithmetic and multimodal support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
import json
import os
from datetime import datetime

from pentary_nn_layers import (
    PentaryLinear, PentaryReLU, PentaryConv2D, PentaryMaxPool2D,
    PentaryBatchNorm, PentaryQuantizer
)
from pentary_multimodal import MultimodalFusion, MultimodalProcessor


class PentaryModel:
    """Complete neural network model with pentary layers"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        """
        Initialize pentary model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims else [128, 64]
        
        self.layers = []
        self._build_model()
    
    def _build_model(self):
        """Build the model architecture"""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            linear = PentaryLinear(dims[i], dims[i + 1], bias=True)
            linear.initialize_weights('xavier')
            self.layers.append(('linear', linear))
            
            # Activation (except last layer)
            if i < len(dims) - 2:
                self.layers.append(('relu', PentaryReLU()))
    
    def forward(self, x: np.ndarray, use_pentary: bool = True) -> np.ndarray:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            use_pentary: Use pentary arithmetic
            
        Returns:
            Output tensor
        """
        output = x
        
        for layer_type, layer in self.layers:
            if layer_type == 'linear':
                output = layer.forward(output, use_pentary=use_pentary)
            elif layer_type == 'relu':
                output = layer.forward(output)
        
        return output
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> List[Tuple]:
        """
        Backward pass through the model.
        
        Args:
            grad_output: Gradient w.r.t. output
            x: Input tensor
            
        Returns:
            List of gradients for each layer
        """
        gradients = []
        current_grad = grad_output
        
        # Forward pass to get intermediate activations
        activations = [x]
        output = x
        
        for layer_type, layer in self.layers:
            if layer_type == 'linear':
                output = layer.forward(output, use_pentary=False)
                activations.append(output)
            elif layer_type == 'relu':
                output = layer.forward(output)
                activations.append(output)
        
        # Backward pass
        for i in range(len(self.layers) - 1, -1, -1):
            layer_type, layer = self.layers[i]
            
            if layer_type == 'relu':
                current_grad = layer.backward(current_grad, activations[i])
            elif layer_type == 'linear':
                grad_input, grad_weight, grad_bias = layer.backward(
                    current_grad, activations[i]
                )
                gradients.insert(0, (grad_weight, grad_bias))
                current_grad = grad_input
        
        return gradients
    
    def update_weights(self, gradients: List[Tuple], learning_rate: float):
        """Update weights using gradients"""
        linear_idx = 0
        for layer_type, layer in self.layers:
            if layer_type == 'linear':
                grad_weight, grad_bias = gradients[linear_idx]
                layer.update_weights(grad_weight, grad_bias, learning_rate)
                linear_idx += 1


class LossFunction:
    """Loss functions for training"""
    
    @staticmethod
    def mse_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Mean Squared Error loss.
        
        Args:
            pred: Predictions
            target: Targets
            
        Returns:
            (loss_value, gradient)
        """
        loss = np.mean((pred - target) ** 2)
        grad = 2 * (pred - target) / pred.size
        return loss, grad
    
    @staticmethod
    def cross_entropy_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Cross-entropy loss (with softmax).
        
        Args:
            pred: Logits (before softmax)
            target: One-hot encoded targets or class indices
            
        Returns:
            (loss_value, gradient)
        """
        # Softmax
        exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
        
        # Handle target format
        if len(target.shape) == 1:
            # Convert indices to one-hot
            one_hot = np.zeros_like(softmax)
            one_hot[np.arange(len(target)), target] = 1
            target = one_hot
        
        # Cross-entropy
        loss = -np.mean(np.sum(target * np.log(softmax + 1e-10), axis=-1))
        
        # Gradient
        grad = (softmax - target) / pred.shape[0]
        
        return loss, grad


class PentaryTrainer:
    """Training framework for pentary neural networks"""
    
    def __init__(self, model: PentaryModel, loss_fn: Callable = LossFunction.mse_loss,
                 optimizer: str = 'sgd', learning_rate: float = 0.01,
                 use_multimodal: bool = False, modalities: List[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: Pentary model to train
            loss_fn: Loss function
            optimizer: Optimizer type ('sgd', 'adam')
            learning_rate: Learning rate
            use_multimodal: Whether to use multimodal processing
            modalities: List of modalities if multimodal
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_multimodal = use_multimodal
        
        # Multimodal fusion
        if use_multimodal and modalities:
            self.fusion = MultimodalFusion(modalities, fusion_method='concat')
        else:
            self.fusion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Single training step.
        
        Args:
            x: Input data
            y: Target labels
            
        Returns:
            (loss, accuracy)
        """
        # Forward pass
        pred = self.model.forward(x, use_pentary=False)  # Use FP for training
        
        # Compute loss
        loss, grad_loss = self.loss_fn(pred, y)
        
        # Backward pass
        gradients = self.model.backward(grad_loss, x)
        
        # Update weights
        self.model.update_weights(gradients, self.learning_rate)
        
        # Compute accuracy (for classification)
        if len(y.shape) == 1:  # Class indices
            accuracy = np.mean(np.argmax(pred, axis=-1) == y)
        else:  # One-hot
            accuracy = np.mean(np.argmax(pred, axis=-1) == np.argmax(y, axis=-1))
        
        return loss, accuracy
    
    def validate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Validation step.
        
        Args:
            x: Input data
            y: Target labels
            
        Returns:
            (loss, accuracy)
        """
        # Forward pass (use pentary for inference)
        pred = self.model.forward(x, use_pentary=True)
        
        # Compute loss
        loss, _ = self.loss_fn(pred, y)
        
        # Compute accuracy
        if len(y.shape) == 1:
            accuracy = np.mean(np.argmax(pred, axis=-1) == y)
        else:
            accuracy = np.mean(np.argmax(pred, axis=-1) == np.argmax(y, axis=-1))
        
        return loss, accuracy
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 10, batch_size: int = 32, verbose: bool = True):
        """
        Train the model.
        
        Args:
            train_data: (x_train, y_train) tuple
            val_data: Optional (x_val, y_val) tuple
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress
        """
        x_train, y_train = train_data
        
        if verbose:
            print("=" * 70)
            print("Training Pentary Neural Network")
            print("=" * 70)
            print(f"Model: {self.model.input_dim} -> {self.model.hidden_dims} -> {self.model.output_dim}")
            print(f"Epochs: {epochs}, Batch size: {batch_size}")
            print(f"Learning rate: {self.learning_rate}")
            print("=" * 70)
        
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            # Training batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Process multimodal data if needed
                if self.use_multimodal and self.fusion:
                    x_batch = self._process_multimodal_batch(x_batch)
                
                loss, acc = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                epoch_acc += acc
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            
            # Validation
            val_loss = None
            val_acc = None
            if val_data is not None:
                x_val, y_val = val_data
                if self.use_multimodal and self.fusion:
                    x_val = self._process_multimodal_batch(x_val)
                val_loss, val_acc = self.validate(x_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}", end="")
                if val_loss is not None:
                    print(f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print()
        
        if verbose:
            print("=" * 70)
            print("Training completed!")
            print("=" * 70)
    
    def _process_multimodal_batch(self, batch_data: Union[List, np.ndarray]) -> np.ndarray:
        """Process a batch of multimodal data"""
        if isinstance(batch_data, list):
            processed = []
            for item in batch_data:
                if isinstance(item, dict):
                    processed.append(self.fusion.process(item))
                else:
                    processed.append(item)
            return np.array(processed)
        return batch_data
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'input_dim': self.model.input_dim,
            'output_dim': self.model.output_dim,
            'hidden_dims': self.model.hidden_dims,
            'layers': []
        }
        
        for layer_type, layer in self.model.layers:
            if layer_type == 'linear':
                layer_data = {
                    'type': 'linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features,
                    'use_bias': layer.use_bias,
                    'weight_pentary': layer.weight_pentary.tolist(),
                    'bias_pentary': layer.bias_pentary.tolist() if layer.bias_pentary is not None else None,
                    'weight_scale': layer.weight_scale,
                    'bias_scale': layer.bias_scale
                }
                model_data['layers'].append(layer_data)
            elif layer_type == 'relu':
                model_data['layers'].append({'type': 'relu'})
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.model.input_dim = model_data['input_dim']
        self.model.output_dim = model_data['output_dim']
        self.model.hidden_dims = model_data['hidden_dims']
        self.model.layers = []
        
        for layer_data in model_data['layers']:
            if layer_data['type'] == 'linear':
                linear = PentaryLinear(
                    layer_data['in_features'],
                    layer_data['out_features'],
                    layer_data['use_bias']
                )
                linear.weight_pentary = np.array(layer_data['weight_pentary'], dtype=np.int32)
                linear.bias_pentary = np.array(layer_data['bias_pentary'], dtype=np.int32) if layer_data['bias_pentary'] else None
                linear.weight_scale = layer_data['weight_scale']
                linear.bias_scale = layer_data['bias_scale']
                self.model.layers.append(('linear', linear))
            elif layer_data['type'] == 'relu':
                self.model.layers.append(('relu', PentaryReLU()))


def main():
    """Test pentary trainer"""
    print("=" * 70)
    print("Pentary Neural Network Trainer Test")
    print("=" * 70)
    
    # Create simple dataset
    n_samples = 1000
    input_dim = 64
    output_dim = 10
    
    # Generate random data
    x_train = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_train = np.random.randint(0, output_dim, n_samples)
    
    # Create model
    model = PentaryModel(input_dim=input_dim, output_dim=output_dim, hidden_dims=[128, 64])
    
    # Create trainer
    trainer = PentaryTrainer(
        model=model,
        loss_fn=LossFunction.cross_entropy_loss,
        learning_rate=0.01
    )
    
    # Split data
    split_idx = int(0.8 * n_samples)
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Train
    trainer.train(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=True
    )
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
