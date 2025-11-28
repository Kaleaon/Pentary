#!/usr/bin/env python3
"""
Pentary Model Utilities
Helper functions for model deployment and inference
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pentary_trainer import PentaryModel, PentaryTrainer
from pentary_simulator import PentaryProcessor


class PentaryModelInference:
    """Inference engine for pentary models"""
    
    def __init__(self, model_path: Optional[str] = None, model: Optional[PentaryModel] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved model JSON file
            model: PentaryModel instance (alternative to loading from file)
        """
        if model_path:
            trainer = PentaryTrainer(model=PentaryModel(1, 1))  # Dummy model
            trainer.load_model(model_path)
            self.model = trainer.model
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")
    
    def predict(self, x: np.ndarray, use_pentary: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data
            use_pentary: Use pentary arithmetic for inference
            
        Returns:
            Predictions
        """
        return self.model.forward(x, use_pentary=use_pentary)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (with softmax).
        
        Args:
            x: Input data
            
        Returns:
            Class probabilities
        """
        logits = self.predict(x, use_pentary=True)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs
    
    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input data
            
        Returns:
            Predicted class indices
        """
        logits = self.predict(x, use_pentary=True)
        return np.argmax(logits, axis=-1)


class PentaryModelAnalyzer:
    """Analyze pentary models"""
    
    @staticmethod
    def count_parameters(model: PentaryModel) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PentaryModel instance
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        
        for layer_type, layer in model.layers:
            if layer_type == 'linear':
                # Weights
                weight_params = layer.in_features * layer.out_features
                total_params += weight_params
                trainable_params += weight_params
                
                # Bias
                if layer.use_bias:
                    bias_params = layer.out_features
                    total_params += bias_params
                    trainable_params += bias_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': 0
        }
    
    @staticmethod
    def get_model_size(model_path: str) -> Dict[str, float]:
        """
        Get model size information.
        
        Args:
            model_path: Path to model JSON file
            
        Returns:
            Dictionary with size information
        """
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Count pentary weights
        total_pentary_weights = 0
        total_scales = 0
        
        for layer in model_data.get('layers', []):
            if layer['type'] == 'linear':
                if 'weight_pentary' in layer:
                    weights = np.array(layer['weight_pentary'])
                    total_pentary_weights += weights.size
                    total_scales += 1
                
                if 'bias_pentary' in layer and layer['bias_pentary']:
                    bias = np.array(layer['bias_pentary'])
                    total_pentary_weights += bias.size
                    total_scales += 1
        
        # Estimate sizes
        # Pentary weights: 3 bits per value (5 levels = log2(5) ≈ 2.32 bits)
        # But stored as int32 in JSON, so 4 bytes per value
        # Scales: float32 = 4 bytes each
        
        weight_size_bytes = total_pentary_weights * 4  # int32
        scale_size_bytes = total_scales * 4  # float32
        total_size_bytes = weight_size_bytes + scale_size_bytes
        
        # If quantized to actual pentary (3 bits), would be:
        theoretical_size_bits = total_pentary_weights * 3  # 3 bits per pentary value
        theoretical_size_bytes = theoretical_size_bits / 8
        
        return {
            'total_pentary_weights': total_pentary_weights,
            'total_scales': total_scales,
            'current_size_bytes': total_size_bytes,
            'current_size_mb': total_size_bytes / (1024 * 1024),
            'theoretical_size_bytes': theoretical_size_bytes,
            'theoretical_size_mb': theoretical_size_bytes / (1024 * 1024),
            'compression_ratio': total_size_bytes / theoretical_size_bytes if theoretical_size_bytes > 0 else 0
        }
    
    @staticmethod
    def analyze_sparsity(model_path: str) -> Dict[str, float]:
        """
        Analyze weight sparsity (zero values).
        
        Args:
            model_path: Path to model JSON file
            
        Returns:
            Dictionary with sparsity information
        """
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        total_weights = 0
        zero_weights = 0
        
        for layer in model_data.get('layers', []):
            if layer['type'] == 'linear':
                if 'weight_pentary' in layer:
                    weights = np.array(layer['weight_pentary'])
                    total_weights += weights.size
                    zero_weights += np.sum(weights == 0)
                
                if 'bias_pentary' in layer and layer['bias_pentary']:
                    bias = np.array(layer['bias_pentary'])
                    total_weights += bias.size
                    zero_weights += np.sum(bias == 0)
        
        sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
        
        return {
            'total_weights': total_weights,
            'zero_weights': zero_weights,
            'sparsity_ratio': sparsity,
            'sparsity_percent': sparsity * 100
        }


def export_to_processor_code(model_path: str, output_path: str):
    """
    Export model to pentary processor assembly code.
    
    Args:
        model_path: Path to model JSON file
        output_path: Output path for assembly file
    """
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    assembly_lines = [
        "# Pentary Processor Assembly Code",
        "# Generated from model: " + model_path,
        "",
        ".data",
        ""
    ]
    
    # Export weights and biases
    for i, layer in enumerate(model_data.get('layers', [])):
        if layer['type'] == 'linear':
            layer_name = f"layer_{i}"
            
            # Export weights
            weights = np.array(layer['weight_pentary'])
            assembly_lines.append(f"# Layer {i} weights ({weights.shape})")
            assembly_lines.append(f"{layer_name}_weights:")
            
            # Flatten weights for storage
            weights_flat = weights.flatten()
            for j, weight in enumerate(weights_flat):
                if j % 16 == 0:
                    assembly_lines.append(f"  .pent {weight}")
                else:
                    assembly_lines[-1] += f", {weight}"
            
            assembly_lines.append("")
            
            # Export bias
            if 'bias_pentary' in layer and layer['bias_pentary']:
                bias = np.array(layer['bias_pentary'])
                assembly_lines.append(f"# Layer {i} bias")
                assembly_lines.append(f"{layer_name}_bias:")
                assembly_lines.append(f"  .pent {', '.join(map(str, bias))}")
                assembly_lines.append("")
            
            # Export scales
            assembly_lines.append(f"# Layer {i} scales")
            assembly_lines.append(f"{layer_name}_weight_scale: .float {layer.get('weight_scale', 1.0)}")
            if 'bias_scale' in layer:
                assembly_lines.append(f"{layer_name}_bias_scale: .float {layer['bias_scale']}")
            assembly_lines.append("")
    
    # Export inference code
    assembly_lines.extend([
        ".text",
        "",
        "# Inference function",
        "inference:",
        "  # Load input to P1",
        "  LOADV P1, input_address, 0",
        ""
    ])
    
    # Generate forward pass code
    for i, layer in enumerate(model_data.get('layers', [])):
        if layer['type'] == 'linear':
            assembly_lines.extend([
                f"  # Layer {i}: Linear",
                f"  LOAD P2, {f'layer_{i}_weights'}, 0",
                f"  MATVEC P3, P2, P1",
                f"  LOAD P4, {f'layer_{i}_bias'}, 0",
                f"  ADD P3, P3, P4",
                "  MOVI P1, P3  # Output becomes next input",
                ""
            ])
        elif layer['type'] == 'relu':
            assembly_lines.extend([
                f"  # Layer {i}: ReLU",
                "  RELU P1, P1",
                ""
            ])
    
    assembly_lines.extend([
        "  # Store output",
        "  STOREV P1, output_address, 0",
        "  RET",
        ""
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(assembly_lines))
    
    print(f"Assembly code exported to {output_path}")


def main():
    """Test model utilities"""
    print("=" * 70)
    print("Pentary Model Utilities Test")
    print("=" * 70)
    
    # Create a dummy model for testing
    model = PentaryModel(input_dim=64, output_dim=10, hidden_dims=[32])
    
    # Test parameter counting
    print("\n1. Parameter Counting:")
    print("-" * 70)
    params = PentaryModelAnalyzer.count_parameters(model)
    print(f"Total parameters: {params['total_parameters']:,}")
    print(f"Trainable parameters: {params['trainable_parameters']:,}")
    
    # Save model for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        trainer = PentaryTrainer(model=model)
        trainer.save_model(temp_path)
    
    # Test model size
    print("\n2. Model Size Analysis:")
    print("-" * 70)
    size_info = PentaryModelAnalyzer.get_model_size(temp_path)
    print(f"Total pentary weights: {size_info['total_pentary_weights']:,}")
    print(f"Current size: {size_info['current_size_mb']:.4f} MB")
    print(f"Theoretical size (3-bit): {size_info['theoretical_size_mb']:.4f} MB")
    print(f"Compression ratio: {size_info['compression_ratio']:.2f}x")
    
    # Test sparsity
    print("\n3. Sparsity Analysis:")
    print("-" * 70)
    sparsity_info = PentaryModelAnalyzer.analyze_sparsity(temp_path)
    print(f"Total weights: {sparsity_info['total_weights']:,}")
    print(f"Zero weights: {sparsity_info['zero_weights']:,}")
    print(f"Sparsity: {sparsity_info['sparsity_percent']:.2f}%")
    
    # Test inference
    print("\n4. Inference Test:")
    print("-" * 70)
    inference = PentaryModelInference(model=model)
    test_input = np.random.randn(5, 64).astype(np.float32)
    predictions = inference.predict(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")
    
    # Cleanup
    import os
    os.unlink(temp_path)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
