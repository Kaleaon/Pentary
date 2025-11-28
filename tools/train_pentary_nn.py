#!/usr/bin/env python3
"""
Main Training Script for Pentary Neural Networks
Supports multimodal training with text, image, and audio
"""

import argparse
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from pentary_trainer import PentaryTrainer, PentaryModel, LossFunction
from pentary_multimodal import MultimodalFusion, TextProcessor, ImageProcessor, AudioProcessor
from pentary_quantizer import ModelQuantizer


def create_multimodal_dataset(n_samples: int, modalities: List[str]) -> Tuple[Dict, np.ndarray]:
    """
    Create a dummy multimodal dataset for testing.
    
    Args:
        n_samples: Number of samples
        modalities: List of modalities to include
        
    Returns:
        (data_dict, labels) where data_dict maps modality names to data
    """
    data = {}
    labels = np.random.randint(0, 10, n_samples)
    
    if 'text' in modalities:
        # Generate dummy text data
        texts = [f"Sample text {i} with some content" for i in range(n_samples)]
        data['text'] = texts
    
    if 'image' in modalities:
        # Generate dummy image data
        images = np.random.randint(0, 255, (n_samples, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        data['image'] = images
    
    if 'audio' in modalities:
        # Generate dummy audio data
        audio = np.random.randn(n_samples, 16000).astype(np.float32)
        data['audio'] = audio
    
    return data, labels


def load_data_from_files(data_dir: str, modalities: List[str]) -> Tuple[Dict, np.ndarray]:
    """
    Load multimodal data from files.
    
    Args:
        data_dir: Directory containing data files
        modalities: List of modalities to load
        
    Returns:
        (data_dict, labels)
    """
    data = {}
    labels = None
    
    # Load labels
    labels_path = os.path.join(data_dir, 'labels.npy')
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    
    # Load each modality
    if 'text' in modalities:
        text_path = os.path.join(data_dir, 'text.json')
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                data['text'] = json.load(f)
    
    if 'image' in modalities:
        image_path = os.path.join(data_dir, 'images.npy')
        if os.path.exists(image_path):
            data['image'] = np.load(image_path)
    
    if 'audio' in modalities:
        audio_path = os.path.join(data_dir, 'audio.npy')
        if os.path.exists(audio_path):
            data['audio'] = np.load(audio_path)
    
    return data, labels


def prepare_multimodal_data(data: Dict, labels: np.ndarray, fusion: MultimodalFusion) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare multimodal data for training.
    
    Args:
        data: Dictionary of modality data
        labels: Labels array
        fusion: Multimodal fusion processor
        
    Returns:
        (x_processed, y_processed)
    """
    n_samples = len(labels)
    processed_samples = []
    
    for i in range(n_samples):
        sample_data = {}
        for modality in fusion.modalities:
            if modality in data:
                if isinstance(data[modality], list):
                    sample_data[modality] = data[modality][i]
                elif isinstance(data[modality], np.ndarray):
                    sample_data[modality] = data[modality][i]
        
        processed = fusion.process(sample_data)
        processed_samples.append(processed)
    
    x = np.array(processed_samples)
    
    # Convert labels to one-hot if needed
    if len(labels.shape) == 1:
        n_classes = len(np.unique(labels))
        y = np.zeros((len(labels), n_classes), dtype=np.float32)
        y[np.arange(len(labels)), labels] = 1.0
    else:
        y = labels
    
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Train Pentary Neural Network with Multimodal Support')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=512,
                       help='Input dimension (for non-multimodal)')
    parser.add_argument('--output-dim', type=int, default=10,
                       help='Output dimension (number of classes)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer dimensions')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                       choices=['mse', 'cross_entropy'],
                       help='Loss function')
    
    # Multimodal arguments
    parser.add_argument('--multimodal', action='store_true',
                       help='Use multimodal processing')
    parser.add_argument('--modalities', type=str, nargs='+',
                       choices=['text', 'image', 'audio'],
                       default=['text', 'image'],
                       help='Modalities to use')
    parser.add_argument('--fusion-method', type=str, default='concat',
                       choices=['concat', 'add', 'multiply'],
                       help='Fusion method for multimodal data')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing data files')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples (for dummy data)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for saved models')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name (default: auto-generated)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate model name if not provided
    if args.model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.model_name = f'pentary_model_{timestamp}'
    
    print("=" * 70)
    print("Pentary Neural Network Training")
    print("=" * 70)
    print(f"Model: {args.input_dim} -> {args.hidden_dims} -> {args.output_dim}")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}, LR {args.learning_rate}")
    print(f"Multimodal: {args.multimodal}")
    if args.multimodal:
        print(f"Modalities: {args.modalities}")
        print(f"Fusion method: {args.fusion_method}")
    print("=" * 70)
    
    # Prepare data
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"\nLoading data from {args.data_dir}...")
        data, labels = load_data_from_files(args.data_dir, args.modalities if args.multimodal else [])
    else:
        print(f"\nGenerating dummy dataset with {args.n_samples} samples...")
        if args.multimodal:
            data, labels = create_multimodal_dataset(args.n_samples, args.modalities)
        else:
            # Generate simple tabular data
            data = np.random.randn(args.n_samples, args.input_dim).astype(np.float32)
            labels = np.random.randint(0, args.output_dim, args.n_samples)
    
    # Setup multimodal fusion if needed
    fusion = None
    if args.multimodal:
        fusion = MultimodalFusion(args.modalities, fusion_method=args.fusion_method)
        x_train, y_train = prepare_multimodal_data(data, labels, fusion)
        input_dim = x_train.shape[1]
    else:
        # Split data
        split_idx = int(0.8 * len(data))
        x_train = data[:split_idx]
        y_train = labels[:split_idx]
        x_val = data[split_idx:]
        y_val = labels[split_idx:]
        input_dim = args.input_dim
    
    # Convert labels to one-hot if needed
    if len(y_train.shape) == 1:
        n_classes = len(np.unique(y_train))
        y_train_onehot = np.zeros((len(y_train), n_classes), dtype=np.float32)
        y_train_onehot[np.arange(len(y_train)), y_train] = 1.0
        y_train = y_train_onehot
        
        if not args.multimodal:
            y_val_onehot = np.zeros((len(y_val), n_classes), dtype=np.float32)
            y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
            y_val = y_val_onehot
    
    # Create model
    print(f"\nCreating model with input dimension {input_dim}...")
    model = PentaryModel(input_dim=input_dim, output_dim=args.output_dim,
                        hidden_dims=args.hidden_dims)
    
    # Select loss function
    if args.loss == 'mse':
        loss_fn = LossFunction.mse_loss
    else:
        loss_fn = LossFunction.cross_entropy_loss
    
    # Create trainer
    trainer = PentaryTrainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=args.learning_rate,
        use_multimodal=args.multimodal,
        modalities=args.modalities if args.multimodal else None
    )
    
    # Train
    if args.multimodal:
        # For multimodal, we need to process validation separately
        split_idx = int(0.8 * len(x_train))
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
    
    trainer.train(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val) if not args.multimodal or len(x_val) > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.output_dir, f'{args.model_name}.json')
        print(f"\nSaving model to {model_path}...")
        trainer.save_model(model_path)
        print("Model saved!")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f'{args.model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
