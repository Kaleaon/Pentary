#!/usr/bin/env python3
"""
Example: Training a Pentary Neural Network with Multimodal Data
This example demonstrates how to train a neural network using the pentary processor
with native support for text, image, and audio modalities.
"""

import numpy as np
from pentary_trainer import PentaryModel, PentaryTrainer, LossFunction
from pentary_multimodal import MultimodalFusion, TextProcessor, ImageProcessor, AudioProcessor


def example_single_modality():
    """Example: Training with single modality (tabular data)"""
    print("=" * 70)
    print("Example 1: Single Modality Training")
    print("=" * 70)
    
    # Create dummy dataset
    n_samples = 1000
    input_dim = 128
    output_dim = 10
    
    x_train = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_train = np.random.randint(0, output_dim, n_samples)
    
    # Split train/val
    split_idx = int(0.8 * n_samples)
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Create model
    model = PentaryModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[64, 32]
    )
    
    # Create trainer
    trainer = PentaryTrainer(
        model=model,
        loss_fn=LossFunction.cross_entropy_loss,
        learning_rate=0.01
    )
    
    # Convert labels to one-hot
    y_train_onehot = np.zeros((len(y_train), output_dim), dtype=np.float32)
    y_train_onehot[np.arange(len(y_train)), y_train] = 1.0
    y_val_onehot = np.zeros((len(y_val), output_dim), dtype=np.float32)
    y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
    
    # Train
    trainer.train(
        train_data=(x_train, y_train_onehot),
        val_data=(x_val, y_val_onehot),
        epochs=5,
        batch_size=32,
        verbose=True
    )
    
    print("\nTraining completed!")


def example_multimodal():
    """Example: Training with multiple modalities"""
    print("\n" + "=" * 70)
    print("Example 2: Multimodal Training")
    print("=" * 70)
    
    # Create multimodal fusion
    fusion = MultimodalFusion(
        modalities=['text', 'image', 'audio'],
        fusion_method='concat'
    )
    
    # Create dummy multimodal dataset
    n_samples = 500
    
    # Text data
    texts = [f"Sample text {i} with multimodal content" for i in range(n_samples)]
    
    # Image data (224x224 RGB)
    images = np.random.randint(0, 255, (n_samples, 224, 224, 3), dtype=np.uint8).astype(np.float32)
    
    # Audio data (1 second at 16kHz)
    audio = np.random.randn(n_samples, 16000).astype(np.float32)
    
    # Labels
    labels = np.random.randint(0, 10, n_samples)
    
    # Process and fuse multimodal data
    print("Processing multimodal data...")
    processed_samples = []
    for i in range(n_samples):
        multimodal_data = {
            'text': texts[i],
            'image': images[i],
            'audio': audio[i]
        }
        fused = fusion.process(multimodal_data)
        processed_samples.append(fused)
    
    x_train = np.array(processed_samples)
    
    # Split train/val
    split_idx = int(0.8 * n_samples)
    x_val = x_train[split_idx:]
    y_val = labels[split_idx:]
    x_train = x_train[:split_idx]
    y_train = labels[:split_idx]
    
    # Convert labels to one-hot
    output_dim = 10
    y_train_onehot = np.zeros((len(y_train), output_dim), dtype=np.float32)
    y_train_onehot[np.arange(len(y_train)), y_train] = 1.0
    y_val_onehot = np.zeros((len(y_val), output_dim), dtype=np.float32)
    y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
    
    # Create model with fused input dimension
    input_dim = x_train.shape[1]
    print(f"Fused input dimension: {input_dim}")
    
    model = PentaryModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[128, 64]
    )
    
    # Create trainer with multimodal support
    trainer = PentaryTrainer(
        model=model,
        loss_fn=LossFunction.cross_entropy_loss,
        learning_rate=0.01,
        use_multimodal=True,
        modalities=['text', 'image', 'audio']
    )
    
    # Train
    trainer.train(
        train_data=(x_train, y_train_onehot),
        val_data=(x_val, y_val_onehot),
        epochs=5,
        batch_size=16,  # Smaller batch for multimodal
        verbose=True
    )
    
    print("\nMultimodal training completed!")


def example_text_only():
    """Example: Training with text only"""
    print("\n" + "=" * 70)
    print("Example 3: Text-Only Training")
    print("=" * 70)
    
    # Create text processor
    text_proc = TextProcessor(
        vocab_size=1000,
        max_length=128,
        embedding_dim=64
    )
    
    # Create text dataset
    texts = [
        "This is a positive review about the product",
        "I really enjoyed using this service",
        "The quality is excellent and highly recommended",
        "This is a negative review with complaints",
        "Poor quality and not worth the money",
        "Very disappointed with the purchase"
    ] * 100  # Repeat for more samples
    
    labels = [1, 1, 1, 0, 0, 0] * 100  # Binary classification
    
    # Process texts
    print("Processing text data...")
    processed_texts = []
    for text in texts:
        processed = text_proc.process(text)
        processed_texts.append(processed.flatten())  # Flatten embeddings
    
    x_train = np.array(processed_texts)
    
    # Split
    split_idx = int(0.8 * len(x_train))
    x_val = x_train[split_idx:]
    y_val = np.array(labels[split_idx:])
    x_train = x_train[:split_idx]
    y_train = np.array(labels[:split_idx])
    
    # Convert to one-hot
    y_train_onehot = np.zeros((len(y_train), 2), dtype=np.float32)
    y_train_onehot[np.arange(len(y_train)), y_train] = 1.0
    y_val_onehot = np.zeros((len(y_val), 2), dtype=np.float32)
    y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
    
    # Create model
    input_dim = x_train.shape[1]
    model = PentaryModel(input_dim=input_dim, output_dim=2, hidden_dims=[32])
    
    # Train
    trainer = PentaryTrainer(
        model=model,
        loss_fn=LossFunction.cross_entropy_loss,
        learning_rate=0.01
    )
    
    trainer.train(
        train_data=(x_train, y_train_onehot),
        val_data=(x_val, y_val_onehot),
        epochs=5,
        batch_size=32,
        verbose=True
    )
    
    print("\nText-only training completed!")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("Pentary Neural Network Training Examples")
    print("=" * 70)
    
    # Run examples
    try:
        example_single_modality()
        example_multimodal()
        example_text_only()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
