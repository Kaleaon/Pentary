# Pentary Neural Network Training Tools

This directory contains comprehensive tools for training neural networks using the pentary processor architecture with native multimodal support.

## Overview

The training framework provides:
- **Pentary Neural Network Layers**: Linear, Conv2D, ReLU, MaxPool2D, BatchNorm
- **Multimodal Support**: Native processing for text, images, and audio
- **Training Framework**: Complete backpropagation with pentary quantization
- **Model Conversion**: Convert ONNX and PyTorch models to pentary format
- **Easy-to-use CLI**: Command-line interface for training

## Files

- `pentary_nn_layers.py`: Core neural network layers with pentary arithmetic
- `pentary_multimodal.py`: Multimodal data processors (text, image, audio)
- `pentary_trainer.py`: Training framework with backpropagation
- `pentary_quantizer.py`: Model quantization utilities
- `train_pentary_nn.py`: Main training script with CLI

## Quick Start

### Basic Training (Single Modality)

```bash
python tools/train_pentary_nn.py \
    --input-dim 128 \
    --output-dim 10 \
    --hidden-dims 64 32 \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.01 \
    --save-model
```

### Multimodal Training

```bash
python tools/train_pentary_nn.py \
    --multimodal \
    --modalities text image audio \
    --fusion-method concat \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --save-model
```

### Training with Custom Data

```bash
python tools/train_pentary_nn.py \
    --data-dir /path/to/data \
    --multimodal \
    --modalities text image \
    --epochs 50 \
    --save-model \
    --output-dir ./trained_models
```

## Usage Examples

### Python API

#### 1. Create a Simple Model

```python
from pentary_trainer import PentaryModel, PentaryTrainer, LossFunction

# Create model
model = PentaryModel(
    input_dim=128,
    output_dim=10,
    hidden_dims=[64, 32]
)

# Create trainer
trainer = PentaryTrainer(
    model=model,
    loss_fn=LossFunction.cross_entropy_loss,
    learning_rate=0.01
)

# Prepare data
x_train = np.random.randn(1000, 128).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Train
trainer.train(
    train_data=(x_train, y_train),
    epochs=10,
    batch_size=32
)
```

#### 2. Multimodal Training

```python
from pentary_multimodal import MultimodalFusion
from pentary_trainer import PentaryModel, PentaryTrainer

# Create multimodal fusion
fusion = MultimodalFusion(['text', 'image', 'audio'], fusion_method='concat')

# Process multimodal data
multimodal_data = {
    'text': "Sample text input",
    'image': np.random.randn(3, 224, 224).astype(np.float32),
    'audio': np.random.randn(16000).astype(np.float32)
}

fused_tensor = fusion.process(multimodal_data)

# Create model with fused input dimension
input_dim = fused_tensor.shape[0]
model = PentaryModel(input_dim=input_dim, output_dim=10)

# Train with multimodal data
trainer = PentaryTrainer(
    model=model,
    use_multimodal=True,
    modalities=['text', 'image', 'audio']
)
```

#### 3. Model Quantization

```python
from pentary_quantizer import ModelQuantizer

# Create quantizer
quantizer = ModelQuantizer(quantization_method='per_tensor')

# Quantize model weights
model_dict = {
    'layers': [
        {
            'type': 'linear',
            'weights': np.random.randn(64, 128).tolist(),
            'bias': np.random.randn(64).tolist()
        }
    ]
}

quantized_model = quantizer.quantize_model_from_dict(model_dict)
quantizer.save_quantized_model(quantized_model, 'quantized_model.json')
```

#### 4. Convert PyTorch Model

```python
from pentary_quantizer import PyTorchToPentaryConverter
import torch

# Load your PyTorch model
pytorch_model = torch.load('model.pth')

# Convert to pentary
converter = PyTorchToPentaryConverter()
converter.convert(pytorch_model, 'pentary_model.json', input_shape=(1, 3, 224, 224))
```

## Architecture Details

### Pentary Quantization

Weights are quantized to 5 levels: `{-2, -1, 0, +1, +2}`

- **Per-tensor quantization**: Single scale factor for entire tensor
- **Per-channel quantization**: Separate scale per channel (for conv layers)

### Multimodal Fusion Methods

1. **Concat**: Flatten and concatenate all modalities
2. **Add**: Element-wise addition (requires matching shapes)
3. **Multiply**: Element-wise multiplication (requires matching shapes)

### Supported Modalities

- **Text**: Tokenization + embedding (configurable vocab size, max length)
- **Image**: Resize + normalization (ImageNet-style)
- **Audio**: Mel spectrogram extraction

## Command-Line Arguments

### Model Arguments
- `--input-dim`: Input dimension (for non-multimodal)
- `--output-dim`: Number of output classes
- `--hidden-dims`: Hidden layer dimensions (space-separated)

### Training Arguments
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--loss`: Loss function (`mse` or `cross_entropy`)

### Multimodal Arguments
- `--multimodal`: Enable multimodal processing
- `--modalities`: Modalities to use (`text`, `image`, `audio`)
- `--fusion-method`: Fusion method (`concat`, `add`, `multiply`)

### Data Arguments
- `--data-dir`: Directory containing data files
- `--n-samples`: Number of samples (for dummy data)

### Output Arguments
- `--output-dir`: Output directory for models
- `--model-name`: Model name (auto-generated if not provided)
- `--save-model`: Save trained model

## Data Format

### For Custom Data Directory

```
data_dir/
├── labels.npy          # NumPy array of labels
├── text.json           # List of text strings (optional)
├── images.npy          # NumPy array of images (optional)
└── audio.npy           # NumPy array of audio waveforms (optional)
```

### Text Format
JSON file with list of strings:
```json
["First text sample", "Second text sample", ...]
```

### Image Format
NumPy array: `(n_samples, height, width, channels)` or `(n_samples, channels, height, width)`
Values in range [0, 255] (uint8) or [0, 1] (float32)

### Audio Format
NumPy array: `(n_samples, n_samples_per_audio)`
Values in range [-1, 1] (float32)

## Model Output Format

Saved models are in JSON format:

```json
{
  "input_dim": 512,
  "output_dim": 10,
  "hidden_dims": [128, 64],
  "layers": [
    {
      "type": "linear",
      "in_features": 512,
      "out_features": 128,
      "use_bias": true,
      "weight_pentary": [[...], [...]],
      "bias_pentary": [...],
      "weight_scale": 0.05,
      "bias_scale": 0.01
    },
    {
      "type": "relu"
    },
    ...
  ]
}
```

## Integration with Pentary Processor

Trained models can be deployed on the pentary processor:

1. **Quantized weights** are already in pentary format `{-2, -1, 0, +1, +2}`
2. **Scale factors** are stored separately for dequantization
3. **Model structure** matches pentary processor ISA

### Deployment Example

```python
from pentary_trainer import PentaryTrainer
from pentary_simulator import PentaryProcessor

# Load trained model
trainer = PentaryTrainer(model=PentaryModel(128, 10))
trainer.load_model('trained_model.json')

# Deploy on processor
processor = PentaryProcessor()
# Convert model to processor instructions
# (implementation depends on processor ISA)
```

## Performance Considerations

- **Training**: Uses floating-point for backpropagation, quantizes after each update
- **Inference**: Uses pentary arithmetic for efficiency
- **Memory**: Pentary weights use ~45% less memory than 8-bit binary
- **Power**: Zero-state sparsity provides automatic power savings

## Limitations

- Current implementation uses NumPy (CPU-only)
- For GPU acceleration, consider converting to PyTorch/TensorFlow
- Multimodal fusion methods are simplified (production may need attention mechanisms)
- Audio processing uses simplified STFT (use librosa for production)

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Attention-based multimodal fusion
- [ ] Advanced quantization schemes (QAT, PTQ)
- [ ] Integration with popular frameworks (PyTorch, TensorFlow)
- [ ] Distributed training support
- [ ] Model compression techniques

## Troubleshooting

### Import Errors
```bash
# Install required packages
pip install numpy
# For ONNX conversion
pip install onnx
# For PyTorch conversion
pip install torch
```

### Memory Issues
- Reduce batch size
- Use smaller hidden dimensions
- Process modalities separately and fuse later

### Training Not Converging
- Lower learning rate
- Increase model capacity
- Check data preprocessing
- Verify loss function matches task type

## References

- Pentary Processor Architecture: `../architecture/pentary_processor_architecture.md`
- Pentary Arithmetic: `pentary_arithmetic.py`
- Pentary Converter: `pentary_converter.py`

## License

See main project LICENSE file.
