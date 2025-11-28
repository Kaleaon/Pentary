#!/usr/bin/env python3
"""
Multimodal Data Processors for Pentary Neural Networks
Handles text, image, audio, and other modalities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import json


class MultimodalProcessor(ABC):
    """Base class for multimodal data processors"""
    
    @abstractmethod
    def process(self, data: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """
        Process input data into a tensor.
        
        Args:
            data: Raw input data (format depends on modality)
            
        Returns:
            Processed tensor ready for neural network
        """
        pass
    
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get the output shape of processed data"""
        pass


class TextProcessor(MultimodalProcessor):
    """Text data processor with tokenization and embedding"""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512,
                 embedding_dim: int = 128, use_pretrained: bool = False):
        """
        Initialize text processor.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            use_pretrained: Whether to use pretrained embeddings
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        
        # Simple vocabulary (in practice, use proper tokenizer)
        self.word_to_id = {}
        self.id_to_word = {}
        self.embedding_matrix = None
        
        # Initialize with dummy vocab
        self._init_vocab()
    
    def _init_vocab(self):
        """Initialize vocabulary"""
        # In practice, load from tokenizer
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into word IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        # Simple word-level tokenization
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                # Add to vocab if not full
                if len(self.word_to_id) < self.vocab_size:
                    token_id = len(self.word_to_id)
                    self.word_to_id[word] = token_id
                    self.id_to_word[token_id] = word
                    token_ids.append(token_id)
                else:
                    token_ids.append(self.word_to_id['<UNK>'])
        
        return token_ids
    
    def process(self, data: str) -> np.ndarray:
        """
        Process text into embedding tensor.
        
        Args:
            data: Input text string
            
        Returns:
            Embedding tensor (max_length, embedding_dim)
        """
        # Tokenize
        token_ids = self.tokenize(data)
        
        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.word_to_id['<PAD>']] * (self.max_length - len(token_ids))
        
        # Convert to embedding
        if self.embedding_matrix is None:
            # Initialize random embeddings
            self.embedding_matrix = np.random.randn(self.vocab_size, self.embedding_dim).astype(np.float32)
            self.embedding_matrix = self.embedding_matrix * 0.1  # Scale down
        
        # Get embeddings
        embeddings = self.embedding_matrix[token_ids]
        
        return embeddings
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape: (max_length, embedding_dim)"""
        return (self.max_length, self.embedding_dim)


class ImageProcessor(MultimodalProcessor):
    """Image data processor with normalization and resizing"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True, mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            mean: Mean for normalization (per channel)
            std: Standard deviation for normalization (per channel)
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # Default ImageNet normalization
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process image into tensor.
        
        Args:
            data: Input image array (H, W, C) or (C, H, W), values in [0, 255] or [0, 1]
            
        Returns:
            Processed image tensor (C, H, W) normalized to [0, 1] or standardized
        """
        # Ensure data is float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Handle different input formats
        if len(data.shape) == 3:
            if data.shape[2] == 3 or data.shape[2] == 1:  # (H, W, C)
                data = np.transpose(data, (2, 0, 1))  # Convert to (C, H, W)
        
        # Resize if needed (simple nearest neighbor for now)
        if data.shape[1] != self.target_size[0] or data.shape[2] != self.target_size[1]:
            data = self._resize(data, self.target_size)
        
        # Normalize pixel values to [0, 1] if needed
        if data.max() > 1.0:
            data = data / 255.0
        
        # Apply normalization (ImageNet style)
        if self.normalize:
            if data.shape[0] == 3:  # RGB
                for c in range(3):
                    data[c] = (data[c] - self.mean[c]) / self.std[c]
            elif data.shape[0] == 1:  # Grayscale
                data[0] = (data[0] - self.mean[0]) / self.std[0]
        
        return data
    
    def _resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Simple resize using nearest neighbor interpolation.
        
        Args:
            image: Input image (C, H, W)
            target_size: Target size (H, W)
            
        Returns:
            Resized image
        """
        C, H, W = image.shape
        target_h, target_w = target_size
        
        # Create output array
        resized = np.zeros((C, target_h, target_w), dtype=np.float32)
        
        # Scale factors
        scale_h = H / target_h
        scale_w = W / target_w
        
        for c in range(C):
            for th in range(target_h):
                for tw in range(target_w):
                    # Nearest neighbor sampling
                    sh = int(th * scale_h)
                    sw = int(tw * scale_w)
                    sh = min(sh, H - 1)
                    sw = min(sw, W - 1)
                    resized[c, th, tw] = image[c, sh, sw]
        
        return resized
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape: (C, H, W)"""
        # Assume RGB by default
        return (3, self.target_size[0], self.target_size[1])


class AudioProcessor(MultimodalProcessor):
    """Audio data processor with spectrogram extraction"""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop_length: int = 256, n_mels: int = 128,
                 duration: float = 1.0):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            duration: Duration of audio in seconds
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process audio into mel spectrogram.
        
        Args:
            data: Audio waveform (n_samples,) or (n_samples, channels)
            
        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        # Handle stereo/mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert to mono
        
        # Ensure correct length
        if len(data) > self.n_samples:
            data = data[:self.n_samples]
        elif len(data) < self.n_samples:
            data = np.pad(data, (0, self.n_samples - len(data)), mode='constant')
        
        # Normalize
        if data.max() > 1.0 or data.min() < -1.0:
            data = data / np.max(np.abs(data))
        
        # Compute mel spectrogram (simplified version)
        # In practice, use librosa or similar
        spectrogram = self._compute_mel_spectrogram(data)
        
        return spectrogram
    
    def _compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram (simplified implementation).
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Mel spectrogram
        """
        # Simplified STFT
        n_frames = (len(waveform) - self.n_fft) // self.hop_length + 1
        spectrogram = np.zeros((self.n_mels, n_frames), dtype=np.float32)
        
        # Simple approximation: use windowed FFT
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end > len(waveform):
                break
            
            window = waveform[start:end]
            # Apply window function (Hanning)
            window = window * np.hanning(len(window))
            
            # FFT (simplified - in practice use proper FFT)
            fft = np.fft.rfft(window, n=self.n_fft)
            magnitude = np.abs(fft)
            
            # Map to mel scale (simplified)
            # In practice, use proper mel filter banks
            mel_bins = np.linspace(0, len(magnitude), self.n_mels, dtype=int)
            for j in range(self.n_mels):
                start_bin = mel_bins[j] if j == 0 else mel_bins[j-1]
                end_bin = mel_bins[j+1] if j < self.n_mels - 1 else len(magnitude)
                spectrogram[j, i] = np.mean(magnitude[start_bin:end_bin])
        
        # Log scale
        spectrogram = np.log(spectrogram + 1e-10)
        
        return spectrogram
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape: (n_mels, time_frames)"""
        n_frames = (self.n_samples - self.n_fft) // self.hop_length + 1
        return (self.n_mels, n_frames)


class MultimodalFusion:
    """Fuses multiple modalities into a unified representation"""
    
    def __init__(self, modalities: List[str], fusion_method: str = 'concat'):
        """
        Initialize multimodal fusion.
        
        Args:
            modalities: List of modality names (e.g., ['text', 'image', 'audio'])
            fusion_method: Fusion method ('concat', 'add', 'multiply', 'attention')
        """
        self.modalities = modalities
        self.fusion_method = fusion_method
        self.processors = {}
        
        # Initialize processors for each modality
        if 'text' in modalities:
            self.processors['text'] = TextProcessor()
        if 'image' in modalities:
            self.processors['image'] = ImageProcessor()
        if 'audio' in modalities:
            self.processors['audio'] = AudioProcessor()
    
    def process(self, data: Dict[str, Union[str, np.ndarray]]) -> np.ndarray:
        """
        Process and fuse multiple modalities.
        
        Args:
            data: Dictionary mapping modality names to raw data
            
        Returns:
            Fused tensor
        """
        processed = {}
        
        # Process each modality
        for modality, processor in self.processors.items():
            if modality in data:
                processed[modality] = processor.process(data[modality])
        
        # Fuse modalities
        if self.fusion_method == 'concat':
            # Flatten and concatenate
            flattened = []
            for modality in self.modalities:
                if modality in processed:
                    flattened.append(processed[modality].flatten())
            return np.concatenate(flattened)
        
        elif self.fusion_method == 'add':
            # Add (requires same shape)
            result = None
            for modality in self.modalities:
                if modality in processed:
                    if result is None:
                        result = processed[modality]
                    else:
                        # Resize to match if needed
                        if result.shape != processed[modality].shape:
                            # Simple resize (in practice, use proper interpolation)
                            result = self._resize_to_match(result, processed[modality])
                        result = result + processed[modality]
            return result
        
        elif self.fusion_method == 'multiply':
            # Element-wise multiply
            result = None
            for modality in self.modalities:
                if modality in processed:
                    if result is None:
                        result = processed[modality]
                    else:
                        result = self._resize_to_match(result, processed[modality])
                        result = result * processed[modality]
            return result
        
        else:  # concat by default
            flattened = []
            for modality in self.modalities:
                if modality in processed:
                    flattened.append(processed[modality].flatten())
            return np.concatenate(flattened)
    
    def _resize_to_match(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Resize source to match target shape"""
        if target.shape == source.shape:
            return source
        
        # Simple nearest neighbor resize
        # In practice, use proper interpolation
        result = np.zeros_like(target)
        scale = [s / t for s, t in zip(source.shape, target.shape)]
        
        for idx in np.ndindex(target.shape):
            source_idx = tuple(int(i * s) for i, s in zip(idx, scale))
            source_idx = tuple(min(s, d - 1) for s, d in zip(source_idx, source.shape))
            result[idx] = source[source_idx]
        
        return result
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape after fusion"""
        shapes = []
        for modality in self.modalities:
            if modality in self.processors:
                shape = self.processors[modality].get_output_shape()
                shapes.append(np.prod(shape))
        
        if self.fusion_method == 'concat':
            return (sum(shapes),)
        else:
            # For add/multiply, return first shape (they should match)
            return self.processors[self.modalities[0]].get_output_shape()


def main():
    """Test multimodal processors"""
    print("=" * 70)
    print("Multimodal Data Processors Test")
    print("=" * 70)
    
    # Test text processor
    print("\n1. Testing Text Processor:")
    print("-" * 70)
    text_proc = TextProcessor(vocab_size=1000, max_length=128, embedding_dim=64)
    text = "This is a sample text for testing the pentary neural network"
    text_tensor = text_proc.process(text)
    print(f"Input text: {text[:50]}...")
    print(f"Output shape: {text_tensor.shape}")
    print(f"Output dtype: {text_tensor.dtype}")
    
    # Test image processor
    print("\n2. Testing Image Processor:")
    print("-" * 70)
    image_proc = ImageProcessor(target_size=(224, 224))
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
    image_tensor = image_proc.process(dummy_image)
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Output shape: {image_tensor.shape}")
    print(f"Output range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # Test audio processor
    print("\n3. Testing Audio Processor:")
    print("-" * 70)
    audio_proc = AudioProcessor(sample_rate=16000, duration=1.0)
    # Create dummy audio
    dummy_audio = np.random.randn(16000).astype(np.float32)
    audio_tensor = audio_proc.process(dummy_audio)
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Output shape: {audio_tensor.shape}")
    print(f"Output range: [{audio_tensor.min():.3f}, {audio_tensor.max():.3f}]")
    
    # Test multimodal fusion
    print("\n4. Testing Multimodal Fusion:")
    print("-" * 70)
    fusion = MultimodalFusion(['text', 'image', 'audio'], fusion_method='concat')
    multimodal_data = {
        'text': text,
        'image': dummy_image,
        'audio': dummy_audio
    }
    fused_tensor = fusion.process(multimodal_data)
    print(f"Fused tensor shape: {fused_tensor.shape}")
    print(f"Fused tensor dtype: {fused_tensor.dtype}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
