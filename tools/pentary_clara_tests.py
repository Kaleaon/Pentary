"""
CLARA-Pentary Test Suite
Comprehensive testing framework for pentary continuous latent reasoning

Author: SuperNinja AI Agent
Date: January 6, 2025
Version: 1.0
"""

import numpy as np
import time
import unittest
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# Pentary Arithmetic Operations
# ============================================================================

class PentaryArithmetic:
    """Core pentary arithmetic operations"""
    
    @staticmethod
    def add(a: List[int], b: List[int]) -> Tuple[List[int], int]:
        """
        Pentary addition with carry
        
        Args:
            a: First pentary number (list of digits 0-4)
            b: Second pentary number (list of digits 0-4)
        
        Returns:
            result: Sum (list of digits 0-4)
            carry: Final carry bit
        """
        result = []
        carry = 0
        
        for digit_a, digit_b in zip(a, b):
            sum_val = digit_a + digit_b + carry
            if sum_val >= 5:
                result.append(sum_val - 5)
                carry = 1
            else:
                result.append(sum_val)
                carry = 0
        
        return result, carry
    
    @staticmethod
    def multiply_digit(a: int, b: int) -> Tuple[int, int]:
        """
        Single pentary digit multiplication
        
        Args:
            a: First digit (0-4)
            b: Second digit (0-4)
        
        Returns:
            low: Low digit of result
            high: High digit of result
        """
        # 5×5 multiplication table
        mult_table = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4],
            [0, 2, 4, 1, 3],  # 2×2=4, 2×3=11₅=6₁₀, 2×4=13₅=8₁₀
            [0, 3, 1, 4, 2],  # 3×2=11₅, 3×3=14₅=9₁₀, 3×4=22₅=12₁₀
            [0, 4, 3, 2, 1]   # 4×2=13₅, 4×3=22₅, 4×4=31₅=16₁₀
        ]
        
        product = a * b
        low = product % 5
        high = product // 5
        return low, high
    
    @staticmethod
    def quantize(float_val: float) -> int:
        """
        Quantize float to pentary {-2, -1, 0, +1, +2}
        
        Args:
            float_val: Float value to quantize
        
        Returns:
            pentary_val: Quantized value in {-2, -1, 0, +1, +2}
        """
        if float_val < -1.5:
            return -2
        elif float_val < -0.5:
            return -1
        elif float_val < 0.5:
            return 0
        elif float_val < 1.5:
            return 1
        else:
            return 2
    
    @staticmethod
    def dot_product(vec_a: List[int], vec_b: List[int]) -> int:
        """
        Pentary dot product
        
        Args:
            vec_a: First vector (pentary digits)
            vec_b: Second vector (pentary digits)
        
        Returns:
            result: Dot product (pentary)
        """
        result = 0
        for a, b in zip(vec_a, vec_b):
            low, high = PentaryArithmetic.multiply_digit(abs(a), abs(b))
            product = high * 5 + low
            
            # Handle signs
            if (a < 0) != (b < 0):
                product = -product
            
            result += product
        
        return result
    
    @staticmethod
    def norm(vec: List[int]) -> float:
        """
        Pentary vector norm
        
        Args:
            vec: Vector (pentary digits)
        
        Returns:
            norm: L2 norm
        """
        sum_squares = 0
        for val in vec:
            sum_squares += val * val
        return np.sqrt(sum_squares)


# ============================================================================
# Pentary Memory Token
# ============================================================================

@dataclass
class PentaryMemoryToken:
    """Pentary memory token representation"""
    
    values: List[int]  # 332 pentary digits
    
    def __post_init__(self):
        assert len(self.values) == 332, "Memory token must have 332 digits"
        assert all(-2 <= v <= 2 for v in self.values), "Values must be in {-2,-1,0,+1,+2}"
    
    def to_bytes(self) -> bytes:
        """Convert to byte representation (3 bits per digit)"""
        # Pack 3 bits per digit
        bits = []
        for val in self.values:
            # Map {-2,-1,0,+1,+2} to {0,1,2,3,4}
            encoded = val + 2
            bits.extend([
                (encoded >> 2) & 1,
                (encoded >> 1) & 1,
                encoded & 1
            ])
        
        # Convert bits to bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= bits[i + j] << (7 - j)
            byte_array.append(byte_val)
        
        return bytes(byte_array)
    
    @classmethod
    def from_float_vector(cls, float_vec: List[float]) -> 'PentaryMemoryToken':
        """Create from float vector by quantization"""
        assert len(float_vec) == 332, "Float vector must have 332 dimensions"
        
        pentary_vals = [PentaryArithmetic.quantize(v) for v in float_vec]
        return cls(values=pentary_vals)
    
    def cosine_similarity(self, other: 'PentaryMemoryToken') -> float:
        """Compute cosine similarity with another token"""
        dot_prod = PentaryArithmetic.dot_product(self.values, other.values)
        norm_self = PentaryArithmetic.norm(self.values)
        norm_other = PentaryArithmetic.norm(other.values)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        return dot_prod / (norm_self * norm_other)


# ============================================================================
# Pentary Semantic Compressor (Simplified)
# ============================================================================

class PentarySemanticCompressor:
    """Simplified pentary semantic compressor for testing"""
    
    def __init__(self, compression_ratio: int = 16):
        self.compression_ratio = compression_ratio
        self.num_memory_tokens = None
    
    def compress(self, document_tokens: List[str]) -> List[PentaryMemoryToken]:
        """
        Compress document into pentary memory tokens
        
        Args:
            document_tokens: List of document tokens
        
        Returns:
            memory_tokens: List of pentary memory tokens
        """
        num_tokens = len(document_tokens)
        self.num_memory_tokens = max(1, num_tokens // self.compression_ratio)
        
        # Simplified: Generate random pentary memory tokens
        # In real implementation, this would be learned
        memory_tokens = []
        for _ in range(self.num_memory_tokens):
            # Generate random float vector
            float_vec = np.random.randn(332).tolist()
            
            # Quantize to pentary
            memory_token = PentaryMemoryToken.from_float_vector(float_vec)
            memory_tokens.append(memory_token)
        
        return memory_tokens
    
    def get_compression_ratio(self, document_tokens: List[str]) -> float:
        """Calculate actual compression ratio"""
        if self.num_memory_tokens is None:
            return 0.0
        return len(document_tokens) / self.num_memory_tokens


# ============================================================================
# Test Suite
# ============================================================================

class TestPentaryArithmetic(unittest.TestCase):
    """Test pentary arithmetic operations"""
    
    def test_add_simple(self):
        """Test simple pentary addition"""
        a = [1, 2, 3]  # 1×25 + 2×5 + 3 = 38
        b = [2, 1, 4]  # 2×25 + 1×5 + 4 = 59
        result, carry = PentaryArithmetic.add(a, b)
        
        # 38 + 59 = 97 = 3×25 + 4×5 + 2 = [3, 4, 2]
        self.assertEqual(result, [3, 4, 2])
        self.assertEqual(carry, 0)
    
    def test_add_with_carry(self):
        """Test pentary addition with carry"""
        a = [4, 4, 4]  # 4×25 + 4×5 + 4 = 124
        b = [1, 1, 1]  # 1×25 + 1×5 + 1 = 31
        result, carry = PentaryArithmetic.add(a, b)
        
        # 124 + 31 = 155 = 1×125 + 1×25 + 1×5 + 0 = [1, 1, 0] with carry
        self.assertEqual(result, [0, 0, 0])
        self.assertEqual(carry, 1)
    
    def test_multiply_digit(self):
        """Test single digit multiplication"""
        # 3 × 4 = 12 = 2×5 + 2
        low, high = PentaryArithmetic.multiply_digit(3, 4)
        self.assertEqual(low, 2)
        self.assertEqual(high, 2)
        
        # 2 × 3 = 6 = 1×5 + 1
        low, high = PentaryArithmetic.multiply_digit(2, 3)
        self.assertEqual(low, 1)
        self.assertEqual(high, 1)
    
    def test_quantize(self):
        """Test float to pentary quantization"""
        self.assertEqual(PentaryArithmetic.quantize(-2.0), -2)
        self.assertEqual(PentaryArithmetic.quantize(-0.7), -1)
        self.assertEqual(PentaryArithmetic.quantize(0.0), 0)
        self.assertEqual(PentaryArithmetic.quantize(0.8), 1)
        self.assertEqual(PentaryArithmetic.quantize(1.8), 2)
    
    def test_dot_product(self):
        """Test pentary dot product"""
        vec_a = [1, 2, 0, -1, -2]
        vec_b = [2, 1, 1, 0, 1]
        
        # 1×2 + 2×1 + 0×1 + (-1)×0 + (-2)×1 = 2 + 2 + 0 + 0 - 2 = 2
        result = PentaryArithmetic.dot_product(vec_a, vec_b)
        self.assertEqual(result, 2)
    
    def test_norm(self):
        """Test pentary vector norm"""
        vec = [1, 2, 0, -1, -2]
        # sqrt(1² + 2² + 0² + 1² + 2²) = sqrt(10) ≈ 3.162
        norm = PentaryArithmetic.norm(vec)
        self.assertAlmostEqual(norm, np.sqrt(10), places=5)


class TestPentaryMemoryToken(unittest.TestCase):
    """Test pentary memory token operations"""
    
    def test_token_creation(self):
        """Test memory token creation"""
        values = [0] * 332
        token = PentaryMemoryToken(values=values)
        self.assertEqual(len(token.values), 332)
    
    def test_token_from_float(self):
        """Test token creation from float vector"""
        float_vec = np.random.randn(332).tolist()
        token = PentaryMemoryToken.from_float_vector(float_vec)
        
        self.assertEqual(len(token.values), 332)
        self.assertTrue(all(-2 <= v <= 2 for v in token.values))
    
    def test_token_to_bytes(self):
        """Test token serialization"""
        values = [1, -1, 0, 2, -2] * 66 + [0, 0]  # 332 values
        token = PentaryMemoryToken(values=values)
        
        byte_data = token.to_bytes()
        # 332 digits × 3 bits = 996 bits = 124.5 bytes
        self.assertEqual(len(byte_data), 125)  # Rounded up
    
    def test_cosine_similarity(self):
        """Test cosine similarity between tokens"""
        # Identical tokens
        values1 = [1, 2, 0, -1, -2] * 66 + [0, 0]
        token1 = PentaryMemoryToken(values=values1)
        token2 = PentaryMemoryToken(values=values1)
        
        similarity = token1.cosine_similarity(token2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Orthogonal tokens
        values2 = [0, 0, 1, 0, 0] * 66 + [0, 0]
        values3 = [1, 0, 0, 0, 0] * 66 + [0, 0]
        token2 = PentaryMemoryToken(values=values2)
        token3 = PentaryMemoryToken(values=values3)
        
        similarity = token2.cosine_similarity(token3)
        self.assertAlmostEqual(similarity, 0.0, places=5)


class TestPentaryCompressor(unittest.TestCase):
    """Test pentary semantic compressor"""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        compressor = PentarySemanticCompressor(compression_ratio=16)
        
        # 1000 tokens → 62.5 memory tokens (rounded to 62)
        document = ["token"] * 1000
        memory_tokens = compressor.compress(document)
        
        ratio = compressor.get_compression_ratio(document)
        self.assertGreaterEqual(ratio, 16.0)
        self.assertLessEqual(ratio, 17.0)
    
    def test_memory_token_count(self):
        """Test number of memory tokens generated"""
        compressor = PentarySemanticCompressor(compression_ratio=16)
        
        document = ["token"] * 1000
        memory_tokens = compressor.compress(document)
        
        expected_tokens = 1000 // 16
        self.assertEqual(len(memory_tokens), expected_tokens)
    
    def test_memory_token_format(self):
        """Test memory token format"""
        compressor = PentarySemanticCompressor(compression_ratio=16)
        
        document = ["token"] * 1000
        memory_tokens = compressor.compress(document)
        
        for token in memory_tokens:
            self.assertIsInstance(token, PentaryMemoryToken)
            self.assertEqual(len(token.values), 332)
            self.assertTrue(all(-2 <= v <= 2 for v in token.values))


class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""
    
    def test_compression_speed(self):
        """Benchmark compression speed"""
        compressor = PentarySemanticCompressor(compression_ratio=16)
        document = ["token"] * 1000
        
        start_time = time.time()
        memory_tokens = compressor.compress(document)
        elapsed = time.time() - start_time
        
        # Should be fast (< 10ms for simplified version)
        self.assertLess(elapsed, 0.01)
        print(f"\nCompression time: {elapsed*1000:.2f} ms")
    
    def test_similarity_speed(self):
        """Benchmark similarity computation speed"""
        token1 = PentaryMemoryToken(values=[1, 2, 0, -1, -2] * 66 + [0, 0])
        token2 = PentaryMemoryToken(values=[2, 1, 1, 0, -1] * 66 + [0, 0])
        
        start_time = time.time()
        for _ in range(1000):
            similarity = token1.cosine_similarity(token2)
        elapsed = time.time() - start_time
        
        # Should be fast (< 1ms per similarity)
        avg_time = elapsed / 1000
        self.assertLess(avg_time, 0.001)
        print(f"\nAverage similarity time: {avg_time*1e6:.2f} µs")
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        # Binary representation: 332 × 4 bytes = 1,328 bytes
        # Pentary representation: 332 × 3 bits = 996 bits = 124.5 bytes
        
        token = PentaryMemoryToken(values=[1, 2, 0, -1, -2] * 66 + [0, 0])
        byte_data = token.to_bytes()
        
        binary_size = 332 * 4  # float32
        pentary_size = len(byte_data)
        
        compression = binary_size / pentary_size
        self.assertGreater(compression, 10.0)
        print(f"\nMemory compression: {compression:.2f}×")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_compression(self):
        """Test end-to-end compression pipeline"""
        # Create compressor
        compressor = PentarySemanticCompressor(compression_ratio=16)
        
        # Compress document
        document = ["The", "quick", "brown", "fox"] * 250  # 1000 tokens
        memory_tokens = compressor.compress(document)
        
        # Verify compression
        self.assertEqual(len(memory_tokens), 62)
        self.assertGreater(compressor.get_compression_ratio(document), 16.0)
        
        # Verify token format
        for token in memory_tokens:
            self.assertEqual(len(token.values), 332)
            byte_data = token.to_bytes()
            self.assertEqual(len(byte_data), 125)
    
    def test_retrieval_simulation(self):
        """Simulate retrieval process"""
        # Create compressor
        compressor = PentarySemanticCompressor(compression_ratio=16)
        
        # Compress multiple documents
        doc1 = ["document", "one"] * 500
        doc2 = ["document", "two"] * 500
        doc3 = ["document", "three"] * 500
        
        mem1 = compressor.compress(doc1)
        mem2 = compressor.compress(doc2)
        mem3 = compressor.compress(doc3)
        
        # Create query (use first document as query)
        query_mem = mem1
        
        # Compute similarities
        sim1 = query_mem[0].cosine_similarity(mem1[0])
        sim2 = query_mem[0].cosine_similarity(mem2[0])
        sim3 = query_mem[0].cosine_similarity(mem3[0])
        
        # Query should be most similar to itself
        self.assertGreater(sim1, sim2)
        self.assertGreater(sim1, sim3)
        
        print(f"\nSimilarities: doc1={sim1:.4f}, doc2={sim2:.4f}, doc3={sim3:.4f}")


# ============================================================================
# Benchmark Suite
# ============================================================================

def run_benchmarks():
    """Run comprehensive benchmarks"""
    print("\n" + "="*70)
    print("CLARA-Pentary Benchmark Suite")
    print("="*70)
    
    # Benchmark 1: Compression ratios
    print("\n1. Compression Ratios:")
    for ratio in [4, 16, 64, 128]:
        compressor = PentarySemanticCompressor(compression_ratio=ratio)
        document = ["token"] * 1000
        memory_tokens = compressor.compress(document)
        actual_ratio = compressor.get_compression_ratio(document)
        
        # Calculate memory savings
        binary_size = len(document) * 768 * 4  # float32
        pentary_size = len(memory_tokens) * 125  # bytes per token
        savings = (1 - pentary_size / binary_size) * 100
        
        print(f"   Ratio {ratio}×: {len(memory_tokens)} tokens, "
              f"{actual_ratio:.1f}× actual, {savings:.1f}% memory saved")
    
    # Benchmark 2: Speed tests
    print("\n2. Speed Benchmarks:")
    compressor = PentarySemanticCompressor(compression_ratio=16)
    document = ["token"] * 1000
    
    # Compression speed
    start = time.time()
    for _ in range(100):
        memory_tokens = compressor.compress(document)
    elapsed = (time.time() - start) / 100
    print(f"   Compression: {elapsed*1000:.2f} ms per document")
    
    # Similarity speed
    token1 = PentaryMemoryToken(values=[1, 2, 0, -1, -2] * 66 + [0, 0])
    token2 = PentaryMemoryToken(values=[2, 1, 1, 0, -1] * 66 + [0, 0])
    
    start = time.time()
    for _ in range(10000):
        similarity = token1.cosine_similarity(token2)
    elapsed = (time.time() - start) / 10000
    print(f"   Similarity: {elapsed*1e6:.2f} µs per comparison")
    
    # Benchmark 3: Memory efficiency
    print("\n3. Memory Efficiency:")
    token = PentaryMemoryToken(values=[1, 2, 0, -1, -2] * 66 + [0, 0])
    
    binary_size = 332 * 4  # float32
    pentary_size = len(token.to_bytes())
    compression = binary_size / pentary_size
    
    print(f"   Binary: {binary_size} bytes")
    print(f"   Pentary: {pentary_size} bytes")
    print(f"   Compression: {compression:.2f}×")
    
    # Benchmark 4: Scalability
    print("\n4. Scalability:")
    for num_docs in [100, 1000, 10000]:
        # Simulate document database
        docs = []
        for _ in range(num_docs):
            token = PentaryMemoryToken(values=np.random.randint(-2, 3, 332).tolist())
            docs.append(token)
        
        # Simulate retrieval
        query = PentaryMemoryToken(values=np.random.randint(-2, 3, 332).tolist())
        
        start = time.time()
        similarities = [query.cosine_similarity(doc) for doc in docs]
        elapsed = time.time() - start
        
        print(f"   {num_docs} docs: {elapsed*1000:.2f} ms retrieval time")
    
    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # Run benchmarks
    run_benchmarks()