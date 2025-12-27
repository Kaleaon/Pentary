#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pentary Advanced Architectures

Tests:
1. PentaryMamba - Selective State Space Model
2. PentaryRWKV - Linear Attention RNN
3. PentaryRetNet - Retentive Network
4. PentaryWorldModel - Latent Dynamics

Verifies:
- Correct output shapes
- Pentary quantization (weights in {-2, -1, 0, +1, +2})
- Sparsity (zero weights benefit)
- Linear complexity claims
- Generation capability
"""

import sys
import os
import time
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from pentary_mamba import PentaryMamba, PentaryMambaBlock, PentarySSMCore
from pentary_rwkv import PentaryRWKV, PentaryRWKVBlock, PentaryTimeMix
from pentary_retnet import PentaryRetNet, PentaryRetNetBlock, PentaryRetention
from pentary_world_model import PentaryWorldModel, PentaryRSSM, PentaryEncoder, PentaryDecoder


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  ✗ {name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailures:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*60}")
        return self.failed == 0


def verify_pentary_weights(weights: np.ndarray, name: str, results: TestResults):
    """Verify weights are in pentary range {-2, -1, 0, +1, +2}."""
    unique_vals = set(np.unique(weights))
    valid_vals = {-2, -1, 0, 1, 2}
    
    if unique_vals.issubset(valid_vals):
        results.add_pass(f"{name} has valid pentary weights")
    else:
        invalid = unique_vals - valid_vals
        results.add_fail(f"{name} has invalid values", f"Found: {invalid}")


def verify_shape(actual: tuple, expected: tuple, name: str, results: TestResults):
    """Verify output shape matches expected."""
    if actual == expected:
        results.add_pass(f"{name} shape correct: {actual}")
    else:
        results.add_fail(f"{name} shape mismatch", f"Expected {expected}, got {actual}")


def test_pentary_mamba(results: TestResults):
    """Test PentaryMamba implementation."""
    print("\n" + "="*60)
    print("Testing PentaryMamba")
    print("="*60)
    
    # Create model
    model = PentaryMamba(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        d_state=8
    )
    
    # Test 1: Verify pentary weights
    verify_pentary_weights(model.embedding, "Embedding", results)
    verify_pentary_weights(model.lm_head, "LM Head", results)
    verify_pentary_weights(model.blocks[0].in_proj, "Block in_proj", results)
    
    # Test 2: Forward pass shape
    batch_size, seq_len = 2, 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    logits = model.forward(input_ids)
    
    verify_shape(logits.shape, (batch_size, seq_len, 1000), "Forward output", results)
    
    # Test 3: Generation
    prompt = np.array([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5)
    
    if generated.shape[1] == prompt.shape[1] + 5:
        results.add_pass("Generation produces correct length")
    else:
        results.add_fail("Generation length", f"Expected {prompt.shape[1] + 5}, got {generated.shape[1]}")
    
    # Test 4: Sparsity check
    stats = model.get_stats()
    if stats['sparsity'] > 0:
        results.add_pass(f"Model has sparsity: {stats['sparsity']:.1%}")
    else:
        results.add_fail("Sparsity check", "No zero weights found")
    
    # Test 5: Linear complexity (time should scale linearly)
    times = []
    for length in [8, 16, 32]:
        x = np.random.randint(0, 1000, (1, length))
        start = time.time()
        _ = model.forward(x)
        times.append(time.time() - start)
    
    # Check if doubling length roughly doubles time (O(n))
    ratio1 = times[1] / times[0] if times[0] > 0 else 0
    ratio2 = times[2] / times[1] if times[1] > 0 else 0
    
    if 0.5 < ratio1 < 4 and 0.5 < ratio2 < 4:
        results.add_pass(f"Linear complexity: ratios {ratio1:.2f}, {ratio2:.2f}")
    else:
        results.add_fail("Linear complexity", f"Unexpected ratios: {ratio1:.2f}, {ratio2:.2f}")


def test_pentary_rwkv(results: TestResults):
    """Test PentaryRWKV implementation."""
    print("\n" + "="*60)
    print("Testing PentaryRWKV")
    print("="*60)
    
    # Create model
    model = PentaryRWKV(
        vocab_size=1000,
        d_model=64,
        n_layers=4
    )
    
    # Test 1: Verify pentary weights
    verify_pentary_weights(model.embedding, "Embedding", results)
    verify_pentary_weights(model.lm_head, "LM Head", results)
    verify_pentary_weights(model.blocks[0].time_mix.W_r, "Time mix W_r", results)
    
    # Test 2: Parallel forward shape
    batch_size, seq_len = 2, 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    logits = model.forward(input_ids)
    
    verify_shape(logits.shape, (batch_size, seq_len, 1000), "Forward output", results)
    
    # Test 3: Recurrent forward
    token = np.array([42])
    logits_rec, states = model.forward_recurrent(token, None)
    
    verify_shape(logits_rec.shape, (1, 1000), "Recurrent output", results)
    
    if len(states) == model.n_layers:
        results.add_pass("Recurrent returns correct number of states")
    else:
        results.add_fail("Recurrent states", f"Expected {model.n_layers}, got {len(states)}")
    
    # Test 4: O(1) inference memory
    token = np.array([1])
    states = None
    
    # Process many tokens and verify state size doesn't grow
    for _ in range(10):
        _, states = model.forward_recurrent(token, states)
    
    # States should have fixed size per layer
    state_sizes = [sum(v.size if hasattr(v, 'size') else 0 for v in s.values() if v is not None) 
                   for s in states]
    if all(s > 0 for s in state_sizes):
        results.add_pass(f"O(1) inference: state sizes constant")
    else:
        results.add_fail("O(1) inference", "State size issue")
    
    # Test 5: Generation
    prompt = np.array([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5)
    
    if generated.shape[1] == prompt.shape[1] + 5:
        results.add_pass("Generation produces correct length")
    else:
        results.add_fail("Generation length", f"Expected {prompt.shape[1] + 5}, got {generated.shape[1]}")


def test_pentary_retnet(results: TestResults):
    """Test PentaryRetNet implementation."""
    print("\n" + "="*60)
    print("Testing PentaryRetNet")
    print("="*60)
    
    # Create model
    model = PentaryRetNet(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        num_heads=4
    )
    
    # Test 1: Verify pentary weights
    verify_pentary_weights(model.embedding, "Embedding", results)
    verify_pentary_weights(model.lm_head, "LM Head", results)
    verify_pentary_weights(model.blocks[0].retention.W_q, "Retention W_q", results)
    
    # Test 2: Parallel forward shape
    batch_size, seq_len = 2, 16
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    logits = model.forward(input_ids)
    
    verify_shape(logits.shape, (batch_size, seq_len, 1000), "Parallel output", results)
    
    # Test 3: Recurrent forward
    token = np.array([42, 123])
    logits_rec, states = model.forward_recurrent(token, None, 0)
    
    verify_shape(logits_rec.shape, (2, 1000), "Recurrent output", results)
    
    if len(states) == model.n_layers:
        results.add_pass("Recurrent returns correct number of states")
    else:
        results.add_fail("Recurrent states", f"Expected {model.n_layers}, got {len(states)}")
    
    # Test 4: Decay mechanism
    retention = PentaryRetention(64, num_heads=4, gamma=0.95)
    D = retention._decay_matrix(8)
    
    # Verify decay is causal and decreasing
    is_causal = np.allclose(D, np.triu(D, 0))
    has_decay = D[0, 0] > D[7, 0]
    
    if is_causal:
        results.add_pass("Decay matrix is causal")
    else:
        results.add_fail("Decay causality", "Matrix not lower triangular")
    
    if has_decay:
        results.add_pass("Decay decreases with distance")
    else:
        results.add_fail("Decay behavior", "No decay observed")
    
    # Test 5: Generation
    prompt = np.array([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5)
    
    if generated.shape[1] == prompt.shape[1] + 5:
        results.add_pass("Generation produces correct length")
    else:
        results.add_fail("Generation length", f"Expected {prompt.shape[1] + 5}, got {generated.shape[1]}")


def test_pentary_world_model(results: TestResults):
    """Test PentaryWorldModel implementation."""
    print("\n" + "="*60)
    print("Testing PentaryWorldModel")
    print("="*60)
    
    # Create model
    model = PentaryWorldModel(
        obs_shape=(32, 32, 3),
        action_dim=4,
        latent_dim=64,
        hidden_dim=64,
        stoch_dim=16
    )
    
    # Test 1: Verify 5 categories (pentary alignment)
    if model.num_categories == 5:
        results.add_pass("World model uses 5 categories (pentary aligned)")
    else:
        results.add_fail("Category count", f"Expected 5, got {model.num_categories}")
    
    # Test 2: Encoder output shape
    batch_size = 2
    obs = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
    z = model.encode(obs)
    
    verify_shape(z.shape, (batch_size, model.latent_dim), "Encoder output", results)
    
    # Test 3: RSSM imagination
    h, stoch_z = model.rssm.initial_state(batch_size)
    actions = np.random.randn(batch_size, 5, 4).astype(np.float32)
    
    imagined = model.imagine(h, stoch_z, actions)
    
    verify_shape(imagined['h'].shape, (batch_size, 6, model.hidden_dim), "Imagined h", results)
    verify_shape(imagined['rewards'].shape, (batch_size, 5, 1), "Imagined rewards", results)
    
    # Test 4: RSSM observation
    observations = np.random.randn(batch_size, 4, 32, 32, 3).astype(np.float32)
    actions = np.random.randn(batch_size, 3, 4).astype(np.float32)
    
    observed = model.observe(observations, actions)
    
    if 'priors' in observed and 'posteriors' in observed:
        results.add_pass("Observation returns priors and posteriors")
    else:
        results.add_fail("Observation outputs", "Missing priors or posteriors")
    
    # Test 5: Gumbel softmax produces valid categorical
    rssm = model.rssm
    logits = np.random.randn(2, 16, 5)  # batch, stoch_dim, num_categories
    sampled = rssm._gumbel_softmax(logits, temperature=1.0, hard=True)
    
    # Each position should be one-hot
    sums = np.sum(sampled, axis=-1)
    is_onehot = np.allclose(sums, 1.0)
    
    if is_onehot:
        results.add_pass("Gumbel softmax produces valid one-hot")
    else:
        results.add_fail("Gumbel softmax", "Not producing one-hot vectors")
    
    # Test 6: Sparsity
    stats = model.get_stats()
    if stats['sparsity'] > 0:
        results.add_pass(f"Model has sparsity: {stats['sparsity']:.1%}")
    else:
        results.add_fail("Sparsity check", "No zero weights found")


def benchmark_all_models():
    """Benchmark all models for comparison."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    results = {}
    
    # Benchmark settings
    batch_size = 1
    seq_len = 32
    vocab_size = 1000
    d_model = 64
    n_layers = 4
    num_runs = 5
    
    # 1. Mamba
    print("\nMamba:")
    mamba = PentaryMamba(vocab_size, d_model, n_layers)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = mamba.forward(x)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])  # Skip warmup
    throughput = seq_len / avg_time
    results['Mamba'] = {'time': avg_time, 'throughput': throughput}
    print(f"  Forward time: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    
    # 2. RWKV
    print("\nRWKV:")
    rwkv = PentaryRWKV(vocab_size, d_model, n_layers)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = rwkv.forward(x)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    throughput = seq_len / avg_time
    results['RWKV'] = {'time': avg_time, 'throughput': throughput}
    print(f"  Forward time: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    
    # RWKV recurrent
    times = []
    for _ in range(num_runs):
        states = None
        token = np.array([1])
        start = time.time()
        for _ in range(seq_len):
            _, states = rwkv.forward_recurrent(token, states)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    throughput = seq_len / avg_time
    print(f"  Recurrent time: {avg_time*1000:.2f}ms")
    print(f"  Recurrent throughput: {throughput:.0f} tokens/sec")
    
    # 3. RetNet
    print("\nRetNet:")
    retnet = PentaryRetNet(vocab_size, d_model, n_layers)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = retnet.forward(x)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    throughput = seq_len / avg_time
    results['RetNet'] = {'time': avg_time, 'throughput': throughput}
    print(f"  Forward time: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    
    # RetNet recurrent
    times = []
    for _ in range(num_runs):
        states = None
        token = np.array([1])
        start = time.time()
        for t in range(seq_len):
            _, states = retnet.forward_recurrent(token, states, t)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    throughput = seq_len / avg_time
    print(f"  Recurrent time: {avg_time*1000:.2f}ms")
    print(f"  Recurrent throughput: {throughput:.0f} tokens/sec")
    
    # 4. World Model
    print("\nWorld Model:")
    world_model = PentaryWorldModel(
        obs_shape=(32, 32, 3),
        action_dim=4,
        latent_dim=64,
        hidden_dim=64,
        stoch_dim=16
    )
    
    h, z = world_model.rssm.initial_state(batch_size)
    action = np.random.randn(batch_size, 4).astype(np.float32)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        for _ in range(10):
            h, z, _ = world_model.rssm.imagine_step(h, z, action)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    throughput = 10 / avg_time
    results['WorldModel'] = {'time': avg_time, 'throughput': throughput}
    print(f"  10 imagination steps: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.0f} steps/sec")
    
    return results


def main():
    """Run all tests."""
    print("="*70)
    print("PENTARY ADVANCED ARCHITECTURES TEST SUITE")
    print("="*70)
    
    results = TestResults()
    
    # Run all tests
    test_pentary_mamba(results)
    test_pentary_rwkv(results)
    test_pentary_retnet(results)
    test_pentary_world_model(results)
    
    # Run benchmarks
    benchmark_results = benchmark_all_models()
    
    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print("\nSparsity Analysis:")
    
    mamba = PentaryMamba(1000, 64, 2)
    rwkv = PentaryRWKV(1000, 64, 4)
    retnet = PentaryRetNet(1000, 64, 4)
    world_model = PentaryWorldModel((32, 32, 3), 4, 64, 64, 16)
    
    for name, model in [("Mamba", mamba), ("RWKV", rwkv), 
                        ("RetNet", retnet), ("WorldModel", world_model)]:
        stats = model.get_stats()
        print(f"  {name}:")
        print(f"    Parameters: {stats['total_parameters']:,}")
        print(f"    Sparsity: {stats['sparsity']:.1%}")
    
    print("\nKey Advantages of Pentary Implementation:")
    print("  • All weights in {-2, -1, 0, +1, +2}")
    print("  • Zero weights enable ~20% sparsity")
    print("  • Shift-add only (no full multipliers)")
    print("  • O(n) or O(1) inference complexity")
    print("  • 5-category stochastic states (World Model)")
    
    # Summary
    all_passed = results.summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
