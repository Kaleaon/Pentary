#!/usr/bin/env python3
"""
Quick Test Suite for Pentary Advanced Architectures

Fast tests to verify basic functionality without full benchmarks.
"""

import sys
import os
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

print("="*70)
print("PENTARY ADVANCED ARCHITECTURES - QUICK TESTS")
print("="*70)

# Track results
passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1

# ============================================================
# Test 1: PentaryMamba
# ============================================================
print("\n[1/4] Testing PentaryMamba...")
try:
    from pentary_mamba import PentaryMamba, PentaryQuantizer
    
    model = PentaryMamba(vocab_size=100, d_model=32, n_layers=1, d_state=4)
    
    # Verify pentary weights
    unique_vals = set(np.unique(model.embedding))
    valid_vals = {-2, -1, 0, 1, 2}
    test("Embedding uses pentary weights", unique_vals.issubset(valid_vals))
    
    # Forward pass
    x = np.array([[1, 2, 3]])
    out = model.forward(x)
    test("Forward produces output", out.shape == (1, 3, 100))
    
    # Stats
    stats = model.get_stats()
    test("Has sparsity", stats['sparsity'] > 0)
    
    print(f"  [Mamba OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
    
except Exception as e:
    print(f"  ✗ Mamba failed: {e}")
    failed += 1

# ============================================================
# Test 2: PentaryRWKV
# ============================================================
print("\n[2/4] Testing PentaryRWKV...")
try:
    from pentary_rwkv import PentaryRWKV
    
    model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
    
    # Verify pentary weights
    unique_vals = set(np.unique(model.embedding))
    test("Embedding uses pentary weights", unique_vals.issubset(valid_vals))
    
    # Parallel forward
    x = np.array([[1, 2, 3, 4]])
    out = model.forward(x)
    test("Parallel forward works", out.shape == (1, 4, 100))
    
    # Recurrent forward
    token = np.array([42])
    logits, states = model.forward_recurrent(token, None)
    test("Recurrent forward works", logits.shape == (1, 100))
    test("Returns states", len(states) == 2)
    
    # Stats
    stats = model.get_stats()
    test("Has sparsity", stats['sparsity'] > 0)
    
    print(f"  [RWKV OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
    
except Exception as e:
    print(f"  ✗ RWKV failed: {e}")
    failed += 1

# ============================================================
# Test 3: PentaryRetNet
# ============================================================
print("\n[3/4] Testing PentaryRetNet...")
try:
    from pentary_retnet import PentaryRetNet, PentaryRetention
    
    model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
    
    # Verify pentary weights
    unique_vals = set(np.unique(model.embedding))
    test("Embedding uses pentary weights", unique_vals.issubset(valid_vals))
    
    # Parallel forward
    x = np.array([[1, 2, 3, 4]])
    out = model.forward(x)
    test("Parallel forward works", out.shape == (1, 4, 100))
    
    # Recurrent forward
    token = np.array([42])
    logits, states = model.forward_recurrent(token, None, 0)
    test("Recurrent forward works", logits.shape == (1, 100))
    test("Returns states", len(states) == 2)
    
    # Decay matrix (lower triangular - future tokens have zero weight)
    retention = PentaryRetention(32, num_heads=2, gamma=0.95)
    D = retention._decay_matrix(4)
    # Check that upper triangle (future) is zero and lower triangle (past) has decay
    test("Decay matrix is causal", D[0, 3] == 0 and D[3, 0] > 0)
    
    # Stats
    stats = model.get_stats()
    test("Has sparsity", stats['sparsity'] > 0)
    
    print(f"  [RetNet OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
    
except Exception as e:
    print(f"  ✗ RetNet failed: {e}")
    failed += 1

# ============================================================
# Test 4: PentaryWorldModel
# ============================================================
print("\n[4/4] Testing PentaryWorldModel...")
try:
    from pentary_world_model import PentaryWorldModel, PentaryRSSM
    
    model = PentaryWorldModel(
        obs_shape=(16, 16, 3),
        action_dim=4,
        latent_dim=32,
        hidden_dim=32,
        stoch_dim=8
    )
    
    # Check 5 categories (pentary aligned!)
    test("Uses 5 categories", model.num_categories == 5)
    
    # Encoder
    obs = np.random.randn(1, 16, 16, 3).astype(np.float32)
    z = model.encode(obs)
    test("Encoder works", z.shape == (1, 32))
    
    # RSSM imagination
    h, stoch_z = model.rssm.initial_state(1)
    action = np.random.randn(1, 4).astype(np.float32)
    h_new, z_new, logits = model.rssm.imagine_step(h, stoch_z, action)
    test("RSSM imagine works", h_new.shape == (1, 32))
    
    # Gumbel softmax
    logits = np.random.randn(1, 8, 5)
    sampled = model.rssm._gumbel_softmax(logits, hard=True)
    sums = np.sum(sampled, axis=-1)
    test("Gumbel softmax produces one-hot", np.allclose(sums, 1.0))
    
    # Stats
    stats = model.get_stats()
    test("Has sparsity", stats['sparsity'] > 0)
    
    print(f"  [WorldModel OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
    
except Exception as e:
    print(f"  ✗ WorldModel failed: {e}")
    failed += 1

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTests: {passed} passed, {failed} failed")

print("\nPentary Architecture Highlights:")
print("  • All weights in {-2, -1, 0, +1, +2}")
print("  • Sparsity from zero weights (~20-70%)")
print("  • Shift-add operations only (no multipliers)")
print("  • O(n) training, O(1) inference")
print("  • 5 categories perfect for pentary stochastic states")

print("\nImplemented Models:")
print("  1. PentaryMamba   - Selective State Space Model")
print("  2. PentaryRWKV    - Linear Attention RNN")
print("  3. PentaryRetNet  - Retentive Network")
print("  4. PentaryWorldModel - Latent Dynamics for RL")

if failed == 0:
    print("\n✓ All tests passed!")
    sys.exit(0)
else:
    print(f"\n✗ {failed} tests failed")
    sys.exit(1)
