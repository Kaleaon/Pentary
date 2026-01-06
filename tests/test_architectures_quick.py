#!/usr/bin/env python3
"""
Quick Test Suite for Pentary Advanced Architectures

Fast tests to verify basic functionality without full benchmarks.
Can be run with pytest or directly.
"""

import sys
import os
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

# Valid pentary values (defined globally to avoid scope issues)
VALID_PENTARY_VALS = {-2, -1, 0, 1, 2}


class TestPentaryMamba:
    """Tests for PentaryMamba implementation."""
    
    def test_mamba_import(self):
        """Test that PentaryMamba can be imported."""
        from pentary_mamba import PentaryMamba, PentaryQuantizer
        assert PentaryMamba is not None
        assert PentaryQuantizer is not None
    
    def test_mamba_pentary_weights(self):
        """Test that Mamba uses pentary weights."""
        from pentary_mamba import PentaryMamba
        model = PentaryMamba(vocab_size=100, d_model=32, n_layers=1, d_state=4)
        unique_vals = set(np.unique(model.embedding))
        assert unique_vals.issubset(VALID_PENTARY_VALS), f"Invalid values: {unique_vals - VALID_PENTARY_VALS}"
    
    def test_mamba_forward(self):
        """Test Mamba forward pass."""
        from pentary_mamba import PentaryMamba
        model = PentaryMamba(vocab_size=100, d_model=32, n_layers=1, d_state=4)
        x = np.array([[1, 2, 3]])
        out = model.forward(x)
        assert out.shape == (1, 3, 100), f"Expected (1, 3, 100), got {out.shape}"
    
    def test_mamba_sparsity(self):
        """Test that Mamba has sparsity from zero weights."""
        from pentary_mamba import PentaryMamba
        model = PentaryMamba(vocab_size=100, d_model=32, n_layers=1, d_state=4)
        stats = model.get_stats()
        assert stats['sparsity'] > 0, "Expected non-zero sparsity"


class TestPentaryRWKV:
    """Tests for PentaryRWKV implementation."""
    
    def test_rwkv_import(self):
        """Test that PentaryRWKV can be imported."""
        from pentary_rwkv import PentaryRWKV
        assert PentaryRWKV is not None
    
    def test_rwkv_pentary_weights(self):
        """Test that RWKV uses pentary weights."""
        from pentary_rwkv import PentaryRWKV
        model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
        unique_vals = set(np.unique(model.embedding))
        assert unique_vals.issubset(VALID_PENTARY_VALS), f"Invalid values: {unique_vals - VALID_PENTARY_VALS}"
    
    def test_rwkv_parallel_forward(self):
        """Test RWKV parallel forward pass."""
        from pentary_rwkv import PentaryRWKV
        model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
        x = np.array([[1, 2, 3, 4]])
        out = model.forward(x)
        assert out.shape == (1, 4, 100), f"Expected (1, 4, 100), got {out.shape}"
    
    def test_rwkv_recurrent_forward(self):
        """Test RWKV recurrent forward pass."""
        from pentary_rwkv import PentaryRWKV
        model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
        token = np.array([42])
        logits, states = model.forward_recurrent(token, None)
        assert logits.shape == (1, 100), f"Expected (1, 100), got {logits.shape}"
        assert len(states) == 2, f"Expected 2 states, got {len(states)}"
    
    def test_rwkv_sparsity(self):
        """Test that RWKV has sparsity."""
        from pentary_rwkv import PentaryRWKV
        model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
        stats = model.get_stats()
        assert stats['sparsity'] > 0, "Expected non-zero sparsity"


class TestPentaryRetNet:
    """Tests for PentaryRetNet implementation."""
    
    def test_retnet_import(self):
        """Test that PentaryRetNet can be imported."""
        from pentary_retnet import PentaryRetNet, PentaryRetention
        assert PentaryRetNet is not None
        assert PentaryRetention is not None
    
    def test_retnet_pentary_weights(self):
        """Test that RetNet uses pentary weights."""
        from pentary_retnet import PentaryRetNet
        model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
        unique_vals = set(np.unique(model.embedding))
        assert unique_vals.issubset(VALID_PENTARY_VALS), f"Invalid values: {unique_vals - VALID_PENTARY_VALS}"
    
    def test_retnet_parallel_forward(self):
        """Test RetNet parallel forward pass."""
        from pentary_retnet import PentaryRetNet
        model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
        x = np.array([[1, 2, 3, 4]])
        out = model.forward(x)
        assert out.shape == (1, 4, 100), f"Expected (1, 4, 100), got {out.shape}"
    
    def test_retnet_recurrent_forward(self):
        """Test RetNet recurrent forward pass."""
        from pentary_retnet import PentaryRetNet
        model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
        token = np.array([42])
        logits, states = model.forward_recurrent(token, None, 0)
        assert logits.shape == (1, 100), f"Expected (1, 100), got {logits.shape}"
        assert len(states) == 2, f"Expected 2 states, got {len(states)}"
    
    def test_retnet_decay_matrix_causal(self):
        """Test that RetNet decay matrix is causal."""
        from pentary_retnet import PentaryRetention
        retention = PentaryRetention(32, num_heads=2, gamma=0.95)
        D = retention._decay_matrix(4)
        # Check that upper triangle (future) is zero and lower triangle (past) has decay
        assert D[0, 3] == 0, "Future tokens should have zero weight"
        assert D[3, 0] > 0, "Past tokens should have positive weight"
    
    def test_retnet_sparsity(self):
        """Test that RetNet has sparsity."""
        from pentary_retnet import PentaryRetNet
        model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
        stats = model.get_stats()
        assert stats['sparsity'] > 0, "Expected non-zero sparsity"


class TestPentaryWorldModel:
    """Tests for PentaryWorldModel implementation."""
    
    def test_worldmodel_import(self):
        """Test that PentaryWorldModel can be imported."""
        from pentary_world_model import PentaryWorldModel, PentaryRSSM
        assert PentaryWorldModel is not None
        assert PentaryRSSM is not None
    
    def test_worldmodel_categories(self):
        """Test that WorldModel uses 5 categories (pentary aligned)."""
        from pentary_world_model import PentaryWorldModel
        model = PentaryWorldModel(
            obs_shape=(16, 16, 3),
            action_dim=4,
            latent_dim=32,
            hidden_dim=32,
            stoch_dim=8
        )
        assert model.num_categories == 5, f"Expected 5 categories, got {model.num_categories}"
    
    def test_worldmodel_encoder(self):
        """Test WorldModel encoder."""
        from pentary_world_model import PentaryWorldModel
        model = PentaryWorldModel(
            obs_shape=(16, 16, 3),
            action_dim=4,
            latent_dim=32,
            hidden_dim=32,
            stoch_dim=8
        )
        obs = np.random.randn(1, 16, 16, 3).astype(np.float32)
        z = model.encode(obs)
        assert z.shape == (1, 32), f"Expected (1, 32), got {z.shape}"
    
    def test_worldmodel_rssm_imagine(self):
        """Test WorldModel RSSM imagination step."""
        from pentary_world_model import PentaryWorldModel
        model = PentaryWorldModel(
            obs_shape=(16, 16, 3),
            action_dim=4,
            latent_dim=32,
            hidden_dim=32,
            stoch_dim=8
        )
        h, stoch_z = model.rssm.initial_state(1)
        action = np.random.randn(1, 4).astype(np.float32)
        h_new, z_new, logits = model.rssm.imagine_step(h, stoch_z, action)
        assert h_new.shape == (1, 32), f"Expected (1, 32), got {h_new.shape}"
    
    def test_worldmodel_gumbel_softmax(self):
        """Test Gumbel softmax produces one-hot outputs."""
        from pentary_world_model import PentaryWorldModel
        model = PentaryWorldModel(
            obs_shape=(16, 16, 3),
            action_dim=4,
            latent_dim=32,
            hidden_dim=32,
            stoch_dim=8
        )
        logits = np.random.randn(1, 8, 5)
        sampled = model.rssm._gumbel_softmax(logits, hard=True)
        sums = np.sum(sampled, axis=-1)
        assert np.allclose(sums, 1.0), "Gumbel softmax should produce one-hot vectors"
    
    def test_worldmodel_sparsity(self):
        """Test that WorldModel has sparsity."""
        from pentary_world_model import PentaryWorldModel
        model = PentaryWorldModel(
            obs_shape=(16, 16, 3),
            action_dim=4,
            latent_dim=32,
            hidden_dim=32,
            stoch_dim=8
        )
        stats = model.get_stats()
        assert stats['sparsity'] > 0, "Expected non-zero sparsity"


def run_quick_tests():
    """Run all quick tests manually (for standalone execution)."""
    print("="*70)
    print("PENTARY ADVANCED ARCHITECTURES - QUICK TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    def test(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1
    
    # Test 1: PentaryMamba
    print("\n[1/4] Testing PentaryMamba...")
    try:
        from pentary_mamba import PentaryMamba, PentaryQuantizer
        
        model = PentaryMamba(vocab_size=100, d_model=32, n_layers=1, d_state=4)
        
        unique_vals = set(np.unique(model.embedding))
        test("Embedding uses pentary weights", unique_vals.issubset(VALID_PENTARY_VALS))
        
        x = np.array([[1, 2, 3]])
        out = model.forward(x)
        test("Forward produces output", out.shape == (1, 3, 100))
        
        stats = model.get_stats()
        test("Has sparsity", stats['sparsity'] > 0)
        
        print(f"  [Mamba OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
        
    except Exception as e:
        print(f"  ✗ Mamba failed: {e}")
        failed += 1
    
    # Test 2: PentaryRWKV
    print("\n[2/4] Testing PentaryRWKV...")
    try:
        from pentary_rwkv import PentaryRWKV
        
        model = PentaryRWKV(vocab_size=100, d_model=32, n_layers=2)
        
        unique_vals = set(np.unique(model.embedding))
        test("Embedding uses pentary weights", unique_vals.issubset(VALID_PENTARY_VALS))
        
        x = np.array([[1, 2, 3, 4]])
        out = model.forward(x)
        test("Parallel forward works", out.shape == (1, 4, 100))
        
        token = np.array([42])
        logits, states = model.forward_recurrent(token, None)
        test("Recurrent forward works", logits.shape == (1, 100))
        test("Returns states", len(states) == 2)
        
        stats = model.get_stats()
        test("Has sparsity", stats['sparsity'] > 0)
        
        print(f"  [RWKV OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
        
    except Exception as e:
        print(f"  ✗ RWKV failed: {e}")
        failed += 1
    
    # Test 3: PentaryRetNet
    print("\n[3/4] Testing PentaryRetNet...")
    try:
        from pentary_retnet import PentaryRetNet, PentaryRetention
        
        model = PentaryRetNet(vocab_size=100, d_model=32, n_layers=2, num_heads=2)
        
        unique_vals = set(np.unique(model.embedding))
        test("Embedding uses pentary weights", unique_vals.issubset(VALID_PENTARY_VALS))
        
        x = np.array([[1, 2, 3, 4]])
        out = model.forward(x)
        test("Parallel forward works", out.shape == (1, 4, 100))
        
        token = np.array([42])
        logits, states = model.forward_recurrent(token, None, 0)
        test("Recurrent forward works", logits.shape == (1, 100))
        test("Returns states", len(states) == 2)
        
        retention = PentaryRetention(32, num_heads=2, gamma=0.95)
        D = retention._decay_matrix(4)
        test("Decay matrix is causal", D[0, 3] == 0 and D[3, 0] > 0)
        
        stats = model.get_stats()
        test("Has sparsity", stats['sparsity'] > 0)
        
        print(f"  [RetNet OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
        
    except Exception as e:
        print(f"  ✗ RetNet failed: {e}")
        failed += 1
    
    # Test 4: PentaryWorldModel
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
        
        test("Uses 5 categories", model.num_categories == 5)
        
        obs = np.random.randn(1, 16, 16, 3).astype(np.float32)
        z = model.encode(obs)
        test("Encoder works", z.shape == (1, 32))
        
        h, stoch_z = model.rssm.initial_state(1)
        action = np.random.randn(1, 4).astype(np.float32)
        h_new, z_new, logits = model.rssm.imagine_step(h, stoch_z, action)
        test("RSSM imagine works", h_new.shape == (1, 32))
        
        logits = np.random.randn(1, 8, 5)
        sampled = model.rssm._gumbel_softmax(logits, hard=True)
        sums = np.sum(sampled, axis=-1)
        test("Gumbel softmax produces one-hot", np.allclose(sums, 1.0))
        
        stats = model.get_stats()
        test("Has sparsity", stats['sparsity'] > 0)
        
        print(f"  [WorldModel OK: {stats['total_parameters']:,} params, {stats['sparsity']:.1%} sparse]")
        
    except Exception as e:
        print(f"  ✗ WorldModel failed: {e}")
        failed += 1
    
    # Summary
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
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_quick_tests()
    if failed == 0:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} tests failed")
        sys.exit(1)
