#!/usr/bin/env python3
"""
Tests for compiler proof metadata and pentary AI quantization certificates.
"""

import os
import sys

# Add repository paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
LANG_DIR = os.path.join(ROOT_DIR, 'language')
sys.path.insert(0, LANG_DIR)
sys.path.insert(0, ROOT_DIR)


def test_compiler_register_proof():
    from pent_compiler import Compiler

    source = """
    fn main() {
        let x = 1;
        let y = 2;
        let z = x + y;
        return z;
    }
    """

    compiler = Compiler()
    assembly, proof = compiler.compile_with_proof(source)

    assert assembly
    assert proof["passed"] is True
    assert proof["register_limit"] == 28
    assert proof["max_register"] <= proof["register_limit"]
    assert proof["uses_zero_register"] is False
    assert proof["registers_used"]


def test_transformer_quantization_certificate_bounds():
    from tools.pentary_transformer import PentaryTransformer

    model = PentaryTransformer(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        num_layers=1,
        d_ff=32,
        max_seq_len=8
    )

    certificates = model.get_quantization_certificates()
    embedding_cert = certificates["embeddings"]["token_embedding"]
    output_cert = certificates["embeddings"]["output_proj"]

    assert embedding_cert["max_abs_error"] <= embedding_cert["error_bound"] + 1e-6
    assert output_cert["max_abs_error"] <= output_cert["error_bound"] + 1e-6

    block_cert = certificates["blocks"]["block_0"]["attention"]["W_q"]
    assert block_cert["max_abs_error"] <= block_cert["error_bound"] + 1e-6
