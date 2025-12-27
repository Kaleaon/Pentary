# Pentary Research References

This directory contains research papers, historical documents, and technical references related to pentary computing, ternary logic, 3-transistor circuits, neural network quantization, and open-source PDK implementations.

**Total Papers:** 79  
**Total Size:** ~309MB  
**Last Updated:** December 2024

## Directory Structure

```
references/
├── papers/              # Academic papers and publications
├── research/            # Technical research documents  
├── trinary-systems/     # Historical trinary/ternary computing
├── PAPER_ARCHIVE.md     # Comprehensive paper index with compatibility ratings
├── RESEARCH_INDEX.md    # Research topic index
└── README.md            # This file
```

---

## Paper Categories

### 1. Neural Network Quantization (12 papers)

Essential papers for understanding low-bit quantization techniques applicable to pentary:

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `gptq.pdf` | GPTQ: 3-4 bit post-training quantization | ⭐⭐⭐⭐⭐ Foundation for pentary weights |
| `awq.pdf` | AWQ: Activation-aware weight quantization | ⭐⭐⭐⭐⭐ Preserves salient channels |
| `lsq.pdf` | Learned Step Size Quantization | ⭐⭐⭐⭐⭐ Training-aware quantization |
| `bitsandbytes_qlora.pdf` | QLoRA: 4-bit finetuning | ⭐⭐⭐⭐⭐ Efficient pentary finetuning |
| `smoothquant.pdf` | SmoothQuant: Activation smoothing | ⭐⭐⭐⭐⭐ W8A8 quantization |
| `squeezellm.pdf` | SqueezeLLM: Dense-sparse quantization | ⭐⭐⭐⭐ Non-uniform quantization |
| `llm_int8_original.pdf` | LLM.int8(): 8-bit inference | ⭐⭐⭐⭐ Large model inference |
| `hqq_quantization.pdf` | HQQ: Fast calibration-free | ⭐⭐⭐⭐⭐ Zero-shot quantization |
| `marlin_kernel.pdf` | Marlin: Fast 4-bit kernels | ⭐⭐⭐⭐ Inference optimization |

### 2. State Space Models & Linear Attention (8 papers)

Architectures highly compatible with pentary computing:

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `mamba.pdf` | Mamba: Selective SSM | ⭐⭐⭐⭐⭐ Linear O(n), ideal for pentary |
| `mamba2_ssd.pdf` | Mamba-2: SSM-Attention duality | ⭐⭐⭐⭐⭐ 2-8× faster |
| `rwkv.pdf` | RWKV: RNN-Transformer hybrid | ⭐⭐⭐⭐⭐ O(1) inference |
| `retnet.pdf` | RetNet: Retention mechanism | ⭐⭐⭐⭐⭐ Three paradigms |
| `griffin.pdf` | Griffin: Gated recurrence + attention | ⭐⭐⭐⭐⭐ Hybrid efficiency |
| `jamba.pdf` | Jamba: Mamba-Transformer hybrid | ⭐⭐⭐⭐⭐ Best of both |
| `samba.pdf` | Samba: Simple hybrid SSM | ⭐⭐⭐⭐⭐ Practical hybrid |
| `titans_memory.pdf` | Titans: Test-time memorization | ⭐⭐⭐⭐⭐ Memory-efficient |

### 3. Foundation Models (15 papers)

Open-source LLMs for benchmarking pentary implementations:

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `llama.pdf` | LLaMA: Original efficient LLM | ⭐⭐⭐⭐ Baseline architecture |
| `llama2.pdf` | Llama 2: RLHF + longer context | ⭐⭐⭐⭐ Improved baseline |
| `llama3.pdf` | Llama 3: Latest architecture | ⭐⭐⭐⭐ Current SOTA |
| `mistral7b.pdf` | Mistral 7B: Sliding window | ⭐⭐⭐⭐ Efficient attention |
| `mixtral_moe.pdf` | Mixtral: Sparse MoE | ⭐⭐⭐⭐ Expert routing |
| `deepseekv2.pdf` | DeepSeek-V2: MLA architecture | ⭐⭐⭐⭐⭐ Extreme efficiency |
| `deepseek_moe.pdf` | DeepSeekMoE: Efficient MoE | ⭐⭐⭐⭐ MoE design |
| `deepseek_llm.pdf` | DeepSeek LLM: Open foundation | ⭐⭐⭐⭐ Efficient training |
| `gemma.pdf` | Gemma: Google open model | ⭐⭐⭐⭐ Clean architecture |
| `gemma2.pdf` | Gemma 2: Improved Gemma | ⭐⭐⭐⭐ Better efficiency |
| `phi2.pdf` | Phi-2: Small but capable | ⭐⭐⭐⭐⭐ Data-efficient |
| `phi3.pdf` | Phi-3: Latest small model | ⭐⭐⭐⭐⭐ Best small model |
| `olmo.pdf` | OLMo: Fully open-source | ⭐⭐⭐⭐ Reproducible |
| `qwen.pdf` | Qwen: Multilingual | ⭐⭐⭐⭐ Strong baseline |
| `qwen2.pdf` | Qwen2: Improved Qwen | ⭐⭐⭐⭐ Better efficiency |

### 4. Low-Bit Neural Networks (6 papers)

Direct comparison points for pentary:

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `binarized_nn.pdf` | Binary Neural Networks | ⭐⭐⭐⭐ 1-bit baseline |
| `ternary_weight_networks.pdf` | Ternary {-1,0,+1} weights | ⭐⭐⭐⭐⭐ **Direct comparison** |
| `bitnet.pdf` | BitNet: 1-bit LLMs | ⭐⭐⭐⭐⭐ **Direct comparison** |
| `bitnet_1_58.pdf` | BitNet b1.58: Ternary LLMs | ⭐⭐⭐⭐⭐ **Key competitor** |
| `sparse_llm.pdf` | SparseGPT: 50% sparsity | ⭐⭐⭐⭐ Sparsity techniques |
| `knowledge_distillation.pdf` | Knowledge Distillation | ⭐⭐⭐⭐ Model compression |

### 5. Attention & Memory Optimization (8 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `flash_attention.pdf` | FlashAttention: IO-aware | ⭐⭐⭐⭐ Tiling for pentary |
| `flashattention2.pdf` | FlashAttention-2: 2× faster | ⭐⭐⭐⭐ Better parallelism |
| `flashattention3.pdf` | FlashAttention-3: Async | ⭐⭐⭐⭐ Latest optimizations |
| `ring_attention.pdf` | Ring Attention: Distributed | ⭐⭐⭐⭐ Long context |
| `gqa.pdf` | GQA: Grouped-Query Attention | ⭐⭐⭐⭐ Memory efficient |
| `pageattention_vllm.pdf` | vLLM: Paged KV cache | ⭐⭐⭐⭐⭐ Memory management |
| `kv_cache_compression.pdf` | KV Cache Compression | ⭐⭐⭐⭐⭐ Pentary KV cache |
| `longllama.pdf` | LongLLaMA: Long context | ⭐⭐⭐⭐ Context extension |

### 6. Speculative Decoding (4 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `speculative_decoding.pdf` | Staged Speculative Decoding | ⭐⭐⭐⭐ Draft models |
| `eagle_speculative.pdf` | EAGLE: Feature speculation | ⭐⭐⭐⭐ 2-3× speedup |
| `medusa.pdf` | Medusa: Multiple heads | ⭐⭐⭐⭐ Parallel speculation |
| `lookahead_decoding.pdf` | Lookahead Decoding | ⭐⭐⭐⭐ N-gram generation |

### 7. Positional Encodings (4 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `rope.pdf` | RoPE: Rotary Position | ⭐⭐⭐⭐ Standard encoding |
| `pope.pdf` | PoPE: Polar coordinates | ⭐⭐⭐⭐⭐ Angular = pentary |
| `yarn.pdf` | YaRN: Context scaling | ⭐⭐⭐⭐ Long context |
| `longllama.pdf` | Long context techniques | ⭐⭐⭐⭐ Extension methods |

### 8. Hardware & Accelerators (5 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `tpu_v1.pdf` | Google TPU Architecture | ⭐⭐⭐⭐⭐ Benchmark target |
| `groq_inference.pdf` | Groq: Deterministic inference | ⭐⭐⭐⭐ Comparison point |
| `research/llm_efficiency_survey.pdf` | LLM Efficiency Survey | ⭐⭐⭐⭐⭐ Comprehensive |
| `neuromorphic_review.pdf` | Neuromorphic Computing | ⭐⭐⭐⭐⭐ Implementation path |
| `moe_survey.pdf` | MoE Survey | ⭐⭐⭐⭐ Expert systems |

### 9. In-Memory & Analog Computing (5 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `research/in_memory_computing_review.pdf` | In-Memory Computing | ⭐⭐⭐⭐⭐ Core concept |
| `research/analog_ai_hardware.pdf` | Analog AI Hardware | ⭐⭐⭐⭐⭐ Implementation |
| `research/memristor_crossbar_nn.pdf` | Memristor Crossbar | ⭐⭐⭐⭐⭐ Hardware design |
| `research/Memristor_CMOS_Ternary_Logic.pdf` | Memristor-CMOS Hybrid | ⭐⭐⭐⭐⭐ Circuit design |

### 10. Ternary/Multi-Valued Logic (6 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `trinary-systems/Setun_Ternary_Computer_HAL.pdf` | Setun Computer | ⭐⭐⭐⭐⭐ Historical |
| `trinary-systems/Ternary_Computing_Cybersecurity.pdf` | Ternary Security | ⭐⭐⭐⭐ Applications |
| `research/Ternary_CMOS_Standard_Cell_Design.pdf` | Ternary Cells | ⭐⭐⭐⭐⭐ Cell design |
| `research/Efficient_Ternary_Logic_Circuits.pdf` | Ternary Optimization | ⭐⭐⭐⭐⭐ Arithmetic |
| `research/Ternary_Logic_Integrated_Circuits.pdf` | Ternary ICs | ⭐⭐⭐⭐⭐ Full methodology |

### 11. Diffusion Models (2 papers)

| Paper | Description | Pentary Relevance |
|-------|-------------|-------------------|
| `ddpm.pdf` | DDPM: Denoising Diffusion | ⭐⭐⭐ Many timesteps |
| `latent_diffusion.pdf` | Latent Diffusion | ⭐⭐⭐⭐ Latent space |

---

## Quick Reference

### Key Findings

1. **Ternary is Proven**: Multiple successful implementations in CMOS
2. **3TL Works**: 25% transistor reduction validated
3. **Pentary Potential**: Higher information density (2.32 bits/digit)
4. **Quantization Ready**: 4-5 bit quantization widely validated
5. **Open-Source Ready**: Compatible with sky130A PDK

### Most Important Papers for Pentary

| Priority | Paper | Why |
|----------|-------|-----|
| 1 | `bitnet_1_58.pdf` | Direct comparison: ternary vs pentary |
| 2 | `mamba.pdf` + `mamba2_ssd.pdf` | Ideal architecture for pentary |
| 3 | `awq.pdf` + `gptq.pdf` | Quantization methodology |
| 4 | `tpu_v1.pdf` | Hardware benchmark |
| 5 | `research/Ternary_CMOS_Standard_Cell_Design.pdf` | Circuit design |

### Application to Pentary

- **Voltage level generation**: From ternary research
- **Comparator-based detection**: Proven techniques
- **Arithmetic circuits**: Extend ternary to pentary
- **Standard cells**: Create pentary library

---

## Citation

When using these references, please cite the original authors. See individual PDF files for full citation information.

## Contributing

To add new research:
1. Download PDF to appropriate subdirectory
2. Update `PAPER_ARCHIVE.md` with details
3. Add entry to this README
4. Commit with descriptive message

## License

Research papers are copyright of their respective authors and publishers. Included here for educational and research purposes under fair use.

---

**Last Updated:** December 2024  
**Total Papers:** 79  
**Total Size:** ~309MB
