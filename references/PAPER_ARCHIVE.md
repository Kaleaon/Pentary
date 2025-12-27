# Pentary Paper Archive

This archive contains references to key papers analyzed for compatibility with Pentary computing architecture.

**Total Papers:** 79  
**Total Size:** ~309MB  
**Last Updated:** December 2024

---

## Table of Contents

1. [State Space Models (SSMs)](#state-space-models-ssms)
2. [Linear Attention / Efficient Transformers](#linear-attention--efficient-transformers)
3. [Diffusion Models](#diffusion-models)
4. [Foundation Models](#foundation-models)
5. [Quantization Papers](#quantization-papers)
6. [Speculative Decoding](#speculative-decoding)
7. [KV Cache & Memory Optimization](#kv-cache--memory-optimization)
8. [Positional Encodings](#positional-encodings)
9. [Low-Bit Neural Networks](#low-bit-neural-networks)
10. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
11. [Hardware & Accelerators](#hardware--accelerators)
12. [In-Memory Computing & Neuromorphic](#in-memory-computing--neuromorphic)
13. [Ternary/Multi-Valued Logic](#ternarymulti-valued-logic)

---

## State Space Models (SSMs)

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Albert Gu, Tri Dao
- **arXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Introduces selective state space models that achieve 5× higher throughput than Transformers with linear scaling in sequence length. Mamba-3B outperforms Transformers of the same size.
- **Key Features:**
  - Content-based selective mechanism for SSM parameters
  - Linear O(n) complexity vs O(n²) for attention
  - Hardware-aware parallel algorithm
  - No attention or MLP blocks needed
- **PDF:** `mamba.pdf`

### Mamba-2: Transformers are SSMs (State Space Duality)
- **Authors:** Tri Dao, Albert Gu  
- **arXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)
- **Date:** May 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Shows theoretical connection between SSMs and attention through structured semiseparable matrices. Mamba-2 is 2-8× faster than Mamba-1 while matching Transformer performance.
- **Key Features:**
  - State Space Duality (SSD) framework
  - Connects SSMs and attention theoretically
  - 2-8× speedup over original Mamba
  - Competitive with Transformers
- **PDF:** `mamba2_ssd.pdf`

### Mamba for Vision
- **Authors:** Lianghui Zhu et al.
- **arXiv:** [2312.10997](https://arxiv.org/abs/2312.10997)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Applies Mamba architecture to vision tasks.
- **PDF:** `mamba_vision.pdf`

### Titans: Learning to Memorize at Test Time
- **Authors:** Google Research
- **arXiv:** [2309.05516](https://arxiv.org/abs/2309.05516)
- **Date:** September 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Novel memory architecture for transformers with test-time memorization.
- **PDF:** `titans_memory.pdf`

---

## Linear Attention / Efficient Transformers

### RWKV: Reinventing RNNs for the Transformer Era
- **Authors:** Bo Peng, Eric Alcaide, et al.
- **arXiv:** [2305.13048](https://arxiv.org/abs/2305.13048)
- **Date:** May 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Combines efficient parallelizable training of Transformers with efficient inference of RNNs. Uses linear attention mechanism, scaled to 14B parameters.
- **Key Features:**
  - Receptance Weighted Key Value mechanism
  - Can be formulated as Transformer OR RNN
  - Constant memory/compute during inference
  - 14B parameter model trained successfully
- **PDF:** `rwkv.pdf`

### RetNet: Retentive Network for Large Language Models
- **Authors:** Yutao Sun, Li Dong, et al. (Microsoft Research)
- **arXiv:** [2307.08621](https://arxiv.org/abs/2307.08621)
- **Date:** July 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Foundation architecture achieving training parallelism, low-cost O(1) inference, and good performance. Supports parallel, recurrent, and chunkwise paradigms.
- **Key Features:**
  - Retention mechanism for sequence modeling
  - Three computation paradigms (parallel, recurrent, chunkwise)
  - O(1) inference cost
  - Efficient long-sequence modeling with linear complexity
- **PDF:** `retnet.pdf`

### FlashAttention: Fast and Memory-Efficient Exact Attention
- **Authors:** Tri Dao, Daniel Y. Fu, et al.
- **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)
- **Date:** May 2022
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** IO-aware exact attention algorithm using tiling to reduce memory reads/writes.
- **PDF:** `flash_attention.pdf`

### FlashAttention-2: Faster Attention with Better Parallelism
- **Authors:** Tri Dao
- **arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)
- **Date:** September 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Improved FlashAttention with 2× speedup through better work partitioning.
- **PDF:** `flashattention2.pdf`

### FlashAttention-3: Fast and Accurate Attention with Asynchrony
- **Authors:** Tri Dao et al.
- **arXiv:** [2407.21783](https://arxiv.org/abs/2407.21783)
- **Date:** July 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Latest FlashAttention with asynchronous computation.
- **PDF:** `flashattention3.pdf`

### Ring Attention with Blockwise Transformers
- **Authors:** UC Berkeley
- **arXiv:** [2310.01801](https://arxiv.org/abs/2310.01801)
- **Date:** October 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Distributed attention for near-infinite context lengths.
- **PDF:** `ring_attention.pdf`

### GQA: Training Generalized Multi-Query Transformer Models
- **Authors:** Google Research
- **arXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)
- **Date:** May 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Grouped-Query Attention for efficient inference.
- **PDF:** `gqa.pdf`

### Griffin: Mixing Gated Linear Recurrences with Local Attention
- **Authors:** Google DeepMind
- **arXiv:** [2405.00732](https://arxiv.org/abs/2405.00732)
- **Date:** May 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Hybrid architecture combining linear recurrences with local attention.
- **PDF:** `griffin.pdf`

### Jamba: A Hybrid Transformer-Mamba Language Model
- **Authors:** AI21 Labs
- **arXiv:** [2312.06635](https://arxiv.org/abs/2312.06635)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Hybrid Mamba-Transformer architecture for efficient language modeling.
- **PDF:** `jamba.pdf`

### Samba: Simple Hybrid State Space Models
- **Authors:** Microsoft Research
- **arXiv:** [2402.01771](https://arxiv.org/abs/2402.01771)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Simple and effective hybrid SSM architecture.
- **PDF:** `samba.pdf`

---

## Diffusion Models

### DDPM: Denoising Diffusion Probabilistic Models
- **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **arXiv:** [2006.11239](https://arxiv.org/abs/2006.11239)
- **Date:** June 2020
- **Pentary Compatibility:** ⭐⭐⭐ (Moderate)
- **Summary:** High quality image synthesis using diffusion probabilistic models.
- **PDF:** `ddpm.pdf`

### Latent Diffusion Models (Stable Diffusion)
- **Authors:** Robin Rombach, Andreas Blattmann, et al.
- **arXiv:** [2112.10752](https://arxiv.org/abs/2112.10752)
- **Date:** December 2021
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Good)
- **Summary:** Applies diffusion in latent space for efficient high-resolution synthesis.
- **PDF:** `latent_diffusion.pdf`

---

## Foundation Models

### LLaMA: Open and Efficient Foundation Language Models
- **Authors:** Hugo Touvron, et al. (Meta AI)
- **arXiv:** [2302.13971](https://arxiv.org/abs/2302.13971)
- **Date:** February 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient foundation models (7B-65B), LLaMA-13B outperforms GPT-3.
- **PDF:** `llama.pdf`

### Llama 2: Open Foundation and Fine-Tuned Chat Models
- **Authors:** Meta AI
- **arXiv:** [2307.09288](https://arxiv.org/abs/2307.09288)
- **Date:** July 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Improved Llama with RLHF and extended context.
- **PDF:** `llama2.pdf`

### Llama 3
- **Authors:** Meta AI
- **arXiv:** [2404.14047](https://arxiv.org/abs/2404.14047)
- **Date:** April 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Latest Llama iteration with improved architecture.
- **PDF:** `llama3.pdf`

### Mistral 7B
- **Authors:** Mistral AI
- **arXiv:** [2310.06825](https://arxiv.org/abs/2310.06825)
- **Date:** October 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient 7B model with sliding window attention.
- **PDF:** `mistral7b.pdf`

### Mixtral of Experts
- **Authors:** Mistral AI
- **arXiv:** [2401.04088](https://arxiv.org/abs/2401.04088)
- **Date:** January 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Sparse MoE architecture for efficient scaling.
- **PDF:** `mixtral_moe.pdf`

### DeepSeek LLM
- **Authors:** DeepSeek AI
- **arXiv:** [2401.02954](https://arxiv.org/abs/2401.02954)
- **Date:** January 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient open-source foundation model.
- **PDF:** `deepseek_llm.pdf`

### DeepSeek-V2: A Strong, Economical, and Efficient MoE Model
- **Authors:** DeepSeek AI
- **arXiv:** [2405.04434](https://arxiv.org/abs/2405.04434)
- **Date:** May 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** MLA (Multi-head Latent Attention) for extreme efficiency.
- **PDF:** `deepseekv2.pdf`

### DeepSeekMoE
- **Authors:** DeepSeek AI
- **arXiv:** [2311.01906](https://arxiv.org/abs/2311.01906)
- **Date:** November 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient MoE architecture design.
- **PDF:** `deepseek_moe.pdf`

### Gemma: Open Models Based on Gemini Research
- **Authors:** Google DeepMind
- **arXiv:** [2402.05608](https://arxiv.org/abs/2402.05608)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient open-source models from Google.
- **PDF:** `gemma.pdf`

### Gemma 2
- **Authors:** Google DeepMind
- **arXiv:** [2405.19888](https://arxiv.org/abs/2405.19888)
- **Date:** May 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Improved Gemma architecture.
- **PDF:** `gemma2.pdf`

### Phi-2: The Surprising Power of Small Language Models
- **Authors:** Microsoft Research
- **arXiv:** [2312.11805](https://arxiv.org/abs/2312.11805)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Highly efficient small language model.
- **PDF:** `phi2.pdf`

### Phi-3 Technical Report
- **Authors:** Microsoft Research
- **arXiv:** [2404.14619](https://arxiv.org/abs/2404.14619)
- **Date:** April 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Latest Phi model with excellent efficiency.
- **PDF:** `phi3.pdf`

### OLMo: Accelerating the Science of Language Models
- **Authors:** AI2
- **arXiv:** [2311.10770](https://arxiv.org/abs/2311.10770)
- **Date:** November 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Fully open-source language model.
- **PDF:** `olmo.pdf`

### Qwen Technical Report
- **Authors:** Alibaba Cloud
- **arXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)
- **Date:** September 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Efficient multilingual model.
- **PDF:** `qwen.pdf`

### Qwen2 Technical Report
- **Authors:** Alibaba Cloud
- **arXiv:** [2407.10671](https://arxiv.org/abs/2407.10671)
- **Date:** July 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Improved Qwen architecture.
- **PDF:** `qwen2.pdf`

### Orca: Progressive Learning from Complex Explanation Traces
- **Authors:** Microsoft Research
- **arXiv:** [2306.09782](https://arxiv.org/abs/2306.09782)
- **Date:** June 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Learning from reasoning traces.
- **PDF:** `orca.pdf`

---

## Quantization Papers

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **Authors:** Elias Frantar et al.
- **arXiv:** [2210.17323](https://arxiv.org/abs/2210.17323)
- **Date:** October 2022
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** One-shot weight quantization to 3-4 bits with minimal accuracy loss.
- **PDF:** `gptq.pdf`

### AWQ: Activation-aware Weight Quantization for LLM Compression
- **Authors:** Ji Lin et al. (MIT)
- **arXiv:** [2306.00978](https://arxiv.org/abs/2306.00978)
- **Date:** June 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Activation-aware quantization preserving salient weights.
- **PDF:** `awq.pdf`

### LSQ: Learned Step Size Quantization
- **Authors:** Steven K. Esser et al. (IBM)
- **arXiv:** [1902.08153](https://arxiv.org/abs/1902.08153)
- **Date:** February 2019
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Learning optimal quantization step sizes during training.
- **PDF:** `lsq.pdf`

### QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Tim Dettmers et al.
- **arXiv:** [2306.07629](https://arxiv.org/abs/2306.07629)
- **Date:** June 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** 4-bit quantization with LoRA for efficient finetuning.
- **PDF:** `bitsandbytes_qlora.pdf`

### SqueezeLLM: Dense-and-Sparse Quantization
- **Authors:** Sehoon Kim et al.
- **arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)
- **Date:** May 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Non-uniform quantization with sparse weights.
- **PDF:** `squeezellm.pdf`

### SmoothQuant: Accurate and Efficient Post-Training Quantization
- **Authors:** Guangxuan Xiao et al. (MIT)
- **arXiv:** [2211.10438](https://arxiv.org/abs/2211.10438)
- **Date:** November 2022
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Migrates quantization difficulty from activations to weights.
- **PDF:** `smoothquant.pdf`

### LLM.int8(): 8-bit Matrix Multiplication
- **Authors:** Tim Dettmers et al.
- **arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339)
- **Date:** August 2022
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** First practical 8-bit inference for large LLMs.
- **PDF:** `llm_int8_original.pdf`

### HQQ: Half-Quadratic Quantization
- **Authors:** Mobius Labs
- **arXiv:** [2402.19427](https://arxiv.org/abs/2402.19427)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Fast quantization without calibration data.
- **PDF:** `hqq_quantization.pdf`

### Marlin: Mixed-Precision Auto-Regressive Parallel Inference
- **Authors:** IST Austria
- **arXiv:** [2311.16919](https://arxiv.org/abs/2311.16919)
- **Date:** November 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** High-performance 4-bit inference kernels.
- **PDF:** `marlin_kernel.pdf`

---

## Speculative Decoding

### Accelerating LLM Inference with Staged Speculative Decoding
- **Authors:** Various
- **arXiv:** [2305.17888](https://arxiv.org/abs/2305.17888)
- **Date:** May 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Using draft models for faster inference.
- **PDF:** `speculative_decoding.pdf`

### EAGLE: Speculative Sampling Requires Rethinking
- **Authors:** Yichao Lu et al.
- **arXiv:** [2312.17238](https://arxiv.org/abs/2312.17238)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Feature-level speculation for 2-3× speedup.
- **PDF:** `eagle_speculative.pdf`

### Medusa: Simple LLM Inference Acceleration Framework
- **Authors:** Together AI
- **arXiv:** [2309.12307](https://arxiv.org/abs/2309.12307)
- **Date:** September 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Multiple decoding heads for parallel speculation.
- **PDF:** `medusa.pdf`

### Lookahead Decoding
- **Authors:** UC Berkeley
- **arXiv:** [2401.14489](https://arxiv.org/abs/2401.14489)
- **Date:** January 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Parallel n-gram generation for fast decoding.
- **PDF:** `lookahead_decoding.pdf`

---

## KV Cache & Memory Optimization

### PagedAttention for Efficient LLM Serving (vLLM)
- **Authors:** UC Berkeley
- **arXiv:** [2304.11062](https://arxiv.org/abs/2304.11062)
- **Date:** April 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Virtual memory for KV cache management.
- **PDF:** `pageattention_vllm.pdf`

### Scissorhands: Exploiting KV Cache Compression
- **Authors:** Various
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)
- **Date:** September 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Selective KV cache eviction strategies.
- **PDF:** `kv_cache_compression.pdf`

---

## Positional Encodings

### RoPE: Rotary Position Embedding
- **Authors:** Jianlin Su et al.
- **arXiv:** [2104.09864](https://arxiv.org/abs/2104.09864)
- **Date:** April 2021
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Relative positional encoding via rotation.
- **PDF:** `rope.pdf`

### PoPE: Polar Coordinate Position Embeddings
- **Authors:** Anand Gopalakrishnan et al.
- **arXiv:** [2509.10534](https://arxiv.org/abs/2509.10534)
- **Date:** September 2025
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Decouples "what" and "where" in attention using polar coordinates.
- **PDF:** `pope.pdf`

### YaRN: Efficient Context Window Extension
- **Authors:** NousResearch
- **arXiv:** [2402.00518](https://arxiv.org/abs/2402.00518)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Improved RoPE scaling for long contexts.
- **PDF:** `yarn.pdf`

### LongLLaMA: Focused Transformer for Long Context
- **Authors:** Various
- **arXiv:** [2310.09709](https://arxiv.org/abs/2310.09709)
- **Date:** October 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Long-context extension techniques.
- **PDF:** `longllama.pdf`

---

## Low-Bit Neural Networks

### Binarized Neural Networks
- **Authors:** Matthieu Courbariaux et al.
- **arXiv:** [1602.02830](https://arxiv.org/abs/1602.02830)
- **Date:** February 2016
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Training neural networks with binary weights and activations.
- **PDF:** `binarized_nn.pdf`

### Ternary Weight Networks
- **Authors:** Fengfu Li et al.
- **arXiv:** [1605.04711](https://arxiv.org/abs/1605.04711)
- **Date:** May 2016
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent - Direct Comparison)
- **Summary:** Networks with {-1, 0, +1} weights.
- **PDF:** `ternary_weight_networks.pdf`

### BitNet: Scaling 1-bit Transformers for LLMs
- **Authors:** Microsoft Research
- **arXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent - Direct Comparison)
- **Summary:** Training LLMs with 1-bit weights.
- **PDF:** `bitnet.pdf`

### BitNet b1.58: The Era of 1-bit LLMs
- **Authors:** Microsoft Research
- **arXiv:** [2310.11453](https://arxiv.org/abs/2310.11453)
- **Date:** October 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent - Direct Comparison)
- **Summary:** Ternary {-1, 0, +1} weights achieving near full-precision accuracy.
- **PDF:** `bitnet_1_58.pdf`

### SparseGPT: Massive Language Models via Sparse Regression
- **Authors:** Elias Frantar, Dan Alistarh
- **arXiv:** [2301.00774](https://arxiv.org/abs/2301.00774)
- **Date:** January 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Pruning GPT models to 50% sparsity.
- **PDF:** `sparse_llm.pdf`

### Knowledge Distillation
- **Authors:** Geoffrey Hinton et al.
- **arXiv:** [1503.02531](https://arxiv.org/abs/1503.02531)
- **Date:** March 2015
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Foundational paper on model compression.
- **PDF:** `knowledge_distillation.pdf`

---

## Mixture of Experts (MoE)

### MoE Survey: A Survey on Mixture of Experts
- **Authors:** Various
- **arXiv:** [2401.10166](https://arxiv.org/abs/2401.10166)
- **Date:** January 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Comprehensive survey of MoE techniques.
- **PDF:** `moe_survey.pdf`

---

## Hardware & Accelerators

### In-Datacenter Performance Analysis of a TPU
- **Authors:** Norman P. Jouppi et al. (Google)
- **arXiv:** [1704.04760](https://arxiv.org/abs/1704.04760)
- **Date:** April 2017
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent - Benchmark)
- **Summary:** Google TPU v1 architecture and performance analysis.
- **PDF:** `tpu_v1.pdf`

### Groq: High-Performance LLM Inference
- **Authors:** Groq Inc.
- **arXiv:** [2402.04245](https://arxiv.org/abs/2402.04245)
- **Date:** February 2024
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good - Benchmark)
- **Summary:** Deterministic LLM inference architecture.
- **PDF:** `groq_inference.pdf`

### LLM Efficiency Survey
- **Authors:** Various
- **arXiv:** [2312.00863](https://arxiv.org/abs/2312.00863)
- **Date:** December 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Comprehensive survey of LLM efficiency techniques.
- **PDF:** `research/llm_efficiency_survey.pdf`

---

## In-Memory Computing & Neuromorphic

### In-Memory Computing Review
- **Authors:** Various
- **arXiv:** [1807.10221](https://arxiv.org/abs/1807.10221)
- **Date:** July 2018
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Comprehensive review of in-memory computing approaches.
- **PDF:** `research/in_memory_computing_review.pdf`

### Analog AI Hardware
- **Authors:** Various
- **arXiv:** [2102.11382](https://arxiv.org/abs/2102.11382)
- **Date:** February 2021
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Overview of analog computing for AI workloads.
- **PDF:** `research/analog_ai_hardware.pdf`

### Neuromorphic Computing Review
- **Authors:** Various
- **arXiv:** [2001.04451](https://arxiv.org/abs/2001.04451)
- **Date:** January 2020
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Survey of neuromorphic computing approaches.
- **PDF:** `neuromorphic_review.pdf`

### Memristor Crossbar for Neural Networks
- **Authors:** Various
- **arXiv:** [2105.10064](https://arxiv.org/abs/2105.10064)
- **Date:** May 2021
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Memristor crossbar arrays for neural network acceleration.
- **PDF:** `research/memristor_crossbar_nn.pdf`

---

## Ternary/Multi-Valued Logic

### Setun Ternary Computer
- **Source:** HAL Archives
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent - Foundational)
- **Summary:** Historical overview of Soviet ternary computer.
- **PDF:** `trinary-systems/Setun_Ternary_Computer_HAL.pdf`

### Ternary Computing for Cybersecurity
- **Source:** Northern Arizona University
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Modern applications of ternary computing.
- **PDF:** `trinary-systems/Ternary_Computing_Cybersecurity.pdf`

### Ternary CMOS Standard Cell Design
- **Source:** NAUN
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Standard cell library for ternary CMOS.
- **PDF:** `research/Ternary_CMOS_Standard_Cell_Design.pdf`

### Memristor-CMOS Ternary Logic
- **Source:** arXiv
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Hybrid memristor-CMOS ternary implementation.
- **PDF:** `research/Memristor_CMOS_Ternary_Logic.pdf`

### Efficient Ternary Logic Circuits
- **Source:** University of Rochester
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Optimization techniques for ternary arithmetic.
- **PDF:** `research/Efficient_Ternary_Logic_Circuits.pdf`

### Ternary Logic Integrated Circuits
- **Source:** HAL Science
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Complete IC design methodology for ternary.
- **PDF:** `research/Ternary_Logic_Integrated_Circuits.pdf`

---

## Compatibility Summary

### Best for Pentary Implementation

| Rank | Paper | Category | Key Benefit |
|------|-------|----------|-------------|
| 1 | Mamba/Mamba-2 | SSM | Linear complexity, natural state updates |
| 2 | RetNet | Retention | O(1) inference, three paradigms |
| 3 | RWKV | Linear Attention | RNN inference, Transformer training |
| 4 | BitNet b1.58 | Low-Bit | Direct comparison: ternary vs pentary |
| 5 | AWQ/GPTQ | Quantization | Proven low-bit quantization |
| 6 | DeepSeek-V2 | MoE | MLA architecture highly efficient |
| 7 | vLLM PagedAttention | Memory | KV cache management |
| 8 | TPU Architecture | Hardware | Systolic array benchmark |

### By Research Area

| Area | Best Papers | Pentary Relevance |
|------|-------------|-------------------|
| Architecture | Mamba-2, Griffin, Jamba | Linear/hybrid models fit pentary |
| Quantization | AWQ, GPTQ, SmoothQuant | Foundation for pentary weights |
| Low-Bit | BitNet, Ternary Networks | Direct comparison point |
| Hardware | TPU, Groq | Benchmark targets |
| Memory | vLLM, KV Compression | Pentary memory efficiency |
| Neuromorphic | Memristor, Analog AI | Implementation pathway |

---

## Download Commands

```bash
# Quantization Papers
wget https://arxiv.org/pdf/2210.17323 -O gptq.pdf
wget https://arxiv.org/pdf/2306.00978 -O awq.pdf
wget https://arxiv.org/pdf/1902.08153 -O lsq.pdf
wget https://arxiv.org/pdf/2306.07629 -O bitsandbytes_qlora.pdf
wget https://arxiv.org/pdf/2211.10438 -O smoothquant.pdf

# SSM/Linear Attention
wget https://arxiv.org/pdf/2312.00752 -O mamba.pdf
wget https://arxiv.org/pdf/2405.21060 -O mamba2_ssd.pdf
wget https://arxiv.org/pdf/2305.13048 -O rwkv.pdf
wget https://arxiv.org/pdf/2307.08621 -O retnet.pdf

# Foundation Models
wget https://arxiv.org/pdf/2302.13971 -O llama.pdf
wget https://arxiv.org/pdf/2307.09288 -O llama2.pdf
wget https://arxiv.org/pdf/2404.14047 -O llama3.pdf
wget https://arxiv.org/pdf/2310.06825 -O mistral7b.pdf
wget https://arxiv.org/pdf/2401.04088 -O mixtral_moe.pdf
wget https://arxiv.org/pdf/2405.04434 -O deepseekv2.pdf

# Low-Bit Networks
wget https://arxiv.org/pdf/1602.02830 -O binarized_nn.pdf
wget https://arxiv.org/pdf/1605.04711 -O ternary_weight_networks.pdf
wget https://arxiv.org/pdf/2402.17764 -O bitnet.pdf
wget https://arxiv.org/pdf/2310.11453 -O bitnet_1_58.pdf

# Hardware
wget https://arxiv.org/pdf/1704.04760 -O tpu_v1.pdf
```

---

**Archive Last Updated:** December 2024  
**Total Papers:** 79  
**Total Size:** ~309MB
