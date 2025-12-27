# Pentary Paper Archive

This archive contains references to key papers analyzed for compatibility with Pentary computing architecture.

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
- **PDF:** https://arxiv.org/pdf/2312.00752

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
- **PDF:** https://arxiv.org/pdf/2405.21060

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
- **PDF:** https://arxiv.org/pdf/2305.13048

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
- **PDF:** https://arxiv.org/pdf/2307.08621

### FlashAttention: Fast and Memory-Efficient Exact Attention
- **Authors:** Tri Dao, Daniel Y. Fu, et al.
- **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)
- **Date:** May 2022
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** IO-aware exact attention algorithm using tiling to reduce memory reads/writes. Enables longer context and faster training.
- **Key Features:**
  - IO-aware algorithm design
  - Tiling for memory efficiency
  - 3× speedup on GPT-2
  - Enables 64K sequence length
- **PDF:** https://arxiv.org/pdf/2205.14135

---

## Diffusion Models

### DDPM: Denoising Diffusion Probabilistic Models
- **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **arXiv:** [2006.11239](https://arxiv.org/abs/2006.11239)
- **Date:** June 2020
- **Pentary Compatibility:** ⭐⭐⭐ (Moderate)
- **Summary:** High quality image synthesis using diffusion probabilistic models. Achieves state-of-the-art FID score of 3.17 on CIFAR10.
- **Key Features:**
  - Denoising score matching with Langevin dynamics
  - Progressive lossy decompression
  - State-of-the-art image quality
  - Generalizes autoregressive decoding
- **PDF:** https://arxiv.org/pdf/2006.11239

### Latent Diffusion Models (Stable Diffusion)
- **Authors:** Robin Rombach, Andreas Blattmann, et al.
- **arXiv:** [2112.10752](https://arxiv.org/abs/2112.10752)
- **Date:** December 2021
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Good)
- **Summary:** Applies diffusion in latent space of pretrained autoencoders, enabling high-resolution synthesis with reduced computational requirements.
- **Key Features:**
  - Latent space diffusion (not pixel space)
  - Cross-attention for conditioning
  - Significantly reduced compute requirements
  - Text-to-image, inpainting, super-resolution
- **PDF:** https://arxiv.org/pdf/2112.10752

---

## Foundation Models

### LLaMA: Open and Efficient Foundation Language Models
- **Authors:** Hugo Touvron, Thibaut Lavril, et al. (Meta AI)
- **arXiv:** [2302.13971](https://arxiv.org/abs/2302.13971)
- **Date:** February 2023
- **Pentary Compatibility:** ⭐⭐⭐⭐ (Very Good)
- **Summary:** Collection of foundation language models (7B-65B parameters) trained on public datasets. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks.
- **Key Features:**
  - 7B to 65B parameter models
  - Trained on public datasets only
  - LLaMA-13B > GPT-3 (175B)
  - Open source weights
- **PDF:** https://arxiv.org/pdf/2302.13971

---

## Positional Encodings

### PoPE: Polar Coordinate Position Embeddings
- **Authors:** Anand Gopalakrishnan, Robert Csordás, Jürgen Schmidhuber, Michael C. Mozer
- **arXiv:** [2509.10534](https://arxiv.org/abs/2509.10534)
- **Date:** September 2025
- **Pentary Compatibility:** ⭐⭐⭐⭐⭐ (Excellent)
- **Summary:** Decouples "what" and "where" in attention by using polar coordinates. Position in angle, content in magnitude.
- **Key Features:**
  - Eliminates what-where entanglement in RoPE
  - Strong zero-shot length extrapolation
  - Works across NLP, music, genomics
  - Better perplexity than RoPE and YaRN
- **PDF:** https://arxiv.org/pdf/2509.10534

---

## Compatibility Summary

| Paper | Architecture | Pentary Score | Key Benefit |
|-------|--------------|---------------|-------------|
| Mamba | SSM | ⭐⭐⭐⭐⭐ | Linear complexity, perfect for pentary state updates |
| Mamba-2 | SSM | ⭐⭐⭐⭐⭐ | 2-8× faster, theoretical SSM-attention connection |
| RWKV | Linear Attention | ⭐⭐⭐⭐⭐ | RNN inference, Transformer training |
| RetNet | Retention | ⭐⭐⭐⭐⭐ | O(1) inference, linear complexity |
| FlashAttention | Attention | ⭐⭐⭐⭐ | IO-aware tiling for Pentary caches |
| DDPM | Diffusion | ⭐⭐⭐ | Requires many timesteps (challenging) |
| Latent Diffusion | Diffusion | ⭐⭐⭐⭐ | Latent space compression aligns with pentary |
| LLaMA | Transformer | ⭐⭐⭐⭐ | Standard architecture, proven quantization |
| PoPE | Positional | ⭐⭐⭐⭐⭐ | Angular encoding perfect for pentary |

---

## Download Commands

```bash
# Mamba
wget https://arxiv.org/pdf/2312.00752 -O mamba.pdf

# Mamba-2
wget https://arxiv.org/pdf/2405.21060 -O mamba2.pdf

# RWKV
wget https://arxiv.org/pdf/2305.13048 -O rwkv.pdf

# RetNet
wget https://arxiv.org/pdf/2307.08621 -O retnet.pdf

# FlashAttention
wget https://arxiv.org/pdf/2205.14135 -O flash_attention.pdf

# DDPM
wget https://arxiv.org/pdf/2006.11239 -O ddpm.pdf

# Latent Diffusion
wget https://arxiv.org/pdf/2112.10752 -O latent_diffusion.pdf

# LLaMA
wget https://arxiv.org/pdf/2302.13971 -O llama.pdf

# PoPE
wget https://arxiv.org/pdf/2509.10534 -O pope.pdf
```

---

**Archive Last Updated:** December 2024
