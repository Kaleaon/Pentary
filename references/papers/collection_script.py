#!/usr/bin/env python3
"""
Paper Collection Script for Pentary Research
Collects papers on neuromorphic computing and infinite context AI
"""

import json
from datetime import datetime

# Paper collection data structure
papers_collection = {
    "metadata": {
        "collection_date": datetime.now().isoformat(),
        "total_papers": 0,
        "categories": {
            "neuromorphic_computing": 0,
            "infinite_context": 0,
            "hybrid_systems": 0
        }
    },
    "papers": []
}

# Neuromorphic Computing Papers
neuromorphic_papers = [
    {
        "id": 1,
        "title": "Biologically-Inspired Technologies: Integrating Brain-Computer Interfaces",
        "arxiv_id": "2410.23639",
        "url": "https://arxiv.org/abs/2410.23639",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "brain_computer_interface",
        "relevance": "high"
    },
    {
        "id": 2,
        "title": "Quantum Computing and Neuromorphic Computing",
        "arxiv_id": "2408.03884",
        "url": "https://arxiv.org/abs/2408.03884",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "quantum_neuromorphic",
        "relevance": "high"
    },
    {
        "id": 3,
        "title": "Neuromorphic Computing for Embodied Intelligence in Autonomous Systems",
        "arxiv_id": "2507.18139",
        "url": "https://arxiv.org/abs/2507.18139",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "embodied_ai",
        "relevance": "high"
    },
    {
        "id": 4,
        "title": "Continual Learning with Neuromorphic Computing",
        "arxiv_id": "2410.09218",
        "url": "https://arxiv.org/html/2410.09218v3",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "continual_learning",
        "relevance": "high"
    },
    {
        "id": 5,
        "title": "Neuromorphic Computing with Large Scale Spiking Neural Networks",
        "url": "https://www.preprints.org/manuscript/202503.1505",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "high"
    },
    {
        "id": 6,
        "title": "Neuromorphic Computing: A Theoretical Framework for Time, Space, and Energy Scaling",
        "arxiv_id": "2507.17886",
        "url": "https://arxiv.org/abs/2507.17886",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "theoretical_framework",
        "relevance": "high"
    },
    {
        "id": 7,
        "title": "Neuromorphic Computing - An Overview",
        "arxiv_id": "2510.06721",
        "url": "https://arxiv.org/abs/2510.06721",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "overview",
        "relevance": "high"
    },
    {
        "id": 8,
        "title": "Bridging Brains and Machines: A Unified Frontier in Neuroscience",
        "arxiv_id": "2507.10722",
        "url": "https://arxiv.org/html/2507.10722v1",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "brain_machine_interface",
        "relevance": "high"
    },
    {
        "id": 9,
        "title": "Neuromorphic Wireless Split Computing with Multi-Level Spikes",
        "url": "https://ui.adsabs.harvard.edu/abs/2024arXiv241104728W/abstract",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "wireless_computing",
        "relevance": "medium"
    },
    {
        "id": 10,
        "title": "Neuromorphic computing for robotic vision: algorithms to hardware",
        "url": "https://www.nature.com/articles/s44172-025-00492-5",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "robotic_vision",
        "relevance": "high"
    }
]

# Spiking Neural Networks Papers
snn_papers = [
    {
        "id": 11,
        "title": "Spiking Neural Networks: The Future of Brain-Inspired Computing",
        "arxiv_id": "2510.27379",
        "url": "https://arxiv.org/abs/2510.27379",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "high"
    },
    {
        "id": 12,
        "title": "Benchmarking Spiking Neural Network Learning Methods",
        "arxiv_id": "2402.01782",
        "url": "https://arxiv.org/abs/2402.01782",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "high"
    },
    {
        "id": 13,
        "title": "Exploring Spiking Neural Networks for Binary Classification",
        "arxiv_id": "2510.20997",
        "url": "https://arxiv.org/abs/2510.20997",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "medium"
    },
    {
        "id": 14,
        "title": "Spiking Neural Network as Adaptive Event Stream Slicer",
        "arxiv_id": "2410.02249",
        "url": "https://arxiv.org/abs/2410.02249",
        "year": 2024,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "medium"
    },
    {
        "id": 15,
        "title": "Spiking Neural Network Architecture Search: A Survey",
        "arxiv_id": "2510.14235",
        "url": "https://arxiv.org/html/2510.14235v1",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "neural_architecture_search",
        "relevance": "high"
    },
    {
        "id": 16,
        "title": "Spiking Neural Networks with Random Network Architecture",
        "arxiv_id": "2505.13622",
        "url": "https://arxiv.org/html/2505.13622v1",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "spiking_neural_networks",
        "relevance": "medium"
    },
    {
        "id": 17,
        "title": "Spiking Neural Networks: a theoretical framework for Universal Computation",
        "arxiv_id": "2509.21920",
        "url": "https://arxiv.org/abs/2509.21920",
        "year": 2025,
        "category": "neuromorphic_computing",
        "subcategory": "theoretical_framework",
        "relevance": "high"
    }
]

# Infinite Context Papers
infinite_context_papers = [
    {
        "id": 18,
        "title": "Infinite Retrieval: Attention Enhanced LLMs in Long-Context",
        "arxiv_id": "2502.12962",
        "url": "https://arxiv.org/html/2502.12962v1",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "long_context_transformers",
        "relevance": "high"
    },
    {
        "id": 19,
        "title": "Efficient Infinite Context Transformers with Infini-attention",
        "arxiv_id": "2404.07143",
        "url": "https://arxiv.org/abs/2404.07143",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "attention_mechanisms",
        "relevance": "high"
    },
    {
        "id": 20,
        "title": "A Memory-Efficient Infinite-Context Transformer for Edge Devices",
        "arxiv_id": "2503.22196",
        "url": "https://arxiv.org/html/2503.22196v1",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "edge_computing",
        "relevance": "high"
    },
    {
        "id": 21,
        "title": "ReAttention: Training-Free Infinite Context with Finite Attention Scope",
        "arxiv_id": "2407.15176",
        "url": "https://arxiv.org/html/2407.15176v3",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "attention_mechanisms",
        "relevance": "high"
    },
    {
        "id": 22,
        "title": "Human-Inspired Episodic Memory for Infinite Context",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2025/file/c05144b635df16ac9bbf8246bbbd55ca-Paper-Conference.pdf",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "memory_systems",
        "relevance": "high"
    },
    {
        "id": 23,
        "title": "Token Memory Transformer with Infinite Context",
        "url": "https://www.researchgate.net/publication/393965346_Token_Memory_Transformer_with_Infinite_Context",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "high"
    }
]

# Memory-Augmented Networks Papers
memory_augmented_papers = [
    {
        "id": 24,
        "title": "Survey on Memory-Augmented Neural Networks: Cognitive Insights to AI",
        "arxiv_id": "2312.06141",
        "url": "https://arxiv.org/abs/2312.06141",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "high"
    },
    {
        "id": 25,
        "title": "Memory-Augmented Transformers: A Systematic Review",
        "arxiv_id": "2508.10824",
        "url": "https://arxiv.org/html/2508.10824v1",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "high"
    },
    {
        "id": 26,
        "title": "Hebbian Memory-Augmented Recurrent Networks: Engram Neurons",
        "arxiv_id": "2507.21474",
        "url": "https://arxiv.org/abs/2507.21474",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "high"
    },
    {
        "id": 27,
        "title": "Benchmarking and Enhancing Long-Term Memory in LLMs",
        "arxiv_id": "2510.27246",
        "url": "https://www.arxiv.org/pdf/2510.27246",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "memory_systems",
        "relevance": "high"
    },
    {
        "id": 28,
        "title": "Heterogenous Memory Augmented Neural Networks",
        "arxiv_id": "2310.10909",
        "url": "https://arxiv.org/abs/2310.10909",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "medium"
    },
    {
        "id": 29,
        "title": "Exploring Learnability in Memory-Augmented Recurrent Neural Networks",
        "arxiv_id": "2410.03154",
        "url": "https://arxiv.org/abs/2410.03154",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "memory_augmented",
        "relevance": "medium"
    }
]

# State Space Models Papers
ssm_papers = [
    {
        "id": 30,
        "title": "Bi-Mamba: Towards Accurate 1-Bit State Space Models",
        "arxiv_id": "2411.11843",
        "url": "https://arxiv.org/abs/2411.11843",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    },
    {
        "id": 31,
        "title": "Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models",
        "arxiv_id": "2501.16295",
        "url": "https://arxiv.org/abs/2501.16295",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    },
    {
        "id": 32,
        "title": "A Comprehensive Survey on Structured State Space Models",
        "arxiv_id": "2503.18970",
        "url": "https://arxiv.org/abs/2503.18970",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    },
    {
        "id": 33,
        "title": "Computational Limits of State-Space Models & Mamba",
        "arxiv_id": "2412.06148",
        "url": "https://arxiv.org/abs/2412.06148",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    },
    {
        "id": 34,
        "title": "Mamba-360: Survey of State Space Models",
        "arxiv_id": "2404.16112",
        "url": "https://arxiv.org/abs/2404.16112",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    },
    {
        "id": 35,
        "title": "Spatial-Mamba: Effective Visual State Space Models",
        "arxiv_id": "2410.15091",
        "url": "https://arxiv.org/abs/2410.15091",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "medium"
    },
    {
        "id": 36,
        "title": "GroupMamba: Efficient Group-Based Visual State Space Model",
        "arxiv_id": "2407.13772",
        "url": "https://arxiv.org/abs/2407.13772",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "medium"
    },
    {
        "id": 37,
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "arxiv_id": "2312.00752",
        "url": "https://arxiv.org/abs/2312.00752",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "state_space_models",
        "relevance": "high"
    }
]

# RAG Papers
rag_papers = [
    {
        "id": 38,
        "title": "Retrieval-Augmented Generation with Graphs (GraphRAG)",
        "arxiv_id": "2501.00309",
        "url": "https://arxiv.org/abs/2501.00309",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 39,
        "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
        "arxiv_id": "2410.05779",
        "url": "https://arxiv.org/abs/2410.05779",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 40,
        "title": "Sufficient Context: A New Lens on Retrieval Augmented Generation",
        "arxiv_id": "2411.06037",
        "url": "https://arxiv.org/abs/2411.06037",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 41,
        "title": "Retrieval Augmented Generation (RAG) and Beyond",
        "arxiv_id": "2409.14924",
        "url": "https://arxiv.org/abs/2409.14924",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 42,
        "title": "A Comprehensive Survey of Retrieval-Augmented Generation (RAG)",
        "arxiv_id": "2410.12837",
        "url": "https://arxiv.org/abs/2410.12837",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 43,
        "title": "Retrieval-Augmented Generation: A Comprehensive Survey",
        "arxiv_id": "2506.00054",
        "url": "https://arxiv.org/abs/2506.00054",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 44,
        "title": "Towards Agentic RAG with Deep Reasoning",
        "arxiv_id": "2507.09477",
        "url": "https://arxiv.org/abs/2507.09477",
        "year": 2025,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    },
    {
        "id": 45,
        "title": "Retrieval Augmented Generation or Long-Context LLMs?",
        "arxiv_id": "2407.16833",
        "url": "https://arxiv.org/abs/2407.16833",
        "year": 2024,
        "category": "infinite_context",
        "subcategory": "retrieval_augmented",
        "relevance": "high"
    }
]

# Combine all papers
all_papers = (neuromorphic_papers + snn_papers + infinite_context_papers + 
              memory_augmented_papers + ssm_papers + rag_papers)

papers_collection["papers"] = all_papers
papers_collection["metadata"]["total_papers"] = len(all_papers)

# Count by category
for paper in all_papers:
    if paper["category"] == "neuromorphic_computing":
        papers_collection["metadata"]["categories"]["neuromorphic_computing"] += 1
    elif paper["category"] == "infinite_context":
        papers_collection["metadata"]["categories"]["infinite_context"] += 1

# Save to JSON
with open("paper_collection.json", "w") as f:
    json.dump(papers_collection, f, indent=2)

print(f"Collected {len(all_papers)} papers")
print(f"Neuromorphic: {papers_collection['metadata']['categories']['neuromorphic_computing']}")
print(f"Infinite Context: {papers_collection['metadata']['categories']['infinite_context']}")