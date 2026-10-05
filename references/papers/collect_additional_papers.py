#!/usr/bin/env python3
"""
Additional Paper Collection Script
Collects recent cutting-edge papers to expand the research collection
"""

import requests
import os
import json
import time
from datetime import datetime

# New cutting-edge papers to add
NEW_PAPERS = {
    "neuromorphic": [
        {
            "title": "Neuromorphic Photonic Processing and Memory with Spiking Resonant Tunnelling Diode Neurons and Neural Networks",
            "arxiv_id": "2507.20866",
            "year": "2024",
            "category": "Photonic Neuromorphic",
            "relevance": 95
        },
        {
            "title": "Experimental neuromorphic computing based on quantum memristor",
            "arxiv_id": "2504.18694",
            "year": "2025",
            "category": "Quantum Neuromorphic",
            "relevance": 98
        },
        {
            "title": "A spiking photonic neural network of 40,000 neurons, trained with backpropagation",
            "arxiv_id": "2411.19209",
            "year": "2024",
            "category": "Photonic Neuromorphic",
            "relevance": 92
        },
        {
            "title": "Optical Spiking Neurons Enable High-Speed and Energy-Efficient Neuromorphic Computing",
            "arxiv_id": "2409.05726",
            "year": "2024",
            "category": "Photonic Neuromorphic",
            "relevance": 90
        },
        {
            "title": "Towards Optimal Deployment of Spiking Networks on Neuromorphic Hardware",
            "arxiv_id": "2510.15542",
            "year": "2024",
            "category": "Spiking Neural Networks",
            "relevance": 88
        },
        {
            "title": "Spiking Neural Networks: The Future of Brain-Inspired Computing",
            "arxiv_id": "2510.27379",
            "year": "2024",
            "category": "Spiking Neural Networks",
            "relevance": 85
        },
        {
            "title": "3-dimensional multistate memristor structures based neuromorphic computing",
            "arxiv_id": "26000094",
            "year": "2024",
            "category": "Memristor Hardware",
            "relevance": 87
        },
        {
            "title": "Recent Progress in Neuromorphic Computing from Memristive Devices",
            "arxiv_id": "0044",
            "year": "2024",
            "category": "Memristor Hardware",
            "relevance": 89
        },
        {
            "title": "Recent Advancements in 2D Material-Based Memristor Technology",
            "arxiv_id": "1451",
            "year": "2024",
            "category": "Memristor Hardware",
            "relevance": 86
        },
        {
            "title": "Integrated photonic synapses, neurons, memristors, and neural networks",
            "arxiv_id": "250011",
            "year": "2025",
            "category": "Photonic Neuromorphic",
            "relevance": 91
        }
    ],
    "infinite_context": [
        {
            "title": "A Memory-Efficient Infinite-Context Transformer for Edge Devices",
            "arxiv_id": "2503.22196",
            "year": "2025",
            "category": "Long Context Transformers",
            "relevance": 94
        },
        {
            "title": "Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing",
            "arxiv_id": "2502.12962",
            "year": "2025",
            "category": "Memory-Augmented Networks",
            "relevance": 96
        },
        {
            "title": "ReAttention: Training-Free Infinite Context with Finite Attention Scope",
            "arxiv_id": "2407.15176",
            "year": "2024",
            "category": "Long Context Transformers",
            "relevance": 88
        },
        {
            "title": "Human-Inspired Episodic Memory for Infinite Context Processing",
            "arxiv_id": "c05144b635df",
            "year": "2025",
            "category": "Memory-Augmented Networks",
            "relevance": 91
        },
        {
            "title": "InfiniPot: Infinite Context Processing on Memory-Constrained LLMs",
            "arxiv_id": "897",
            "year": "2024",
            "category": "Long Context Transformers",
            "relevance": 87
        },
        {
            "title": "RingAttention with Blockwise Transformers for Near-Infinite Context",
            "arxiv_id": "1119587863e7",
            "year": "2024",
            "category": "Long Context Transformers",
            "relevance": 93
        },
        {
            "title": "Retrieval-Augmented Generation with Graphs (GraphRAG)",
            "arxiv_id": "2501.00309",
            "year": "2025",
            "category": "Retrieval-Augmented Generation",
            "relevance": 95
        },
        {
            "title": "When to use Graphs in RAG: A Comprehensive Analysis",
            "arxiv_id": "2506.05690",
            "year": "2024",
            "category": "Retrieval-Augmented Generation",
            "relevance": 89
        },
        {
            "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
            "arxiv_id": "2410.05779",
            "year": "2024",
            "category": "Retrieval-Augmented Generation",
            "relevance": 92
        },
        {
            "title": "Retrieval-Augmented Generation with Extreme Long-Context Videos",
            "arxiv_id": "2502.01549",
            "year": "2025",
            "category": "Retrieval-Augmented Generation",
            "relevance": 88
        },
        {
            "title": "Sufficient Context: A New Lens on Retrieval Augmented Generation",
            "arxiv_id": "2411.06037",
            "year": "2024",
            "category": "Retrieval-Augmented Generation",
            "relevance": 86
        },
        {
            "title": "Graph of Records: Boosting Retrieval Augmented Generation",
            "arxiv_id": "2410.11001",
            "year": "2024",
            "category": "Retrieval-Augmented Generation",
            "relevance": 90
        },
        {
            "title": "SimRAG: Self-Improving Retrieval-Augmented Generation",
            "arxiv_id": "2410.17952",
            "year": "2024",
            "category": "Retrieval-Augmented Generation",
            "relevance": 87
        }
    ],
    "emerging_topics": [
        {
            "title": "Transformers are SSMs: Generalized Models and Efficient Algorithms",
            "arxiv_id": "2405.21060",
            "year": "2024",
            "category": "State Space Models",
            "relevance": 94
        },
        {
            "title": "SPATIAL-MAMBA: EFFECTIVE VISUAL STATE SPACE MODELS",
            "arxiv_id": "b7216f4a3248",
            "year": "2025",
            "category": "State Space Models",
            "relevance": 91
        },
        {
            "title": "Essential difficulties of the Mamba architecture demonstrated by analysis",
            "arxiv_id": "2509.17514",
            "year": "2024",
            "category": "State Space Models",
            "relevance": 85
        },
        {
            "title": "Characterizing the Behavior of Training Mamba-based State Space Models",
            "arxiv_id": "2508.17679",
            "year": "2024",
            "category": "State Space Models",
            "relevance": 87
        },
        {
            "title": "Mamba-ST: State Space Model for Efficient Style Transfer",
            "arxiv_id": "Botti_Mamba-ST",
            "year": "2025",
            "category": "State Space Models",
            "relevance": 83
        },
        {
            "title": "High-performance ternary logic circuits and neural networks",
            "arxiv_id": "PMC11721562",
            "year": "2024",
            "category": "Ternary Computing",
            "relevance": 89
        },
        {
            "title": "Neuromorphic Programming: Emerging Directions for Brain-Inspired Computing",
            "arxiv_id": "2410.22352",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 86
        },
        {
            "title": "Recent trends in neuromorphic systems for non-von Neumann architectures",
            "arxiv_id": "041304",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 84
        }
    ]
}

def sanitize_filename(title):
    """Create a safe filename from paper title"""
    # Remove or replace problematic characters
    safe_chars = []
    for char in title:
        if char.isalnum() or char in (' ', '-', '_', '(', ')', '[', ']', '.', ','):
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    filename = ''.join(safe_chars)
    # Limit length and replace spaces
    filename = filename.replace(' ', '_')
    if len(filename) > 100:
        filename = filename[:100] + '...'
    
    return filename

def download_paper(arxiv_id, title, category):
    """Download a paper from arXiv"""
    
    # Handle different ID formats
    if '/' in arxiv_id:
        # Already a full URL or has category
        if arxiv_id.startswith('http'):
            pdf_url = arxiv_id
        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        # Standard arXiv ID
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join("neuromorphic" if "neuromorphic" in category.lower() or "quantum" in category.lower() or "photonic" in category.lower() or "memristor" in category.lower() or "spiking" in category.lower() else "infinite_context", filename)
    
    print(f"Downloading: {title}")
    print(f"URL: {pdf_url}")
    print(f"Saving to: {filepath}")
    
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Successfully downloaded: {filename}")
        return True, filepath
        
    except Exception as e:
        print(f"✗ Failed to download {title}: {e}")
        return False, None

def main():
    """Main collection function"""
    print("🔬 Collecting Additional Cutting-Edge Research Papers")
    print("=" * 60)
    
    downloaded_papers = []
    failed_papers = []
    
    # Process each category
    for category, papers in NEW_PAPERS.items():
        print(f"\n📚 Processing {category.replace('_', ' ').title()} Papers:")
        print("-" * 40)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}/{len(papers)}: {paper['title']}")
            
            success, filepath = download_paper(
                paper['arxiv_id'], 
                paper['title'], 
                paper['category']
            )
            
            if success:
                paper_info = {
                    **paper,
                    'filepath': filepath,
                    'downloaded': True,
                    'download_date': datetime.now().isoformat()
                }
                downloaded_papers.append(paper_info)
            else:
                paper['downloaded'] = False
                failed_papers.append(paper)
            
            # Rate limiting
            time.sleep(1)
    
    # Save results
    results = {
        'collection_date': datetime.now().isoformat(),
        'total_papers_attempted': sum(len(papers) for papers in NEW_PAPERS.values()),
        'successful_downloads': len(downloaded_papers),
        'failed_downloads': len(failed_papers),
        'papers': downloaded_papers + failed_papers
    }
    
    with open('additional_papers_collection.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total papers attempted: {results['total_papers_attempted']}")
    print(f"Successfully downloaded: {results['successful_downloads']}")
    print(f"Failed downloads: {results['failed_downloads']}")
    print(f"Success rate: {results['successful_downloads']/results['total_papers_attempted']*100:.1f}%")
    
    if failed_papers:
        print("\n❌ Failed Downloads:")
        for paper in failed_papers:
            print(f"  - {paper['title']} ({paper['arxiv_id']})")
    
    print(f"\n✅ Collection completed! Results saved to 'additional_papers_collection.json'")

if __name__ == "__main__":
    main()