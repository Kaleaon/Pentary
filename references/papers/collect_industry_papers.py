#!/usr/bin/env python3
"""
Industry and Conference Paper Collection Script
Collects recent papers from major AI labs and conferences
"""

import requests
import os
import json
import time
from datetime import datetime

# Industry and conference papers to add
INDUSTRY_PAPERS = {
    "neuromorphic": [
        {
            "title": "Neuromorphic Principles for Efficient Large Language Models",
            "arxiv_id": "qaDM1R2nlm",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 94,
            "source": "OpenReview (NeurIPS/ICML)"
        },
        {
            "title": "Enhancing Efficiency of Neuromorphic In-Memory Computing",
            "arxiv_id": "2407.00641",
            "year": "2024",
            "category": "In-Memory Computing",
            "relevance": 91,
            "source": "arXiv"
        },
        {
            "title": "Neuromorphic Intelligence",
            "arxiv_id": "2509.11940",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 88,
            "source": "arXiv"
        },
        {
            "title": "Variable-precision neuromorphic state space model for on-edge devices",
            "arxiv_id": "0487X",
            "year": "2025",
            "category": "State Space Models",
            "relevance": 89,
            "source": "Neurocomputing"
        },
        {
            "title": "Modern Neuromorphic AI: From Intra-Token to Inter-Token Processing",
            "arxiv_id": "2601.00245",
            "year": "2025",
            "category": "Neuromorphic Computing",
            "relevance": 86,
            "source": "arXiv"
        },
        {
            "title": "Neuromorphic Computing in the Era of Large Models",
            "arxiv_id": "10977800",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 92,
            "source": "IEEE"
        },
        {
            "title": "Bridging Brains and Machines: A Unified Frontier in Neuroscience",
            "arxiv_id": "2507.10722",
            "year": "2024",
            "category": "Neuromorphic Computing",
            "relevance": 87,
            "source": "arXiv"
        },
        {
            "title": "Compute-in-Memory Implementation of State Space Models for Event Sequence Processing",
            "arxiv_id": "2511.13912",
            "year": "2024",
            "category": "State Space Models",
            "relevance": 93,
            "source": "Nature Communications"
        }
    ],
    "efficient_ai": [
        {
            "title": "Towards Efficient and Reliable AI Through Neuromorphic Principles",
            "arxiv_id": "2309.15942",
            "year": "2024",
            "category": "Efficient AI",
            "relevance": 90,
            "source": "arXiv"
        },
        {
            "title": "NeuroNAS: Enhancing Efficiency of Neuromorphic In-Memory Computing",
            "arxiv_id": "2407.00641",
            "year": "2024",
            "category": "Neural Architecture Search",
            "relevance": 89,
            "source": "arXiv"
        },
        {
            "title": "Hardware-Inspired Architectures for Energy-Efficient AI",
            "arxiv_id": "396002563",
            "year": "2024",
            "category": "Efficient AI",
            "relevance": 85,
            "source": "ResearchGate"
        }
    ],
    "emerging_applications": [
        {
            "title": "Can neuromorphic computing help reduce AI's high energy cost?",
            "arxiv_id": "PMC12595464",
            "year": "2024",
            "category": "Energy Efficiency",
            "relevance": 91,
            "source": "NIH"
        }
    ]
}

def sanitize_filename(title):
    """Create a safe filename from paper title"""
    safe_chars = []
    for char in title:
        if char.isalnum() or char in (' ', '-', '_', '(', ')', '[', ']', '.', ','):
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    filename = ''.join(safe_chars)
    filename = filename.replace(' ', '_')
    if len(filename) > 100:
        filename = filename[:100] + '...'
    
    return filename

def download_paper(paper_info):
    """Download a paper from various sources"""
    arxiv_id = paper_info['arxiv_id']
    title = paper_info['title']
    category = paper_info['category']
    
    # Determine download URL based on source
    if paper_info.get('source') == 'OpenReview':
        pdf_url = f"https://openreview.net/pdf?id={arxiv_id}"
    elif paper_info.get('source') == 'Nature Communications':
        pdf_url = f"https://www.nature.com/articles/s41467-025-68227-w.pdf"
    elif paper_info.get('source') == 'IEEE':
        pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arxiv_id}"
    elif paper_info.get('source') == 'ResearchGate':
        # For ResearchGate, we'll need to handle differently
        pdf_url = f"https://www.researchgate.net/publication/{arxiv_id}"
    elif paper_info.get('source') == 'NIH':
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{arxiv_id}/pdf"
    elif '/' in arxiv_id:
        if arxiv_id.startswith('http'):
            pdf_url = arxiv_id
        else:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    filename = sanitize_filename(title) + ".pdf"
    
    # Determine directory based on category
    if "neuromorphic" in category.lower() or "memory" in category.lower():
        directory = "neuromorphic"
    else:
        directory = "infinite_context"
    
    filepath = os.path.join(directory, filename)
    
    print(f"Downloading: {title}")
    print(f"Source: {paper_info.get('source', 'arXiv')}")
    print(f"URL: {pdf_url}")
    print(f"Saving to: {filepath}")
    
    try:
        # Special handling for different sources
        if paper_info.get('source') == 'ResearchGate':
            print("⚠️  ResearchGate papers may require manual download")
            return False, None
        elif paper_info.get('source') == 'IEEE':
            print("⚠️  IEEE papers may require subscription access")
            return False, None
        
        response = requests.get(pdf_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        os.makedirs(directory, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Successfully downloaded: {filename}")
        return True, filepath
        
    except Exception as e:
        print(f"✗ Failed to download {title}: {e}")
        return False, None

def main():
    """Main collection function"""
    print("🏢 Collecting Industry and Conference Research Papers")
    print("=" * 60)
    
    downloaded_papers = []
    failed_papers = []
    
    # Process each category
    for category, papers in INDUSTRY_PAPERS.items():
        print(f"\n📚 Processing {category.replace('_', ' ').title()} Papers:")
        print("-" * 40)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}/{len(papers)}: {paper['title']}")
            
            success, filepath = download_paper(paper)
            
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
        'total_papers_attempted': sum(len(papers) for papers in INDUSTRY_PAPERS.values()),
        'successful_downloads': len(downloaded_papers),
        'failed_downloads': len(failed_papers),
        'papers': downloaded_papers + failed_papers
    }
    
    with open('industry_papers_collection.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 INDUSTRY COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total papers attempted: {results['total_papers_attempted']}")
    print(f"Successfully downloaded: {results['successful_downloads']}")
    print(f"Failed downloads: {results['failed_downloads']}")
    print(f"Success rate: {results['successful_downloads']/results['total_papers_attempted']*100:.1f}%")
    
    if failed_papers:
        print("\n❌ Failed Downloads (may require manual access):")
        for paper in failed_papers:
            print(f"  - {paper['title']} ({paper.get('source', 'Unknown source')})")
    
    print(f"\n✅ Industry collection completed! Results saved to 'industry_papers_collection.json'")

if __name__ == "__main__":
    main()