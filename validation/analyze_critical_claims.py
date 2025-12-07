#!/usr/bin/env python3
"""
Analyze and prioritize critical claims for validation.
"""

import json
from collections import defaultdict
from typing import List, Dict

def load_claims(filename: str) -> List[Dict]:
    """Load claims from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_critical_claims(claims: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract the most critical claims that need validation."""
    
    critical_keywords = [
        # Performance multipliers
        '5×', '10×', '2×', '3×', '4×', 'faster', 'speedup',
        # Efficiency claims
        '99.9%', '40%', '60%', 'reduction', 'lower power',
        # Cost claims
        '$8,000', '$5-75', 'cost',
        # Capacity claims
        '10M+', '2.5M', 'neurons', 'tokens',
        # Comparisons
        'vs TPU', 'vs H100', 'vs Bitcoin', 'vs Ethereum',
        # Specific technologies
        'memristor', 'quantum', 'neuromorphic'
    ]
    
    critical = defaultdict(list)
    
    for claim in claims:
        text_lower = claim['text'].lower()
        match_lower = claim['match'].lower()
        
        # Check for critical keywords
        for keyword in critical_keywords:
            if keyword.lower() in text_lower or keyword.lower() in match_lower:
                # Categorize by file type
                file_path = claim['file']
                if 'research/' in file_path:
                    category = 'research'
                elif 'architecture/' in file_path:
                    category = 'architecture'
                elif 'hardware/' in file_path:
                    category = 'hardware'
                else:
                    category = 'general'
                
                critical[category].append(claim)
                break
    
    return critical

def identify_top_claims(claims: List[Dict], n: int = 50) -> List[Dict]:
    """Identify top N most impactful claims."""
    
    # Score claims by impact
    scored_claims = []
    
    for claim in claims:
        score = 0
        text = claim['text'].lower()
        match = claim['match'].lower()
        
        # High impact multipliers
        if any(x in match for x in ['10×', '10x', '5×', '5x']):
            score += 10
        elif any(x in match for x in ['3×', '3x', '4×', '4x']):
            score += 7
        elif any(x in match for x in ['2×', '2x']):
            score += 5
        
        # High percentage claims
        if '99' in match or '90' in match:
            score += 8
        elif any(x in match for x in ['50%', '60%', '70%', '80%']):
            score += 6
        
        # Comparison claims (high impact)
        if any(x in text for x in ['vs tpu', 'vs h100', 'vs nvidia', 'vs google']):
            score += 8
        
        # Cost claims
        if '$' in match:
            score += 5
        
        # Capacity claims
        if any(x in text for x in ['10m+', '2.5m', 'million', 'billion']):
            score += 6
        
        # Research papers (need validation)
        if 'research/' in claim['file']:
            score += 3
        
        scored_claims.append((score, claim))
    
    # Sort by score and return top N
    scored_claims.sort(reverse=True, key=lambda x: x[0])
    return [claim for score, claim in scored_claims[:n]]

def generate_validation_priorities(critical_claims: Dict[str, List[Dict]], 
                                   top_claims: List[Dict]) -> str:
    """Generate a prioritized validation report."""
    
    report = "# Critical Claims Validation Priorities\n\n"
    report += "## Executive Summary\n\n"
    report += f"- **Total Critical Claims:** {sum(len(v) for v in critical_claims.values())}\n"
    report += f"- **Top Priority Claims:** {len(top_claims)}\n\n"
    
    report += "## Top 50 Highest Impact Claims Requiring Validation\n\n"
    
    for i, claim in enumerate(top_claims, 1):
        report += f"### {i}. {claim['match']}\n\n"
        report += f"**File:** `{claim['file']}`  \n"
        report += f"**Line:** {claim['line']}  \n"
        report += f"**Type:** {claim['type']}  \n"
        report += f"**Context:** {claim['text'][:150]}...\n\n"
        report += "**Validation Required:**\n"
        
        # Determine validation method
        text_lower = claim['text'].lower()
        if any(x in text_lower for x in ['faster', 'speedup', 'performance']):
            report += "- [ ] Performance benchmark\n"
            report += "- [ ] Simulation test\n"
        if any(x in text_lower for x in ['power', 'energy', 'efficiency']):
            report += "- [ ] Power consumption analysis\n"
            report += "- [ ] Energy efficiency calculation\n"
        if any(x in text_lower for x in ['vs', 'compared to', 'better than']):
            report += "- [ ] Comparative benchmark\n"
            report += "- [ ] Literature review\n"
        if '$' in claim['match']:
            report += "- [ ] Cost breakdown analysis\n"
            report += "- [ ] Market price comparison\n"
        if any(x in text_lower for x in ['neurons', 'tokens', 'capacity']):
            report += "- [ ] Capacity calculation\n"
            report += "- [ ] Memory analysis\n"
        
        report += "\n---\n\n"
    
    report += "## Critical Claims by Category\n\n"
    
    for category, cat_claims in sorted(critical_claims.items()):
        report += f"### {category.title()} ({len(cat_claims)} claims)\n\n"
        
        # Group by file
        by_file = defaultdict(list)
        for claim in cat_claims:
            by_file[claim['file']].append(claim)
        
        for file, file_claims in sorted(by_file.items())[:5]:  # Top 5 files
            report += f"#### {file} ({len(file_claims)} claims)\n\n"
            for claim in file_claims[:3]:  # Top 3 claims per file
                report += f"- **{claim['match']}** (Line {claim['line']})\n"
            if len(file_claims) > 3:
                report += f"- ... and {len(file_claims) - 3} more\n"
            report += "\n"
    
    return report

if __name__ == "__main__":
    print("Loading claims...")
    claims = load_claims("claims_extracted.json")
    
    print(f"Analyzing {len(claims)} claims...")
    
    critical_claims = extract_critical_claims(claims)
    print(f"\nCritical claims by category:")
    for category, cat_claims in critical_claims.items():
        print(f"  {category}: {len(cat_claims)}")
    
    top_claims = identify_top_claims(claims, n=50)
    print(f"\nIdentified top {len(top_claims)} highest impact claims")
    
    report = generate_validation_priorities(critical_claims, top_claims)
    
    with open("validation_priorities.md", "w") as f:
        f.write(report)
    
    print("\nGenerated validation_priorities.md")
    
    # Save top claims to separate JSON for easier processing
    with open("top_claims.json", "w") as f:
        json.dump(top_claims, f, indent=2)
    
    print("Saved top_claims.json")