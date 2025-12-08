#!/usr/bin/env python3
"""
Extract all quantitative claims from Pentary repository markdown files.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json

class ClaimExtractor:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.claims = []
        
        # Patterns to identify claims
        self.patterns = [
            # Performance improvements: "5× faster", "2.5x speedup"
            (r'(\d+\.?\d*)[×x]\s*(faster|speedup|improvement|better|higher|more efficient)', 'performance_multiplier'),
            
            # Percentage improvements: "99.9% reduction", "40% lower"
            (r'(\d+\.?\d*)%\s*(reduction|lower|decrease|improvement|increase|higher|better|more)', 'percentage_change'),
            
            # Absolute values: "100 TPS", "2.5M neurons", "10M+ tokens"
            (r'(\d+\.?\d*[KMBT]?\+?)\s*([A-Z]{2,}|neurons|tokens|operations|watts|bytes)', 'absolute_value'),
            
            # Comparisons: "vs", "compared to", "better than"
            (r'(vs\.?|compared to|better than|faster than|more efficient than)\s+([A-Z][A-Za-z0-9\s]+)', 'comparison'),
            
            # Ranges: "2-3× faster", "$5-75"
            (r'(\d+\.?\d*)-(\d+\.?\d*)([×x%]?)\s*([a-zA-Z]+)', 'range'),
            
            # Cost claims: "$8,000", "$0-50"
            (r'\$(\d+,?\d*\.?\d*)', 'cost'),
            
            # Time claims: "2-4 weeks", "30 minutes"
            (r'(\d+\.?\d*)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)', 'time'),
            
            # Power claims: "500W", "8-16 TFLOPS/W"
            (r'(\d+\.?\d*)\s*(W|TFLOPS/W|GFLOPS/W|mW)', 'power'),
        ]
    
    def extract_from_file(self, file_path: Path) -> List[Dict]:
        """Extract claims from a single markdown file."""
        claims = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, claim_type in self.patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            claim = {
                                'file': str(file_path.relative_to(self.repo_path)),
                                'line': line_num,
                                'type': claim_type,
                                'text': line.strip(),
                                'match': match.group(0),
                                'context': self._get_context(lines, line_num),
                                'needs_validation': True
                            }
                            claims.append(claim)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return claims
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 2) -> str:
        """Get surrounding context for a claim."""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return '\n'.join(lines[start:end])
    
    def extract_all_claims(self) -> List[Dict]:
        """Extract claims from all markdown files in repository."""
        md_files = list(self.repo_path.rglob('*.md'))
        
        print(f"Found {len(md_files)} markdown files")
        
        all_claims = []
        for md_file in md_files:
            file_claims = self.extract_from_file(md_file)
            all_claims.extend(file_claims)
            if file_claims:
                print(f"  {md_file.name}: {len(file_claims)} claims")
        
        return all_claims
    
    def categorize_claims(self, claims: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize claims by type and file."""
        categorized = {
            'performance': [],
            'efficiency': [],
            'cost': [],
            'capacity': [],
            'comparison': [],
            'time': [],
            'other': []
        }
        
        for claim in claims:
            claim_type = claim['type']
            
            if 'performance' in claim_type or 'speedup' in claim['text'].lower():
                categorized['performance'].append(claim)
            elif 'reduction' in claim['text'].lower() or 'efficiency' in claim['text'].lower():
                categorized['efficiency'].append(claim)
            elif claim_type == 'cost':
                categorized['cost'].append(claim)
            elif 'neurons' in claim['text'].lower() or 'tokens' in claim['text'].lower():
                categorized['capacity'].append(claim)
            elif claim_type == 'comparison':
                categorized['comparison'].append(claim)
            elif claim_type == 'time':
                categorized['time'].append(claim)
            else:
                categorized['other'].append(claim)
        
        return categorized
    
    def save_claims(self, claims: List[Dict], output_file: str):
        """Save extracted claims to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(claims, f, indent=2)
        print(f"\nSaved {len(claims)} claims to {output_file}")
    
    def generate_report(self, claims: List[Dict], categorized: Dict[str, List[Dict]]) -> str:
        """Generate a markdown report of all claims."""
        report = "# Pentary Repository Claims Extraction Report\n\n"
        report += f"**Total Claims Found:** {len(claims)}\n\n"
        
        report += "## Claims by Category\n\n"
        for category, cat_claims in categorized.items():
            report += f"### {category.title()} ({len(cat_claims)} claims)\n\n"
            
            # Group by file
            by_file = {}
            for claim in cat_claims:
                file = claim['file']
                if file not in by_file:
                    by_file[file] = []
                by_file[file].append(claim)
            
            for file, file_claims in sorted(by_file.items()):
                report += f"#### {file}\n\n"
                for claim in file_claims[:10]:  # Limit to first 10 per file
                    report += f"- **Line {claim['line']}:** {claim['match']}\n"
                    report += f"  - Context: `{claim['text'][:100]}...`\n"
                if len(file_claims) > 10:
                    report += f"  - ... and {len(file_claims) - 10} more claims\n"
                report += "\n"
        
        report += "## Claims Requiring Validation\n\n"
        report += "All extracted claims need validation through:\n"
        report += "1. Mathematical proofs\n"
        report += "2. Simulations\n"
        report += "3. Benchmarks\n"
        report += "4. Literature references\n\n"
        
        return report

if __name__ == "__main__":
    extractor = ClaimExtractor("pentary-repo")
    
    print("Extracting claims from Pentary repository...")
    claims = extractor.extract_all_claims()
    
    print(f"\nTotal claims extracted: {len(claims)}")
    
    categorized = extractor.categorize_claims(claims)
    
    print("\nClaims by category:")
    for category, cat_claims in categorized.items():
        print(f"  {category}: {len(cat_claims)}")
    
    # Save to JSON
    extractor.save_claims(claims, "claims_extracted.json")
    
    # Generate report
    report = extractor.generate_report(claims, categorized)
    with open("claims_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\nGenerated claims_report.md")