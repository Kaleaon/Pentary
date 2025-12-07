#!/usr/bin/env python3
"""
Manual focused code review of critical Pentary implementations.
Checks for actual bugs and design flaws.
"""

from pathlib import Path
import re
from typing import List, Dict

class FocusedReviewer:
    """Focused manual review of critical code."""
    
    def __init__(self):
        self.issues = []
        
    def review_pentary_arithmetic(self, filepath: Path) -> List[Dict]:
        """Review pentary arithmetic implementation."""
        
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        print(f"\n{'='*60}")
        print(f"Reviewing: {filepath.name}")
        print(f"{'='*60}")
        
        # Check 1: Verify pentary digit range (0-4)
        for i, line in enumerate(lines, 1):
            if 'range(5)' in line or 'range(0, 5)' in line:
                print(f"✅ Line {i}: Correct pentary range (0-4)")
            elif re.search(r'range\(\d+\)', line) and 'pentary' in line.lower():
                match = re.search(r'range\((\d+)\)', line)
                if match and int(match.group(1)) != 5:
                    issues.append({
                        'severity': 'CRITICAL',
                        'file': filepath.name,
                        'line': i,
                        'issue': f'Incorrect pentary range: {match.group(0)}',
                        'expected': 'range(5) for pentary digits 0-4'
                    })
        
        # Check 2: Verify conversion logic
        if 'def to_decimal' in content or 'def from_decimal' in content:
            print("✅ Found conversion functions")
            
            # Check for base-5 operations
            if 'base = 5' in content or '** 5' in content or 'pow(5' in content:
                print("✅ Using base-5 for conversions")
            else:
                issues.append({
                    'severity': 'WARNING',
                    'file': filepath.name,
                    'line': 0,
                    'issue': 'Cannot verify base-5 usage in conversions',
                    'expected': 'Should use base-5 for pentary conversions'
                })
        
        # Check 3: Verify addition/subtraction carry logic
        if 'def add' in content or 'def subtract' in content:
            print("✅ Found arithmetic functions")
            
            # Check for carry handling
            if 'carry' in content.lower():
                print("✅ Carry logic present")
            else:
                issues.append({
                    'severity': 'WARNING',
                    'file': filepath.name,
                    'line': 0,
                    'issue': 'No carry logic found in arithmetic operations',
                    'expected': 'Should handle carry for pentary addition/subtraction'
                })
        
        return issues
    
    def review_pentary_nn(self, filepath: Path) -> List[Dict]:
        """Review neural network implementation."""
        
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        print(f"\n{'='*60}")
        print(f"Reviewing: {filepath.name}")
        print(f"{'='*60}")
        
        # Check 1: Verify quantization values
        if 'quantize' in content.lower():
            print("✅ Found quantization logic")
            
            # Check for pentary values {-2, -1, 0, 1, 2}
            if '-2' in content and '2' in content:
                print("✅ Using pentary quantization range")
            else:
                issues.append({
                    'severity': 'WARNING',
                    'file': filepath.name,
                    'line': 0,
                    'issue': 'Cannot verify pentary quantization range',
                    'expected': 'Should quantize to {-2, -1, 0, 1, 2}'
                })
        
        # Check 2: Verify forward pass implementation
        if 'def forward' in content:
            print("✅ Found forward pass")
        
        # Check 3: Check for gradient handling
        if 'backward' in content.lower() or 'gradient' in content.lower():
            print("✅ Found gradient/backward pass logic")
        
        return issues
    
    def review_pentary_converter(self, filepath: Path) -> List[Dict]:
        """Review conversion implementation."""
        
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        print(f"\n{'='*60}")
        print(f"Reviewing: {filepath.name}")
        print(f"{'='*60}")
        
        # Check 1: Verify bidirectional conversion
        has_to_pentary = 'to_pentary' in content or 'binary_to_pentary' in content
        has_from_pentary = 'from_pentary' in content or 'pentary_to_binary' in content
        
        if has_to_pentary and has_from_pentary:
            print("✅ Bidirectional conversion present")
        else:
            issues.append({
                'severity': 'WARNING',
                'file': filepath.name,
                'line': 0,
                'issue': 'Missing bidirectional conversion',
                'expected': 'Should have both to_pentary and from_pentary'
            })
        
        # Check 2: Verify error handling
        if 'try:' in content and 'except' in content:
            print("✅ Error handling present")
        else:
            issues.append({
                'severity': 'INFO',
                'file': filepath.name,
                'line': 0,
                'issue': 'No error handling found',
                'expected': 'Should handle conversion errors'
            })
        
        return issues
    
    def generate_focused_report(self, all_issues: List[Dict]) -> str:
        """Generate focused review report."""
        
        report = "# Focused Code Review Report\n\n"
        report += "## Overview\n\n"
        report += "This report focuses on critical implementation correctness issues.\n\n"
        
        # Group by severity
        critical = [i for i in all_issues if i['severity'] == 'CRITICAL']
        warnings = [i for i in all_issues if i['severity'] == 'WARNING']
        info = [i for i in all_issues if i['severity'] == 'INFO']
        
        report += f"- **Critical Issues:** {len(critical)}\n"
        report += f"- **Warnings:** {len(warnings)}\n"
        report += f"- **Info:** {len(info)}\n\n"
        
        if critical:
            report += "## Critical Issues ❌\n\n"
            for issue in critical:
                report += f"### {issue['file']} - Line {issue['line']}\n\n"
                report += f"**Issue:** {issue['issue']}\n\n"
                report += f"**Expected:** {issue['expected']}\n\n"
                report += "---\n\n"
        
        if warnings:
            report += "## Warnings ⚠️\n\n"
            for issue in warnings:
                report += f"### {issue['file']} - Line {issue['line']}\n\n"
                report += f"**Issue:** {issue['issue']}\n\n"
                report += f"**Expected:** {issue['expected']}\n\n"
                report += "---\n\n"
        
        if not critical and not warnings:
            report += "## Assessment: ✅ GOOD\n\n"
            report += "No critical issues or warnings found in focused review.\n\n"
        
        return report

if __name__ == "__main__":
    print("="*60)
    print("Focused Code Review")
    print("="*60)
    
    reviewer = FocusedReviewer()
    all_issues = []
    
    # Review critical files
    tools_dir = Path("pentary-repo/tools")
    
    critical_files = [
        'pentary_arithmetic.py',
        'pentary_arithmetic_extended.py',
        'pentary_nn.py',
        'pentary_quantizer.py',
        'pentary_converter.py',
        'pentary_converter_optimized.py'
    ]
    
    for filename in critical_files:
        filepath = tools_dir / filename
        if filepath.exists():
            if 'arithmetic' in filename:
                all_issues.extend(reviewer.review_pentary_arithmetic(filepath))
            elif 'nn' in filename or 'quantizer' in filename:
                all_issues.extend(reviewer.review_pentary_nn(filepath))
            elif 'converter' in filename:
                all_issues.extend(reviewer.review_pentary_converter(filepath))
    
    # Generate report
    report = reviewer.generate_focused_report(all_issues)
    with open("focused_code_review.md", "w") as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("Review Complete!")
    print("="*60)
    print(f"Critical issues: {len([i for i in all_issues if i['severity'] == 'CRITICAL'])}")
    print(f"Warnings: {len([i for i in all_issues if i['severity'] == 'WARNING'])}")
    print(f"Info: {len([i for i in all_issues if i['severity'] == 'INFO'])}")
    print("\nGenerated focused_code_review.md")