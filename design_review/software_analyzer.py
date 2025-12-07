#!/usr/bin/env python3
"""
Comprehensive software implementation analyzer for Python files.
Checks for correctness, edge cases, and potential bugs.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

class PythonAnalyzer:
    """Analyzes Python implementations for correctness and bugs."""
    
    def __init__(self):
        self.issues = []
        
    def analyze_file(self, filepath: Path) -> Dict:
        """Analyze a Python file."""
        
        print(f"\nAnalyzing: {filepath.name}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        results = {
            'file': str(filepath),
            'critical_issues': [],
            'warnings': [],
            'code_smells': [],
            'metrics': {}
        }
        
        # Try to parse AST
        try:
            tree = ast.parse(content)
            results['critical_issues'].extend(self.check_arithmetic_correctness(tree, content))
            results['warnings'].extend(self.check_edge_cases(tree, content))
            results['code_smells'].extend(self.check_code_quality(tree, content))
            results['metrics'] = self.calculate_metrics(tree, content)
        except SyntaxError as e:
            results['critical_issues'].append({
                'severity': 'CRITICAL',
                'type': 'syntax_error',
                'message': f'Syntax error: {e}',
                'line': e.lineno
            })
        
        return results
    
    def check_arithmetic_correctness(self, tree: ast.AST, content: str) -> List[Dict]:
        """Check arithmetic operations for correctness."""
        issues = []
        
        # Check for division by zero
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                # Check if divisor could be zero
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    issues.append({
                        'severity': 'CRITICAL',
                        'type': 'division_by_zero',
                        'message': 'Division by zero detected',
                        'line': node.lineno
                    })
        
        # Check for integer overflow in pentary operations
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                # Check for potential overflow in multiplication
                if isinstance(node.op, ast.Mult):
                    # Look for pentary digit operations
                    if hasattr(node, 'lineno'):
                        issues.append({
                            'severity': 'WARNING',
                            'type': 'potential_overflow',
                            'message': 'Multiplication operation - verify overflow handling',
                            'line': node.lineno
                        })
        
        return issues
    
    def check_edge_cases(self, tree: ast.AST, content: str) -> List[Dict]:
        """Check for missing edge case handling."""
        warnings = []
        
        # Check for array indexing without bounds checking
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # Check if there's bounds checking nearby
                # This is a simplified check
                if hasattr(node, 'lineno'):
                    warnings.append({
                        'severity': 'WARNING',
                        'type': 'unchecked_indexing',
                        'message': 'Array indexing - verify bounds checking',
                        'line': node.lineno
                    })
        
        # Check for missing None checks
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check if there's a None check
                if hasattr(node, 'lineno'):
                    warnings.append({
                        'severity': 'INFO',
                        'type': 'potential_none',
                        'message': 'Attribute access - verify None checking',
                        'line': node.lineno
                    })
        
        return warnings
    
    def check_code_quality(self, tree: ast.AST, content: str) -> List[Dict]:
        """Check code quality issues."""
        smells = []
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    smells.append({
                        'severity': 'INFO',
                        'type': 'long_function',
                        'message': f'Function "{node.name}" is {func_lines} lines long - consider refactoring',
                        'line': node.lineno
                    })
        
        # Check for deeply nested code
        max_depth = self.calculate_nesting_depth(tree)
        if max_depth > 4:
            smells.append({
                'severity': 'INFO',
                'type': 'deep_nesting',
                'message': f'Maximum nesting depth is {max_depth} - consider refactoring',
                'line': 0
            })
        
        return smells
    
    def calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                child_depth = self.calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self.calculate_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def calculate_metrics(self, tree: ast.AST, content: str) -> Dict:
        """Calculate code metrics."""
        
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'functions': 0,
            'classes': 0,
            'complexity': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['functions'] += 1
            elif isinstance(node, ast.ClassDef):
                metrics['classes'] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                metrics['complexity'] += 1
        
        return metrics
    
    def analyze_directory(self, directory: Path, pattern: str = "pentary*.py") -> Dict:
        """Analyze Python files in directory."""
        
        results = {
            'files': [],
            'summary': {
                'total_files': 0,
                'total_critical': 0,
                'total_warnings': 0,
                'total_smells': 0
            }
        }
        
        python_files = sorted(list(directory.glob(pattern)))
        results['summary']['total_files'] = len(python_files)
        
        for pyfile in python_files:
            file_results = self.analyze_file(pyfile)
            results['files'].append(file_results)
            
            results['summary']['total_critical'] += len(file_results['critical_issues'])
            results['summary']['total_warnings'] += len(file_results['warnings'])
            results['summary']['total_smells'] += len(file_results['code_smells'])
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate analysis report."""
        
        report = "# Software Implementation Analysis Report\n\n"
        report += "## Summary\n\n"
        report += f"- **Files Analyzed:** {results['summary']['total_files']}\n"
        report += f"- **Critical Issues:** {results['summary']['total_critical']}\n"
        report += f"- **Warnings:** {results['summary']['total_warnings']}\n"
        report += f"- **Code Smells:** {results['summary']['total_smells']}\n\n"
        
        # Critical issues
        if results['summary']['total_critical'] > 0:
            report += "## Critical Issues ❌\n\n"
            for file_result in results['files']:
                if file_result['critical_issues']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for issue in file_result['critical_issues']:
                        report += f"- **Line {issue['line']}**: {issue['message']}\n"
                    report += "\n"
        
        # Warnings
        if results['summary']['total_warnings'] > 0:
            report += "## Warnings ⚠️\n\n"
            report += "Note: Many warnings are informational and may be false positives.\n\n"
            for file_result in results['files']:
                if file_result['warnings']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    report += f"- Total warnings: {len(file_result['warnings'])}\n"
                    report += "- Review for: bounds checking, None handling, type validation\n\n"
        
        # Metrics
        report += "## Code Metrics\n\n"
        report += "| File | Lines | Functions | Classes | Complexity |\n"
        report += "|------|-------|-----------|---------|------------|\n"
        
        for file_result in results['files']:
            m = file_result['metrics']
            report += f"| {Path(file_result['file']).name} | {m['lines_of_code']} | "
            report += f"{m['functions']} | {m['classes']} | {m['complexity']} |\n"
        
        return report

if __name__ == "__main__":
    print("="*60)
    print("Software Implementation Analysis")
    print("="*60)
    
    analyzer = PythonAnalyzer()
    
    # Analyze tools directory
    results = analyzer.analyze_directory(Path("pentary-repo/tools"))
    
    # Generate report
    report = analyzer.generate_report(results)
    with open("software_analysis_report.md", "w") as f:
        f.write(report)
    
    # Save raw results
    with open("software_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Files analyzed: {results['summary']['total_files']}")
    print(f"Critical issues: {results['summary']['total_critical']}")
    print(f"Warnings: {results['summary']['total_warnings']}")
    print("\nGenerated software_analysis_report.md")