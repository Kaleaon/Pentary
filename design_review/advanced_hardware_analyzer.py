#!/usr/bin/env python3
"""
Advanced hardware design analyzer with deep Verilog understanding.
Focuses on real design flaws, not false positives.
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

class AdvancedVerilogAnalyzer:
    """Advanced Verilog analyzer with context-aware checking."""
    
    def __init__(self):
        self.critical_issues = []
        self.design_flaws = []
        self.optimization_opportunities = []
        
    def analyze_file(self, filepath: Path) -> Dict:
        """Perform deep analysis of Verilog file."""
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {filepath.name}")
        print(f"{'='*60}")
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Remove comments for analysis
        clean_content = self.remove_comments(content)
        clean_lines = clean_content.split('\n')
        
        results = {
            'file': str(filepath),
            'critical_issues': [],
            'design_flaws': [],
            'optimizations': [],
            'analysis': {}
        }
        
        # Deep analysis checks
        results['critical_issues'].extend(self.check_signal_bit_width_errors(clean_content, clean_lines))
        results['critical_issues'].extend(self.check_always_block_issues(clean_content, clean_lines))
        results['design_flaws'].extend(self.check_timing_issues(clean_content, clean_lines))
        results['design_flaws'].extend(self.check_state_machine_completeness(clean_content, clean_lines))
        results['design_flaws'].extend(self.check_reset_synchronization(clean_content, clean_lines))
        results['optimizations'].extend(self.check_resource_usage(clean_content, clean_lines))
        results['analysis'] = self.perform_structural_analysis(clean_content, clean_lines)
        
        return results
    
    def remove_comments(self, content: str) -> str:
        """Remove comments from Verilog code."""
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content
    
    def check_signal_bit_width_errors(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for actual bit width mismatches in operations."""
        issues = []
        
        # Extract all signal declarations with their widths
        signals = {}
        for i, line in enumerate(lines, 1):
            # Match various declaration patterns
            patterns = [
                r'(wire|reg|input|output)\s+\[(\d+):(\d+)\]\s+(\w+)',
                r'(wire|reg|input|output)\s+(\w+)\s*\[(\d+):(\d+)\]'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) == 4:
                        sig_type, msb, lsb, name = groups
                        # Skip if msb/lsb are not numeric
                        if not msb.isdigit() or not lsb.isdigit():
                            continue
                        width = int(msb) - int(lsb) + 1
                        signals[name] = {'width': width, 'line': i}
        
        # Check arithmetic operations for width mismatches
        for i, line in enumerate(lines, 1):
            # Check additions/subtractions
            if '+' in line or '-' in line:
                # Extract operands
                match = re.search(r'(\w+)\s*[+\-]\s*(\w+)', line)
                if match:
                    op1, op2 = match.groups()
                    if op1 in signals and op2 in signals:
                        if signals[op1]['width'] != signals[op2]['width']:
                            issues.append({
                                'severity': 'CRITICAL',
                                'type': 'width_mismatch',
                                'message': f'Width mismatch in operation: {op1}[{signals[op1]["width"]}] and {op2}[{signals[op2]["width"]}]',
                                'line': i,
                                'fix': f'Ensure both operands have same width or use explicit width conversion'
                            })
        
        return issues
    
    def check_always_block_issues(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for issues in always blocks."""
        issues = []
        
        in_always = False
        always_type = None
        always_start = 0
        sensitivity_list = []
        
        for i, line in enumerate(lines, 1):
            # Detect always block start
            if 'always' in line and '@' in line:
                in_always = True
                always_start = i
                
                # Determine type
                if 'posedge' in line or 'negedge' in line:
                    always_type = 'sequential'
                    # Extract sensitivity list
                    match = re.search(r'@\s*\((.*?)\)', line)
                    if match:
                        sensitivity_list = [s.strip() for s in match.group(1).split('or')]
                else:
                    always_type = 'combinational'
                    match = re.search(r'@\s*\((.*?)\)', line)
                    if match:
                        sensitivity_list = [s.strip() for s in match.group(1).split('or')]
            
            # Check assignments within always block
            if in_always:
                # Check for blocking vs non-blocking
                if '=' in line and not '<=' in line:
                    # Blocking assignment
                    if always_type == 'sequential':
                        # Extract signal name
                        match = re.search(r'(\w+)\s*=', line)
                        if match and 'if' not in line and 'else' not in line:
                            issues.append({
                                'severity': 'CRITICAL',
                                'type': 'blocking_in_sequential',
                                'message': f'Blocking assignment in sequential always block at line {i}',
                                'line': i,
                                'fix': 'Use non-blocking assignment (<=) in sequential always blocks'
                            })
                
                elif '<=' in line:
                    # Non-blocking assignment
                    if always_type == 'combinational':
                        match = re.search(r'(\w+)\s*<=', line)
                        if match:
                            issues.append({
                                'severity': 'CRITICAL',
                                'type': 'nonblocking_in_combinational',
                                'message': f'Non-blocking assignment in combinational always block at line {i}',
                                'line': i,
                                'fix': 'Use blocking assignment (=) in combinational always blocks'
                            })
                
                # Check for incomplete sensitivity list in combinational blocks
                if always_type == 'combinational' and sensitivity_list:
                    # Extract signals used in RHS
                    rhs_signals = re.findall(r'\b(\w+)\b', line)
                    for sig in rhs_signals:
                        if sig not in sensitivity_list and sig not in ['if', 'else', 'case', 'begin', 'end']:
                            # This is a potential incomplete sensitivity list
                            pass  # Too many false positives, skip for now
            
            # Detect end of always block
            if in_always and 'end' in line:
                # Check if this is the end of the always block
                if line.strip() == 'end' or line.strip().startswith('end'):
                    in_always = False
                    always_type = None
                    sensitivity_list = []
        
        return issues
    
    def check_timing_issues(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for potential timing issues."""
        issues = []
        
        # Check for long combinational paths
        assign_chains = {}
        for i, line in enumerate(lines, 1):
            if 'assign' in line:
                match = re.search(r'assign\s+(\w+)\s*=\s*(.+);', line)
                if match:
                    lhs, rhs = match.groups()
                    # Count operations in RHS
                    ops = rhs.count('+') + rhs.count('-') + rhs.count('*') + rhs.count('&') + rhs.count('|')
                    if ops > 5:
                        issues.append({
                            'severity': 'WARNING',
                            'type': 'long_combinational_path',
                            'message': f'Long combinational path detected ({ops} operations) in assign statement',
                            'line': i,
                            'fix': 'Consider pipelining this operation'
                        })
        
        # Check for potential clock domain crossing
        clocks = set()
        for line in lines:
            match = re.findall(r'posedge\s+(\w+)', line)
            clocks.update(match)
        
        if len(clocks) > 1:
            issues.append({
                'severity': 'WARNING',
                'type': 'multiple_clock_domains',
                'message': f'Multiple clock domains detected: {", ".join(clocks)}',
                'line': 0,
                'fix': 'Ensure proper clock domain crossing synchronization'
            })
        
        return issues
    
    def check_state_machine_completeness(self, content: str, lines: List[str]) -> List[Dict]:
        """Check state machine for completeness."""
        issues = []
        
        # Look for state machine
        has_state = False
        states = set()
        transitions = {}
        default_case = False
        
        for i, line in enumerate(lines, 1):
            # Find state declarations
            if 'parameter' in line or 'localparam' in line:
                match = re.search(r'(\w+)\s*=\s*\d+', line)
                if match and ('STATE' in match.group(1).upper() or 'state' in match.group(1).lower()):
                    states.add(match.group(1))
                    has_state = True
            
            # Find case statements
            if 'case' in line and 'state' in line.lower():
                in_case = True
            
            if 'default' in line and ':' in line:
                default_case = True
        
        if has_state:
            if not default_case:
                issues.append({
                    'severity': 'WARNING',
                    'type': 'incomplete_state_machine',
                    'message': 'State machine found but no default case in case statement',
                    'line': 0,
                    'fix': 'Add default case to handle unexpected states'
                })
        
        return issues
    
    def check_reset_synchronization(self, content: str, lines: List[str]) -> List[Dict]:
        """Check reset logic for proper synchronization."""
        issues = []
        
        # Check for asynchronous reset
        async_reset = False
        sync_reset = False
        
        for line in lines:
            if 'posedge' in line and 'reset' in line.lower():
                if 'or' in line:
                    async_reset = True
                else:
                    sync_reset = True
        
        if async_reset and sync_reset:
            issues.append({
                'severity': 'WARNING',
                'type': 'mixed_reset_style',
                'message': 'Mixed asynchronous and synchronous reset detected',
                'line': 0,
                'fix': 'Use consistent reset style throughout design'
            })
        
        return issues
    
    def check_resource_usage(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for resource usage optimization opportunities."""
        optimizations = []
        
        # Check for large multipliers
        for i, line in enumerate(lines, 1):
            if '*' in line:
                # Check if multiplying by power of 2
                match = re.search(r'\*\s*(\d+)', line)
                if match:
                    value = int(match.group(1))
                    if value > 0 and (value & (value - 1)) == 0:
                        # Power of 2
                        shift = value.bit_length() - 1
                        optimizations.append({
                            'severity': 'INFO',
                            'type': 'optimization',
                            'message': f'Multiplication by {value} can be replaced with left shift by {shift}',
                            'line': i,
                            'fix': f'Replace * {value} with << {shift}'
                        })
        
        # Check for redundant logic
        for i, line in enumerate(lines, 1):
            # Check for double negation
            if '!!' in line:
                optimizations.append({
                    'severity': 'INFO',
                    'type': 'optimization',
                    'message': 'Double negation detected',
                    'line': i,
                    'fix': 'Remove double negation'
                })
        
        return optimizations
    
    def perform_structural_analysis(self, content: str, lines: List[str]) -> Dict:
        """Perform structural analysis of the design."""
        
        analysis = {
            'modules': [],
            'always_blocks': 0,
            'assign_statements': 0,
            'registers': 0,
            'wires': 0,
            'parameters': 0,
            'instantiations': 0,
            'complexity_score': 0
        }
        
        # Count modules
        modules = re.findall(r'module\s+(\w+)', content)
        analysis['modules'] = modules
        
        # Count constructs
        analysis['always_blocks'] = content.count('always')
        analysis['assign_statements'] = content.count('assign')
        analysis['registers'] = len(re.findall(r'\breg\b', content))
        analysis['wires'] = len(re.findall(r'\bwire\b', content))
        analysis['parameters'] = len(re.findall(r'\bparameter\b', content))
        
        # Count module instantiations
        for line in lines:
            if re.match(r'\s*\w+\s+\w+\s*\(', line):
                if 'module' not in line and 'function' not in line:
                    analysis['instantiations'] += 1
        
        # Calculate complexity score
        analysis['complexity_score'] = (
            analysis['always_blocks'] * 10 +
            analysis['assign_statements'] * 2 +
            analysis['registers'] * 1 +
            analysis['instantiations'] * 5
        )
        
        return analysis
    
    def analyze_directory(self, directory: Path) -> Dict:
        """Analyze all Verilog files in directory."""
        
        results = {
            'files': [],
            'summary': {
                'total_files': 0,
                'total_critical': 0,
                'total_design_flaws': 0,
                'total_optimizations': 0
            }
        }
        
        verilog_files = sorted(list(directory.glob('*.v')))
        # Exclude testbenches for now
        verilog_files = [f for f in verilog_files if not f.name.startswith('testbench')]
        
        results['summary']['total_files'] = len(verilog_files)
        
        for vfile in verilog_files:
            file_results = self.analyze_file(vfile)
            results['files'].append(file_results)
            
            results['summary']['total_critical'] += len(file_results['critical_issues'])
            results['summary']['total_design_flaws'] += len(file_results['design_flaws'])
            results['summary']['total_optimizations'] += len(file_results['optimizations'])
        
        return results
    
    def generate_detailed_report(self, results: Dict) -> str:
        """Generate detailed markdown report."""
        
        report = "# Advanced Hardware Design Analysis Report\n\n"
        report += "## Executive Summary\n\n"
        report += f"- **Files Analyzed:** {results['summary']['total_files']}\n"
        report += f"- **Critical Issues:** {results['summary']['total_critical']}\n"
        report += f"- **Design Flaws:** {results['summary']['total_design_flaws']}\n"
        report += f"- **Optimization Opportunities:** {results['summary']['total_optimizations']}\n\n"
        
        # Overall assessment
        if results['summary']['total_critical'] == 0:
            report += "### Overall Assessment: ✅ GOOD\n\n"
            report += "No critical issues found. Design appears structurally sound.\n\n"
        elif results['summary']['total_critical'] < 5:
            report += "### Overall Assessment: ⚠️ NEEDS ATTENTION\n\n"
            report += "Some critical issues found that should be addressed.\n\n"
        else:
            report += "### Overall Assessment: ❌ REQUIRES FIXES\n\n"
            report += "Multiple critical issues found that must be fixed before synthesis.\n\n"
        
        # Critical issues
        if results['summary']['total_critical'] > 0:
            report += "## Critical Issues ❌\n\n"
            report += "These issues must be fixed before synthesis:\n\n"
            
            for file_result in results['files']:
                if file_result['critical_issues']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for issue in file_result['critical_issues']:
                        report += f"#### Line {issue['line']}: {issue['type']}\n\n"
                        report += f"**Problem:** {issue['message']}\n\n"
                        report += f"**Fix:** {issue['fix']}\n\n"
                    report += "---\n\n"
        
        # Design flaws
        if results['summary']['total_design_flaws'] > 0:
            report += "## Design Flaws ⚠️\n\n"
            report += "These issues may cause problems in synthesis or operation:\n\n"
            
            for file_result in results['files']:
                if file_result['design_flaws']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for flaw in file_result['design_flaws'][:5]:  # Limit to 5
                        report += f"- **{flaw['type']}** (Line {flaw['line']}): {flaw['message']}\n"
                        report += f"  - *Fix:* {flaw['fix']}\n"
                    if len(file_result['design_flaws']) > 5:
                        report += f"  - ... and {len(file_result['design_flaws']) - 5} more\n"
                    report += "\n"
        
        # Optimizations
        if results['summary']['total_optimizations'] > 0:
            report += "## Optimization Opportunities 💡\n\n"
            
            for file_result in results['files']:
                if file_result['optimizations']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for opt in file_result['optimizations'][:3]:  # Limit to 3
                        report += f"- **Line {opt['line']}**: {opt['message']}\n"
                        report += f"  - *Suggestion:* {opt['fix']}\n"
                    if len(file_result['optimizations']) > 3:
                        report += f"  - ... and {len(file_result['optimizations']) - 3} more\n"
                    report += "\n"
        
        # Structural analysis
        report += "## Design Complexity Analysis\n\n"
        report += "| File | Modules | Always Blocks | Registers | Wires | Complexity |\n"
        report += "|------|---------|---------------|-----------|-------|------------|\n"
        
        for file_result in results['files']:
            a = file_result['analysis']
            modules_str = ', '.join(a['modules']) if a['modules'] else 'None'
            report += f"| {Path(file_result['file']).name} | {modules_str} | "
            report += f"{a['always_blocks']} | {a['registers']} | {a['wires']} | "
            report += f"{a['complexity_score']} |\n"
        
        report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if results['summary']['total_critical'] > 0:
            report += "### Immediate Actions Required\n\n"
            report += "1. Fix all critical issues before attempting synthesis\n"
            report += "2. Review blocking/non-blocking assignment usage\n"
            report += "3. Verify signal width consistency\n\n"
        
        if results['summary']['total_design_flaws'] > 0:
            report += "### Design Improvements\n\n"
            report += "1. Address timing issues and long combinational paths\n"
            report += "2. Complete state machine implementations\n"
            report += "3. Ensure proper reset synchronization\n\n"
        
        if results['summary']['total_optimizations'] > 0:
            report += "### Optimization Suggestions\n\n"
            report += "1. Replace multiplications with shifts where possible\n"
            report += "2. Remove redundant logic\n"
            report += "3. Consider resource sharing opportunities\n\n"
        
        return report

if __name__ == "__main__":
    print("="*60)
    print("Advanced Hardware Design Analysis")
    print("="*60)
    
    analyzer = AdvancedVerilogAnalyzer()
    results = analyzer.analyze_directory(Path("pentary-repo/hardware"))
    
    # Generate report
    report = analyzer.generate_detailed_report(results)
    with open("advanced_hardware_analysis.md", "w") as f:
        f.write(report)
    
    # Save raw results
    with open("advanced_hardware_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Files analyzed: {results['summary']['total_files']}")
    print(f"Critical issues: {results['summary']['total_critical']}")
    print(f"Design flaws: {results['summary']['total_design_flaws']}")
    print(f"Optimizations: {results['summary']['total_optimizations']}")
    print("\nGenerated advanced_hardware_analysis.md")
    print("Saved advanced_hardware_analysis.json")