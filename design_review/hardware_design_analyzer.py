#!/usr/bin/env python3
"""
Comprehensive hardware design analyzer for Verilog files.
Checks for common errors, timing issues, and design flaws.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json

class VerilogAnalyzer:
    """Analyzes Verilog files for design flaws and errors."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    def analyze_file(self, filepath: Path) -> Dict:
        """Analyze a single Verilog file."""
        
        print(f"Analyzing {filepath.name}...")
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        results = {
            'file': str(filepath),
            'issues': [],
            'warnings': [],
            'info': [],
            'metrics': {}
        }
        
        # Run all checks
        results['issues'].extend(self.check_syntax_errors(content, lines))
        results['warnings'].extend(self.check_signal_widths(content, lines))
        results['warnings'].extend(self.check_uninitialized_signals(content, lines))
        results['issues'].extend(self.check_blocking_nonblocking(content, lines))
        results['warnings'].extend(self.check_race_conditions(content, lines))
        results['warnings'].extend(self.check_state_machines(content, lines))
        results['info'].extend(self.check_clock_domains(content, lines))
        results['warnings'].extend(self.check_reset_logic(content, lines))
        results['warnings'].extend(self.check_combinational_loops(content, lines))
        results['metrics'] = self.calculate_metrics(content, lines)
        
        return results
    
    def check_syntax_errors(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for common syntax errors."""
        issues = []
        
        # Check for unmatched begin/end
        begin_count = content.count('begin')
        end_count = content.count('end')
        if begin_count != end_count:
            issues.append({
                'severity': 'ERROR',
                'type': 'syntax',
                'message': f'Unmatched begin/end: {begin_count} begin, {end_count} end',
                'line': 0
            })
        
        # Check for unmatched parentheses
        paren_count = content.count('(') - content.count(')')
        if paren_count != 0:
            issues.append({
                'severity': 'ERROR',
                'type': 'syntax',
                'message': f'Unmatched parentheses: difference of {paren_count}',
                'line': 0
            })
        
        # Check for missing semicolons (common error)
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('//'):
                # Check if line should end with semicolon
                if any(keyword in line for keyword in ['assign', 'wire', 'reg', 'input', 'output']):
                    if not line.endswith(';') and not line.endswith(',') and not line.endswith('('):
                        if 'begin' not in line and 'end' not in line:
                            issues.append({
                                'severity': 'ERROR',
                                'type': 'syntax',
                                'message': 'Possible missing semicolon',
                                'line': i,
                                'code': line[:80]
                            })
        
        return issues
    
    def check_signal_widths(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for signal width mismatches."""
        warnings = []
        
        # Extract signal declarations
        signals = {}
        for i, line in enumerate(lines, 1):
            # Match wire/reg declarations with widths
            match = re.search(r'(wire|reg)\s*\[(\d+):(\d+)\]\s*(\w+)', line)
            if match:
                sig_type, msb, lsb, name = match.groups()
                width = int(msb) - int(lsb) + 1
                signals[name] = {'width': width, 'line': i, 'type': sig_type}
        
        # Check for width mismatches in assignments
        for i, line in enumerate(lines, 1):
            # Check assignments
            if '=' in line and not line.strip().startswith('//'):
                # Extract LHS and RHS
                parts = line.split('=')
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip().rstrip(';')
                    
                    # Check if signals are in our dictionary
                    lhs_name = re.search(r'(\w+)', lhs)
                    if lhs_name and lhs_name.group(1) in signals:
                        lhs_width = signals[lhs_name.group(1)]['width']
                        
                        # Check for literal width mismatches
                        literal_match = re.search(r"(\d+)'[bdh](\w+)", rhs)
                        if literal_match:
                            literal_width = int(literal_match.group(1))
                            if literal_width != lhs_width:
                                warnings.append({
                                    'severity': 'WARNING',
                                    'type': 'width_mismatch',
                                    'message': f'Width mismatch: {lhs_name.group(1)} is {lhs_width}-bit but assigned {literal_width}-bit literal',
                                    'line': i,
                                    'code': line.strip()[:80]
                                })
        
        return warnings
    
    def check_uninitialized_signals(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for potentially uninitialized signals."""
        warnings = []
        
        # Find all reg declarations
        regs = set()
        for line in lines:
            match = re.search(r'reg\s+(?:\[\d+:\d+\]\s+)?(\w+)', line)
            if match:
                regs.add(match.group(1))
        
        # Check if regs are initialized in always blocks
        initialized = set()
        in_always = False
        for i, line in enumerate(lines, 1):
            if 'always' in line:
                in_always = True
            if in_always and '=' in line:
                match = re.search(r'(\w+)\s*[<]?=', line)
                if match:
                    initialized.add(match.group(1))
            if in_always and 'end' in line:
                in_always = False
        
        # Report uninitialized regs
        uninitialized = regs - initialized
        for reg in uninitialized:
            warnings.append({
                'severity': 'WARNING',
                'type': 'uninitialized',
                'message': f'Register "{reg}" may not be initialized',
                'line': 0
            })
        
        return warnings
    
    def check_blocking_nonblocking(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for improper mixing of blocking and non-blocking assignments."""
        issues = []
        
        in_always_comb = False
        in_always_seq = False
        
        for i, line in enumerate(lines, 1):
            # Detect always block type
            if 'always @' in line or 'always_comb' in line:
                if 'posedge' in line or 'negedge' in line:
                    in_always_seq = True
                    in_always_comb = False
                else:
                    in_always_comb = True
                    in_always_seq = False
            
            # Check assignments
            if in_always_seq and '=' in line and '<=' not in line:
                if not line.strip().startswith('//'):
                    issues.append({
                        'severity': 'ERROR',
                        'type': 'blocking_in_sequential',
                        'message': 'Blocking assignment (=) in sequential always block, should use non-blocking (<=)',
                        'line': i,
                        'code': line.strip()[:80]
                    })
            
            if in_always_comb and '<=' in line:
                if not line.strip().startswith('//'):
                    issues.append({
                        'severity': 'ERROR',
                        'type': 'nonblocking_in_combinational',
                        'message': 'Non-blocking assignment (<=) in combinational always block, should use blocking (=)',
                        'line': i,
                        'code': line.strip()[:80]
                    })
            
            if 'end' in line:
                in_always_comb = False
                in_always_seq = False
        
        return issues
    
    def check_race_conditions(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for potential race conditions."""
        warnings = []
        
        # Check for multiple drivers
        drivers = {}
        for i, line in enumerate(lines, 1):
            if '=' in line or '<=' in line:
                match = re.search(r'(\w+)\s*[<]?=', line)
                if match:
                    signal = match.group(1)
                    if signal in drivers:
                        drivers[signal].append(i)
                    else:
                        drivers[signal] = [i]
        
        # Report signals with multiple drivers
        for signal, lines_list in drivers.items():
            if len(lines_list) > 1:
                warnings.append({
                    'severity': 'WARNING',
                    'type': 'multiple_drivers',
                    'message': f'Signal "{signal}" has multiple drivers at lines: {lines_list}',
                    'line': lines_list[0]
                })
        
        return warnings
    
    def check_state_machines(self, content: str, lines: List[str]) -> List[Dict]:
        """Check state machine implementations."""
        warnings = []
        
        # Look for state machine patterns
        has_state_reg = False
        has_next_state = False
        has_state_transition = False
        
        for line in lines:
            if 'state' in line.lower() and 'reg' in line:
                has_state_reg = True
            if 'next_state' in line.lower():
                has_next_state = True
            if 'state <=' in line.lower() or 'state=' in line.lower():
                has_state_transition = True
        
        # Check for proper state machine structure
        if has_state_reg:
            if not has_next_state:
                warnings.append({
                    'severity': 'WARNING',
                    'type': 'state_machine',
                    'message': 'State machine found but no next_state logic detected',
                    'line': 0
                })
            if not has_state_transition:
                warnings.append({
                    'severity': 'WARNING',
                    'type': 'state_machine',
                    'message': 'State machine found but no state transitions detected',
                    'line': 0
                })
        
        return warnings
    
    def check_clock_domains(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for clock domain crossing issues."""
        info = []
        
        # Find all clock signals
        clocks = set()
        for line in lines:
            if 'posedge' in line or 'negedge' in line:
                match = re.search(r'(posedge|negedge)\s+(\w+)', line)
                if match:
                    clocks.add(match.group(2))
        
        if len(clocks) > 1:
            info.append({
                'severity': 'INFO',
                'type': 'clock_domains',
                'message': f'Multiple clock domains detected: {", ".join(clocks)}',
                'line': 0
            })
        
        return info
    
    def check_reset_logic(self, content: str, lines: List[str]) -> List[Dict]:
        """Check reset logic implementation."""
        warnings = []
        
        has_reset = False
        has_async_reset = False
        has_sync_reset = False
        
        for line in lines:
            if 'reset' in line.lower() or 'rst' in line.lower():
                has_reset = True
                if 'posedge' in line or 'negedge' in line:
                    if 'or' in line:
                        has_async_reset = True
                    else:
                        has_sync_reset = True
        
        if not has_reset:
            warnings.append({
                'severity': 'WARNING',
                'type': 'reset',
                'message': 'No reset signal detected in module',
                'line': 0
            })
        
        if has_async_reset and has_sync_reset:
            warnings.append({
                'severity': 'WARNING',
                'type': 'reset',
                'message': 'Mixed async and sync reset detected',
                'line': 0
            })
        
        return warnings
    
    def check_combinational_loops(self, content: str, lines: List[str]) -> List[Dict]:
        """Check for potential combinational loops."""
        warnings = []
        
        # Build dependency graph
        dependencies = {}
        for i, line in enumerate(lines, 1):
            if 'assign' in line or ('=' in line and 'always' not in line):
                match = re.search(r'(\w+)\s*=\s*(.+);', line)
                if match:
                    lhs = match.group(1)
                    rhs = match.group(2)
                    # Extract signals from RHS
                    rhs_signals = re.findall(r'\b(\w+)\b', rhs)
                    dependencies[lhs] = rhs_signals
        
        # Check for self-dependencies (simple loop detection)
        for signal, deps in dependencies.items():
            if signal in deps:
                warnings.append({
                    'severity': 'WARNING',
                    'type': 'combinational_loop',
                    'message': f'Potential combinational loop: signal "{signal}" depends on itself',
                    'line': 0
                })
        
        return warnings
    
    def calculate_metrics(self, content: str, lines: List[str]) -> Dict:
        """Calculate design metrics."""
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
            'comment_lines': len([l for l in lines if l.strip().startswith('//')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'modules': content.count('module '),
            'always_blocks': content.count('always'),
            'assign_statements': content.count('assign'),
            'registers': len(re.findall(r'\breg\b', content)),
            'wires': len(re.findall(r'\bwire\b', content)),
        }
        
        return metrics
    
    def analyze_directory(self, directory: Path) -> Dict:
        """Analyze all Verilog files in directory."""
        
        results = {
            'files': [],
            'summary': {
                'total_files': 0,
                'total_issues': 0,
                'total_warnings': 0,
                'total_info': 0
            }
        }
        
        verilog_files = list(directory.glob('*.v'))
        results['summary']['total_files'] = len(verilog_files)
        
        for vfile in verilog_files:
            file_results = self.analyze_file(vfile)
            results['files'].append(file_results)
            
            results['summary']['total_issues'] += len(file_results['issues'])
            results['summary']['total_warnings'] += len(file_results['warnings'])
            results['summary']['total_info'] += len(file_results['info'])
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate markdown report."""
        
        report = "# Hardware Design Analysis Report\n\n"
        report += "## Summary\n\n"
        report += f"- **Total Files Analyzed:** {results['summary']['total_files']}\n"
        report += f"- **Total Issues:** {results['summary']['total_issues']}\n"
        report += f"- **Total Warnings:** {results['summary']['total_warnings']}\n"
        report += f"- **Total Info:** {results['summary']['total_info']}\n\n"
        
        # Issues by severity
        if results['summary']['total_issues'] > 0:
            report += "## Critical Issues ❌\n\n"
            for file_result in results['files']:
                if file_result['issues']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for issue in file_result['issues']:
                        report += f"- **Line {issue['line']}**: {issue['message']}\n"
                        if 'code' in issue:
                            report += f"  ```verilog\n  {issue['code']}\n  ```\n"
                    report += "\n"
        
        # Warnings
        if results['summary']['total_warnings'] > 0:
            report += "## Warnings ⚠️\n\n"
            for file_result in results['files']:
                if file_result['warnings']:
                    report += f"### {Path(file_result['file']).name}\n\n"
                    for warning in file_result['warnings'][:10]:  # Limit to first 10
                        report += f"- **Line {warning['line']}**: {warning['message']}\n"
                        if 'code' in warning:
                            report += f"  ```verilog\n  {warning['code']}\n  ```\n"
                    if len(file_result['warnings']) > 10:
                        report += f"  ... and {len(file_result['warnings']) - 10} more warnings\n"
                    report += "\n"
        
        # Metrics
        report += "## Design Metrics\n\n"
        report += "| File | Lines | Modules | Always Blocks | Registers | Wires |\n"
        report += "|------|-------|---------|---------------|-----------|-------|\n"
        
        for file_result in results['files']:
            m = file_result['metrics']
            report += f"| {Path(file_result['file']).name} | {m['code_lines']} | {m['modules']} | "
            report += f"{m['always_blocks']} | {m['registers']} | {m['wires']} |\n"
        
        return report

if __name__ == "__main__":
    print("Starting hardware design analysis...")
    
    analyzer = VerilogAnalyzer()
    results = analyzer.analyze_directory(Path("pentary-repo/hardware"))
    
    # Generate report
    report = analyzer.generate_report(results)
    with open("hardware_analysis_report.md", "w") as f:
        f.write(report)
    
    # Save raw results
    with open("hardware_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"Files analyzed: {results['summary']['total_files']}")
    print(f"Issues found: {results['summary']['total_issues']}")
    print(f"Warnings found: {results['summary']['total_warnings']}")
    print(f"\nGenerated hardware_analysis_report.md")
    print(f"Saved hardware_analysis_results.json")