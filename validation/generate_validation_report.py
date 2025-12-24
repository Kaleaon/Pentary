#!/usr/bin/env python3
"""
Pentary Processor - Comprehensive Validation Report Generator
Analyzes all validation results and generates a detailed report
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

class ValidationReportGenerator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'synthesis': {},
            'testbenches': {},
            'simulations': {},
            'overall': {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
    def parse_synthesis_results(self, results_dir):
        """Parse Yosys synthesis results"""
        print("Parsing synthesis results...")
        
        synthesis_dir = Path(results_dir) / 'synthesis_results'
        if not synthesis_dir.exists():
            print(f"  Warning: {synthesis_dir} not found")
            return
            
        for log_file in synthesis_dir.glob('*_synth.log'):
            module_name = log_file.stem.replace('_synth', '')
            
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract statistics
            stats = {}
            
            # Look for cell count
            cell_match = re.search(r'Number of cells:\s+(\d+)', content)
            if cell_match:
                stats['cells'] = int(cell_match.group(1))
                
            # Look for wire count
            wire_match = re.search(r'Number of wires:\s+(\d+)', content)
            if wire_match:
                stats['wires'] = int(wire_match.group(1))
                
            # Check for errors
            has_error = 'ERROR' in content or 'Error' in content
            has_warning = 'WARNING' in content or 'Warning' in content
            
            self.results['synthesis'][module_name] = {
                'status': 'FAILED' if has_error else 'PASSED',
                'statistics': stats,
                'warnings': has_warning,
                'log_file': str(log_file)
            }
            
            self.results['overall']['total_checks'] += 1
            if has_error:
                self.results['overall']['failed'] += 1
            else:
                self.results['overall']['passed'] += 1
            if has_warning:
                self.results['overall']['warnings'] += 1
                
    def parse_testbench_results(self, results_dir):
        """Parse testbench execution results"""
        print("Parsing testbench results...")
        
        test_dir = Path(results_dir) / 'test_results'
        if not test_dir.exists():
            print(f"  Warning: {test_dir} not found")
            return
            
        for log_file in test_dir.glob('*_run.log'):
            tb_name = log_file.stem.replace('_run', '')
            
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Check for pass/fail
            passed = any(keyword in content for keyword in ['PASS', 'SUCCESS', 'All tests passed'])
            failed = any(keyword in content for keyword in ['FAIL', 'ERROR'])
            
            # Count test cases
            test_count = len(re.findall(r'Test \d+:', content))
            pass_count = len(re.findall(r'PASS', content))
            fail_count = len(re.findall(r'FAIL', content))
            
            self.results['testbenches'][tb_name] = {
                'status': 'FAILED' if failed else ('PASSED' if passed else 'UNKNOWN'),
                'total_tests': test_count,
                'passed_tests': pass_count,
                'failed_tests': fail_count,
                'log_file': str(log_file)
            }
            
            self.results['overall']['total_checks'] += 1
            if failed:
                self.results['overall']['failed'] += 1
            elif passed:
                self.results['overall']['passed'] += 1
                
    def parse_spice_results(self, sim_dir):
        """Parse SPICE simulation results"""
        print("Parsing SPICE simulation results...")
        
        sim_path = Path(sim_dir)
        if not sim_path.exists():
            print(f"  Warning: {sim_path} not found")
            return
            
        for spice_file in sim_path.glob('*.spice'):
            sim_name = spice_file.stem
            
            # Look for corresponding results
            result_file = sim_path / f"{sim_name}_results.raw"
            log_file = sim_path / f"{sim_name}.log"
            
            status = 'NOT_RUN'
            if result_file.exists():
                status = 'COMPLETED'
            elif log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                    if 'Error' in content or 'ERROR' in content:
                        status = 'FAILED'
                    else:
                        status = 'COMPLETED'
                        
            self.results['simulations'][sim_name] = {
                'status': status,
                'spice_file': str(spice_file),
                'result_file': str(result_file) if result_file.exists() else None
            }
            
            self.results['overall']['total_checks'] += 1
            if status == 'COMPLETED':
                self.results['overall']['passed'] += 1
            elif status == 'FAILED':
                self.results['overall']['failed'] += 1
                
    def generate_markdown_report(self, output_file):
        """Generate markdown validation report"""
        print(f"Generating markdown report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Pentary Processor - Validation Report\n\n")
            f.write(f"**Generated**: {self.results['timestamp']}\n\n")
            f.write("---\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            overall = self.results['overall']
            pass_rate = (overall['passed'] / overall['total_checks'] * 100) if overall['total_checks'] > 0 else 0
            
            f.write(f"- **Total Checks**: {overall['total_checks']}\n")
            f.write(f"- **Passed**: {overall['passed']} ✅\n")
            f.write(f"- **Failed**: {overall['failed']} ❌\n")
            f.write(f"- **Warnings**: {overall['warnings']} ⚠️\n")
            f.write(f"- **Pass Rate**: {pass_rate:.1f}%\n\n")
            
            if overall['failed'] == 0:
                f.write("🎉 **All validation checks passed!**\n\n")
            else:
                f.write(f"⚠️ **{overall['failed']} checks failed - review required**\n\n")
                
            f.write("---\n\n")
            
            # Synthesis results
            f.write("## Synthesis Results\n\n")
            if self.results['synthesis']:
                f.write("| Module | Status | Cells | Wires | Warnings |\n")
                f.write("|--------|--------|-------|-------|----------|\n")
                
                for module, data in sorted(self.results['synthesis'].items()):
                    status_icon = "✅" if data['status'] == 'PASSED' else "❌"
                    cells = data['statistics'].get('cells', 'N/A')
                    wires = data['statistics'].get('wires', 'N/A')
                    warnings = "⚠️" if data['warnings'] else ""
                    
                    f.write(f"| {module} | {status_icon} {data['status']} | {cells} | {wires} | {warnings} |\n")
            else:
                f.write("*No synthesis results found*\n")
                
            f.write("\n---\n\n")
            
            # Testbench results
            f.write("## Testbench Results\n\n")
            if self.results['testbenches']:
                f.write("| Testbench | Status | Total | Passed | Failed |\n")
                f.write("|-----------|--------|-------|--------|--------|\n")
                
                for tb, data in sorted(self.results['testbenches'].items()):
                    status_icon = "✅" if data['status'] == 'PASSED' else "❌"
                    
                    f.write(f"| {tb} | {status_icon} {data['status']} | "
                           f"{data['total_tests']} | {data['passed_tests']} | {data['failed_tests']} |\n")
            else:
                f.write("*No testbench results found*\n")
                
            f.write("\n---\n\n")
            
            # SPICE simulation results
            f.write("## SPICE Simulation Results\n\n")
            if self.results['simulations']:
                f.write("| Simulation | Status |\n")
                f.write("|------------|--------|\n")
                
                for sim, data in sorted(self.results['simulations'].items()):
                    status_icon = "✅" if data['status'] == 'COMPLETED' else ("❌" if data['status'] == 'FAILED' else "⏳")
                    
                    f.write(f"| {sim} | {status_icon} {data['status']} |\n")
            else:
                f.write("*No SPICE simulation results found*\n")
                
            f.write("\n---\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if overall['failed'] > 0:
                f.write("### Critical Issues\n\n")
                
                # List failed synthesis
                failed_synth = [m for m, d in self.results['synthesis'].items() if d['status'] == 'FAILED']
                if failed_synth:
                    f.write("**Failed Synthesis:**\n")
                    for module in failed_synth:
                        f.write(f"- {module}: Review synthesis log for errors\n")
                    f.write("\n")
                    
                # List failed testbenches
                failed_tests = [t for t, d in self.results['testbenches'].items() if d['status'] == 'FAILED']
                if failed_tests:
                    f.write("**Failed Testbenches:**\n")
                    for tb in failed_tests:
                        f.write(f"- {tb}: Review test log for failures\n")
                    f.write("\n")
                    
            if overall['warnings'] > 0:
                f.write("### Warnings\n\n")
                f.write(f"- {overall['warnings']} modules have synthesis warnings\n")
                f.write("- Review warnings to ensure they are not critical\n\n")
                
            f.write("### Next Steps\n\n")
            if overall['failed'] == 0:
                f.write("1. ✅ All validation checks passed\n")
                f.write("2. Proceed with FPGA prototyping\n")
                f.write("3. Begin software toolchain development\n")
                f.write("4. Prepare for tape-out\n")
            else:
                f.write("1. Fix all failed synthesis and testbenches\n")
                f.write("2. Re-run validation suite\n")
                f.write("3. Address all warnings\n")
                f.write("4. Perform additional verification\n")
                
            f.write("\n---\n\n")
            f.write("*End of Validation Report*\n")
            
    def generate_json_report(self, output_file):
        """Generate JSON validation report"""
        print(f"Generating JSON report: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def run(self, validation_dir, output_dir):
        """Run complete validation report generation"""
        print("=" * 60)
        print("Pentary Processor - Validation Report Generator")
        print("=" * 60)
        print()
        
        # Parse all results
        self.parse_synthesis_results(validation_dir)
        self.parse_testbench_results(validation_dir)
        self.parse_spice_results(Path(validation_dir).parent / 'simulations')
        
        # Generate reports
        os.makedirs(output_dir, exist_ok=True)
        
        md_report = Path(output_dir) / 'VALIDATION_REPORT.md'
        json_report = Path(output_dir) / 'validation_results.json'
        
        self.generate_markdown_report(md_report)
        self.generate_json_report(json_report)
        
        print()
        print("=" * 60)
        print("Report Generation Complete")
        print("=" * 60)
        print()
        print(f"Markdown report: {md_report}")
        print(f"JSON report: {json_report}")
        print()
        
        # Print summary
        overall = self.results['overall']
        print(f"Total checks: {overall['total_checks']}")
        print(f"Passed: {overall['passed']} ✅")
        print(f"Failed: {overall['failed']} ❌")
        print(f"Warnings: {overall['warnings']} ⚠️")
        
        if overall['failed'] == 0:
            print()
            print("🎉 All validation checks passed!")
            return 0
        else:
            print()
            print(f"⚠️ {overall['failed']} checks failed - review required")
            return 1

if __name__ == '__main__':
    import sys
    
    # Default paths
    validation_dir = 'validation'
    output_dir = 'validation/reports'
    
    # Allow command-line override
    if len(sys.argv) > 1:
        validation_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    generator = ValidationReportGenerator()
    exit_code = generator.run(validation_dir, output_dir)
    sys.exit(exit_code)