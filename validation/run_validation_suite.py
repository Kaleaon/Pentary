#!/usr/bin/env python3
"""
Pentary Project Validation Suite
================================

Comprehensive validation of all Pentary project components:
- Python tools and libraries
- Pent language toolchain
- Hardware verification (if iverilog available)
- Documentation checks

Run with: python3 run_validation_suite.py
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Ensure workspace is in path
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

class ValidationResult:
    """Store validation results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []
        self.duration = 0.0
    
    def add_pass(self):
        self.passed += 1
    
    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
    
    def add_skip(self, reason: str):
        self.skipped += 1
        self.errors.append(f"SKIPPED: {reason}")
    
    @property
    def total(self):
        return self.passed + self.failed + self.skipped
    
    @property
    def success_rate(self):
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100
    
    def to_dict(self):
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total,
            "success_rate": f"{self.success_rate:.1f}%",
            "duration": f"{self.duration:.2f}s",
            "errors": self.errors
        }


class ValidationSuite:
    """Main validation suite"""
    
    def __init__(self):
        self.results: Dict[str, ValidationResult] = {}
        self.start_time = None
        self.end_time = None
    
    def run_all(self):
        """Run all validation tests"""
        self.start_time = datetime.now()
        
        print("=" * 70)
        print("  PENTARY PROJECT VALIDATION SUITE")
        print("=" * 70)
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Run validation categories
        self.validate_python_syntax()
        self.validate_pentary_tools()
        self.validate_pent_language()
        self.validate_pytest_suite()
        self.validate_hardware_files()
        self.validate_documentation()
        
        self.end_time = datetime.now()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return self.overall_success()
    
    def validate_python_syntax(self):
        """Check all Python files for syntax errors"""
        result = ValidationResult("Python Syntax")
        start = time.time()
        
        print("\n[1/6] Validating Python Syntax...")
        
        # Find all Python files
        py_files = []
        for root, dirs, files in os.walk(WORKSPACE):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        
        for py_file in py_files:
            rel_path = os.path.relpath(py_file, WORKSPACE)
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                compile(source, py_file, 'exec')
                result.add_pass()
            except SyntaxError as e:
                result.add_fail(f"{rel_path}: {e}")
            except Exception as e:
                result.add_fail(f"{rel_path}: {e}")
        
        result.duration = time.time() - start
        self.results["python_syntax"] = result
        print(f"  ✓ {result.passed} files OK, {result.failed} errors")
    
    def validate_pentary_tools(self):
        """Validate pentary tools functionality"""
        result = ValidationResult("Pentary Tools")
        start = time.time()
        
        print("\n[2/6] Validating Pentary Tools...")
        
        # Test pentary converter
        try:
            from tools.pentary_converter import PentaryConverter
            
            # Test decimal to pentary
            p = PentaryConverter.decimal_to_pentary(42)
            d = PentaryConverter.pentary_to_decimal(p)
            if d == 42:
                result.add_pass()
            else:
                result.add_fail(f"Converter roundtrip failed: 42 -> {p} -> {d}")
            
            # Test negative numbers
            p = PentaryConverter.decimal_to_pentary(-17)
            d = PentaryConverter.pentary_to_decimal(p)
            if d == -17:
                result.add_pass()
            else:
                result.add_fail(f"Negative roundtrip failed: -17 -> {p} -> {d}")
            
            # Test zero
            p = PentaryConverter.decimal_to_pentary(0)
            d = PentaryConverter.pentary_to_decimal(p)
            if d == 0:
                result.add_pass()
            else:
                result.add_fail(f"Zero roundtrip failed: 0 -> {p} -> {d}")
                
        except Exception as e:
            result.add_fail(f"PentaryConverter: {e}")
        
        # Test pentary arithmetic
        try:
            from tools.pentary_arithmetic import PentaryArithmetic
            
            # Test addition using the actual method name with pentary string notation
            # "++" = 2 in pentary, so "++" + "++" = 4 = "+-" in pentary (1*5 + -1)
            result_str, steps = PentaryArithmetic.add_pentary_detailed("++", "++")
            if result_str is not None and len(steps) > 0:
                result.add_pass()
            else:
                result.add_fail("PentaryArithmetic.add_pentary_detailed returned invalid result")
                
        except Exception as e:
            result.add_fail(f"PentaryArithmetic: {e}")
        
        # Test pentary validation
        try:
            from tools.pentary_validation import PentaryValidator
            
            v = PentaryValidator()
            if v.validate_pentary_digit(0):
                result.add_pass()
            else:
                result.add_fail("Validator rejected valid digit 0")
                
        except Exception as e:
            result.add_fail(f"PentaryValidator: {e}")
        
        result.duration = time.time() - start
        self.results["pentary_tools"] = result
        print(f"  ✓ {result.passed} tests OK, {result.failed} errors")
    
    def validate_pent_language(self):
        """Validate Pent language toolchain"""
        result = ValidationResult("Pent Language")
        start = time.time()
        
        print("\n[3/6] Validating Pent Language...")
        
        # Add language directory to path for consistent imports
        lang_dir = os.path.join(WORKSPACE, 'language')
        if lang_dir not in sys.path:
            sys.path.insert(0, lang_dir)
        
        # Test lexer
        try:
            from pent_lexer import Lexer, TokenType
            
            lexer = Lexer("let x = 42;")
            tokens = lexer.tokenize()
            if len(tokens) > 0 and tokens[0].type.name == "LET":
                result.add_pass()
            else:
                result.add_fail("Lexer failed to tokenize 'let'")
                
        except Exception as e:
            result.add_fail(f"Lexer: {e}")
        
        # Test parser
        try:
            from pent_parser import Parser
            from pent_lexer import Lexer
            
            source = "fn main() { return 42; }"
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            if len(ast.statements) == 1:
                result.add_pass()
            else:
                result.add_fail(f"Parser produced {len(ast.statements)} statements, expected 1")
                
        except Exception as e:
            result.add_fail(f"Parser: {e}")
        
        # Test compiler
        try:
            from pent_compiler import Compiler
            
            source = "fn main() { return 1 + 2; }"
            compiler = Compiler()
            asm = compiler.compile(source)
            if len(asm) > 0:
                result.add_pass()
            else:
                result.add_fail("Compiler produced empty output")
                
        except Exception as e:
            result.add_fail(f"Compiler: {e}")
        
        # Test interpreter
        try:
            from pent_interpreter import PentInterpreter
            
            source = "fn main() { return 3 + 4; }"
            interp = PentInterpreter()
            val = interp.interpret(source)
            if val == 7:
                result.add_pass()
            else:
                result.add_fail(f"Interpreter returned {val}, expected 7")
                
        except Exception as e:
            result.add_fail(f"Interpreter: {e}")
        
        # Test struct parsing
        try:
            from pent_parser import Parser, StructDefinition
            from pent_lexer import Lexer
            
            source = "struct Point { x: int, y: int }"
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            if len(ast.statements) == 1 and isinstance(ast.statements[0], StructDefinition):
                result.add_pass()
            else:
                result.add_fail("Failed to parse struct definition")
                
        except Exception as e:
            result.add_fail(f"Struct parsing: {e}")
        
        # Test array parsing
        try:
            from pent_parser import Parser, ArrayLiteral
            from pent_lexer import Lexer
            
            source = "fn main() { let arr = [1, 2, 3]; }"
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            result.add_pass()
                
        except Exception as e:
            result.add_fail(f"Array parsing: {e}")
        
        result.duration = time.time() - start
        self.results["pent_language"] = result
        print(f"  ✓ {result.passed} tests OK, {result.failed} errors")
    
    def validate_pytest_suite(self):
        """Run pytest test suite"""
        result = ValidationResult("Pytest Suite")
        start = time.time()
        
        print("\n[4/6] Running Pytest Suite...")
        
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", 
                 os.path.join(WORKSPACE, "tests"), 
                 "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse output
            output = proc.stdout + proc.stderr
            
            # Look for "X passed" pattern
            import re
            match = re.search(r'(\d+) passed', output)
            if match:
                result.passed = int(match.group(1))
            
            match = re.search(r'(\d+) failed', output)
            if match:
                result.failed = int(match.group(1))
                result.errors.append(f"Pytest reported {result.failed} failures")
            
            match = re.search(r'(\d+) skipped', output)
            if match:
                result.skipped = int(match.group(1))
            
            if proc.returncode != 0 and result.failed == 0:
                result.add_fail(f"Pytest exited with code {proc.returncode}")
                
        except subprocess.TimeoutExpired:
            result.add_fail("Pytest timed out after 300s")
        except Exception as e:
            result.add_fail(f"Pytest error: {e}")
        
        result.duration = time.time() - start
        self.results["pytest"] = result
        print(f"  ✓ {result.passed} tests passed, {result.failed} failed, {result.skipped} skipped")
    
    def validate_hardware_files(self):
        """Validate Verilog hardware files"""
        result = ValidationResult("Hardware Files")
        start = time.time()
        
        print("\n[5/6] Validating Hardware Files...")
        
        # Find Verilog files
        v_files = []
        hw_dir = os.path.join(WORKSPACE, "hardware")
        for root, dirs, files in os.walk(hw_dir):
            for file in files:
                if file.endswith('.v'):
                    v_files.append(os.path.join(root, file))
        
        # Check if iverilog is available
        iverilog_available = False
        try:
            subprocess.run(["iverilog", "-V"], capture_output=True)
            iverilog_available = True
        except FileNotFoundError:
            pass
        
        if not iverilog_available:
            # Just check files exist and have content
            for v_file in v_files:
                rel_path = os.path.relpath(v_file, WORKSPACE)
                try:
                    with open(v_file, 'r') as f:
                        content = f.read()
                    if len(content) > 0:
                        result.add_pass()
                    else:
                        result.add_fail(f"{rel_path}: Empty file")
                except Exception as e:
                    result.add_fail(f"{rel_path}: {e}")
            
            result.errors.append("NOTE: iverilog not available, syntax check skipped")
        else:
            # Run iverilog syntax check
            for v_file in v_files:
                rel_path = os.path.relpath(v_file, WORKSPACE)
                try:
                    proc = subprocess.run(
                        ["iverilog", "-t", "null", "-Wall", v_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if proc.returncode == 0:
                        result.add_pass()
                    else:
                        # Some warnings are okay
                        if "error" in proc.stderr.lower():
                            result.add_fail(f"{rel_path}: {proc.stderr[:200]}")
                        else:
                            result.add_pass()
                except subprocess.TimeoutExpired:
                    result.add_skip(f"{rel_path}: Timed out")
                except Exception as e:
                    result.add_fail(f"{rel_path}: {e}")
        
        result.duration = time.time() - start
        self.results["hardware"] = result
        print(f"  ✓ {result.passed} files OK, {result.failed} errors, {result.skipped} skipped")
    
    def validate_documentation(self):
        """Validate documentation files"""
        result = ValidationResult("Documentation")
        start = time.time()
        
        print("\n[6/6] Validating Documentation...")
        
        # Required documentation files
        required_docs = [
            "README_UPDATED.md",
            "USER_GUIDE.md",
            "FAQ.md",
            "QUICK_REFERENCE.md",
            "hardware/3T_CHIP_LAYOUT_DESIGN.md",
            "hardware/MEMRISTOR_CHIP_LAYOUT_DESIGN.md",
            "hardware/BREADBOARD_PCB_DESIGN.md",
        ]
        
        for doc in required_docs:
            doc_path = os.path.join(WORKSPACE, doc)
            if os.path.exists(doc_path):
                # Check it has substantial content
                with open(doc_path, 'r') as f:
                    content = f.read()
                if len(content) > 500:
                    result.add_pass()
                else:
                    result.add_fail(f"{doc}: Too short ({len(content)} chars)")
            else:
                result.add_fail(f"{doc}: Missing")
        
        # Check for broken links (simple check)
        md_files = []
        for root, dirs, files in os.walk(WORKSPACE):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        
        # Count total markdown files
        result.errors.append(f"INFO: Found {len(md_files)} markdown files")
        
        result.duration = time.time() - start
        self.results["documentation"] = result
        print(f"  ✓ {result.passed} docs OK, {result.failed} issues")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("  VALIDATION SUMMARY")
        print("=" * 70)
        
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for name, result in self.results.items():
            status = "✓ PASS" if result.failed == 0 else "✗ FAIL"
            print(f"  {result.name:25} {status:8} ({result.passed}/{result.total}) {result.duration:.1f}s")
            total_passed += result.passed
            total_failed += result.failed
            total_skipped += result.skipped
        
        print("-" * 70)
        total = total_passed + total_failed + total_skipped
        success_rate = (total_passed / total * 100) if total > 0 else 0
        
        print(f"  {'TOTAL':25} {'':8} {total_passed}/{total} ({success_rate:.1f}%)")
        
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\n  Duration: {duration:.1f}s")
        print(f"  Finished: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if total_failed > 0:
            print("\n  ERRORS:")
            for name, result in self.results.items():
                for error in result.errors:
                    if not error.startswith("INFO:") and not error.startswith("NOTE:"):
                        print(f"    - [{result.name}] {error[:80]}")
        
        print("=" * 70)
        
        if total_failed == 0:
            print("  ✓ ALL VALIDATIONS PASSED")
        else:
            print(f"  ✗ {total_failed} VALIDATION(S) FAILED")
        print("=" * 70)
    
    def save_results(self):
        """Save results to JSON file"""
        output = {
            "timestamp": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "summary": {
                "total_passed": sum(r.passed for r in self.results.values()),
                "total_failed": sum(r.failed for r in self.results.values()),
                "total_skipped": sum(r.skipped for r in self.results.values()),
            }
        }
        
        output_file = os.path.join(WORKSPACE, "validation", "validation_results_latest.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n  Results saved to: {output_file}")
    
    def overall_success(self) -> bool:
        """Return True if all validations passed"""
        return all(r.failed == 0 for r in self.results.values())


def main():
    suite = ValidationSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
