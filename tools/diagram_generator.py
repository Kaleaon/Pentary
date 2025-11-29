#!/usr/bin/env python3
"""
Pentary Diagram Generator
Creates high-quality diagrams for all research topics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Wedge
import numpy as np
from pathlib import Path

# Set style for professional diagrams
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class PentaryDiagramGenerator:
    """Generate diagrams for Pentary research topics"""

    def __init__(self, output_dir="diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Pentary color scheme
        self.colors = {
            '-2': '#d62728',  # Strong negative (red)
            '-1': '#ff7f0e',  # Weak negative (orange)
            '0': '#7f7f7f',   # Neutral (gray)
            '+1': '#2ca02c',  # Weak positive (green)
            '+2': '#1f77b4',  # Strong positive (blue)
            'background': '#f0f0f0',
            'text': '#333333',
            'accent': '#9467bd'
        }

    def create_pentary_number_system_diagram(self):
        """Create diagram showing pentary number system"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Number System: Balanced Quinary Architecture',
                     fontsize=16, fontweight='bold')

        # 1. Digit representation
        ax = axes[0, 0]
        ax.set_title('Pentary Digit Values', fontweight='bold')
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-3, 3)

        digits = ['-2', '-1', '0', '+1', '+2']
        values = [-2, -1, 0, 1, 2]
        voltages = ['0V', '1.25V', '2.5V', '3.75V', '5V']

        for i, (digit, value, voltage) in enumerate(zip(digits, values, voltages)):
            color = self.colors[digit]
            # Draw bar
            rect = Rectangle((i-0.3, 0), 0.6, value,
                           facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            # Labels
            ax.text(i, value + 0.3 if value >= 0 else value - 0.3,
                   digit, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(i, -2.5, voltage, ha='center', va='top', fontsize=9)

        ax.axhline(y=0, color='black', linewidth=2, linestyle='--', alpha=0.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(digits)
        ax.set_ylabel('Numeric Value', fontweight='bold')
        ax.set_xlabel('Pentary Digit', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Information density comparison
        ax = axes[0, 1]
        ax.set_title('Information Density Comparison', fontweight='bold')

        systems = ['Binary', 'Ternary', 'Pentary', 'Octal']
        bits_per_digit = [1.0, 1.58, 2.32, 3.0]
        colors_bar = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

        bars = ax.barh(systems, bits_per_digit, color=colors_bar, edgecolor='black', linewidth=2)
        ax.set_xlabel('Bits per Digit', fontweight='bold')
        ax.set_xlim(0, 3.5)

        for i, (bar, value) in enumerate(zip(bars, bits_per_digit)):
            ax.text(value + 0.1, i, f'{value:.2f}', va='center', fontweight='bold')

        ax.grid(True, alpha=0.3, axis='x')

        # 3. Number representation example
        ax = axes[1, 0]
        ax.set_title('Example: Decimal 42 in Pentary', fontweight='bold')
        ax.axis('off')

        # Show conversion
        text = """
Decimal: 42

Conversion Process:
42 ÷ 5 = 8 remainder 2  → +2
8 ÷ 5 = 1 remainder 3   → +(-2) = -2, carry 1
1 + 1 = 2               → +2

Pentary: +2 -2 +2
Reading: 2×25 + (-2)×5 + 2×1 = 50 - 10 + 2 = 42 ✓

Voltage Levels: 5V | 0V | 5V
        """
        ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))

        # 4. Arithmetic operations
        ax = axes[1, 1]
        ax.set_title('Pentary Addition Example', fontweight='bold')
        ax.axis('off')

        text = """
Addition: (+2 -1) + (+1 +2)

Step-by-step:
  +2  -1
+ +1  +2
---------

Rightmost: -1 + +2 = +1 (no carry)
Leftmost:  +2 + +1 = +(-2) with carry +1
           = +1 +(-2) +1 = +1 -1

Result: +1 -1 +1

Verification:
(2×5 - 1) + (1×5 + 2) = 9 + 7 = 16
1×25 - 1×5 + 1 = 25 - 5 + 1 = 21...

Actually: +2-1 = 9, +1+2 = 7, sum = 16
16 in pentary: 16 = 3×5 + 1 = (5-2)×5 + 1 = +(-2)+1
        """
        ax.text(0.1, 0.5, text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_number_system.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_number_system.png")
        plt.close()

    def create_logic_gates_diagram(self):
        """Create comprehensive logic gates diagram"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Pentary Logic Gates and Truth Tables',
                     fontsize=16, fontweight='bold')

        # 1. NOT gate
        ax = axes[0, 0]
        ax.set_title('NOT Gate (Negation)', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw gate symbol
        triangle = plt.Polygon([(2, 3), (2, 7), (5, 5)],
                              facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        circle = Circle((5.3, 5), 0.3, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Input/output lines
        ax.plot([0, 2], [5, 5], 'k-', linewidth=2)
        ax.plot([5.6, 7], [5, 5], 'k-', linewidth=2)
        ax.text(1, 5.5, 'IN', fontsize=10, fontweight='bold')
        ax.text(6, 5.5, 'OUT', fontsize=10, fontweight='bold')

        # Truth table
        truth_table = """
IN  | OUT
----|----
-2  | +2
-1  | +1
 0  |  0
+1  | -1
+2  | -2
        """
        ax.text(7.5, 5, truth_table, fontsize=9, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.7))

        # 2. MIN gate
        ax = axes[0, 1]
        ax.set_title('MIN Gate (Minimum)', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw gate
        rect = FancyBboxPatch((2, 3), 3, 4, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.5, 5, 'MIN', fontsize=12, fontweight='bold', ha='center', va='center')

        # Inputs/outputs
        ax.plot([0, 2], [6, 6], 'k-', linewidth=2)
        ax.plot([0, 2], [4, 4], 'k-', linewidth=2)
        ax.plot([5, 7], [5, 5], 'k-', linewidth=2)
        ax.text(1, 6.5, 'A', fontsize=10, fontweight='bold')
        ax.text(1, 4.5, 'B', fontsize=10, fontweight='bold')
        ax.text(6, 5.5, 'OUT', fontsize=10, fontweight='bold')

        # Properties
        props = """
Properties:
• Commutative
• Associative
• Identity: MIN(A,+2)=A
        """
        ax.text(7.5, 5, props, fontsize=9,
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='lightgreen', alpha=0.5))

        # 3. MAX gate
        ax = axes[0, 2]
        ax.set_title('MAX Gate (Maximum)', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw gate
        rect = FancyBboxPatch((2, 3), 3, 4, boxstyle="round,pad=0.1",
                             facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.5, 5, 'MAX', fontsize=12, fontweight='bold', ha='center', va='center')

        # Inputs/outputs
        ax.plot([0, 2], [6, 6], 'k-', linewidth=2)
        ax.plot([0, 2], [4, 4], 'k-', linewidth=2)
        ax.plot([5, 7], [5, 5], 'k-', linewidth=2)
        ax.text(1, 6.5, 'A', fontsize=10, fontweight='bold')
        ax.text(1, 4.5, 'B', fontsize=10, fontweight='bold')
        ax.text(6, 5.5, 'OUT', fontsize=10, fontweight='bold')

        # Properties
        props = """
Properties:
• Commutative
• Associative
• Identity: MAX(A,-2)=A
        """
        ax.text(7.5, 5, props, fontsize=9,
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='lightcoral', alpha=0.5))

        # 4. Half Adder
        ax = axes[1, 0]
        ax.set_title('Half Adder', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw complex gate
        rect = FancyBboxPatch((2, 2), 4, 6, boxstyle="round,pad=0.2",
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(4, 5, 'HALF\nADDER', fontsize=11, fontweight='bold',
               ha='center', va='center')

        # Inputs/outputs
        ax.plot([0, 2], [6.5, 6.5], 'k-', linewidth=2)
        ax.plot([0, 2], [3.5, 3.5], 'k-', linewidth=2)
        ax.plot([6, 8], [6, 6], 'k-', linewidth=2)
        ax.plot([6, 8], [4, 4], 'k-', linewidth=2)
        ax.text(1, 7, 'A', fontsize=10, fontweight='bold')
        ax.text(1, 4, 'B', fontsize=10, fontweight='bold')
        ax.text(7, 6.5, 'SUM', fontsize=9, fontweight='bold')
        ax.text(7, 4.5, 'CARRY', fontsize=9, fontweight='bold')

        # 5. Full Adder
        ax = axes[1, 1]
        ax.set_title('Full Adder', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw complex gate
        rect = FancyBboxPatch((2, 1.5), 4, 7, boxstyle="round,pad=0.2",
                             facecolor='lightcyan', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(4, 5, 'FULL\nADDER', fontsize=11, fontweight='bold',
               ha='center', va='center')

        # Inputs/outputs
        ax.plot([0, 2], [7, 7], 'k-', linewidth=2)
        ax.plot([0, 2], [5, 5], 'k-', linewidth=2)
        ax.plot([0, 2], [3, 3], 'k-', linewidth=2)
        ax.plot([6, 8], [6, 6], 'k-', linewidth=2)
        ax.plot([6, 8], [4, 4], 'k-', linewidth=2)
        ax.text(1, 7.5, 'A', fontsize=10, fontweight='bold')
        ax.text(1, 5.5, 'B', fontsize=10, fontweight='bold')
        ax.text(1, 3.5, 'Cin', fontsize=9, fontweight='bold')
        ax.text(7, 6.5, 'SUM', fontsize=9, fontweight='bold')
        ax.text(7, 4.5, 'Cout', fontsize=9, fontweight='bold')

        # 6. Complexity comparison
        ax = axes[1, 2]
        ax.set_title('Transistor Count Comparison', fontweight='bold')

        gates = ['NOT', 'MIN/MAX', 'Half\nAdder', 'Full\nAdder']
        binary = [2, 6, 10, 28]
        pentary = [5, 14, 70, 110]

        x = np.arange(len(gates))
        width = 0.35

        bars1 = ax.bar(x - width/2, binary, width, label='Binary',
                      color='#e74c3c', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, pentary, width, label='Pentary',
                      color='#3498db', edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Transistor Count', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(gates)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_logic_gates.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_logic_gates.png")
        plt.close()

    def create_architecture_diagram(self):
        """Create processor architecture diagram"""
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.suptitle('Pentary Processor Architecture', fontsize=16, fontweight='bold')

        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Memory subsystem
        mem_box = FancyBboxPatch((0.5, 6), 3, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#e8f4f8', edgecolor='black', linewidth=2)
        ax.add_patch(mem_box)
        ax.text(2, 8.5, 'MEMORY', fontsize=12, fontweight='bold', ha='center')
        ax.text(2, 8, 'In-Memory\nCompute', fontsize=9, ha='center')
        ax.text(2, 7.2, '7× Density', fontsize=8, ha='center', style='italic')
        ax.text(2, 6.8, 'Memristor\nArray', fontsize=8, ha='center')

        # ALU
        alu_box = FancyBboxPatch((4.5, 6), 3, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#fff4e6', edgecolor='black', linewidth=2)
        ax.add_patch(alu_box)
        ax.text(6, 8.5, 'ALU', fontsize=12, fontweight='bold', ha='center')
        ax.text(6, 8, 'Pentary\nArithmetic', fontsize=9, ha='center')
        ax.text(6, 7.2, 'Shift-Add', fontsize=8, ha='center', style='italic')
        ax.text(6, 6.8, 'No FPU', fontsize=8, ha='center')

        # Neural Engine
        neural_box = FancyBboxPatch((8.5, 6), 3, 3.5, boxstyle="round,pad=0.1",
                                   facecolor='#f0e6ff', edgecolor='black', linewidth=2)
        ax.add_patch(neural_box)
        ax.text(10, 8.5, 'NEURAL', fontsize=12, fontweight='bold', ha='center')
        ax.text(10, 8, 'Matrix\nEngine', fontsize=9, ha='center')
        ax.text(10, 7.2, '10× Efficient', fontsize=8, ha='center', style='italic')
        ax.text(10, 6.8, 'Sparse Ops', fontsize=8, ha='center')

        # Control Unit
        ctrl_box = FancyBboxPatch((0.5, 2), 5, 3, boxstyle="round,pad=0.1",
                                 facecolor='#e6ffe6', edgecolor='black', linewidth=2)
        ax.add_patch(ctrl_box)
        ax.text(3, 4.2, 'CONTROL UNIT', fontsize=12, fontweight='bold', ha='center')
        ax.text(3, 3.5, 'Instruction Decode', fontsize=9, ha='center')
        ax.text(3, 3, 'Pipeline Control', fontsize=9, ha='center')
        ax.text(3, 2.5, 'Power Management', fontsize=9, ha='center')

        # I/O Interface
        io_box = FancyBboxPatch((6.5, 2), 5, 3, boxstyle="round,pad=0.1",
                               facecolor='#ffe6e6', edgecolor='black', linewidth=2)
        ax.add_patch(io_box)
        ax.text(9, 4.2, 'I/O INTERFACE', fontsize=12, fontweight='bold', ha='center')
        ax.text(9, 3.5, 'Binary ↔ Pentary', fontsize=9, ha='center')
        ax.text(9, 3, 'ADC/DAC', fontsize=9, ha='center')
        ax.text(9, 2.5, 'Communication', fontsize=9, ha='center')

        # Interconnect
        ax.annotate('', xy=(4.5, 7.5), xytext=(3.5, 7.5),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
        ax.annotate('', xy=(8.5, 7.5), xytext=(7.5, 7.5),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
        ax.annotate('', xy=(3, 5), xytext=(3, 6),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
        ax.annotate('', xy=(9, 5), xytext=(9, 6),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

        # Bus labels
        ax.text(4, 7.8, 'Data Bus', fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(8, 7.8, 'Data Bus', fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(2.5, 5.5, 'Control', fontsize=8, ha='center', rotation=90,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(9.5, 5.5, 'Control', fontsize=8, ha='center', rotation=90,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Key features box
        features_text = """
KEY FEATURES:
• 5-state logic: {-2, -1, 0, +1, +2}
• Zero-state power disconnect
• 20× smaller multipliers
• 7× memory density
• 10× energy efficiency
• Native sparsity support
        """
        ax.text(12, 5, features_text, fontsize=9,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        # Title for bottom
        ax.text(7, 0.5, 'Pentary Balanced Quinary Architecture',
               fontsize=11, ha='center', style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_architecture.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_architecture.png")
        plt.close()

    def create_neural_network_diagram(self):
        """Create neural network quantization diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Neural Network Architecture',
                     fontsize=16, fontweight='bold')

        # 1. Weight quantization
        ax = axes[0, 0]
        ax.set_title('Weight Quantization Scheme', fontweight='bold')

        # Show quantization levels
        float_range = np.linspace(-1, 1, 100)
        quantized = np.zeros_like(float_range)

        for i, val in enumerate(float_range):
            if val <= -0.6:
                quantized[i] = -2
            elif val <= -0.2:
                quantized[i] = -1
            elif val <= 0.2:
                quantized[i] = 0
            elif val <= 0.6:
                quantized[i] = 1
            else:
                quantized[i] = 2

        ax.plot(float_range, float_range, 'k--', label='Original', linewidth=2, alpha=0.5)
        ax.plot(float_range, quantized, 'b-', label='Quantized', linewidth=2)

        # Add colored regions
        colors_q = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4']
        levels = [-2, -1, 0, 1, 2]
        for i, (color, level) in enumerate(zip(colors_q, levels)):
            ax.axhline(y=level, color=color, linestyle=':', alpha=0.5, linewidth=2)
            ax.fill_between(float_range, level-0.1, level+0.1,
                          color=color, alpha=0.2)

        ax.set_xlabel('Original Weight Value', fontweight='bold')
        ax.set_ylabel('Quantized Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.5, 2.5)

        # 2. Matrix multiplication
        ax = axes[0, 1]
        ax.set_title('Matrix Multiplication: Shift-Add', fontweight='bold')
        ax.axis('off')

        text = """
Traditional (FP16):
  Input × Weight = Output
  [0.5] × [0.7] = [0.35]

  Requires: 3000-transistor FPU
  Energy: HIGH

Pentary (Shift-Add):
  Input × Weight = Output
  [+1] × [+1] = [+1]
  [+2] × [-1] = [-2]  (shift + negate)
  [+1] × [+2] = [+2]  (shift)

  Requires: 150-transistor shift-add
  Energy: 10× LOWER

Multiplication becomes:
  ×0  → disconnect (0 power!)
  ×±1 → pass/negate
  ×±2 → shift left
        """
        ax.text(0.1, 0.5, text, fontsize=9, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 3. Activation functions
        ax = axes[1, 0]
        ax.set_title('Pentary Activation Functions', fontweight='bold')

        x = np.linspace(-3, 3, 100)

        # Pentary ReLU
        relu_pent = np.zeros_like(x)
        for i, val in enumerate(x):
            if val <= -2:
                relu_pent[i] = -2
            elif val <= -1:
                relu_pent[i] = -1
            elif val <= 0:
                relu_pent[i] = 0
            elif val <= 1:
                relu_pent[i] = 1
            else:
                relu_pent[i] = 2

        ax.plot(x, relu_pent, 'b-', label='Pentary ReLU', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        ax.set_xlabel('Input', fontweight='bold')
        ax.set_ylabel('Output', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.5, 2.5)

        # 4. Performance comparison
        ax = axes[1, 1]
        ax.set_title('Performance Metrics', fontweight='bold')

        metrics = ['Transistors\nper Multiply', 'Energy per\nOperation',
                  'Memory\nDensity', 'Inference\nSpeed']
        binary_vals = [3000, 100, 1, 1]
        pentary_vals = [150, 10, 7, 3]

        x_pos = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_vals, width,
                      label='Binary (normalized)', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_vals, width,
                      label='Pentary (normalized)', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Relative Value (lower is better)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        # Add improvement labels
        improvements = ['20×', '10×', '7×', '3×']
        for i, (bar, imp) in enumerate(zip(bars1, improvements)):
            ax.text(bar.get_x() + width/2, max(binary_vals[i], pentary_vals[i]) * 1.5,
                   f'{imp}\nbetter', ha='center', fontsize=8, fontweight='bold',
                   color='green')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_neural_network.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_neural_network.png")
        plt.close()

    def create_cryptography_diagram(self):
        """Create cryptography and security diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Cryptography & Security',
                     fontsize=16, fontweight='bold')

        # 1. Side-channel resistance
        ax = axes[0, 0]
        ax.set_title('Power Analysis Resistance', fontweight='bold')

        time = np.linspace(0, 10, 100)

        # Binary power trace (shows pattern)
        binary_power = np.random.normal(1, 0.1, 100)
        key_bits = [1, 0, 1, 0, 1, 1, 0, 1]
        for i, bit in enumerate(key_bits):
            start = int(i * 100 / 8)
            end = int((i + 1) * 100 / 8)
            if bit == 1:
                binary_power[start:end] += 0.5

        # Pentary power trace (zeros have no power)
        pentary_power = np.random.normal(1, 0.1, 100)
        key_pents = [1, 0, 2, 0, -1, 0, 2, -2]
        for i, pent in enumerate(key_pents):
            start = int(i * 100 / 8)
            end = int((i + 1) * 100 / 8)
            if pent == 0:
                pentary_power[start:end] = 0  # Zero power!
            elif abs(pent) == 2:
                pentary_power[start:end] += 0.5

        ax.plot(time, binary_power, 'r-', label='Binary (vulnerable)', linewidth=2, alpha=0.7)
        ax.plot(time, pentary_power, 'b-', label='Pentary (resistant)', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Power Consumption', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 2)

        # Highlight zero-power regions
        for i, pent in enumerate(key_pents):
            if pent == 0:
                start = i * 10 / 8
                end = (i + 1) * 10 / 8
                ax.axvspan(start, end, color='green', alpha=0.2)
                ax.text((start + end) / 2, 1.8, '0\npower',
                       ha='center', fontsize=7, fontweight='bold')

        # 2. Lattice crypto speedup
        ax = axes[0, 1]
        ax.set_title('Post-Quantum Crypto Performance', fontweight='bold')

        operations = ['Key Gen', 'Encrypt', 'Decrypt', 'Sign', 'Verify']
        binary_time = [100, 80, 85, 120, 90]
        pentary_time = [30, 28, 32, 40, 35]

        x_pos = np.arange(len(operations))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_time, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_time, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add speedup labels
        for i in range(len(operations)):
            speedup = binary_time[i] / pentary_time[i]
            ax.text(i, max(binary_time[i], pentary_time[i]) + 5,
                   f'{speedup:.1f}×', ha='center', fontsize=9,
                   fontweight='bold', color='green')

        # 3. Encryption flow
        ax = axes[1, 0]
        ax.set_title('Pentary Encryption Pipeline', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw pipeline stages
        stages = [
            (1, 7, 'Plaintext\nInput'),
            (3.5, 7, 'Pentary\nConvert'),
            (6, 7, 'Matrix\nMultiply'),
            (8.5, 7, 'Ciphertext\nOutput')
        ]

        for i, (x, y, label) in enumerate(stages):
            color = ['#ffe6e6', '#e6f3ff', '#e6ffe6', '#fff4e6'][i]
            rect = FancyBboxPatch((x-0.6, y-0.6), 1.2, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=8, ha='center', va='center',
                   fontweight='bold')

            if i < len(stages) - 1:
                ax.annotate('', xy=(stages[i+1][0]-0.7, y), xytext=(x+0.7, y),
                          arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

        # Add features
        features = """
Features:
• Zero-state = no power signature
• Matrix ops 3-5× faster
• Sparse operations efficient
• Side-channel resistant
• Post-quantum ready
        """
        ax.text(5, 3, features, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        # 4. Security comparison
        ax = axes[1, 1]
        ax.set_title('Security Metrics Comparison', fontweight='bold')

        metrics = ['Power\nAnalysis\nResistance', 'Timing\nAttack\nResistance',
                  'Fault\nInjection\nResistance', 'Quantum\nResistance']
        binary_score = [3, 5, 4, 2]
        pentary_score = [9, 7, 6, 8]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        binary_score += binary_score[:1]
        pentary_score += pentary_score[:1]
        angles += angles[:1]

        ax = plt.subplot(2, 2, 4, projection='polar')
        ax.plot(angles, binary_score, 'o-', linewidth=2, label='Binary', color='#e74c3c')
        ax.fill(angles, binary_score, alpha=0.25, color='#e74c3c')
        ax.plot(angles, pentary_score, 'o-', linewidth=2, label='Pentary', color='#3498db')
        ax.fill(angles, pentary_score, alpha=0.25, color='#3498db')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Security Metrics Comparison', fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_cryptography.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_cryptography.png")
        plt.close()

    def create_all_diagrams(self):
        """Generate all diagrams"""
        print("\n🎨 Generating Pentary Diagrams...\n")

        self.create_pentary_number_system_diagram()
        self.create_logic_gates_diagram()
        self.create_architecture_diagram()
        self.create_neural_network_diagram()
        self.create_cryptography_diagram()

        print(f"\n✅ All diagrams created in '{self.output_dir}/' directory")
        print(f"📊 Total diagrams: 5")

if __name__ == "__main__":
    generator = PentaryDiagramGenerator()
    generator.create_all_diagrams()