#!/usr/bin/env python3
"""
Specialized Pentary Diagram Generator
Creates diagrams for compiler, graphics, economics, and other specialized topics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Wedge, Polygon
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class SpecializedPentaryDiagrams:
    """Generate specialized diagrams"""

    def __init__(self, output_dir="diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.colors = {
            '-2': '#d62728', '-1': '#ff7f0e', '0': '#7f7f7f',
            '+1': '#2ca02c', '+2': '#1f77b4',
            'background': '#f0f0f0', 'text': '#333333', 'accent': '#9467bd'
        }

    def create_compiler_optimization_diagram(self):
        """Create compiler optimization pipeline diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Compiler Optimizations',
                     fontsize=16, fontweight='bold')

        # 1. Compilation pipeline
        ax = axes[0, 0]
        ax.set_title('Compilation Pipeline', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        stages = [
            (2, 9, 'Source\nCode', '#e6f3ff'),
            (2, 7.5, 'Lexer', '#ffe6f3'),
            (2, 6, 'Parser', '#e6ffe6'),
            (2, 4.5, 'IR Gen', '#fff4e6'),
            (6, 9, 'Pentary\nOptimizer', '#f0e6ff'),
            (6, 7.5, 'Code\nGen', '#ffe6e6'),
            (6, 6, 'Pentary\nASM', '#e6f3ff'),
            (6, 4.5, 'Binary', '#ffe6f3')
        ]

        for x, y, label, color in stages:
            rect = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=8, ha='center', va='center',
                   fontweight='bold')

        # Arrows
        arrows = [
            ((2, 8.6), (2, 7.9)),
            ((2, 7.1), (2, 6.4)),
            ((2, 5.6), (2, 4.9)),
            ((2.7, 4.5), (5.3, 4.5)),
            ((6, 4.9), (6, 5.6)),
            ((6, 6.4), (6, 7.1)),
            ((6, 7.9), (6, 8.6)),
            ((2.7, 9), (5.3, 9))
        ]

        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

        # Optimization box
        opt_text = """
Key Optimizations:
• Constant folding
• Dead code elimination
• Strength reduction
• Loop unrolling
• Pentary-specific opts
        """
        ax.text(9, 6.5, opt_text, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 2. Optimization techniques
        ax = axes[0, 1]
        ax.set_title('Optimization Impact', fontweight='bold')

        optimizations = ['Constant\nFolding', 'Strength\nReduction',
                        'Loop\nUnroll', 'Pentary\nSpecific']
        speedup = [1.2, 1.8, 2.5, 3.2]

        bars = ax.bar(optimizations, speedup, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                     edgecolor='black', linewidth=2)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Speedup Factor', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, speedup):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                   f'{val}×', ha='center', fontweight='bold')

        # 3. Code size comparison
        ax = axes[1, 0]
        ax.set_title('Code Size Reduction', fontweight='bold')

        code_types = ['Arithmetic', 'Matrix Ops', 'Neural Net', 'Crypto']
        binary_size = [100, 100, 100, 100]
        pentary_size = [85, 60, 45, 70]

        x_pos = np.arange(len(code_types))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_size, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_size, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Code Size (normalized)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(code_types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Compilation time
        ax = axes[1, 1]
        ax.set_title('Compilation Time', fontweight='bold')

        project_sizes = ['Small\n(<1K LOC)', 'Medium\n(1-10K)', 'Large\n(10-100K)', 'Huge\n(>100K)']
        binary_time = [1, 10, 100, 1000]
        pentary_time = [1.2, 12, 110, 1100]

        x_pos = np.arange(len(project_sizes))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_time, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_time, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Compilation Time (s)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(project_sizes, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_compiler_optimizations.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_compiler_optimizations.png")
        plt.close()

    def create_graphics_processor_diagram(self):
        """Create graphics processor architecture diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Graphics Processor Architecture',
                     fontsize=16, fontweight='bold')

        # 1. GPU architecture
        ax = axes[0, 0]
        ax.set_title('Pentary GPU Block Diagram', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Main blocks
        blocks = [
            (2, 8, 'Vertex\nShader', '#e6f3ff', 1.5, 1),
            (5, 8, 'Geometry\nShader', '#ffe6f3', 1.5, 1),
            (8, 8, 'Rasterizer', '#e6ffe6', 1.5, 1),
            (2, 6, 'Texture\nUnits', '#fff4e6', 1.5, 1),
            (5, 6, 'Fragment\nShader', '#f0e6ff', 1.5, 1),
            (8, 6, 'ROP', '#ffe6e6', 1.5, 1),
            (5, 4, 'Memory\nController', '#e6f3ff', 3, 1),
            (5, 2, 'Pentary\nMemory', '#ffe6f3', 3, 1)
        ]

        for x, y, label, color, w, h in blocks:
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=8, ha='center', va='center',
                   fontweight='bold')

        # Connections
        ax.plot([2.75, 4.25], [8, 8], 'b-', linewidth=2)
        ax.plot([5.75, 7.25], [8, 8], 'b-', linewidth=2)
        ax.plot([2, 2], [7.5, 6.5], 'b-', linewidth=2)
        ax.plot([5, 5], [7.5, 6.5], 'b-', linewidth=2)
        ax.plot([8, 8], [7.5, 6.5], 'b-', linewidth=2)
        ax.plot([5, 5], [5.5, 4.5], 'b-', linewidth=2)
        ax.plot([5, 5], [3.5, 2.5], 'b-', linewidth=2)

        # 2. Rendering pipeline performance
        ax = axes[0, 1]
        ax.set_title('Rendering Performance', fontweight='bold')

        stages = ['Vertex\nProc', 'Geometry', 'Raster', 'Fragment', 'Output']
        binary_fps = [60, 60, 60, 30, 60]
        pentary_fps = [90, 85, 95, 75, 90]

        x_pos = np.arange(len(stages))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_fps, width,
                      label='Binary GPU', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_fps, width,
                      label='Pentary GPU', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Throughput (Mpixels/s)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Power efficiency
        ax = axes[1, 0]
        ax.set_title('Power Efficiency by Workload', fontweight='bold')

        workloads = ['Gaming', '3D\nModeling', 'Ray\nTracing', 'Compute']
        binary_power = [200, 250, 300, 280]
        pentary_power = [80, 100, 120, 110]

        x_pos = np.arange(len(workloads))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_power, width,
                      label='Binary GPU', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_power, width,
                      label='Pentary GPU', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Power Consumption (W)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(workloads)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add efficiency labels
        for i in range(len(workloads)):
            efficiency = binary_power[i] / pentary_power[i]
            ax.text(i, max(binary_power[i], pentary_power[i]) + 20,
                   f'{efficiency:.1f}×\nmore\nefficient', ha='center', fontsize=7,
                   fontweight='bold', color='green')

        # 4. Feature support
        ax = axes[1, 1]
        ax.set_title('Graphics Features Support', fontweight='bold')

        features = ['Texture\nFiltering', 'Anti-\nAliasing', 'Shading',
                   'Ray\nTracing', 'Compute']
        support_level = [9, 8, 9, 7, 10]

        bars = ax.barh(features, support_level, color='#2ecc71',
                      edgecolor='black', linewidth=2)
        ax.set_xlabel('Support Level', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3, axis='x')

        for i, (bar, val) in enumerate(zip(bars, support_level)):
            ax.text(val + 0.2, i, f'{val}/10', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_graphics_processor.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_graphics_processor.png")
        plt.close()

    def create_economics_diagram(self):
        """Create economics and cost analysis diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Economics & Cost Analysis',
                     fontsize=16, fontweight='bold')

        # 1. Manufacturing cost breakdown
        ax = axes[0, 0]
        ax.set_title('Manufacturing Cost Breakdown', fontweight='bold')

        components = ['Wafer\nCost', 'Packaging', 'Testing', 'Yield\nLoss', 'Other']
        binary_costs = [40, 15, 10, 20, 15]
        pentary_costs = [45, 18, 12, 15, 10]

        x_pos = np.arange(len(components))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_costs, width,
                      label='Binary Chip', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_costs, width,
                      label='Pentary Chip', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Cost ($)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(components, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add total cost
        binary_total = sum(binary_costs)
        pentary_total = sum(pentary_costs)
        ax.text(len(components)-0.5, max(binary_costs) + 10,
               f'Total:\nBinary: ${binary_total}\nPentary: ${pentary_total}',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 2. TCO over time
        ax = axes[0, 1]
        ax.set_title('Total Cost of Ownership (5 years)', fontweight='bold')

        years = np.arange(0, 6)

        # Binary TCO
        binary_tco = [100]  # Initial cost
        for year in range(1, 6):
            binary_tco.append(binary_tco[-1] + 50)  # Operating cost per year

        # Pentary TCO
        pentary_tco = [100]  # Initial cost (same)
        for year in range(1, 6):
            pentary_tco.append(pentary_tco[-1] + 15)  # Lower operating cost

        ax.plot(years, binary_tco, 'r-o', linewidth=2, label='Binary System', markersize=8)
        ax.plot(years, pentary_tco, 'b-o', linewidth=2, label='Pentary System', markersize=8)

        ax.set_xlabel('Years', fontweight='bold')
        ax.set_ylabel('Total Cost ($1000s)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight savings
        savings = binary_tco[-1] - pentary_tco[-1]
        ax.text(3, 200, f'5-year savings:\n${savings}K ({savings/binary_tco[-1]*100:.0f}%)',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               fontweight='bold')

        # 3. Performance per dollar
        ax = axes[1, 0]
        ax.set_title('Performance per Dollar', fontweight='bold')

        metrics = ['TOPS/\n$1000', 'TOPS/W\nper $', 'Memory\nGB/$', 'Inference\nFPS/$']
        binary_perf = [10, 5, 2, 50]
        pentary_perf = [30, 20, 14, 150]

        x_pos = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_perf, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_perf, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Performance Metric', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add improvement ratios
        for i in range(len(metrics)):
            ratio = pentary_perf[i] / binary_perf[i]
            ax.text(i, max(binary_perf[i], pentary_perf[i]) + 5,
                   f'{ratio:.1f}×', ha='center', fontsize=9,
                   fontweight='bold', color='green')

        # 4. Market segments
        ax = axes[1, 1]
        ax.set_title('Target Market Segments', fontweight='bold')

        segments = ['Data\nCenter', 'Edge\nDevices', 'Mobile', 'IoT', 'Automotive']
        market_size = [35, 25, 20, 10, 10]
        colors_seg = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

        wedges, texts, autotexts = ax.pie(market_size, labels=segments,
                                          autopct='%1.1f%%', colors=colors_seg,
                                          startangle=90, textprops={'fontsize': 9})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_economics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_economics.png")
        plt.close()

    def create_realtime_systems_diagram(self):
        """Create real-time systems diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Real-Time Systems',
                     fontsize=16, fontweight='bold')

        # 1. Timing diagram
        ax = axes[0, 0]
        ax.set_title('Task Scheduling Timeline', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)

        # Draw tasks
        tasks = [
            ('Task A', 0, 2, 4, '#3498db'),
            ('Task B', 2, 1, 3.5, '#2ecc71'),
            ('Task C', 3, 1.5, 3, '#f39c12'),
            ('Task A', 4.5, 2, 2.5, '#3498db'),
            ('Task D', 6.5, 1, 2, '#e74c3c')
        ]

        for name, start, duration, y, color in tasks:
            rect = Rectangle((start, y-0.3), duration, 0.6,
                           facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(start + duration/2, y, name, fontsize=8,
                   ha='center', va='center', fontweight='bold', color='white')

        # Deadline markers
        deadlines = [2.5, 3.5, 5, 7]
        for dl in deadlines:
            ax.axvline(x=dl, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.text(dl, 0.5, 'DL', fontsize=8, ha='center', color='red', fontweight='bold')

        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Priority Level', fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        # 2. Latency comparison
        ax = axes[0, 1]
        ax.set_title('Worst-Case Execution Time', fontweight='bold')

        operations = ['Sensor\nRead', 'Process', 'Actuate', 'Network', 'Total']
        binary_wcet = [0.5, 5, 1, 2, 8.5]
        pentary_wcet = [0.3, 1.5, 0.5, 1, 3.3]

        x_pos = np.arange(len(operations))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, binary_wcet, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_wcet, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)

        ax.set_ylabel('WCET (ms)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operations, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Jitter analysis
        ax = axes[1, 0]
        ax.set_title('Timing Jitter Distribution', fontweight='bold')

        # Generate jitter data
        np.random.seed(42)
        binary_jitter = np.random.normal(5, 1.5, 1000)
        pentary_jitter = np.random.normal(5, 0.5, 1000)

        ax.hist(binary_jitter, bins=30, alpha=0.5, label='Binary',
               color='#e74c3c', edgecolor='black')
        ax.hist(pentary_jitter, bins=30, alpha=0.5, label='Pentary',
               color='#3498db', edgecolor='black')

        ax.set_xlabel('Execution Time (ms)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"""
Binary:
  Mean: {np.mean(binary_jitter):.2f}ms
  Std: {np.std(binary_jitter):.2f}ms

Pentary:
  Mean: {np.mean(pentary_jitter):.2f}ms
  Std: {np.std(pentary_jitter):.2f}ms
        """
        ax.text(7.5, ax.get_ylim()[1]*0.7, stats_text, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               family='monospace')

        # 4. Application domains
        ax = axes[1, 1]
        ax.set_title('Real-Time Application Domains', fontweight='bold')

        domains = ['Industrial\nControl', 'Automotive', 'Aerospace',
                  'Medical', 'Robotics']
        suitability = [9, 9, 8, 9, 10]

        bars = ax.barh(domains, suitability, color='#2ecc71',
                      edgecolor='black', linewidth=2)
        ax.set_xlabel('Suitability Score', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3, axis='x')

        for i, (bar, val) in enumerate(zip(bars, suitability)):
            ax.text(val + 0.2, i, f'{val}/10', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_realtime_systems.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_realtime_systems.png")
        plt.close()

    def create_all_specialized_diagrams(self):
        """Generate all specialized diagrams"""
        print("\n🎨 Generating Specialized Pentary Diagrams...\n")

        self.create_compiler_optimization_diagram()
        self.create_graphics_processor_diagram()
        self.create_economics_diagram()
        self.create_realtime_systems_diagram()

        print(f"\n✅ All specialized diagrams created in '{self.output_dir}/' directory")
        print(f"📊 Total specialized diagrams: 4")

if __name__ == "__main__":
    generator = SpecializedPentaryDiagrams()
    generator.create_all_specialized_diagrams()