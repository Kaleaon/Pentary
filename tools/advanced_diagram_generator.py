#!/usr/bin/env python3
"""
Advanced Pentary Diagram Generator
Creates specialized diagrams for advanced research topics
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

class AdvancedPentaryDiagrams:
    """Generate advanced diagrams for specialized topics"""
    
    def __init__(self, output_dir="diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.colors = {
            '-2': '#d62728', '-1': '#ff7f0e', '0': '#7f7f7f',
            '+1': '#2ca02c', '+2': '#1f77b4',
            'background': '#f0f0f0', 'text': '#333333', 'accent': '#9467bd'
        }
    
    def create_quantum_interface_diagram(self):
        """Create quantum-pentary interface diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary-Quantum Computing Interface', 
                     fontsize=16, fontweight='bold')
        
        # 1. Qubit encoding
        ax = axes[0, 0]
        ax.set_title('Quantum State Encoding', fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # Bloch sphere representation
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.contour(x, y, z, levels=10, alpha=0.3, colors='gray')
        
        # Mark pentary encoding points
        states = [
            (0, 0, 1, '|0⟩ → 0'),
            (0, 0, -1, '|1⟩ → +2'),
            (1, 0, 0, '|+⟩ → +1'),
            (-1, 0, 0, '|-⟩ → -1'),
            (0, 1, 0, '|i⟩ → -2')
        ]
        
        for x, y, z, label in states:
            ax.plot([0, x], [0, y], 'b-', linewidth=2, alpha=0.7)
            ax.scatter(x, y, s=100, c='red', zorder=5)
            ax.text(x*1.2, y*1.2, label, fontsize=8, ha='center')
        
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Qubit → Pentary Mapping', fontweight='bold')
        
        # 2. Hybrid algorithm flow
        ax = axes[0, 1]
        ax.set_title('Variational Quantum Algorithm', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw flow
        boxes = [
            (2, 8, 'Initialize\nParameters', '#e6f3ff'),
            (2, 6, 'Quantum\nCircuit', '#ffe6f3'),
            (2, 4, 'Measure\nResults', '#e6ffe6'),
            (6, 6, 'Pentary\nOptimizer', '#fff4e6'),
            (6, 4, 'Update\nParameters', '#f0e6ff')
        ]
        
        for x, y, label, color in boxes:
            rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=9, ha='center', va='center',
                   fontweight='bold')
        
        # Arrows
        arrows = [
            ((2, 7.5), (2, 6.5)),
            ((2, 5.5), (2, 4.5)),
            ((2.8, 4), (5.2, 4)),
            ((6, 4.5), (6, 5.5)),
            ((5.2, 6), (2.8, 6)),
            ((2, 8.5), (2, 9.5))
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        ax.text(5, 9, 'Convergence?', fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 3. Error correction
        ax = axes[1, 0]
        ax.set_title('Quantum Error Correction', fontweight='bold')
        
        # Show error rates
        error_types = ['Bit Flip', 'Phase Flip', 'Depolarizing', 'Amplitude\nDamping']
        binary_ecc = [0.001, 0.001, 0.002, 0.0015]
        pentary_ecc = [0.0005, 0.0006, 0.0008, 0.0007]
        
        x_pos = np.arange(len(error_types))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, binary_ecc, width,
                      label='Binary ECC', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_ecc, width,
                      label='Pentary ECC', color='#3498db',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Error Rate', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(error_types, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # 4. Performance metrics
        ax = axes[1, 1]
        ax.set_title('Hybrid Computing Performance', fontweight='bold')
        
        metrics = ['Circuit\nDepth', 'Gate\nCount', 'Coherence\nTime', 'Fidelity']
        improvement = [1.2, 1.5, 1.8, 1.4]
        
        bars = ax.barh(metrics, improvement, color='#2ecc71',
                      edgecolor='black', linewidth=2)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax.set_xlabel('Improvement Factor', fontweight='bold')
        ax.set_xlim(0, 2)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, improvement)):
            ax.text(val + 0.05, i, f'{val}×', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_quantum_interface.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_quantum_interface.png")
        plt.close()
    
    def create_edge_computing_diagram(self):
        """Create edge computing topology diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Edge Computing Architecture', 
                     fontsize=16, fontweight='bold')
        
        # 1. Network topology
        ax = axes[0, 0]
        ax.set_title('Edge Network Topology', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Cloud
        cloud = Circle((5, 9), 0.8, facecolor='#e6f3ff', edgecolor='black', linewidth=2)
        ax.add_patch(cloud)
        ax.text(5, 9, 'Cloud\nData Center', fontsize=8, ha='center', va='center',
               fontweight='bold')
        
        # Edge servers
        edge_positions = [(2, 6), (5, 6), (8, 6)]
        for i, (x, y) in enumerate(edge_positions):
            rect = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#ffe6f3', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, f'Edge\nServer {i+1}', fontsize=7, ha='center', va='center',
                   fontweight='bold')
            
            # Connect to cloud
            ax.plot([x, 5], [y+0.4, 8.2], 'b--', linewidth=1.5, alpha=0.5)
        
        # IoT devices
        device_positions = [
            (1, 3), (2, 3), (3, 3),
            (4, 3), (5, 3), (6, 3),
            (7, 3), (8, 3), (9, 3)
        ]
        
        for i, (x, y) in enumerate(device_positions):
            device = Circle((x, y), 0.3, facecolor='#e6ffe6', 
                          edgecolor='black', linewidth=1.5)
            ax.add_patch(device)
            ax.text(x, y, 'IoT', fontsize=6, ha='center', va='center')
            
            # Connect to nearest edge server
            edge_idx = i // 3
            edge_x, edge_y = edge_positions[edge_idx]
            ax.plot([x, edge_x], [y+0.3, edge_y-0.4], 'g-', linewidth=1, alpha=0.5)
        
        # 2. Power consumption
        ax = axes[0, 1]
        ax.set_title('Power Consumption by Location', fontweight='bold')
        
        locations = ['Cloud', 'Edge\nServer', 'IoT\nDevice']
        binary_power = [1000, 100, 10]
        pentary_power = [800, 20, 2]
        
        x_pos = np.arange(len(locations))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, binary_power, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_power, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Power (Watts)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(locations)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # 3. Latency comparison
        ax = axes[1, 0]
        ax.set_title('Processing Latency', fontweight='bold')
        
        scenarios = ['Cloud\nOnly', 'Edge\n(Binary)', 'Edge\n(Pentary)', 'Local\n(Pentary)']
        latencies = [150, 50, 15, 5]
        colors_lat = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        
        bars = ax.bar(scenarios, latencies, color=colors_lat,
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, lat in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, lat + 5,
                   f'{lat}ms', ha='center', fontweight='bold')
        
        # 4. Use case distribution
        ax = axes[1, 1]
        ax.set_title('Edge Computing Use Cases', fontweight='bold')
        
        use_cases = ['Smart\nCities', 'Industrial\nIoT', 'Healthcare', 
                    'Autonomous\nVehicles', 'AR/VR']
        percentages = [25, 30, 15, 20, 10]
        colors_pie = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        wedges, texts, autotexts = ax.pie(percentages, labels=use_cases,
                                          autopct='%1.1f%%', colors=colors_pie,
                                          startangle=90, textprops={'fontsize': 9})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_edge_computing.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_edge_computing.png")
        plt.close()
    
    def create_signal_processing_diagram(self):
        """Create signal processing pipeline diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Signal Processing', 
                     fontsize=16, fontweight='bold')
        
        # 1. Filter response
        ax = axes[0, 0]
        ax.set_title('Pentary FIR Filter Response', fontweight='bold')
        
        # Generate frequency response
        freq = np.linspace(0, 0.5, 1000)
        
        # Binary filter (8-bit coefficients)
        binary_response = np.ones_like(freq)
        for i in range(1, 5):
            binary_response *= np.sinc(freq * i)
        
        # Pentary filter (5-level coefficients)
        pentary_response = np.ones_like(freq)
        for i in range(1, 5):
            pentary_response *= np.sinc(freq * i * 0.9)
        
        ax.plot(freq, 20*np.log10(np.abs(binary_response)), 'r-', 
               label='Binary (8-bit)', linewidth=2, alpha=0.7)
        ax.plot(freq, 20*np.log10(np.abs(pentary_response)), 'b-',
               label='Pentary (5-level)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Normalized Frequency', fontweight='bold')
        ax.set_ylabel('Magnitude (dB)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-60, 5)
        
        # 2. Processing pipeline
        ax = axes[0, 1]
        ax.set_title('Signal Processing Pipeline', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        stages = [
            (1.5, 8, 'ADC\n5-level', '#e6f3ff'),
            (1.5, 6, 'Pentary\nFilter', '#ffe6f3'),
            (1.5, 4, 'FFT', '#e6ffe6'),
            (5, 8, 'Feature\nExtract', '#fff4e6'),
            (5, 6, 'Classify', '#f0e6ff'),
            (5, 4, 'Output', '#ffe6e6')
        ]
        
        for x, y, label, color in stages:
            rect = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, label, fontsize=8, ha='center', va='center',
                   fontweight='bold')
        
        # Connect stages
        connections = [
            ((1.5, 7.5), (1.5, 6.5)),
            ((1.5, 5.5), (1.5, 4.5)),
            ((2.1, 8), (4.4, 8)),
            ((2.1, 6), (4.4, 6)),
            ((5, 7.5), (5, 6.5)),
            ((5, 5.5), (5, 4.5))
        ]
        
        for start, end in connections:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # 3. Computational efficiency
        ax = axes[1, 0]
        ax.set_title('Operations per Second', fontweight='bold')
        
        operations = ['MAC\nOps', 'FFT\n1024', 'Filter\n64-tap', 'Correlation']
        binary_ops = [1e9, 5e7, 2e8, 1e8]
        pentary_ops = [3e9, 1.5e8, 6e8, 3e8]
        
        x_pos = np.arange(len(operations))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, binary_ops, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_ops, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Operations/Second', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operations, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # Add speedup labels
        for i in range(len(operations)):
            speedup = pentary_ops[i] / binary_ops[i]
            ax.text(i, max(binary_ops[i], pentary_ops[i]) * 1.5,
                   f'{speedup:.1f}×', ha='center', fontsize=9,
                   fontweight='bold', color='green')
        
        # 4. Application areas
        ax = axes[1, 1]
        ax.set_title('Signal Processing Applications', fontweight='bold')
        
        applications = ['Audio\nProcessing', 'Radar', 'Communications',
                       'Biomedical', 'Image\nProcessing']
        performance = [8, 7, 9, 6, 8]
        
        angles = np.linspace(0, 2 * np.pi, len(applications), endpoint=False).tolist()
        performance += performance[:1]
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        ax.plot(angles, performance, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, performance, alpha=0.25, color='#3498db')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(applications, fontsize=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=7)
        ax.set_title('Performance Score', fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_signal_processing.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_signal_processing.png")
        plt.close()
    
    def create_database_graphs_diagram(self):
        """Create database and graph processing diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pentary Database & Graph Processing', 
                     fontsize=16, fontweight='bold')
        
        # 1. Graph representation
        ax = axes[0, 0]
        ax.set_title('Pentary Graph Encoding', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw sample graph
        nodes = [
            (2, 8, 'A', '+2'),
            (5, 8, 'B', '+1'),
            (8, 8, 'C', '0'),
            (2, 5, 'D', '-1'),
            (5, 5, 'E', '+2'),
            (8, 5, 'F', '-2')
        ]
        
        edges = [
            (0, 1, '+1'), (1, 2, '+2'), (0, 3, '-1'),
            (1, 4, '+1'), (2, 5, '0'), (3, 4, '+2'), (4, 5, '-1')
        ]
        
        # Draw edges first
        for i, j, weight in edges:
            x1, y1, _, _ = nodes[i]
            x2, y2, _, _ = nodes[j]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.5)
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y, weight, fontsize=8, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw nodes
        for x, y, label, value in nodes:
            color = self.colors.get(value, '#7f7f7f')
            circle = Circle((x, y), 0.4, facecolor=color, 
                          edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, label, fontsize=12, ha='center', va='center',
                   fontweight='bold', color='white')
            ax.text(x, y-0.7, value, fontsize=8, ha='center')
        
        # 2. Query performance
        ax = axes[0, 1]
        ax.set_title('Graph Query Performance', fontweight='bold')
        
        queries = ['Shortest\nPath', 'PageRank', 'Community\nDetection',
                  'Pattern\nMatch']
        binary_time = [100, 200, 300, 150]
        pentary_time = [30, 60, 80, 45]
        
        x_pos = np.arange(len(queries))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, binary_time, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_time, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Query Time (ms)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(queries, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add speedup labels
        for i in range(len(queries)):
            speedup = binary_time[i] / pentary_time[i]
            ax.text(i, max(binary_time[i], pentary_time[i]) + 20,
                   f'{speedup:.1f}×', ha='center', fontsize=9,
                   fontweight='bold', color='green')
        
        # 3. Storage efficiency
        ax = axes[1, 0]
        ax.set_title('Storage Efficiency', fontweight='bold')
        
        data_types = ['Adjacency\nMatrix', 'Edge\nList', 'Node\nAttributes',
                     'Index\nStructures']
        binary_size = [100, 80, 60, 40]
        pentary_size = [70, 55, 40, 25]
        
        x_pos = np.arange(len(data_types))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, binary_size, width,
                      label='Binary', color='#e74c3c',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, pentary_size, width,
                      label='Pentary', color='#3498db',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Storage Size (MB)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data_types, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Use cases
        ax = axes[1, 1]
        ax.set_title('Database Use Cases', fontweight='bold')
        
        use_cases = ['Social\nNetworks', 'Knowledge\nGraphs', 'Recommendation',
                    'Fraud\nDetection', 'Network\nAnalysis']
        suitability = [9, 8, 9, 7, 8]
        
        bars = ax.barh(use_cases, suitability, color='#2ecc71',
                      edgecolor='black', linewidth=2)
        ax.set_xlabel('Suitability Score', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, suitability)):
            ax.text(val + 0.2, i, f'{val}/10', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pentary_database_graphs.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created: pentary_database_graphs.png")
        plt.close()
    
    def create_all_advanced_diagrams(self):
        """Generate all advanced diagrams"""
        print("\n🎨 Generating Advanced Pentary Diagrams...\n")
        
        self.create_quantum_interface_diagram()
        self.create_edge_computing_diagram()
        self.create_signal_processing_diagram()
        self.create_database_graphs_diagram()
        
        print(f"\n✅ All advanced diagrams created in '{self.output_dir}/' directory")
        print(f"📊 Total advanced diagrams: 4")

if __name__ == "__main__":
    generator = AdvancedPentaryDiagrams()
    generator.create_all_advanced_diagrams()