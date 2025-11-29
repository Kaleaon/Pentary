#!/usr/bin/env python3
"""
Integrate Diagrams into Research Documents
Adds diagram references to existing research markdown files
"""

from pathlib import Path
import re

class DiagramIntegrator:
    """Integrate diagrams into research documents"""

    def __init__(self, research_dir="research", diagrams_dir="diagrams"):
        self.research_dir = Path(research_dir)
        self.diagrams_dir = Path(diagrams_dir)

        # Mapping of research files to their primary diagrams
        self.diagram_mapping = {
            'pentary_foundations.md': ['pentary_number_system.png', 'pentary_architecture.png'],
            'pentary_logic_gates.md': ['pentary_logic_gates.png'],
            'pentary_cryptography.md': ['pentary_cryptography.png'],
            'pentary_quantum_interface.md': ['pentary_quantum_interface.png'],
            'pentary_compiler_optimizations.md': ['pentary_compiler_optimizations.png'],
            'pentary_database_graphs.md': ['pentary_database_graphs.png'],
            'pentary_edge_computing.md': ['pentary_edge_computing.png'],
            'pentary_realtime_systems.md': ['pentary_realtime_systems.png'],
            'pentary_reliability.md': ['pentary_reliability.png'],
            'pentary_scientific_computing.md': ['pentary_scientific_computing.png'],
            'pentary_signal_processing.md': ['pentary_signal_processing.png'],
            'pentary_economics.md': ['pentary_economics.png'],
            'pentary_gaussian_splatting.md': ['pentary_gaussian_splatting.png'],
            'pentary_graphics_processor.md': ['pentary_graphics_processor.png'],
        }

    def add_diagram_to_document(self, doc_path, diagram_name):
        """Add diagram reference to document if not already present"""

        with open(doc_path, 'r') as f:
            content = f.read()

        # Check if diagram already referenced
        if diagram_name in content:
            print(f"  ⊙ Diagram already present in {doc_path.name}")
            return False

        # Find appropriate insertion point (after first heading)
        lines = content.split('\n')
        insert_index = 0

        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_index = i + 1
                break

        # Insert diagram reference
        diagram_ref = f"\n![Diagram](../diagrams/{diagram_name})\n"
        lines.insert(insert_index, diagram_ref)

        # Write back
        with open(doc_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  ✓ Added {diagram_name} to {doc_path.name}")
        return True

    def integrate_all(self):
        """Integrate diagrams into all research documents"""
        print("\n📊 Integrating Diagrams into Research Documents...\n")

        total_added = 0

        for doc_name, diagrams in self.diagram_mapping.items():
            doc_path = self.research_dir / doc_name

            if not doc_path.exists():
                print(f"  ⚠ Document not found: {doc_name}")
                continue

            print(f"Processing: {doc_name}")

            for diagram in diagrams:
                if self.add_diagram_to_document(doc_path, diagram):
                    total_added += 1

        print(f"\n✅ Integration complete!")
        print(f"📈 Total diagrams added: {total_added}")
        print(f"📄 Documents processed: {len(self.diagram_mapping)}")

if __name__ == "__main__":
    integrator = DiagramIntegrator()
    integrator.integrate_all()