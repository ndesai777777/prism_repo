"""Verify the DR notebook structure and execute the code cells."""
import json
from pathlib import Path

nb_path = Path(r'c:\Users\i.Nikesh.Desai\OneDrive - Acentra\Documents\GitHub\prism_repo\prism_repo\Code\PRISM_Doubly_Robust_Modeling_Workflow.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
print(f"Cells: {len(nb['cells'])}")
print(f"Format: {nb['nbformat']}.{nb['nbformat_minor']}")

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"Code cells: {len(code_cells)}, Markdown cells: {len(md_cells)}")

# Execute code cells in sequence
import sys, os
sys.path.insert(0, str(nb_path.parent))
os.chdir(nb_path.parent)

for i, cell in enumerate(code_cells):
    source = ''.join(cell['source'])
    try:
        exec(compile(source, f'<cell_{i}>', 'exec'))
        print(f"  Cell {i}: OK")
    except Exception as e:
        print(f"  Cell {i}: ERROR - {type(e).__name__}: {e}")
        break

print("\nVerification complete.")
