#!/usr/bin/env python3

import json
import re

def fix_ada_matplotlib_style():
    """Fix matplotlib style issue in ADA notebook"""
    
    notebook_path = "notebooks/individual_models/ada_training.ipynb"
    
    print("🔧 Fixing matplotlib style in ADA notebook...")
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells_fixed = 0
        
        # Process each cell
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source_lines = cell.get('source', [])
                
                # Convert source to single string for easier processing
                if isinstance(source_lines, list):
                    source_text = ''.join(source_lines)
                else:
                    source_text = source_lines
                
                original_source = source_text
                
                # Fix matplotlib style issue
                if "plt.style.use('seaborn-v0_8')" in source_text:
                    print(f"   📍 Fixing matplotlib style in cell {i}")
                    
                    # Replace the problematic style line with safe error handling
                    source_text = source_text.replace(
                        "plt.style.use('seaborn-v0_8')",
                        """try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        print("⚠️ Using default matplotlib style")
        pass"""
                    )
                    
                    cells_fixed += 1
                
                # Update the cell if changes were made
                if source_text != original_source:
                    # Convert back to list format
                    cell['source'] = source_text.split('\n')
                    # Ensure each line ends with \n except the last
                    cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                    for i, line in enumerate(cell['source'])]
        
        # Write the corrected notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✅ Fixed ADA notebook: {cells_fixed} cells updated")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing ADA notebook: {e}")
        return False

if __name__ == "__main__":
    success = fix_ada_matplotlib_style()
    if success:
        print("🎉 ADA notebook matplotlib style fixed successfully!")
    else:
        print("💥 Failed to fix ADA notebook matplotlib style")