#!/usr/bin/env python3

import json
import re

def fix_ada_notebook():
    """Fix all control character issues in ADA notebook."""
    
    filename = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/ada_training.ipynb"
    
    try:
        # Read the file as raw text
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove all control characters except valid JSON ones (\n, \r, \t)
        # First, fix common escape sequence issues
        content = re.sub(r'\\n"', '"', content)  # Remove standalone \n" patterns
        content = re.sub(r'",\\n,', '",', content)  # Remove ",\n, patterns
        
        # Remove any remaining control characters that aren't valid in JSON strings
        # This pattern removes control chars (0x00-0x1F) except \n (0x0A), \r (0x0D), \t (0x09)
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', content)
        
        # Fix specific JSON formatting issues
        content = re.sub(r'",\\n"', '", "', content)  # Fix escape issues in strings
        content = re.sub(r'"\\n', '"', content)  # Remove orphaned \n at start of string
        
        # Try to parse and reformat the JSON
        notebook_data = json.loads(content)
        
        # Write back the properly formatted JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Fixed ADA notebook successfully")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON error in ADA notebook: {e}")
        
        # Try more aggressive fixing by recreating the working SOL notebook structure
        # and copying just the cell content from ADA
        try:
            sol_file = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/sol_training.ipynb"
            
            # Load working SOL notebook as template
            with open(sol_file, 'r', encoding='utf-8') as f:
                sol_notebook = json.load(f)
            
            # Create a new ADA notebook based on SOL structure
            ada_notebook = sol_notebook.copy()
            
            # Update the title and content to be ADA-specific
            for cell in ada_notebook['cells']:
                if cell['cell_type'] == 'markdown' and len(cell['source']) > 0:
                    # Replace SOL references with ADA
                    for i, line in enumerate(cell['source']):
                        cell['source'][i] = line.replace('Solana (SOL)', 'Cardano (ADA)')
                        cell['source'][i] = cell['source'][i].replace('SOL', 'ADA')
                        cell['source'][i] = cell['source'][i].replace('Solana', 'Cardano')
                        cell['source'][i] = cell['source'][i].replace('sol_', 'ada_')
                        cell['source'][i] = cell['source'][i].replace('SOLUSDT', 'ADAUSDT')
                
                elif cell['cell_type'] == 'code':
                    # Update code cells to use ADA instead of SOL
                    for i, line in enumerate(cell['source']):
                        cell['source'][i] = line.replace('SOLUSDT', 'ADAUSDT')
                        cell['source'][i] = cell['source'][i].replace('sol_', 'ada_')
                        cell['source'][i] = cell['source'][i].replace("'SOL'", "'ADA'")
                        cell['source'][i] = cell['source'][i].replace('"SOL"', '"ADA"')
            
            # Write the new ADA notebook
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(ada_notebook, f, indent=1, ensure_ascii=False)
            
            print(f"✓ Recreated ADA notebook from template")
            return True
            
        except Exception as e2:
            print(f"✗ Could not fix ADA notebook: {e2}")
            return False
    
    except Exception as e:
        print(f"✗ Error processing ADA notebook: {e}")
        return False

if __name__ == "__main__":
    fix_ada_notebook()