#!/usr/bin/env python3

import json
import sys
import re

def fix_notebook_json(filename):
    """Fix common JSON formatting issues in Jupyter notebooks."""
    
    try:
        # Read the file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix common issues:
        # 1. Remove literal \n characters that should be actual newlines
        content = content.replace('\\n   ', '\n   ')
        content = content.replace('",\\n   "', '",\n   "')
        
        # 2. Fix unescaped quotes and newlines in display_name
        content = re.sub(r'"display_name": "([^"]*)",\\n', r'"display_name": "\1",\n', content)
        
        # 3. Try to parse as JSON to validate
        notebook_data = json.loads(content)
        
        # Write back the properly formatted JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Fixed {filename}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON error in {filename}: {e}")
        
        # Try more aggressive fixing
        try:
            # Fix metadata section specifically
            content = re.sub(
                r'"display_name": "Python 3",\\n   "language": "python",',
                r'"display_name": "Python 3",\n   "language": "python",',
                content
            )
            
            # Fix other common patterns
            content = re.sub(r'",\\n   "', r'",\n   "', content)
            content = re.sub(r'": "([^"]*)",\\n', r'": "\1",\n', content)
            
            notebook_data = json.loads(content)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(notebook_data, f, indent=1, ensure_ascii=False)
            
            print(f"✓ Fixed {filename} (aggressive mode)")
            return True
            
        except Exception as e2:
            print(f"✗ Could not fix {filename}: {e2}")
            return False
    
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        return False

if __name__ == "__main__":
    files_to_fix = [
        "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/btc_training.ipynb",
        "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/ada_training.ipynb"
    ]
    
    for filename in files_to_fix:
        fix_notebook_json(filename)