#!/usr/bin/env python3

import json

def fix_major_coins_notebook():
    """Fix major coins portfolio notebook JSON issues."""
    
    filename = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/multi_asset_portfolios/major_coins_portfolio.ipynb"
    
    try:
        # Read file as text to fix JSON issues
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix common JSON formatting issues
        content = content.replace('\\n   "', '\n   "')
        content = content.replace('",\\n   "', '",\n   "')
        
        # Try to load and reformat
        notebook_data = json.loads(content)
        
        # Write properly formatted JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=1, ensure_ascii=False)
        
        print("✅ Fixed major_coins_portfolio.ipynb")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON error: {e}")
        
        # Use working SOL template as base
        try:
            sol_file = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/sol_training.ipynb"
            
            with open(sol_file, 'r') as f:
                sol_notebook = json.load(f)
            
            # Modify for major coins portfolio
            for cell in sol_notebook['cells']:
                if cell['cell_type'] == 'markdown' and len(cell['source']) > 0:
                    for i, line in enumerate(cell['source']):
                        if 'Solana (SOL)' in line:
                            cell['source'][i] = line.replace('Solana (SOL)', 'Major Coins Portfolio (BTC, ETH, BNB)')
                        elif 'SOL' in line and 'SOLUSDT' not in line:
                            cell['source'][i] = line.replace('SOL', 'BTC, ETH, BNB')
                        elif 'sol_' in line:
                            cell['source'][i] = line.replace('sol_', 'major_coins_')
                elif cell['cell_type'] == 'code':
                    for i, line in enumerate(cell['source']):
                        if "assets = ['SOL']" in line:
                            cell['source'][i] = line.replace("assets = ['SOL']", "assets = ['BTC', 'ETH', 'BNB']")
                        elif 'SOLUSDT' in line:
                            cell['source'][i] = line.replace('SOLUSDT', 'BTCUSDT')
                        elif 'sol_' in line:
                            cell['source'][i] = line.replace('sol_', 'major_coins_')
            
            # Write the corrected notebook
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(sol_notebook, f, indent=1, ensure_ascii=False)
            
            print("✅ Recreated major_coins_portfolio.ipynb from template")
            return True
            
        except Exception as e2:
            print(f"❌ Could not fix: {e2}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_major_coins_notebook()