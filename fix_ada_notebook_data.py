#!/usr/bin/env python3

import json
import re

def fix_ada_notebook_data_loading():
    """Fix ADA notebook to use proper data loading and fix all SOL references."""
    
    filename = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/ada_training.ipynb"
    
    try:
        # Read the notebook
        with open(filename, 'r') as f:
            notebook = json.load(f)
        
        # Fix each cell
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # Update source code in each cell
                for i, line in enumerate(cell['source']):
                    # Fix data loading function name
                    if 'def load_ada_data():' in line:
                        cell['source'][i] = line.replace('def load_ada_data():', 'def load_ada_data():')
                    
                    # Fix function docstring
                    elif 'Load SOL cryptocurrency data with Solana-specific preprocessing' in line:
                        cell['source'][i] = line.replace('Load SOL cryptocurrency data with Solana-specific preprocessing', 'Load ADA cryptocurrency data with Cardano-specific preprocessing')
                    
                    # Fix CSV loading attempt
                    elif "df = pd.read_csv('../../data/ADAUSDT_5m.csv')" in line:
                        cell['source'][i] = "        # Try to load from existing CSV data first\n        csv_files = ['crypto_5currencies_2years.csv', 'crypto_5min_2years.csv', 'crypto_test_data.csv']\n        df = None\n        \n        for csv_file in csv_files:\n            try:\n                temp_df = pd.read_csv(csv_file)\n                if 'tic' in temp_df.columns and 'ADA' in temp_df['tic'].unique():\n                    df = temp_df[temp_df['tic'] == 'ADA'].copy()\n                    print(f\"Loaded ADA data from {csv_file}\")\n                    break\n            except FileNotFoundError:\n                continue"
                    
                    elif "print(f\"Loaded {len(df)} rows of SOL data\")" in line:
                        cell['source'][i] = "        if df is not None:\n            print(f\"Loaded {len(df)} rows of ADA data\")"
                    
                    # Fix fallback download
                    elif 'print("CSV not found, downloading fresh SOL data...")' in line:
                        cell['source'][i] = "        if df is None:\n            print(\"CSV not found, downloading fresh ADA data...\")"
                    
                    # Fix YahooDownloader call
                    elif "ticker_list=['SOL-USD']" in line:
                        cell['source'][i] = line.replace("ticker_list=['SOL-USD']", "ticker_list=['ADA-USD']")
                    
                    # Fix ticker assignment
                    elif "df['tic'] = 'ADAUSDT'" in line:
                        cell['source'][i] = "        if 'tic' not in df.columns:\n            df['tic'] = 'ADA'"
                    
                    # Fix all SOL references in print statements and comments
                    elif 'SOL' in line and 'SOLUSDT' not in line:
                        cell['source'][i] = line.replace('SOL', 'ADA')
                    
                    # Fix Solana references
                    elif 'Solana' in line:
                        cell['source'][i] = line.replace('Solana', 'Cardano')
                    
                    # Fix function names
                    elif 'ada_' in line and 'SOL' in line:
                        cell['source'][i] = line.replace('SOL', 'ADA')
                    
                    # Fix create_ada_features function
                    elif 'def create_ada_features(df):' in line:
                        cell['source'][i] = line.replace('def create_ada_features(df):', 'def create_ada_features(df):')
                    
                    elif 'Create technical indicators optimized for SOL' in line:
                        cell['source'][i] = line.replace('Create technical indicators optimized for SOL', 'Create technical indicators optimized for ADA')
                    
                    # Fix all other SOL-specific references in comments
                    elif '# SOL' in line:
                        cell['source'][i] = line.replace('# SOL', '# ADA')
                    
                    elif 'SOL-specific' in line:
                        cell['source'][i] = line.replace('SOL-specific', 'ADA-specific')
                    
                    # Fix chart titles and labels
                    elif "'SOL " in line:
                        cell['source'][i] = line.replace("'SOL ", "'ADA ")
                    
                    elif '"SOL ' in line:
                        cell['source'][i] = line.replace('"SOL ', '"ADA ')
                    
                    # Fix environment setup print
                    elif 'Environment setup complete for SOL (Solana) trading' in line:
                        cell['source'][i] = line.replace('Environment setup complete for SOL (Solana) trading', 'Environment setup complete for ADA (Cardano) trading')
            
            # Fix markdown cells too
            elif cell['cell_type'] == 'markdown':
                for i, line in enumerate(cell['source']):
                    if 'SOL (Solana)' in line:
                        cell['source'][i] = line.replace('SOL (Solana)', 'ADA (Cardano)')
                    elif 'Solana' in line:
                        cell['source'][i] = line.replace('Solana', 'Cardano')
                    elif 'SOL' in line and 'SOLUTION' not in line.upper():
                        cell['source'][i] = line.replace('SOL', 'ADA')
        
        # Write the corrected notebook
        with open(filename, 'w') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print("✅ Fixed ADA notebook data loading and references")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing ADA notebook: {e}")
        return False

if __name__ == "__main__":
    fix_ada_notebook_data_loading()