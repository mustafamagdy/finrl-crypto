#!/usr/bin/env python3

import json

def create_portfolio_notebook(template_file, output_file, assets, portfolio_name, description):
    """Create a portfolio notebook from template."""
    
    try:
        # Load template
        with open(template_file, 'r') as f:
            notebook = json.load(f)
        
        # Update notebook content
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and len(cell['source']) > 0:
                for i, line in enumerate(cell['source']):
                    # Update title and description
                    if 'Solana (SOL)' in line:
                        cell['source'][i] = line.replace('Solana (SOL)', f'{portfolio_name} ({", ".join(assets)})')
                    elif 'SOL' in line and 'SOLUSDT' not in line and 'sol_' not in line:
                        cell['source'][i] = line.replace('SOL', f'{", ".join(assets)}')
                    elif 'sol_training' in line:
                        cell['source'][i] = line.replace('sol_training', f'{portfolio_name.lower().replace(" ", "_")}_portfolio')
                    elif 'high-performance trading' in line:
                        cell['source'][i] = line.replace('high-performance trading', description)
                    
            elif cell['cell_type'] == 'code':
                for i, line in enumerate(cell['source']):
                    # Update asset list
                    if "assets = ['SOL']" in line:
                        assets_str = str(assets).replace("'", '"')
                        cell['source'][i] = line.replace("assets = ['SOL']", f"assets = {assets_str}")
                    elif 'SOLUSDT' in line:
                        cell['source'][i] = line.replace('SOLUSDT', f'{assets[0]}USDT')
                    elif 'sol_' in line:
                        prefix = portfolio_name.lower().replace(" ", "_")
                        cell['source'][i] = line.replace('sol_', f'{prefix}_')
                    elif f'"{assets[0]}"' in line and 'Target Assets:' in line:
                        cell['source'][i] = line.replace(f'"{assets[0]}"', f'"{", ".join(assets)}"')
        
        # Write the new notebook
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✅ Created {portfolio_name} portfolio notebook: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating {portfolio_name} notebook: {e}")
        return False

def main():
    """Create all portfolio notebooks."""
    
    # Template file
    template_file = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/sol_training.ipynb"
    
    # Portfolio configurations
    portfolios = [
        {
            'name': 'Balanced Portfolio',
            'assets': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'],
            'description': 'balanced multi-asset portfolio trading',
            'filename': 'balanced_portfolio.ipynb'
        },
        {
            'name': 'Full Portfolio',
            'assets': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC', 'DOT', 'LINK'],
            'description': 'comprehensive 8-asset portfolio trading',
            'filename': 'full_portfolio.ipynb'
        }
    ]
    
    # Create portfolios
    output_dir = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/multi_asset_portfolios/"
    
    for portfolio in portfolios:
        output_file = output_dir + portfolio['filename']
        create_portfolio_notebook(
            template_file=template_file,
            output_file=output_file,
            assets=portfolio['assets'],
            portfolio_name=portfolio['name'],
            description=portfolio['description']
        )

if __name__ == "__main__":
    main()