#!/usr/bin/env python3

import json

def fix_ada_notebook_final():
    """Final fix for ADA notebook to handle ADAUSDT ticker correctly."""
    
    filename = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/ada_training.ipynb"
    
    try:
        # Read the notebook
        with open(filename, 'r') as f:
            notebook = json.load(f)
        
        # Find the data loading cell and fix it
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and any('csv_files = [' in line for line in cell.get('source', [])):
                # Replace the entire data loading function
                new_source = [
                    "# Section 2: ADA Data Loading and Market Analysis\n",
                    "def load_ada_data():\n",
                    "    \"\"\"Load ADA cryptocurrency data with Cardano-specific preprocessing\"\"\"\n",
                    "    \n",
                    "    # Try to load from existing CSV data first\n",
                    "    csv_files = ['crypto_5currencies_2years.csv', 'crypto_5min_2years.csv', 'crypto_test_data.csv']\n",
                    "    df = None\n",
                    "    \n",
                    "    for csv_file in csv_files:\n",
                    "        try:\n",
                    "            temp_df = pd.read_csv(csv_file)\n",
                    "            if 'tic' in temp_df.columns:\n",
                    "                # Look for both ADA and ADAUSDT\n",
                    "                ada_tickers = [t for t in temp_df['tic'].unique() if 'ADA' in t]\n",
                    "                if ada_tickers:\n",
                    "                    ada_ticker = ada_tickers[0]  # Use the first ADA ticker found\n",
                    "                    df = temp_df[temp_df['tic'] == ada_ticker].copy()\n",
                    "                    # Standardize ticker to ADA\n",
                    "                    df['tic'] = 'ADA'\n",
                    "                    print(f\"Loaded ADA data from {csv_file} (ticker: {ada_ticker})\")\n",
                    "                    break\n",
                    "        except FileNotFoundError:\n",
                    "            continue\n",
                    "    \n",
                    "    if df is None:\n",
                    "        print(\"CSV not found, downloading fresh ADA data...\")\n",
                    "        # Fallback to download if CSV doesn't exist\n",
                    "        end_date = datetime.now()\n",
                    "        start_date = end_date - timedelta(days=365*2)  # 2 years\n",
                    "        \n",
                    "        df = YahooDownloader(start_date=start_date.strftime('%Y-%m-%d'),\n",
                    "                           end_date=end_date.strftime('%Y-%m-%d'),\n",
                    "                           ticker_list=['ADA-USD']).fetch_data()\n",
                    "    \n",
                    "    # Standardize column names\n",
                    "    if 'open_time' in df.columns:\n",
                    "        df['date'] = pd.to_datetime(df['open_time'])\n",
                    "    elif 'date' not in df.columns:\n",
                    "        df.reset_index(inplace=True)\n",
                    "        if 'Date' in df.columns:\n",
                    "            df['date'] = pd.to_datetime(df['Date'])\n",
                    "        else:\n",
                    "            df['date'] = pd.to_datetime(df['date'])\n",
                    "    else:\n",
                    "        df['date'] = pd.to_datetime(df['date'])\n",
                    "    \n",
                    "    # Required columns for FinRL\n",
                    "    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']\n",
                    "    \n",
                    "    # Map columns if needed\n",
                    "    column_mapping = {\n",
                    "        'open_price': 'open',\n",
                    "        'high_price': 'high', \n",
                    "        'low_price': 'low',\n",
                    "        'close_price': 'close',\n",
                    "        'volume': 'volume'\n",
                    "    }\n",
                    "    \n",
                    "    for old_name, new_name in column_mapping.items():\n",
                    "        if old_name in df.columns:\n",
                    "            df[new_name] = df[old_name]\n",
                    "    \n",
                    "    # Ensure we have all required columns\n",
                    "    available_cols = [col for col in required_cols if col in df.columns]\n",
                    "    if 'tic' in df.columns:\n",
                    "        available_cols.append('tic')\n",
                    "    \n",
                    "    df = df[available_cols]\n",
                    "    \n",
                    "    # Add ticker if not present\n",
                    "    if 'tic' not in df.columns:\n",
                    "        df['tic'] = 'ADA'\n",
                    "    \n",
                    "    # Sort by date\n",
                    "    df = df.sort_values('date').reset_index(drop=True)\n",
                    "    \n",
                    "    # Basic data cleaning\n",
                    "    df = df.dropna()\n",
                    "    \n",
                    "    if len(df) == 0:\n",
                    "        raise ValueError(\"No ADA data found after processing\")\n",
                    "    \n",
                    "    print(f\"📊 ADA Data shape: {df.shape}\")\n",
                    "    print(f\"📅 Date range: {df['date'].min()} to {df['date'].max()}\")\n",
                    "    print(f\"💰 Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}\")\n",
                    "    print(f\"📈 Average daily volume: {df['volume'].mean():,.0f}\")\n",
                    "    \n",
                    "    # Cardano-specific market analysis\n",
                    "    price_changes = df['close'].pct_change().dropna()\n",
                    "    high_vol_periods = price_changes[abs(price_changes) > price_changes.std() * 2]\n",
                    "    \n",
                    "    print(f\"\\n🔥 ADA Market Characteristics:\")\n",
                    "    print(f\"   Average 5min return: {price_changes.mean()*100:.4f}%\")\n",
                    "    print(f\"   Volatility (std): {price_changes.std()*100:.4f}%\")\n",
                    "    print(f\"   High volatility periods: {len(high_vol_periods)} ({len(high_vol_periods)/len(price_changes)*100:.1f}%)\")\n",
                    "    print(f\"   Max single period gain: {price_changes.max()*100:.2f}%\")\n",
                    "    print(f\"   Max single period loss: {price_changes.min()*100:.2f}%\")\n",
                    "    \n",
                    "    return df\n",
                    "\n",
                    "# Load the ADA data\n",
                    "raw_data = load_ada_data()\n",
                    "\n",
                    "# Display basic statistics\n",
                    "raw_data.describe()"
                ]
                cell['source'] = new_source
                break
        
        # Write the corrected notebook
        with open(filename, 'w') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print("✅ Final fix applied to ADA notebook - data loading should work now")
        return True
        
    except Exception as e:
        print(f"❌ Error applying final fix: {e}")
        return False

if __name__ == "__main__":
    fix_ada_notebook_final()