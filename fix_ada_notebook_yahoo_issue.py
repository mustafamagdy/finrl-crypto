#!/usr/bin/env python3

import json

def fix_ada_notebook_yahoo_issue():
    """Fix ADA notebook to avoid YahooDownloader issues and use direct yfinance."""
    
    filename = "/Users/mustafamagdy/Home/Dev/my-dev/finrl-bot01/notebooks/individual_models/ada_training.ipynb"
    
    try:
        # Read the notebook
        with open(filename, 'r') as f:
            notebook = json.load(f)
        
        # Find the data loading cell and fix it
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and any('def load_ada_data():' in line for line in cell.get('source', [])):
                # Replace with a more robust data loading function
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
                    "                    print(f\"✅ Loaded ADA data from {csv_file} (ticker: {ada_ticker})\")\n",
                    "                    break\n",
                    "        except FileNotFoundError:\n",
                    "            continue\n",
                    "    \n",
                    "    if df is None:\n",
                    "        print(\"📥 CSV not found, downloading fresh ADA data using yfinance...\")\n",
                    "        # Use yfinance directly to avoid YahooDownloader issues\n",
                    "        import yfinance as yf\n",
                    "        \n",
                    "        try:\n",
                    "            # Download 2 years of 5-minute data\n",
                    "            ticker = yf.Ticker(\"ADA-USD\")\n",
                    "            \n",
                    "            # Try different periods to get maximum data\n",
                    "            periods = ['2y', '1y', '6mo', '3mo']\n",
                    "            \n",
                    "            for period in periods:\n",
                    "                try:\n",
                    "                    hist_data = ticker.history(period=period, interval='5m')\n",
                    "                    if len(hist_data) > 0:\n",
                    "                        print(f\"✅ Downloaded {len(hist_data)} rows of ADA data ({period} period)\")\n",
                    "                        break\n",
                    "                except Exception as e:\n",
                    "                    print(f\"⚠️ Failed to download {period} data: {e}\")\n",
                    "                    continue\n",
                    "            \n",
                    "            if len(hist_data) == 0:\n",
                    "                raise ValueError(\"No data could be downloaded\")\n",
                    "            \n",
                    "            # Convert to FinRL format\n",
                    "            df = hist_data.reset_index()\n",
                    "            df['tic'] = 'ADA'\n",
                    "            \n",
                    "            # Rename columns to match FinRL format\n",
                    "            column_mapping = {\n",
                    "                'Datetime': 'date',\n",
                    "                'Open': 'open',\n",
                    "                'High': 'high',\n",
                    "                'Low': 'low',\n",
                    "                'Close': 'close',\n",
                    "                'Volume': 'volume'\n",
                    "            }\n",
                    "            \n",
                    "            for old_name, new_name in column_mapping.items():\n",
                    "                if old_name in df.columns:\n",
                    "                    df[new_name] = df[old_name]\n",
                    "            \n",
                    "        except Exception as e:\n",
                    "            print(f\"❌ Failed to download ADA data: {e}\")\n",
                    "            # Create synthetic data as last resort\n",
                    "            print(\"🔧 Creating synthetic ADA data for testing...\")\n",
                    "            \n",
                    "            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5min')\n",
                    "            np.random.seed(42)\n",
                    "            \n",
                    "            # Generate realistic ADA price movement\n",
                    "            initial_price = 0.35\n",
                    "            returns = np.random.normal(0, 0.002, len(dates))  # 0.2% volatility\n",
                    "            prices = [initial_price]\n",
                    "            \n",
                    "            for ret in returns[1:]:\n",
                    "                prices.append(prices[-1] * (1 + ret))\n",
                    "            \n",
                    "            df = pd.DataFrame({\n",
                    "                'date': dates,\n",
                    "                'open': prices,\n",
                    "                'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],\n",
                    "                'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],\n",
                    "                'close': prices,\n",
                    "                'volume': np.random.lognormal(15, 0.5, len(dates)),\n",
                    "                'tic': 'ADA'\n",
                    "            })\n",
                    "    \n",
                    "    # Ensure date column is datetime\n",
                    "    if 'date' not in df.columns:\n",
                    "        if 'Date' in df.columns:\n",
                    "            df['date'] = pd.to_datetime(df['Date'])\n",
                    "        elif 'Datetime' in df.columns:\n",
                    "            df['date'] = pd.to_datetime(df['Datetime'])\n",
                    "        else:\n",
                    "            df.reset_index(inplace=True)\n",
                    "            df['date'] = pd.to_datetime(df.index)\n",
                    "    else:\n",
                    "        df['date'] = pd.to_datetime(df['date'])\n",
                    "    \n",
                    "    # Ensure all required columns exist\n",
                    "    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']\n",
                    "    \n",
                    "    for col in required_cols:\n",
                    "        if col not in df.columns:\n",
                    "            if col == 'tic':\n",
                    "                df[col] = 'ADA'\n",
                    "            else:\n",
                    "                print(f\"⚠️ Missing column {col}, using close price as fallback\")\n",
                    "                df[col] = df['close'] if 'close' in df.columns else 0.35\n",
                    "    \n",
                    "    # Select only required columns\n",
                    "    df = df[required_cols]\n",
                    "    \n",
                    "    # Sort by date and clean\n",
                    "    df = df.sort_values('date').reset_index(drop=True)\n",
                    "    df = df.dropna()\n",
                    "    \n",
                    "    # Ensure we have data\n",
                    "    if len(df) == 0:\n",
                    "        raise ValueError(\"No ADA data available after processing\")\n",
                    "    \n",
                    "    print(f\"\\n📊 ADA Data Summary:\")\n",
                    "    print(f\"   Shape: {df.shape}\")\n",
                    "    print(f\"   Date range: {df['date'].min()} to {df['date'].max()}\")\n",
                    "    print(f\"   Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}\")\n",
                    "    print(f\"   Average volume: {df['volume'].mean():,.0f}\")\n",
                    "    \n",
                    "    # Cardano-specific market analysis\n",
                    "    price_changes = df['close'].pct_change().dropna()\n",
                    "    high_vol_periods = price_changes[abs(price_changes) > price_changes.std() * 2]\n",
                    "    \n",
                    "    print(f\"\\n🔥 ADA Market Characteristics:\")\n",
                    "    print(f\"   Average return: {price_changes.mean()*100:.4f}%\")\n",
                    "    print(f\"   Volatility: {price_changes.std()*100:.4f}%\")\n",
                    "    print(f\"   High volatility periods: {len(high_vol_periods)} ({len(high_vol_periods)/len(price_changes)*100:.1f}%)\")\n",
                    "    print(f\"   Best period: +{price_changes.max()*100:.2f}%\")\n",
                    "    print(f\"   Worst period: {price_changes.min()*100:.2f}%\")\n",
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
        
        print("✅ Fixed ADA notebook to avoid YahooDownloader issues")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing ADA notebook: {e}")
        return False

if __name__ == "__main__":
    fix_ada_notebook_yahoo_issue()