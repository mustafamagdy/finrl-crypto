#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test if we can load ADA from CSV
csv_files = ['crypto_5currencies_2years.csv', 'crypto_5min_2years.csv', 'crypto_test_data.csv']

for csv_file in csv_files:
    try:
        temp_df = pd.read_csv(csv_file)
        if 'tic' in temp_df.columns:
            ada_tickers = [t for t in temp_df['tic'].unique() if 'ADA' in t]
            if ada_tickers:
                ada_ticker = ada_tickers[0]
                df = temp_df[temp_df['tic'] == ada_ticker].copy()
                df['tic'] = 'ADA'
                print(f'✅ Test successful! Found ADA data in {csv_file}')
                print(f'   Records: {len(df):,}')
                print(f'   Date range: {df["date"].min()} to {df["date"].max()}')
                print(f'   Price range: ${df["close"].min():.4f} - ${df["close"].max():.4f}')
                break
    except Exception as e:
        print(f'❌ Error with {csv_file}: {e}')
        continue
else:
    print('❌ No ADA data found in CSV files - will use yfinance download')