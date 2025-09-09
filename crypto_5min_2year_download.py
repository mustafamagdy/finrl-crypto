import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def download_crypto_5min_2years(symbols, timeframe='5m', days=730):  # 2 years = ~730 days
    """
    Download cryptocurrency data from Binance - 5 minute data for 2 years
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # Calculate start time (2 years ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"Downloading {timeframe} data from {start_time.date()} to {end_time.date()}")
    
    all_data = []
    
    for symbol in symbols:
        print(f"\nDownloading {timeframe} data for {symbol}...")
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        
        symbol_data = []
        batch_count = 0
        
        while since < int(end_time.timestamp() * 1000):
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                
                if not ohlcv:
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['tic'] = symbol.replace('/', '')
                
                symbol_data.append(df)
                batch_count += 1
                
                # Update since to the last timestamp
                since = ohlcv[-1][0] + 1
                
                # Progress reporting every 50 batches
                if batch_count % 50 == 0:
                    print(f"  Downloaded {batch_count} batches, last date: {df['datetime'].iloc[-1]}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                time.sleep(2)
                continue
        
        if symbol_data:
            symbol_df = pd.concat(symbol_data, ignore_index=True)
            symbol_df = symbol_df.drop_duplicates(subset=['timestamp'])
            symbol_df = symbol_df.sort_values('timestamp')
            all_data.append(symbol_df)
            print(f"Total records for {symbol}: {len(symbol_df):,}")
            print(f"Date range: {symbol_df['datetime'].min()} to {symbol_df['datetime'].max()}")
    
    if all_data:
        # Combine all symbols
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Rename columns to match FinRL format
        final_df = final_df.rename(columns={'datetime': 'date'})
        final_df = final_df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        final_df = final_df.sort_values(['date', 'tic'])
        
        return final_df
    else:
        return pd.DataFrame()

def calculate_expected_records(timeframe='5m', days=730):
    """Calculate expected number of records"""
    minutes_per_day = 24 * 60  # 1440 minutes per day
    if timeframe == '5m':
        records_per_day = minutes_per_day // 5  # 288 records per day per symbol
    elif timeframe == '1m':
        records_per_day = minutes_per_day  # 1440 records per day per symbol
    elif timeframe == '1h':
        records_per_day = 24  # 24 records per day per symbol
    else:
        records_per_day = 288  # default to 5m
    
    return records_per_day * days

if __name__ == "__main__":
    # Major crypto symbols
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    # Calculate expected records
    expected_per_symbol = calculate_expected_records('5m', 730)
    total_expected = expected_per_symbol * len(crypto_symbols)
    
    print("Starting crypto data download - 5-minute data for 2 years")
    print("="*60)
    print(f"Symbols: {crypto_symbols}")
    print(f"Timeframe: 5 minutes")
    print(f"Period: Last 2 years (~730 days)")
    print(f"Expected records per symbol: {expected_per_symbol:,}")
    print(f"Total expected records: {total_expected:,}")
    print("="*60)
    
    start_download_time = datetime.now()
    
    # Download data
    crypto_df = download_crypto_5min_2years(crypto_symbols, timeframe='5m', days=730)
    
    end_download_time = datetime.now()
    download_duration = end_download_time - start_download_time
    
    if not crypto_df.empty:
        # Save to CSV
        filename = 'crypto_5min_2years.csv'
        crypto_df.to_csv(filename, index=False)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"File saved: {filename}")
        print(f"Total records: {len(crypto_df):,}")
        print(f"Records per symbol: {len(crypto_df) // len(crypto_df['tic'].unique()):,}")
        print(f"Date range: {crypto_df['date'].min()} to {crypto_df['date'].max()}")
        print(f"Symbols: {sorted(crypto_df['tic'].unique())}")
        print(f"Download duration: {download_duration}")
        
        # Data quality checks
        print(f"\nData Quality Checks:")
        print(f"  Unique dates: {crypto_df['date'].nunique():,}")
        print(f"  Missing values: {crypto_df.isnull().sum().sum()}")
        print(f"  Duplicate records: {crypto_df.duplicated().sum()}")
        
        # Sample data
        print(f"\nFirst 5 records:")
        print(crypto_df.head())
        
        print(f"\nLast 5 records:")
        print(crypto_df.tail())
        
    else:
        print("❌ No data downloaded! Check your connection and try again.")