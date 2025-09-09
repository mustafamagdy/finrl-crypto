import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def download_longer_crypto_data(symbols, timeframe='1h', days=1825):  # Use 1h instead of 1m for 5 years
    """
    Download cryptocurrency data from Binance
    Using hourly data for longer periods to avoid too much data
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # Calculate start time (5 years ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_data = []
    
    for symbol in symbols:
        print(f"Downloading {timeframe} data for {symbol}...")
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        
        symbol_data = []
        
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
                
                # Update since to the last timestamp
                since = ohlcv[-1][0] + 1
                
                # Rate limiting
                time.sleep(0.1)
                
                print(f"  Downloaded batch: {len(ohlcv)} records, last date: {df['datetime'].iloc[-1]}")
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                time.sleep(1)
                break
        
        if symbol_data:
            symbol_df = pd.concat(symbol_data, ignore_index=True)
            symbol_df = symbol_df.drop_duplicates(subset=['timestamp'])
            symbol_df = symbol_df.sort_values('timestamp')
            all_data.append(symbol_df)
            print(f"Total records for {symbol}: {len(symbol_df)}")
    
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

if __name__ == "__main__":
    # Major crypto symbols
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    print("Starting crypto data download - 5 years hourly data...")
    print(f"Symbols: {crypto_symbols}")
    print("Timeframe: 1 hour")
    print("Period: Last 5 years")
    
    # Download data
    crypto_df = download_longer_crypto_data(crypto_symbols, timeframe='1h', days=1825)
    
    if not crypto_df.empty:
        # Save to CSV
        crypto_df.to_csv('crypto_5year_hourly.csv', index=False)
        
        print("\nData Summary:")
        print(f"Total records: {len(crypto_df)}")
        print(f"Date range: {crypto_df['date'].min()} to {crypto_df['date'].max()}")
        print(f"Symbols: {crypto_df['tic'].unique()}")
        print(f"Records per symbol: {len(crypto_df) // len(crypto_df['tic'].unique())}")
        print("\nFirst few records:")
        print(crypto_df.head())
    else:
        print("No data downloaded!")