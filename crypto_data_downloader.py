import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def download_crypto_data(symbols, timeframe='1m', days=1825):  # 5 years = ~1825 days
    """
    Download cryptocurrency data from Binance
    
    Args:
        symbols: List of crypto symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframe: Timeframe for data (1m, 5m, 1h, 1d)
        days: Number of days to download
    
    Returns:
        DataFrame with crypto data
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'rateLimit': 1200,  # Rate limit in milliseconds
        'enableRateLimit': True,
    })
    
    # Calculate start time (5 years ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_data = []
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        
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
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                time.sleep(1)
                break
        
        if symbol_data:
            symbol_df = pd.concat(symbol_data, ignore_index=True)
            symbol_df = symbol_df.drop_duplicates(subset=['timestamp'])
            symbol_df = symbol_df.sort_values('timestamp')
            all_data.append(symbol_df)
            print(f"Downloaded {len(symbol_df)} records for {symbol}")
    
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

def save_crypto_data(df, filename='crypto_data.csv'):
    """Save crypto data to CSV file"""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_crypto_data(filename='crypto_data.csv'):
    """Load crypto data from CSV file"""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        return None

if __name__ == "__main__":
    # Major crypto symbols
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    print("Starting crypto data download...")
    print(f"Symbols: {crypto_symbols}")
    print("Timeframe: 1 minute")
    print("Period: Last 5 years")
    
    # Download data
    crypto_df = download_crypto_data(crypto_symbols, timeframe='1m', days=1825)
    
    if not crypto_df.empty:
        # Save to CSV
        save_crypto_data(crypto_df, 'crypto_1min_5years.csv')
        
        print("\nData Summary:")
        print(f"Total records: {len(crypto_df)}")
        print(f"Date range: {crypto_df['date'].min()} to {crypto_df['date'].max()}")
        print(f"Symbols: {crypto_df['tic'].unique()}")
        print("\nFirst few records:")
        print(crypto_df.head())
    else:
        print("No data downloaded!")