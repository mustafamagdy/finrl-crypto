import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def download_crypto_test_data(symbols, timeframe='1m', days=30):  # Test with 30 days
    """
    Download cryptocurrency data from Binance (test version)
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    
    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_data = []
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        
        since = int(start_time.timestamp() * 1000)
        symbol_data = []
        
        while since < int(end_time.timestamp() * 1000):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                
                if not ohlcv:
                    break
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['tic'] = symbol.replace('/', '')
                
                symbol_data.append(df)
                since = ohlcv[-1][0] + 1
                
                print(f"  Downloaded batch: {len(ohlcv)} records, last date: {df['datetime'].iloc[-1]}")
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                break
        
        if symbol_data:
            symbol_df = pd.concat(symbol_data, ignore_index=True)
            symbol_df = symbol_df.drop_duplicates(subset=['timestamp'])
            symbol_df = symbol_df.sort_values('timestamp')
            all_data.append(symbol_df)
            print(f"Total records for {symbol}: {len(symbol_df)}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.rename(columns={'datetime': 'date'})
        final_df = final_df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        final_df = final_df.sort_values(['date', 'tic'])
        
        return final_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    print("Testing crypto data download...")
    crypto_df = download_crypto_test_data(crypto_symbols, timeframe='1m', days=30)
    
    if not crypto_df.empty:
        crypto_df.to_csv('crypto_test_data.csv', index=False)
        
        print(f"\nTest completed!")
        print(f"Total records: {len(crypto_df)}")
        print(f"Date range: {crypto_df['date'].min()} to {crypto_df['date'].max()}")
        print(f"Symbols: {crypto_df['tic'].unique()}")
        print("\nSample data:")
        print(crypto_df.head())
    else:
        print("Failed to download test data!")