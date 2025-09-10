import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def download_5_cryptos_data():
    """Download 5-minute data for 5 additional popular cryptocurrencies for 2 years"""
    
    print("🚀 DOWNLOADING 5 ADDITIONAL CRYPTOCURRENCIES")
    print("="*60)
    
    # 5 popular cryptocurrencies with high volume
    symbols = [
        'ADAUSDT',   # Cardano
        'SOLUSDT',   # Solana  
        'MATICUSDT', # Polygon
        'DOTUSDT',   # Polkadot
        'LINKUSDT'   # Chainlink
    ]
    
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'rateLimit': 1200,  # 1.2 seconds between requests
        'enableRateLimit': True,
        'sandbox': False,
    })
    
    # Time range - same as original (2 years)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=730)  # 2 years
    
    print(f"📅 Time Range: {start_time} to {end_time}")
    print(f"⏰ Timeframe: 5 minutes")
    print(f"💰 Symbols: {symbols}")
    
    all_data = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n📊 [{i}/5] Downloading {symbol}...")
        
        try:
            # Calculate expected records (288 per day * 730 days)
            expected_records = 288 * 730
            print(f"   📈 Expected records: ~{expected_records:,}")
            
            # Fetch OHLCV data
            current_time = start_time
            symbol_data = []
            
            while current_time < end_time:
                # Fetch data in chunks
                since_ms = int(current_time.timestamp() * 1000)
                
                try:
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, 
                        '5m', 
                        since=since_ms, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    # Process the data
                    for candle in ohlcv:
                        timestamp_ms, open_price, high, low, close, volume = candle
                        dt = datetime.fromtimestamp(timestamp_ms / 1000)
                        
                        if dt >= end_time:
                            break
                            
                        symbol_data.append({
                            'date': dt,
                            'tic': symbol,
                            'open': float(open_price),
                            'high': float(high),
                            'low': float(low),
                            'close': float(close),
                            'volume': float(volume) if volume else 0
                        })
                    
                    # Update current_time to the last timestamp + 5 minutes
                    if ohlcv:
                        last_timestamp = ohlcv[-1][0]
                        current_time = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=5)
                        
                        # Progress update
                        progress = ((current_time - start_time).days / 730) * 100
                        print(f"   ⏳ Progress: {progress:.1f}% ({len(symbol_data):,} records)")
                    
                    # Rate limiting
                    time.sleep(1.5)
                    
                except Exception as e:
                    print(f"   ❌ Error fetching batch: {e}")
                    current_time += timedelta(hours=1)  # Skip ahead
                    continue
            
            print(f"   ✅ {symbol}: {len(symbol_data):,} records downloaded")
            all_data.extend(symbol_data)
            
            # Small break between symbols
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Failed to download {symbol}: {e}")
            continue
    
    # Create DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Save to CSV
        filename = 'crypto_5currencies_2years.csv'
        df.to_csv(filename, index=False)
        
        print(f"\n✅ DOWNLOAD COMPLETE")
        print(f"📄 File: {filename}")
        print(f"📊 Total records: {len(df):,}")
        print(f"💰 Symbols: {sorted(df['tic'].unique())}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Records per symbol
        for symbol in sorted(df['tic'].unique()):
            count = len(df[df['tic'] == symbol])
            print(f"   {symbol}: {count:,} records")
        
        return filename
    else:
        print("❌ No data downloaded")
        return None

def main():
    """Main execution"""
    try:
        filename = download_5_cryptos_data()
        if filename:
            print(f"\n🎉 Successfully downloaded 5 cryptocurrencies data!")
            print(f"📁 Ready for training: {filename}")
        else:
            print("❌ Download failed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()