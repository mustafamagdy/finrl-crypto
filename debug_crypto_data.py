import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# Load crypto data
df = pd.read_csv('crypto_test_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'tic']).reset_index(drop=True)

print("Original data:")
print(df.head())
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Add technical indicators
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=['rsi_30', 'macd', 'cci_30', 'dx_30'],
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False
)

processed_df = fe.preprocess_data(df)

print("\nProcessed data:")
print(processed_df.head())
print(f"Shape: {processed_df.shape}")
print(f"Columns: {processed_df.columns.tolist()}")

# Check for any issues
print(f"\nData types:")
print(processed_df.dtypes)

# Check first day data for one symbol
first_day_data = processed_df[processed_df['tic'] == 'BTCUSDT'].head()
print(f"\nFirst day BTCUSDT data:")
print(first_day_data)

# Check if we have the required columns
required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
print(f"\nRequired columns present: {all(col in processed_df.columns for col in required_cols)}")