import yfinance as yf
import os
from datetime import datetime

print("\n" + "="*70)
print("DOWNLOADING 7 YEARS OF STOCK DATA")
print("="*70)

os.makedirs('data/stocks', exist_ok=True)

stocks = {
    'IOC': 'IOC.NS',
    'AAPL': 'AAPL',
    'ITC': 'ITC.NS'
}

for name, ticker in stocks.items():
    print(f"\nDownloading {name} ({ticker})...")
    try:
        data = yf.download(ticker, period='7y', progress=True)
        filepath = f'data/stocks/{name}.csv'
        data.to_csv(filepath)
        print(f"✓ {name} saved: {len(data)} rows")
        print(f"  Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")

print("\n" + "="*70)
print("✓ All data downloaded successfully!")
print("="*70 + "\n")