import yfinance as yf
import os

os.makedirs('data/stocks', exist_ok=True)

print('Downloading IOC...')
ioc = yf.download('IOC.NS', period='7y', progress=True)
ioc.to_csv('data/stocks/IOC.csv')
print('✓ IOC saved')

print('\nDownloading AAPL...')
aapl = yf.download('AAPL', period='7y', progress=True)
aapl.to_csv('data/stocks/AAPL.csv')
print('✓ AAPL saved')

print('\n✓ All data downloaded!')