import yfinance as yf
import os
from core.predictor import StockPredictorEnsemble

print("\n" + "="*70)
print("ITC STOCK PREDICTION - DOWNLOAD & ANALYZE")
print("="*70)

# Step 1: Download data
print("\nSTEP 1: Downloading ITC data...")
os.makedirs('data/stocks', exist_ok=True)

try:
    itc = yf.download('ITC.NS', period='7y', progress=True)
    itc.to_csv('data/stocks/ITC.csv')
    print(f"✓ ITC data downloaded and saved")
    print(f"✓ Total trading days: {len(itc)}")
except Exception as e:
    print(f"✗ Error downloading: {e}")
    exit()

# Step 2: Make prediction
print("\nSTEP 2: Analyzing ITC stock...")

predictor = StockPredictorEnsemble()
result = predictor.predict("ITC")

# Step 3: Display results
if result:
    print("\n" + "="*70)
    print("ITC PREDICTION RESULTS")
    print("="*70)
    
    print(f"\n✓ Stock: {result['ticker']} (Indian Tobacco Company)")
    print(f"✓ Current Price: ₹{result['current_price']:.2f}")
    
    print(f"\n📊 PREDICTION SIGNAL:")
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    
    print(f"\n💰 PRICE TARGETS:")
    print(f"  Entry Price: ₹{result['current_price']:.2f}")
    print(f"  Target Price: ₹{result['target']:.2f}")
    print(f"  Stop Loss: ₹{result['stop_loss']:.2f}")
    
    if result['signal'] == "BUY":
        profit = result['target'] - result['current_price']
        loss = result['current_price'] - result['stop_loss']
        print(f"  Potential Profit: ₹{profit:.2f} per share")
        print(f"  Potential Loss: ₹{loss:.2f} per share")
    else:
        profit = result['current_price'] - result['target']
        loss = result['stop_loss'] - result['current_price']
        print(f"  Potential Profit: ₹{profit:.2f} per share")
        print(f"  Potential Loss: ₹{loss:.2f} per share")
    
    print(f"\n📈 TECHNICAL INDICATORS:")
    print(f"  RSI: {result['rsi']:.2f}")
    print(f"  MACD: {result['macd']:.4f}")
    print(f"  Volatility: {result['volatility']:.2f}%")
    
    print("\n" + "="*70)
    print("✓ Analysis Complete!")
    print("="*70 + "\n")
else:
    print("✗ Failed to make prediction")
