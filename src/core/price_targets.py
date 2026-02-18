import pandas as pd
import numpy as np

class PriceTargetCalculator:
    """Calculate professional trading prices using 7-year volatility"""
    
    def __init__(self):
        print("✓ Price Target Calculator initialized")
    
    def load_data(self, ticker):
        """Load stock data"""
        try:
            df = pd.read_csv(f'data/stocks/{ticker}.csv', index_col=0, parse_dates=True)
            
            # Keep numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in numeric_cols if col in df.columns]]
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            return df
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (volatility measure)"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1]
    
    def calculate_volatility_percentages(self, df):
        """Calculate volatility for different periods"""
        current_price = df['Close'].iloc[-1]
        
        # Short-term volatility (20 days)
        vol_20 = df['Close'].pct_change().tail(20).std() * 100
        
        # Medium-term volatility (252 days = 1 year)
        vol_252 = df['Close'].pct_change().tail(252).std() * 100
        
        # Long-term volatility (all data = 7 years)
        vol_all = df['Close'].pct_change().std() * 100
        
        # Average volatility (weighted)
        avg_volatility = (vol_20 * 0.3 + vol_252 * 0.4 + vol_all * 0.3)
        
        return {
            'vol_short': vol_20,
            'vol_medium': vol_252,
            'vol_long': vol_all,
            'vol_avg': avg_volatility
        }
    
    def calculate_support_resistance(self, df):
        """Calculate support & resistance levels"""
        recent_data = df['Close'].tail(252)  # Last 1 year
        
        # Support (1-year low)
        support = recent_data.min()
        
        # Resistance (1-year high)
        resistance = recent_data.max()
        
        # Pivot point
        pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
        
        return {
            'support': support,
            'resistance': resistance,
            'pivot': pivot
        }
    
    def calculate_targets_and_stops(self, df, signal):
        """Calculate entry, target, and stop loss prices"""
        current_price = df['Close'].iloc[-1]
        
        # Get volatility
        volatilities = self.calculate_volatility_percentages(df)
        avg_vol = volatilities['vol_avg']
        
        # Get ATR
        atr = self.calculate_atr(df, period=14)
        
        # Get support/resistance
        levels = self.calculate_support_resistance(df)
        
        # Calculate based on signal
        if signal == "BUY":
            # For BUY: Target is ABOVE current price
            target = current_price + (atr * 2.0)  # 2x ATR
            stop_loss = current_price - (atr * 0.75)  # 0.75x ATR below
            
            # Ensure target is not above resistance
            target = min(target, levels['resistance'])
        else:
            # For SELL: Target is BELOW current price
            target = current_price - (atr * 2.0)  # 2x ATR
            stop_loss = current_price + (atr * 0.75)  # 0.75x ATR above
            
            # Ensure target is not below support
            target = max(target, levels['support'])
        
        return {
            'current_price': current_price,
            'entry': current_price,
            'target': target,
            'stop_loss': stop_loss,
            'atr': atr,
            'support': levels['support'],
            'resistance': levels['resistance'],
            'pivot': levels['pivot'],
            'volatilities': volatilities
        }
    
    def calculate_risk_reward(self, current_price, target, stop_loss):
        """Calculate risk/reward ratio"""
        if current_price == stop_loss:
            return None
        
        potential_profit = abs(target - current_price)
        potential_loss = abs(current_price - stop_loss)
        
        if potential_loss == 0:
            return None
        
        risk_reward_ratio = potential_profit / potential_loss
        
        return {
            'profit': potential_profit,
            'loss': potential_loss,
            'ratio': risk_reward_ratio,
            'profit_percent': (potential_profit / current_price) * 100,
            'loss_percent': (potential_loss / current_price) * 100
        }
    
    def calculate_position_size(self, current_price, stop_loss, risk_amount=10000):
        """Calculate how many shares to buy for given risk"""
        risk_per_share = abs(current_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        
        return {
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'position_size': int(position_size),
            'total_investment': int(position_size) * current_price
        }
    
    def generate_report(self, ticker, signal):
        """Generate complete price target report"""
        print(f"\n{'='*80}")
        print(f"PRICE TARGET ANALYSIS - {ticker}")
        print(f"{'='*80}")
        
        # Load data
        df = self.load_data(ticker)
        if df is None:
            print("✗ Failed to load data")
            return None
        
        # Calculate targets
        targets = self.calculate_targets_and_stops(df, signal)
        
        # Calculate risk/reward
        rr = self.calculate_risk_reward(
            targets['current_price'],
            targets['target'],
            targets['stop_loss']
        )
        
        # Calculate position size
        position = self.calculate_position_size(
            targets['current_price'],
            targets['stop_loss'],
            risk_amount=10000
        )
        
        # Display report
        print(f"\n✓ Stock: {ticker}")
        print(f"✓ Signal: {signal}")
        print(f"✓ Current Price: ₹{targets['current_price']:.2f}")
        
        print(f"\n💰 TRADING LEVELS:")
        print(f"  Entry Price: ₹{targets['entry']:.2f}")
        print(f"  Target Price: ₹{targets['target']:.2f}")
        print(f"  Stop Loss: ₹{targets['stop_loss']:.2f}")
        
        print(f"\n📊 SUPPORT & RESISTANCE:")
        print(f"  Support (1Y Low): ₹{targets['support']:.2f}")
        print(f"  Resistance (1Y High): ₹{targets['resistance']:.2f}")
        print(f"  Pivot Point: ₹{targets['pivot']:.2f}")
        
        print(f"\n📈 VOLATILITY ANALYSIS:")
        print(f"  Short-term (20d): {targets['volatilities']['vol_short']:.2f}%")
        print(f"  Medium-term (1Y): {targets['volatilities']['vol_medium']:.2f}%")
        print(f"  Long-term (7Y): {targets['volatilities']['vol_long']:.2f}%")
        print(f"  Average: {targets['volatilities']['vol_avg']:.2f}%")
        print(f"  ATR (14-day): ₹{targets['atr']:.2f}")
        
        if rr:
            print(f"\n💹 RISK/REWARD:")
            print(f"  Potential Profit: ₹{rr['profit']:.2f} ({rr['profit_percent']:.2f}%)")
            print(f"  Potential Loss: ₹{rr['loss']:.2f} ({rr['loss_percent']:.2f}%)")
            print(f"  Risk/Reward Ratio: 1:{rr['ratio']:.2f}")
            
            if rr['ratio'] >= 2:
                rating = "EXCELLENT ⭐⭐⭐"
            elif rr['ratio'] >= 1.5:
                rating = "GOOD ⭐⭐"
            elif rr['ratio'] >= 1:
                rating = "ACCEPTABLE ⭐"
            else:
                rating = "POOR ✗"
            
            print(f"  Rating: {rating}")
        
        if position:
            print(f"\n📍 POSITION SIZING (For ₹{position['risk_amount']:,} risk):")
            print(f"  Risk per Share: ₹{position['risk_per_share']:.2f}")
            print(f"  Position Size: {position['position_size']:,} shares")
            print(f"  Total Investment: ₹{position['total_investment']:,}")
        
        print(f"\n{'='*80}\n")
        
        return {
            'ticker': ticker,
            'signal': signal,
            'targets': targets,
            'risk_reward': rr,
            'position': position
        }


if __name__ == "__main__":
    calculator = PriceTargetCalculator()
    
    # Test with all stocks
    stocks = ['IOC', 'AAPL', 'ITC']
    signals = ['BUY', 'BUY', 'BUY']  # You can change these
    
    for ticker, signal in zip(stocks, signals):
        calculator.generate_report(ticker, signal)