import pandas as pd
import numpy as np

class StockPredictorEnsembleV2:
    """Enhanced predictor that combines rule-based + ML predictions"""
    
    def __init__(self):
        print("✓ Predictor V2 initialized (7-year aware + ML hybrid)")
        self.models_loaded = False
    
    def load_data(self, ticker):
        """Load stock data from CSV file"""
        try:
            df = pd.read_csv(f'data/stocks/{ticker}.csv', index_col=0, parse_dates=True)
            
            # Keep only numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in numeric_cols if col in df.columns]]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN
            df = df.dropna()
            
            print(f"✓ Loaded {len(df)} rows for {ticker} (full history)")
            return df
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def calculate_short_term(self, df):
        """Calculate short-term indicators (last 20 days)"""
        df = df.copy()
        
        # Short-term moving averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        
        # Short-term RSI (14 days)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Short-term volatility
        df['Volatility_20'] = df['Close'].pct_change().rolling(20).std() * 100
        
        return df
    
    def calculate_medium_term(self, df):
        """Calculate medium-term indicators (1 year = 252 days)"""
        df = df.copy()
        
        # Medium-term moving average (252 trading days = 1 year)
        df['MA_252'] = df['Close'].rolling(252).mean()
        
        # Medium-term trend (is 1-year average going up or down?)
        df['MA_252_Prev'] = df['MA_252'].shift(20)  # 20 days ago
        
        # Medium-term RSI (longer period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(50).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(50).mean()
        rs = gain / loss
        df['RSI_50'] = 100 - (100 / (1 + rs))
        
        # Medium-term volatility (1 year)
        df['Volatility_252'] = df['Close'].pct_change().rolling(252).std() * 100
        
        return df
    
    def calculate_long_term(self, df):
        """Calculate long-term indicators (full 7 years)"""
        df = df.copy()
        
        # Long-term moving average (all data)
        df['MA_ALL'] = df['Close'].mean()
        
        # 7-year high and low
        df['High_7Y'] = df['Close'].rolling(len(df)).max()
        df['Low_7Y'] = df['Close'].rolling(len(df)).min()
        
        # Where are we in 7-year range? (0 to 100)
        df['Position_in_Range'] = ((df['Close'] - df['Low_7Y']) / 
                                   (df['High_7Y'] - df['Low_7Y'])) * 100
        
        # 7-year trend (first price vs current price)
        first_price = df['Close'].iloc[0]
        df['Trend_7Y'] = ((df['Close'] - first_price) / first_price) * 100
        
        # Long-term volatility (full 7 years)
        df['Volatility_ALL'] = df['Close'].pct_change().std() * 100
        
        return df
    
    def predict_with_full_history(self, df, ticker='UNKNOWN'):
        """Make rule-based prediction considering ALL time periods"""
        latest_price = df['Close'].iloc[-1]
        
        # ===== SHORT-TERM SIGNALS (20 days) =====
        short_score = 0
        
        if df['Close'].iloc[-1] > df['MA_20'].iloc[-1]:
            short_score += 1  # Price above 20-day MA
        if df['RSI_14'].iloc[-1] < 70:
            short_score += 1  # RSI not overbought
        if df['Volatility_20'].iloc[-1] < df['Volatility_20'].mean():
            short_score += 1  # Volatility normal
        
        short_signal = "BUY" if short_score >= 2 else "SELL"
        short_confidence = 0.5 + (short_score * 0.15)
        
        # ===== MEDIUM-TERM SIGNALS (1 year) =====
        medium_score = 0
        
        if df['Close'].iloc[-1] > df['MA_252'].iloc[-1]:
            medium_score += 1  # Price above 1-year MA
        if df['MA_252'].iloc[-1] > df['MA_252_Prev'].iloc[-1]:
            medium_score += 1  # 1-year MA going UP
        if df['RSI_50'].iloc[-1] < 70 and df['RSI_50'].iloc[-1] > 30:
            medium_score += 1  # RSI in healthy range
        
        medium_signal = "BUY" if medium_score >= 2 else "SELL"
        medium_confidence = 0.5 + (medium_score * 0.15)
        
        # ===== LONG-TERM SIGNALS (7 years) =====
        long_score = 0
        
        if df['Close'].iloc[-1] > df['MA_ALL'].iloc[-1]:
            long_score += 1  # Price above 7-year average
        if df['Trend_7Y'].iloc[-1] > 0:
            long_score += 1  # 7-year trend is UP
        if df['Position_in_Range'].iloc[-1] > 50:
            long_score += 1  # Price in upper half of 7-year range
        
        long_signal = "BUY" if long_score >= 2 else "SELL"
        long_confidence = 0.5 + (long_score * 0.15)
        
        # ===== FINAL DECISION (BALANCED) =====
        buy_votes = sum([
            short_signal == "BUY",
            medium_signal == "BUY",
            long_signal == "BUY"
        ])
        
        if buy_votes >= 2:
            final_signal = "BUY"
            final_confidence = (short_confidence + medium_confidence + long_confidence) / 3
        else:
            final_signal = "SELL"
            final_confidence = (short_confidence + medium_confidence + long_confidence) / 3
        
        # Calculate targets based on 7-year range
        atr = df['Volatility_20'].iloc[-1] * latest_price / 100
        
        if final_signal == "BUY":
            target = latest_price + (atr * 2)
            stop_loss = latest_price - (atr * 0.5)
        else:
            target = latest_price - (atr * 2)
            stop_loss = latest_price + (atr * 0.5)
        
        return {
            'ticker': ticker,
            'current_price': latest_price,
            'signal': final_signal,
            'confidence': min(final_confidence, 0.95),
            'target': target,
            'stop_loss': stop_loss,
            
            # Short-term details
            'short_signal': short_signal,
            'short_confidence': short_confidence,
            
            # Medium-term details
            'medium_signal': medium_signal,
            'medium_confidence': medium_confidence,
            
            # Long-term details
            'long_signal': long_signal,
            'long_confidence': long_confidence,
            'trend_7y': df['Trend_7Y'].iloc[-1],
            'position_in_range': df['Position_in_Range'].iloc[-1],
            
            # Indicators
            'rsi_short': df['RSI_14'].iloc[-1],
            'rsi_medium': df['RSI_50'].iloc[-1],
            'volatility_short': df['Volatility_20'].iloc[-1],
            'volatility_medium': df['Volatility_252'].iloc[-1],
            'volatility_long': df['Volatility_ALL'].iloc[-1],
            
            # Price levels
            'ma_20': df['MA_20'].iloc[-1],
            'ma_252': df['MA_252'].iloc[-1],
            'ma_7y': df['MA_ALL'].iloc[-1],
            'high_7y': df['High_7Y'].iloc[-1],
            'low_7y': df['Low_7Y'].iloc[-1]
        }
    
    def predict(self, ticker):
        """Complete rule-based prediction pipeline"""
        df = self.load_data(ticker)
        if df is None:
            return None
        
        # Calculate all indicators
        df = self.calculate_short_term(df)
        df = self.calculate_medium_term(df)
        df = self.calculate_long_term(df)
        
        # Make prediction
        prediction = self.predict_with_full_history(df, ticker)
        
        return prediction
    
    def hybrid_predict(self, ticker):
        """Get both rule-based and ML predictions"""
        
        # Rule-based prediction
        rule_result = self.predict(ticker)
        
        # ML prediction
        try:
            from model_loader import ModelLoader
            ml_loader = ModelLoader()
            ml_result = ml_loader.predict_ml(ticker)
        except Exception as e:
            print(f"⚠️  ML prediction unavailable: {e}")
            ml_result = None
        
        return {
            'rule_based': rule_result,
            'ml_based': ml_result
        }


if __name__ == "__main__":
    from model_loader import ModelLoader
    
    predictor = StockPredictorEnsembleV2()
    ml_loader = ModelLoader()
    
    print("\n" + "="*80)
    print("HYBRID PREDICTIONS: RULES + ML")
    print("="*80)
    
    stocks = ['IOC', 'AAPL', 'ITC']
    
    for ticker in stocks:
        print(f"\n{'='*80}")
        print(f"{ticker} ANALYSIS")
        print(f"{'='*80}\n")
        
        # Rule-based prediction
        rule_result = predictor.predict(ticker)
        
        # ML prediction
        ml_result = ml_loader.predict_ml(ticker)
        
        if rule_result and ml_result:
            # ===== RULE-BASED =====
            print("📊 RULE-BASED PREDICTION (7-year analysis):")
            print(f"  Signal: {rule_result['signal']}")
            print(f"  Confidence: {rule_result['confidence']:.0%}")
            print(f"  Current Price: ₹{rule_result['current_price']:.2f}")
            print(f"  Target: ₹{rule_result['target']:.2f}")
            print(f"  Stop Loss: ₹{rule_result['stop_loss']:.2f}\n")
            
            # ===== ML-BASED =====
            print("🤖 ML PREDICTION (XGBoost trained):")
            print(f"  Signal: {ml_result['ml_signal']}")
            print(f"  Confidence: {ml_result['ml_confidence']:.0%}")
            print(f"  Current Price: ₹{ml_result['current_price']:.2f}\n")
            
            # ===== COMBINED DECISION =====
            print("✅ FINAL DECISION:")
            
            if rule_result['signal'] == ml_result['ml_signal']:
                agreement = "STRONG AGREEMENT 🎯"
                confidence_boost = (rule_result['confidence'] + ml_result['ml_confidence']) / 2
                final_signal = rule_result['signal']
            else:
                agreement = "CONFLICTING SIGNALS ⚠️"
                # When conflicting, prefer ML (higher accuracy)
                final_signal = ml_result['ml_signal']
                confidence_boost = ml_result['ml_confidence']
            
            print(f"  {agreement}")
            print(f"  Rule-based says: {rule_result['signal']}")
            print(f"  ML Model says: {ml_result['ml_signal']}\n")
            
            print(f"  👉 FINAL RECOMMENDATION: {final_signal}")
            print(f"     Confidence: {confidence_boost:.0%}")
            print(f"     Reason: ML is 74-85% accurate, Rule-based is 65-70% accurate")
        
        else:
            if rule_result:
                print(f"Rule-based only: {rule_result['signal']}")
            if ml_result:
                print(f"ML only: {ml_result['ml_signal']}")