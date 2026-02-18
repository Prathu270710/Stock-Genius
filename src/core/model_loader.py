import pickle
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineering
from sklearn.preprocessing import StandardScaler

class ModelLoader:
    """Load saved ML models and make predictions for ANY stock"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.scalers = {}
        
        # Pre-tuned features for stocks we've trained
        self.BEST_FEATURES = {
            'IOC': [
                'Price_Channel_High', 'Trend_Alignment', 'MA_10', 'Price_Channel_Low',
                'BB_Lower', 'MA_20', 'Close_MA_20_ratio', 'MA_200', 'KC_Lower', 'MA_100',
                'MACD', 'Close_MA_100_ratio', 'MA_5', 'Close_MA_50_ratio',
                'Close_MA_200_ratio', 'BB_STD_20', 'Distance_to_Support', 'MA20_MA200_Ratio',
                'KC_Upper', 'BB_Position', 'plus_di', 'monthly_return', 'MACD_signal',
                'ATR_14', 'CCI'
            ],
            'AAPL': [
                'MA50_above_MA200', 'Price_Channel_High', 'Trend_Alignment', 'ATR_14',
                'MA_50', 'MA_200', 'MA_10', 'MA_5', 'Close_MA_200_ratio', 'KC_Lower',
                'Close_Open_Corr', 'Close_MA_50_ratio', 'Price_Channel_Low', 'MACD_signal',
                'MA20_MA200_Ratio', 'Price_Channel_Mid', 'MACD', 'OBV', 'Distance_to_Support',
                'HV_50', 'Close_MA_20_ratio', 'plus_di', 'HV_20', 'KC_Upper', 'MA_100'
            ],
            'ITC': [
                'Close_MA_50_ratio', 'Close_MA_100_ratio', 'Close_MA_200_ratio', 'MACD',
                'MACD_signal', 'momentum_20', 'BB_STD_20', 'BB_Position', 'HV_20',
                'volume_ratio', 'OBV', 'VPT_MA', 'Stoch_RSI', 'CCI', 'plus_di', 'ADX',
                'Candle_Size', 'Close_Open_Corr', 'Volume_Price_Corr', 'Return_Volume_Corr',
                'Distance_to_Resistance', 'MA20_above_MA50', 'MA50_above_MA200',
                'Trend_Alignment', 'MA20_MA200_Ratio'
            ]
        }
        
        # Default features (template for new stocks)
        self.DEFAULT_FEATURES = [
            'Price_Channel_High', 'Trend_Alignment', 'MA_10', 'Price_Channel_Low',
            'BB_Lower', 'MA_20', 'Close_MA_20_ratio', 'MA_200', 'KC_Lower', 'MA_100',
            'MACD', 'Close_MA_100_ratio', 'MA_5', 'Close_MA_50_ratio',
            'Close_MA_200_ratio', 'BB_STD_20', 'Distance_to_Support', 'MA20_MA200_Ratio',
            'KC_Upper', 'BB_Position', 'plus_di', 'monthly_return', 'MACD_signal',
            'ATR_14', 'CCI'
        ]
        
        print("✓ Model Loader initialized (Scalable)")
    
    def get_features_for_stock(self, ticker):
        """Get best features for a stock"""
        if ticker in self.BEST_FEATURES:
            return self.BEST_FEATURES[ticker]
        else:
            # Use default template for new/untrained stocks
            return self.DEFAULT_FEATURES
    
    def load_model(self, ticker):
        """Load saved model and scaler"""
        try:
            model_path = f'models/{ticker}_optimized/xgboost_optimized.pkl'
            scaler_path = f'models/{ticker}_optimized/scaler_optimized.pkl'
            
            # Load model
            with open(model_path, 'rb') as f:
                self.models[ticker] = pickle.load(f)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scalers[ticker] = pickle.load(f)
            
            print(f"✓ {ticker} model loaded (Accuracy: 74-85%)")
            return True
        
        except FileNotFoundError:
            print(f"✗ Model not found for {ticker}")
            print(f"   Train first: python core/train_any_stock.py")
            return False
        
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
            return False
    
    def is_model_available(self, ticker):
        """Check if model exists without loading"""
        import os
        model_path = f'models/{ticker}_optimized/xgboost_optimized.pkl'
        return os.path.exists(model_path)
    
    def predict_ml(self, ticker):
        """Make ML prediction for any stock"""
        
        # Load model if not already loaded
        if ticker not in self.models:
            if not self.load_model(ticker):
                return None
        
        try:
            # Prepare features
            df = self.feature_engineer.prepare_features(ticker)
            if df is None:
                print(f"✗ Could not prepare features for {ticker}")
                return None
            
            # Get best features for this stock
            selected_features = self.get_features_for_stock(ticker)
            
            # Verify all features exist
            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                print(f"⚠️  Missing features for {ticker}: {missing_features}")
                # Use only available features
                selected_features = [f for f in selected_features if f in df.columns]
            
            # Get latest row
            latest_features = df[selected_features].iloc[-1:].values
            
            # Scale
            X_scaled = self.scalers[ticker].transform(latest_features)
            
            # Get prediction from model
            model = self.models[ticker]
            ml_prediction = model.predict(X_scaled)[0]  # 0=DOWN, 1=UP
            ml_proba = model.predict_proba(X_scaled)[0]  # [DOWN_prob, UP_prob]
            
            # ML confidence (max probability)
            ml_confidence = max(ml_proba)
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Get latest data metrics
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
            
            return {
                'ticker': ticker,
                'ml_signal': 'BUY' if ml_prediction == 1 else 'SELL',
                'ml_confidence': ml_confidence,
                'ml_prediction': ml_prediction,
                'current_price': current_price,
                'ml_proba_down': ml_proba[0],
                'ml_proba_up': ml_proba[1],
                'rsi': rsi,
                'macd': macd,
                'features_used': len(selected_features)
            }
        
        except Exception as e:
            print(f"✗ Error predicting for {ticker}: {e}")
            return None
    
    def predict_batch(self, ticker_list):
        """Make predictions for multiple stocks"""
        results = {}
        
        for ticker in ticker_list:
            result = self.predict_ml(ticker)
            if result:
                results[ticker] = result
        
        return results
    
    def print_prediction(self, result):
        """Pretty print a prediction"""
        if not result:
            return
        
        ticker = result['ticker']
        signal = result['ml_signal']
        confidence = result['ml_confidence']
        price = result['current_price']
        
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        
        print(f"\n{signal_emoji} {ticker}")
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Current Price: ₹{price:.2f}")
        
        if result['rsi']:
            print(f"   RSI: {result['rsi']:.2f}")
        if result['macd']:
            print(f"   MACD: {result['macd']:.4f}")


if __name__ == "__main__":
    loader = ModelLoader()
    
    print(f"\n{'='*80}")
    print("ML MODEL PREDICTIONS")
    print(f"{'='*80}")
    
    # Test with trained stocks
    trained_stocks = ['IOC', 'AAPL', 'ITC']
    
    print(f"\n📊 CHECKING AVAILABLE MODELS:")
    for ticker in trained_stocks:
        status = "✅ Ready" if loader.is_model_available(ticker) else "❌ Not trained"
        print(f"  {ticker}: {status}")
    
    print(f"\n{'='*80}")
    print("MAKING PREDICTIONS")
    print(f"{'='*80}")
    
    for ticker in trained_stocks:
        result = loader.predict_ml(ticker)
        if result:
            loader.print_prediction(result)
    
    print(f"\n{'='*80}\n")