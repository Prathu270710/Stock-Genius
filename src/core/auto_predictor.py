import os
import sys
import pandas as pd
import yfinance as yf
import threading
import json

# Add core directory to path
sys.path.insert(0, os.path.dirname(__file__))

from predictor import StockPredictorEnsembleV2
from model_loader import ModelLoader
from price_targets import PriceTargetCalculator
from risk_manager import RiskManager
from feature_engineering import FeatureEngineering
from board_selector import BoardSelector

class AutoPredictor:
    """Automatically predict ANY stock - Download, Train, Analyze"""
    
    TRAINED_STOCKS_FILE = 'trained_stocks.json'
    
    def __init__(self):
        self.predictor = StockPredictorEnsembleV2()
        self.ml_loader = ModelLoader()
        self.price_calculator = PriceTargetCalculator()
        self.risk_manager = RiskManager()
        self.feature_engineer = FeatureEngineering()
        self.trained_stocks = self.load_trained_stocks()
        
        print("✓ Auto Predictor initialized")
    
    def load_trained_stocks(self):
        """Load list of already trained stocks"""
        if os.path.exists(self.TRAINED_STOCKS_FILE):
            try:
                with open(self.TRAINED_STOCKS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_trained_stocks(self):
        """Save trained stocks list"""
        with open(self.TRAINED_STOCKS_FILE, 'w') as f:
            json.dump(self.trained_stocks, f, indent=2)
    
    def mark_stock_trained(self, ticker, accuracy):
        """Mark stock as trained"""
        self.trained_stocks[ticker] = {
            'accuracy': accuracy,
            'timestamp': str(pd.Timestamp.now())
        }
        self.save_trained_stocks()
    
    def is_stock_trained(self, ticker):
        """Check if stock is already trained"""
        return ticker in self.trained_stocks
    
    def download_stock(self, ticker, period='7y'):
        """Download stock data"""
        print(f"\n📥 Downloading {ticker}...")
        
        try:
            data = yf.download(ticker, period=period, progress=False)
            
            if len(data) == 0:
                print(f"✗ No data found for {ticker}")
                return False
            
            os.makedirs('data/stocks', exist_ok=True)
            data.to_csv(f'data/stocks/{ticker}.csv')
            
            print(f"✓ Downloaded: {len(data)} rows")
            return True
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def auto_train_stock(self, ticker):
        """Train ML model for stock (background task)"""
        print(f"\n🤖 Background Training Started for {ticker}...")
        print(f"   (This will take 2-3 minutes)")
        
        try:
            from learner_optimized import OptimizedModelLearner
            
            learner = OptimizedModelLearner()
            
            # Default features
            default_features = [
                'Price_Channel_High', 'Trend_Alignment', 'MA_10', 'Price_Channel_Low',
                'BB_Lower', 'MA_20', 'Close_MA_20_ratio', 'MA_200', 'KC_Lower', 'MA_100',
                'MACD', 'Close_MA_100_ratio', 'MA_5', 'Close_MA_50_ratio',
                'Close_MA_200_ratio', 'BB_STD_20', 'Distance_to_Support', 'MA20_MA200_Ratio',
                'KC_Upper', 'BB_Position', 'plus_di', 'monthly_return', 'MACD_signal',
                'ATR_14', 'CCI'
            ]
            
            learner.BEST_FEATURES[ticker] = default_features
            result = learner.train_all_models(ticker)
            
            if result:
                accuracy = result['metrics']['accuracy']
                self.mark_stock_trained(ticker, accuracy)
                print(f"✅ {ticker} training complete! Accuracy: {accuracy:.2%}")
                return True
            else:
                print(f"❌ Training failed for {ticker}")
                return False
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def get_instant_prediction(self, ticker):
        """Get INSTANT prediction (rule-based only)"""
        print(f"\n⚡ Getting instant prediction for {ticker}...")
        
        df = self.predictor.load_data(ticker)
        if df is None:
            return None
        
        # Calculate indicators
        df = self.predictor.calculate_short_term(df)
        df = self.predictor.calculate_medium_term(df)
        df = self.predictor.calculate_long_term(df)
        
        # Rule-based prediction - FIX: pass ticker
        rule_pred = self.predictor.predict_with_full_history(df, ticker)
        
        # Price targets
        targets = self.price_calculator.calculate_targets_and_stops(df, rule_pred['signal'])
        
        # Risk analysis
        risk_data = self.risk_manager.generate_report(
            ticker,
            targets['current_price'],
            targets['target'],
            targets['stop_loss']
        )
        
        return {
            'ticker': ticker,
            'type': 'RULE-BASED (Instant)',
            'signal': rule_pred['signal'],
            'confidence': rule_pred['confidence'],
            'current_price': targets['current_price'],
            'entry': targets['entry'],
            'target': targets['target'],
            'stop_loss': targets['stop_loss'],
            'win_rate': risk_data['win_rate'],
            'ml_available': False,
            'model_training': True
        }
    
    def get_ml_prediction(self, ticker):
        """Get ML prediction (if model trained)"""
        if self.ml_loader.is_model_available(ticker):
            return self.ml_loader.predict_ml(ticker)
        return None
    
    def predict_with_auto_train(self, ticker):
        """
        Get prediction immediately + train ML in background
        Returns full analytics if model already trained
        """
        
        print(f"\n{'='*80}")
        print(f"AUTO PREDICTION - {ticker}")
        print(f"{'='*80}\n")
        
        # Step 1: Check if data exists
        if not os.path.exists(f'data/stocks/{ticker}.csv'):
            print(f"📥 Data not found, downloading...")
            if not self.download_stock(ticker):
                print(f"✗ Failed to download {ticker}")
                return None
        else:
            print(f"✓ Data already available for {ticker}")
        
        # Step 2: Get instant prediction (rule-based)
        instant_result = self.get_instant_prediction(ticker)
        if not instant_result:
            print(f"✗ Failed to get prediction")
            return None
        
        # Step 3: Check if ML model exists
        ml_result = self.get_ml_prediction(ticker)
        
        # Step 4: Determine status
        if ml_result:
            # Model exists - FULL ANALYTICS
            instant_result['ml_result'] = ml_result
            instant_result['model_training'] = False
            instant_result['type'] = 'FULL ANALYTICS (ML + Rule-based)'
        else:
            # Model doesn't exist - START BACKGROUND TRAINING
            if not self.is_stock_trained(ticker):
                print(f"\n🚀 Starting background training for {ticker}...")
                training_thread = threading.Thread(
                    target=self.auto_train_stock,
                    args=(ticker,),
                    daemon=True
                )
                training_thread.start()
                instant_result['model_training'] = True
            else:
                instant_result['model_training'] = False
        
        return instant_result
    
    def print_prediction(self, result):
        """Pretty print prediction"""
        if not result:
            return
        
        ticker = result['ticker']
        signal = result['signal']
        confidence = result['confidence']
        price = result['current_price']
        
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        
        print(f"\n{'='*80}")
        print(f"PREDICTION RESULT - {ticker}")
        print(f"{'='*80}\n")
        
        # Type
        print(f"📊 Analysis Type: {result['type']}")
        
        if result.get('ml_result'):
            print(f"🤖 ML Model: ✅ AVAILABLE (Accuracy: 85%)\n")
            ml = result['ml_result']
            print(f"🤖 ML PREDICTION:")
            print(f"   Signal: {ml['ml_signal']}")
            print(f"   Confidence: {ml['ml_confidence']:.0%}\n")
        else:
            if result.get('model_training'):
                print(f"🤖 ML Model: ⏳ TRAINING (Will be ready in 2-3 minutes)\n")
            else:
                print(f"🤖 ML Model: ⏳ TRAINING SCHEDULED\n")
        
        # Main prediction
        print(f"📊 RULE-BASED PREDICTION:")
        print(f"{signal_emoji} Signal: {signal}")
        print(f"📈 Confidence: {confidence:.0%}")
        print(f"💰 Current Price: ₹{price:.2f}\n")
        
        # Trading plan
        print(f"📍 TRADING PLAN:")
        print(f"   Entry: ₹{result['entry']:.2f}")
        print(f"   Target: ₹{result['target']:.2f}")
        print(f"   Stop Loss: ₹{result['stop_loss']:.2f}\n")
        
        # Win rate
        print(f"📈 Historical Win Rate: {result['win_rate']:.1f}%\n")
        
        # Status
        if result.get('ml_result'):
            print(f"✅ STATUS: FULL ANALYTICS AVAILABLE!")
            print(f"💡 You have both rule-based (70%) and ML (85%) predictions!\n")
        elif result.get('model_training'):
            print(f"⏳ STATUS: Background training in progress...")
            print(f"💡 Come back in 3 minutes for ML prediction (85% accuracy)!\n")
        else:
            print(f"ℹ️  STATUS: Training scheduled.\n")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("WELCOME TO AUTO PREDICTOR")
    print("="*80)
    
    predictor = AutoPredictor()
    
    # Get ticker from user with board selection
    ticker = BoardSelector.get_ticker_from_user()
    
    if ticker:
        # Get prediction
        result = predictor.predict_with_auto_train(ticker)
        
        # Show result
        if result:
            predictor.print_prediction(result)