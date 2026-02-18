import os
import sys
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineering
from learner_optimized import OptimizedModelLearner

class ScalableTrainer:
    """Train models for ANY stock - Works for multiple stocks"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.learner = OptimizedModelLearner()
        
        print("✓ Scalable Trainer initialized")
    
    def download_stock_data(self, ticker, period='7y'):
        """Download data for any stock"""
        import yfinance as yf
        
        print(f"\n📥 Downloading {ticker} ({period})...")
        
        try:
            data = yf.download(ticker, period=period, progress=False)
            
            if len(data) == 0:
                print(f"✗ No data found for {ticker}")
                return False
            
            os.makedirs('data/stocks', exist_ok=True)
            data.to_csv(f'data/stocks/{ticker}.csv')
            
            print(f"✓ {ticker} downloaded: {len(data)} rows")
            return True
        
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {e}")
            return False
    
    def train_stock_model(self, ticker):
        """Train model for any stock"""
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL FOR {ticker}")
        print(f"{'='*80}\n")
        
        # Use default features
        default_features = [
            'Price_Channel_High', 'Trend_Alignment', 'MA_10', 'Price_Channel_Low',
            'BB_Lower', 'MA_20', 'Close_MA_20_ratio', 'MA_200', 'KC_Lower', 'MA_100',
            'MACD', 'Close_MA_100_ratio', 'MA_5', 'Close_MA_50_ratio',
            'Close_MA_200_ratio', 'BB_STD_20', 'Distance_to_Support', 'MA20_MA200_Ratio',
            'KC_Upper', 'BB_Position', 'plus_di', 'monthly_return', 'MACD_signal',
            'ATR_14', 'CCI'
        ]
        
        # Add to learner
        self.learner.BEST_FEATURES[ticker] = default_features
        
        # Train
        result = self.learner.train_all_models(ticker)
        
        return result
    
    def batch_train_stocks(self, stock_list, download=True):
        """Train multiple stocks at once"""
        print(f"\n{'='*80}")
        print(f"BATCH TRAINING: {len(stock_list)} STOCKS")
        print(f"{'='*80}\n")
        
        results = {}
        failed = []
        
        for idx, ticker in enumerate(stock_list, 1):
            print(f"\n[{idx}/{len(stock_list)}] Processing {ticker}...")
            
            # Download
            if download:
                if not self.download_stock_data(ticker):
                    print(f"⏭️  Skipping {ticker} (download failed)")
                    failed.append(ticker)
                    continue
            
            # Train
            result = self.train_stock_model(ticker)
            
            if result:
                results[ticker] = result
                print(f"✅ {ticker} training complete!")
            else:
                print(f"❌ {ticker} training failed")
                failed.append(ticker)
        
        # Summary
        self.print_summary(results, failed)
        
        return results
    
    def print_summary(self, results, failed):
        """Print training summary"""
        print(f"\n\n{'='*80}")
        print("TRAINING SUMMARY - ALL STOCKS")
        print(f"{'='*80}\n")
        
        if results:
            print(f"{'Ticker':<12} | {'Accuracy':<12} | {'Precision':<12} | "
                  f"{'Recall':<10} | {'AUC-ROC':<10} | {'Status':<10}")
            print("-" * 100)
            
            total_accuracy = 0
            
            for ticker in sorted(results.keys()):
                result = results[ticker]
                m = result['metrics']
                accuracy = m['accuracy'] * 100
                precision = m['precision'] * 100
                recall = m['recall'] * 100
                auc = m['auc'] * 100
                
                status = "✅ Good" if accuracy >= 70 else "⚠️  Fair"
                
                print(f"{ticker:<12} | {accuracy:>10.2f}% | {precision:>10.2f}% | "
                      f"{recall:>8.2f}% | {auc:>8.2f}% | {status:<10}")
                
                total_accuracy += accuracy
            
            avg_accuracy = total_accuracy / len(results)
            print("-" * 100)
            print(f"{'AVERAGE':<12} | {avg_accuracy:>10.2f}%")
        
        if failed:
            print(f"\n⚠️  FAILED STOCKS ({len(failed)}):")
            for ticker in failed:
                print(f"  - {ticker}")
        
        print(f"\n{'='*80}")
        print(f"✓ Successfully trained {len(results)} out of {len(results) + len(failed)} stocks!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    trainer = ScalableTrainer()
    
    # ========== CHOOSE YOUR STOCKS ==========
    
    # Option 1: Just the 3 core stocks (already trained)
    stocks_to_train = ['IOC.NS', 'AAPL', 'ITC', 'GOOGL', 'TSLA']

    
    # Option 2: Add US Tech stocks
    # stocks_to_train = ['IOC', 'AAPL', 'ITC', 'GOOGL', 'MSFT', 'TSLA']
    
    # Option 3: Add Indian stocks
    # stocks_to_train = ['IOC', 'ITC', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'WIPRO.NS']
    
    # Option 4: Mixed - US + India
    # stocks_to_train = ['IOC', 'ITC', 'AAPL', 'GOOGL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']
    
    # Option 5: Extended portfolio (20+ stocks)
    # stocks_to_train = [
    #     # Indian
    #     'IOC', 'ITC', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'BAJAJ-AUTO.NS',
    #     # US Tech
    #     'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN',
    #     # US Finance
    #     'JPM', 'BAC', 'WFC',
    #     # US Consumer
    #     'PG', 'KO', 'MCD'
    # ]
    
    print(f"\n🚀 Training {len(stocks_to_train)} stocks...")
    print(f"Stocks: {', '.join(stocks_to_train)}\n")
    
    # Train all!
    results = trainer.batch_train_stocks(stocks_to_train, download=True)