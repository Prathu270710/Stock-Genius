from core.trading_summary import TradingSummary
from core.model_loader import ModelLoader
from core.predictor import StockPredictorEnsembleV2

def test_all_stocks():
    """Test all stocks with complete system"""
    
    print("\n" + "="*80)
    print("COMPLETE SYSTEM TEST")
    print("="*80 + "\n")
    
    summary = TradingSummary()
    ml_loader = ModelLoader()
    predictor = StockPredictorEnsembleV2()
    
    # Test with trained stocks
    stocks = ['IOC', 'AAPL', 'ITC']
    
    for ticker in stocks:
        print(f"\n{'='*80}")
        print(f"TESTING {ticker}")
        print(f"{'='*80}\n")
        
        # Check if model available
        if ml_loader.is_model_available(ticker):
            print(f"✅ Model available for {ticker}")
        else:
            print(f"⚠️  Model not trained for {ticker}")
        
        # Generate report
        report = summary.generate_full_report(ticker, show_expert=False)
        
        if report:
            print(f"\n✅ Report generated successfully!")
        else:
            print(f"\n❌ Failed to generate report")
    
    print(f"\n{'='*80}")
    print("✓ TEST COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_all_stocks()