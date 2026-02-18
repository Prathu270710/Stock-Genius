import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from core.auto_predictor import AutoPredictor
from core.board_selector import BoardSelector

def main():
    """Main application entry point"""
    
    try:
        while True:
            print("\n" + "="*80)
            print("STOCK PREDICTION & ANALYTICS SYSTEM")
            print("="*80 + "\n")
            
            print("OPTIONS:")
            print("1. Predict a Stock")
            print("2. View Trained Stocks")
            print("3. Exit\n")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                try:
                    # Get ticker from user
                    ticker = BoardSelector.get_ticker_from_user()
                    
                    if ticker:
                        # Get prediction
                        predictor = AutoPredictor()
                        result = predictor.predict_with_auto_train(ticker)
                        
                        # Show result
                        if result:
                            predictor.print_prediction(result)
                            
                            # Expand to full view
                            try:
                                view_expert = input("\nView expert details? (y/n): ").strip().lower()
                                if view_expert == 'y':
                                    from core.trading_summary import TradingSummary
                                    summary = TradingSummary()
                                    data = summary.generate_simple_view(ticker)
                                    if data:
                                        summary.generate_expert_view(data)
                            except KeyboardInterrupt:
                                print("\n\n⏹️  Skipping expert details...\n")
                                continue
                
                except KeyboardInterrupt:
                    print("\n\n⏹️  Stock search cancelled.\n")
                    continue
            
            elif choice == '2':
                # Show trained stocks
                predictor = AutoPredictor()
                trained = predictor.trained_stocks
                
                if trained:
                    print("\n" + "="*80)
                    print("TRAINED STOCKS (Ready for full analytics)")
                    print("="*80 + "\n")
                    
                    for ticker, info in trained.items():
                        print(f"✅ {ticker:<15} Accuracy: {info['accuracy']:.2%}")
                    
                    print()
                else:
                    print("\n⏳ No stocks trained yet. Start by predicting a stock!\n")
            
            elif choice == '3':
                print("\n" + "="*80)
                print("THANK YOU FOR USING STOCK PREDICTION SYSTEM")
                print("="*80)
                print("Goodbye! 👋\n")
                break
            
            else:
                print("Invalid choice. Try again.")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⏹️  PROGRAM STOPPED BY USER")
        print("="*80)
        print("\nThank you for using Stock Prediction System!")
        print("Goodbye! 👋\n")

if __name__ == "__main__":
    main()