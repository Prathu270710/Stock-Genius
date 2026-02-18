import pandas as pd
import numpy as np
from predictor import StockPredictorEnsembleV2
from price_targets import PriceTargetCalculator
from risk_manager import RiskManager
from model_loader import ModelLoader

class TradingSummary:
    """Simple, customer-friendly trading summary with ML + Rule-based predictions"""
    
    def __init__(self):
        self.predictor = StockPredictorEnsembleV2()
        self.price_calculator = PriceTargetCalculator()
        self.risk_manager = RiskManager()
        self.ml_loader = ModelLoader()
    
    def generate_simple_view(self, ticker):
        """Generate SIMPLE view for regular customers"""
        print(f"\n{'='*80}")
        print(f"TRADING RECOMMENDATION - {ticker}")
        print(f"{'='*80}\n")
        
        # Get rule-based prediction
        rule_prediction = self.predictor.predict(ticker)
        if not rule_prediction:
            print("✗ Failed to analyze")
            return None
        
        # Get ML prediction
        ml_prediction = self.ml_loader.predict_ml(ticker)
        
        # Get price targets
        df = self.price_calculator.load_data(ticker)
        targets = self.price_calculator.calculate_targets_and_stops(df, rule_prediction['signal'])
        
        # Get risk metrics
        risk_data = self.risk_manager.generate_report(
            ticker, 
            targets['current_price'],
            targets['target'],
            targets['stop_loss']
        )
        
        # ===== HYBRID SIGNAL DECISION =====
        # Use ML if available, otherwise use rule-based
        if ml_prediction:
            final_signal = ml_prediction['ml_signal']
            final_confidence = ml_prediction['ml_confidence']
            ml_confidence = ml_prediction['ml_confidence']
        else:
            final_signal = rule_prediction['signal']
            final_confidence = rule_prediction['confidence']
            ml_confidence = None
        
        # ===== SIMPLE VIEW (MAIN) =====
        print("🎯 WHAT YOU NEED TO KNOW:\n")
        
        # Signal
        if final_signal == "BUY":
            print(f"📈 RECOMMENDATION: BUY ✅")
            signal_emoji = "🟢"
        else:
            print(f"📉 RECOMMENDATION: SELL ⚠️")
            signal_emoji = "🔴"
        
        print(f"   Confidence: {final_confidence:.0%}")
        print(f"   Status: {signal_emoji} {final_signal}\n")
        
        # Current Price
        current = targets['current_price']
        print(f"💰 CURRENT PRICE: ₹{current:.2f}\n")
        
        # Entry, Target, Stop (THE MOST IMPORTANT)
        entry = targets['entry']
        target = targets['target']
        stop = targets['stop_loss']
        
        print("📍 YOUR TRADING PLAN:")
        print(f"   Entry Price:    ₹{entry:.2f}  (BUY HERE)")
        print(f"   Target Price:   ₹{target:.2f}  (SELL FOR PROFIT)")
        print(f"   Stop Loss:      ₹{stop:.2f}  (PROTECT IF WRONG)\n")
        
        # Simple Profit/Loss
        profit = target - entry
        loss = entry - stop
        
        print("💵 PROFIT & LOSS:")
        print(f"   Potential Profit: ₹{profit:.2f} per share")
        print(f"   Max Loss: ₹{loss:.2f} per share\n")
        
        # Risk/Reward (SIMPLIFIED)
        rr_ratio = profit / loss if loss > 0 else 0
        
        print("⚡ IS THIS A GOOD TRADE?")
        if rr_ratio >= 2.5:
            rating = "EXCELLENT ⭐⭐⭐ (Profit is 2.5x the risk)"
        elif rr_ratio >= 1.5:
            rating = "GOOD ⭐⭐ (Profit is 1.5x the risk)"
        elif rr_ratio >= 1:
            rating = "OKAY ⭐ (Profit = Risk)"
        else:
            rating = "POOR ✗ (Risk > Profit - Avoid)"
        
        print(f"   {rating}\n")
        
        # Position Size
        position = self.price_calculator.calculate_position_size(entry, stop, 10000)
        
        print("📊 HOW MUCH TO INVEST:")
        print(f"   For ₹10,000 risk: Buy {position['position_size']:,} shares")
        print(f"   Total investment: ₹{position['total_investment']:,.0f}\n")
        
        # Win Probability (SIMPLIFIED for users)
        win_rate = risk_data['win_rate']
        
        print("📈 SUCCESS RATE:")
        print(f"   This strategy worked {win_rate:.1f}% of the time")
        print(f"   in the last 7 years\n")
        
        # ===== ML PREDICTION (if available) =====
        if ml_prediction:
            print("🤖 AI MODEL ANALYSIS:")
            print(f"   ML Signal: {ml_prediction['ml_signal']}")
            print(f"   ML Confidence: {ml_prediction['ml_confidence']:.0%}")
            print(f"   Accuracy: 74-85% (trained on 7 years data)\n")
        
        # Final Decision
        print("✅ FINAL DECISION:")
        
        if final_signal == "BUY" and rr_ratio >= 1.5 and win_rate > 50:
            print(f"   ✅ STRONG BUY - Take this trade!")
            print(f"      Why: Good reward, acceptable risk, {win_rate:.0f}% historical success\n")
        elif final_signal == "BUY" and rr_ratio >= 1:
            print(f"   🟡 CAUTIOUS BUY - Consider this trade")
            print(f"      Why: Acceptable reward/risk\n")
        elif final_signal == "SELL":
            print(f"   ⚠️  SELL - Consider exiting\n")
        else:
            print(f"   ❌ SKIP - Unfavorable trade\n")
        
        # Store data for expert view
        return {
            'ticker': ticker,
            'signal': final_signal,
            'confidence': final_confidence,
            'current': current,
            'entry': entry,
            'target': target,
            'stop': stop,
            'profit': profit,
            'loss': loss,
            'rr_ratio': rr_ratio,
            'win_rate': win_rate,
            'position': position,
            'rule_prediction': rule_prediction,
            'ml_prediction': ml_prediction,
            'targets': targets,
            'risk_data': risk_data
        }
    
    def generate_expert_view(self, data):
        """Generate EXPERT view (collapsible advanced section)"""
        if not data:
            return
        
        print("="*80)
        print("📚 EXPERT ANALYSIS (Advanced Metrics)")
        print("="*80 + "\n")
        
        rule_pred = data['rule_prediction']
        ml_pred = data['ml_prediction']
        targets = data['targets']
        risk = data['risk_data']
        
        # ===== SIGNAL COMPARISON =====
        print("📊 SIGNAL COMPARISON:")
        print(f"  Rule-based (7-year): {rule_pred['signal']} ({rule_pred['confidence']:.0%})")
        if ml_pred:
            print(f"  ML Model (XGBoost): {ml_pred['ml_signal']} ({ml_pred['ml_confidence']:.0%})")
            
            if rule_pred['signal'] == ml_pred['ml_signal']:
                print(f"  Status: ✅ AGREEMENT\n")
            else:
                print(f"  Status: ⚠️  CONFLICTING\n")
        else:
            print(f"  ML Model: Not available\n")
        
        # ===== TIMEFRAME SIGNALS =====
        print("⏱️  TIMEFRAME SIGNALS:")
        print(f"  Short-term (20 days):   {rule_pred['short_signal']}  ({rule_pred['short_confidence']:.0%})")
        print(f"  Medium-term (1 year):   {rule_pred['medium_signal']}  ({rule_pred['medium_confidence']:.0%})")
        print(f"  Long-term (7 years):    {rule_pred['long_signal']}  ({rule_pred['long_confidence']:.0%})")
        print(f"  7-Year Trend: {rule_pred['trend_7y']:+.2f}%\n")
        
        # ===== SUPPORT & RESISTANCE =====
        print("📈 SUPPORT & RESISTANCE:")
        print(f"  Support (1Y Low): ₹{targets['support']:.2f}")
        print(f"  Resistance (1Y High): ₹{targets['resistance']:.2f}")
        print(f"  Pivot Point: ₹{targets['pivot']:.2f}\n")
        
        # ===== VOLATILITY =====
        print("📊 VOLATILITY METRICS:")
        print(f"  Short-term (20d): {targets['volatilities']['vol_short']:.2f}%")
        print(f"  Medium-term (1Y): {targets['volatilities']['vol_medium']:.2f}%")
        print(f"  Long-term (7Y): {targets['volatilities']['vol_long']:.2f}%")
        print(f"  Average: {targets['volatilities']['vol_avg']:.2f}%\n")
        
        # ===== RISK METRICS =====
        print("⚡ RISK METRICS:")
        print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {risk['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {risk['max_drawdown']:.2f}%")
        print(f"  Value at Risk (VaR): {risk['var']:.2f}%")
        print(f"  Expected Value per Trade: ₹{risk['expected_value']:.2f}\n")
        
        # ===== TECHNICAL INDICATORS =====
        print("📈 TECHNICAL INDICATORS:")
        if 'rsi_short' in rule_pred:
            print(f"  RSI (14-day): {rule_pred['rsi_short']:.2f}")
        if 'rsi_medium' in rule_pred:
            print(f"  RSI (50-day): {rule_pred['rsi_medium']:.2f}")
        if 'position_in_range' in rule_pred:
            print(f"  Position in 7Y Range: {rule_pred['position_in_range']:.1f}%\n")
        
        # ===== HISTORICAL PERFORMANCE =====
        print("📊 HISTORICAL PERFORMANCE (Last 7 years):")
        if 'high_7y' in targets:
            print(f"  7-Year High: ₹{targets['high_7y']:.2f}")
        else:
            print(f"  7-Year High: Data not available")
    
        if 'low_7y' in targets:
            print(f"  7-Year Low: ₹{targets['low_7y']:.2f}")
        else:
            print(f"  7-Year Low: Data not available")
    
        if 'ma_7y' in targets:
            print(f"  7-Year Average: ₹{targets['ma_7y']:.2f}\n")
        else:
            print(f"  7-Year Average: Data not available\n")
        
        # ===== RISK RATING =====
        print("🎯 OVERALL RISK RATING:")
        print(f"  {risk['risk_rating']}")
        print(f"  Score: {risk['risk_score']:.1f}/100\n")
        
        # ===== MODEL ACCURACY =====
        print("🤖 MODEL ACCURACY:")
        print(f"  Rule-based: 65-70%")
        print(f"  ML Model: 74-85%")
        print(f"  Recommendation: Use ML when available\n")
        
        print("="*80 + "\n")
    
    def generate_full_report(self, ticker, show_expert=False):
        """Generate both views"""
        data = self.generate_simple_view(ticker)
        
        if show_expert and data:
            input("\n[Press ENTER to see Expert Analysis...]")
            self.generate_expert_view(data)
        
        return data


if __name__ == "__main__":
    summary = TradingSummary()
    
    print("\n" + "="*80)
    print("TRADING RECOMMENDATIONS FOR CUSTOMERS")
    print("="*80)
    
    stocks = ['IOC', 'AAPL', 'ITC']
    
    for ticker in stocks:
        summary.generate_full_report(ticker, show_expert=True)