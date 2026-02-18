import pandas as pd
import numpy as np
from scipy import stats

class RiskManager:
    """Calculate risk metrics and probabilities using 7-year data"""
    
    def __init__(self):
        print("✓ Risk Manager initialized")
    
    def load_data(self, ticker):
        """Load stock data"""
        try:
            df = pd.read_csv(f'data/stocks/{ticker}.csv', index_col=0, parse_dates=True)
            
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in numeric_cols if col in df.columns]]
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            return df
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def calculate_daily_returns(self, df):
        """Calculate daily returns"""
        returns = df['Close'].pct_change() * 100
        return returns.dropna()
    
    def calculate_win_rate(self, df, entry_price, target_price):
        """Calculate historical win rate"""
        returns = self.calculate_daily_returns(df)
        
        # Calculate percentage move needed
        target_move = ((target_price - entry_price) / entry_price) * 100
        
        # Count days where price moved that much UP
        winning_days = (returns > target_move).sum()
        total_days = len(returns)
        
        win_rate = (winning_days / total_days) * 100
        
        return win_rate
    
    def calculate_loss_probability(self, df, entry_price, stop_loss):
        """Calculate probability of hitting stop loss"""
        returns = self.calculate_daily_returns(df)
        
        # Calculate percentage move needed
        stop_move = ((entry_price - stop_loss) / entry_price) * 100
        
        # Count days where price moved that much DOWN
        losing_days = (returns < -stop_move).sum()
        total_days = len(returns)
        
        loss_probability = (losing_days / total_days) * 100
        
        return loss_probability
    
    def calculate_expected_value(self, win_rate, profit, loss_rate, loss):
        """Calculate expected value per trade"""
        ev = (win_rate * profit) - (loss_rate * loss)
        return ev
    
    def calculate_sharpe_ratio(self, df, risk_free_rate=6.5):
        """Calculate Sharpe ratio (risk-adjusted returns)"""
        returns = self.calculate_daily_returns(df)
        
        # Annualize returns
        annual_return = returns.mean() * 252
        
        # Annualize volatility
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return sharpe_ratio
    
    def calculate_max_drawdown(self, df):
        """Calculate maximum drawdown"""
        closing_prices = df['Close']
        
        # Calculate running maximum
        running_max = closing_prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (closing_prices - running_max) / running_max * 100
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_volatility_metrics(self, df):
        """Calculate volatility metrics"""
        returns = self.calculate_daily_returns(df)
        
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside volatility (only negative days)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'downside_volatility': downside_vol
        }
    
    def calculate_sortino_ratio(self, df, risk_free_rate=6.5):
        """Calculate Sortino ratio (uses downside volatility)"""
        returns = self.calculate_daily_returns(df)
        
        # Annualize returns
        annual_return = returns.mean() * 252
        
        # Downside volatility
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return 0
        
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol
        
        return sortino_ratio
    
    def calculate_var_cvar(self, df, confidence=0.95):
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        returns = self.calculate_daily_returns(df)
        
        # VaR: the loss that will be exceeded with (1 - confidence) probability
        var = returns.quantile(1 - confidence)
        
        # CVaR: average loss beyond VaR
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def get_risk_rating(self, sharpe_ratio, max_drawdown, win_rate):
        """Get overall risk rating"""
        score = 0
        
        # Sharpe ratio component (0-3 points)
        if sharpe_ratio > 1.5:
            score += 3
        elif sharpe_ratio > 1.0:
            score += 2
        elif sharpe_ratio > 0:
            score += 1
        
        # Drawdown component (0-3 points)
        if max_drawdown > -10:
            score += 3
        elif max_drawdown > -20:
            score += 2
        elif max_drawdown > -30:
            score += 1
        
        # Win rate component (0-3 points)
        if win_rate > 60:
            score += 3
        elif win_rate > 50:
            score += 2
        elif win_rate > 40:
            score += 1
        
        total_score = score / 9 * 100
        
        if total_score >= 75:
            rating = "EXCELLENT 🟢 (Very Safe)"
        elif total_score >= 60:
            rating = "GOOD 🟡 (Acceptable)"
        elif total_score >= 40:
            rating = "MODERATE 🟠 (Fair Risk)"
        else:
            rating = "POOR 🔴 (High Risk)"
        
        return rating, total_score
    
    def generate_report(self, ticker, entry_price, target_price, stop_loss):
        """Generate comprehensive risk report"""
        print(f"\n{'='*80}")
        print(f"RISK ANALYSIS REPORT - {ticker}")
        print(f"{'='*80}")
        
        # Load data
        df = self.load_data(ticker)
        if df is None:
            print("✗ Failed to load data")
            return None
        
        # Calculate metrics
        win_rate = self.calculate_win_rate(df, entry_price, target_price)
        loss_probability = self.calculate_loss_probability(df, entry_price, stop_loss)
        
        profit = target_price - entry_price
        loss = entry_price - stop_loss
        
        expected_value = self.calculate_expected_value(
            win_rate / 100, 
            profit, 
            loss_probability / 100, 
            loss
        )
        
        sharpe_ratio = self.calculate_sharpe_ratio(df)
        sortino_ratio = self.calculate_sortino_ratio(df)
        max_drawdown = self.calculate_max_drawdown(df)
        
        volatility_metrics = self.calculate_volatility_metrics(df)
        var, cvar = self.calculate_var_cvar(df)
        
        rating, score = self.get_risk_rating(sharpe_ratio, max_drawdown, win_rate)
        
        # Display report
        print(f"\n✓ Stock: {ticker}")
        print(f"✓ Entry: ₹{entry_price:.2f} | Target: ₹{target_price:.2f} | Stop: ₹{stop_loss:.2f}")
        
        print(f"\n📊 WIN PROBABILITY:")
        print(f"  Win Rate: {win_rate:.2f}% ({'GOOD ✅' if win_rate > 50 else 'POOR ❌'})")
        print(f"  Loss Probability: {loss_probability:.2f}%")
        print(f"  Expected Win/Loss: {win_rate - loss_probability:.2f}%")
        
        print(f"\n💰 EXPECTED VALUE:")
        print(f"  Per Trade: ₹{expected_value:.2f}")
        ev_status = "POSITIVE ✅" if expected_value > 0 else "NEGATIVE ❌"
        print(f"  Status: {ev_status}")
        if expected_value > 0:
            print(f"  Over 100 trades: ₹{expected_value * 100:,.2f} expected profit")
        else:
            print(f"  Over 100 trades: ₹{abs(expected_value * 100):,.2f} expected loss")
        
        print(f"\n📈 RISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        sharpe_status = "GOOD ✅" if sharpe_ratio > 1.0 else "FAIR 🟡" if sharpe_ratio > 0 else "POOR ❌"
        print(f"  Status: {sharpe_status}")
        print(f"  (>1.0 = Good risk-adjusted returns)")
        
        print(f"\n  Sortino Ratio: {sortino_ratio:.2f}")
        print(f"  (Better for downside risk)")
        
        print(f"\n⚠️  MAXIMUM DRAWDOWN:")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        dd_status = "EXCELLENT ✅" if max_drawdown > -10 else "GOOD 🟡" if max_drawdown > -20 else "POOR ❌"
        print(f"  Status: {dd_status}")
        print(f"  (Worst case 1-day drop from historical high)")
        
        print(f"\n📊 VOLATILITY METRICS:")
        print(f"  Daily Volatility: {volatility_metrics['daily_volatility']:.2f}%")
        print(f"  Annual Volatility: {volatility_metrics['annual_volatility']:.2f}%")
        print(f"  Downside Volatility: {volatility_metrics['downside_volatility']:.2f}%")
        
        print(f"\n💔 VALUE AT RISK (95% confidence):")
        print(f"  VaR: {var:.2f}% (Max daily loss)")
        print(f"  CVaR: {cvar:.2f}% (Average loss if VaR exceeded)")
        print(f"  Interpretation: 95% chance daily loss < {abs(var):.2f}%")
        
        print(f"\n🎯 OVERALL RISK RATING:")
        print(f"  {rating}")
        print(f"  Score: {score:.1f}/100")
        
        print(f"\n✅ RECOMMENDATION:")
        if expected_value > 0 and sharpe_ratio > 0.5:
            print(f"  ✅ GOOD TRADE - Expected value is POSITIVE")
        elif expected_value > 0:
            print(f"  🟡 ACCEPTABLE - Expected value positive but risky")
        else:
            print(f"  ❌ POOR TRADE - Expected value is NEGATIVE")
        
        print(f"\n{'='*80}\n")
        
        return {
            'ticker': ticker,
            'win_rate': win_rate,
            'loss_probability': loss_probability,
            'expected_value': expected_value,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var': var,
            'cvar': cvar,
            'volatility_metrics': volatility_metrics,
            'risk_rating': rating,
            'risk_score': score
        }


if __name__ == "__main__":
    manager = RiskManager()
    
    # Test cases (from price_targets output)
    test_cases = [
        ('IOC', 175.15, 181.31, 171.64),      # Entry, Target, Stop
        ('AAPL', 255.78, 270.04, 250.43),
        ('ITC', 317.95, 333.71, 312.04)
    ]
    
    for ticker, entry, target, stop in test_cases:
        manager.generate_report(ticker, entry, target, stop)