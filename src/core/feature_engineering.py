import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    """Create 50+ advanced features for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        print("✓ Feature Engineering initialized (Advanced)")
    
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
    
    # ========== STEP 1: PRICE FEATURES ==========
    def create_price_features(self, df):
        """Create price-based features"""
        df = df.copy()
        
        df['daily_return'] = df['Close'].pct_change() * 100
        df['weekly_return'] = df['Close'].pct_change(5) * 100
        df['monthly_return'] = df['Close'].pct_change(20) * 100
        
        df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    # ========== STEP 2: MOVING AVERAGE FEATURES ==========
    def create_moving_average_features(self, df):
        """Create moving average features"""
        df = df.copy()
        
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
            df[f'Close_MA_{period}_ratio'] = df['Close'] / df[f'MA_{period}']
        
        return df
    
    # ========== STEP 3: MOMENTUM FEATURES ==========
    def create_momentum_features(self, df):
        """Create momentum indicators"""
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # Momentum
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_20'] = df['Close'] - df['Close'].shift(20)
        
        return df
    
    # ========== STEP 4: VOLATILITY FEATURES ==========
    def create_volatility_features(self, df):
        """Create volatility features"""
        df = df.copy()
        
        # Bollinger Bands
        df['BB_MA_20'] = df['Close'].rolling(20).mean()
        df['BB_STD_20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_MA_20'] + (df['BB_STD_20'] * 2)
        df['BB_Lower'] = df['BB_MA_20'] - (df['BB_STD_20'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        
        # Historical volatility
        df['HV_20'] = df['Close'].pct_change().rolling(20).std() * 100
        df['HV_50'] = df['Close'].pct_change().rolling(50).std() * 100
        
        return df
    
    # ========== STEP 5: VOLUME FEATURES ==========
    def create_volume_features(self, df):
        """Create volume-based features"""
        df = df.copy()
        
        df['volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_MA_20']
        
        df['price_change'] = df['Close'].diff()
        df['OBV'] = (np.sign(df['price_change']) * df['Volume']).fillna(0).cumsum()
        
        # Volume Price Trend
        df['VPT'] = df['Volume'] * df['daily_return']
        df['VPT_MA'] = df['VPT'].rolling(14).mean()
        
        return df
    
    # ========== STEP 6: ADVANCED TECHNICAL FEATURES ==========
    def create_advanced_technical_features(self, df):
        """Create advanced technical indicators"""
        df = df.copy()
        
        # ===== STOCHASTIC OSCILLATOR =====
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        df['Stoch_RSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min() + 1e-10)
        
        # ===== WILLIAMS %R =====
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
        
        # ===== CCI (Commodity Channel Index) =====
        sma_20 = df['Close'].rolling(20).mean()
        mad = df['Close'].rolling(20).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
        df['CCI'] = (df['Close'] - sma_20) / (0.015 * mad + 1e-10)
        
        # ===== ADX (Average Directional Index) =====
        df['plus_dm'] = np.where(df['High'].diff() > df['Low'].diff().abs(), df['High'].diff(), 0)
        df['minus_dm'] = np.where(df['Low'].diff().abs() > df['High'].diff(), df['Low'].diff().abs(), 0)
        df['plus_di'] = 100 * df['plus_dm'].rolling(14).mean() / (df['ATR_14'] + 1e-10)
        df['minus_di'] = 100 * df['minus_dm'].rolling(14).mean() / (df['ATR_14'] + 1e-10)
        df['ADX'] = abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        
        # ===== PRICE CHANNELS =====
        df['Price_Channel_High'] = df['High'].rolling(20).max()
        df['Price_Channel_Low'] = df['Low'].rolling(20).min()
        df['Price_Channel_Mid'] = (df['Price_Channel_High'] + df['Price_Channel_Low']) / 2
        
        # ===== KELTNER CHANNELS =====
        df['KC_Mid'] = df['Close'].rolling(20).mean()
        df['KC_ATR'] = df['ATR_14'] * 2
        df['KC_Upper'] = df['KC_Mid'] + df['KC_ATR']
        df['KC_Lower'] = df['KC_Mid'] - df['KC_ATR']
        
        return df
    
    # ========== STEP 7: ADVANCED PRICE ACTION FEATURES ==========
    def create_price_action_features(self, df):
        """Create advanced price action and correlation features"""
        df = df.copy()
        
        # ===== CANDLE PATTERNS =====
        df['Candle_Direction'] = np.where(df['Close'] > df['Open'], 1, 0)
        df['Candle_Size'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Lower_Wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        # ===== PRICE MOMENTUM =====
        df['Price_Acceleration'] = df['momentum_10'].diff()
        df['Price_Velocity'] = df['Close'].diff()
        
        # ===== CORRELATION FEATURES =====
        df['Close_Open_Corr'] = df['Close'].rolling(20).corr(df['Open'])
        df['Volume_Price_Corr'] = df['Volume'].rolling(20).corr(df['Close'].pct_change().abs())
        df['Return_Volume_Corr'] = df['daily_return'].rolling(20).corr(df['volume_ratio'])
        
        # ===== SUPPORT & RESISTANCE =====
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        df['Distance_to_Support'] = df['Close'] - df['Support']
        df['Distance_to_Resistance'] = df['Resistance'] - df['Close']
        
        # ===== TREND STRENGTH =====
        df['Close_above_MA20'] = np.where(df['Close'] > df['MA_20'], 1, 0)
        df['MA20_above_MA50'] = np.where(df['MA_20'] > df['MA_50'], 1, 0)
        df['MA50_above_MA200'] = np.where(df['MA_50'] > df['MA_200'], 1, 0)
        df['Trend_Alignment'] = df['Close_above_MA20'] + df['MA20_above_MA50'] + df['MA50_above_MA200']
        
        # ===== RELATIVE STRENGTH =====
        df['Close_to_Open_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['High_to_Low_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Close_Position_in_Range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # ===== MULTI-TIMEFRAME FEATURES =====
        df['MA5_MA20_Ratio'] = df['MA_5'] / (df['MA_20'] + 1e-10)
        df['MA20_MA200_Ratio'] = df['MA_20'] / (df['MA_200'] + 1e-10)
        df['RSI_Position'] = df['RSI'] / 100  # Normalize to 0-1
        
        return df
    
    def create_target_variable(self, df, lookforward=5):
        """Create target variable for ML"""
        df = df.copy()
        
        df['future_return'] = df['Close'].shift(-lookforward).pct_change(lookforward) * 100
        df['target'] = (df['future_return'] > 0).astype(int)
        
        return df
    
    def prepare_features(self, ticker):
        """Create ALL features and prepare for ML"""
        print(f"\nPreparing advanced features for {ticker}...")
        
        df = self.load_data(ticker)
        if df is None:
            return None
        
        # Step 1: Price features
        df = self.create_price_features(df)
        print("  ✓ Step 1: Price features")
        
        # Step 2: Moving averages
        df = self.create_moving_average_features(df)
        print("  ✓ Step 2: Moving average features")
        
        # Step 3: Momentum
        df = self.create_momentum_features(df)
        print("  ✓ Step 3: Momentum features")
        
        # Step 4: Volatility
        df = self.create_volatility_features(df)
        print("  ✓ Step 4: Volatility features")
        
        # Step 5: Volume
        df = self.create_volume_features(df)
        print("  ✓ Step 5: Volume features")
        
        # Step 6: Advanced technical (NEW!)
        df = self.create_advanced_technical_features(df)
        print("  ✓ Step 6: Advanced technical features")
        
        # Step 7: Price action (NEW!)
        df = self.create_price_action_features(df)
        print("  ✓ Step 7: Price action features")
        
        # Target variable
        df = self.create_target_variable(df)
        
        # Remove NaN rows
        df = df.dropna()
        
        print(f"✓ Created {len(df.columns)} total features")
        print(f"✓ Samples: {len(df)} rows")
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature columns (excluding target)"""
        features = [
            # Step 1: Price features
            'daily_return', 'weekly_return', 'monthly_return',
            'close_open_ratio', 'high_low_ratio',
            
            # Step 2: Moving averages
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100', 'MA_200',
            'Close_MA_5_ratio', 'Close_MA_10_ratio', 'Close_MA_20_ratio',
            'Close_MA_50_ratio', 'Close_MA_100_ratio', 'Close_MA_200_ratio',
            
            # Step 3: Momentum
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'momentum_10', 'momentum_20',
            
            # Step 4: Volatility
            'BB_MA_20', 'BB_STD_20', 'BB_Upper', 'BB_Lower', 'BB_Position',
            'ATR_14', 'HV_20', 'HV_50',
            
            # Step 5: Volume
            'volume_ratio', 'OBV', 'VPT_MA',
            
            # Step 6: Advanced technical (NEW!)
            'Stoch_K', 'Stoch_D', 'Stoch_RSI', 'Williams_R', 'CCI',
            'plus_di', 'minus_di', 'ADX',
            'Price_Channel_High', 'Price_Channel_Low', 'Price_Channel_Mid',
            'KC_Upper', 'KC_Lower',
            
            # Step 7: Price action (NEW!)
            'Candle_Direction', 'Candle_Size', 'Body_Size', 'Upper_Wick', 'Lower_Wick',
            'Price_Acceleration', 'Price_Velocity',
            'Close_Open_Corr', 'Volume_Price_Corr', 'Return_Volume_Corr',
            'Support', 'Resistance', 'Distance_to_Support', 'Distance_to_Resistance',
            'Close_above_MA20', 'MA20_above_MA50', 'MA50_above_MA200', 'Trend_Alignment',
            'Close_to_Open_Pct', 'High_to_Low_Pct', 'Close_Position_in_Range',
            'MA5_MA20_Ratio', 'MA20_MA200_Ratio', 'RSI_Position'
        ]
        return features


if __name__ == "__main__":
    fe = FeatureEngineering()
    
    for ticker in ['IOC', 'AAPL', 'ITC']:
        print(f"\n{'='*80}")
        print(f"Feature Engineering - {ticker}")
        print(f"{'='*80}")
        
        df = fe.prepare_features(ticker)
        
        if df is not None:
            print(f"\nFeature Sample (first 5 rows):")
            print(df.head())
            print(f"\nTarget distribution:")
            print(f"  UP (1): {(df['target'] == 1).sum()} ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
            print(f"  DOWN (0): {(df['target'] == 0).sum()} ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
