import pandas as pd
import numpy as np
import os

class DataPipeline:
    """Handle data loading, validation, and preparation"""
    
    def __init__(self, data_dir='data/stocks'):
        self.data_dir = data_dir
        self.df = None
        self.ticker = None
    
    def load_stock_data(self, ticker):
        """Load stock data from CSV file"""
        try:
            file_path = f'{self.data_dir}/{ticker}.csv'
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"✗ File not found: {file_path}")
                return None
            
            # Load CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"✓ Loaded raw data: {len(df)} rows")
            
            # Store
            self.df = df
            self.ticker = ticker
            
            return df
        
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def validate_data(self, df):
        """Validate data quality"""
        print(f"\nValidating data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"  ⚠️  Missing values found:")
            for col in missing[missing > 0].index:
                print(f"     {col}: {missing[col]} rows")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠️  Duplicate rows: {duplicates}")
        
        # Check columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✓ Data validation passed")
        return True
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print(f"Cleaning data...")
        
        # Keep only required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in required_cols if col in df.columns]]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN
        initial_rows = len(df)
        df = df.dropna()
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"  ⚠️  Removed {removed} rows with missing values")
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        print(f"  ✓ Cleaned data: {len(df)} rows remaining")
        
        return df
    
    def get_summary(self, df):
        """Get data summary"""
        if df is None or len(df) == 0:
            return None
    
    # Convert index to datetime if needed
        if isinstance(df.index[0], str):
            df.index = pd.to_datetime(df.index)
    
        summary = {
            'ticker': self.ticker,
            'rows': len(df),
            'date_range': f"{str(df.index[0])[:10]} to {str(df.index[-1])[:10]}",
            'current_price': df['Close'].iloc[-1],
            'high_52w': df['Close'].tail(252).max(),
            'low_52w': df['Close'].tail(252).min(),
            'avg_volume': df['Volume'].mean(),
            'current_volume': df['Volume'].iloc[-1]
        }
    
        return summary
    
    def process(self, ticker):
        """Complete pipeline: load → validate → clean"""
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")
        
        # Load
        df = self.load_stock_data(ticker)
        if df is None:
            return None
        
        # Validate
        if not self.validate_data(df):
            return None
        
        # Clean
        df = self.clean_data(df)
        
        # Summary
        summary = self.get_summary(df)
        print(f"\nData Summary:")
        print(f"  Ticker: {summary['ticker']}")
        print(f"  Rows: {summary['rows']}")
        print(f"  Date Range: {summary['date_range']}")
        print(f"  Current Price: ₹{summary['current_price']:.2f}")
        print(f"  52-Week High: ₹{summary['high_52w']:.2f}")
        print(f"  52-Week Low: ₹{summary['low_52w']:.2f}")
        print(f"  Average Volume: {summary['avg_volume']:.0f}")
        print(f"  Current Volume: {summary['current_volume']:.0f}")
        
        print(f"\n{'='*60}")
        print(f"✓ {ticker} data is ready!")
        print(f"{'='*60}\n")
        
        return df


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline()
    
    # Process IOC
    ioc_data = pipeline.process("IOC")
    
    # Process AAPL
    aapl_data = pipeline.process("AAPL")
    
    # Process ITC
    itc_data = pipeline.process("ITC")