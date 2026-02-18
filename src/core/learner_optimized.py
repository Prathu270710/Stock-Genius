import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pickle
import os
from feature_engineering import FeatureEngineering

class OptimizedModelLearner:
    """Train models with ONLY selected features"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.scaler = StandardScaler()
        print("✓ Optimized Model Learner initialized")
    
    # Define best features for each stock (from feature selection)
    BEST_FEATURES = {
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
    
    def prepare_data(self, ticker):
        """Prepare data with ONLY selected features"""
        print(f"\n{'='*80}")
        print(f"PREPARING OPTIMIZED DATA - {ticker}")
        print(f"{'='*80}")
        
        df = self.feature_engineer.prepare_features(ticker)
        if df is None:
            return None, None, None, None
        
        # Get ONLY the best features for this stock
        selected_features = self.BEST_FEATURES[ticker]
        
        X = df[selected_features]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)
        
        print(f"✓ Selected Features: {len(selected_features)}")
        print(f"✓ Samples: {len(X)}")
        print(f"✓ Target distribution: {sum(y)} UP, {len(y) - sum(y)} DOWN")
        
        return X_scaled, y, selected_features, df
    
    def split_data(self, X, y):
        """Split data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_optimized_xgboost(self, X_train, X_test, y_train, y_test, ticker):
        """Train optimized XGBoost with selected features"""
        print(f"\n{'='*80}")
        print(f"TRAINING OPTIMIZED XGBOOST - {ticker}")
        print(f"{'='*80}")
        
        # Optimized parameters
        model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 OPTIMIZED XGBOOST RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        
        # Feature importance
        importance = model.feature_importances_
        feature_names = X_train.columns
        top_indices = np.argsort(importance)[-5:][::-1]
        
        print(f"\n  Top 5 Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"    {i}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def save_model(self, model, ticker):
        """Save trained model"""
        print(f"\n{'='*80}")
        print(f"SAVING OPTIMIZED MODEL - {ticker}")
        print(f"{'='*80}")
        
        models_dir = f'models/{ticker}_optimized'
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = f'{models_dir}/xgboost_optimized.pkl'
        pickle.dump(model, open(model_path, 'wb'))
        print(f"✓ XGBoost model saved: {model_path}")
        
        scaler_path = f'{models_dir}/scaler_optimized.pkl'
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        print(f"✓ Scaler saved: {scaler_path}")
    
    def train_all_models(self, ticker):
        """Train optimized model"""
        X, y, features, df = self.prepare_data(ticker)
        if X is None:
            return None
        
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        model, metrics = self.train_optimized_xgboost(X_train, X_test, y_train, y_test, ticker)
        self.save_model(model, ticker)
        
        return {
            'ticker': ticker,
            'metrics': metrics,
            'features': len(features),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }


if __name__ == "__main__":
    learner = OptimizedModelLearner()
    
    stocks = ['IOC', 'AAPL', 'ITC']
    results = {}
    
    for ticker in stocks:
        result = learner.train_all_models(ticker)
        if result:
            results[ticker] = result
    
    # Summary
    print(f"\n\n{'='*80}")
    print("OPTIMIZED TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    print("Stock     | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Features")
    print("-"*80)
    
    for ticker, result in results.items():
        m = result['metrics']
        print(f"{ticker:8} | {m['accuracy']*100:7.2f}% | {m['precision']*100:8.2f}% | "
              f"{m['recall']*100:6.2f}% | {m['f1']*100:7.2f}% | {m['auc']*100:7.2f}% | "
              f"{result['features']:8}")
    
    print(f"\n{'='*80}")
    print("✓ All optimized models trained and saved!")
    print(f"{'='*80}\n")