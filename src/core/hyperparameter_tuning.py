import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from feature_engineering import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HyperparameterTuning:
    """Find optimal XGBoost parameters"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.scaler = StandardScaler()
    
    def prepare_data(self, ticker):
        """Prepare data"""
        df = self.feature_engineer.prepare_features(ticker)
        feature_cols = self.feature_engineer.get_feature_columns()
        
        X = df[feature_cols]
        y = df['target']
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def tune_xgboost(self, X_train, y_train, X_test, y_test, ticker):
        """Use GridSearchCV to find best parameters"""
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING - {ticker}")
        print(f"{'='*80}\n")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [150, 200, 250, 300],
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1, 0.5, 1]
        }
        
        # Base model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        # GridSearchCV with 5-fold cross-validation
        print("🔍 Searching optimal parameters (this takes 5-10 minutes)...\n")
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\n✅ Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\n✅ Best CV Score (AUC-ROC): {best_score:.4f}")
        
        # Test on test set
        best_model = grid_search.best_estimator_
        test_auc = best_model.score(X_test, y_test)
        print(f"✅ Test Set Score: {test_auc:.4f}")
        
        return best_model, best_params
    
    def optimize(self, ticker):
        """Run optimization"""
        X_train, X_test, y_train, y_test = self.prepare_data(ticker)
        best_model, best_params = self.tune_xgboost(X_train, y_train, X_test, y_test, ticker)
        
        return best_model, best_params


if __name__ == "__main__":
    tuner = HyperparameterTuning()
    
    for ticker in ['AAPL']:  # Start with AAPL (highest potential)
        model, params = tuner.optimize(ticker)