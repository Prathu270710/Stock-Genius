import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
from feature_engineering import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    """Select only the best features"""
    
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
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def select_by_statistical_test(self, X_train, y_train, feature_cols, k=25):
        """Use statistical test (f_classif)"""
        print(f"\n{'='*80}")
        print(f"METHOD 1: Statistical Test (SelectKBest)")
        print(f"{'='*80}\n")
        
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        
        # Get scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'Feature': feature_cols,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        print("Top 10 Features (Statistical):")
        for i, row in feature_scores.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Score']:.4f}")
        
        return selected_features, feature_scores
    
    def select_by_xgboost_importance(self, X_train, y_train, feature_cols, k=25):
        """Use XGBoost feature importance"""
        print(f"\n{'='*80}")
        print(f"METHOD 2: XGBoost Importance")
        print(f"{'='*80}\n")
        
        # Train XGBoost to get importance
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Get importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Select top k
        top_features = feature_importance.head(k)['Feature'].tolist()
        
        print(f"Top 10 Features (XGBoost):")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return top_features, feature_importance
    
    def evaluate_feature_set(self, X_train, X_test, y_train, y_test, features, method_name):
        """Train model with selected features and evaluate"""
        print(f"\n{'='*80}")
        print(f"EVALUATION: {method_name}")
        print(f"{'='*80}\n")
        
        # Select only chosen features
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]
        
        # Train
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        accuracy = model.score(X_test_selected, y_test)
        
        print(f"Features used: {len(features)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy, model
    
    def select_best_features(self, ticker):
        """Run complete feature selection"""
        print(f"\n{'='*80}")
        print(f"FEATURE SELECTION - {ticker}")
        print(f"{'='*80}\n")
        
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(ticker)
        
        # Method 1: Statistical
        stat_features, stat_scores = self.select_by_statistical_test(X_train, y_train, feature_cols, k=25)
        
        # Method 2: XGBoost Importance
        xgb_features, xgb_importance = self.select_by_xgboost_importance(X_train, y_train, feature_cols, k=25)
        
        # Evaluate with each method
        stat_acc, stat_model = self.evaluate_feature_set(X_train, X_test, y_train, y_test, stat_features, "Statistical Features")
        xgb_acc, xgb_model = self.evaluate_feature_set(X_train, X_test, y_train, y_test, xgb_features, "XGBoost Features")
        
        # Choose best
        if xgb_acc > stat_acc:
            best_features = xgb_features
            best_acc = xgb_acc
            best_method = "XGBoost"
        else:
            best_features = stat_features
            best_acc = stat_acc
            best_method = "Statistical"
        
        print(f"\n{'='*80}")
        print(f"BEST METHOD: {best_method}")
        print(f"Accuracy: {best_acc*100:.2f}%")
        print(f"Features: {len(best_features)}")
        print(f"{'='*80}\n")
        
        print(f"Selected Features:")
        for i, feat in enumerate(best_features, 1):
            print(f"  {i}. {feat}")
        
        return best_features, best_acc


if __name__ == "__main__":
    selector = FeatureSelector()
    
    for ticker in ['IOC', 'AAPL', 'ITC']:
        best_features, best_acc = selector.select_best_features(ticker)