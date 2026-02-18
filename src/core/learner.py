import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pickle
import os
from feature_engineering import FeatureEngineering

class ModelLearner:
    """Train XGBoost models for stock prediction"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.scaler = StandardScaler()
        self.xgb_models = {}
        print("✓ Model Learner initialized (XGBoost only)")
    
    def prepare_data(self, ticker):
        """Prepare data for training"""
        print(f"\n{'='*80}")
        print(f"PREPARING DATA - {ticker}")
        print(f"{'='*80}")
        
        # Create features
        df = self.feature_engineer.prepare_features(ticker)
        if df is None:
            return None, None, None, None
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns()
        
        # Separate features and target
        X = df[feature_cols]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Samples: {len(X)}")
        print(f"✓ Target distribution: {sum(y)} UP, {len(y) - sum(y)} DOWN")
        
        return X_scaled, y, feature_cols, df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, ticker):
        """Train XGBoost model"""
        print(f"\n{'='*80}")
        print(f"TRAINING XGBOOST - {ticker}")
        print(f"{'='*80}")
        
        # Create and train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 XGBOOST RESULTS:")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1-Score:  {f1:.2%}")
        print(f"  AUC-ROC:   {auc:.2%}")
        
        # Feature importance
        importance = model.feature_importances_
        print(f"\n  Top 5 Important Features:")
        feature_names = X_train.columns
        top_indices = np.argsort(importance)[-5:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"    {i}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train_ensemble(self, X_train, X_test, y_train, y_test, ticker):
        """Train ensemble of multiple XGBoost models"""
        print(f"\n{'='*80}")
        print(f"TRAINING ENSEMBLE (5 Models) - {ticker}")
        print(f"{'='*80}")
        
        ensemble_predictions = np.zeros((len(X_test), 5))
        metrics_list = []
        
        for i in range(5):
            print(f"\n  Training model {i+1}/5...")
            
            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.12,
                subsample=0.75,
                colsample_bytree=0.75,
                random_state=42 + i,  # Different random seed
                eval_metric='logloss',
                verbosity=0
            )
            
            model.fit(X_train, y_train, verbose=False)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            ensemble_predictions[:, i] = y_pred_proba
            
            # Calculate accuracy
            y_pred = (y_pred_proba > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            print(f"    Model {i+1} Accuracy: {acc:.2%}")
            metrics_list.append(acc)
        
        # Average ensemble prediction
        ensemble_pred_proba = ensemble_predictions.mean(axis=1)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Calculate ensemble metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_pred_proba)
        
        print(f"\n📊 ENSEMBLE RESULTS (Average of 5 models):")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1-Score:  {f1:.2%}")
        print(f"  AUC-ROC:   {auc:.2%}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def save_model(self, model, ticker):
        """Save trained model"""
        print(f"\n{'='*80}")
        print(f"SAVING MODEL - {ticker}")
        print(f"{'='*80}")
        
        # Create models directory
        models_dir = f'models/{ticker}'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save XGBoost
        model_path = f'{models_dir}/xgboost_model.pkl'
        pickle.dump(model, open(model_path, 'wb'))
        print(f"✓ XGBoost model saved: {model_path}")
        
        # Save scaler
        scaler_path = f'{models_dir}/scaler.pkl'
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        print(f"✓ Scaler saved: {scaler_path}")
    
    def train_all_models(self, ticker):
        """Train all models for a stock"""
        # Prepare data
        X, y, feature_cols, df = self.prepare_data(ticker)
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train main XGBoost
        xgb_model, xgb_metrics = self.train_xgboost(X_train, X_test, y_train, y_test, ticker)
        
        # Train ensemble
        ensemble_metrics = self.train_ensemble(X_train, X_test, y_train, y_test, ticker)
        
        # Save main model
        self.save_model(xgb_model, ticker)
        
        # Return metrics
        return {
            'ticker': ticker,
            'xgb_metrics': xgb_metrics,
            'ensemble_metrics': ensemble_metrics,
            'feature_count': len(feature_cols),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }


if __name__ == "__main__":
    learner = ModelLearner()
    
    stocks = ['IOC', 'AAPL', 'ITC']
    results = {}
    
    for ticker in stocks:
        result = learner.train_all_models(ticker)
        if result:
            results[ticker] = result
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for ticker, result in results.items():
        print(f"📊 {ticker}")
        print(f"  XGBoost Accuracy:  {result['xgb_metrics']['accuracy']:.2%}")
        print(f"  Ensemble Accuracy: {result['ensemble_metrics']['accuracy']:.2%}")
        print(f"  Improvement: {(result['ensemble_metrics']['accuracy'] - result['xgb_metrics']['accuracy']):.2%}\n")
    
    print(f"{'='*80}")
    print("✓ All models trained and saved!")
    print(f"{'='*80}\n")