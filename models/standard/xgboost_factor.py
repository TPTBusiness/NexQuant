"""
XGBoost Factor Model - Standard Version

Usage:
    from rdagent.components.model_loader import load_model
    model = load_model("xgboost_factor")
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path


class XGBoostFactorModel:
    """
    XGBoost-based factor model for EUR/USD trading.
    
    Features:
    - Handles tabular data efficiently
    - Built-in feature importance
    - Fast training and inference
    """
    
    def __init__(self, **params):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            **params
        }
        self.model = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None, **fit_params):
        """Train the model."""
        self.feature_names = feature_names
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, **fit_params)
        
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        if self.feature_names is not None:
            indices = np.argsort(importance)[::-1][:top_n]
            return [(self.feature_names[i], importance[i]) for i in indices]
        return importance
    
    def save(self, path: str):
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        print(f"✓ Model loaded from {path}")
        return self


# Convenience function
def create_xgboost_factor_model(**params):
    """Create XGBoost factor model."""
    return XGBoostFactorModel(**params)


if __name__ == "__main__":
    # Test
    print("=== XGBoost Factor Model Test ===")
    model = create_xgboost_factor_model()
    print(f"✓ Model created with params: {model.params}")
