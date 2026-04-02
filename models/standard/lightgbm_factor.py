"""
LightGBM Factor Model - Standard Version

Usage:
    from rdagent.components.model_loader import load_model
    model = load_model("lightgbm_factor")
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path


class LightGBMFactorModel:
    """
    LightGBM-based factor model for EUR/USD trading.
    
    Features:
    - Faster than XGBoost
    - Lower memory usage
    - Good for large datasets
    """
    
    def __init__(self, **params):
        self.params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            **params
        }
        self.model = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None, **fit_params):
        """Train the model."""
        self.feature_names = feature_names
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names if feature_names else 'auto')
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            **fit_params
        )
        
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n=10, importance_type='gain'):
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
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
        self.model = lgb.Booster(model_file=path)
        print(f"✓ Model loaded from {path}")
        return self


# Convenience function
def create_lightgbm_factor_model(**params):
    """Create LightGBM factor model."""
    return LightGBMFactorModel(**params)


if __name__ == "__main__":
    # Test
    print("=== LightGBM Factor Model Test ===")
    model = create_lightgbm_factor_model()
    print(f"✓ Model created with params: {model.params}")
