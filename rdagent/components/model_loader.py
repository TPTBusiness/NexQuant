"""
Predix Model Loader

Loads models from:
1. models/local/*.py (your improved models - not in Git)
2. models/standard/*.py (default models - in Git)

Usage:
    from rdagent.components.model_loader import load_model
    
    # Load XGBoost model
    model = load_model("xgboost_factor")
    
    # Load your improved version (if exists in models/local/)
    model = load_model("transformer_factor")  # Auto-loads from local if exists
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Any


# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # Predix/
MODELS_DIR = BASE_DIR / "models"
LOCAL_MODELS_DIR = MODELS_DIR / "local"
STANDARD_MODELS_DIR = MODELS_DIR / "standard"


def get_local_model_path(name: str) -> Optional[Path]:
    """Find local model file by name.
    
    Priority:
    1. {name}_v2.py (latest version)
    2. {name}_v1.py
    3. {name}.py
    """
    if not LOCAL_MODELS_DIR.exists():
        return None
    
    # Try versioned files first (v2, v1, etc.)
    for version in ["v2", "v1"]:
        path = LOCAL_MODELS_DIR / f"{name}_{version}.py"
        if path.exists():
            print(f"  (found versioned: {name}_{version}.py)")
            return path
    
    # Try exact name
    path = LOCAL_MODELS_DIR / f"{name}.py"
    if path.exists():
        return path
    
    return None


def get_standard_model_path(name: str) -> Optional[Path]:
    """Find standard model file by name."""
    if not STANDARD_MODELS_DIR.exists():
        return None
    
    path = STANDARD_MODELS_DIR / f"{name}.py"
    if path.exists():
        return path
    
    return None


def load_module_from_path(path: Path, module_name: str) -> Any:
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # nosec
    
    return module


def load_model(name: str, local_only: bool = False, fallback_to_standard: bool = True):
    """
    Load a model by name.
    
    Priority:
    1. models/local/{name}.py (if exists)
    2. models/standard/{name}.py (if fallback_to_standard=True)
    
    Args:
        name: Model name (e.g., "xgboost_factor", "transformer_factor")
        local_only: Only load from local/, raise error if not found
        fallback_to_standard: If True, fall back to standard models
    
    Returns:
        Model class or instance
    
    Raises:
        FileNotFoundError: If model not found
        ImportError: If model cannot be loaded
    """
    # Try local models first
    local_path = get_local_model_path(name)
    
    if local_path:
        print(f"✓ Loading model '{name}' from local: {local_path}")
        module = load_module_from_path(local_path, f"local_{name}")
        
        # Try to find create_* or Model class
        for attr_name in dir(module):
            if attr_name.startswith('create_') and name.replace('_', '') in attr_name.replace('create_', ''):
                return getattr(module, attr_name)
            if attr_name.endswith('Model') and name.replace('_', '') in attr_name.lower():
                return getattr(module, attr_name)
        
        # Return module if no specific class found
        return module
    
    # Local not found
    if local_only:
        raise FileNotFoundError(f"Local model '{name}' not found in {LOCAL_MODELS_DIR}")
    
    # Try standard models
    if not fallback_to_standard:
        raise FileNotFoundError(f"Model '{name}' not found")
    
    standard_path = get_standard_model_path(name)
    if not standard_path:
        raise FileNotFoundError(f"Model '{name}' not found in standard or local directories")
    
    print(f"✓ Loading model '{name}' from standard: {standard_path}")
    module = load_module_from_path(standard_path, f"standard_{name}")
    
    # Try to find create_* or Model class
    for attr_name in dir(module):
        if attr_name.startswith('create_') and name.replace('_', '') in attr_name.replace('create_', ''):
            return getattr(module, attr_name)
        if attr_name.endswith('Model') and name.replace('_', '') in attr_name.lower():
            return getattr(module, attr_name)
    
    return module


def list_available_models() -> dict:
    """List all available models."""
    result = {"standard": [], "local": []}
    
    # Standard models
    if STANDARD_MODELS_DIR.exists():
        result["standard"] = [p.stem for p in STANDARD_MODELS_DIR.glob("*.py") if not p.name.startswith('_')]
    
    # Local models
    if LOCAL_MODELS_DIR.exists():
        result["local"] = [p.stem for p in LOCAL_MODELS_DIR.glob("*.py") if not p.name.startswith('_')]
    
    return result


# Convenience functions for specific models
def get_xgboost_model(**params):
    """Get XGBoost model."""
    return load_model("xgboost_factor")(**params)


def get_lightgbm_model(**params):
    """Get LightGBM model."""
    return load_model("lightgbm_factor")(**params)


def get_randomforest_model(**params):
    """Get RandomForest model."""
    return load_model("randomforest_factor")(**params)


# Test function
if __name__ == "__main__":
    print("=== Available Models ===")
    available = list_available_models()
    print(f"Standard: {available['standard']}")
    print(f"Local: {available['local']}")
    
    print("\n=== Testing Model Load ===")
    try:
        # Test XGBoost
        xgb_factory = load_model("xgboost_factor")
        print(f"✓ Loaded xgboost_factor")
        
        # Test LightGBM
        lgb_factory = load_model("lightgbm_factor")
        print(f"✓ Loaded lightgbm_factor")
        
    except Exception as e:
        print(f"✗ Error: {e}")
