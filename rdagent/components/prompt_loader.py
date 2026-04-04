"""
Predix Prompt Loader

Loads prompts from:
1. prompts/local/*.yaml (your improved prompts - not in Git)
2. prompts/standard_prompts.yaml (default prompts - in Git)

Usage:
    from rdagent.components.prompt_loader import load_prompt
    
    # Load factor discovery prompt
    prompt = load_prompt("factor_discovery")
    
    # Load with custom local prompt
    prompt = load_prompt("factor_discovery", local_only=True)
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # Predix/
PROMPTS_DIR = BASE_DIR / "prompts"
LOCAL_PROMPTS_DIR = PROMPTS_DIR / "local"
STANDARD_PROMPTS_FILE = PROMPTS_DIR / "standard_prompts.yaml"


def get_local_prompt_path(name: str) -> Optional[Path]:
    """Find local prompt file by name.
    
    Priority:
    1. {name}_v2.yaml (latest version)
    2. {name}_v1.yaml
    3. {name}.yaml
    """
    if not LOCAL_PROMPTS_DIR.exists():
        return None
    
    # Try versioned files first (v3, v2, v1, etc.)
    for version in ["v3", "v2", "v1"]:
        for ext in ["yaml", "yml"]:
            path = LOCAL_PROMPTS_DIR / f"{name}_{version}.{ext}"
            if path.exists():
                print(f"  (found versioned: {name}_{version}.{ext})")
                return path
    
    # Try exact name
    for ext in ["yaml", "yml"]:
        path = LOCAL_PROMPTS_DIR / f"{name}.{ext}"
        if path.exists():
            return path
    
    # Try subdirectories
    for subdir in LOCAL_PROMPTS_DIR.iterdir():
        if subdir.is_dir():
            for ext in ["yaml", "yml"]:
                path = subdir / f"{name}.{ext}"
                if path.exists():
                    return path
    
    return None


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompt(
    name: str,
    section: Optional[str] = None,
    local_only: bool = False,
    fallback_to_standard: bool = True
) -> str:
    """
    Load a prompt by name.
    
    Priority:
    1. prompts/local/{name}.yaml (if exists)
    2. prompts/standard_prompts.yaml (if fallback_to_standard=True)
    
    Args:
        name: Prompt name (e.g., "factor_discovery")
        section: Specific section in YAML (e.g., "system" or "user")
        local_only: Only load from local/, raise error if not found
        fallback_to_standard: If True, fall back to standard prompts
    
    Returns:
        Prompt text
    
    Raises:
        FileNotFoundError: If prompt not found
    """
    # Try local prompts first
    local_path = get_local_prompt_path(name)
    
    if local_path:
        print(f"✓ Loading prompt '{name}' from local: {local_path}")
        data = load_yaml_file(local_path)
        
        if section:
            return data.get(section, "")
        
        # If data is dict with 'system' and 'user', return full dict
        if isinstance(data, dict):
            return data
        return str(data)
    
    # Local not found
    if local_only:
        raise FileNotFoundError(f"Local prompt '{name}' not found in {LOCAL_PROMPTS_DIR}")
    
    # Try standard prompts
    if not fallback_to_standard:
        raise FileNotFoundError(f"Prompt '{name}' not found")
    
    if not STANDARD_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Standard prompts file not found: {STANDARD_PROMPTS_FILE}")
    
    print(f"✓ Loading prompt '{name}' from standard prompts")
    data = load_yaml_file(STANDARD_PROMPTS_FILE)
    
    # Get section from standard prompts
    if name in data:
        prompt_data = data[name]
        
        if section and isinstance(prompt_data, dict):
            return prompt_data.get(section, "")
        
        return prompt_data
    
    raise FileNotFoundError(f"Prompt '{name}' not found in standard prompts")


def load_all_prompts() -> Dict[str, Any]:
    """Load all available prompts."""
    result = {}
    
    # Load standard prompts
    if STANDARD_PROMPTS_FILE.exists():
        result["standard"] = load_yaml_file(STANDARD_PROMPTS_FILE)
    
    # Load local prompts
    if LOCAL_PROMPTS_DIR.exists():
        result["local"] = {}
        for path in LOCAL_PROMPTS_DIR.glob("*.yaml"):
            result["local"][path.stem] = load_yaml_file(path)
    
    return result


def list_available_prompts() -> Dict[str, list]:
    """List all available prompts."""
    result = {"standard": [], "local": []}
    
    # Standard prompts
    if STANDARD_PROMPTS_FILE.exists():
        data = load_yaml_file(STANDARD_PROMPTS_FILE)
        result["standard"] = list(data.keys())
    
    # Local prompts
    if LOCAL_PROMPTS_DIR.exists():
        result["local"] = [p.stem for p in LOCAL_PROMPTS_DIR.glob("*.yaml")]
    
    return result


# Convenience functions for specific prompts
def get_factor_discovery_prompt() -> Dict[str, str]:
    """Get factor discovery prompt (system + user)."""
    return load_prompt("factor_discovery")


def get_factor_evolution_prompt() -> Dict[str, str]:
    """Get factor evolution prompt."""
    return load_prompt("factor_evolution")


def get_model_coder_prompt() -> Dict[str, str]:
    """Get model coder prompt."""
    return load_prompt("model_coder")


def get_trading_strategy_prompt() -> Dict[str, str]:
    """Get trading strategy prompt."""
    return load_prompt("trading_strategy")


# Test function
if __name__ == "__main__":
    print("=== Available Prompts ===")
    available = list_available_prompts()
    print(f"Standard: {available['standard']}")
    print(f"Local: {available['local']}")
    
    print("\n=== Testing Prompt Load ===")
    try:
        prompt = load_prompt("factor_discovery")
        print(f"✓ Loaded factor_discovery prompt")
        
        # Handle nested dict structure (local prompts)
        if isinstance(prompt, dict):
            if 'factor_discovery' in prompt:
                # Local prompt structure
                fd = prompt['factor_discovery']
                print(f"  System: {len(fd.get('system', ''))} chars")
                print(f"  User: {len(fd.get('user', ''))} chars")
            else:
                # Standard prompt structure
                print(f"  System: {len(prompt.get('system', ''))} chars")
                print(f"  User: {len(prompt.get('user', ''))} chars")
        else:
            print(f"  Content: {len(str(prompt))} chars")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
