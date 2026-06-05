import os
import yaml
import importlib

from typing import Optional, Dict, List, Type
from argparse import Namespace

# Registry for wrapper classes
WRAPPER_REGISTRY = {}

def register_wrapper(name: str, dynamic: str):
    """
    Decorator to register a wrapper class
    """
    def decorator(cls):
        if dynamic not in WRAPPER_REGISTRY:
            WRAPPER_REGISTRY[dynamic] = {}
        WRAPPER_REGISTRY[dynamic][name] = cls
        return cls
    return decorator

def get_wrapper_class(dynamic: str, name: str) -> Type:
    """
    Get wrapper class by dynamic type and name
    """
    if dynamic not in WRAPPER_REGISTRY or name not in WRAPPER_REGISTRY[dynamic]:
        raise ValueError(f"Wrapper {dynamic}.{name} not found in registry")
    return WRAPPER_REGISTRY[dynamic][name]

def load_config_and_class(args: Namespace):
    """
    Load configuration and wrapper class based on arguments
    
    Args:
        args: Command line arguments containing 'dynamic' and 'style' keys
        
    Returns:
        Tuple of (config, wrapper_cls)
    """
    dynamic = getattr(args, "dynamic", 'token_dynamic')
    style = getattr(args, "style", 'skipgpt')
    config_name = getattr(args, "config_name", style)
    if config_name == "": config_name = style

    # Load all wrapper classes for the specified dynamic type
    _load_wrapper_classes(dynamic)

    # Load yaml config
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, f"wrapper/{dynamic}/{config_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get wrapper class
    wrapper_cls = get_wrapper_class(dynamic, style)
    
    return config, wrapper_cls, config_path

def _load_wrapper_classes(dynamic: str):
    """
    Dynamically import and register all wrapper classes for a given dynamic type
    
    Args:
        dynamic: The dynamic type (e.g., 'token_dynamic')
    """
    # Skip if already loaded
    if dynamic in WRAPPER_REGISTRY:
        return
    
    # Get the path to the dynamic module
    base_path = os.path.dirname(os.path.abspath(__file__))
    dynamic_path = os.path.join(base_path, f"wrapper/{dynamic}")
    
    if not os.path.exists(dynamic_path):
        raise ValueError(f"Dynamic wrapper path not found: {dynamic_path}")
    
    # Import the module to trigger registration
    module_path = f"wrapper.{dynamic}"
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_path}: {e}")