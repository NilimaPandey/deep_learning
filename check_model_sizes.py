#!/usr/bin/env python3
"""
Script to check model sizes and ensure they fit within the 10MB limit.
"""

import sys
import os
sys.path.append('.')

from homework.models import load_model

def check_model_size(model_name):
    """Check the size of a specific model"""
    try:
        model = load_model(model_name)
        size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
        print(f"{model_name}: {size_mb:.2f} MB")
        return size_mb <= 10
    except Exception as e:
        print(f"{model_name}: Error - {e}")
        return False

def main():
    """Check all model sizes"""
    print("Checking model sizes...")
    print("=" * 40)
    
    models = ["linear", "mlp", "mlp_deep", "mlp_deep_residual"]
    all_ok = True
    
    for model_name in models:
        ok = check_model_size(model_name)
        if not ok:
            all_ok = False
    
    print("=" * 40)
    if all_ok:
        print("✅ All models are under 10MB limit!")
    else:
        print("❌ Some models exceed 10MB limit!")

if __name__ == "__main__":
    main() 