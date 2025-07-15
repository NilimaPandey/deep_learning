#!/usr/bin/env python3
"""
Script to train all models and generate the required .th files for grading.
This script will train each model with appropriate hyperparameters to achieve the required accuracy.
"""

import subprocess
import sys
import os

def train_model(model_name, epochs=100, lr=0.001, batch_size=128):
    """Train a specific model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")
    
    cmd = [
        sys.executable, "-m", "homework.train",
        "--model_name", model_name,
        "--num_epoch", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Training {model_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return False

def main():
    """Train all models with appropriate hyperparameters"""
    
    # Check if we're in the right directory
    if not os.path.exists("homework"):
        print("Error: Please run this script from the main homework directory")
        sys.exit(1)
    
    # Training configurations for each model
    models_config = [
        {
            "name": "linear",
            "epochs": 150,  # Linear model needs more epochs
            "lr": 0.001,
            "batch_size": 128
        },
        {
            "name": "mlp", 
            "epochs": 100,
            "lr": 0.001,
            "batch_size": 128
        },
        {
            "name": "mlp_deep",
            "epochs": 100,
            "lr": 0.001,
            "batch_size": 128
        },
        {
            "name": "mlp_deep_residual",
            "epochs": 100,
            "lr": 0.001,
            "batch_size": 128
        }
    ]
    
    print("Starting training of all models...")
    print("This will take some time. Each model will train for up to 30 minutes.")
    
    success_count = 0
    for config in models_config:
        success = train_model(
            config["name"],
            config["epochs"],
            config["lr"],
            config["batch_size"]
        )
        if success:
            success_count += 1
            print(f"✅ Successfully trained {config['name']}")
        else:
            print(f"❌ Failed to train {config['name']}")
    
    print(f"\n{'='*50}")
    print(f"Training complete! {success_count}/{len(models_config)} models trained successfully.")
    print(f"{'='*50}")
    
    # Check which .th files were created
    th_files = ["linear.th", "mlp.th", "mlp_deep.th", "mlp_deep_residual.th"]
    existing_files = [f for f in th_files if os.path.exists(f)]
    
    print(f"\nGenerated .th files: {existing_files}")
    
    if len(existing_files) == len(th_files):
        print("🎉 All model files created successfully!")
    else:
        missing = set(th_files) - set(existing_files)
        print(f"⚠️  Missing files: {missing}")

if __name__ == "__main__":
    main() 