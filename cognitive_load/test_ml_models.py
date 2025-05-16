"""
Test script to verify that ML models can be trained correctly
This script loads a small subset of data and trains a simplified model
to verify that the pipeline works end-to-end.
"""

import numpy as np
import os
from data_loader import prepare_all_experiments
from train_ml_models import (
    extract_features_from_signals,
    run_ml_experiment,
    ML_MODELS
)

def main():
    """Test ML model training"""
    print("Testing ML model training functionality...\n")
    
    # Try to load the previously failing experiment
    experiment_names = ['remote_ppg_remote_resp']
    experiment_data = prepare_all_experiments(experiment_names)
    
    if not experiment_data or not experiment_names[0] in experiment_data:
        print("❌ Error: Failed to load experiment data.")
        return False
    
    # Get the first experiment
    exp_data = experiment_data[experiment_names[0]]
    
    # Limit to a small subset for quick testing
    max_samples = 5  # Use just a few samples for quick testing
    
    # Create a simplified version of the experiment data
    simplified_exp_data = {}
    for split in ['train', 'val', 'test']:
        X, y, keys = exp_data[split]
        # Limit number of samples for faster testing
        if len(X) > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
            keys = keys[:max_samples] if keys is not None else None
        simplified_exp_data[split] = (X, y, keys)
    
    # Store original config
    simplified_exp_data['config'] = exp_data['config']
    
    # Create simplified model configs with fewer estimators for faster training
    test_models = {
        'rf': {
            'name': 'Random Forest (TEST)',
            'class': ML_MODELS['rf']['class'],
            'params': {
                'n_estimators': 10,  # Reduced for faster testing
                'max_depth': 3,
                'random_state': 2025
            }
        }
    }
    
    try:
        print("Extracting features from simplified dataset...")
        # Test feature extraction
        X_train, y_train, _ = simplified_exp_data['train']
        X_train_features = extract_features_from_signals(X_train)
        
        print(f"✅ Successfully extracted {X_train_features.shape[1]} features from {X_train_features.shape[0]} samples.")
        
        print("\nTesting model training pipeline...")
        # Run a minimal experiment with just Random Forest
        results = run_ml_experiment(simplified_exp_data, model_configs=test_models)
        
        if not results or 'rf' not in results:
            print("❌ Error: Failed to get results from the experiment.")
            return False
        
        # Check for model and evaluation results
        if 'model' not in results['rf']:
            print("❌ Error: Model not present in results.")
            return False
        
        if 'test_results' not in results['rf']:
            print("❌ Error: Test results not present in results.")
            return False
        
        # Print test accuracy
        test_acc = results['rf']['test_results']['accuracy']
        print(f"\n✅ Model trained successfully with test accuracy: {test_acc:.4f}")
        
        print("\n✅ ML model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during ML testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 