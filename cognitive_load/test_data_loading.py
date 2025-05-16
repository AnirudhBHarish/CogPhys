"""
Test script to verify that data loading works correctly
"""

import numpy as np
import os
from data_loader import prepare_all_experiments, EXPERIMENT_CONFIGS

def main():
    """Test data loading functions"""
    print("Testing data loading functionality...\n")
    
    # Try to load the previously failing experiment
    experiment_names = ['remote_ppg_remote_resp']
    experiment_data = prepare_all_experiments(experiment_names)
    
    if not experiment_data:
        print("❌ Error: No experiment data was loaded.")
        return False
    
    print("\nChecking experiment data structure...")
    for name, data in experiment_data.items():
        if data is None:
            print(f"❌ Error: Data for experiment '{name}' is None.")
            continue
        
        print(f"\nExperiment: {EXPERIMENT_CONFIGS[name]['description']}")
        
        # Check if data has all required keys
        for split in ['train', 'val', 'test']:
            if split not in data:
                print(f"❌ Error: Missing '{split}' split in experiment data.")
                continue
                
            X, y, keys = data[split]
            
            if X is None or len(X) == 0:
                print(f"❌ Error: X_{split} is empty or None.")
                continue
                
            print(f"✅ {split.capitalize()} data: {X.shape}, labels: {y.shape}")
            
            # Check signal dimensions
            if len(X.shape) != 3:
                print(f"❌ Error: X_{split} should have 3 dimensions (n_samples, n_signals, signal_length)")
                continue
                
            print(f"  - Number of signals per sample: {X.shape[1]}")
            print(f"  - Signal length: {X.shape[2]}")
            
            # Check if we have any NaN or Inf values
            if np.isnan(X).any():
                print(f"❌ Error: X_{split} contains NaN values.")
            if np.isinf(X).any():
                print(f"❌ Error: X_{split} contains Inf values.")
                
            # Check class distribution
            if len(np.unique(y)) != 2:
                print(f"❌ Error: y_{split} does not have exactly 2 classes.")
            else:
                class_counts = np.bincount(y)
                print(f"  - Class distribution: {class_counts}")
                
    print("\n✅ Data loading test completed!")
    return True

if __name__ == "__main__":
    main() 