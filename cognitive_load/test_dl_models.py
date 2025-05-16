"""
Test script to verify that DL models can be trained correctly
This script loads a small subset of data and trains a simplified model
for just a few epochs to verify that the pipeline works end-to-end.
"""

import numpy as np
import os
import torch
import lightning.pytorch as pl
from data_loader import prepare_all_experiments
from train_dl_models import (
    MultiChannelTSDataModule,
    prepare_indices,
    DL_MODELS
)

def main():
    """Test DL model training"""
    print("Testing DL model training functionality...\n")
    
    # Disable progress bar to reduce output noise
    pl.trainer.trainer.Trainer.progress_bar_callback = None
    
    # Try to load just one experiment to save time
    experiment_names = ['remote_ppg_contact_resp']
    experiment_data = prepare_all_experiments(experiment_names)
    
    if not experiment_data or not experiment_names[0] in experiment_data:
        print("❌ Error: Failed to load experiment data.")
        return False
    
    # Get the first experiment
    exp_name = experiment_names[0]
    exp_data = experiment_data[exp_name]
    
    # Limit to a small subset for quick testing
    max_samples = 10  # Use just a few samples for quick testing
    
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
    
    try:
        print("Preparing data module...")
        # Prepare indices for data module
        X_all, y_all, train_indices, val_indices, test_indices = prepare_indices(simplified_exp_data)
        
        # Create data module with small batch size for testing
        data_module = MultiChannelTSDataModule(
            X_all, y_all, train_indices, val_indices, test_indices, batch_size=4
        )
        
        # Set up data module to get number of channels
        data_module.setup()
        
        print(f"✅ Data module created successfully with input shape: {data_module.X_train.shape}")
        
        # Get number of channels from the data
        n_channels = data_module.X_train.shape[1]  # [batch, channels, time_steps]
        
        print("\nCreating model...")
        # Create a simplified CNN model for testing
        cnn_config = DL_MODELS['cnn']
        
        # Update in_channels parameter
        model_params = cnn_config['params'].copy()
        model_params['in_channels'] = n_channels
        
        # Create model
        model = cnn_config['class'](**model_params)
        
        print("✅ Model created successfully.")
        
        print("\nTesting forward pass...")
        # Test forward pass with a sample from the training data
        with torch.no_grad():
            sample_input = data_module.X_train[:2]  # Take first 2 samples
            output = model(sample_input)
            
            if output.shape != torch.Size([2, 2]):  # 2 samples, 2 classes
                print(f"❌ Error: Unexpected output shape: {output.shape}, expected [2, 2]")
                return False
                
        print(f"✅ Forward pass successful. Output shape: {output.shape}")
        
        print("\nTesting model training (1 epoch)...")
        # Create a very simplified trainer for testing
        trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False
        )
        
        # Train for 1 epoch
        trainer.fit(model, data_module)
        
        print("\nTesting model evaluation...")
        # Test the model
        test_results = trainer.test(model, data_module)
        
        if not test_results:
            print("❌ Error: No test results returned.")
            return False
            
        print(f"✅ Model evaluated successfully. Test accuracy: {test_results[0]['test_acc']:.4f}")
        
        print("\n✅ DL model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during DL testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 