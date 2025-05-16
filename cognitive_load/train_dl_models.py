"""
Deep Learning models training script for cognitive load classification
Trains and evaluates DL models on the different experiment configurations:
1. Remote PPG + Contact Resp
2. Remote PPG + Remote Resp
3. Remote PPG + Remote Resp + Blink Markers

Supported models:
- 1D CNN
- LSTM
- ResNet1D
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os
from datetime import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Import from parent directory
import sys
sys.path.append('..')
from models.trainers import CNN1D, LSTMModel, ResNet1DModel

# Import custom modules
from data_loader import prepare_all_experiments
from utils import EXPERIMENT_CONFIGS

# DL model configurations
DL_MODELS = {
    'cnn': {
        'name': '1D CNN',
        'class': CNN1D,
        'params': {
            'in_channels': 1,  # Will be updated based on input data
            'num_classes': 2,
            'learning_rate': 0.0005
        }
    },
    'lstm': {
        'name': 'LSTM',
        'class': LSTMModel,
        'params': {
            'in_channels': 1,  # Will be updated based on input data
            'num_classes': 2,
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.0005
        }
    },
    'resnet': {
        'name': 'ResNet1D',
        'class': ResNet1DModel,
        'params': {
            'in_channels': 1,  # Will be updated based on input data
            'num_classes': 2,
            'learning_rate': 0.0005
        }
    }
}

# Training parameters
MAX_EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 10  # For early stopping

# Flag to control checkpoint saving and loading
SAVE_CHECKPOINTS = False  # Set to False to skip checkpoint saving/loading

class MultiChannelTSDataModule(pl.LightningDataModule):
    """
    Data module for multi-channel time series data
    Extends the TSDataModule from train_dl_baselines.py to handle multiple channels
    """
    def __init__(self, X, y, train_indices, val_indices, test_indices, batch_size=32):
        super().__init__()
        self.X = X
        self.y = y
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        # Extract data for each split
        X_train = self.X[self.train_indices]
        X_val = self.X[self.val_indices]
        X_test = self.X[self.test_indices]
        
        # Determine number of channels (signals) in the data
        if len(X_train.shape) == 3:  # [samples, channels, time_steps]
            self.n_channels = X_train.shape[1]
        else:
            self.n_channels = 1
        
        # Reshape data if needed
        if len(X_train.shape) == 3:
            # Already in [samples, channels, time_steps] format
            # Just need to ensure channels are first dimension for PyTorch
            # Transpose from [samples, channels, time_steps] to [samples, time_steps, channels]
            X_train = np.transpose(X_train, (0, 2, 1))
            X_val = np.transpose(X_val, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
        
        # Normalize each channel individually using RobustScaler
        # which is more robust to outliers in physiological signals
        for i in range(len(X_train)):
            for c in range(X_train.shape[2]):  # iterate over channels
                if np.std(X_train[i, :, c]) > 0:  # Avoid division by zero
                    median = np.median(X_train[i, :, c])
                    q75, q25 = np.percentile(X_train[i, :, c], [75, 25])
                    iqr = q75 - q25 or 1  # Use 1 if IQR is 0
                    X_train[i, :, c] = (X_train[i, :, c] - median) / iqr
        
        for i in range(len(X_val)):
            for c in range(X_val.shape[2]):
                if np.std(X_val[i, :, c]) > 0:
                    median = np.median(X_val[i, :, c])
                    q75, q25 = np.percentile(X_val[i, :, c], [75, 25])
                    iqr = q75 - q25 or 1
                    X_val[i, :, c] = (X_val[i, :, c] - median) / iqr
                
        for i in range(len(X_test)):
            for c in range(X_test.shape[2]):
                if np.std(X_test[i, :, c]) > 0:
                    median = np.median(X_test[i, :, c])
                    q75, q25 = np.percentile(X_test[i, :, c], [75, 25])
                    iqr = q75 - q25 or 1
                    X_test[i, :, c] = (X_test[i, :, c] - median) / iqr
        
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y[self.train_indices], dtype=torch.long)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y[self.val_indices], dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y[self.test_indices], dtype=torch.long)
        
        # Reshape for models - [batch, channels, time_steps]
        # Need to transpose from [batch, time_steps, channels] to [batch, channels, time_steps]
        self.X_train = self.X_train.permute(0, 2, 1)
        self.X_val = self.X_val.permute(0, 2, 1)
        self.X_test = self.X_test.permute(0, 2, 1)
        
        # Print shapes to verify
        print(f"Training data shape: {self.X_train.shape}, Labels: {self.y_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}, Labels: {self.y_val.shape}")
        print(f"Test data shape: {self.X_test.shape}, Labels: {self.y_test.shape}")
        
        # Print class distribution
        print(f"Train class distribution: {torch.bincount(self.y_train)}")
        print(f"Train class balance: {torch.bincount(self.y_train)[0]/len(self.y_train):.2f}:{torch.bincount(self.y_train)[1]/len(self.y_train):.2f}")
        print(f"Validation class distribution: {torch.bincount(self.y_val)}")
        print(f"Validation class balance: {torch.bincount(self.y_val)[0]/len(self.y_val):.2f}:{torch.bincount(self.y_val)[1]/len(self.y_val):.2f}")
        print(f"Test class distribution: {torch.bincount(self.y_test)}")
        print(f"Test class balance: {torch.bincount(self.y_test)[0]/len(self.y_test):.2f}:{torch.bincount(self.y_test)[1]/len(self.y_test):.2f}")
        
    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        val_dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4)

def prepare_indices(experiment_data):
    """
    Prepare indices for train/val/test splits
    
    Args:
        experiment_data: Dictionary with data from data_loader
        
    Returns:
        Tuple of indices for train, val, test
    """
    X_train, y_train, _ = experiment_data['train']
    X_val, y_val, _ = experiment_data['val']
    X_test, y_test, _ = experiment_data['test']
    
    # Create indices arrays
    train_indices = np.arange(len(X_train))
    val_indices = np.arange(len(X_val)) + len(X_train)
    test_indices = np.arange(len(X_test)) + len(X_train) + len(X_val)
    
    # Combine all data
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    return X_all, y_all, train_indices, val_indices, test_indices

def evaluate_model(trainer, model, data_module):
    """
    Evaluate a trained model
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Trained model
        data_module: Data module with test data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get test predictions
    results = trainer.test(model, data_module)
    test_acc = results[0]['test_acc']
    
    # Get detailed predictions
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        
        for batch in data_module.test_dataloader():
            x, y = batch
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    # Calculate metrics
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"True Negatives: {conf_matrix[0][0]}, False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}, True Positives: {conf_matrix[1][1]}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (Recall/True Positive Rate): {recall:.4f}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Low', 'High'], output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'report': report,
        'y_pred': y_pred,
        'y_true': y_true,
        'confusion_matrix': conf_matrix
    }

def get_predictions(model, data_loader):
    """Get predictions from a model on a given data loader"""
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    return np.array(y_pred), np.array(y_true)

def run_dl_experiment(experiment_name, experiment_data, model_configs=None):
    """
    Run DL experiment on the given data
    
    Args:
        experiment_name: Name of the experiment
        experiment_data: Data for the experiment (from data_loader)
        model_configs: Dictionary of model configurations to use (default: all)
        
    Returns:
        Dictionary of results for each model
    """
    if model_configs is None:
        model_configs = DL_MODELS
    
    # Prepare indices for data module
    X_all, y_all, train_indices, val_indices, test_indices = prepare_indices(experiment_data)
    
    # Create data module
    data_module = MultiChannelTSDataModule(
        X_all, y_all, train_indices, val_indices, test_indices, batch_size=BATCH_SIZE
    )
    
    # Set up data module to get number of channels
    data_module.setup()
    n_channels = data_module.X_train.shape[1]  # [batch, channels, time_steps]
    
    results = {}
    
    # Create directory for model checkpoints if needed
    models_dir = f"./models/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if SAVE_CHECKPOINTS:
        os.makedirs(models_dir, exist_ok=True)
    
    # Train and evaluate each model
    for model_key, model_config in model_configs.items():
        model_name = model_config['name']
        print(f"\nTraining {model_name} for {experiment_name}...")
        
        # Update in_channels parameter
        model_params = model_config['params'].copy()
        model_params['in_channels'] = n_channels
        
        # Create model
        model = model_config['class'](**model_params)
        
        # Setup callbacks
        callbacks = []
        checkpoint_callback = None
        best_model_path = None
        
        if SAVE_CHECKPOINTS:
            # Create model-specific directory
            model_dir = os.path.join(models_dir, model_key)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_dir,
                filename=f"{model_key}-{{epoch:02d}}-{{val_acc:.4f}}",
                monitor='val_acc',
                mode='max',
                save_top_k=1
            )
            callbacks.append(checkpoint_callback)
        
        # Always add early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            mode='min'
        )
        callbacks.append(early_stop_callback)
        
        # Setup logger (or disable if not saving checkpoints)
        logger = False
        if SAVE_CHECKPOINTS:
            logger = TensorBoardLogger(
                save_dir=os.path.join(models_dir, model_key),
                name=model_key
            )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            enable_checkpointing=SAVE_CHECKPOINTS
        )
        
        # Train model
        trainer.fit(model, data_module)
        
        # Decide which model to use for evaluation
        best_model = model  # Default to the final model
        
        # Try to load best checkpoint if available and requested
        if SAVE_CHECKPOINTS and checkpoint_callback is not None:
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                try:
                    best_model = model_config['class'].load_from_checkpoint(best_model_path)
                    print(f"Loaded best model from {best_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint {best_model_path}: {e}")
                    print("Using final model instead")
                    best_model = model
            else:
                print("No valid checkpoint found. Using final model for evaluation.")
        
        # Evaluate on training data
        print("\n--- Training Set Evaluation ---")
        y_train_pred, y_train_true = get_predictions(best_model, data_module.train_dataloader())
        train_acc = accuracy_score(y_train_true, y_train_pred)
        train_f1 = f1_score(y_train_true, y_train_pred)
        train_conf_matrix = confusion_matrix(y_train_true, y_train_pred)
        print(f"Training accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print("Training Confusion Matrix:")
        print(train_conf_matrix)
        
        # Evaluate on validation data
        print("\n--- Validation Set Evaluation ---")
        y_val_pred, y_val_true = get_predictions(best_model, data_module.val_dataloader())
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred)
        val_conf_matrix = confusion_matrix(y_val_true, y_val_pred)
        print(f"Validation accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print("Validation Confusion Matrix:")
        print(val_conf_matrix)
        
        # Evaluate on test data
        print("\n--- Test Set Evaluation ---")
        test_results = evaluate_model(trainer, best_model, data_module)
        print(f"Test accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
        
        # Store results
        results[model_key] = {
            'model_path': best_model_path if SAVE_CHECKPOINTS and best_model_path else None,
            'train_results': {
                'accuracy': train_acc,
                'f1': train_f1,
                'confusion_matrix': train_conf_matrix
            },
            'val_results': {
                'accuracy': val_acc,
                'f1': val_f1,
                'confusion_matrix': val_conf_matrix
            },
            'test_results': test_results,
            'model_params': model_params
        }
    
    return results

def print_experiment_summary(experiment_name, results):
    """
    Print summary of experiment results
    
    Args:
        experiment_name: Name of the experiment
        results: Dictionary of results for each model
    """
    print("\n" + "="*80)
    print(f"SUMMARY FOR EXPERIMENT: {EXPERIMENT_CONFIGS[experiment_name]['description']}")
    print("="*80)
    
    # Table header
    print(f"{'Model':<20} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10} {'Test Prec':<10} {'Test Recall':<10}")
    print("-"*80)
    
    # Results for each model
    for model_key, model_results in results.items():
        model_name = DL_MODELS[model_key]['name']
        train_results = model_results['train_results']
        val_results = model_results['val_results']
        test_results = model_results['test_results']
        
        print(f"{model_name:<20} {train_results['accuracy']:.4f}    {val_results['accuracy']:.4f}    "
              f"{test_results['accuracy']:.4f}    {test_results['f1']:.4f}    "
              f"{test_results['precision']:.4f}    {test_results['recall']:.4f}")
    
    print("-"*80)
    print("="*80 + "\n")

def main():
    """Main entry point for the script"""
    print("\n--- Running DL Classification Pipeline with Raw Signals ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for all experiments
    all_experiment_data = prepare_all_experiments()
    
    all_results = {}
    
    # Run experiments
    for experiment_name, experiment_data in all_experiment_data.items():
        if experiment_data is None:
            print(f"Skipping experiment {experiment_name} due to missing data")
            continue
        
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT: {EXPERIMENT_CONFIGS[experiment_name]['description']}")
        print("="*80)
        
        # Run DL experiment
        experiment_results = run_dl_experiment(experiment_name, experiment_data)
        
        # Print summary
        print_experiment_summary(experiment_name, experiment_results)
        
        # Store results
        all_results[experiment_name] = experiment_results
    
    # Save all results to file
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"dl_results_{timestamp}.pkl")
    
    try:
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 