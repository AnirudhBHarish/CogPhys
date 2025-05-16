"""
Machine Learning models training script for cognitive load classification
Trains and evaluates ML models on the different experiment configurations:
1. Remote PPG + Contact Resp
2. Remote PPG + Remote Resp
3. Remote PPG + Remote Resp + Blink Markers

Supported models:
- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import heartpy as hp
from scipy.signal import find_peaks
from scipy.fft import fft

# Import custom modules
from data_loader import prepare_all_experiments
from utils import EXPERIMENT_CONFIGS

# ML models configurations
ML_MODELS = {
    'rf': {
        'name': 'Random Forest',
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 2025,
            'class_weight': 'balanced'
        }
    },
    'gb': {
        'name': 'Gradient Boosting',
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'random_state': 2025
        }
    },
    'svm': {
        'name': 'Support Vector Machine',
        'class': SVC,
        'params': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 2025
        }
    },
    'lr': {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'params': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'random_state': 2025
        }
    }
}

def extract_time_domain_features(signal):
    """
    Extract time domain features from a signal
    
    Args:
        signal: 1D numpy array
        
    Returns:
        Dictionary of time domain features
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = np.ptp(signal)
    features['median'] = np.median(signal)
    features['skewness'] = 0 if features['std'] == 0 else np.mean((signal - features['mean'])**3) / (features['std']**3)
    features['kurtosis'] = 0 if features['std'] == 0 else np.mean((signal - features['mean'])**4) / (features['std']**4)
    
    # Percentiles
    features['p25'] = np.percentile(signal, 25)
    features['p75'] = np.percentile(signal, 75)
    features['iqr'] = features['p75'] - features['p25']
    
    # Peak-based features
    peaks, _ = find_peaks(signal, height=features['mean'], distance=10)
    if len(peaks) > 1:
        features['peak_count'] = len(peaks)
        features['peak_mean'] = np.mean(signal[peaks])
        features['peak_std'] = np.std(signal[peaks])
        peak_intervals = np.diff(peaks)
        features['peak_interval_mean'] = np.mean(peak_intervals)
        features['peak_interval_std'] = np.std(peak_intervals) if len(peak_intervals) > 1 else 0
    else:
        features['peak_count'] = 0
        features['peak_mean'] = 0
        features['peak_std'] = 0
        features['peak_interval_mean'] = 0
        features['peak_interval_std'] = 0
    
    # Signal energy and power
    features['energy'] = np.sum(signal**2)
    features['power'] = features['energy'] / len(signal)
    
    return features

def extract_frequency_domain_features(signal, sampling_rate=30.0):
    """
    Extract frequency domain features from a signal
    
    Args:
        signal: 1D numpy array
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of frequency domain features
    """
    features = {}
    
    # Compute FFT
    n = len(signal)
    signal_fft = fft(signal)
    # Get the magnitude spectrum
    magnitude = np.abs(signal_fft[:n//2])
    # Frequency values
    freq = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
    
    # Basic features
    try:
        features['fft_mean'] = np.mean(magnitude)
        features['fft_std'] = np.std(magnitude)
        features['fft_max'] = np.max(magnitude)
        features['fft_sum'] = np.sum(magnitude)
        
        # Energy in different frequency bands
        # VLF: Very low frequency (0.0033-0.04 Hz)
        vlf_mask = (freq >= 0.0033) & (freq < 0.04)
        features['vlf_energy'] = np.sum(magnitude[vlf_mask]**2) if np.any(vlf_mask) else 0
        
        # LF: Low frequency (0.04-0.15 Hz)
        lf_mask = (freq >= 0.04) & (freq < 0.15)
        features['lf_energy'] = np.sum(magnitude[lf_mask]**2) if np.any(lf_mask) else 0
        
        # HF: High frequency (0.15-0.4 Hz)
        hf_mask = (freq >= 0.15) & (freq < 0.4)
        features['hf_energy'] = np.sum(magnitude[hf_mask]**2) if np.any(hf_mask) else 0
        
        # Respiratory band (0.15-0.4 Hz)
        resp_mask = (freq >= 0.15) & (freq < 0.4)
        features['resp_energy'] = np.sum(magnitude[resp_mask]**2) if np.any(resp_mask) else 0
        
        # LF/HF ratio (commonly used in HRV analysis)
        features['lf_hf_ratio'] = features['lf_energy'] / features['hf_energy'] if features['hf_energy'] > 0 else 0
        
        # Dominant frequency
        if len(magnitude) > 0:
            idx_max = np.argmax(magnitude)
            features['dominant_freq'] = freq[idx_max]
            features['dominant_power'] = magnitude[idx_max]
        else:
            features['dominant_freq'] = 0
            features['dominant_power'] = 0
    
    except Exception as e:
        print(f"Error extracting frequency domain features: {e}")
        # Set default values for all frequency features
        for feature_name in ['fft_mean', 'fft_std', 'fft_max', 'fft_sum', 
                            'vlf_energy', 'lf_energy', 'hf_energy', 'resp_energy', 
                            'lf_hf_ratio', 'dominant_freq', 'dominant_power']:
            features[feature_name] = 0
    
    return features

def extract_features_from_signals(signals_array):
    """
    Extract features from multiple signals
    
    Args:
        signals_array: Array of signals with shape (n_samples, n_signals, signal_length)
        
    Returns:
        Array of feature vectors
    """
    all_features = []
    
    for sample_idx in range(signals_array.shape[0]):
        sample_features = []
        
        # Process each signal in the sample
        for signal_idx in range(signals_array.shape[1]):
            signal = signals_array[sample_idx, signal_idx]
            
            # Extract time domain features
            time_features = extract_time_domain_features(signal)
            # Extract frequency domain features
            freq_features = extract_frequency_domain_features(signal)
            
            # Combine all features
            signal_features = {}
            # Add signal type indicator to feature name
            signal_type = {0: 'ppg', 1: 'resp', 2: 'blink'}
            prefix = signal_type.get(signal_idx, f'sig{signal_idx}')
            
            for name, value in time_features.items():
                signal_features[f'{prefix}_{name}'] = value
            
            for name, value in freq_features.items():
                signal_features[f'{prefix}_{name}'] = value
            
            # Convert dictionary to flat list
            sample_features.extend(list(signal_features.values()))
        
        all_features.append(sample_features)
    
    return np.array(all_features)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Low', 'High'], output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"True Negatives: {conf_matrix[0][0]}, False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}, True Positives: {conf_matrix[1][1]}")
    
    # Calculate and print additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (Recall/True Positive Rate): {recall:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'report': report,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def run_ml_experiment(experiment_data, model_configs=None):
    """
    Run ML experiment on the given data
    
    Args:
        experiment_data: Data for the experiment (from data_loader)
        model_configs: Dictionary of model configurations to use (default: all)
        
    Returns:
        Dictionary of results for each model
    """
    if model_configs is None:
        model_configs = ML_MODELS
    
    # Unpack data
    X_train, y_train, _ = experiment_data['train']
    X_val, y_val, _ = experiment_data['val']
    X_test, y_test, _ = experiment_data['test']
    
    # Extract features from signals
    print("Extracting features from signals...")
    X_train_features = extract_features_from_signals(X_train)
    X_val_features = extract_features_from_signals(X_val)
    X_test_features = extract_features_from_signals(X_test)
    
    print(f"Extracted features: {X_train_features.shape[1]} features per sample")
    
    # Print class distribution for each set
    print("\nClass Distribution:")
    print(f"Training set: {np.bincount(y_train)}, Class balance: {np.bincount(y_train)[0]/len(y_train):.2f}:{np.bincount(y_train)[1]/len(y_train):.2f}")
    print(f"Validation set: {np.bincount(y_val)}, Class balance: {np.bincount(y_val)[0]/len(y_val):.2f}:{np.bincount(y_val)[1]/len(y_val):.2f}")
    print(f"Test set: {np.bincount(y_test)}, Class balance: {np.bincount(y_test)[0]/len(y_test):.2f}:{np.bincount(y_test)[1]/len(y_test):.2f}")
    
    results = {}
    
    # Train and evaluate each model
    for model_key, model_config in model_configs.items():
        model_name = model_config['name']
        print(f"\nTraining {model_name}...")
        
        # Create and fit the model
        start_time = time.time()
        
        # Create pipeline with standardization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_config['class'](**model_config['params']))
        ])
        
        # Fit on training data
        pipeline.fit(X_train_features, y_train)
        
        # Measure training time
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on training data
        print("\n--- Training Set Evaluation ---")
        train_results = evaluate_model(pipeline, X_train_features, y_train)
        print(f"Training accuracy: {train_results['accuracy']:.4f}, F1: {train_results['f1']:.4f}")
        
        # Evaluate on validation data
        print("\n--- Validation Set Evaluation ---")
        val_results = evaluate_model(pipeline, X_val_features, y_val)
        print(f"Validation accuracy: {val_results['accuracy']:.4f}, F1: {val_results['f1']:.4f}")
        
        # Evaluate on test data
        print("\n--- Test Set Evaluation ---")
        test_results = evaluate_model(pipeline, X_test_features, y_test)
        print(f"Test accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
        
        # Store results
        results[model_key] = {
            'model': pipeline,
            'train_time': train_time,
            'train_results': train_results,
            'val_results': val_results,
            'test_results': test_results,
            'features': {
                'n_features': X_train_features.shape[1]
            }
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
    print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*60)
    
    # Results for each model
    for model_key, model_results in results.items():
        model_name = ML_MODELS[model_key]['name']
        test_results = model_results['test_results']
        print(f"{model_name:<20} {test_results['accuracy']:.4f}    {test_results['f1']:.4f}    "
              f"{test_results['precision']:.4f}    {test_results['recall']:.4f}")
    
    print("-"*60)
    print(f"Class distribution: {np.bincount(results[list(results.keys())[0]]['test_results']['y_pred'])}")
    print("="*80 + "\n")

def main():
    """Main entry point for the script"""
    print("\n--- Running ML Classification Pipeline with Extracted Features ---")
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
        
        # Run ML experiment
        experiment_results = run_ml_experiment(experiment_data)
        
        # Print summary
        print_experiment_summary(experiment_name, experiment_results)
        
        # Store results
        all_results[experiment_name] = experiment_results
    
    # Save all results to file
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"ml_results_{timestamp}.pkl")
    
    try:
        # Remove actual model objects before saving to reduce file size
        for exp_results in all_results.values():
            for model_results in exp_results.values():
                if 'model' in model_results:
                    del model_results['model']
        
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 