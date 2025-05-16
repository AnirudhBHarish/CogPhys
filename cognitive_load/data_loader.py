"""
Data loader module for physiological signal processing
Handles loading and preprocessing of:
- rPPG waveforms (from rppg_waveforms.pkl)
- Respiration waveforms (from resp_fusion_waveforms.pkl)
- Blink markers (from eos_dict.pkl)
"""

import numpy as np
import pandas as pd
import pickle
import os
import heartpy as hp

# Import utilities from utils.py
from utils import (
    load_pickle, create_median_split_labels, apply_smoothing_filter,
    resample_signal, normalize_signal, extract_participant_info,
    filter_by_participants, RPPG_WAVEFORMS_FILE, RESP_FUSION_WAVEFORMS_FILE,
    BLINK_MARKERS_FILE, FOLDS_FILE, NASA_TLX_FILE, EXPERIMENT_CONFIGS,
    RPPG_SAMPLING_RATE, RESP_SAMPLING_RATE, BLINK_SAMPLING_RATE,
    SIGNAL_DURATION, RPPG_SIGNAL_LENGTH, RESP_SIGNAL_LENGTH, BLINK_SIGNAL_LENGTH,
    filter_signals_with_missing_data
)

def load_rppg_waveforms():
    """Load rPPG waveforms from pickle file"""
    try:
        data = load_pickle(RPPG_WAVEFORMS_FILE)
        print(f"Loaded {len(data)} rPPG waveforms")
        return data
    except Exception as e:
        print(f"Error loading rPPG waveforms: {e}")
        return None

def load_resp_waveforms():
    """Load respiration waveforms from pickle file"""
    try:
        data = load_pickle(RESP_FUSION_WAVEFORMS_FILE)
        print(f"Loaded {len(data)} respiration waveforms")
        return data
    except Exception as e:
        print(f"Error loading respiration waveforms: {e}")
        return None

def load_blink_markers():
    """Load blink markers from pickle file"""
    try:
        data = load_pickle(BLINK_MARKERS_FILE)
        print(f"Loaded {len(data)} blink marker sets")
        return data
    except Exception as e:
        print(f"Error loading blink markers: {e}")
        return None

def prepare_data_for_experiment(experiment_config, label_map, fold_info):
    """
    Prepare data for a specific experiment configuration
    
    Args:
        experiment_config: Dictionary with experiment configuration
        label_map: Dictionary mapping (participant_id, task_name) to label
        fold_info: Dictionary with train/val/test participant lists
        
    Returns:
        Dictionary with prepared data for each split
    """
    # Load data based on experiment configuration
    rppg_data = load_rppg_waveforms() if experiment_config['use_remote_ppg'] else None
    resp_data = load_resp_waveforms() if experiment_config['use_remote_resp'] else None
    blink_data = load_blink_markers() if experiment_config['use_blink'] else None
    
    # Check if required data was loaded
    if experiment_config['use_remote_ppg'] and rppg_data is None:
        print("Error: rPPG data required but not loaded")
        return None
    
    if experiment_config['use_remote_resp'] and resp_data is None:
        print("Error: Respiration data required but not loaded")
        return None
    
    if experiment_config['use_blink'] and blink_data is None:
        print("Error: Blink marker data required but not loaded")
        return None
    
    # Initialize data containers
    X_train, y_train, keys_train = [], [], []
    X_val, y_val, keys_val = [], [], []
    X_test, y_test, keys_test = [], [], []
    
    # Determine how many channels we will have based on the experiment config
    n_channels = 1  # Always have RPPG
    if experiment_config['use_remote_resp']:
        n_channels += 1
    if experiment_config['use_blink']:
        n_channels += 1
    
    # Process all keys in rPPG data (used as the base)
    for key in rppg_data:
        participant_id, task_name = extract_participant_info(key)
        
        # Skip if no label for this combination
        label = label_map.get((participant_id, task_name))
        if label is None:
            continue
        
        # Get rPPG signal and apply preprocessing
        try:
            # Extract 'pred' rPPG signal (the predicted remote PPG)
            rppg_signal = rppg_data[key].get('pred')
            if rppg_signal is None or len(rppg_signal) < 100:
                continue
            
            # Apply smoothing and ensure correct length
            rppg_signal = apply_smoothing_filter(rppg_signal)
            if len(rppg_signal) > RPPG_SIGNAL_LENGTH:
                rppg_signal = rppg_signal[:RPPG_SIGNAL_LENGTH]
            elif len(rppg_signal) < RPPG_SIGNAL_LENGTH:
                # Pad with zeros if needed
                pad_length = RPPG_SIGNAL_LENGTH - len(rppg_signal)
                rppg_signal = np.pad(rppg_signal, (0, pad_length), 'constant')
            
            # Get and process respiration signal if needed
            resp_signal = None
            if experiment_config['use_remote_resp']:
                if key in resp_data:
                    resp_signal = resp_data[key].get('pred')
                    if resp_signal is not None and len(resp_signal) >= 100:
                        # Resample to match rPPG rate if needed
                        if len(resp_signal) != RPPG_SIGNAL_LENGTH:
                            resp_signal = resample_signal(resp_signal, 
                                                      RESP_SIGNAL_LENGTH, 
                                                      RPPG_SIGNAL_LENGTH)
                
                # If we couldn't get a valid resp signal, create a zero array
                if resp_signal is None:
                    print(f"Warning: No valid respiration signal for {key}, using zeros")
                    resp_signal = np.zeros(RPPG_SIGNAL_LENGTH)
            
            # Get and process blink signal if needed
            blink_signal = None
            if experiment_config['use_blink']:
                if key in blink_data:
                    # Use average of left and right eye
                    blink_signal = blink_data[key].get('eo_signal_avg')
                    if blink_signal is None and 'eo_signal_left' in blink_data[key]:
                        blink_signal = blink_data[key]['eo_signal_left']
                
                # If we couldn't get a valid blink signal, create a zero array
                if blink_signal is None:
                    print(f"Warning: No valid blink signal for {key}, using zeros")
                    blink_signal = np.zeros(RPPG_SIGNAL_LENGTH)
            
            # Normalize all signals
            rppg_signal = normalize_signal(rppg_signal)
            if resp_signal is not None:
                resp_signal = normalize_signal(resp_signal)
            if blink_signal is not None:
                blink_signal = normalize_signal(blink_signal)
            
            # Create a numpy array with the right shape from the start
            combined_signal = np.zeros((n_channels, RPPG_SIGNAL_LENGTH))
            
            # Fill the array with our signals
            channel_idx = 0
            combined_signal[channel_idx] = rppg_signal
            channel_idx += 1
            
            if experiment_config['use_remote_resp']:
                combined_signal[channel_idx] = resp_signal
                channel_idx += 1
                
            if experiment_config['use_blink']:
                combined_signal[channel_idx] = blink_signal
            
            # Determine which split this belongs to
            if participant_id in fold_info['train']:
                X_train.append(combined_signal)
                y_train.append(label)
                keys_train.append(key)
            elif participant_id in fold_info['valid']:
                X_val.append(combined_signal)
                y_val.append(label)
                keys_val.append(key)
            elif participant_id in fold_info['test']:
                X_test.append(combined_signal)
                y_test.append(label)
                keys_test.append(key)
        
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue
    
    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Filter out samples with missing signals
    print("\nFiltering out samples with missing signals...")
    X_train, y_train, keys_train = filter_signals_with_missing_data(X_train, y_train, keys_train, experiment_config)
    X_val, y_val, keys_val = filter_signals_with_missing_data(X_val, y_val, keys_val, experiment_config)
    X_test, y_test, keys_test = filter_signals_with_missing_data(X_test, y_test, keys_test, experiment_config)
    
    # Print statistics
    print(f"\nData prepared for: {experiment_config['description']}")
    print(f"Training: {len(X_train)} samples, Class distribution: {np.bincount(y_train) if len(y_train) > 0 else 'N/A'}")
    print(f"Validation: {len(X_val)} samples, Class distribution: {np.bincount(y_val) if len(y_val) > 0 else 'N/A'}")
    print(f"Testing: {len(X_test)} samples, Class distribution: {np.bincount(y_test) if len(y_test) > 0 else 'N/A'}")
    
    return {
        'train': (X_train, y_train, keys_train),
        'val': (X_val, y_val, keys_val),
        'test': (X_test, y_test, keys_test),
        'config': experiment_config
    }

def prepare_all_experiments(experiment_names=None):
    """
    Prepare data for all or selected experiments
    
    Args:
        experiment_names: List of experiment names to prepare, or None for all
        
    Returns:
        Dictionary with prepared data for each experiment
    """
    # Create labels
    label_map = create_median_split_labels()
    
    # Load predefined folds
    folds = load_pickle(FOLDS_FILE)
    fold_0 = folds[0]  # Use fold 0 as specified
    
    print(f"Using fold 0 with:")
    print(f"  Train: {len(fold_0['train'])} participants")
    print(f"  Validation: {len(fold_0['valid'])} participants")
    print(f"  Test: {len(fold_0['test'])} participants")
    
    # Determine which experiments to run
    if experiment_names is None:
        experiments_to_run = EXPERIMENT_CONFIGS.keys()
    else:
        experiments_to_run = [name for name in experiment_names if name in EXPERIMENT_CONFIGS]
    
    # Prepare data for each experiment
    experiment_data = {}
    for experiment_name in experiments_to_run:
        config = EXPERIMENT_CONFIGS[experiment_name]
        print(f"\nPreparing data for experiment: {config['description']}")
        experiment_data[experiment_name] = prepare_data_for_experiment(config, label_map, fold_0)
    
    return experiment_data

# Simple test function
if __name__ == "__main__":
    print("Testing data loader...")
    experiment_data = prepare_all_experiments()
    for name, data in experiment_data.items():
        if data:
            print(f"\nExperiment: {EXPERIMENT_CONFIGS[name]['description']}")
            print(f"Features shape: {data['train'][0].shape}")
            print(f"Labels shape: {data['train'][1].shape}") 