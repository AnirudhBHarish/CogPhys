"""
Data loader module for physiological signal processing
Handles loading and preprocessing of:
- rPPG waveforms (from rppg_waveforms.pkl)
- Respiration waveforms (from resp_fusion_waveforms.pkl)
- Blink markers (from eos_dict.pkl)
- Contact PPG and respiration (from GT.pkl)
"""

import numpy as np
import heartpy as hp

# Import utilities from utils.py
from utils import (
    load_pickle, create_median_split_labels, apply_smoothing_filter,
    resample_signal, normalize_signal, extract_participant_info,
    filter_by_participants, RPPG_WAVEFORMS_FILE, RESP_FUSION_WAVEFORMS_FILE,
    CV_RPPG_WAVEFORMS_FILE, CV_RESP_WAVEFORMS_FILE,
    BLINK_MARKERS_FILE, FOLDS_FILE, NASA_TLX_FILE, EXPERIMENT_CONFIGS,
    RPPG_SAMPLING_RATE, RESP_SAMPLING_RATE, BLINK_SAMPLING_RATE,
    SIGNAL_DURATION, RPPG_SIGNAL_LENGTH, RESP_SIGNAL_LENGTH, BLINK_SIGNAL_LENGTH,
    filter_signals_with_missing_data
)

# Constants for contact data
GT_DATA_FILE = './data/GT.pkl'
CONTACT_PPG_SAMPLING_RATE = 60.0  # Hz
CONTACT_RESP_SAMPLING_RATE = 61.0  # Hz
CONTACT_PPG_SIGNAL_LENGTH = int(CONTACT_PPG_SAMPLING_RATE * SIGNAL_DURATION)  # 7200 samples
CONTACT_RESP_SIGNAL_LENGTH = int(CONTACT_RESP_SAMPLING_RATE * SIGNAL_DURATION)  # 2160 samples

def load_rppg_waveforms(use_cv_waveforms=True):
    """Load rPPG waveforms from pickle file"""
    try:
        if use_cv_waveforms:
            data = load_pickle(CV_RPPG_WAVEFORMS_FILE)  # Use cross-validation data
        else:
            data = load_pickle(RPPG_WAVEFORMS_FILE)
        print(f"Loaded {len(data)} rPPG waveforms")
        return data
    except Exception as e:
        print(f"Error loading rPPG waveforms: {e}")
        return None

def load_resp_waveforms(use_cv_waveforms=True):
    """Load respiration waveforms from pickle file"""
    try:
        if use_cv_waveforms:
            data = load_pickle(CV_RESP_WAVEFORMS_FILE)  # Use cross-validation data
        else:
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

def load_contact_data():
    """Load contact PPG and respiration data from GT.pkl"""
    try:
        data = load_pickle(GT_DATA_FILE)
        print(f"Loaded {len(data)} contact data samples from GT.pkl")
        return data
    except Exception as e:
        print(f"Error loading contact data: {e}")
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
    rppg_data = load_rppg_waveforms() if experiment_config.get('use_remote_ppg', False) else None
    resp_data = load_resp_waveforms() if experiment_config.get('use_remote_resp', False) else None
    blink_data = load_blink_markers() if experiment_config.get('use_blink', False) else None
    contact_data = load_contact_data() if experiment_config.get('use_contact_ppg', False) or experiment_config.get('use_contact_resp', False) else None
    
    # Check if required data was loaded
    if experiment_config.get('use_remote_ppg', False) and rppg_data is None:
        print("Error: rPPG data required but not loaded")
        return None
    else:
        if rppg_data is not None:
            print(f"rPPG data loaded with {len(rppg_data)} samples")
    
    if experiment_config.get('use_remote_resp', False) and resp_data is None:
        print("Error: Respiration data required but not loaded")
        return None
    else:
        if resp_data is not None:
            print(f"Respiration data loaded with {len(resp_data)} samples")
    
    if experiment_config.get('use_blink', False) and blink_data is None:
        print("Error: Blink marker data required but not loaded")
        return None
    else:
        if blink_data is not None:
            print(f"Blink marker data loaded with {len(blink_data)} samples")
        
    if (experiment_config.get('use_contact_ppg', False) or experiment_config.get('use_contact_resp', False)) and contact_data is None:
        print("Error: Contact data required but not loaded")
        return None
    else:
        if contact_data is not None:
            print(f"Contact data loaded with {len(contact_data)} samples")
    
    # Initialize data containers
    X_train, y_train, keys_train = [], [], []
    X_val, y_val, keys_val = [], [], []
    X_test, y_test, keys_test = [], [], []
    
    # Determine how many channels we will have based on the experiment config
    n_channels = 0
    if experiment_config.get('use_remote_ppg', False) or experiment_config.get('use_contact_ppg', False):
        n_channels += 1  # PPG channel
    if experiment_config.get('use_remote_resp', False) or experiment_config.get('use_contact_resp', False):
        n_channels += 1  # Respiration channel
    if experiment_config.get('use_blink', False):
        n_channels += 1  # Blink channel
    
    # Determine which data source to use as the base for iteration
    if experiment_config.get('use_contact_ppg', False) or experiment_config.get('use_contact_resp', False):
        base_data = contact_data
    elif experiment_config.get('use_remote_ppg', False):
        base_data = rppg_data
    elif experiment_config.get('use_remote_resp', False):
        base_data = resp_data
    elif experiment_config.get('use_blink', False):
        base_data = blink_data
    else:
        print("Error: No valid data source specified in configuration")
        return None
    
    # Process all keys in the base data
    for key in base_data:
        participant_id, task_name = extract_participant_info(key)
        
        # Skip if no label for this combination
        label = label_map.get((participant_id, task_name))
        if label is None:
            continue
        
        try:
            # Create a numpy array with the right shape from the start
            combined_signal = np.zeros((n_channels, RPPG_SIGNAL_LENGTH))
            
            # Fill the array with our signals
            channel_idx = 0
            
            # Handle PPG signal (either remote or contact)
            if experiment_config.get('use_remote_ppg', False):
                # Extract 'pred' rPPG signal (the predicted remote PPG)
                if key not in rppg_data:
                    continue
                
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
                
                # Normalize signal
                rppg_signal = normalize_signal(rppg_signal)
                combined_signal[channel_idx] = rppg_signal
                channel_idx += 1
            
            elif experiment_config.get('use_contact_ppg', False):
                # Extract contact PPG signal
                if 'ppg' not in contact_data[key]:
                    continue
                
                contact_ppg = contact_data[key]['ppg']
                if contact_ppg is None or len(contact_ppg) < 100:
                    continue
                
                # Apply smoothing
                contact_ppg = apply_smoothing_filter(contact_ppg)
                
                # Resample to match the target length
                if len(contact_ppg) != RPPG_SIGNAL_LENGTH:
                    contact_ppg = resample_signal(contact_ppg, CONTACT_PPG_SIGNAL_LENGTH, RPPG_SIGNAL_LENGTH)
                    
                if contact_ppg is None:
                    continue
                
                # Normalize signal
                contact_ppg = normalize_signal(contact_ppg)
                combined_signal[channel_idx] = contact_ppg
                channel_idx += 1
            
            # Handle respiration signal (either remote or contact)
            if experiment_config.get('use_remote_resp', False):
                # Extract remote respiration signal
                resp_signal = None
                if key in resp_data:
                    resp_signal = resp_data[key].get('pred')
                    if resp_signal is not None and len(resp_signal) >= 100:
                        # Resample to match rPPG rate if needed
                        if len(resp_signal) != RPPG_SIGNAL_LENGTH:
                            resp_signal = resample_signal(resp_signal, RESP_SIGNAL_LENGTH, RPPG_SIGNAL_LENGTH)
                
                # If we couldn't get a valid resp signal, create a zero array
                if resp_signal is None:
                    print(f"Warning: No valid remote respiration signal for {key}, using zeros")
                    resp_signal = np.zeros(RPPG_SIGNAL_LENGTH)
                
                # Normalize signal
                resp_signal = normalize_signal(resp_signal)
                combined_signal[channel_idx] = resp_signal
                channel_idx += 1
                
            elif experiment_config.get('use_contact_resp', False):
                # Extract contact respiration signal
                resp_signal = None
                if 'respiration' in contact_data[key]:
                    resp_signal = contact_data[key]['respiration']
                    if resp_signal is not None and len(resp_signal) >= 100:
                        # Resample to match rPPG rate if needed
                        if len(resp_signal) != RPPG_SIGNAL_LENGTH:
                            resp_signal = resample_signal(resp_signal, CONTACT_RESP_SIGNAL_LENGTH, RPPG_SIGNAL_LENGTH)
                
                # If we couldn't get a valid resp signal, create a zero array
                if resp_signal is None:
                    print(f"Warning: No valid contact respiration signal for {key}, using zeros")
                    resp_signal = np.zeros(RPPG_SIGNAL_LENGTH)
                
                # Normalize signal
                resp_signal = normalize_signal(resp_signal)
                combined_signal[channel_idx] = resp_signal
                channel_idx += 1
            
            # Handle blink signals
            if experiment_config.get('use_blink', False):
                # Extract blink signal
                blink_signal = None
                if key in blink_data:
                    # Use average of left and right eye
                    blink_signal = blink_data[key].get('eo_signal_avg')
                    if blink_signal is None and 'eo_signal_left' in blink_data[key]:
                        blink_signal = blink_data[key]['eo_signal_left']
                
                # If we couldn't get a valid blink signal, create a zero array
                if blink_signal is None:
                    print(f"Warning: No valid blink signal for {key}, using zeros")
                    blink_signal = np.zeros(RPPG_SIGNAL_LENGTH)
                
                # Normalize signal
                blink_signal = normalize_signal(blink_signal)
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

def prepare_all_experiments(experiment_names=None, split_idx=0):
    """
    Prepare data for all or selected experiments
    
    Args:
        experiment_names: List of experiment names to prepare, or None for all
        
    Returns:
        Dictionary with prepared data for each experiment
    """
    # Create labels
    label_map = create_median_split_labels()
    print(label_map)
    
    # Load predefined folds
    folds = load_pickle(FOLDS_FILE)
    fold_split_dict = folds[split_idx]  # Default to fold 0
        
    # fold_split_dict = load_pickle("data/CogPhys_extra_folds.pkl")[0] # for main paper results, fix the split


    print(f"Using fold {split_idx} with:")
    print(f"  Train: {len(fold_split_dict['train'])} participants")
    print(f"  Validation: {len(fold_split_dict['valid'])} participants")
    print(f"  Test: {len(fold_split_dict['test'])} participants")
    
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
        experiment_data[experiment_name] = prepare_data_for_experiment(config, label_map, fold_split_dict)
    
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