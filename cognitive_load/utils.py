"""
Utility functions and constants for cognitive load classification
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy import signal
from scipy.signal import resample

# Constants for file paths
BASE_PATH = '/rdf/user/pg34/code/GM/lab_iccv/'
NASA_TLX_FILE = os.path.join(BASE_PATH, 'data/GM_NASA_TLX.csv')
FOLDS_FILE = os.path.join(BASE_PATH, 'data/CogPhys_all_Folds_new.pkl')
RPPG_WAVEFORMS_FILE = os.path.join(BASE_PATH, 'data/rppg_waveforms.pkl')
RESP_FUSION_WAVEFORMS_FILE = os.path.join(BASE_PATH, 'data/resp_fusion_waveforms.pkl')
BLINK_MARKERS_FILE = os.path.join(BASE_PATH, 'data/eos_dict.pkl')

# Signal parameters
RPPG_SAMPLING_RATE = 30.0  # Hz
RESP_SAMPLING_RATE = 15.0  # Hz
BLINK_SAMPLING_RATE = 30.0  # Hz
SIGNAL_DURATION = 120  # seconds

# Target signal lengths
RPPG_SIGNAL_LENGTH = int(RPPG_SAMPLING_RATE * SIGNAL_DURATION)  # 3600 samples
RESP_SIGNAL_LENGTH = int(RESP_SAMPLING_RATE * SIGNAL_DURATION)  # 1800 samples
BLINK_SIGNAL_LENGTH = int(BLINK_SAMPLING_RATE * SIGNAL_DURATION)  # 3600 samples

# Task definitions
LOW_LOAD_TASKS = ['still', 'read_rest', 'pattern_rest']
HIGH_LOAD_TASKS = ['pattern', 'number', 'read']
CORRUPTED_RESP = []  # Add any known corrupted data points here
SIGNAL_TYPES = ['ppg', 'resp', 'ecg', 'accel']  # All available signal types

# Experimental configurations
EXPERIMENT_CONFIGS = {
    'remote_ppg_contact_resp': {
        'use_remote_ppg': True,
        'use_remote_resp': False,
        'use_blink': False,
        'description': 'Remote PPG + Contact Resp'
    },
    'remote_ppg_remote_resp': {
        'use_remote_ppg': True,
        'use_remote_resp': True, 
        'use_blink': False,
        'description': 'Remote PPG + Remote Resp'
    },
    'remote_ppg_remote_resp_blink': {
        'use_remote_ppg': True,
        'use_remote_resp': True,
        'use_blink': True,
        'description': 'Remote PPG + Remote Resp + Blink Markers'
    }
}

def load_pickle(filename):
    """Load data from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def apply_smoothing_filter(signal_data, window_size=15, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth the signal
    
    Args:
        signal_data: numpy array containing the signal
        window_size: size of the window (must be odd)
        polyorder: order of the polynomial
        
    Returns:
        Smoothed signal as numpy array
    """
    # Make sure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Apply Savitzky-Golay filter
    try:
        smoothed_signal = signal.savgol_filter(signal_data, window_size, polyorder)
        return smoothed_signal
    except Exception as e:
        print(f"Error applying smoothing filter: {e}")
        # Fallback to simpler moving average if savgol fails
        window = np.ones(window_size) / window_size
        smoothed_signal = np.convolve(signal_data, window, mode='same')
        return smoothed_signal

def resample_signal(signal_data, original_length, target_length):
    """
    Resample a signal to a different length
    
    Args:
        signal_data: numpy array containing the signal
        original_length: current length of the signal
        target_length: desired length of the signal
        
    Returns:
        Resampled signal as numpy array
    """
    if len(signal_data) != original_length:
        print(f"Warning: Expected signal length {original_length} but got {len(signal_data)}")
    
    try:
        resampled_signal = resample(signal_data, target_length)
        return resampled_signal
    except Exception as e:
        print(f"Error resampling signal: {e}")
        return None

def normalize_signal(signal_data):
    """
    Normalize a signal to zero mean and unit variance
    
    Args:
        signal_data: numpy array containing the signal
        
    Returns:
        Normalized signal as numpy array
    """
    if np.std(signal_data) > 0:
        return (signal_data - np.mean(signal_data)) / np.std(signal_data)
    return signal_data

def extract_participant_info(key):
    """
    Extract participant ID and task name from key
    
    Args:
        key: String in format 'vXX_taskname'
        
    Returns:
        Tuple of (participant_id, task_name)
    """
    parts = key.split('_')
    participant_id = parts[0]
    task_name = '_'.join(parts[1:])
    return participant_id, task_name

def filter_by_participants(data, participant_list):
    """
    Filter data to only include specified participants
    
    Args:
        data: Dictionary with keys in format 'vXX_taskname'
        participant_list: List of participant IDs to include
        
    Returns:
        Filtered dictionary
    """
    filtered_data = {}
    for key in data:
        participant_id, _ = extract_participant_info(key)
        if participant_id in participant_list:
            filtered_data[key] = data[key]
    return filtered_data

def filter_signals_with_missing_data(X, y, keys, config):
    """
    Filter out samples with missing or zero-filled signals
    
    Args:
        X: Array of signal data with shape (n_samples, n_channels, signal_length)
        y: Array of labels
        keys: List of keys corresponding to each sample
        config: Experiment configuration dictionary
        
    Returns:
        Tuple of filtered (X, y, keys)
    """
    if len(X) == 0:
        return X, y, keys
    
    valid_indices = []
    
    # Determine which channels to check based on config
    channels_to_check = [0]  # Always check rPPG (first channel)
    
    if config['use_remote_resp']:
        channels_to_check.append(1)  # Check respiration (second channel)
    
    if config['use_blink']:
        channels_to_check.append(2 if config['use_remote_resp'] else 1)  # Check blink channel
    
    # Check each sample
    for i in range(len(X)):
        valid = True
        
        for channel in channels_to_check:
            # Check if channel exists
            if channel >= X[i].shape[0]:
                valid = False
                break
                
            # Check if signal is all zeros or has very low variance
            signal = X[i][channel]
            if np.all(signal == 0) or np.std(signal) < 1e-6:
                valid = False
                break
        
        if valid:
            valid_indices.append(i)
    
    # Filter arrays
    filtered_X = X[valid_indices] if len(valid_indices) > 0 else X
    filtered_y = y[valid_indices] if len(valid_indices) > 0 else y
    filtered_keys = [keys[i] for i in valid_indices] if len(valid_indices) > 0 else keys
    
    print(f"Filtered out {len(X) - len(filtered_X)} samples with missing or zero-filled signals")
    print(f"Remaining samples: {len(filtered_X)}")
    
    return filtered_X, filtered_y, filtered_keys

def load_nasa_tlx_scores(filename=NASA_TLX_FILE):
    """Loads NASA-TLX, calculates sum score, returns map."""
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    score_cols = ['Q3_1', 'Q3_3', 'Q3_5', 'Q3_6'] # exclude physical demand and performance
    df['raw_score_sum'] = df[score_cols].sum(axis=1)
    
    score_map = {}
    for _, row in df.iterrows():
        participant_id = f"v{row['Participant ID']}"
        task_name = row['Q2'].strip().lower()
        score_map[(participant_id, task_name)] = row['raw_score_sum'] 
    print(f"Loaded {len(score_map)} NASA-TLX scores.")
    return score_map

def create_median_split_labels(filename=NASA_TLX_FILE, exclude_still_for_median=True):
    """
    Loads NASA-TLX, calculates sum score, and creates binary labels
    (0=Low, 1=High) based on each participant's median score, optionally
    excluding the 'still' task when calculating the median.
    Returns a map: (participant_id, task_name) -> label (0 or 1)
    """
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    score_cols = ['Q3_1', 'Q3_3', 'Q3_5', 'Q3_6'] 
    df['raw_score_sum'] = df[score_cols].sum(axis=1)
    
    # --- Calculate median score ---
    if exclude_still_for_median:
        # Filter out 'still' task BEFORE calculating median
        active_tasks_df = df[df['Q2'] != 'still']
        median_scores = active_tasks_df.groupby('Participant ID')['raw_score_sum'].median()
        print("Calculated median excluding 'still' task.")
    else:
        # Calculate median using all tasks
        median_scores = df.groupby('Participant ID')['raw_score_sum'].median()
        print("Calculated median including 'still' task.")
    # -----------------------------

    label_map = {}
    skipped_count = 0
    # --- Iterate through the ORIGINAL df to label ALL tasks ---
    for _, row in df.iterrows(): 
        participant_id_num = row['Participant ID']
        participant_id_str = f"v{participant_id_num}" 
        task_name = row['Q2'].strip().lower()
        
        # Get the participant's calculated median score
        participant_median = median_scores.get(participant_id_num)
        
        if participant_median is None:
            # This might happen if a participant ONLY had 'still' data and exclude_still=True
            skipped_count += 1
            continue 
            
        current_score = row['raw_score_sum']
        
        # Assign label: 0 if <= median, 1 if > median
        label = 0 if current_score < participant_median else 1
        
        label_map[(participant_id_str, task_name)] = label
        
    print(f"Created median-split labels for {len(label_map)} entries.")
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} entries due to missing median.")
        
    return label_map 