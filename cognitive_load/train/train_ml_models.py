"""
Machine Learning models training script for cognitive load classification
Trains and evaluates ML models on the different experiment configurations:
1. Remote PPG + Contact Resp
2. Remote PPG + Remote Resp
3. Remote PPG + Remote Resp + Blink Markers
4. Remote PPG Only
5. Blink Markers Only

Supported models:
- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression
- LDA (Linear Discriminant Analysis)
- KNN (K-Nearest Neighbors)
- DT (Decision Tree)
- MLP (Multi-Layer Perceptron)
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
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
    },
    'lda': {
        'name': 'Linear Discriminant Analysis',
        'class': LinearDiscriminantAnalysis,
        'params': {
            'solver': 'svd',
            'store_covariance': True,
            'tol': 1e-4
        }
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2  # Euclidean distance
        }
    },
    'dt': {
        'name': 'Decision Tree',
        'class': DecisionTreeClassifier,
        'params': {
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 2025
        }
    },
    'mlp': {
        'name': 'Multi-Layer Perceptron',
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'max_iter': 200,
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
    Extract features from multiple physiological signals
    
    Args:
        signals_array: Array of signals with shape (n_samples, n_signals, signal_length)
        
    Returns:
        Array of feature vectors
    """
    all_features = []
    
    for sample_idx in range(signals_array.shape[0]):
        sample_features = {}
        
        # Process each signal in the sample
        for signal_idx in range(signals_array.shape[1]):
            signal = signals_array[sample_idx, signal_idx]
            
            # Determine signal type based on index
            signal_type = {0: 'ppg', 1: 'resp', 2: 'blink'}.get(signal_idx, f'sig{signal_idx}')
            
            # Apply signal-specific processing
            if signal_type == 'ppg':
                # PPG-specific features
                ppg_features = extract_ppg_features(signal)
                for name, value in ppg_features.items():
                    sample_features[f'{signal_type}_{name}'] = value
            elif signal_type == 'resp':
                # Respiratory-specific features
                resp_features = extract_respiratory_features(signal)
                for name, value in resp_features.items():
                    sample_features[f'{signal_type}_{name}'] = value
            elif signal_type == 'blink':
                # Blink-specific features
                blink_features = extract_blink_features(signal)
                for name, value in blink_features.items():
                    sample_features[f'{signal_type}_{name}'] = value
            else:
                # Generic time and frequency domain features for unknown signal types
                time_features = extract_time_domain_features(signal)
                freq_features = extract_frequency_domain_features(signal)
                
                for name, value in time_features.items():
                    sample_features[f'{signal_type}_{name}'] = value
                
                for name, value in freq_features.items():
                    sample_features[f'{signal_type}_{name}'] = value
        
        # Convert dictionary to flat list
        feature_values = list(sample_features.values())
        
        # Ensure all features are finite (not NaN or inf)
        for i, value in enumerate(feature_values):
            if value is None or np.isnan(value) or np.isinf(value):
                print(f"Warning: Non-finite feature value detected at position {i}, replacing with 0.0")
                feature_values[i] = 0.0
        
        all_features.append(feature_values)
    
    result = np.array(all_features)
    
    # Final check for NaN values in entire array
    if np.isnan(result).any():
        print("WARNING: NaN values detected in final feature array!")
        # Replace any remaining NaNs with zeros
        result = np.nan_to_num(result, nan=0.0)
    
    return result

def extract_ppg_features(signal, sampling_rate=30.0):
    """
    Extract PPG-specific features
    
    Args:
        signal: 1D numpy array containing PPG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of PPG features
    """
    features = {}
    
    # Apply bandpass filter to isolate cardiac frequency range (0.8-3.0 Hz, ~48-180 BPM)
    try:
        from scipy.signal import butter, filtfilt
        # Bandpass filter parameters
        low_cutoff = 0.8  # Hz
        high_cutoff = 3.0  # Hz
        nyquist = 0.5 * sampling_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        # Apply bandpass filter
        b, a = butter(3, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        # Detrend to remove baseline wander
        detrended_signal = signal - np.convolve(signal, np.ones(int(sampling_rate*3))/int(sampling_rate*3), mode='same')
    except Exception as e:
        print(f"Error in PPG filtering: {e}")
        filtered_signal = signal
        detrended_signal = signal
    
    # Basic statistical features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['range'] = np.ptp(signal)
    
    # Peak detection with physiological constraints
    try:
        from scipy.signal import find_peaks
        
        # Find peaks with constraints specific to PPG
        # Minimum distance between peaks (0.5s in samples)
        min_distance = int(0.5 * sampling_rate)
        
        # Minimum height for peaks (use signal characteristics)
        height_threshold = np.mean(filtered_signal) + 0.2 * np.std(filtered_signal)
        
        # Find peaks
        peaks, _ = find_peaks(filtered_signal, height=height_threshold, distance=min_distance)
        
        if len(peaks) > 1:
            # Calculate heart rate
            ibi = np.diff(peaks) / sampling_rate  # Inter-beat intervals in seconds
            # Filter out physiologically implausible IBIs (< 0.3s or > 1.5s, i.e. HR outside 40-200 BPM)
            ibi = ibi[(ibi >= 0.3) & (ibi <= 1.5)]
            features['hr_mean'] = 60 / np.mean(ibi)  # HR in BPM
            features['hr_std'] = np.std(60 / ibi) if len(ibi) > 1 else 0
            
            # HRV features
            if len(ibi) > 1:
                features['ibi_mean'] = np.mean(ibi) * 1000  # IBI in ms
                features['ibi_std'] = np.std(ibi) * 1000    # IBI std in ms
                features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(ibi)))) * 1000  # RMSSD in ms
                features['sdnn'] = np.std(ibi) * 1000  # SDNN in ms
                
                # pNN50: percentage of successive RR intervals that differ by more than 50 ms
                nn50 = np.sum(np.abs(np.diff(ibi)) > 0.05)  # Count intervals > 50ms
                features['pnn50'] = (nn50 / len(ibi)) * 100 if len(ibi) > 0 else 0
                
                # Poincaré plot features
                sd1 = np.std(np.diff(ibi) / np.sqrt(2)) * 1000
                sd2 = np.std(ibi) * 1000
                features['sd1'] = sd1  # Short-term variability
                features['sd2'] = sd2  # Long-term variability
                features['sd1_sd2_ratio'] = sd1 / sd2 if sd2 > 0 else 0
            else:
                features['ibi_mean'] = 0
                features['ibi_std'] = 0
                features['rmssd'] = 0
                features['sdnn'] = 0
                features['pnn50'] = 0
                features['sd1'] = 0
                features['sd2'] = 0
                features['sd1_sd2_ratio'] = 0
            
            # Pulse amplitude features
            pulse_amplitudes = filtered_signal[peaks]
            features['pulse_amp_mean'] = np.mean(pulse_amplitudes)
            features['pulse_amp_std'] = np.std(pulse_amplitudes)
            
            # Pulse width features (approximated)
            pulse_widths = []
            for i in range(len(peaks)-1):
                half_amp = (filtered_signal[peaks[i]] + filtered_signal[peaks[i+1]]) / 2
                # Find width at half amplitude
                width = np.sum(filtered_signal[peaks[i]:peaks[i+1]] > half_amp) / sampling_rate
                pulse_widths.append(width)
            
            if pulse_widths:
                features['pulse_width_mean'] = np.mean(pulse_widths)
                features['pulse_width_std'] = np.std(pulse_widths) if len(pulse_widths) > 1 else 0
            else:
                features['pulse_width_mean'] = 0
                features['pulse_width_std'] = 0
        else:
            # Default values if not enough peaks
            features['hr_mean'] = 0
            features['hr_std'] = 0
            features['ibi_mean'] = 0
            features['ibi_std'] = 0
            features['rmssd'] = 0
            features['sdnn'] = 0
            features['pnn50'] = 0
            features['sd1'] = 0
            features['sd2'] = 0
            features['sd1_sd2_ratio'] = 0
            features['pulse_amp_mean'] = 0
            features['pulse_amp_std'] = 0
            features['pulse_width_mean'] = 0
            features['pulse_width_std'] = 0
    except Exception as e:
        print(f"Error extracting PPG peak features: {e}")
        # Default values if peak detection fails
        features['hr_mean'] = 0
        features['hr_std'] = 0
        features['ibi_mean'] = 0
        features['ibi_std'] = 0
        features['rmssd'] = 0
        features['sdnn'] = 0
        features['pnn50'] = 0
        features['sd1'] = 0
        features['sd2'] = 0
        features['sd1_sd2_ratio'] = 0
        features['pulse_amp_mean'] = 0
        features['pulse_amp_std'] = 0
        features['pulse_width_mean'] = 0
        features['pulse_width_std'] = 0
    
    # Frequency domain features
    try:
        # Compute FFT
        n = len(filtered_signal)
        signal_fft = np.fft.fft(filtered_signal)
        # Get the magnitude spectrum
        magnitude = np.abs(signal_fft[:n//2])
        # Frequency values
        freq = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
        
        # Physiologically relevant frequency bands for HRV
        # VLF: Very low frequency (0.0033-0.04 Hz)
        vlf_mask = (freq >= 0.0033) & (freq < 0.04)
        features['vlf_power'] = np.sum(magnitude[vlf_mask]**2) if np.any(vlf_mask) else 0
        
        # LF: Low frequency (0.04-0.15 Hz)
        lf_mask = (freq >= 0.04) & (freq < 0.15)
        features['lf_power'] = np.sum(magnitude[lf_mask]**2) if np.any(lf_mask) else 0
        
        # HF: High frequency (0.15-0.4 Hz)
        hf_mask = (freq >= 0.15) & (freq < 0.4)
        features['hf_power'] = np.sum(magnitude[hf_mask]**2) if np.any(hf_mask) else 0
        
        # LF/HF ratio (commonly used in HRV analysis)
        features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
        
        # Cardiac frequency band (0.8-3.0 Hz)
        cardiac_mask = (freq >= 0.8) & (freq < 3.0)
        features['cardiac_power'] = np.sum(magnitude[cardiac_mask]**2) if np.any(cardiac_mask) else 0
        
        # Dominant frequency in cardiac band
        if np.any(cardiac_mask):
            cardiac_freqs = freq[cardiac_mask]
            cardiac_mags = magnitude[cardiac_mask]
            if len(cardiac_mags) > 0:
                idx_max = np.argmax(cardiac_mags)
                features['dominant_freq'] = cardiac_freqs[idx_max]
                features['dominant_power'] = cardiac_mags[idx_max]
            else:
                features['dominant_freq'] = 0
                features['dominant_power'] = 0
        else:
            features['dominant_freq'] = 0
            features['dominant_power'] = 0
    except Exception as e:
        print(f"Error extracting PPG frequency features: {e}")
        features['vlf_power'] = 0
        features['lf_power'] = 0
        features['hf_power'] = 0
        features['lf_hf_ratio'] = 0
        features['cardiac_power'] = 0
        features['dominant_freq'] = 0
        features['dominant_power'] = 0
    
    return features

def extract_respiratory_features(signal, sampling_rate=30.0):
    """
    Extract respiratory-specific features
    
    Args:
        signal: 1D numpy array containing respiratory signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of respiratory features
    """
    features = {}
    
    # Apply bandpass filter to isolate respiratory frequency range (0.05-1.0 Hz, ~3-60 breaths/min)
    try:
        from scipy.signal import butter, filtfilt
        # Bandpass filter parameters
        low_cutoff = 0.05  # Hz (3 breaths per minute)
        high_cutoff = 1.0  # Hz (60 breaths per minute)
        nyquist = 0.5 * sampling_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        # Apply bandpass filter
        b, a = butter(2, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
    except Exception as e:
        print(f"Error in respiratory filtering: {e}")
        filtered_signal = signal
    
    # Basic statistical features
    features['mean'] = np.mean(signal) if len(signal) > 0 else 0
    features['std'] = np.std(signal) if len(signal) > 0 else 0
    features['range'] = np.ptp(signal) if len(signal) > 0 else 0
    
    # Frequency domain features
    try:
        # Compute FFT
        n = len(filtered_signal)
        signal_fft = np.fft.fft(filtered_signal)
        # Get the magnitude spectrum
        magnitude = np.abs(signal_fft[:n//2])
        # Frequency values
        freq = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
        
        # Respiratory frequency band (0.05-1.0 Hz)
        resp_mask = (freq >= 0.05) & (freq < 1.0)
        features['resp_power'] = np.sum(magnitude[resp_mask]**2) if np.any(resp_mask) else 0
        
        # Dominant frequency in respiratory band
        if np.any(resp_mask):
            resp_freqs = freq[resp_mask]
            resp_mags = magnitude[resp_mask]
            if len(resp_mags) > 0:
                idx_max = np.argmax(resp_mags)
                features['dominant_freq'] = resp_freqs[idx_max]
                features['dominant_power'] = resp_mags[idx_max]
            else:
                features['dominant_freq'] = 0
                features['dominant_power'] = 0
        else:
            features['dominant_freq'] = 0
            features['dominant_power'] = 0
        
        # Spectral entropy (measure of signal complexity)
        if np.sum(magnitude) > 0:
            normalized_magnitude = magnitude / np.sum(magnitude)
            entropy = -np.sum(normalized_magnitude * np.log2(normalized_magnitude + 1e-10))
            features['spectral_entropy'] = entropy
        else:
            features['spectral_entropy'] = 0
    except Exception as e:
        print(f"Error extracting respiratory frequency features: {e}")
        features['resp_power'] = 0
        features['dominant_freq'] = 0
        features['dominant_power'] = 0
        features['spectral_entropy'] = 0
    
    return features

def interpolate_nans(signal):
    """
    Replace NaN values in a signal with linearly interpolated values
    
    Args:
        signal: 1D numpy array that may contain NaN values
        
    Returns:
        Signal with NaN values replaced by interpolated values
    """
    # Create a copy of the signal
    interpolated_signal = np.copy(signal)
    
    # Find indices of NaN values
    nan_indices = np.isnan(signal)
    
    # Check if there are any NaN values
    if np.any(nan_indices):
        # print(f"Found {np.sum(nan_indices)} NaN values in signal, interpolating...")
        
        # Get indices of non-NaN values
        valid_indices = np.where(~nan_indices)[0]
        
        # Get values at valid indices
        valid_values = signal[valid_indices]
        
        # If we have valid values, interpolate NaNs
        if len(valid_indices) > 0:
            # Create interpolation function using valid points
            if len(valid_indices) > 1:  # Need at least 2 points for interpolation
                from scipy import interpolate
                interp_func = interpolate.interp1d(
                    valid_indices, 
                    valid_values,
                    kind='linear', 
                    bounds_error=False,  # Allow extrapolation
                    fill_value=(valid_values[0], valid_values[-1])  # Fill with nearest valid values
                )
                
                # Create indices array for all points
                all_indices = np.arange(len(signal))
                
                # Apply interpolation to all points (will only modify NaN positions)
                interpolated_signal = interp_func(all_indices)
            else:
                # If only one valid value, fill all NaNs with that value
                interpolated_signal = np.full_like(signal, valid_values[0])
        else:
            # If all values are NaN, replace with zeros
            print("Warning: All values in signal are NaN, replacing with zeros")
            interpolated_signal = np.zeros_like(signal)
    
    return interpolated_signal

def extract_blink_features(signal, sampling_rate=30.0):
    """
    Extract blink-specific features by detecting troughs in the signal
    Since the amplitude represents eye openness, blinks are detected as troughs (minima)
    
    Args:
        signal: 1D numpy array containing blink signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of blink features
    """
    # Initialize features with default values to avoid NaN
    features = {
        'mean': 0.0,
        'std': 0.0,
        'range': 0.0,
        'blink_rate': 0.0,
        'blink_duration_mean': 0.0,
        'blink_duration_std': 0.0,
        'blink_interval_mean': 0.0,
        'blink_interval_std': 0.0,
        'blink_interval_cv': 0.0,
        'blink_amp_mean': 0.0,
        'blink_amp_std': 0.0,
        'blink_density': 0.0,
        'blink_depth_mean': 0.0,
        'blink_depth_std': 0.0
    }
    
    # Interpolate NaN values in the signal
    signal_no_nans = interpolate_nans(signal)
    
    # Basic statistical features
    if len(signal_no_nans) > 0:
        features['mean'] = np.mean(signal_no_nans)
        features['std'] = np.std(signal_no_nans)
        features['range'] = np.ptp(signal_no_nans)
    
    # Apply smoothing filter
    try:
        from scipy.signal import savgol_filter
        # Ensure window_length is odd
        window_length = 15
        if window_length % 2 == 0:
            window_length += 1
        smoothed_signal = savgol_filter(signal_no_nans, window_length=window_length, polyorder=3)
    except Exception as e:
        print(f"Error in blink signal smoothing: {e}")
        smoothed_signal = signal_no_nans
    
    # Blink detection (troughs)
    try:
        from scipy.signal import find_peaks
        
        # Normalize signal
        signal_range = np.max(smoothed_signal) - np.min(smoothed_signal)
        if signal_range > 1e-10:
            normalized_signal = (smoothed_signal - np.min(smoothed_signal)) / signal_range
        else:
            normalized_signal = np.zeros_like(smoothed_signal)
        
        # Invert the signal to convert troughs to peaks
        inverted_signal = 1.0 - normalized_signal
        
        # Find peaks in inverted signal (which are troughs in original signal)
        # Blinks typically have a minimum duration of 100-400ms
        min_distance = int(0.1 * sampling_rate)  # Minimum 100ms between blinks
        height_threshold = 0.4  # Threshold for blink detection (inverted signal)
        
        peaks, properties = find_peaks(inverted_signal, 
                                       height=height_threshold, 
                                       distance=min_distance, 
                                       width=int(0.1*sampling_rate),  # Min width of 100ms
                                       prominence=0.15)  # Minimum prominence to be considered a blink
        
        # These peaks represent blinks (eye closures)
        if len(peaks) > 0:
            # Blink rate (blinks per minute)
            features['blink_rate'] = len(peaks) / (len(signal) / sampling_rate) * 60
            
            # Blink duration features
            widths = properties['widths'] / sampling_rate  # Convert to seconds
            features['blink_duration_mean'] = np.mean(widths)
            features['blink_duration_std'] = np.std(widths) if len(widths) > 1 else 0
            
            # Blink interval features
            if len(peaks) > 1:
                intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
                features['blink_interval_mean'] = np.mean(intervals)
                features['blink_interval_std'] = np.std(intervals) if len(intervals) > 1 else 0
                features['blink_interval_cv'] = (features['blink_interval_std'] / features['blink_interval_mean'] 
                                               if features['blink_interval_mean'] > 0 else 0)
            
            # Blink amplitude features (using prominence of peaks in inverted signal)
            amplitudes = properties['prominences']
            features['blink_amp_mean'] = np.mean(amplitudes)
            features['blink_amp_std'] = np.std(amplitudes) if len(amplitudes) > 1 else 0
            
            # Additional blink metrics
            features['blink_density'] = len(peaks) / (len(signal) / sampling_rate)  # Blinks per second
            
            # Calculate mean and std of blink depths (how closed the eyes get)
            blink_depths = inverted_signal[peaks]
            features['blink_depth_mean'] = np.mean(blink_depths)
            features['blink_depth_std'] = np.std(blink_depths) if len(blink_depths) > 1 else 0
        else:
            # Default values if no blinks detected
            features['blink_rate'] = 0
            features['blink_duration_mean'] = 0
            features['blink_duration_std'] = 0
            features['blink_interval_mean'] = 0
            features['blink_interval_std'] = 0
            features['blink_interval_cv'] = 0
            features['blink_amp_mean'] = 0
            features['blink_amp_std'] = 0
            features['blink_density'] = 0
            features['blink_depth_mean'] = 0
            features['blink_depth_std'] = 0
    except Exception as e:
        print(f"Error extracting blink features: {e}")
    
    # Final safety check to ensure no NaN values
    for key in features:
        if np.isnan(features[key]):
            print(f"Warning: NaN detected in blink feature '{key}', setting to 0.0")
            features[key] = 0.0
    
    return features

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