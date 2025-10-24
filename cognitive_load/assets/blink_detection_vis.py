#!/usr/bin/env python3
"""
Improved script to visualize blink waveforms and peak detection results.
This version includes parameter tuning capabilities and enhanced visualizations.
Saves visualizations to ./assets/blink/
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import random
from matplotlib.widgets import Slider, Button, CheckButtons
import argparse

# Paths
BLINK_MARKERS_FILE = '../data/eos_dict.pkl'
OUTPUT_DIR = 'assets/blink'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_blink_data():
    """Load blink marker data from pickle file"""
    try:
        with open(BLINK_MARKERS_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded blink data with {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading blink data: {e}")
        return None

def normalize_signal(signal):
    """Normalize signal to range [0,1]"""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min > 1e-10:
        return (signal - signal_min) / (signal_max - signal_min)
    return np.zeros_like(signal)

def detect_blinks(signal, params=None, sampling_rate=30.0):
    """
    Detect blinks in the signal with customizable parameters
    
    Args:
        signal: 1D numpy array containing blink signal
        params: Dictionary of detection parameters (overrides defaults)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with detected peaks, properties, and processed signal
    """
    # Default parameters
    default_params = {
        'window_length': 15,
        'polyorder': 3,
        'height': 0.5,
        'distance': int(0.1 * sampling_rate),
        'width': int(0.1 * sampling_rate),
        'prominence': 0.2
    }
    
    # Use provided params or defaults
    if params is None:
        params = default_params
    else:
        # Use defaults for any missing params
        for k, v in default_params.items():
            if k not in params:
                params[k] = v
    
    # Apply smoothing filter
    try:
        # Ensure window_length is odd
        window_length = params['window_length']
        if window_length % 2 == 0:
            window_length += 1
            
        smoothed_signal = savgol_filter(signal, window_length, params['polyorder'])
    except Exception as e:
        print(f"Error in blink signal smoothing: {e}")
        smoothed_signal = signal
    
    # Normalize signal
    normalized_signal = normalize_signal(smoothed_signal)
    
    # Find peaks (blinks)
    peaks, properties = find_peaks(
        normalized_signal, 
        height=params['height'], 
        distance=params['distance'],
        width=params['width'], 
        prominence=params['prominence']
    )
    
    return {
        'original': signal,
        'smoothed': smoothed_signal,
        'normalized': normalized_signal,
        'peaks': peaks,
        'properties': properties,
        'params': params
    }

def extract_blink_features(detection_results, sampling_rate=30.0):
    """
    Extract blink features from detection results
    
    Args:
        detection_results: Dictionary with detection results
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of blink features
    """
    peaks = detection_results['peaks']
    properties = detection_results['properties']
    signal = detection_results['original']
    
    features = {}
    
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
            features['blink_interval_std'] = np.std(intervals)
            features['blink_interval_cv'] = (features['blink_interval_std'] / features['blink_interval_mean'] 
                                            if features['blink_interval_mean'] > 0 else 0)
        else:
            features['blink_interval_mean'] = 0
            features['blink_interval_std'] = 0
            features['blink_interval_cv'] = 0
        
        # Blink amplitude features
        amplitudes = properties['prominences']
        features['blink_amp_mean'] = np.mean(amplitudes)
        features['blink_amp_std'] = np.std(amplitudes) if len(amplitudes) > 1 else 0
    else:
        features['blink_rate'] = 0
        features['blink_duration_mean'] = 0
        features['blink_duration_std'] = 0
        features['blink_interval_mean'] = 0
        features['blink_interval_std'] = 0
        features['blink_interval_cv'] = 0
        features['blink_amp_mean'] = 0
        features['blink_amp_std'] = 0
    
    return features

def visualize_blink_detection(key, signal, detection_results, features, time_window=None):
    """
    Enhanced visualization of blink detection results
    
    Args:
        key: Sample identifier
        signal: Original signal
        detection_results: Dictionary with detection results
        features: Dictionary of extracted features
        time_window: Optional tuple of (start_sec, end_sec) to zoom
    """
    peaks = detection_results['peaks']
    properties = detection_results['properties']
    smoothed = detection_results['smoothed']
    normalized = detection_results['normalized']
    params = detection_results['params']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Time axis (in seconds)
    time = np.arange(len(signal)) / 30.0  # 30 Hz sampling rate
    
    # Apply time window if specified
    if time_window:
        start_sec, end_sec = time_window
        start_idx = max(0, int(start_sec * 30.0))
        end_idx = min(len(signal), int(end_sec * 30.0))
        time_slice = slice(start_idx, end_idx)
        display_range = (start_sec, end_sec)
    else:
        time_slice = slice(0, len(signal))
        display_range = (0, len(signal) / 30.0)
    
    # Plot original signal
    ax = axes[0]
    ax.plot(time[time_slice], signal[time_slice], label='Original Signal')
    ax.set_title(f'Original Blink Signal - {key}')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    
    # Plot smoothed signal
    ax = axes[1]
    ax.plot(time[time_slice], smoothed[time_slice], label='Smoothed Signal')
    ax.set_title(f'Smoothed Signal (Savitzky-Golay filter, window={params["window_length"]}, polyorder={params["polyorder"]})')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    
    # Plot normalized signal with detected peaks
    ax = axes[2]
    ax.plot(time[time_slice], normalized[time_slice], label='Normalized Signal')
    
    if len(peaks) > 0:
        # Find peaks within the time window
        visible_peaks = [p for p in peaks if p in range(time_slice.start, time_slice.stop)]
        
        # Plot detected peaks
        peak_times = [time[p] for p in visible_peaks]
        peak_values = [normalized[p] for p in visible_peaks]
        ax.plot(peak_times, peak_values, 'ro', label='Detected Blinks')
        
        # Highlight peak widths for visible peaks
        for peak in visible_peaks:
            # Find the index of this peak in the original peaks array
            peak_idx = np.where(peaks == peak)[0][0]
            
            # Get width and boundary indices
            width = properties['widths'][peak_idx]
            left_ips = properties['left_ips'][peak_idx]
            right_ips = properties['right_ips'][peak_idx]
            
            # Convert to integer indices
            left_idx = int(left_ips)
            right_idx = int(right_ips)
            
            # Ensure indices are within bounds
            left_idx = max(0, left_idx)
            right_idx = min(len(normalized) - 1, right_idx)
            
            # Highlight width of peak
            ax.axvspan(time[left_idx], time[right_idx], alpha=0.3, color='green')
            
            # Add label with peak info
            ax.annotate(f"{peak_idx}", (time[peak], normalized[peak]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red')
    
    ax.set_title('Normalized Signal with Detected Blinks')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Normalized Amplitude')
    ax.legend()
    ax.grid(True)
    
    # Set x-axis limits to the specific time window
    for a in axes:
        a.set_xlim(display_range)
    
    # Add feature annotations
    plt.figtext(0.02, 0.02, '\n'.join([
        f"Blink Rate: {features['blink_rate']:.2f} blinks/min",
        f"Detected Blinks: {len(peaks)}",
        f"Avg Blink Duration: {features['blink_duration_mean']*1000:.2f} ms",
        f"Avg Blink Interval: {features['blink_interval_mean']:.2f} sec"
    ]), fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # Add detection parameters
    plt.figtext(0.7, 0.02, '\n'.join([
        f"Detection Parameters:",
        f"Height threshold: {params['height']:.2f}",
        f"Min distance: {params['distance']} samples ({params['distance']/30:.2f} sec)",
        f"Min width: {params['width']} samples ({params['width']/30*1000:.1f} ms)",
        f"Min prominence: {params['prominence']:.2f}"
    ]), fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for annotations
    
    # Save figure
    if time_window:
        start_str = f"{time_window[0]:.1f}"
        end_str = f"{time_window[1]:.1f}"
        output_path = os.path.join(OUTPUT_DIR, f'blink_detection_{key}_t{start_str}-{end_str}.png')
    else:
        output_path = os.path.join(OUTPUT_DIR, f'blink_detection_{key}.png')
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to {output_path}")

def create_interactive_tuning_plot(key, signal):
    """
    Create an interactive plot for tuning blink detection parameters
    
    Args:
        key: Sample identifier
        signal: Original signal
    """
    # Initial parameter values
    initial_params = {
        'window_length': 15,
        'polyorder': 3,
        'height': 0.5,
        'distance': 3,  # in samples
        'width': 3,     # in samples
        'prominence': 0.2
    }
    
    # Create initial detection results
    detection_results = detect_blinks(signal, initial_params)
    features = extract_blink_features(detection_results)
    
    # Create figure with time-series data
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.35)  # Make room for sliders
    
    # Time axis (in seconds)
    time = np.arange(len(signal)) / 30.0  # 30 Hz sampling rate
    
    # Initialize plot elements
    line_original, = ax.plot(time, signal, alpha=0.5, label='Original')
    line_smoothed, = ax.plot(time, detection_results['smoothed'], 'g-', label='Smoothed')
    line_normalized, = ax.plot(time, detection_results['normalized'], 'b-', label='Normalized')
    
    # Initial peak markers
    peaks = detection_results['peaks']
    peak_markers, = ax.plot(time[peaks], detection_results['normalized'][peaks], 'ro', markersize=8, label='Detected Blinks')
    
    # Add peak count text
    peak_text = ax.text(0.05, 0.95, f"Detected {len(peaks)} blinks", transform=ax.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    ax.set_title(f'Interactive Blink Detection Tuning - {key}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    
    # Create sliders for parameter adjustment
    ax_window = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_poly = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_height = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_distance = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_width = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_prominence = plt.axes([0.25, 0.00, 0.65, 0.03])
    
    # Create sliders with appropriate ranges
    slider_window = Slider(ax_window, 'Window Length', 5, 51, valinit=initial_params['window_length'], valstep=2)
    slider_poly = Slider(ax_poly, 'Poly Order', 1, 5, valinit=initial_params['polyorder'], valstep=1)
    slider_height = Slider(ax_height, 'Height', 0.0, 1.0, valinit=initial_params['height'])
    slider_distance = Slider(ax_distance, 'Min Distance (s)', 0.0, 1.0, valinit=initial_params['distance']/30.0)
    slider_width = Slider(ax_width, 'Min Width (s)', 0.0, 0.5, valinit=initial_params['width']/30.0)
    slider_prominence = Slider(ax_prominence, 'Prominence', 0.0, 1.0, valinit=initial_params['prominence'])
    
    # Function to update plot based on slider changes
    def update(val):
        # Get current parameter values
        params = {
            'window_length': int(slider_window.val),
            'polyorder': int(slider_poly.val),
            'height': slider_height.val,
            'distance': int(slider_distance.val * 30.0),  # convert to samples
            'width': int(slider_width.val * 30.0),        # convert to samples
            'prominence': slider_prominence.val
        }
        
        # Recompute with new parameters
        detection_results = detect_blinks(signal, params)
        features = extract_blink_features(detection_results)
        
        # Update plot data
        line_smoothed.set_ydata(detection_results['smoothed'])
        line_normalized.set_ydata(detection_results['normalized'])
        
        # Update peak markers
        peaks = detection_results['peaks']
        if len(peaks) > 0:
            peak_markers.set_data(time[peaks], detection_results['normalized'][peaks])
        else:
            peak_markers.set_data([], [])
        
        # Update peak count
        peak_text.set_text(f"Detected {len(peaks)} blinks\nBlink rate: {features['blink_rate']:.1f} bpm")
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_window.on_changed(update)
    slider_poly.on_changed(update)
    slider_height.on_changed(update)
    slider_distance.on_changed(update)
    slider_width.on_changed(update)
    slider_prominence.on_changed(update)
    
    # Button to save current configuration
    ax_save = plt.axes([0.05, 0.12, 0.15, 0.05])
    button_save = Button(ax_save, 'Save')
    
    def save_config(event):
        # Get current parameter values
        params = {
            'window_length': int(slider_window.val),
            'polyorder': int(slider_poly.val),
            'height': slider_height.val,
            'distance': int(slider_distance.val * 30.0),  # convert to samples
            'width': int(slider_width.val * 30.0),        # convert to samples
            'prominence': slider_prominence.val
        }
        
        # Generate detection results with current parameters
        detection_results = detect_blinks(signal, params)
        features = extract_blink_features(detection_results)
        
        # Save static figure for current parameters
        visualize_blink_detection(key, signal, detection_results, features)
        
        # Also save a zoomed-in view of a representative segment (10 seconds around a peak)
        if len(detection_results['peaks']) > 0:
            # Find a peak near the middle of the signal
            mid_idx = len(signal) // 2
            nearest_peak_idx = np.argmin(np.abs(detection_results['peaks'] - mid_idx))
            peak_time = detection_results['peaks'][nearest_peak_idx] / 30.0
            
            # Create a 10-second window around the peak
            start_sec = max(0, peak_time - 5.0)
            end_sec = min(len(signal) / 30.0, peak_time + 5.0)
            
            # Save zoomed view
            visualize_blink_detection(key, signal, detection_results, features, (start_sec, end_sec))
        
        print(f"Saved current configuration for {key}")
        print(f"Parameters: {params}")
        print(f"Features: {features}")
    
    button_save.on_clicked(save_config)
    
    # Show the plot
    plt.show()

def create_task_comparison(blink_data, task_types=None):
    """
    Create comparison plots showing blink rates across different tasks
    
    Args:
        blink_data: Dictionary of blink data
        task_types: List of task types to compare (default: still, read, pattern, number)
    """
    if task_types is None:
        task_types = ['still', 'read', 'pattern', 'number']
    
    # Dictionary to store blink rates by task
    task_blink_rates = {task: [] for task in task_types}
    task_blink_durations = {task: [] for task in task_types}
    task_blink_intervals = {task: [] for task in task_types}
    
    # Process all samples
    for key in blink_data:
        # Find which task this sample belongs to
        matching_tasks = [task for task in task_types if task in key]
        if not matching_tasks:
            continue
        
        task = matching_tasks[0]
        
        # Use average blink signal if available, otherwise use left eye
        if 'eo_signal_avg' in blink_data[key]:
            signal = blink_data[key]['eo_signal_avg']
        elif 'eo_signal_left' in blink_data[key]:
            signal = blink_data[key]['eo_signal_left']
        else:
            continue
        
        # Ensure signal is valid
        if signal is None or len(signal) < 100:
            continue
        
        # Detect blinks with default parameters
        detection_results = detect_blinks(signal)
        features = extract_blink_features(detection_results)
        
        # Store features for this task
        task_blink_rates[task].append(features['blink_rate'])
        
        if features['blink_duration_mean'] > 0:
            task_blink_durations[task].append(features['blink_duration_mean'] * 1000)  # Convert to ms
        
        if features['blink_interval_mean'] > 0:
            task_blink_intervals[task].append(features['blink_interval_mean'])
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot blink rates
    ax = axes[0]
    box_data = [task_blink_rates[task] for task in task_types]
    ax.boxplot(box_data, labels=task_types, patch_artist=True)
    ax.set_title('Blink Rate Comparison Across Tasks')
    ax.set_ylabel('Blinks per Minute')
    ax.grid(True, axis='y')
    
    # Add sample counts
    for i, task in enumerate(task_types):
        count = len(task_blink_rates[task])
        ax.annotate(f"n={count}", xy=(i+1, 0), xytext=(0, -20), 
                   textcoords='offset points', ha='center', va='top')
    
    # Plot blink durations
    ax = axes[1]
    box_data = [task_blink_durations[task] for task in task_types]
    ax.boxplot(box_data, labels=task_types, patch_artist=True)
    ax.set_title('Blink Duration Comparison Across Tasks')
    ax.set_ylabel('Blink Duration (ms)')
    ax.grid(True, axis='y')
    
    # Plot blink intervals
    ax = axes[2]
    box_data = [task_blink_intervals[task] for task in task_types]
    ax.boxplot(box_data, labels=task_types, patch_artist=True)
    ax.set_title('Blink Interval Comparison Across Tasks')
    ax.set_ylabel('Interval Between Blinks (s)')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved task comparison plot to {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics by Task:")
    for task in task_types:
        rates = task_blink_rates[task]
        durations = task_blink_durations[task]
        intervals = task_blink_intervals[task]
        
        print(f"\n{task.upper()} (n={len(rates)}):")
        if rates:
            print(f"  Blink Rate: {np.mean(rates):.2f} ± {np.std(rates):.2f} bpm")
        if durations:
            print(f"  Blink Duration: {np.mean(durations):.2f} ± {np.std(durations):.2f} ms")
        if intervals:
            print(f"  Blink Interval: {np.mean(intervals):.2f} ± {np.std(intervals):.2f} s")

def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Blink detection visualization tool')
    parser.add_argument('--mode', choices=['basic', 'interactive', 'comparison', 'all'], 
                       default='basic', help='Visualization mode')
    parser.add_argument('--sample', type=str, help='Process a specific sample key')
    parser.add_argument('--task', type=str, help='Process samples for a specific task')
    parser.add_argument('--count', type=int, default=2, 
                       help='Number of samples per task (for basic mode)')
    
    args = parser.parse_args()
    
    # Load blink data
    blink_data = load_blink_data()
    if blink_data is None:
        print("Failed to load blink data")
        return
    
    # List of all keys
    all_keys = list(blink_data.keys())
    
    # Process a specific sample if requested
    if args.sample and args.sample in blink_data:
        selected_keys = [args.sample]
    # Process samples for a specific task if requested
    elif args.task:
        selected_keys = [key for key in all_keys if args.task in key]
        # Limit to a reasonable number
        if len(selected_keys) > args.count and args.mode != 'comparison':
            selected_keys = random.sample(selected_keys, args.count)
    else:
        # Default: select samples from different task types
        task_types = ['still', 'read', 'pattern', 'number']
        selected_keys = []
        for task in task_types:
            # Find keys matching this task
            matching_keys = [key for key in all_keys if task in key]
            
            # Select random samples for each task type
            if matching_keys:
                selected_keys.extend(random.sample(matching_keys, min(args.count, len(matching_keys))))
    
    # Perform requested visualization mode
    if args.mode == 'basic' or args.mode == 'all':
        # Process each selected sample with basic visualization
        for key in selected_keys:
            print(f"\nProcessing sample: {key}")
            
            # Use average blink signal if available, otherwise use left eye
            if 'eo_signal_avg' in blink_data[key]:
                signal = blink_data[key]['eo_signal_avg']
            elif 'eo_signal_left' in blink_data[key]:
                signal = blink_data[key]['eo_signal_left']
            else:
                print(f"No valid blink signal found for {key}")
                continue
            
            # Ensure signal is valid
            if signal is None or len(signal) < 100:
                print(f"Invalid signal for {key}")
                continue
            
            # Detect blinks with default parameters
            detection_results = detect_blinks(signal)
            features = extract_blink_features(detection_results)
            
            # Visualize full signal
            visualize_blink_detection(key, signal, detection_results, features)
            
            # Also create a detailed view of a 10-second segment if there are peaks
            if len(detection_results['peaks']) > 0:
                # Find a peak near the middle of the signal
                mid_idx = len(signal) // 2
                nearest_peak_idx = np.argmin(np.abs(detection_results['peaks'] - mid_idx))
                peak_time = detection_results['peaks'][nearest_peak_idx] / 30.0
                
                # Create a 10-second window around the peak
                start_sec = max(0, peak_time - 5.0)
                end_sec = min(len(signal) / 30.0, peak_time + 5.0)
                
                # Save zoomed view
                visualize_blink_detection(key, signal, detection_results, features, (start_sec, end_sec))
            
            # Print features
            print(f"Blink features for {key}:")
            for k, v in features.items():
                print(f"  {k}: {v}")
    
    if args.mode == 'interactive' or args.mode == 'all':
        # Only use the first selected key for interactive mode
        if selected_keys:
            key = selected_keys[0]
            print(f"\nInteractive mode for sample: {key}")
            
            # Use average blink signal if available, otherwise use left eye
            if 'eo_signal_avg' in blink_data[key]:
                signal = blink_data[key]['eo_signal_avg']
            elif 'eo_signal_left' in blink_data[key]:
                signal = blink_data[key]['eo_signal_left']
            else:
                print(f"No valid blink signal found for {key}")
                return
            
            # Ensure signal is valid
            if signal is None or len(signal) < 100:
                print(f"Invalid signal for {key}")
                return
            
            # Create interactive plot
            create_interactive_tuning_plot(key, signal)
    
    if args.mode == 'comparison' or args.mode == 'all':
        # Create comparison plots across tasks
        create_task_comparison(blink_data)

if __name__ == "__main__":
    main() 