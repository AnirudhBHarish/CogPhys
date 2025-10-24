"""
Launcher script for running all experiments
This script runs both ML and DL experiments with the different configurations
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Import experiment configurations
from utils import EXPERIMENT_CONFIGS

def main():
    """Main function to parse arguments and run experiments"""
    parser = argparse.ArgumentParser(description='Run cognitive load classification experiments')
    parser.add_argument('--ml', action='store_true', help='Run Machine Learning experiments')
    parser.add_argument('--dl', action='store_true', help='Run Deep Learning experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--exp', type=str, nargs='+', choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='Specific experiment(s) to run')
    parser.add_argument('--split', type=int, default=0,
                        help='Split index for cross-validation (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Display detailed output (default: only show summaries)')
    
    args = parser.parse_args()
    
    # Default to all if no specific flags
    if not (args.ml or args.dl or args.all or args.exp):
        args.all = True
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Track start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file
    log_file = f"./results/experiments_log_split{args.split}_{timestamp}.txt"
    log_file = f"./results/experiments_log_split{args.split}.txt"
    print(f"Log file will be saved to: {log_file}")
    # Define markers for summary sections
    summary_markers = [
        "SUMMARY FOR EXPERIMENT:"
    ]
    
    # Open log file
    with open(log_file, 'w') as log:
        # Store original stdout
        original_stdout = sys.stdout
        
        print(f"===== Starting Experiments at {timestamp} =====")
        log.write(f"===== Starting Experiments at {timestamp} =====\n")
        
        print(f"Verbose mode: {'ON' if args.verbose else 'OFF - showing only summary tables'}")
        log.write(f"Verbose mode: {'ON' if args.verbose else 'OFF - showing only summary tables'}\n")
        
        # Run ML experiments
        if args.ml or args.all:
            print("\n===== Running Machine Learning Models =====")
            log.write("\n===== Running Machine Learning Models =====\n")
            start_ml = time.time()
            
            try:
                # Create a special stdout capture class
                class StdoutCapture:
                    def __init__(self):
                        self.in_summary_section = False
                        self.buffer = []
                        self.captured_lines = 0
                        self.max_buffer_lines = 15  # Maximum expected lines in a summary table
                    
                    def write(self, text):
                        # Always write to console
                        original_stdout.write(text)
                        
                        # For log file, filter based on verbose flag
                        if args.verbose:
                            # In verbose mode, write everything to log
                            log.write(text)
                        else:
                            # In non-verbose mode, only write summary tables and important messages
                            
                            # Check if this is a summary section start marker
                            if "SUMMARY FOR EXPERIMENT:" in text:
                                self.in_summary_section = True
                                self.buffer = []
                                self.captured_lines = 0
                                # Store beginning of summary
                                self.buffer.append("\n" + "="*80 + "\n")
                                self.buffer.append(text)
                            
                            # Continue capturing if in summary section
                            elif self.in_summary_section:
                                # Add line to buffer
                                self.buffer.append(text)
                                self.captured_lines += 1
                                
                                # Check if this might be the end of the summary section
                                if "=" * 20 in text and self.captured_lines > 5:
                                    # Write entire buffer to log
                                    for line in self.buffer:
                                        log.write(line)
                                    # Reset buffer
                                    self.buffer = []
                                    self.in_summary_section = False
                                    
                                # Safety check - if buffer gets too large, write what we have
                                if self.captured_lines >= self.max_buffer_lines:
                                    for line in self.buffer:
                                        log.write(line)
                                    self.buffer = []
                            
                            # Always write section headers and completion messages
                            elif text.strip().startswith("===== Running") or "===== All Experiments Completed" in text:
                                log.write(text)
                            elif text.startswith("ML experiments completed") or text.startswith("DL experiments completed"):
                                log.write(text)
                            elif text.startswith("Total execution time:") or text.startswith("Log file saved to:"):
                                log.write(text)
                            elif text.startswith("===== Starting Experiments"):
                                log.write(text)
                    
                    def flush(self):
                        original_stdout.flush()
                        log.flush()
                
                # Use the capture class temporarily
                sys.stdout = StdoutCapture()
                
                if args.exp:
                    # Run with specific experiment filter
                    from data_loader import prepare_all_experiments
                    experiment_data = prepare_all_experiments(args.exp, split_idx=args.split)
                    from train_ml_models import run_ml_experiment, print_experiment_summary
                    for exp_name, exp_data in experiment_data.items():
                        if exp_data:
                            results = run_ml_experiment(exp_data)
                            print_experiment_summary(exp_name, results)
                else:
                    # Run all ML experiments
                    import train_ml_models
                    train_ml_models.main()
                
                ml_time = time.time() - start_ml
                print(f"ML experiments completed in {ml_time:.2f} seconds")
            
            except Exception as e:
                print(f"Error in ML experiments: {str(e)}")
            
            finally:
                # Restore stdout
                sys.stdout = original_stdout
        
        # Run DL experiments
        if args.dl or args.all:
            print("\n===== Running Deep Learning Models =====")
            log.write("\n===== Running Deep Learning Models =====\n")
            start_dl = time.time()
            
            try:
                # Create a special stdout capture class for DL part
                class StdoutCapture:
                    def __init__(self):
                        self.in_summary_section = False
                        self.buffer = []
                        self.captured_lines = 0
                        self.max_buffer_lines = 15  # Maximum expected lines in a summary table
                    
                    def write(self, text):
                        # Always write to console
                        original_stdout.write(text)
                        
                        # For log file, filter based on verbose flag
                        if args.verbose:
                            # In verbose mode, write everything to log
                            log.write(text)
                        else:
                            # In non-verbose mode, only write summary tables and important messages
                            
                            # Check if this is a summary section start marker
                            if "SUMMARY FOR EXPERIMENT:" in text:
                                self.in_summary_section = True
                                self.buffer = []
                                self.captured_lines = 0
                                # Store beginning of summary
                                self.buffer.append("\n" + "="*80 + "\n")
                                self.buffer.append(text)
                            
                            # Continue capturing if in summary section
                            elif self.in_summary_section:
                                # Add line to buffer
                                self.buffer.append(text)
                                self.captured_lines += 1
                                
                                # Check if this might be the end of the summary section
                                if "=" * 20 in text and self.captured_lines > 5:
                                    # Write entire buffer to log
                                    for line in self.buffer:
                                        log.write(line)
                                    # Reset buffer
                                    self.buffer = []
                                    self.in_summary_section = False
                                    
                                # Safety check - if buffer gets too large, write what we have
                                if self.captured_lines >= self.max_buffer_lines:
                                    for line in self.buffer:
                                        log.write(line)
                                    self.buffer = []
                            
                            # Always write section headers and completion messages
                            elif text.strip().startswith("===== Running") or "===== All Experiments Completed" in text:
                                log.write(text)
                            elif text.startswith("ML experiments completed") or text.startswith("DL experiments completed"):
                                log.write(text)
                            elif text.startswith("Total execution time:") or text.startswith("Log file saved to:"):
                                log.write(text)
                            elif text.startswith("===== Starting Experiments"):
                                log.write(text)
                    
                    def flush(self):
                        original_stdout.flush()
                        log.flush()
                
                # Use the capture class temporarily
                sys.stdout = StdoutCapture()
                
                if args.exp:
                    # Run with specific experiment filter
                    from data_loader import prepare_all_experiments
                    experiment_data = prepare_all_experiments(args.exp)
                    
                    from train_dl_models import run_dl_experiment, print_experiment_summary
                    for exp_name, exp_data in experiment_data.items():
                        if exp_data:
                            results = run_dl_experiment(exp_name, exp_data)
                            print_experiment_summary(exp_name, results)
                else:
                    # Run all DL experiments
                    import train_dl_models
                    train_dl_models.main()
                
                dl_time = time.time() - start_dl
                print(f"DL experiments completed in {dl_time:.2f} seconds")
            
            except Exception as e:
                print(f"Error in DL experiments: {str(e)}")
            
            finally:
                # Restore stdout
                sys.stdout = original_stdout
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n===== All Experiments Completed =====")
        log.write(f"\n===== All Experiments Completed =====\n")
        print(f"Total execution time: {total_time:.2f} seconds")
        log.write(f"Total execution time: {total_time:.2f} seconds\n")
        print(f"Log file saved to: {log_file}")
        log.write(f"Log file saved to: {log_file}\n")

if __name__ == '__main__':
    main() 