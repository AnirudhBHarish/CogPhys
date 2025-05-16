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
    log_file = f"./results/experiments_log_{timestamp}.txt"
    
    # Redirect stdout to log file and console
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(log_file, 'w') as log:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log)
        
        print(f"===== Starting Experiments at {timestamp} =====")
        
        # Run ML experiments
        if args.ml or args.all:
            print("\n===== Running Machine Learning Models =====")
            start_ml = time.time()
            
            try:
                if args.exp:
                    # Run with specific experiment filter
                    from data_loader import prepare_all_experiments
                    experiment_data = prepare_all_experiments(args.exp)
                    
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
        
        # Run DL experiments
        if args.dl or args.all:
            print("\n===== Running Deep Learning Models =====")
            start_dl = time.time()
            
            try:
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
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n===== All Experiments Completed =====")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Log file saved to: {log_file}")
        
        # Restore original stdout
        sys.stdout = original_stdout

if __name__ == '__main__':
    main() 