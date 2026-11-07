import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn # For nn.CrossEntropyLoss
import numpy as np
import time
import os
import pickle as pkl

import misc # misc.py
import optimizers # optimizers.py
from networks import MLP, CNN, fit # networks.py

def run_experiment(args):
    """Main function to run experiments."""
    # --- Setup ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed) # For numpy operations if any
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True # Can slow down, but good for reproducibility
            torch.backends.cudnn.benchmark = False   # Disable benchmarking for determinism

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    dataset_seed = args.seed if args.seed is not None else 42 # Seed for data splitting
    if args.dataset == 'cifar':
        train_dataset, val_dataset = misc.load_cifar(num_train=args.num_train, num_val=args.num_val, seed=dataset_seed)
        NetModel = CNN # CNN class from networks.py
        criterion = nn.CrossEntropyLoss()
        # For CNN, input_channels might be needed if not fixed in CNN class
        # net_params = {'num_input_channels': 3, 'num_classes': 10} # Example
    elif args.dataset == 'mnist':
        train_dataset, val_dataset = misc.load_mnist(num_train=args.num_train, num_val=args.num_val, seed=dataset_seed)
        NetModel = MLP # MLP class from networks.py
        criterion = nn.CrossEntropyLoss()
        # net_params = {'num_features': 784, 'num_outputs': 10} # Example
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    print(f"Loaded {args.dataset.upper()} data: Train {len(train_dataset)}, Val {len(val_dataset)}")

    # --- Optimizer Configurations ---
    all_optimizer_configs_to_test = misc.get_optimizer_configs()
    
    if args.optimizer_key: # Filter if a specific optimizer base key is provided
        filtered_configs = [cfg for cfg in all_optimizer_configs_to_test if cfg['name'] == args.optimizer_key]
        if filtered_configs:
            all_optimizer_configs_to_test = filtered_configs
            print(f"Filtered to {len(all_optimizer_configs_to_test)} configurations for optimizer key '{args.optimizer_key}'.")
        else:
            print(f"Warning: Optimizer key '{args.optimizer_key}' not found. Running all {len(all_optimizer_configs_to_test)} configurations.")

    # --- Results Storage ---
    # This dictionary will store aggregated results (avg, std over inits) for each opt config
    aggregated_results_for_all_configs = {} 
    
    # Path for the final aggregated results .pkl file
    overall_results_dir = f'results_{args.dataset}/'
    os.makedirs(overall_results_dir, exist_ok=True)
    aggregated_pkl_filepath = os.path.join(overall_results_dir, f'all_optimizer_results_aggregated_{args.dataset}.pkl')

    # --- Main Experiment Loop ---
    if not args.only_plot:
        for opt_config_dict in all_optimizer_configs_to_test:
            detailed_opt_label = opt_config_dict['label']
            print(f"\n===== Running Experiment for: {detailed_opt_label} =====")

            # Store metrics from each initialization run for THIS optimizer config
            current_config_train_loss_histories = []
            current_config_val_loss_histories = []
            current_config_train_acc_histories = []
            current_config_val_acc_histories = []
            current_config_total_times_per_init = []

            for i_init in range(args.num_inits):
                print(f"--- Initialization {i_init+1}/{args.num_inits} for {detailed_opt_label} ---")
                
                # Set seed for this specific initialization run for model weights
                # Data splitting seed is handled once above.
                current_init_seed = (args.seed if args.seed is not None else np.random.randint(0, 10000)) + i_init
                torch.manual_seed(current_init_seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_init_seed)
                
                # Initialize Model (model __init__ moves it to device)
                # Pass dataset-specific params if your NetModel class requires them
                model_instance = NetModel() # e.g., NetModel(**net_params) if needed
                
                # Optimizer Instantiation
                opt_class = opt_config_dict['optimizer_class']
                opt_kwargs_for_class = opt_config_dict['kwargs']
                optimizer = opt_class(model_instance.parameters(), **opt_kwargs_for_class)
                
                # Optimizer Wrapping (if any)
                wrapper_type = opt_config_dict.get('wrapper_type')
                if wrapper_type:
                    wrapper_specific_kwargs = opt_config_dict.get('wrapper_kwargs', {})
                    optimizer = misc.wrap_optimizer(optimizer, wrapper_type, wrapper_specific_kwargs, 
                                                    task_name_for_logic=opt_config_dict.get('task_name_for_wrapper_logic'))
                
                # Learning Rate Scheduler
                lr_sched = None
                if args.lr_schedule_type == 'ReduceLROnPlateau':
                    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                                        patience=args.lr_patience, verbose=False)
                elif args.lr_schedule_type == 'StepLR':
                    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
                
                # --- Training & Validation for this init run ---
                run_start_time = time.time()
                train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = fit(
                    net_model=model_instance, 
                    train_data_loader=train_loader, 
                    val_data_loader=val_loader, 
                    optimizer_instance=optimizer,
                    criterion=criterion,
                    num_epochs_to_run=args.num_epochs, 
                    device_to_use=device,
                    lr_scheduler=lr_sched,
                    epoch_print_interval=args.verbose_interval
                )
                run_elapsed_time = time.time() - run_start_time
                print(f"Initialization {i_init+1} finished in {run_elapsed_time:.2f}s.")

                # Save raw history for this init run (optional, but good for deep dives)
                # misc.save_metric_history(train_loss_hist, args.dataset, detailed_opt_label, 'train_loss_raw', i_init)
                # ... (save other raw histories if needed) ...

                current_config_train_loss_histories.append(train_loss_hist)
                current_config_val_loss_histories.append(val_loss_hist)
                current_config_train_acc_histories.append(train_acc_hist)
                current_config_val_acc_histories.append(val_acc_hist)
                current_config_total_times_per_init.append(run_elapsed_time)

            # --- Aggregate results for the current optimizer configuration ---
            if current_config_train_loss_histories: # If any inits were successful
                # Helper to average/std over lists of lists (histories from inits)
                # Ensures all inner lists are same length (num_epochs) for np operations
                def aggregate_histories(list_of_histories, num_epochs_expected):
                    valid_histories = [h for h in list_of_histories if h is not None and len(h) == num_epochs_expected]
                    if not valid_histories: # No valid histories to aggregate
                        return np.array([]), np.array([]) 
                    try:
                        np_array_of_histories = np.array(valid_histories)
                        return np.mean(np_array_of_histories, axis=0), np.std(np_array_of_histories, axis=0)
                    except Exception as e:
                        print(f"Error during history aggregation: {e}. Histories: {list_of_histories}")
                        return np.array([]), np.array([])


                avg_train_loss, std_train_loss = aggregate_histories(current_config_train_loss_histories, args.num_epochs)
                avg_val_loss, std_val_loss = aggregate_histories(current_config_val_loss_histories, args.num_epochs)
                avg_train_acc, std_train_acc = aggregate_histories(current_config_train_acc_histories, args.num_epochs)
                avg_val_acc, std_val_acc = aggregate_histories(current_config_val_acc_histories, args.num_epochs)
                
                aggregated_results_for_all_configs[detailed_opt_label] = {
                    'avg_train_loss': avg_train_loss.tolist() if avg_train_loss.size > 0 else [],
                    'std_train_loss': std_train_loss.tolist() if std_train_loss.size > 0 else [],
                    'avg_val_loss': avg_val_loss.tolist() if avg_val_loss.size > 0 else [],
                    'std_val_loss': std_val_loss.tolist() if std_val_loss.size > 0 else [],
                    'avg_train_acc': avg_train_acc.tolist() if avg_train_acc.size > 0 else [],
                    'std_train_acc': std_train_acc.tolist() if std_train_acc.size > 0 else [],
                    'avg_val_acc': avg_val_acc.tolist() if avg_val_acc.size > 0 else [],
                    'std_val_acc': std_val_acc.tolist() if std_val_acc.size > 0 else [],
                    'avg_total_time': np.mean(current_config_total_times_per_init) if current_config_total_times_per_init else 0,
                    'std_total_time': np.std(current_config_total_times_per_init) if len(current_config_total_times_per_init) > 1 else 0,
                    'num_successful_inits': len([h for h in current_config_train_loss_histories if h is not None and len(h) == args.num_epochs])
                }
                print(f"Aggregated results for {detailed_opt_label} over {aggregated_results_for_all_configs[detailed_opt_label]['num_successful_inits']} inits.")
        
        # Save the dictionary containing all aggregated results
        if aggregated_results_for_all_configs:
            with open(aggregated_pkl_filepath, 'wb') as f:
                pkl.dump(aggregated_results_for_all_configs, f)
            print(f"\nSaved all aggregated results to: {aggregated_pkl_filepath}")
        else:
            print("\nNo training was performed or no results to save.")

    # --- Plotting and Summary ---
    if args.only_plot or aggregated_results_for_all_configs: # Plot if we just trained or if in only_plot mode
        if args.only_plot: # If only_plot, try to load the aggregated results
            try:
                with open(aggregated_pkl_filepath, 'rb') as f:
                    aggregated_results_for_all_configs = pkl.load(f)
                print(f"Loaded aggregated results from: {aggregated_pkl_filepath}")
            except FileNotFoundError:
                print(f"Error: Aggregated results file '{aggregated_pkl_filepath}' not found for plotting. Run training first.")
                return
            except Exception as e:
                print(f"Error loading aggregated results file: {e}")
                return
        
        if not aggregated_results_for_all_configs:
            print("No results available to plot or summarize.")
            return

        print("\n----- Generating Plots -----")
        misc.plot_all_metrics(
            aggregated_results_data_dict=aggregated_results_for_all_configs,
            num_epochs_completed=args.num_epochs, # All runs are for this many epochs
            dataset_name_str=args.dataset,
            plot_std_dev_bands=True # Enable error bands
        )

        print("\n----- Performance & Efficiency Summary -----")
        # Sort by final average validation accuracy (descending)
        # Handle cases where 'avg_val_acc' might be empty or not present for some configs
        def get_final_val_acc(metrics_dict):
            val_acc_hist = metrics_dict.get('avg_val_acc')
            return val_acc_hist[-1] if val_acc_hist and len(val_acc_hist) > 0 else -1.0 # Default to -1 if no acc data

        sorted_configs_by_perf = sorted(aggregated_results_for_all_configs.items(), 
                                        key=lambda item: get_final_val_acc(item[1]), 
                                        reverse=True)
        
        # Header for the summary table
        header = f"{'Optimizer Configuration':<70} | {'Final Val Acc':<18} | {'Final Val Loss':<18} | {'Avg Train Time (s)':<20} | {'Inits OK':<8}"
        print(header)
        print("-" * len(header))

        for opt_label, metrics in sorted_configs_by_perf:
            final_val_acc_str = "N/A"
            if metrics.get('avg_val_acc') and metrics['avg_val_acc']:
                final_val_acc_str = f"{metrics['avg_val_acc'][-1]:.4f} \u00B1 {metrics['std_val_acc'][-1]:.4f}"
            
            final_val_loss_str = "N/A"
            if metrics.get('avg_val_loss') and metrics['avg_val_loss']:
                final_val_loss_str = f"{metrics['avg_val_loss'][-1]:.4f} \u00B1 {metrics['std_val_loss'][-1]:.4f}"

            avg_time_str = f"{metrics.get('avg_total_time', 0):.2f} \u00B1 {metrics.get('std_total_time', 0):.2f}"
            inits_ok_str = str(metrics.get('num_successful_inits',0))

            print(f"{opt_label:<70} | {final_val_acc_str:<18} | {final_val_loss_str:<18} | {avg_time_str:<20} | {inits_ok_str:<8}")
    else:
         print("No results to plot (either training was skipped and no pkl file found, or no results were generated).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Learning Optimizer Comparison Framework")
    
    # Experiment Setup
    parser.add_argument('-dataset', type=str, default='cifar', choices=['cifar', 'mnist'], help="Dataset to use")
    parser.add_argument('-num_epochs', type=int, default=20, help="Number of training epochs per initialization") # Reduced for faster runs
    parser.add_argument('-num_inits', type=int, default=2, help="Number of random initializations per optimizer config") # Default 2 for speed
    parser.add_argument('-seed', type=int, default=None, help="Global random seed for reproducibility (for data splits, init seeds). If None, non-deterministic.")

    # Data Handling
    parser.add_argument('-num_train', type=int, default=5000, help="Number of training samples (subset)")
    parser.add_argument('-num_val', type=int, default=1000, help="Number of validation samples (subset)")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for training and validation") # Smaller batch for potentially smaller datasets
    parser.add_argument('-num_workers', type=int, default=2, help="Dataloader workers. Set to 0 if issues in Colab/Windows.")

    # LR Scheduling
    parser.add_argument('-lr_schedule_type', type=str, default=None, choices=['ReduceLROnPlateau', 'StepLR'], help="Type of LR scheduler")
    parser.add_argument('-lr_patience', type=int, default=3, help="Patience for ReduceLROnPlateau (epochs)") # Shorter patience
    parser.add_argument('-lr_step_size', type=int, default=7, help="Step size for StepLR (epochs)") # Adjusted step size

    # Filtering and Mode
    parser.add_argument('-optimizer_key', type=str, default=None, help="Run only a specific optimizer base key from optim_dict (e.g., 'adam', 'sgd')")
    parser.add_argument('-only_plot', action='store_true', help="Skip training, load aggregated results and plot")
    
    # Logging
    parser.add_argument('-verbose_interval', type=int, default=1, help="Print training progress every N epochs")

    cli_args = parser.parse_args()
    run_experiment(cli_args)