#!/usr/bin/env python
"""
Script to analyze how training and inference time scales with system size (number of orbitals).
This script will run the advanced_delta_ml_model.py script multiple times with different system sizes 
and collect timing data.
"""

import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import pandas as pd

# Define the output directory
OUTPUT_DIR = "orbital_scaling_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the system sizes to test (from our dataset analysis)
SYSTEM_SIZES = [22, 26, 32, 40, 44]

# Dictionary to store timing data
timing_data = {
    'system_sizes': [],           # Number of orbitals
    'training_times': [],         # Training time in seconds
    'inference_times': [],        # Inference time in seconds
    'epoch_times': [],            # Average epoch time
    'train_examples': [],         # Number of training examples used
    'test_examples': [],          # Number of test examples used
    'test_rmse': [],              # Test RMSE
    'test_mae': [],               # Test MAE
    'test_r2': [],                # Test R²
    'test_rel_error': []          # Test relative error
}

# Define functions for curve fitting
def linear_fit(x, a, b):
    """Linear function: a*x + b"""
    return a * x + b

def quadratic_fit(x, a, b, c):
    """Quadratic function: a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def cubic_fit(x, a, b, c, d):
    """Cubic function: a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def exponential_fit(x, a, b, c):
    """Exponential function: a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def power_fit(x, a, b, c):
    """Power function: a * x^b + c"""
    return a * np.power(x, b) + c

def run_experiment(system_size, results_dir):
    """Run the advanced_delta_ml_model.py script with a specific system size and collect timing data."""
    print(f"\n{'='*80}")
    print(f"Running experiment for system size: {system_size} orbitals")
    print(f"{'='*80}")
    
    # Run the model script
    cmd = [
        "python", "advanced_delta_ml_model.py",
        "--system_size", str(system_size),
        "--output_dir", results_dir,
        "--max_epochs", "30",  # Reduced for testing
        "--early_stopping", "5"  # Reduced for testing
    ]
    
    # Print the command being run
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        total_time = time.time() - start_time
        
        print(f"Command completed in {total_time:.2f} seconds")
        print("Output:")
        
        # Extract key information from the output
        output_lines = result.stdout.splitlines()
        
        # Training dataset size
        train_examples = 0
        for line in output_lines:
            if "Filtered training dataset:" in line:
                train_examples = int(line.split(":")[-1].strip().split()[0])
                break
        
        # Test dataset size
        test_examples = 0
        for line in output_lines:
            if "Filtered test dataset:" in line:
                test_examples = int(line.split(":")[-1].strip().split()[0])
                break
        
        # Test metrics
        test_rmse = None
        test_mae = None
        test_r2 = None
        test_rel_error = None
        
        for line in output_lines:
            if "Test RMSE:" in line and "mHa" in line:
                test_rmse = float(line.split("(")[-1].split()[0])  # Extract mHa value
            elif "Test MAE:" in line and "mHa" in line:
                test_mae = float(line.split("(")[-1].split()[0])  # Extract mHa value
            elif "Test R²:" in line:
                test_r2 = float(line.split(":")[-1].strip())
            elif "Test Relative Error:" in line:
                test_rel_error = float(line.split(":")[-1].strip().replace("%", ""))
        
        # Check if timing data was saved
        timing_file = os.path.join(results_dir, "timing_data.json")
        if os.path.exists(timing_file):
            with open(timing_file, "r") as f:
                run_timing_data = json.load(f)
            
            # Add data to our collection
            timing_data["system_sizes"].append(system_size)
            timing_data["training_times"].append(run_timing_data["total_training_time"])
            timing_data["inference_times"].append(run_timing_data["total_inference_time"])
            
            # Average epoch time
            if run_timing_data["epoch_times"]:
                avg_epoch_time = np.mean(run_timing_data["epoch_times"])
                timing_data["epoch_times"].append(avg_epoch_time)
            
            # Add the other metrics
            timing_data["train_examples"].append(train_examples)
            timing_data["test_examples"].append(test_examples)
            timing_data["test_rmse"].append(test_rmse)
            timing_data["test_mae"].append(test_mae)
            timing_data["test_r2"].append(test_r2)
            timing_data["test_rel_error"].append(test_rel_error)
            
            print(f"Successfully collected timing data for system size {system_size}")
            return True
        
        print(f"No timing data found for system size {system_size}")
        return False
    
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def analyze_scaling():
    """Analyze the collected timing data and determine scaling with system size."""
    if not timing_data["system_sizes"]:
        print("No timing data collected. Cannot perform analysis.")
        return
    
    # Convert lists to numpy arrays for analysis
    system_sizes = np.array(timing_data["system_sizes"])
    training_times = np.array(timing_data["training_times"])
    inference_times = np.array(timing_data["inference_times"])
    
    # Ensure we have the correct data order (ascending system sizes)
    sort_idx = np.argsort(system_sizes)
    system_sizes = system_sizes[sort_idx]
    training_times = training_times[sort_idx]
    inference_times = inference_times[sort_idx]
    
    # Save raw data to CSV
    df = pd.DataFrame({
        "System_Size": system_sizes,
        "Training_Time": training_times,
        "Inference_Time": inference_times,
        "Train_Examples": np.array(timing_data["train_examples"])[sort_idx],
        "Test_Examples": np.array(timing_data["test_examples"])[sort_idx],
        "Test_RMSE_mHa": np.array(timing_data["test_rmse"])[sort_idx],
        "Test_MAE_mHa": np.array(timing_data["test_mae"])[sort_idx],
        "Test_R2": np.array(timing_data["test_r2"])[sort_idx],
        "Test_Rel_Error_Percent": np.array(timing_data["test_rel_error"])[sort_idx]
    })
    
    # Save to CSV
    df.to_csv(os.path.join(OUTPUT_DIR, "scaling_data.csv"), index=False)
    print(f"Saved raw data to {os.path.join(OUTPUT_DIR, 'scaling_data.csv')}")
    
    # Display data table
    print("\nSummary of collected data:")
    print(df.to_string(index=False))
    
    # Fit different models to training time data
    models = [
        {"name": "Linear", "func": linear_fit, "color": "red"},
        {"name": "Quadratic", "func": quadratic_fit, "color": "green"},
        {"name": "Cubic", "func": cubic_fit, "color": "blue"},
        {"name": "Power", "func": power_fit, "color": "orange"}
    ]
    
    best_fit = {"name": None, "rmse": float('inf'), "params": None, "func": None}
    
    # Create plot for training time scaling
    plt.figure(figsize=(12, 8))
    plt.scatter(system_sizes, training_times, s=100, color='black', label='Measured data')
    
    # Try fitting different models
    for model in models:
        try:
            params, _ = curve_fit(model["func"], system_sizes, training_times)
            
            # Generate fitted curve
            x_fit = np.linspace(min(system_sizes) * 0.9, max(system_sizes) * 1.5, 1000)
            y_fit = model["func"](x_fit, *params)
            
            # Plot fitted curve
            plt.plot(x_fit, y_fit, '-', color=model["color"], label=f'{model["name"]} fit')
            
            # Calculate RMSE of fit
            y_pred = model["func"](system_sizes, *params)
            rmse = np.sqrt(np.mean((y_pred - training_times)**2))
            
            print(f"{model['name']} fit parameters: {params}")
            print(f"{model['name']} fit RMSE: {rmse}")
            
            # Check if this is the best fit so far
            if rmse < best_fit["rmse"]:
                best_fit["name"] = model["name"]
                best_fit["rmse"] = rmse
                best_fit["params"] = params
                best_fit["func"] = model["func"]
        except Exception as e:
            print(f"Error fitting {model['name']} model: {e}")
    
    # Format the plot
    plt.xlabel('Number of Orbitals', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.title(f'Scaling of Training Time with System Size\nBest fit: {best_fit["name"]} (RMSE: {best_fit["rmse"]:.2f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_time_scaling.png"), dpi=300)
    
    # Estimate maximum feasible system size
    if best_fit["name"]:
        # Assuming a 24-hour training time limit
        max_time = 24 * 60 * 60  # 24 hours in seconds
        
        # Create a function to find the root of (fit_func(x) - max_time)
        def time_limit_func(x):
            return best_fit["func"](x, *best_fit["params"]) - max_time
        
        # Try to find the root using interpolation
        try:
            # Generate x values beyond our current range
            x_extended = np.linspace(max(system_sizes), max(system_sizes) * 10, 10000)
            y_extended = best_fit["func"](x_extended, *best_fit["params"])
            
            # Find where y crosses the max_time threshold
            idx = np.argmin(np.abs(y_extended - max_time))
            max_feasible_size = x_extended[idx]
            
            print(f"\nEstimated maximum feasible system size: {int(max_feasible_size)} orbitals")
            print(f"(Based on 24-hour training time limit)")
            
            # Save this estimate
            with open(os.path.join(OUTPUT_DIR, "max_feasible_size.txt"), "w") as f:
                f.write(f"Estimated maximum feasible system size: {int(max_feasible_size)} orbitals\n")
                f.write(f"Based on 24-hour training time limit and {best_fit['name']} scaling model\n")
                f.write(f"Model parameters: {best_fit['params']}\n")
                
                # Add more details about the scaling function
                if best_fit["name"] == "Linear":
                    a, b = best_fit["params"]
                    f.write(f"Scaling as: {a:.6f} * N + {b:.6f}\n")
                    f.write(f"This suggests an O(N) scaling\n")
                elif best_fit["name"] == "Quadratic":
                    a, b, c = best_fit["params"]
                    f.write(f"Scaling as: {a:.6f} * N² + {b:.6f} * N + {c:.6f}\n")
                    f.write(f"This suggests an O(N²) scaling\n")
                elif best_fit["name"] == "Cubic":
                    a, b, c, d = best_fit["params"]
                    f.write(f"Scaling as: {a:.6f} * N³ + {b:.6f} * N² + {c:.6f} * N + {d:.6f}\n")
                    f.write(f"This suggests an O(N³) scaling\n")
                elif best_fit["name"] == "Power":
                    a, b, c = best_fit["params"]
                    f.write(f"Scaling as: {a:.6f} * N^{b:.6f} + {c:.6f}\n")
                    f.write(f"This suggests an O(N^{b:.2f}) scaling\n")
        except Exception as e:
            print(f"Error estimating maximum feasible size: {e}")
    
    # Also create a plot for inference time
    plt.figure(figsize=(12, 8))
    plt.scatter(system_sizes, inference_times, s=100, color='black', label='Measured data')
    
    # Try fitting a model to inference time
    try:
        params, _ = curve_fit(quadratic_fit, system_sizes, inference_times)
        
        # Generate fitted curve
        x_fit = np.linspace(min(system_sizes) * 0.9, max(system_sizes) * 1.5, 1000)
        y_fit = quadratic_fit(x_fit, *params)
        
        # Plot fitted curve
        plt.plot(x_fit, y_fit, '-', color='red', label='Quadratic fit')
    except Exception as e:
        print(f"Error fitting inference time data: {e}")
    
    # Format the plot
    plt.xlabel('Number of Orbitals', fontsize=14)
    plt.ylabel('Inference Time (seconds)', fontsize=14)
    plt.title('Scaling of Inference Time with System Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "inference_time_scaling.png"), dpi=300)
    
    # Create summary plots for test metrics vs orbital count
    plt.figure(figsize=(12, 8))
    plt.scatter(system_sizes, np.array(timing_data["test_rmse"])[sort_idx], s=100, color='blue', label='Test RMSE (mHa)')
    plt.scatter(system_sizes, np.array(timing_data["test_mae"])[sort_idx], s=100, color='green', label='Test MAE (mHa)')
    plt.xlabel('Number of Orbitals', fontsize=14)
    plt.ylabel('Error (mHa)', fontsize=14)
    plt.title('Test Error vs System Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_metrics_scaling.png"), dpi=300)
    
    # Create a comprehensive summary plot
    plt.figure(figsize=(15, 10))
    
    # Training time subplot
    plt.subplot(2, 1, 1)
    plt.scatter(system_sizes, training_times, s=100, color='black', label='Training time')
    
    # If we have a best fit model, plot it
    if best_fit["name"]:
        x_fit = np.linspace(min(system_sizes) * 0.9, max(system_sizes) * 1.5, 1000)
        y_fit = best_fit["func"](x_fit, *best_fit["params"])
        plt.plot(x_fit, y_fit, '-', color='red', label=f'{best_fit["name"]} fit')
        
        # If we found a max feasible size, show it
        if 'max_feasible_size' in locals():
            plt.axvline(x=max_feasible_size, color='green', linestyle='--', 
                       label=f'Max feasible: {int(max_feasible_size)} orbitals')
    
    plt.xlabel('Number of Orbitals', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.title('Scaling of Training Time with System Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Inference time subplot
    plt.subplot(2, 1, 2)
    plt.scatter(system_sizes, inference_times, s=100, color='black', label='Inference time')
    
    # Try to fit a model
    try:
        params, _ = curve_fit(quadratic_fit, system_sizes, inference_times)
        x_fit = np.linspace(min(system_sizes) * 0.9, max(system_sizes) * 1.5, 1000)
        y_fit = quadratic_fit(x_fit, *params)
        plt.plot(x_fit, y_fit, '-', color='blue', label='Quadratic fit')
    except Exception as e:
        print(f"Error fitting quadratic model to inference time: {e}")
    
    plt.xlabel('Number of Orbitals', fontsize=14)
    plt.ylabel('Inference Time (seconds)', fontsize=14)
    plt.title('Scaling of Inference Time with System Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "complete_scaling_analysis.png"), dpi=300)
    
    # Print scaling summary
    print("\nScaling Analysis Summary:")
    print(f"Best fit model for training time: {best_fit['name']}")
    
    if best_fit["name"] == "Linear":
        a, b = best_fit["params"]
        print(f"Scaling as: {a:.6f} * N + {b:.6f}")
        print(f"This suggests an O(N) scaling")
    elif best_fit["name"] == "Quadratic":
        a, b, c = best_fit["params"]
        print(f"Scaling as: {a:.6f} * N² + {b:.6f} * N + {c:.6f}")
        print(f"This suggests an O(N²) scaling")
    elif best_fit["name"] == "Cubic":
        a, b, c, d = best_fit["params"]
        print(f"Scaling as: {a:.6f} * N³ + {b:.6f} * N² + {c:.6f} * N + {d:.6f}")
        print(f"This suggests an O(N³) scaling")
    elif best_fit["name"] == "Power":
        a, b, c = best_fit["params"]
        print(f"Scaling as: {a:.6f} * N^{b:.6f} + {c:.6f}")
        print(f"This suggests an O(N^{b:.2f}) scaling")

def main():
    """Main function to run experiments for different system sizes."""
    print("Starting orbital scaling analysis...")
    
    # Create result directories for each system size
    result_dirs = {size: f"delta_ml_results_{size}" for size in SYSTEM_SIZES}
    
    for system_size, results_dir in result_dirs.items():
        # Create the directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Run the experiment
        success = run_experiment(system_size, results_dir)
        if not success:
            print(f"Warning: Failed to collect data for system size {system_size}")
    
    # Save collected timing data
    with open(os.path.join(OUTPUT_DIR, "collected_timing_data.json"), "w") as f:
        json.dump(timing_data, f, indent=2)
    
    # Analyze the scaling
    analyze_scaling()

if __name__ == "__main__":
    main() 