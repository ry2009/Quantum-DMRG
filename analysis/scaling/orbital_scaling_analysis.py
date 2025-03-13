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

# Define the output directory
OUTPUT_DIR = "orbital_scaling_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the system sizes to test (number of orbitals)
# You can customize these based on your available data
SYSTEM_SIZES = [22, 26, 32, 40, 44]  # Example sizes in increasing order

# Define output directories for each system size
OUTPUT_DIRS = [f"delta_ml_results_{size}" for size in SYSTEM_SIZES]

# Dictionary to store timing data
timing_data = {
    'system_sizes': [],
    'training_times': [],
    'inference_times': [],
    'epoch_times': [],
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

def run_experiment(system_size, output_dir):
    """Run the advanced_delta_ml_model.py script with a specific system size and collect timing data."""
    print(f"\n{'='*80}")
    print(f"Running experiment for system size: {system_size} orbitals")
    print(f"{'='*80}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the model script (you may need to adjust this command based on your setup)
    # The script should save timing data to the output directory
    cmd = [
        "python", "advanced_delta_ml_model.py",
        "--system_size", str(system_size),
        "--output_dir", output_dir,
        "--max_epochs", "50",  # Reduced epochs for faster experiments
        "--early_stopping", "10"
    ]
    
    # Print the command being run
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    # Check if timing data was saved
    timing_file = os.path.join(output_dir, "timing_data.json")
    if os.path.exists(timing_file):
        with open(timing_file, "r") as f:
            run_timing_data = json.load(f)
        
        # Extract relevant timing data
        if run_timing_data["system_sizes"] and run_timing_data["training_times"]:
            actual_system_size = run_timing_data["system_sizes"][0]  # Use actual measured size
            timing_data["system_sizes"].append(actual_system_size)
            timing_data["training_times"].append(run_timing_data["total_training_time"])
            timing_data["inference_times"].append(run_timing_data["total_inference_time"])
            
            # Also save average epoch time
            if run_timing_data["epoch_times"]:
                avg_epoch_time = np.mean(run_timing_data["epoch_times"])
                timing_data["epoch_times"].append(avg_epoch_time)
            
            print(f"Collected timing data for system size {actual_system_size}")
            return True
    
    print(f"No timing data found for system size {system_size}")
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
    
    # Save raw data to CSV
    with open(os.path.join(OUTPUT_DIR, "scaling_data.csv"), "w") as f:
        f.write("System_Size,Training_Time,Inference_Time\n")
        for size, train_time, inf_time in zip(system_sizes, training_times, inference_times):
            f.write(f"{size},{train_time},{inf_time}\n")
    
    # Fit different models to training time data
    models = [
        {"name": "Linear", "func": linear_fit, "color": "red"},
        {"name": "Quadratic", "func": quadratic_fit, "color": "green"},
        {"name": "Cubic", "func": cubic_fit, "color": "blue"},
        {"name": "Exponential", "func": exponential_fit, "color": "purple"},
        {"name": "Power", "func": power_fit, "color": "orange"}
    ]
    
    best_fit = {"name": None, "rmse": float('inf'), "params": None, "func": None}
    
    # Create plot for training time scaling
    plt.figure(figsize=(12, 8))
    plt.scatter(system_sizes, training_times, s=100, color='black', label='Measured data')
    
    # Try fitting different models
    for model in models:
        try:
            params, _ = curve_fit(model["func"], system_sizes, training_times, maxfev=10000)
            
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
        y_fit = quadratic_fit(x_fit, *params[0], *params[1:])
        plt.plot(x_fit, y_fit, '-', color='blue', label='Quadratic fit')
    except Exception:
        pass
    
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
    elif best_fit["name"] == "Exponential":
        a, b, c = best_fit["params"]
        print(f"Scaling as: {a:.6f} * exp({b:.6f} * N) + {c:.6f}")
        print(f"This suggests an exponential scaling")
    elif best_fit["name"] == "Power":
        a, b, c = best_fit["params"]
        print(f"Scaling as: {a:.6f} * N^{b:.6f} + {c:.6f}")
        print(f"This suggests an O(N^{b:.2f}) scaling")

def main():
    print("Starting orbital scaling analysis...")
    
    for idx, (system_size, output_dir) in enumerate(zip(SYSTEM_SIZES, OUTPUT_DIRS)):
        print(f"\nRunning experiment {idx+1}/{len(SYSTEM_SIZES)}: {system_size} orbitals")
        success = run_experiment(system_size, output_dir)
        
        if not success:
            print(f"Warning: Failed to collect data for system size {system_size}")
    
    # Save collected timing data
    with open(os.path.join(OUTPUT_DIR, "collected_timing_data.json"), "w") as f:
        json.dump(timing_data, f, indent=2)
    
    # Analyze the scaling
    analyze_scaling()

if __name__ == "__main__":
    main() 