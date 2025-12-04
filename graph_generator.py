import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from itertools import product

def analyze_csv_parameters(csv_file, parameters, outputs):
    """
    Analyzes CSV data by plotting relationships between parameters and outputs.
    
    Args:
        csv_file: Path to the CSV file
        parameters: List of parameter column names to analyze
        outputs: List of output column names to plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Validate that all parameters and outputs exist in the dataframe
    missing_params = [p for p in parameters if p not in df.columns]
    missing_outputs = [o for o in outputs if o not in df.columns]
    
    if missing_params:
        raise ValueError(f"Parameters not found in CSV: {missing_params}")
    if missing_outputs:
        raise ValueError(f"Outputs not found in CSV: {missing_outputs}")
    
    # Create main output directory
    output_dir = Path("graphs_out")
    output_dir.mkdir(exist_ok=True)
    
    # For each parameter
    for param in parameters:
        print(f"\nAnalyzing parameter: {param}")
        
        # Get other parameters (the ones to freeze)
        other_params = [p for p in parameters if p != param]
        
        # Create folder for this parameter
        param_folder_path = output_dir / param
        param_folder_path.mkdir(exist_ok=True)
        
        # For each output
        for output in outputs:
            print(f"  - Plotting against output: {output}")
            
            # Get unique values for the current parameter
            param_values = sorted(df[param].unique())
            
            # Dictionary to store all curves for averaging
            all_curves = []
            
            # Get all unique values for other parameters
            other_param_values = {p: sorted(df[p].unique()) for p in other_params}
            
            # Generate all combinations of other parameter values
            if other_params:
                combinations = list(product(*[other_param_values[p] for p in other_params]))
            else:
                combinations = [()]  # Single empty combination if no other params
            
            # For each combination of frozen parameter values
            for combo in combinations:
                # Create filter condition
                mask = pd.Series([True] * len(df))
                
                for i, other_param in enumerate(other_params):
                    mask &= (df[other_param] == combo[i])
                
                # Filter data
                filtered_df = df[mask]
                
                if len(filtered_df) > 0:
                    # Sort by the current parameter
                    filtered_df = filtered_df.sort_values(by=param)
                    
                    # Store the curve (param values and corresponding output values)
                    curve_x = filtered_df[param].values
                    curve_y = filtered_df[output].values
                    
                    if len(curve_x) > 0:
                        all_curves.append((curve_x, curve_y))
            
            # Average all curves
            if all_curves:
                # Create a common x-axis (all unique parameter values)
                common_x = np.array(param_values)
                
                # For each x value, collect all corresponding y values and average them
                averaged_y = []
                
                for x_val in common_x:
                    y_values = []
                    for curve_x, curve_y in all_curves:
                        # Find if this x value exists in this curve
                        indices = np.where(curve_x == x_val)[0]
                        if len(indices) > 0:
                            y_values.append(curve_y[indices[0]])
                    
                    if y_values:
                        averaged_y.append(np.mean(y_values))
                    else:
                        averaged_y.append(np.nan)
                
                averaged_y = np.array(averaged_y)
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(averaged_y)
                plot_x = common_x[valid_mask]
                plot_y = averaged_y[valid_mask]
                
                if len(plot_x) > 0:
                    plt.plot(plot_x, plot_y, 'b-o', linewidth=2, markersize=6)
                    plt.xlabel(param, fontsize=12, fontweight='bold')
                    plt.ylabel(output, fontsize=12, fontweight='bold')
                    plt.title(f'{output} vs {param}\n(Averaged over all combinations of other parameters)', 
                             fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save the plot
                    output_file = param_folder_path / f"{param}_vs_{output}.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"    Saved: {output_file}")
                else:
                    print(f"    No valid data points to plot")
            else:
                print(f"    No curves generated for this combination")
    
    print(f"\nAnalysis complete! Results saved in '{output_dir}' directory")


# Example usage
if __name__ == "__main__":
    # Define your CSV file path
    csv_file = r"results\dataset2_Faculty_entering\parameter_sweep_20251204_044703.csv"  # Replace with your CSV file path
    
    # Define parameters (independent variables)
    parameters = ["compression_rank", "threshold"]
    
    # Define outputs (dependent variables)
    outputs = [
        "reduction_percentage",
        "avg_fps",
        "time_saved_percent",
        "cpu_saved_percent",
        "memory_saved_percent"
    ]
    
    # Run the analysis
    analyze_csv_parameters(csv_file, parameters, outputs)