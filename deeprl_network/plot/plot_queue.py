import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Helper function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, smoothing_factor):
    """
    Calculates Exponential Moving Average (EMA) for a given data array.
    This matches TensorBoard's smoothing logic: s_t = smoothing_factor * s_{t-1} + (1 - smoothing_factor) * x_t.
    """
    if len(data) == 0:
        return np.array([])
    ema = np.zeros_like(data, dtype=np.float32)
    ema[0] = data[0] # Initialize with the first data point
    for i in range(1, len(data)):
        ema[i] = smoothing_factor * ema[i-1] + (1 - smoothing_factor) * data[i]
    return ema

def plot_multiple_traffic_queues_smoothed_with_shade(log_dirs_with_names, smoothing_factor=0.9, std_multiplier=10.0, custom_colors=None):
    """
    Reads files ending with 'traffic.csv' from multiple log directories,
    extracts 'step' (first column) and 'avg_queue' (last column) data.
    Applies EMA smoothing to the main curve and uses smoothed average absolute deviation
    from the raw data to plot shaded regions, mimicking TensorBoard's style.
    Ensures the lower bound of the shaded region does not go below zero.
    The plot title is removed.

    Args:
        log_dirs_with_names (list of tuples): A list where each tuple contains:
                                               (log_directory_path, display_name_for_plot)
        smoothing_factor (float): The EMA smoothing factor for the queue data and its deviation.
                                  A value closer to 1.0 means more smoothing. (e.g., 0.9 for strong smoothing).
        std_multiplier (float): Multiplier for the smoothed average absolute deviation to determine the shade area.
                                E.g., 1.0 for +/- 1 * smoothed_MAD. Increased default for debugging visibility.
        custom_colors (list of str, optional): A list of color strings (e.g., 'blue', '#FF0000', 'green')
                                               to explicitly set colors for each run.
                                               If None, uses matplotlib's default colormap.
                                               The order of colors should match the order in log_dirs_with_names.
    """

    plt.figure(figsize=(10, 6))
    
    if custom_colors and len(custom_colors) >= len(log_dirs_with_names):
        colors = custom_colors
    else:
        print("Warning: Custom colors not provided or not enough colors. Using default colormap.")
        colors_map = plt.cm.get_cmap('tab10', len(log_dirs_with_names))
        colors = [colors_map(i) for i in range(len(log_dirs_with_names))]

    for idx, (log_dir, display_name) in enumerate(log_dirs_with_names):
        print(f"Processing '{display_name}' from directory: {log_dir}")
        
        found_traffic_file = None
        for fname in os.listdir(log_dir):
            if fname.endswith('traffic.csv'):
                found_traffic_file = os.path.join(log_dir, fname)
                break

        if not found_traffic_file:
            print(f"Warning: No file ending with 'traffic.csv' found in '{log_dir}'. Skipping this run.")
            continue

        try:
            df = pd.read_csv(found_traffic_file)
            print(f"  Reading data from: {found_traffic_file}")
            
            if df.shape[1] < 2:
                print(f"Warning: '{found_traffic_file}' does not have enough columns. Skipping.")
                continue

            steps = df.iloc[:, 0].values
            avg_queues = df.iloc[:, -1].values # Still using last column for avg_queue

            if len(steps) == 0 or len(avg_queues) == 0:
                print(f"Warning: No data found in '{found_traffic_file}'. Skipping.")
                continue

            if len(avg_queues) > 0:
                smoothed_queues = calculate_ema(avg_queues, smoothing_factor) 
                
                absolute_deviations = np.abs(avg_queues - smoothed_queues)
                smoothed_deviation = calculate_ema(absolute_deviations, smoothing_factor)
                
                # --- Debugging info for shade width ---
                print(f"  For '{display_name}': Smoothed Deviation - Min: {np.min(smoothed_deviation):.4f}, Max: {np.max(smoothed_deviation):.4f}, Mean: {np.mean(smoothed_deviation):.4f}")
                print(f"  Effective Shade Width (at max): {np.max(smoothed_deviation) * std_multiplier:.4f}")
                
                line_color = colors[idx]
                
                plt.plot(steps, smoothed_queues, label=display_name, color=line_color, linewidth=2)
                
                # Ensure lower bound of shade does not go below 0
                lower_bound = np.maximum(0, smoothed_queues - smoothed_deviation * std_multiplier)
                upper_bound = smoothed_queues + smoothed_deviation * std_multiplier

                plt.fill_between(
                    steps,
                    lower_bound, # Modified here
                    upper_bound,
                    color=line_color,
                    alpha=0.15 
                )
            else:
                print(f"  Not enough data points for EMA smoothing in '{display_name}'. Skipping plot for this run.")

            print(f"  Collected {len(steps)} data points for '{display_name}'.")

        except Exception as e:
            print(f"Error reading '{found_traffic_file}': {e}. Skipping this run.")
            continue

    # Removed plt.title() as requested
    plt.xlabel('Simulation time (sec)', fontsize=14)
    plt.ylabel('Average queue length (veh)', fontsize=14) # Stays avg_queue
    plt.grid(True, linestyle=':', alpha=0.6) 
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=12, loc='upper left') 
    else:
        print("No traffic queue data plotted. Legend will not be displayed.")

    plt.tight_layout()
    
    output_dir = 'plot'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'queue.png') # Stays queue.png
    
    plt.savefig(save_path, dpi=300) 
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    my_traffic_log_configs = [
        # ('expe/0604_1823_eval', 'MA2C baseline'),
        ('expe/0604_1841_eval', 'Optimized MA2C baseline'),
        # ('expe/0604_2210_eval', 'IC3Net'), 
        # ('expe/0604_2215_eval', 'MAPPO'), 
        ('expe/0604_2250_eval', 'Hybrid')
    ]

    custom_traffic_plot_colors = ['#8c564b', '#9467bd'] 
    # '#9467bd', '#8c564b',
    
    print("\n--- Starting Traffic Queue Plotting Process (Smoothed with Shade, Lower Bound >= 0, No Title) ---")
    plot_multiple_traffic_queues_smoothed_with_shade(
        my_traffic_log_configs, 
        smoothing_factor=0.95, 
        std_multiplier=0.0, 
        custom_colors=custom_traffic_plot_colors
    )
    print("--- Traffic Queue Plotting Process Completed ---")