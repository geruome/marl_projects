import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

def plot_tensorboard_subplots_optimized(log_file_path, tags, ema_alpha=0.99):
    """
    Reads specified scalar tags from a TensorBoard event file, applies EMA,
    and plots them in a 2x2 subplot grid on a single figure with optimized y-axis limits.

    Args:
        log_file_path (str): Path to the TensorBoard event file.
        tags (list): A list of strings, where each string is a scalar tag to plot.
                     Expected to be 4 tags for a 2x2 layout.
        ema_alpha (float): Alpha value for Exponential Moving Average (EMA) smoothing.
                           Higher value means more smoothing.
    """
    if len(tags) != 4:
        print("Warning: Expected exactly 4 tags for a 2x2 layout. Adjusting layout for current tags count.")
        # Fallback to dynamic subplot creation if not 4 tags
        num_rows = (len(tags) + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(16, 5 * num_rows))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # 2x2 layout, adjusted figure size
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    data = {tag: [] for tag in tags}
    steps = {tag: [] for tag in tags}

    print(f"Reading data from: {log_file_path}")
    for event in summary_iterator(log_file_path):
        if event.summary:
            for value in event.summary.value:
                if value.tag in tags:
                    # Using .get() with default [] and then append is safer
                    # ensuring the list is initialized if a tag appears out of order
                    data[value.tag].append(value.simple_value)
                    steps[value.tag].append(event.step)

    # Define specific y-axis limits for each tag
    y_limits = {
        "item/clip_frac": [0, 0.02],
        "item/train_reward": [-750, -100], # Keep current range as it seems reasonable from image
        "loss/nc_policy_loss": [-0.08, 0],
        "loss/nc_value_loss": [0, 50] # Adjusted to [0, 50]
    }
    
    # Define plotting colors (optional, but can make it visually cleaner)
    plot_color = '#1f77b4' # A standard matplotlib blue, or '#556B7D' from previous if preferred

    for i, tag in enumerate(tags):
        ax = axes[i] # Get the current subplot axis
        
        if not data.get(tag) or not steps.get(tag):
            print(f"No data found for tag: {tag}")
            ax.set_title(tag)
            ax.grid(True)
            continue

        series = pd.Series(data.get(tag))
        smoothed_data = series.ewm(alpha=1 - ema_alpha).mean()

        ax.plot(steps.get(tag), smoothed_data, label=f'EMA({ema_alpha}) {tag}', color=plot_color)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(tag)
        ax.grid(True, which="both", linestyle='-', alpha=0.7)
        ax.legend()

        # Apply specific y-axis limits
        if tag in y_limits:
            ax.set_ylim(y_limits[tag])

    plt.tight_layout()

    # Save the figure as requested
    output_dir = 'plot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'PPO.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'PPO.png')}")
    # plt.show() # Uncomment if you also want to display the plot

if __name__ == "__main__":
    # --- Configuration ---
    # REPLACE THIS with the actual path to your TensorBoard event file
    my_log_file = "merged_tb_runs/PPO/events.out.tfevents.1749038588.ide-g50-u2-937b6a.27502.0.v2" 

    # List of tags to plot. Ensure these tags match the ones in your TensorBoard log.
    tags_to_plot = [
        "item/clip_frac",
        "item/train_reward",
        "loss/nc_policy_loss",
        "loss/nc_value_loss"
    ]

    ema_smoothing_alpha = 0.99

    # --- Run the plotting function ---
    if os.path.exists(my_log_file):
        plot_tensorboard_subplots_optimized(my_log_file, tags_to_plot, ema_smoothing_alpha)
    else:
        print(f"Error: Log file not found at {my_log_file}")
        print("Please update 'my_log_file' to your actual TensorBoard event file path.")