import tensorflow as tf
from tensorflow.core.util import event_pb2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter # For custom x-axis formatting

# Helper function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, smoothing_factor):
    """
    Calculates Exponential Moving Average (EMA) for a given data array.
    This matches TensorBoard's smoothing logic: s_t = smoothing_factor * s_{t-1} + (1 - smoothing_factor) * x_t.
    """
    if len(data) == 0: # 确保有数据可以平滑
        return np.array([])
    ema = np.zeros_like(data, dtype=np.float32)
    ema[0] = data[0] # Initialize with the first data point
    for i in range(1, len(data)):
        ema[i] = smoothing_factor * ema[i-1] + (1 - smoothing_factor) * data[i]
    return ema

# 将 window_size 参数替换为 smoothing_factor
def plot_multiple_tensorboard_rewards_smoothed(log_dirs_with_names, smoothing_factor=0.99, std_multiplier=1.0, custom_colors=None, reward_plot_height=4): # 添加 reward_plot_height 参数
    """
    Reads 'train_reward' scalar data from multiple TensorBoard log directories
    and plots them on a single graph using TensorBoard's EMA smoothing style
    with shaded confidence intervals based on smoothed average absolute deviation.
    Saves the plot to a file instead of displaying it.

    Args:
        log_dirs_with_names (list of tuples): A list where each tuple contains:
                                               (log_directory_path, display_name_for_plot)
        smoothing_factor (float): The EMA smoothing factor (e.g., 0.99 for strong smoothing).
                                  Matches TensorBoard's smoothing slider.
        std_multiplier (float): Multiplier for the smoothed average absolute deviation to determine the shade area.
                                E.g., 1.0 for +/- 1 * smoothed_MAD.
        custom_colors (list of str, optional): A list of color strings (e.g., 'blue', '#FF0000', 'green')
                                               to explicitly set colors for each run.
                                               If None, uses matplotlib's default colormap.
                                               The order of colors should match the order in log_dirs_with_names.
        reward_plot_height (float): The height of the reward plot figure (in inches). Lower value "flattens" the plot.
    """

    plt.figure(figsize=(10, reward_plot_height)) # 调整 figure height 参数
    
    # Define colors for the plots
    if custom_colors and len(custom_colors) >= len(log_dirs_with_names):
        # 如果提供了自定义颜色且数量足够，则使用自定义颜色
        colors = custom_colors
    else:
        # 否则使用 matplotlib 的 colormap
        print("Warning: Custom colors not provided or not enough colors. Using default colormap.")
        colors_map = plt.cm.get_cmap('tab10', len(log_dirs_with_names))
        colors = [colors_map(i) for i in range(len(log_dirs_with_names))]


    for idx, (log_dir, display_name) in enumerate(log_dirs_with_names):
        print(f"Processing '{display_name}' from directory: {log_dir}")
        
        train_rewards_steps = []
        train_rewards_values = []

        event_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.startswith('events.out.tfevents.')
        ]
        event_files.sort()

        if not event_files:
            print(f"Warning: No .tfevents files found in '{log_dir}'. Skipping this run.")
            continue

        for event_file in event_files:
            try: # 保持读取文件的鲁棒性，以防文件损坏
                for event_str in tf.compat.v1.io.tf_record_iterator(event_file):
                    event = event_pb2.Event.FromString(event_str)
                    
                    if event.step is not None and event.HasField('summary'):
                        for value in event.summary.value:
                            if value.tag.endswith('Avg_reward'):
                                if value.HasField('simple_value'):
                                    train_rewards_steps.append(event.step)
                                    train_rewards_values.append(value.simple_value)
                                    break
            except Exception as e:
                print(f"    Warning: Error reading event file '{event_file}': {e}. Skipping this file.")
                continue
        
        if train_rewards_steps:
            steps = np.array(train_rewards_steps)
            values = np.array(train_rewards_values)

            sort_indices = np.argsort(steps)
            steps = steps[sort_indices]
            values = values[sort_indices]

            if len(values) > 0:
                smoothed_values = calculate_ema(values, smoothing_factor)
                
                absolute_deviations = np.abs(values - smoothed_values)
                smoothed_deviation = calculate_ema(absolute_deviations, smoothing_factor)
                
                line_color = colors[idx] # 使用指定或分配的颜色
                plt.plot(steps, smoothed_values, label=display_name, color=line_color, linewidth=2)
                
                # Ensure lower bound of shade does not go below 0 (rewards can be negative, but visual might be cleaner >= 0)
                # For rewards, it's common to allow negative values, so I'll keep the shade potentially going below zero
                # if the smoothed_values allow it, unless you specifically want to clip it at 0.
                # If you want to clip rewards shade at 0, uncomment the next two lines and change fill_between.
                # lower_bound = np.maximum(0, smoothed_values - smoothed_deviation * std_multiplier)
                # plt.fill_between(steps, lower_bound, smoothed_values + smoothed_deviation * std_multiplier, color=line_color, alpha=0.15)

                plt.fill_between(
                    steps,
                    smoothed_values - smoothed_deviation * std_multiplier,
                    smoothed_values + smoothed_deviation * std_multiplier,
                    color=line_color,
                    alpha=0.15 # Semi-transparent
                )
            else:
                print(f"    Not enough data points for EMA smoothing in '{display_name}'. Skipping plot for this run.")

            print(f"    Collected {len(train_rewards_steps)} 'train_reward' points for '{display_name}'.")
        else:
            print(f"    No 'train_reward' data found in '{display_name}'.")

    # Customize the plot to match the desired style
    plt.title('Average Episode Reward', fontsize=16)
    plt.xlabel('Training step', fontsize=14)
    plt.ylabel('Average episode reward', fontsize=14)
    plt.grid(False)
    
    def millions_formatter(x, pos):
        return f'{x*1e-6:.1f}M'
    plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=12, loc='upper left')
    else:
        print("No reward data plotted. Legend will not be displayed.")

    plt.tight_layout()
    plt.savefig('plot/rewards.png', dpi=300) # Ensure it saves to 'plot' directory
    print("Plot saved to: plot/rewards.png")
    plt.close()

if __name__ == "__main__":
    # Ensure 'plot' directory exists for all generated images
    os.makedirs('plot', exist_ok=True)

    # --- Generate rewards.png with adjusted height ---
    my_log_configs = [
        ('expe/0629164427', 'reward(fan) = 1'),
        ('expe/0629191035', 'reward(fan) = fan'),
    ]

    # custom_plot_colors = ['#1f77b4', '#d62728', '#9467bd', '#8c564b', ] 
    custom_plot_colors = ['#8c564b', '#9467bd', ]

    print("\n--- Starting TensorBoard reward plotting process (EMA smooth style, adjusted height) ---")
    plot_multiple_tensorboard_rewards_smoothed(
        my_log_configs, 
        smoothing_factor=0.99, 
        std_multiplier=0, # std_multiplier for rewards can be adjusted if you want shade
        custom_colors=custom_plot_colors,
        reward_plot_height=3.5 # 调整此值以压扁 rewards 图，例如从 6 调整到 3.5 或更小
    )
    print("--- TensorBoard reward plotting process Completed ---")

    # --- Generate queue.png (assuming previous code is in a separate block/file or called here) ---
    # You need to ensure the plot_multiple_traffic_queues_smoothed_with_shade function is available
    # and called with its parameters. For demonstration, I'll put a placeholder call.
    # Replace this with your actual call to generate queue.png and delay.png
    # For example:
    # from your_queue_delay_plot_script import plot_multiple_traffic_queues_smoothed_with_shade, plot_multiple_traffic_delays_smoothed_with_shade

    # Assuming these functions are defined elsewhere or you will define them here
    # (they were provided in previous turns, just not in this snippet's __main__)
    # For completeness, I'll include placeholder calls to those functions with their latest states.

    # Placeholder for plot_multiple_traffic_queues_smoothed_with_shade
    # This part should be called from your existing queue plotting code
    # (Make sure to import/define plot_multiple_traffic_queues_smoothed_with_shade and plot_multiple_traffic_delays_smoothed_with_shade)
    # The 'plot_multiple_traffic_queues_smoothed_with_shade' function from earlier for 'queue.png'
    # needs to be present in your script or imported.
    # The 'plot_multiple_traffic_delays_smoothed_with_shade' for 'delay.png' also needs to be present.

    # Example of how you would call them after defining/importing:
    # plot_multiple_traffic_queues_smoothed_with_shade(
    #     my_traffic_log_configs,
    #     smoothing_factor=0.9,
    #     std_multiplier=10.0,
    #     custom_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # )
    # plot_multiple_traffic_delays_smoothed_with_shade(
    #     my_traffic_log_configs,
    #     smoothing_factor=0.9,
    #     std_multiplier=10.0,
    #     custom_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # )

