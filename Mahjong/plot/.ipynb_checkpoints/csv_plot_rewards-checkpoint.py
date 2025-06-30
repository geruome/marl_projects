import pandas as pd # Import pandas for CSV reading
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os

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

# 修改函数以从 CSV 文件读取数据，只平滑，不显示标准差阴影
def plot_multiple_csv_rewards_smoothed(csv_paths_with_names, smoothing_factor=0.99, custom_colors=None, reward_plot_height=4):
    """
    Reads 'Avg_reward' scalar data from multiple CSV files
    and plots them on a single graph using TensorBoard's EMA smoothing style.
    Removes the standard deviation shading.
    Saves the plot to a file instead of displaying it.

    Args:
        csv_paths_with_names (list of tuples): A list where each tuple contains:
                                                (csv_file_path, display_name_for_plot)
                                                Each CSV file is expected to have 'Step' and 'Avg_reward' columns.
        smoothing_factor (float): The EMA smoothing factor (e.g., 0.99 for strong smoothing).
                                  Matches TensorBoard's smoothing slider.
        custom_colors (list of str, optional): A list of color strings (e.g., 'blue', '#FF0000', 'green')
                                                to explicitly set colors for each run.
                                                If None, uses matplotlib's default colormap.
                                                The order of colors should match the order in csv_paths_with_names.
        reward_plot_height (float): The height of the reward plot figure (in inches). Lower value "flattens" the plot.
    """

    plt.figure(figsize=(10, reward_plot_height))
    
    # Define colors for the plots
    if custom_colors and len(custom_colors) >= len(csv_paths_with_names):
        colors = custom_colors
    else:
        print("Warning: Custom colors not provided or not enough colors. Using default colormap.")
        colors_map = plt.cm.get_cmap('tab10', len(csv_paths_with_names))
        colors = [colors_map(i) for i in range(len(csv_paths_with_names))]

    for idx, (csv_path, display_name) in enumerate(csv_paths_with_names):
        print(f"Processing '{display_name}' from CSV file: {csv_path}")
        
        try:
            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(csv_path)

            # 检查必要的列是否存在
            if 'Step' not in df.columns or 'win_rate' not in df.columns:
                print(f"警告: CSV文件 '{csv_path}' 缺少 'Step' 或 'Avg_reward' 列。跳过此文件。")
                continue

            steps = df['Step'].values
            values = df['win_rate'].values

            steps = steps[:min(200000, len(steps))]
            values = values[:min(200000, len(values))]
            
            if len(values) > 0:
                smoothed_values = calculate_ema(values, smoothing_factor)
                
                line_color = colors[idx] # 使用指定或分配的颜色
                plt.plot(steps, smoothed_values, label=display_name, color=line_color, linewidth=2)
            else:
                print(f"警告: '{display_name}' 中没有足够的 Avg_reward 数据点进行平滑。跳过此文件的绘图。")

            print(f"  从 '{display_name}' 收集到 {len(values)} 'Avg_reward' 数据点。")

        except FileNotFoundError:
            print(f"错误: 找不到CSV文件 '{csv_path}'。跳过此文件。")
            continue
        except pd.errors.EmptyDataError:
            print(f"警告: CSV文件 '{csv_path}' 为空。跳过此文件。")
            continue
        except Exception as e:
            print(f"错误: 读取或处理CSV文件 '{csv_path}' 时发生异常: {e}. 跳过此文件。")
            continue
            
    # Customize the plot to match the desired style
    plt.title('Average win rate', fontsize=16)
    plt.xlabel('Training step', fontsize=14)
    plt.ylabel('Average win rate', fontsize=14)
    plt.grid(False) # 保持没有网格

    # 保持 x 轴百万格式化
    def millions_formatter(x, pos):
        # 这里的 x 是实际的 step 值，如果你的 step 已经是 1,2,3...
        # 并且你希望在图上显示 1M, 2M 等，你需要根据你的实际 Step 范围调整
        # 如果你的 Step 最大值只有几千，那么这里就不应该用 M
        # 假设你的 Step 已经够大，比如几十万、几百万
        if x >= 1e6:
            return f'{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.1f}K'
        else:
            return f'{x:.0f}'

    plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=12, loc='lower right')
    else:
        print("没有 Avg_reward 数据被绘制。图例将不显示。")

    plt.tight_layout()
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True) # 确保 'plot' 目录存在
    plt.savefig(os.path.join(plot_dir, 'bad_dyn.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    csv_configs = [
        ('csvs/data_gamept.csv', 'reward = fanCnt'),
        ('csvs/data_linear.csv', 'baseline policy method'),
        ('csvs/data_policy.csv', 'reward = actual game score'),
        # ('data_add.csv', 'B=128,Params=0.13M'),
    ]

    custom_plot_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e'] # 示例颜色

    print("\n--- 开始从多个 CSV 文件绘制 EMA 平滑后的 Average Episode Reward ---")
    plot_multiple_csv_rewards_smoothed(
        csv_configs, 
        smoothing_factor=0.999, # 可以调整平滑因子
        custom_colors=custom_plot_colors,
        reward_plot_height=5.5 # 调整图的高度
    )
    print("--- 绘图完成 ---")
