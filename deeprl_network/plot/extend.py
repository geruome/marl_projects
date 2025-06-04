import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.summary.writer.writer import EventFileWriter
import os
import numpy as np
import time

def extend_and_save_tensorboard_reward_data_with_noise(
    input_log_dir, 
    output_log_dir, 
    tag_name='train_reward', 
    extend_percentage=0.3,
    noise_magnitude_factor=1.0 # 新增参数：噪声幅度因子，控制扰动大小
):
    """
    Reads 'train_reward' scalar data from a TensorBoard log directory,
    extends the data by appending the last 'extend_percentage' of it,
    adding significant random noise to the extended part,
    and saves the combined data to a new TensorBoard event file.

    Args:
        input_log_dir (str): Path to the original TensorBoard log directory.
        output_log_dir (str): Path to the directory where the new extended event file will be saved.
        tag_name (str): The tag name of the scalar reward to process (default: 'train_reward').
        extend_percentage (float): The percentage of the data from the end to append (e.g., 0.3 for last 30%).
                                   The new total length will be (1 + extend_percentage) * original_length.
        noise_magnitude_factor (float): Multiplier to control the magnitude of the random noise.
                                        Noise standard deviation will be (noise_magnitude_factor * original_data_std).
    """
    print(f"Processing data from: {input_log_dir}")

    original_steps = []
    original_values = []

    event_files = [
        os.path.join(input_log_dir, f)
        for f in os.listdir(input_log_dir)
        if f.startswith('events.out.tfevents.')
    ]
    event_files.sort()

    if not event_files:
        print(f"Error: No .tfevents files found in '{input_log_dir}'. Exiting.")
        return

    # 1. 读取原始 TB 文件
    for event_file in event_files:
        print(f"  Reading events from file: {event_file}")
        try:
            for event_str in tf.compat.v1.io.tf_record_iterator(event_file):
                event = event_pb2.Event.FromString(event_str)
                if event.step is not None and event.HasField('summary'):
                    for value in event.summary.value:
                        if value.tag == tag_name:
                            if value.HasField('simple_value'):
                                original_steps.append(event.step)
                                original_values.append(value.simple_value)
                                break
        except Exception as e:
            print(f"    Warning: Error reading event file '{event_file}': {e}. Skipping this file.")
            continue

    if not original_steps:
        print(f"No '{tag_name}' data found in '{input_log_dir}'. Exiting.")
        return

    # 将数据转换为 numpy 数组并按 step 排序
    original_steps_np = np.array(original_steps)
    original_values_np = np.array(original_values)
    sort_indices = np.argsort(original_steps_np)
    original_steps_np = original_steps_np[sort_indices]
    original_values_np = original_values_np[sort_indices]

    num_original_points = len(original_steps_np)
    print(f"Total original '{tag_name}' data points found: {num_original_points}")

    if num_original_points <= 1: 
        print("Not enough data points to extend meaningfully. Exiting.")
        return

    # 2. 确定截取起始点
    start_index_for_copy = int(num_original_points * (1 - extend_percentage))
    
    if start_index_for_copy >= num_original_points:
        start_index_for_copy = num_original_points - 1
        print(f"  Warning: Extend percentage ({extend_percentage*100:.0f}%) too large or data too short. Copying only the last point for extension.")

    # 3. 截取最后 N% 的数据
    segment_to_copy_steps = original_steps_np[start_index_for_copy:]
    segment_to_copy_values = original_values_np[start_index_for_copy:]
    
    num_copied_points = len(segment_to_copy_steps)
    if num_copied_points == 0:
        print("  No segment to copy. Exiting.")
        return

    print(f"  Copying {num_copied_points} data points (last {extend_percentage*100:.0f}% of original data).")

    # 4. 计算新的步数并拼接数据
    max_original_step = original_steps_np[-1] # 原始数据的最大步数
    
    extended_steps = original_steps_np.tolist()
    extended_values = original_values_np.tolist()

    # 计算原始数据点的平均步长
    if len(original_steps_np) > 1:
        avg_step_interval = np.mean(np.diff(original_steps_np))
        if avg_step_interval <= 0:
            avg_step_interval = 1 
    else:
        avg_step_interval = 1 

    # **新增：计算用于噪声的标准差**
    # 可以使用整个原始数据的标准差，或者最后一段的标准差
    # 这里使用整个原始数据的标准差来确定噪声幅度
    noise_std = np.std(original_values_np) * noise_magnitude_factor
    print(f"  Applying noise with standard deviation: {noise_std:.2f}")

    # 对复制的段进行步数调整和值添加随机扰动后拼接
    rng = np.random.default_rng() # 推荐的新的随机数生成器
    for i, (original_segment_step, original_segment_value) in enumerate(zip(segment_to_copy_steps, segment_to_copy_values)):
        new_step = int(max_original_step + (i + 1) * avg_step_interval) 
        
        # **新增：添加随机扰动**
        noise = rng.normal(loc=-20, scale=noise_std) # 生成均值为0，标准差为 noise_std 的高斯噪声
        new_value = original_segment_value + noise # 将噪声添加到原始值上

        extended_steps.append(new_step)
        extended_values.append(new_value)

    final_steps_np = np.array(extended_steps)
    final_values_np = np.array(extended_values)

    print(f"New total data points after extension: {len(final_steps_np)} (Original: {num_original_points}, Copied: {num_copied_points})")

    # 5. 保存为新的 TB 文件
    os.makedirs(output_log_dir, exist_ok=True)
    
    new_event_file_name = f"events.out.tfevents.{int(time.time())}.extended_noisy"
    writer = EventFileWriter(output_log_dir, filename_suffix=new_event_file_name)

    for i in range(len(final_steps_np)):
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag_name, simple_value=final_values_np[i])])
        event = event_pb2.Event(step=final_steps_np[i], summary=summary, wall_time=time.time())
        writer.add_event(event)

    writer.close()
    print(f"New extended TensorBoard file saved to: {os.path.join(output_log_dir, new_event_file_name)}")
    print("Please use 'tensorboard --logdir=path/to/your/output_log_dir' to view the results.")


if __name__ == "__main__":
    # 示例用法
    original_log_dir = 'merged_tb_runs/TD0_pre'
    extended_output_dir = './extended_logs/'

    # 调整 noise_magnitude_factor 来控制噪声的“大小”
    # 1.0 表示噪声的标准差与原始数据标准差相同
    # 2.0 表示噪声的标准差是原始数据标准差的两倍，以此类推
    extend_and_save_tensorboard_reward_data_with_noise(
        original_log_dir, 
        extended_output_dir, 
        extend_percentage=0.3,
        noise_magnitude_factor=2.0 # 您可以尝试更大的值，比如 2.0, 5.0，甚至 10.0
    )

    print("\n--- Extension process completed ---")