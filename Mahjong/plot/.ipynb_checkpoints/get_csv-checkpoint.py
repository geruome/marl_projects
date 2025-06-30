import tensorflow as tf
from tensorflow.core.util import event_pb2
import os
import glob
import csv
import time # Not strictly needed for logic, but good for timestamp in comments or future filename generation

def extract_and_reindex_scalar_to_csv(input_log_dir, output_csv_path, target_tag='Avg_reward'):
    """
    从TensorBoard日志目录中提取特定标量（例如 'Avg_reward'）的所有数值，
    并将其以 (连续索引步数, value) 的形式写入 CSV 文件。
    'Step' 列将从 1 开始连续递增。

    Args:
        input_log_dir (str): 原始TensorBoard日志目录的路径。
        output_csv_path (str): 输出CSV文件的完整路径（包括文件名）。
        target_tag (str): 目标标量的标签，例如 'Avg_reward'。
    """
    if not os.path.exists(input_log_dir):
        print(f"错误: 输入日志目录不存在: {input_log_dir}")
        return
    if not os.path.isdir(input_log_dir):
        print(f"错误: 输入路径 '{input_log_dir}' 不是一个目录。请提供 TensorBoard 日志目录。")
        return

    print(f"正在从 '{input_log_dir}' 提取标量 '{target_tag}' 并生成从 1 开始的连续步数到 '{output_csv_path}'...")

    # 查找输入目录中的所有 .tfevents 文件
    input_event_files = glob.glob(os.path.join(input_log_dir, 'events.out.tfevents.*'))
    input_event_files.sort() # 确保按时间戳顺序处理，这对于生成连续的 step 很重要

    if not input_event_files:
        print(f"警告: 在目录 '{input_log_dir}' 中没有找到 TensorBoard 事件文件。没有数据可提取。")
        return

    extracted_data = []
    # 专门用于记录 Avg_reward 数据点在 CSV 中的步数，从 1 开始
    csv_step_counter = 0 

    for event_file_path in input_event_files:
        print(f"  正在处理文件: {os.path.basename(event_file_path)}")
        try:
            for event_str in tf.compat.v1.io.tf_record_iterator(event_file_path):
                original_event = event_pb2.Event.FromString(event_str)
                
                # 检查事件是否包含 Summary
                if original_event.HasField('summary'):
                    for value in original_event.summary.value:
                        if value.tag == target_tag:
                            # 找到目标标量，每次找到一个就增加 CSV 的步数计数器
                            csv_step_counter += 1
                            # 存储 (新的连续步数, 标量值)
                            extracted_data.append([csv_step_counter, value.simple_value / 50])
                            # 如果一个事件可能包含多个同名tag，这里只会记录第一个。
                            # 通常不会有，但如果需要，可以修改逻辑。
                            break # 找到目标tag后，可以跳出内层循环，处理下一个event

        except Exception as e:
            print(f"错误: 处理文件 '{os.path.basename(event_file_path)}' 时发生异常: {e}. 跳过此文件。")
            continue

    if not extracted_data:
        print(f"警告: 没有在日志中找到标签为 '{target_tag}' 的任何数据。")
        return

    # 将提取的数据写入 CSV 文件
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Step', 'win_rate']) # 写入表头
            csv_writer.writerows(extracted_data) # 写入数据行
        print(f"成功将 '{target_tag}' 数据提取并保存到 '{output_csv_path}'")
        print(f"总共提取了 {len(extracted_data)} 条数据。")
    except Exception as e:
        print(f"错误: 写入CSV文件 '{output_csv_path}' 时发生异常: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你的原始 TensorBoard 日志目录
    input_log_directory = 'expe/0626223307' 

    # 定义输出 CSV 文件的路径和文件名
    output_csv_file = 'csvs/data_policy.csv'
    output_csv_path = os.path.join(os.getcwd(), output_csv_file) # 默认保存到当前运行目录下

    # 定义要提取的标量标签
    target_scalar_tag = 'Avg_reward'

    print("\n--- 开始 TensorBoard 标量提取 (一步到位生成连续 Step CSV) ---")
    extract_and_reindex_scalar_to_csv(input_log_directory, output_csv_path, target_scalar_tag)
    print("--- TensorBoard 标量提取完成 ---")

    print(f"\nCSV文件已保存到: {output_csv_path}")