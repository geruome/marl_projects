import tensorflow as tf
from tensorflow.core.util import event_pb2 # Needed for Event.FromString and Event construction
import os
import shutil
import glob
import time # For generating unique filenames

def subsample_tensorboard_logs(input_log_dir, output_log_dir, sample_interval=5):
    if not os.path.exists(input_log_dir):
        print(f"错误: 输入日志目录不存在: {input_log_dir}")
        return
    if not os.path.isdir(input_log_dir):
        print(f"错误: 输入路径 '{input_log_dir}' 不是一个目录。请提供 TensorBoard 日志目录。")
        return
    if sample_interval <= 0:
        print("错误: 抽样间隔必须是正整数。")
        return

    # 创建新的输出日志目录，如果已存在则清空
    if os.path.exists(output_log_dir):
        print(f"警告: 输出日志目录 {output_log_dir} 已存在，将被清空。")
        shutil.rmtree(output_log_dir)
    os.makedirs(output_log_dir)

    print(f"正在从 '{input_log_dir}' 读取并按每 {sample_interval} 个事件抽样...")

    # 生成一个新的事件文件名，确保唯一性
    timestamp = str(int(time.time()))
    output_event_filename = f"events.out.tfevents.{timestamp}.subsampled"
    destination_event_file_path = os.path.join(output_log_dir, output_event_filename)

    tf_record_writer = tf.io.TFRecordWriter(destination_event_file_path)

    # 查找输入目录中的所有 .tfevents 文件
    input_event_files = glob.glob(os.path.join(input_log_dir, 'events.out.tfevents.*'))
    input_event_files.sort() # 确保按时间戳顺序处理

    if not input_event_files:
        print(f"警告: 在目录 '{input_log_dir}' 中没有找到 TensorBoard 事件文件。")
        tf_record_writer.close()
        return

    global_event_counter = 0 # 用于跟踪所有文件的累积事件数量（作为新的合成步数）
    written_events_count = 0

    for event_file_path in input_event_files:
        print(f"  正在处理文件: {os.path.basename(event_file_path)}")            
        for event_str in tf.compat.v1.io.tf_record_iterator(event_file_path):
            original_event = event_pb2.Event.FromString(event_str)
            
            # 累积事件计数器，无论是哪种类型的事件
            global_event_counter += 1

            should_write_this_event = False

            # 1. 对于非 Summary 事件（如文件版本、图定义、日志信息等），直接写入
            #    因为它们不含step且通常只出现少数次，对图表影响小但保持完整性
            if not original_event.HasField('summary'):
                should_write_this_event = True
            # 2. 对于 Summary 事件，根据计数器进行抽样
            else: # original_event.HasField('summary') is True
                if global_event_counter % sample_interval == 0:
                    should_write_this_event = True
            
            if should_write_this_event:
                new_event = event_pb2.Event()
                new_event.CopyFrom(original_event) # 复制原始事件的所有字段

                # 核心：为所有写入的 Summary 事件强制添加/更新合成的 step
                # 确保 TensorBoard 能够显示这些数据，并以我们合成的步数作为横轴
                if new_event.HasField('summary'):
                    new_event.step = global_event_counter
                # 对于非 summary 事件，如果它原本有 step，则保留；如果没有，也不添加
                # 这是为了保持它们的原貌，它们不影响我们关心的横坐标
                # (因为它们一般不会在 Scalar 图中出现)
                
                tf_record_writer.write(new_event.SerializeToString())
                written_events_count += 1



    tf_record_writer.close()
    print(f"处理完成。")
    print(f"总共读取原始事件: {global_event_counter}")
    print(f"总共写入抽样事件 (按 {sample_interval} 间隔): {written_events_count}")
    print(f"新的日志文件已保存到 '{output_log_dir}'")
    print(f"您可以通过 'tensorboard --logdir {output_log_dir}' 查看新日志。")

if __name__ == "__main__":
    input_log_directory = 'merged_tb/great' 

    output_log_directory = 'merged_tb/great2'

    sample_interval_steps = 5 # 10就不行, 5正常

    if os.path.exists(output_log_directory):
        print(f"清理旧的输出目录: {output_log_directory}")
        shutil.rmtree(output_log_directory)

    print("\n--- 开始 TensorBoard 日志抽样 (完全基于事件顺序) ---")
    subsample_tensorboard_logs(input_log_directory, output_log_directory, sample_interval_steps)
    print("--- TensorBoard 日志抽样完成 ---")

    print("\n要查看抽样后的日志，请运行:")
    print(f"tensorboard --logdir {output_log_directory}")