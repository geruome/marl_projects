import tensorflow as tf
import os
import shutil

# Make sure eager execution is enabled at the very start if you intend to use TF2.x features
# tf.compat.v1.enable_eager_execution() # This might conflict with the v1.Graph context later if placed here

def rename_tensorboard_scalar(input_log_dir, output_log_dir, old_tag, new_tag):
    """
    读取指定TensorBoard日志目录中的事件文件，将特定标量标签重命名，
    并保存到新的日志目录。

    Args:
        input_log_dir (str): 原始TensorBoard日志目录的路径。
        output_log_dir (str): 新的TensorBoard日志目录的路径。
        old_tag (str): 需要重命名的旧标量标签（例如 'Avg_reward'）。
        new_tag (str): 新的标量标签（例如 'reward'）。
    """
    if not os.path.exists(input_log_dir):
        print(f"错误: 输入日志目录不存在: {input_log_dir}")
        return
    if not os.path.isdir(input_log_dir): # Add a check to ensure it's a directory
        print(f"错误: 输入路径 '{input_log_dir}' 不是一个目录。请提供 TensorBoard 日志目录。")
        return

    # 创建新的输出日志目录，如果已存在则清空
    if os.path.exists(output_log_dir):
        print(f"警告: 输出日志目录 {output_log_dir} 已存在，将被清空。")
        shutil.rmtree(output_log_dir)
    os.makedirs(output_log_dir)

    print(f"正在从 '{input_log_dir}' 读取并处理日志...")

    # --- Crucial change: Iterate over actual event files ---
    # Disable eager execution for v1 FileWriter compatibility
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Graph().as_default():
        new_writer = tf.compat.v1.summary.FileWriter(output_log_dir)

        # Find all tfevents files in the input directory
        event_files = [os.path.join(input_log_dir, f) 
                       for f in os.listdir(input_log_dir) 
                       if f.startswith('events.out.tfevents.')]
        
        if not event_files:
            print(f"警告: 在目录 '{input_log_dir}' 中没有找到 TensorBoard 事件文件。")
            new_writer.close() # Close writer even if no files
            tf.compat.v1.enable_eager_execution() # Re-enable eager mode
            return

        for event_file_path in event_files:
            print(f"  正在处理文件: {os.path.basename(event_file_path)}")
            # For each event file, create a summary_iterator
            for event_file in tf.compat.v1.train.summary_iterator(event_file_path):
                if event_file.HasField('summary'):
                    new_summary = tf.compat.v1.Summary()
                    for value in event_file.summary.value:
                        if value.tag == old_tag:
                            new_value = new_summary.value.add()
                            new_value.tag = new_tag
                            new_value.simple_value = value.simple_value
                        else:
                            new_summary.value.add().CopyFrom(value)
                    
                    if new_summary.value:
                        new_event = tf.compat.v1.Event(
                            wall_time=event_file.wall_time,
                            step=event_file.step,
                            summary=new_summary
                        )
                        new_writer.add_event(new_event)
                else:
                    new_writer.add_event(event_file)

        new_writer.close()
    
    tf.compat.v1.enable_eager_execution() # Re-enable eager execution

    print(f"处理完成。新的日志文件已保存到 '{output_log_dir}'")
    print(f"您可以通过 'tensorboard --logdir {output_log_dir}' 查看新日志。")


# --- 使用示例 ---
if __name__ == "__main__":
    input_log_directory = 'merged_tb/great'
    output_log_directory = 'merged_tb/great1'

    old_scalar_tag = 'Avg_reward'
    new_scalar_tag = 'reward'

    rename_tensorboard_scalar(input_log_directory, output_log_directory, old_scalar_tag, new_scalar_tag)