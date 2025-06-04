import tensorflow as tf
from tensorflow.core.util import event_pb2
import os # This is the module that contains listdir
import glob

def merge_tensorboard_logs(input_log_dirs, output_log_dir):
    """
    Merges multiple TensorBoard log directories into a single output directory,
    automatically adjusting 'global_step' to create a continuous timeline.

    Args:
        input_log_dirs (list): A list of paths to your existing TensorBoard log directories,
                               in the chronological order you want them merged.
        output_log_dir (str): The directory where the merged log file will be saved.
    """

    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
        print(f"Created output directory: {output_log_dir}")

    temp_writer = tf.summary.create_file_writer(output_log_dir)
    temp_writer.close()

    output_event_files = glob.glob(os.path.join(output_log_dir, 'events.out.tfevents.*'))
    if not output_event_files:
        print(f"Error: No .tfevents file found in {output_log_dir} after initialization. Exiting.")
        return

    destination_event_file_path = output_event_files[0]
    print(f"Merged events will be written to: {destination_event_file_path}")

    tf_record_writer = tf.io.TFRecordWriter(destination_event_file_path)

    current_global_step_offset = 0

    # First pass: Determine max step for each run to calculate offsets.
    run_info = [] # Stores (input_dir, max_step_in_run)
    for input_dir in input_log_dirs:
        print(f"Scanning '{input_dir}' for max step...")
        # *** Check this line carefully in your local file: it must be os.listdir ***
        input_event_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir) # <--- THIS MUST BE os.listdir
            if f.startswith('events.out.tfevents.')
        ]
        input_event_files.sort()

        max_step_in_current_run = -1
        if not input_event_files:
            print(f"No .tfevents files found in '{input_dir}'. Skipping.")
            run_info.append((input_dir, -1))
            continue

        try:
            for event_file in input_event_files:
                for event_str in tf.compat.v1.io.tf_record_iterator(event_file):
                    event = event_pb2.Event.FromString(event_str)
                    if event.step is not None:
                        max_step_in_current_run = max(max_step_in_current_run, event.step)
        except Exception as e:
            print(f"Error scanning '{event_file}' for max step: {e}. Data from this run might be incomplete.")

        print(f"Max step found in '{input_dir}': {max_step_in_current_run}")
        run_info.append((input_dir, max_step_in_current_run))

    # Second pass: Read events, apply offset, and write.
    for input_dir, max_step_pre_scanned in run_info:
        print(f"Processing '{input_dir}' with current step offset: {current_global_step_offset}")

        # *** And check this line too: it must be os.listdir ***
        input_event_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir) # <--- THIS MUST BE os.listdir
            if f.startswith('events.out.tfevents.')
        ]
        input_event_files.sort()

        if not input_event_files:
            continue

        for event_file in input_event_files:
            try:
                for event_str in tf.compat.v1.io.tf_record_iterator(event_file):
                    original_event = event_pb2.Event.FromString(event_str)
                    
                    new_event = event_pb2.Event()
                    new_event.wall_time = original_event.wall_time

                    if original_event.step is not None and original_event.step >= 0:
                        new_event.step = original_event.step + current_global_step_offset
                        
                        if original_event.HasField('summary'):
                            new_event.summary.CopyFrom(original_event.summary)
                    else:
                        if original_event.HasField('file_version'):
                            new_event.file_version = original_event.file_version
                        if original_event.HasField('graph_def'):
                            new_event.graph_def = original_event.graph_def
                        if original_event.HasField('meta_graph_def'):
                            new_event.meta_graph_def = original_event.meta_graph_def
                        if original_event.HasField('log_message'):
                            new_event.log_message.CopyFrom(original_event.log_message)
                        if original_event.HasField('session_log'):
                            new_event.session_log.CopyFrom(original_event.session_log)
                        if original_event.HasField('tagged_run_metadata'):
                            new_event.tagged_run_metadata.CopyFrom(original_event.tagged_run_metadata)
                        
                        if not new_event.HasFields():
                             continue

                    tf_record_writer.write(new_event.SerializeToString())

            except Exception as e:
                print(f"Error processing event in '{event_file}': {e}. Skipping this file.")
                continue

        if max_step_pre_scanned >= 0:
            current_global_step_offset += (max_step_pre_scanned + 1)
        print(f"Finished '{input_dir}'. Next offset: {current_global_step_offset}")

    tf_record_writer.close()
    print(f"Successfully merged all logs into '{output_log_dir}'.")

# --- Example Usage ---
if __name__ == "__main__":
    input_log_dirs_to_merge = [
        "expe/0530_2009_ppo/log",
        "expe/0530_2009_ppo/0531_1412/log",
        "expe/0530_2009_ppo/0601_1529/log",
        "expe/0530_2009_ppo/0602_1001/log",
    ]
    
    merged_output_dir = './merged_tb_runs/aaa'

    if os.path.exists(merged_output_dir):
        import shutil
        print(f"Cleaning up existing output directory: {merged_output_dir}")
        shutil.rmtree(merged_output_dir)

    print("\n--- Starting TensorBoard log merging process ---")
    merge_tensorboard_logs(input_log_dirs_to_merge, merged_output_dir)
    print("--- TensorBoard log merging process completed ---")

    print("\nTo view the merged logs, run:")
    print(f"tensorboard --logdir {merged_output_dir}")