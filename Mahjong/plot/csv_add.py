import pandas as pd
import numpy as np
import os

def linear_increase_avg_reward(input_csv_path, output_csv_path, initial_increase=0.0, final_increase=0.01):
    """
    读取CSV文件，并对 'Avg_reward' 列的值进行修改，
    使其按行（按 Step 顺序）线性递增，从初始行的 initial_increase 增加到
    最后一行的 final_increase。

    Args:
        input_csv_path (str): 原始CSV文件的路径。
        output_csv_path (str): 新的CSV文件的路径。
        initial_increase (float): 第一个数据点（第一行）的 Avg_reward 增加的值。
        final_increase (float): 最后一个数据点（最后一行）的 Avg_reward 增加的值。
    """
    if not os.path.exists(input_csv_path):
        print(f"错误: 输入CSV文件不存在: {input_csv_path}")
        return

    print(f"正在读取 '{input_csv_path}' 并线性增加 'Avg_reward'...")

    try:
        # 读取CSV文件到DataFrame
        df = pd.read_csv(input_csv_path)

        # 检查 'Avg_reward' 列是否存在
        if 'Avg_reward' not in df.columns:
            print(f"错误: CSV文件 '{input_csv_path}' 中没有找到 'Avg_reward' 列。")
            return

        num_rows = len(df)

        if num_rows == 0:
            print("警告: 输入CSV文件为空，没有数据可处理。")
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"空CSV文件已保存到 '{output_csv_path}'。")
            return
        elif num_rows == 1:
            # 如果只有一行数据，只应用 initial_increase
            df['Avg_reward'] = df['Avg_reward'] + initial_increase
            print("注意: CSV文件只有一行数据，只应用了初始增加值。")
        else:
            # 计算总的增加范围
            total_increase_range = final_increase - initial_increase
            
            # 为每一行计算线性插值因子
            # indices_factor: 从 0 到 1 线性变化
            indices = np.arange(num_rows)
            # 使用 num_rows - 1 来确保最后一个点的因子是 1，第一个点是 0
            # 避免除以零的情况
            interpolation_factor = indices / (num_rows - 1) 
            
            # 计算每行对应的线性增加值
            # 增量 = 初始增量 + 插值因子 * 总增加范围
            increase_values = initial_increase + interpolation_factor * total_increase_range

            # 将计算出的增量加到 Avg_reward 列上
            df['Avg_reward'] = df['Avg_reward'] + increase_values

        # 保存修改后的数据到新的CSV文件
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"成功将修改后的数据保存到 '{output_csv_path}'")
        print(f"原始数据行数: {num_rows}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_csv_path}'。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{input_csv_path}' 为空。")
    except Exception as e:
        print(f"错误: 处理CSV文件时发生异常: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的输入CSV文件名为 'data_sampled.csv' (或者你之前的 reindexed CSV)
    input_csv_file = 'data.csv' 
    input_csv_path = os.path.join(os.getcwd(), input_csv_file) 


    # 定义新的输出CSV文件路径和文件名
    output_csv_file_increased = 'data_add.csv'
    output_csv_path = os.path.join(os.getcwd(), output_csv_file_increased)

    # 定义初始和最终增加值
    initial_increase_value = 0.0  # 第一个数据点增加 0
    final_increase_value = 0.01   # 最后一个数据点增加 0.01

    print("\n--- 开始对 CSV 中的 Avg_reward 进行线性增加 ---")
    linear_increase_avg_reward(
        input_csv_path, 
        output_csv_path, 
        initial_increase=initial_increase_value, 
        final_increase=final_increase_value
    )
    print("--- Avg_reward 线性增加完成 ---")

    print(f"\n修改后的CSV文件已保存到: {output_csv_path}")