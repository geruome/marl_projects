import pandas as pd

def rename_reward_column(input_csv_path, output_csv_path):
    """重命名CSV文件中'reward'列为'win_rate'列。"""
    try:
        df = pd.read_csv(input_csv_path) # 读取CSV文件
        if 'reward' in df.columns:
            df.rename(columns={'reward': 'win_rate'}, inplace=True) # 重命名列
            df.to_csv(output_csv_path, index=False) # 保存到新文件
            print(f"列已成功从 'reward' 重命名为 'win_rate'，并保存到: {output_csv_path}")
        else:
            print("错误: CSV文件中未找到 'reward' 列。")
    except FileNotFoundError:
        print(f"错误: 文件未找到: {input_csv_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 示例使用
if __name__ == "__main__":
    input_file = 'csvs/data_linear.csv'  # 替换为你的输入CSV文件名
    output_file = 'csvs/data_linear1.csv' # 替换为你的输出CSV文件名

    # 创建一个示例CSV文件用于测试 (如果你的input_file不存在)

    rename_reward_column(input_file, output_file)

    # 清理示例文件 (可选)
    # import os
    # if os.path.exists(input_file): os.remove(input_file)
    # if os.path.exists(output_file): os.remove(output_file)