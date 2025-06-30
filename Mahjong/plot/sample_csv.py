import pandas as pd
import os

def random_sample_csv(input_csv_path, output_csv_path, sample_percentage):
    """
    从CSV文件中随机抽取指定百分比的数据，并将 'Step' 列重新索引为从 1 开始的连续整数。

    Args:
        input_csv_path (str): 原始CSV文件的路径。
        output_csv_path (str): 新的CSV文件的路径。
        sample_percentage (float): 抽样的百分比，例如 0.10 表示抽样 10% 的数据。
                                   值应在 (0, 1] 之间。
    """
    if not os.path.exists(input_csv_path):
        print(f"错误: 输入CSV文件不存在: {input_csv_path}")
        return
    if not (0 < sample_percentage <= 1):
        print(f"错误: 抽样百分比必须在 (0, 1] 之间。当前值为: {sample_percentage}")
        return

    print(f"正在从 '{input_csv_path}' 随机抽样 {sample_percentage*100:.2f}% 数据，并保存到 '{output_csv_path}'...")

    try:
        # 读取原始CSV文件到DataFrame
        df = pd.read_csv(input_csv_path)

        # 检查 'Step' 列是否存在
        if 'Step' not in df.columns:
            print(f"错误: CSV文件 '{input_csv_path}' 中没有找到 'Step' 列。")
            return

        # 随机抽样数据
        # frac 参数表示要返回的轴项的比例（分数）
        # random_state 用于确保结果可复现，可以设置为任何整数
        sampled_df = df.sample(frac=sample_percentage, random_state=42)

        # 按照原始的 'Step' 列（或者任何你希望的排序依据）进行排序，
        # 以便在重新索引后保持数据的逻辑顺序
        # 如果原始Step本身是乱序的，这里也可以选择不排序，但通常是希望保留时间序列顺序
        sampled_df = sampled_df.sort_values(by='Step').reset_index(drop=True)

        # 重新索引 'Step' 列为从 1 开始的连续整数
        sampled_df['Step'] = range(1, len(sampled_df) + 1)

        # 将抽样后的数据保存到新的CSV文件
        sampled_df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"成功将抽样后的数据保存到 '{output_csv_path}'")
        print(f"原始数据行数: {len(df)}")
        print(f"抽样后数据行数: {len(sampled_df)}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_csv_path}'。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{input_csv_path}' 为空。")
    except Exception as e:
        print(f"错误: 处理CSV文件时发生异常: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的输入CSV文件名为 'data.csv'
    input_csv_file = 'data.csv' 
    input_csv_path = os.path.join(os.getcwd(), input_csv_file) 

    # 定义新的输出CSV文件路径和文件名
    output_csv_file_sampled = 'data_sampled_0.71.csv'
    output_csv_path = os.path.join(os.getcwd(), output_csv_file_sampled)

    percentage_to_sample = 0.71

    print("\n--- 开始从 CSV 随机抽样并重新索引 Step ---")
    random_sample_csv(input_csv_path, output_csv_path, percentage_to_sample)
    print("--- CSV 随机抽样完成 ---")

    print(f"\n抽样后的CSV文件已保存到: {output_csv_path}")