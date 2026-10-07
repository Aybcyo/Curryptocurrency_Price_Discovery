# Author: Gretchen_RuochengGu
# CreateTime: 2025/7/30
#  Filename: graphing2
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设定文件夹路径
folder_path = "/home/featurize/work/RuochengG/data_with_move_ave/Call/ETH"
output_folder = os.path.join(folder_path, "plots")
os.makedirs(output_folder, exist_ok=True)  # 创建保存图片的目录

# 遍历所有 .csv 文件
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)

            # 检查是否有想画的列
            if 'ILS1_MA60' in df.columns and 'ILS2_MA60' in df.columns:
                plt.figure(figsize=(10, 5))
                plt.scatter(df['spot_close_time'], df['ILS1_MA60'], label='ILS1_MA60')
                plt.scatter(df['spot_close_time'], df['ILS2_MA60'], label='ILS2_MA60')
                plt.title(f"Call - ETH - {file} - Moving Average_60 of ILS1 & ILS2")
                plt.xlabel("spot close time")
                plt.ylabel("ILS moving average")
                plt.legend()
                # plt.grid(True)
                plt.show()

                # 保存图片到 output_folder
                plot_filename = os.path.splitext(file)[0] + ".png"
                plot_path = os.path.join(output_folder, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"✅ 绘图完成并保存：{plot_path}")
            else:
                print(f"⚠️ 文件缺少 ILS1/ILS2 列，跳过：{file}")

        except Exception as e:
            print(f"❌ 读取或绘图失败：{file}, 错误：{e}")
