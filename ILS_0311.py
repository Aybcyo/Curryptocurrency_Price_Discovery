# Author: Gretchen_RuochengGu
# CreateTime: 2026/3/11
#  Filename: ILS_0311
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.vector_ar.vecm import VECM
# import warnings
#
# # 忽略 VECM 在协整检验时的收敛警告
# warnings.filterwarnings("ignore")
#
#
# # ==========================================
# # 模块 1：VECM 核心计算 (保持不变，严格对齐论文)
# # ==========================================
#
#
# # ==========================================
# # 模块 2：数据聚合与滑动窗口执行
# # ==========================================
#
# # 1. 读取你已经分好类的单一文件
# # 确保你的 CSV 里有 'open_time_stamp', 'implied_stock_price', 'observed_stock_price' 这几列
# df_raw = pd.read_csv("/home/featurize/work/RuochengG/data_bsm/Call/BTC/short_itm/part-00008-f4e6cd7e-11ff-4ccb-9311-73b36077273b-c000.csv")
#
# # 2. 直接扔进处理函数
# df_result = process_single_category(df_raw, window_size=60)
#
#
# # 这样你的 df_merged 就既有 option_price，又有 ILS1 了！
# print(df_merged.head())
# # 3. 查看或保存结果
# print(df_result[['open_time_stamp', 'IS1', 'ILS1']].head())
# print(df_merged.columns)
# df_result.to_csv("/home/featurize/work/RuochengG/Call_ATM_ILS_calculated.csv", index=False)
import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 之前定义好的 VECM 核心处理函数
def calculate_price_discovery_metrics(p1_series, p2_series, lag_order=1):
    """
    计算 IS, CS, IL, ILS
    """
    if np.any(np.isnan(p1_series)) or np.any(np.isnan(p2_series)):
      return [np.nan] * 8

    data = np.column_stack((p1_series, p2_series))

    try:
      vecm = VECM(data, k_ar_diff=lag_order, deterministic='ci')
      vecm_res = vecm.fit()

      alpha1 = vecm_res.alpha[0, 0]
      alpha2 = vecm_res.alpha[1, 0]

      if alpha2 - alpha1 == 0:
        return [np.nan] * 8

      CS1 = alpha2 / (alpha2 - alpha1)
      CS2 = alpha1 / (alpha1 - alpha2)

      resid = vecm_res.resid
      omega = np.cov(resid.T)
      M = np.linalg.cholesky(omega)
      m11, m12, m22 = M[0, 0], M[1, 0], M[1, 1]

      num1 = (CS1 * m11 + CS2 * m12) ** 2
      num2 = (CS2 * m22) ** 2
      denom = num1 + num2

      IS1 = num1 / denom
      IS2 = num2 / denom

      # 加上微小常数 1e-9 防止除以 0
      IL1 = np.abs((IS1 / (IS2 + 1e-9)) * (CS2 / (CS1 + 1e-9)))
      IL2 = np.abs((IS2 / (IS1 + 1e-9)) * (CS1 / (CS2 + 1e-9)))

      ILS1 = IL1 / (IL1 + IL2)
      ILS2 = IL2 / (IL1 + IL2)

      return IS1, IS2, CS1, CS2, IL1, IL2, ILS1, ILS2

    except Exception:
      return [np.nan] * 8
#
# ==========================================
# (为了代码完整性，这里简写，请直接套用上一轮我们写好的 process_single_category)
def process_single_category(df, window_size=60, lag_order=1):
  """
  处理单个已经分好类的 DataFrame (例如只装有 ATM Call 的数据)
  """
  print(f"原始数据行数: {len(df)}")

  # 1. 必不可少的步骤：时间戳聚合
  # 即使分好类，同一分钟内仍可能有多个不同的期权成交/报价
  # 论文要求取平均，同时这能彻底杜绝你之前遇到的 1:60 重复 bug
  df_agg = df.groupby('open_time_stamp').agg({
    'implied_stock_price': 'mean',
    'spot_close': 'mean'
  }).reset_index()

  # 确保按时间排序
  df_agg = df_agg.sort_values('open_time_stamp').reset_index(drop=True)
  print(f"聚合后唯一时间戳行数: {len(df_agg)}")

  # overall ETH更改
  # ==========================================
  # 🚨 新增：强制洗去脏数据（数据类型转换与安全过滤）
  # ==========================================
  # 1. 强制将这两列转为纯数字类型 (把奇奇怪怪的字符或空值强行变成 NaN)
  df_agg['implied_stock_price'] = pd.to_numeric(df_agg['implied_stock_price'], errors='coerce')
  df_agg['spot_close'] = pd.to_numeric(df_agg['spot_close'], errors='coerce')

  # 2. 剔除价格 <= 0 或为 NaN 的异常数据 (因为对数 log 不能对 0 和负数运算！)
  df_agg = df_agg[(df_agg['implied_stock_price'] > 0) & (df_agg['spot_close'] > 0)].reset_index(drop=True)
  # overall ETH

  # 2. 提前计算对数价格
  df_agg['p1'] = np.log(df_agg['implied_stock_price'])
  df_agg['p2'] = np.log(df_agg['spot_close'])

  # 初始化结果列
  metrics = ['IS1', 'IS2', 'CS1', 'CS2', 'IL1', 'IL2', 'ILS1', 'ILS2']
  for m in metrics:
    df_agg[m] = np.nan

  print(f"开始执行 VECM 滑动窗口 (Window={window_size})...")

  # 3. 严格的滑动窗口计算 (只将结果赋给窗口最末端)
  for i in range(window_size, len(df_agg) + 1):
    window = df_agg.iloc[i - window_size: i]

    p1_seq = window['p1'].values
    p2_seq = window['p2'].values

    res = calculate_price_discovery_metrics(p1_seq, p2_seq, lag_order=lag_order)

    # 将算出的指标记录在窗口终点行 (i-1)
    df_agg.loc[i - 1, metrics] = res

    if i % 1000 == 0:
      print(f"已处理 {i} 个窗口...")

  # 清理过程列，去除前 59 行算不出结果的 NaN
  df_final = df_agg.drop(columns=['p1', 'p2']).dropna(subset=['ILS1'])
  print(f"处理完成！有效指标行数: {len(df_final)}\n")

  return df_final

# def process_single_category(df, window_size=60, lag_order=1):
#   # 1. 聚合去重
#   df_agg = df.groupby('open_time_stamp').agg({
#     'implied_stock_price': 'mean',
#     'observed_stock_price': 'mean',
#     # 如果你想把路径里提取的标签带到最终结果里，可以在这里用 'first' 保留
#     'maturity_label': 'first',
#     'moneyness_label': 'first',
#     'option_type': 'first',
#     'asset': 'first'
#   }).reset_index()
#
#   df_agg = df_agg.sort_values('open_time_stamp').reset_index(drop=True)
#   df_agg['p1'] = np.log(df_agg['implied_stock_price'])
#   df_agg['p2'] = np.log(df_agg['observed_stock_price'])
#
#   metrics = ['IS1', 'IS2', 'CS1', 'CS2', 'IL1', 'IL2', 'ILS1', 'ILS2']
#   for m in metrics: df_agg[m] = np.nan
#
#   for i in range(window_size, len(df_agg) + 1):
#     window = df_agg.iloc[i - window_size: i]
#     # 【此处调用之前的 calculate_price_discovery_metrics 函数】
#     # 这里为了演示略过具体计算，假设返回了一堆指标
#     # res = calculate_price_discovery_metrics(window['p1'].values, window['p2'].values)
#     # df_agg.loc[i - 1, metrics] = res
#     pass
#
#     # df_final = df_agg.drop(columns=['p1', 'p2']).dropna(subset=['ILS1'])
#   return df_agg  # 实际运行请替换为 df_final


# ==========================================
# 自动化遍历与批量执行模块
# ==========================================
# def batch_run_pipeline(base_input_dir, base_output_dir):
#   """
#   遍历目录，提取路径标签，合并CSV，并执行模型计算
#   """
#   # 确保输出总目录存在
#   os.makedirs(base_output_dir, exist_ok=True)
#
#   # 遍历 base_input_dir 下的所有子文件夹
#   for root, dirs, files in os.walk(base_input_dir):
#     # 找出当前文件夹下的所有 csv 文件 (Spark 生成的 part-xxxx.csv)
#     csv_files = [f for f in files if f.endswith('.csv')]
#
#     # 如果当前文件夹没有 csv 文件，说明是上层目录，跳过
#     if not csv_files:
#       continue
#
#     print(f"\n正在处理目录: {root}")
#
#     # --------------------------------------------------
#     # 1. 从路径中提取标签信息
#     # 假设路径: .../data_bsm/Call/BTC/short_itm/
#     # --------------------------------------------------
#     path_parts = root.split(os.sep)
#
#     try:
#       # 找到 'data_bsm' 在路径中的位置，往后推算标签
#       base_idx = path_parts.index('data_bsm')
#       opt_type = path_parts[base_idx + 1]  # 'Call' 或 'Put'
#       asset = path_parts[base_idx + 2]  # 'BTC' 或 'ETH'
#       cat_folder = path_parts[base_idx + 3]  # 'short_itm' 等
#
#       # 拆分 short_itm 为 maturity 和 moneyness
#       if '_' in cat_folder:
#         maturity, moneyness = cat_folder.split('_', 1)
#       else:
#         maturity, moneyness = "unknown", cat_folder
#
#     except (ValueError, IndexError):
#       print(f"路径解析失败，跳过: {root}")
#       continue
#
#     # --------------------------------------------------
#     # 2. 读取并合并当前目录下的所有 Spark 分片 CSV
#     # --------------------------------------------------
#     df_list = []
#     for file in csv_files:
#       file_path = os.path.join(root, file)
#       df_list.append(pd.read_csv(file_path))
#
#     combined_df = pd.concat(df_list, ignore_index=True)
#     print(f"[{opt_type} | {asset} | {maturity} | {moneyness}] 数据读取完毕，共 {len(combined_df)} 行")
#
#     # --------------------------------------------------
#     # 3. 将隐含标签注入到 DataFrame 中 (解决你说的"没有label"问题)
#     # --------------------------------------------------
#     combined_df['maturity_label'] = maturity
#     combined_df['moneyness_label'] = moneyness
#     combined_df['option_type'] = opt_type
#     combined_df['asset'] = asset
#
#     # --------------------------------------------------
#     # 4. 送入模型执行计算
#     # --------------------------------------------------
#     result_df = process_single_category(combined_df, window_size=60)
#     df_merged = pd.merge(
#           combined_df[['open_time_stamp','option_symbol','spot_close_time','option_close_time','spot_close',
#                        'option_close','true_option_price','date_part','expiration_date','close_time','Date','Time_to_Expiration',
#                        'strike_price','rate','implied_volatility','implied_option_price','implied_stock_price']],
#           result_df[['open_time_stamp', 'ILS1', 'ILS2']],
#           on='open_time_stamp',
#           how='left'  # 左连接：保留原始数据的所有行，匹配不上的地方填 NaN
#       )
#     # --------------------------------------------------
#     # 5. 保存结果到对应结构的输出文件夹
#     # --------------------------------------------------
#     # 比如存到: /home/featurize/work/RuochengG/data_ILS_results/Call/BTC/short_itm_ILS.csv
#     out_folder = os.path.join(base_output_dir, opt_type, asset)
#     os.makedirs(out_folder, exist_ok=True)
#
#     out_file_name = f"{cat_folder}_ILS_results0311.csv"
#     out_file_path = os.path.join(out_folder, out_file_name)
#
#     df_merged.to_csv(out_file_path, index=False)
#     print(f"结果已保存至: {out_file_path}")
#
#
# # ==========================================
# # 🚀 启动入口
# # ==========================================
# if __name__ == "__main__":
#   # 你的原始数据根目录
#   INPUT_DIR = "/home/featurize/work/RuochengG/data_bsm"
#
#   # 你想存放计算结果的根目录
#   OUTPUT_DIR = "/home/featurize/work/RuochengG/data_ILS_results"
#
#   batch_run_pipeline(INPUT_DIR, OUTPUT_DIR)

# overall ILS
def run_overall_ils_pipeline(base_input_dir, base_output_dir, target_asset='BTC'):
  """
  计算全市场 Overall ILS (算法A：Pooling)
  遍历所有文件夹，将特定资产（如 BTC）的所有期权数据合为一体送入模型
  """
  print(f"\n" + "=" * 50)
  print(f"🚀 开始收集 {target_asset} 全市场数据 (Pooling)")
  print("=" * 50)

  os.makedirs(base_output_dir, exist_ok=True)
  df_list = []

  # 1. 遍历并收集该资产下的所有 CSV
  for root, dirs, files in os.walk(base_input_dir):
    csv_files = [f for f in files if f.endswith('.csv')]
    if not csv_files:
      continue

    path_parts = root.split(os.sep)

    try:
      base_idx = path_parts.index('data_bsm')
      asset = path_parts[base_idx + 2]  # 获取资产名 (BTC 或 ETH)

      # 只收集目标资产的数据
      if asset == target_asset:
        for file in csv_files:
          file_path = os.path.join(root, file)
          df_list.append(pd.read_csv(file_path))

    except (ValueError, IndexError):
      continue

  if not df_list:
    print(f"⚠️ 未找到 {target_asset} 的数据，请检查路径！")
    return

  # 2. 拼接成一个全市场的巨型 DataFrame
  print(f"正在拼接数据，共找到 {len(df_list)} 个数据块...")
  combined_df = pd.concat(df_list, ignore_index=True)
  print(f"全市场数据拼接完毕！共 {len(combined_df)} 行。")

  # 3. 送入模型执行计算
  # (这里会调用你写好的 process_single_category)
  # 里面的 groupby('open_time_stamp').mean() 会完美执行 Pooling 混合！
  print(f"\n开始计算 Overall ILS (将执行全市场隐含价格大融合)...")
  result_df = process_single_category(combined_df, window_size=60)

  # 4. 保存最终的 Overall 结果
  out_file_name = f"{target_asset}_Overall_ILS.csv"
  out_file_path = os.path.join(base_output_dir, out_file_name)

  # 只保留关键列保存以节省空间
  keep_cols = ['open_time_stamp', 'IS1', 'IS2', 'CS1', 'CS2', 'IL1', 'IL2', 'ILS1', 'ILS2']
  result_df_clean = result_df[keep_cols].copy()  # 加上 .copy() 避免 Pandas 警告

  # ==========================================
  # 👉 新增：计算并写入均值到单独的一列
  # ==========================================
  ils1_mean = result_df_clean['ILS1'].mean()
  ils2_mean = result_df_clean['ILS2'].mean()

  result_df_clean['Overall_ILS1_Mean'] = ils1_mean
  result_df_clean['Overall_ILS2_Mean'] = ils2_mean

  # 顺便把结果打印在终端里，让你心里有数
  print("\n" + "=" * 50)
  print(f"📊 {target_asset} 全市场期权平均主导份额 (ILS1 Mean): {ils1_mean:.2%}")
  print(f"📊 {target_asset} 全市场现货平均主导份额 (ILS2 Mean): {ils2_mean:.2%}")
  print("=" * 50)

  # 保存最终结果
  result_df_clean.to_csv(out_file_path, index=False)
  print(f"🎉 结果已保存至: {out_file_path}")


# ==========================================
# 🚀 启动入口
# ==========================================
if __name__ == "__main__":
  # 你的原始数据根目录
  INPUT_DIR = "/home/featurize/work/RuochengG/data_bsm"

  # 建议给 Overall 结果单独建一个文件夹，避免和细分标签混在一起
  OUTPUT_DIR = "/home/featurize/work/RuochengG/data_ILS_results/Overall"

  # 跑完 BTC，你可以把这里改成 'ETH' 再跑一次
  run_overall_ils_pipeline(INPUT_DIR, OUTPUT_DIR, target_asset='ETH')
