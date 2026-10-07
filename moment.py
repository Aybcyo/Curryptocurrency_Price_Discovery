# Author: Gretchen_RuochengGu
# CreateTime: 2025/8/4
#  Filename: moment
# second
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_files(folder_path):
  df_list = []
  for file in os.listdir(folder_path):
    if file.endswith(".csv"):
      file_path = os.path.join(folder_path,file)
      df = pd.read_csv(file_path)
      df_list.append(df)

  combined_df = pd.concat(df_list, ignore_index = True)
  return combined_df

# Spark 输出目录（注意它是一个文件夹，不是单个 csv 文件）
folder_path1 = "/home/featurize/work/RuochengG/data_with_move_ave/Call/BTC"
folder_path2 = '/home/featurize/work/RuochengG/data_with_move_ave/Call/ETH'
folder_path3 = '/home/featurize/work/RuochengG/data_with_move_ave/Put/BTC'
folder_path4 = '/home/featurize/work/RuochengG/data_with_move_ave/Put/ETH'
Call_BTC = read_csv_files(folder_path1)
Call_ETH = read_csv_files(folder_path2)
Put_BTC = read_csv_files(folder_path3)
Put_ETH = read_csv_files(folder_path4)
df_call = pd.merge(Call_BTC, Call_ETH, on='strike_price', how='inner')
df_put = pd.merge(Put_BTC, Put_ETH, on='strike_price', how='inner')
df = pd.merge(df_call, df_put, on = 'strike_price', how = 'inner')

merged_df = df.sort_values(by='strike_price')

# 提取数值
K = merged_df['strike_price'].values
C = merged_df['implied_option_price'].values
P = merged_df['implied_option_price'].values
T = merged_df['Time_to_Expiration']
r = merged_df['rate'].values
# === 计算 integrand: 0.5 * K^2 * (P + C) ===
integrand2 = 0.5 / K**2 * (P * C)
integrand3 = 6/K**3 *(P*C)
integrand4 = 24/K**4 * (P*C)
discount = np.exp(r * T)
# === 数值积分 ===
M2 = discount * np.trapz(integrand2, K)
M3 = discount * np.trapz(integrand3, K)
M4 = discount * np.trapz(integrand4, K)

print(f"Risk-neutral variance (Second Moment M2): {M2:.6f}")
print(f"Skewnessrelated (Third Moment M3): {M3:.6f}")
print(f"Kurtosisrelated (Forth Moment M4): {M4:.6f}")
print(f"skewness: {M3/M2 ** 1.5:.6f}")
print(f"Kurtosis: {M4/M2**2:.6f}")
print(f"Variance: {M2:.6f}")
print(f"volatility: {np.sqrt(M2):.6f}")
