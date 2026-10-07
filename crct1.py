# Author: Gretchen_RuochengGu
# CreateTime: 2025/8/5
#  Filename: ImpliedVolatilitySurface
import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')  # 忽略一些数学计算的警告


# ==========================================
# 1. 核心计算函数 (已加入强力数据清洗！)
# ==========================================

def compute_surface_macro_features(group):
    # 预设所有你需要的新 Features
    empty_res = {
      'M2': np.nan, 'M3': np.nan, 'M4': np.nan,
      'Skewness': np.nan, 'Kurtosis': np.nan, 'Entropy': np.nan,
      'Sigma_ATM': np.nan, 'P_RightTail': np.nan, 'P_LeftTail': np.nan,
      'IV_Skew': np.nan, 'IV_Curvature': np.nan
    }

    # 1. 强力清洗数据
    clean_group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=['strike_price', 'implied_volatility'])
    curve = clean_group.groupby('strike_price')['implied_volatility'].mean().reset_index().sort_values('strike_price')

    if len(curve) < 5:
      return empty_res

    K_arr = curve['strike_price'].values
    IV_arr = curve['implied_volatility'].values

    S = clean_group['implied_stock_price'].median()
    r = clean_group['rate'].median()
    T = clean_group['Time_to_Expiration'].median()

    if pd.isna(S) or S <= 0 or T <= 0:
      return empty_res

    # 远期价格 F
    F = S * np.exp(r * T)

    try:
      # 三次样条插值拟合隐含波动率曲线
      cs_iv = CubicSpline(K_arr, IV_arr, bc_type='natural')
      sigma_atm = float(cs_iv(F))

      # ==========================================
      # 内部辅助函数：根据插值的IV，反算期权价格 (O_K)
      # ==========================================
      def bs_price_fwd(K, vol, is_call):
        vol = max(min(vol, 5.0), 0.01)  # 防止极值溢出
        d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if is_call:
          return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
          return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

      # O(K) 函数：K < F 取 Put，K >= F 取 Call [公式图3]
      def O_K(K):
        vol = float(cs_iv(K))
        return bs_price_fwd(K, vol, is_call=(K >= F))

      # ==========================================
      # 积分函数 (Integrands)
      # ==========================================
      # M2: O(K) / K^2 [公式图2]
      def integrand_M2(K):
        return O_K(K) / (K ** 2)

      # M3: (6ln(K/F) - 3[ln(K/F)]^2) / K^2 * O(K) [公式图3]
      def integrand_M3(K):
        ln_kf = np.log(K / F)
        return (6 * ln_kf - 3 * (ln_kf ** 2)) / (K ** 2) * O_K(K)

      # M4: (12[ln(K/F)]^2 - 4[ln(K/F)]^3) / K^2 * O(K) [公式图4]
      def integrand_M4(K):
        ln_kf = np.log(K / F)
        return (12 * (ln_kf ** 2) - 4 * (ln_kf ** 3)) / (K ** 2) * O_K(K)

      K_min, K_max = K_arr.min(), K_arr.max()

      # 计算积分 (使用 limit=50 防止计算卡死)
      int_M2, _ = quad(integrand_M2, K_min, K_max, limit=50)
      int_M3, _ = quad(integrand_M3, K_min, K_max, limit=50)
      int_M4, _ = quad(integrand_M4, K_min, K_max, limit=50)

      # 还原 M2, M3, M4 (假设 K0=F，消除漂移项)
      M2 = (2 * np.exp(r * T) / T) * int_M2
      M3 = np.exp(r * T) * int_M3
      M4 = np.exp(r * T) * int_M4

      # 计算标准化偏度和峰度
      skewness = M3 / (M2 ** 1.5) if M2 > 0 else np.nan
      kurtosis = M4 / (M2 ** 2) if M2 > 0 else np.nan

      # ==========================================
      # Implied Entropy (Lognormal Approximation) [公式图5]
      # ==========================================
      if M2 > 0:
        sigma_ent_sq = M2 * T
        mu_ent = np.log(F) - sigma_ent_sq / 2
        # 对数正态分布的信息熵解析解: ln(sigma * sqrt(2*pi*e)) + mu
        entropy = np.log(np.sqrt(sigma_ent_sq * 2 * np.pi * np.e)) + mu_ent
      else:
        entropy = np.nan

      # ==========================================
      # Tail Risk Metrics [公式图7]
      # ==========================================
      tail_move = 3 * sigma_atm * np.sqrt(T)
      K_high = F + tail_move
      K_low = max(F - tail_move, 0.0001)

      def calc_d2(K, vol):
        return (np.log(F / K) - 0.5 * vol ** 2 * T) / (vol * np.sqrt(T))

      # P(S_T > K_high)
      P_RightTail = norm.cdf(calc_d2(K_high, sigma_atm))
      # P(S_T < K_low)
      P_LeftTail = norm.cdf(-calc_d2(K_low, sigma_atm))

      # ==========================================
      # Volatility Skew & Curvature [公式图8, 9]
      # ==========================================
      # 选取典型的 10% 虚实值对数资金面 (m = +/- 0.1)
      m_otm_val, m_itm_val = 0.1, -0.1
      K_otm = min(F * np.exp(m_otm_val), K_max)
      K_itm = max(F * np.exp(m_itm_val), K_min)

      # 实际取到的对数资金面
      m_otm_eff = np.log(K_otm / F)

      vol_otm = float(cs_iv(K_otm))
      vol_itm = float(cs_iv(K_itm))

      # Skewness (Strike-based slope)
      iv_skew = (vol_otm - vol_itm) / (K_otm - K_itm) if K_otm != K_itm else np.nan
      # Smile Curvature (Log-moneyness based)
      iv_curv = (vol_otm + vol_itm - 2 * sigma_atm) / (m_otm_eff ** 2) if m_otm_eff != 0 else np.nan

      # 完美输出所有要求的特征字典
      return {
        'M2': M2, 'M3': M3, 'M4': M4,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Entropy': entropy,
        'Sigma_ATM': sigma_atm,
        'P_RightTail': P_RightTail,
        'P_LeftTail': P_LeftTail,
        'IV_Skew': iv_skew,
        'IV_Curvature': iv_curv
      }
    except Exception:
      # 如果数学引擎由于极值崩溃，安全返回空模板
      return empty_res



# ==========================================
# def read_csv_files(folder_path):
#   if not os.path.exists(folder_path):
#     raise FileNotFoundError(f"❌ 找不到目录: {folder_path}")
#
#   # 【核心改动】：加上 "**" 和 recursive=True，让它穿透所有子文件夹去找 .csv
#   search_pattern = os.path.join(folder_path, "**", "*.csv")
#   file_list = glob.glob(search_pattern, recursive=True)
#
#   # 如果没找到 .csv，找 part- 开头的文件
#   if not file_list:
#     search_pattern_part = os.path.join(folder_path, "**", "part-*")
#     file_list = glob.glob(search_pattern_part, recursive=True)
#
#   # 过滤掉 Spark 生成的无用文件
#   file_list = [f for f in file_list if not f.endswith(('.crc', '_SUCCESS'))]
#
#   if not file_list:
#     raise ValueError(f"⚠️ 警告：目录 '{folder_path}' 及其子目录中没有找到数据！")
#
#   print(f"✅ 在 {folder_path} 及子目录中找到了 {len(file_list)} 个数据文件！")
#
#   df_list = [pd.read_csv(f) for f in file_list]
#   return pd.concat(df_list, ignore_index=True)
# folder_path1 = "/home/featurize/work/RuochengG/data_ILS_results/Call/BTC"
# folder_path2 = '/home/featurize/work/RuochengG/data_ILS_results/Call/ETH'
# folder_path3 = '/home/featurize/work/RuochengG/data_ILS_results/Put/BTC'
# folder_path4 = '/home/featurize/work/RuochengG/data_ILS_results/Put/ETH'
# Call_BTC = read_csv_files(folder_path1)
# Call_ETH = read_csv_files(folder_path2)
# Put_BTC = read_csv_files(folder_path3)
# Put_ETH = read_csv_files(folder_path4)
# ==========================================
print("1. 正在读取并合并全量数据...")
# 请确保你的 Call_BTC 和 Put_BTC 已经用 read_csv_files 读好了
# full_market_df = pd.concat([Call_BTC, Put_BTC], ignore_index=True)
full_market_df = pd.read_csv('')
# 强力洗刷列名，防止 \n 报错
full_market_df.columns = full_market_df.columns.str.strip()
print(f"数据总量: {len(full_market_df)} 行")

# 转换为时间格式并提取【分钟】桶
full_market_df['spot_close_time'] = pd.to_datetime(full_market_df['spot_close_time'])
full_market_df['time_bin'] = full_market_df['spot_close_time'].dt.floor('1min')  # <--- 这里改成了 1min！

# ==========================================
# 3. 计算宏观特征 (防 OOM 的 For 循环机制)
# ==========================================
print("2. 正在计算跨截面的宏观风险特征(硬核循环模式)...")
results_list = []

# 直接遍历 groupby 的结果，绝对不会触发 apply 的玄学合并问题
for (t_bin, expiry), group in full_market_df.groupby(['time_bin', 'expiration_date']):
  res_dict = compute_surface_macro_features(group)
  res_dict['time_bin'] = t_bin
  res_dict['expiration_date'] = expiry
  results_list.append(res_dict)

macro_features_table = pd.DataFrame(results_list)
macro_features_table.dropna(subset=['M2'], inplace=True)
print(f"✅ 成功计算并构建了 {len(macro_features_table)} 个有效宏观特征！")

# ==========================================
# 4. 特征映射与行级 P(ITM) 计算
# ==========================================
# print("3. 正在将宏观特征映射回全量大表，并计算 d2...")
# # 这一次，我们光明正大地将特征合回 full_market_df
# full_market_df = pd.merge(
#   full_market_df,
#   macro_features_table,
#   on=['time_bin', 'expiration_date'],
#   how='left'
# )
#
# # 极速向量化计算 d2
# full_market_df['d2'] = (np.log(full_market_df['implied_stock_price'] / full_market_df['strike_price']) +
#                         (full_market_df['rate'] - 0.5 * full_market_df['implied_volatility'] ** 2) * full_market_df[
#                           'Time_to_Expiration']) / \
#                        (full_market_df['implied_volatility'] * np.sqrt(full_market_df['Time_to_Expiration']))
#
# print("🎉 全量数据处理完毕！")
# # print(full_market_df[['time_bin', 'strike_price', 'M2', 'Skewness', 'd2']].head(15))
# print(full_market_df.head(15))
print("3. 正在将宏观特征映射回全量大表，并计算 d2...")

full_market_df = pd.merge(
    full_market_df,
    macro_features_table,
    on=['time_bin', 'expiration_date'],
    how='left'
)

# ==========================================
# 【新增终极防御】：强制将所有数学计算列转为纯数字 (float64)
# errors='coerce' 会把所有非数字的乱码直接变成 NaN，防止报错
# ==========================================
math_cols = ['implied_stock_price', 'strike_price', 'rate', 'implied_volatility', 'Time_to_Expiration']
for col in math_cols:
    full_market_df[col] = pd.to_numeric(full_market_df[col], errors='coerce')

# 为了绝对安全，去除那些无法计算 d2 的行（比如波动率为 0 或剩余时间为 0，会导致除以 0 报错）
safe_mask = (full_market_df['implied_volatility'] > 0) & (full_market_df['Time_to_Expiration'] > 0)

# 在安全的数据上计算 d2
full_market_df.loc[safe_mask, 'd2'] = (
    np.log(full_market_df.loc[safe_mask, 'implied_stock_price'] / full_market_df.loc[safe_mask, 'strike_price']) +
    (full_market_df.loc[safe_mask, 'rate'] - 0.5 * full_market_df.loc[safe_mask, 'implied_volatility']**2) * full_market_df.loc[safe_mask, 'Time_to_Expiration']
) / (
    full_market_df.loc[safe_mask, 'implied_volatility'] * np.sqrt(full_market_df.loc[safe_mask, 'Time_to_Expiration'])
)
print("4. 正在根据 d2 计算单张期权的价内概率 P(ITM)...")

# 2. 计算连续正态分布的累积概率 N(d2) 和 N(-d2)
# 使用 scipy.stats.norm.cdf 可以极速完成千万级别的概率运算
Nd2 = norm.cdf(full_market_df['d2'].fillna(0))
N_minus_d2 = norm.cdf(-full_market_df['d2'].fillna(0))

# 3. 区分 Call 和 Put，填入正确的 P(ITM)
# 假设你的数据能通过 option_symbol 结尾的 'C' 或 'P' 来区分，如果列名不同请自行替换
is_call = full_market_df['option_symbol'].str.endswith('C')

full_market_df['P_ITM'] = np.where(is_call, Nd2, N_minus_d2)

# 把安全掩码外的无效数据重新置为 NaN
full_market_df.loc[~safe_mask, 'P_ITM'] = np.nan

print("🎉 概率计算完成！")
print("🎉 全量数据处理完毕！")
print(full_market_df[['option_symbol','time_bin', 'strike_price', 'M2', 'Skewness', 'Entropy','IV_Curvature','P_ITM']].head(15))
full_market_df.to_csv("/home/featurize/work/RuochengG/features_BTC_0311.csv", index=False)
print("🎉 BTC数据保存完毕！")
# # 如果需要保存结果，可以直接在这里使用 full_market_df.to_csv("xxx.csv", index=False)
# # ==========================================
# # 执行流程
# # ==========================================
# # print("1. 正在读取并合并所有数据以构建完整的市场表面...")
# # # folder_path = '/home/featurize/work/RuochengG/data_bsm/Call/BTC'
# # # file_pattern = os.path.join(folder_path, '*.csv')
# # #
# # # # 获取所有 csv 文件路径 (确保该文件夹下只有你需要处理的期权数据)
# # # file_list = glob.glob(file_pattern)
# # # df_list = []
# # # for file in file_list:
# # #   df = pd.read_csv(file)
# # #   # 【核心优化】提取文件名（例如 'Call_OTM_data'），作为新的一列打上标签
# # #   base_name = os.path.basename(file).replace('.csv', '')
# # #   df['source_file'] = base_name
# # #   df_list.append(df)
# # # # 一次性物理拼接所有碎片数据
# # #
# # # def read_csv_files(folder_path):
# # #   df_list = []
# # #   for file in os.listdir(folder_path):
# # #     if file.endswith(".csv"):
# # #       file_path = os.path.join(folder_path,file)
# # #       df = pd.read_csv(file_path)
# # #       df_list.append(df)
# # #
# # #   combined_df = pd.concat(df_list, ignore_index = True)
# # #   return combined_df
# #
# #
# #
# #
# # full_market_df = pd.concat([Call_BTC, Put_BTC], ignore_index=True)
# # full_market_df.columns = full_market_df.columns.str.strip()
# #
# #
# # # 统一时间格式，防止 merge 时类型不匹配报错
# # full_market_df['spot_close_time'] = pd.to_datetime(full_market_df['spot_close_time'])
# #
# # # 2. 【核心修正】：正确的小样本测试提取法
# # # 取出前 10 个独一无二的时间戳
# # # ==========================================
# # full_market_df['time_bin'] = full_market_df['spot_close_time'].dt.floor('1min')
# # print(f"数据总量: {len(full_market_df)} 行")
# # # full_market_timestamps = full_market_df['spot_close_time'].unique()[:10]
# # #
# # # # 提取这 10 个时间点对应的所有 Call 和 Put（保持截面完整性！）
# # # test_df = full_market_df[full_market_df['spot_close_time'].isin(full_market_timestamps)].copy()
# # #
# # # print(f"抽取了 {len(test_timestamps)} 个时间截面，共计 {len(test_df)} 行测试数据。")
# #
# # print("2. 正在计算跨截面的宏观风险特征...")
# # macro_features_table = full_market_df.groupby(
# #     ['time_bin', 'expiration_date']
# # ).apply(compute_surface_macro_features).reset_index()
# #
# # # print("2. 正在计算跨截面的宏观风险特征...")
# # # # 按 【时间】 和 【到期日】 切分出独立的市场快照，计算宏观特征
# # # macro_features_table = full_market_df.groupby(
# # #   ['spot_close_time', 'expiration_date']
# # # ).apply(compute_surface_macro_features).reset_index()
# #
# # # 丢弃因数据不足（如点数少于 5 个）导致无法计算的无效截面
# # macro_features_table.dropna(subset=['M2'], inplace=True)
# # print(f"成功计算出 {len(macro_features_table)} 个有效宏观特征！")
# # print(macro_features_table.head(10))
# #
# # print("3. 正在将宏观特征映射回原数据，并逐行计算微观价内概率 P(ITM)...")
# #
# # # 【修正1】：把合并的结果老老实实存回 test_df
# # full_market_df = pd.merge(
# #     full_market_df,
# #     macro_features_table,
# #     on=['time_bin', 'expiration_date'],
# #     how='left'
# # )
# #
# # # 【修正2】：统一全用 test_df 算 d2，不要再混入 full_market_df
# # full_market_df['d2'] = (np.log(full_market_df['implied_stock_price'] / full_market_df['strike_price']) +
# #                  (full_market_df['rate'] - 0.5 * full_market_df['implied_volatility']**2) * full_market_df['Time_to_Expiration']) / \
# #                 (full_market_df['implied_volatility'] * np.sqrt(full_market_df['Time_to_Expiration']))
# #
# # print("🎉 小样本测试圆满成功！")
# # print(full_market_df[['time_bin', 'strike_price', 'M2', 'Skewness', 'd2']].head(10))
# #
# #
# # # 【核心优化】利用刚才打上的 'source_file' 标签，智能判断 Call 和 Put
# # # 只要文件名里包含 'Call' (不区分大小写)，就判定为看涨期权
# # is_call = full_market_df['option_symbol'].str.contains('C', case=False)
# #
# # # 向量化计算 P_ITM (Call 用 d2, Put 用 -d2)
# # full_market_df['P_ITM'] = np.where(is_call, norm.cdf(full_market_df['d2']), norm.cdf(-full_market_df['d2']))
# # print(full_market_df.head())
# # print(full_market_df.columns)
# # print("4. 正在还原并保存赋予了新特征的数据集...")
# # # 根据 'source_file' 标签，将大表重新拆分为原来的小表，并分别保存
# # for source, sub_df in full_market_df.groupby('source_file'):
# #   save_path = os.path.join(folder_path, f"{source}_with_features.csv")
# #
# #   # 保存前，清理掉辅助计算用的中间列，保持数据干净
# #   columns_to_drop = ['source_file', 'd2']
# #   sub_df = sub_df.drop(columns=columns_to_drop, errors='ignore')
# #
# #   # sub_df.to_csv(save_path, index=False)
# #   # print(f"  -> 已保存: {save_path}")
# #
# # print("所有数据处理完毕！完美收工。")
#
#
# # print("1. 正在读取并合并所有数据以构建完整的市场表面...")
# # # 1. 定义文件夹路径和匹配模式 (*.csv 代表所有 csv 文件)
# # folder_path = '/Users/ruocheng.gu/Desktop/Option/'
# # file_pattern = os.path.join(folder_path, '*.csv')
# #
# # # 2. 获取所有符合条件的文件路径列表
# # file_list = glob.glob(file_pattern)
# #
# # # 3. 循环读取所有文件，并存入一个列表中
# # df_list = []
# # for file in file_list:
# #     df = pd.read_csv(file)
# #     df_list.append(df)
# #
# # # 4. 将列表中的所有 DataFrame 上下拼接成一个大的 DataFrame
# # # ignore_index=True 会重新生成 0, 1, 2... 的索引，避免索引冲突
# # df_call_otm = pd.concat(df_list, ignore_index=True)
# # print(f"成功合并了 {len(file_list)} 个文件！")
# # print(df_call_otm.head())
# # # 假设你的文件存在一个文件夹下，格式类似于 Call_OTM_data.csv, Put_ITM_data.csv 等
# # # all_files = glob.glob("/Users/ruocheng.gu/Desktop/Option/*_data.csv")
# # # 1. 物理拼接：把所有分散的表上下堆叠在一起 (不管 Call/Put 还是 ITM/OTM)
# # full_market_df = pd.concat([df_call_otm, df_put_otm, df_call_itm, df_put_itm])
# # # full_market_df = pd.concat([pd.read_csv(f) for f in all_files])
# #
# # # (示例: 如果你已经把它读进一个叫做 full_market_df 的变量里)
# # # full_market_df['spot_close_time'] = pd.to_datetime(full_market_df['spot_close_time'])
# #
# # print("2. 正在计算跨截面的宏观风险特征...")
# # # macro_features_table = full_market_df.groupby('spot_close_time').apply(compute_surface_macro_features).reset_index()
# # # 2. 计算标准：按 【时间】 和 【到期日】 切分出独立的市场快照，计算宏观特征
# # macro_features_table = full_market_df.groupby(
# #     ['spot_close_time', 'expiration_date']  # <--- 这就是合并与计算的标准！
# # ).apply(compute_surface_macro_features).reset_index()
# # macro_features_table.dropna(inplace=True)  # 丢弃无法计算的无效截面
# #
# # print("3. 正在逐个处理分离的数据集，并进行特征映射与行级计算...")
# # # 遍历你分离的那些数据集
# # # for file_path in all_files:
# # #     # 读取特定的切割数据 (比如只包含 Call OTM)
# # #     sub_df = pd.read_csv(file_path)
# # #     sub_df['spot_close_time'] = pd.to_datetime(sub_df['spot_close_time'])
# #
# # #     # --- 核心步骤：将宏观特征匹配到每一行 ---
# # #     sub_df = pd.merge(sub_df, macro_features_table, on='spot_close_time', how='left')
# #
# # #     # --- 核心步骤：逐行计算微观价内概率 P(ITM) ---
# # #     # 利用数据中已有的 BS 隐含参数
# # #     sub_df['d2'] = (np.log(sub_df['implied_stock_price'] / sub_df['strike_price']) +
# # #                     (sub_df['rate'] - 0.5 * sub_df['implied_volatility']**2) * sub_df['Time_to_Expiration']) / \
# # #                    (sub_df['implied_volatility'] * np.sqrt(sub_df['Time_to_Expiration']))
# #
# # #     # 假设你可以从文件名或列中区分 Call 和 Put
# # #     is_call = 'Call' in file_path # 请根据你的实际命名规则修改
# # #     if is_call:
# # #         sub_df['P_ITM'] = norm.cdf(sub_df['d2'])
# # #     else:
# # #         sub_df['P_ITM'] = norm.cdf(-sub_df['d2'])
# #
# # #     # 保存赋予了新特征的数据
# # #     save_path = file_path.replace('.csv', '_with_features.csv')
# # #     sub_df.to_csv(save_path, index=False)
# # #     print(f"完成文件处理并保存: {save_path}")
# #
# # print("所有数据处理完毕！")
