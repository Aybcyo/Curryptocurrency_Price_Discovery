# Author: Gretchen_RuochengGu
# CreateTime: 2026/3/4
#  Filename: ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
print('v3')

def run_ml_and_shap_pipeline(data_encoded, all_features, agg_name, output_dir):
  """
  运行三大模型全样本评估并生成 SHAP 图 (支持动态特征列表)
  """
  print(f"\n" + "=" * 50)
  print(f"🚀 开始处理【{agg_name} 聚合】数据集")
  print("=" * 50)

  X = data_encoded[all_features]
  lower_bound = X.quantile(0.01)
  upper_bound = X.quantile(0.99)
  X = X.clip(lower=lower_bound, upper=upper_bound, axis=1)
  X = X.fillna(X.median())
  y = data_encoded['ILS1']

  # 神经网络专用的标准化数据
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # ---------------------------------------------------------
  # 1. 定义并训练模型 (全样本)
  # ---------------------------------------------------------
  print("正在训练 Random Forest (CPU 全核)...")
  rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
  rf_model.fit(X, y)

  print("正在训练 Neural Network (CPU)...")
  nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
  nn_model.fit(X_scaled, y)

  print("正在训练 XGBoost (GPU 加速)...")
  xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist',
    device='cuda'  # 若无 GPU 环境，请改为 'cpu'
  )
  xgb_model.fit(X, y)

  # ---------------------------------------------------------
  # 2. 模型评估与比较 (In-Sample R2)
  # ---------------------------------------------------------
  models = {
    'Random Forest': (rf_model, X),
    'Neural Network': (nn_model, X_scaled),
    'XGBoost': (xgb_model, X)
  }

  performance_data = []
  best_r2 = -float('inf')
  best_tree_model = None
  best_tree_name = ""

  for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # 将结果存入列表准备转成 DataFrame
    performance_data.append({
      'Aggregation': agg_name,
      'Model': name,
      'MSE': mse,
      'R2_Score': r2
    })

    # 挑选表现最好的树模型给 SHAP
    if name in ['Random Forest', 'XGBoost'] and r2 > best_r2:
      best_r2 = r2
      best_tree_model = model
      best_tree_name = name

  # 提取最优树模型的特征重要性
  importances = best_tree_model.feature_importances_
  importance_df = pd.DataFrame({
    'Feature': X.columns,
    f'Importance_{agg_name}': importances
  }).sort_values(by=f'Importance_{agg_name}', ascending=False)
  # ---------------------------------------------------------
  # 3. SHAP 分析最优树模型 (揭示 Maturity 贡献)
  # ---------------------------------------------------------
  print(f"\n=== 开始 SHAP 分析 (使用表现最好的 {best_tree_name}) ===")

  # 抽样加速计算 (如果你的内存够大，可以调高 n 的值)
  sample_size = min(2000, len(X))
  X_shap_sample = X.sample(n=sample_size, random_state=42)

  explainer = shap.TreeExplainer(best_tree_model)
  shap_values = explainer.shap_values(X_shap_sample)

  os.makedirs(output_dir, exist_ok=True)

  # A. 全局特征重要性柱状图 (最直观的排名)
  plt.figure()
  shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
  plt.title(f"Feature Importance ({best_tree_name} - {agg_name})")
  plt.tight_layout()
  bar_path = os.path.join(output_dir, f"shap_bar_{best_tree_name.replace(' ', '')}_{agg_name}.pdf")
  plt.savefig(bar_path, dpi=300, bbox_inches='tight')
  plt.close()

  # B. 蜂巢散点图 (能看出不同期限的正负面影响)
  plt.figure()
  shap.summary_plot(shap_values, X_shap_sample, show=False)
  plt.title(f"SHAP Summary ({best_tree_name} - {agg_name})")
  plt.tight_layout()
  summary_path = os.path.join(output_dir, f"shap_summary_{best_tree_name.replace(' ', '')}_{agg_name}.pdf")
  plt.savefig(summary_path, dpi=300, bbox_inches='tight')
  plt.close()

  print(f"✅ SHAP 图像已成功保存！(请重点观察 maturity_label 相关的特征排位)")

  return pd.DataFrame(performance_data), importance_df


def main_evaluation_pipeline(merged_data_path, output_dir):
  print(f"1. 正在读取合并后的完整数据集...")
  df = pd.read_csv(merged_data_path)

  os.makedirs(output_dir, exist_ok=True)

  # ==========================================
  # 📊 步骤 A：捕获打印的基础统计数据，存为 DataFrame
  # ==========================================
  total_raw = len(df)
  df = df.dropna(subset=['ILS1'])
  valid_ils = len(df)

  # 1. 整体数据量统计表
  summary_df = pd.DataFrame({
    'Metric': ['Total Raw Rows', 'Valid ILS Rows', 'Retention Rate'],
    'Value': [total_raw, valid_ils, valid_ils / total_raw]
  })

  # 2. Maturity 期限条数分布表
  if 'maturity_label' in df.columns:
    maturity_dist_df = df['maturity_label'].value_counts().reset_index()
    maturity_dist_df.columns = ['Maturity_Label', 'Count']
    maturity_dist_df['Percentage'] = maturity_dist_df['Count'] / valid_ils
  else:
    maturity_dist_df = pd.DataFrame({'Warning': ['maturity_label not found']})

  # 3. 核心特征的描述性统计表 (Mean, Std, Min, Max等，论文必备！)
  base_features = [
    'M2', 'M3', 'M4', 'Skewness', 'Kurtosis', 'Entropy',
    'Sigma_ATM', 'P_RightTail', 'P_LeftTail', 'IV_Skew', 'IV_Curvature'
  ]
  for col in base_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

  # 直接用 df.describe() 生成统计表
  desc_stats_df = df[base_features + ['ILS1']].describe().reset_index()
  desc_stats_df.rename(columns={'index': 'Statistic'}, inplace=True)

  # ==========================================
  # 🤖 步骤 B：执行特征聚合与机器学习流水线
  # ==========================================
  group_keys = ['time_bin', 'option_type', 'maturity_label', 'moneyness_label']
  group_cols = group_keys + ['ILS1']

  df_mean = df.groupby(group_cols)[base_features].mean().reset_index().dropna()
  df_median = df.groupby(group_cols)[base_features].median().reset_index().dropna()

  # 独热编码
  df_mean_encoded = pd.get_dummies(df_mean, columns=['maturity_label'], dtype=int)
  df_median_encoded = pd.get_dummies(df_median, columns=['maturity_label'], dtype=int)
  maturity_cols = [col for col in df_mean_encoded.columns if col.startswith('maturity_label_')]
  all_features = base_features + maturity_cols

  # （这里调用上一轮我们写的 run_ml_and_shap_pipeline，它会返回模型表现和特征重要性）
  # 假设你已经把上一轮的 run_ml_and_shap_pipeline 函数放在了上面
  perf_mean, imp_mean = run_ml_and_shap_pipeline(df_mean_encoded, all_features, "Mean", output_dir)
  perf_median, imp_median = run_ml_and_shap_pipeline(df_median_encoded, all_features, "Median", output_dir)

  final_performance = pd.concat([perf_mean, perf_median], ignore_index=True)
  final_importance = pd.merge(imp_mean, imp_median, on='Feature', how='outer')

  # ==========================================
  # 💾 步骤 C：将所有表格统一打包写入 Excel！
  # ==========================================
  excel_path = os.path.join(output_dir, "Full_Experiment_Results.xlsx")
  print(f"\n📊 正在将所有统计表格写入 Excel: {excel_path}")

  with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # 写入数据基础统计
    summary_df.to_excel(writer, sheet_name='Data_Summary', index=False)
    maturity_dist_df.to_excel(writer, sheet_name='Maturity_Distribution', index=False)
    desc_stats_df.to_excel(writer, sheet_name='Descriptive_Stats', index=False)

    # 写入模型跑出来的结果
    final_performance.to_excel(writer, sheet_name='Model_Performance', index=False)
    final_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)

  print("🎉 完美！你在终端里看到的所有数据，现在都已经在 Excel 的不同 Sheet 里躺好了。")

# def main_evaluation_pipeline(merged_data_path, output_dir):
#   print(f"1. 正在读取合并后的完整数据集...")
#   df = pd.read_csv(merged_data_path)
#
#   os.makedirs(output_dir, exist_ok=True)
#   # ---------------------------------------------------------
#   # 📊 数据基础统计模块
#   # ---------------------------------------------------------
#   print("\n" + "=" * 50)
#   print("📈 原始数据分布统计")
#   print("=" * 50)
#   total_raw = len(df)
#   df = df.dropna(subset=['ILS1'])
#   valid_ils = len(df)
#   print(f"原始数据总行数: {total_raw}")
#   print(f"包含有效 ILS1 的行数: {valid_ils} (留存率: {valid_ils / total_raw:.2%})")
#
#   print("\n按 Maturity 期限的数据条数分布:")
#   if 'maturity_label' in df.columns:
#     print(df['maturity_label'].value_counts().to_string())
#   else:
#     print("⚠️ 未找到 maturity_label 列！")
#
#   # 基础的 11 个数值特征
#   base_features = [
#     'M2', 'M3', 'M4', 'Skewness', 'Kurtosis', 'Entropy',
#     'Sigma_ATM', 'P_RightTail', 'P_LeftTail', 'IV_Skew', 'IV_Curvature'
#   ]
#   for col in base_features:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
#   group_keys = ['time_bin', 'option_type', 'maturity_label', 'moneyness_label']
#   group_cols = group_keys + ['ILS1']
#
#   print("\n2. 正在进行特征聚合 (Mean & Median)...")
#   df_mean = df.groupby(group_cols)[base_features].mean().reset_index().dropna()
#   df_median = df.groupby(group_cols)[base_features].median().reset_index().dropna()
#
#   print(f"   -> 聚合完成！Mean 数据集: {len(df_mean)} 行 | Median 数据集: {len(df_median)} 行")
#
#   # ---------------------------------------------------------
#   # 🤖 核心逻辑：对 Maturity 进行 One-Hot 编码
#   # ---------------------------------------------------------
#   # 使用 get_dummies 将 'short', 'mid', 'long' 变成 0 和 1 的独立特征列
#   df_mean_encoded = pd.get_dummies(df_mean, columns=['maturity_label'], dtype=int)
#   df_median_encoded = pd.get_dummies(df_median, columns=['maturity_label'], dtype=int)
#
#   # 动态获取新生出来的 maturity 特征列名 (例如: maturity_label_short, maturity_label_long)
#   maturity_cols = [col for col in df_mean_encoded.columns if col.startswith('maturity_label_')]
#
#   # 组合成最终喂给机器学习的完整特征矩阵 X
#   all_features = base_features + maturity_cols
#   print(f"   -> 已成功加入期限特征: {maturity_cols}")
#
#   # 执行评估流水线
#   run_ml_and_shap_pipeline(df_mean_encoded, all_features, "Mean", output_dir)
#   run_ml_and_shap_pipeline(df_median_encoded, all_features, "Median", output_dir)
#
#   print("\n🎉 所有模型训练、评估与可解释性分析完毕！")

# ==========================================
# 🚀 执行代码
# ==========================================
MERGED_FILE_PATH = "/home/featurize/work/RuochengG/data_final_ready_for_ML/ETH_final_dataset_with_ILS.csv"
OUTPUT_PLOT_DIR = "/home/featurize/work/RuochengG/ETH_SUM_0312/"
main_evaluation_pipeline(MERGED_FILE_PATH, OUTPUT_PLOT_DIR)

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import shap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib
# matplotlib.use('Agg') # 或者使用 'Qt5Agg'
# import matplotlib.pyplot as plt
# import shap
# print("================ 这是最新修改的版本 v2 ================")
# # ---------------- 1. 准备数据 ----------------
# # 假设 df 是你的数据集，'target' 是你要预测的列
# # 特征列为 M2, M3, M4, Entropy, P(ITM) 等
# df_BTC = pd.read_csv('/home/featurize/work/RuochengG/features_BTC_0303_2.csv')
# df_ETH = pd.read_csv('/home/featurize/work/RuochengG/features_ETH_0303_2.csv')
# df = pd.concat([df_BTC, df_ETH], ignore_index=True)
# features = ['M2', 'M3', 'M4','Skewness','Kurtosis','Entropy','P_ITM','P_RightTail','P_LeftTail', 'IV_Skew', 'IV_Curvature']
# X = df[features]
# y = df['true_option_price']
# # X = X.replace([np.inf, -np.inf], np.nan)
# # X = X.fillna(X.median())
# lower_bound = X.quantile(0.01)
# upper_bound = X.quantile(0.99)
# X = X.clip(lower=lower_bound, upper=upper_bound, axis=1)
# X = X.fillna(X.median())
# # 划分训练集和测试集 (80% 训练, 20% 测试)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 为了神经网络，对数据进行标准化 (RF和XGBoost可以直接用原始数据，但用标准化的数据也不影响它们)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
#
# # ---------------- 2. 定义并训练模型 ----------------
# # A. Random Forest (开启 CPU 全核并行)
# # 加上 n_jobs=-1，让服务器的所有 CPU 核心一起跑，能快非常多
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# rf_model.fit(X_train, y_train)
#
# # B. Neural Network (纯 CPU，预警：依然会比较慢)
# # 建议稍微降低 max_iter，比如 200，防止它无休止地训练
# nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
# nn_model.fit(X_train_scaled, y_train)
#
# # C. XGBoost (开启 4090 GPU 加速)
# # 加上 tree_method 和 device，瞬间秒杀 200 万条数据
# xgb_model = xgb.XGBRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     random_state=42,
#     tree_method='hist',
#     device='cuda'
# )
# xgb_model.fit(X_train, y_train)
#
# # ---------------- 3. 模型评估与比较 ----------------
# models = {
#     'Random Forest': (rf_model, X_test),
#     'Neural Network': (nn_model, X_test_scaled),
#     'XGBoost': (xgb_model, X_test)
# }
# #
# print("=== 模型评估结果 ===")
# for name, (model, test_data) in models.items():
#     y_pred = model.predict(test_data)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f"{name} -> MSE: {mse:.4f} | R2 Score: {r2:.4f}")
#
# # (在实际中，你会根据上面的输出选择一个表现最好的模型，例如 R2 最高的。假设是 XGBoost)
# best_model = rf_model
# #
# # # ---------------- 4. SHAP 分析最优模型 ----------------
# print("\n=== 开始 SHAP 分析 ===")
# best_model = rf_model
#
# X_shap_sample = X_test.sample(n=1000, random_state=42)
#
# explainer = shap.TreeExplainer(best_model)
# # 用抽样后的 1 万条数据计算 SHAP 值
# shap_values = explainer.shap_values(X_shap_sample)
#
# # 1. 保存全局特征重要性柱状图
# shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
# plt.tight_layout()
# plt.savefig("/home/featurize/work/RuochengG/shap_bar_xgb1.png", dpi=300, bbox_inches='tight')
# plt.close()
#
# # 2. 保存蜂巢散点图 (依然用抽样数据)
# shap.summary_plot(shap_values, X_shap_sample, show=False)
# plt.tight_layout()
# plt.savefig("/home/featurize/work/RuochengG/shap_summary_xgb1.png", dpi=300, bbox_inches='tight')
# plt.close()
#
# print("SHAP 图像已成功保存！")
