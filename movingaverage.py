# Author: Gretchen_RuochengGu
# CreateTime: 2025/7/28
#  Filename: movingaverage
# from pyspark.sql import SparkSession
# import os
# os.environ['PYSPARK_PYTHON'] = "/environment/miniconda3/bin/python"
import numpy as np
import pandas as pd
# from pyspark.sql.window import Window
# from statsmodels.tsa.vector_ar.vecm import VECM, select_order
# from numpy.linalg import inv
# from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType, IntegerType
# spark = SparkSession.builder.appName("PyCharm").getOrCreate()
# spark = SparkSession.builder \
#     .appName("CSV Processing") \
#     .config("spark.network.timeout", "600s") \
#     .config("spark.security.manager.enabled", "false") \
#     .config("spark.executor.heartbeatInterval", "100s") \
#     .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
#     .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
#     .config("spark.executor.memory", "16g") \
#     .config("spark.driver.memory", "16g") \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
#     .getOrCreate()
# spark.conf.set("spark.sql.execution.arrow.enabled", "false")


# 加载数据
import pandas as pd
import glob
import os
import csv

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
folder_path = "/home/featurize/work/RuochengG/data_ILS_ws1/Put/ETH/mid_atm"
df = read_csv_files(folder_path)
print(df.head(5))

# 设置窗口大小
window_size = 60

# 计算移动平均
df['ILS1_MA60'] = df['ILS1'].rolling(window=window_size).mean()
df['ILS2_MA60'] = df['ILS2'].rolling(window=window_size).mean()


# print(df[['row_num', 'ILS1', 'ILS1_MA60', 'ILS2', 'ILS2_MA60']].head(70))
print(df.head(70))
# 可选：保存结果
# df.to_csv("/home/featurize/work/RuochengG/data_with_move_ave/Put/ETH/mid_atm.csv", index=False)
