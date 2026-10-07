# Author: Gretchen_RuochengGu
# CreateTime: 2025/2/16
#  Filename: interpolation
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("PyCharm").getOrCreate()
spark = SparkSession.builder \
    .appName("CSV Processing") \
    .config("spark.security.manager.enabled", "false") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
print(spark.version)
path = "/Volumes/PortableSSD/rate/merge/ETH"
df = spark.read.csv(f"{path}/*.csv", header=True, inferSchema=True)
df = df.drop('1_5Mo')

# 计算 `expiration_date` 和 `close_time` 之间的月数差
df = df.withColumn('month_diff', F.datediff('expiration_date', 'close_time') / 30)  # 近似为月数差


time_units = [
    ('1Mo', 1), ('2Mo', 2), ('3Mo', 3), ('4Mo', 4), ('6Mo', 6),
    ('1Yr', 12), ('2Yr', 24), ('3Yr', 36), ('5Yr', 60), ('7Yr', 84), ('10Yr', 120),
    ('20Yr', 240), ('30Yr', 360)
]


def parse_columns(df, time_units):
  for col, months in time_units:
    df = df.withColumn(col, F.col(col).cast('float'))  # 确保数据类型为 float
  return df


df = parse_columns(df, time_units)

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


def linear_interpolate(df, time_units, time_col='month_diff'):
  def interpolate(row):
    month_diff = row[time_col]
    lower_col, lower_months = None, None
    higher_col, higher_months = None, None
    lower_rate, higher_rate = None, None

    # 如果 month_diff 小于 1，直接使用 1Mo 的 rate
    if month_diff < 1:
      lower_col, lower_months = time_units[0]
      lower_rate = row[lower_col]
      return lower_rate

    # 查找最接近的两个时间区间
    for i in range(len(time_units) - 1):
      lower_col, lower_months = time_units[i]
      higher_col, higher_months = time_units[i + 1]

      # 如果 month_diff 在这两个区间之间，执行插值
      if lower_months <= month_diff <= higher_months:
        lower_rate = row[lower_col]
        higher_rate = row[higher_col]
        break

    # 如果没有找到合适的区间或者 rate 为 NULL，则使用之前一个月/之后一个月的 rate 填充
    if lower_rate is None and i > 0:
      lower_col, lower_months = time_units[i - 1]  # 使用之前一个月的 rate
      lower_rate = row[lower_col]

    if higher_rate is None and i < len(time_units) - 1:
      higher_col, higher_months = time_units[i + 1]  # 使用之后一个月的 rate
      higher_rate = row[higher_col]

    # 如果仍然没有找到合适的 rate，则返回 None 或其他值
    if lower_rate is None or higher_rate is None:
      return None

    # 线性插值
    return lower_rate + (higher_rate - lower_rate) * (month_diff - lower_months) / (higher_months - lower_months)

  # 创建 UDF
  interpolate_udf = F.udf(interpolate, FloatType())

  # 使用 UDF 插入列
  return df.withColumn('interpolated_rate',
                       interpolate_udf(F.struct([F.col(c) for c in [col for col, _ in time_units] + [time_col]])))


df = linear_interpolate(df, time_units)


df.show()

output_path = '/Volumes/PortableSSD/rate/interpolation/ETH'
df.write \
    .option("header", "true") \
    .csv(output_path)


