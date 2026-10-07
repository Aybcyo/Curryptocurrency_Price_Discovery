# Author: Gretchen_RuochengGu
# CreateTime: 2025/4/1
#  Filename: classify_9
# BTC: short_itm, short_atm, short_otm, mid_short_itm, mid_atm, mid_short_otm, long_itm, long_atm, long_otm
# ETH: short_itm, short_atm, short_otm
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PyCharm").getOrCreate()
from pyspark.sql.functions import col, udf, lag, when
from pyspark.sql.functions import col, count, isnan, pandas_udf, to_timestamp
spark = SparkSession.builder \
    .appName("CSV Processing") \
    .config("spark.security.manager.enabled", "false") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

path = "/home/featurize/work/RuochengG/strike_price/BTC"
df = spark.read.csv(f"{path}/*.csv", header=True, inferSchema=True)
df = df.filter(~col("option_symbol").contains("-C"))
df = df.withColumn("spot_close_time", to_timestamp(df["spot_close_time"]))
print(df.count())
# short
df_short = df[(df['month_diff'] >= 1.00) & (df['month_diff'] <= 3.00)]
print(df_short.count())
df_short_itm = df_short[(df_short['spot_close'] / df_short["strike_price"] >= 0.8) &
                        (df_short['spot_close'] / df_short["strike_price"] < 1)]
df_short_atm = df_short[(df_short['strike_price'] / df_short['spot_close'] >= 0.99) &
                        (df_short['strike_price'] / df_short['spot_close'] <= 1.01)]
df_short_otm = df_short[(df_short['spot_close'] / df_short["strike_price"] >1) &
                        (df_short['spot_close'] / df_short["strike_price"] <= 1.2)]
# mid
df_mid = df[(df['month_diff'] > 3.00) & (df['month_diff'] <= 6.00)]
print(df_mid.count())
df_mid_itm = df_mid[(df_mid['spot_close'] / df_mid["strike_price"] >= 0.8) &
                        (df_short['spot_close'] / df_short["strike_price"] < 1)]
df_mid_atm = df_mid[(df_mid['strike_price'] / df_mid['spot_close'] >= 0.99) &
                        (df_mid['strike_price'] / df_mid['spot_close'] <= 1.01)]
df_mid_otm = df_mid[(df_mid['spot_close'] / df_mid["strike_price"] >1) &
                        (df_mid['spot_close'] / df_mid["strike_price"] <= 1.2)]
# long
df_long = df[(df['month_diff'] > 6.00) & (df['month_diff'] <= 12.00)]
print(df_long.count())
df_long_itm = df_long[(df_long['spot_close'] / df_long["strike_price"] >= 0.8) &
                        (df_long['spot_close'] / df_long["strike_price"] < 1)]
df_long_atm = df_long[(df_long['strike_price'] / df_long['spot_close'] >= 0.99) &
                        (df_long['strike_price'] / df_long['spot_close'] <= 1.01)]
df_long_otm = df_long[(df_long['spot_close'] / df_long["strike_price"] >1) &
                        (df_long['spot_close'] / df_long["strike_price"] <= 1.2)]
# df_short_otm = df_short_otm[df_short_otm['spot_close'] >= 0]
# df_short_otm = df_short_otm.dropna(subset=['implied_stock_price', 'spot_close'])
df_short_itm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/short_itm.csv", header = True)
df_short_atm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/short_atm.csv", header = True)
df_short_otm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/short_otm.csv", header = True)
df_mid_itm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/mid_itm.csv", header = True)
df_mid_atm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/mid_atm.csv", header = True)
df_mid_otm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/mid_otm.csv", header = True)
df_long_itm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/long_itm.csv", header = True)
df_long_atm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/long_atm.csv", header = True)
df_long_otm.write.csv("/home/featurize/work/RuochengG/data_classification/Put/BTC/long_otm.csv", header = True)
