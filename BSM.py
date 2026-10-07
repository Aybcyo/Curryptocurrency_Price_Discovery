#!/usr/bin/env python
# coding: utf-8
import glob
from pyspark.sql import SparkSession
import os
os.environ['PYSPARK_PYTHON'] = "/environment/miniconda3/bin/python"

import re
import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
import pandas as pd
import os
import math
import logging
import scipy.stats as stats
logger = logging.getLogger(__name__)
from pyspark.sql import functions as F
import logging
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from numpy.linalg import inv
from pyspark.sql.functions import col, count, isnan, pandas_udf, to_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType, IntegerType

from pyspark.sql.functions import col, udf, lag, when
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from scipy.optimize import brentq
import math
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np

# 初始化 SparkSession
spark = SparkSession.builder.appName("PyCharm").getOrCreate()
spark = SparkSession.builder \
    .appName("CSV Processing") \
    .config("spark.network.timeout", "600s") \
    .config("spark.security.manager.enabled", "false") \
    .config("spark.executor.heartbeatInterval", "100s") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
    .getOrCreate()




# 定义 GBS 模型的输入限制
# class _GBS_Limits:
#     MAX32 = 2147483248.0
#
#     MIN_T = 1.0 / 1000.0  # requires some time left before expiration
#     MIN_X = 0.01
#     MIN_FS = 0.01
#
#     # Volatility smaller than 0.5% causes American Options calculations
#     # to fail (Number to large errors).
#     # GBS() should be OK with any positive number. Since vols less
#     # than 0.5% are expected to be extremely rare, and most likely bad inputs,
#     # _gbs() is assigned this limit too
#     MIN_V = 0.005
#
#     MAX_T = 100
#     MAX_X = MAX32
#     MAX_FS = MAX32
#
#     # Asian Option limits
#     # maximum TA is time to expiration for the option
#     MIN_TA = 0
#
#     # This model will work with higher values for b, r, and V. However, such values are extremely uncommon.
#     # To catch some common errors, interest rates and volatility is capped to 200%
#     # This reason for 2 (200%) is mostly to cause the library to throw an exceptions
#     # if a value like 15% is entered as 15 rather than 0.15)
#     MIN_b = -1
#     MIN_r = -1
#
#     MAX_b = 1
#     MAX_r = 2
#     MAX_V = 2
#
#
# # 定义异常类
# class GBS_InputError(Exception):
#   def __init__(self, mismatch):
#     Exception.__init__(self, mismatch)
#
#
# # This class defines the Exception that gets thrown when there is a calculation error
# class GBS_CalculationError(Exception):
#   def __init__(self, mismatch):
#     Exception.__init__(self, mismatch)
#
#
# # 测试输入是否合法
# def _gbs_test_inputs(fs, x, t, r, b, v):
#     if (x < _GBS_Limits.MIN_X) | (x > _GBS_Limits.MAX_X):
#       raise GBS_InputError(
#         "Invalid Input Strike Price (X). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X,
#                                                                                            _GBS_Limits.MAX_X))
#
#     if (fs < _GBS_Limits.MIN_FS) | (fs > _GBS_Limits.MAX_FS):
#       raise GBS_InputError(
#         "Invalid Input Forward/Spot Price (FS). Acceptable range for inputs is {1} to {2}".format(fs,
#                                                                                                   _GBS_Limits.MIN_FS,
#                                                                                                   _GBS_Limits.MAX_FS))
#
#     if (t < _GBS_Limits.MIN_T) | (t > _GBS_Limits.MAX_T):
#       raise GBS_InputError(
#         "Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T,
#                                                                                          _GBS_Limits.MAX_T))
#
#     if (b < _GBS_Limits.MIN_b) | (b > _GBS_Limits.MAX_b):
#       raise GBS_InputError(
#         "Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b,
#                                                                                                   _GBS_Limits.MIN_b,
#                                                                                                   _GBS_Limits.MAX_b))
#
#     if (r < _GBS_Limits.MIN_r) | (r > _GBS_Limits.MAX_r):
#       raise GBS_InputError(
#         "Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r,
#                                                                                                    _GBS_Limits.MIN_r,
#                                                                                                    _GBS_Limits.MAX_r))
#
#     if (v < _GBS_Limits.MIN_V) | (v > _GBS_Limits.MAX_V):
#       raise GBS_InputError(
#         "Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v,
#                                                                                                        _GBS_Limits.MIN_V,
#                                                                                                        _GBS_Limits.MAX_V))
#
#       # GBS 模型计算
# # def _gbs(fs, x, t, r, b, v):
# #       logger.debug("Debugging Information: _gbs()")
# #       # -----------
# #       # Test Inputs (throwing an exception on failure)
# #       _gbs_test_inputs(fs, x, t, r, b, v)
# #
# #       # -----------
# #       # Create preliminary calculations
# #       t__sqrt = math.sqrt(t)
# #       d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
# #       d2 = d1 - v * t__sqrt
# #       logger.debug("     Put Option")
# #       value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
# #       delta = -math.exp((b - r) * t) * norm.cdf(-d1)
# #       gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
# #       theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
# #         (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
# #       vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
# #       rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)
# #
# #       logger.debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
# #       logger.debug(
# #         "     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
# #                                                                                                      theta, vega,
#
# def _gbs(fs, x, t, r, b, v):
#
#       logger.debug("Debugging Information: _gbs()")
#       # -----------
#       # Test Inputs (throwing an exception on failure)
#       _gbs_test_inputs( fs, x, t, r, b, v)
#
#       # -----------
#       # Create preliminary calculations
#       t__sqrt = math.sqrt(t)
#       d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
#       d2 = d1 - v * t__sqrt
#
#       logger.debug("     Call Option")
#       value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
#       delta = math.exp((b - r) * t) * norm.cdf(d1)
#       gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
#       theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
#           (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
#       vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
#       rho = x * t * math.exp(-r * t) * norm.cdf(d2)
#
#       logger.debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
#       logger.debug("     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
#                                                                                                           theta, vega,
#                                                                                                           rho))
#
#
#       return float(value), float(delta), float(gamma), float(theta), float(vega), float(rho)
#
# # 近似隐含波动率
# def _approx_implied_vol(fs, x, t, r, b, cp):
#     b = r
#     ebrt = math.exp((b - r) * t)
#     ert = math.exp(-r * t)
#
#     a = math.sqrt(2 * math.pi) / (fs * ebrt + x * ert)
#
#     # if option_type == "c":
#     payoff = fs * ebrt - x * ert
#     # else:
#     # payoff = x * ert - fs * ebrt
#
#     b = cp - payoff / 2
#     # b = r
#     c = (payoff ** 2) / math.pi
#
#     v = (a * (b + math.sqrt(b ** 2 + c))) / math.sqrt(t)
#     return float(v)
#
# # 牛顿法计算隐含波动率
# def _newton_implied_vol(val_fn, x, fs, t, b, r, cp, precision=.00001, max_steps=500):
#   # Estimate starting Vol, making sure it is allowable range
#   v = _approx_implied_vol(fs, x, t, r, b, cp)
#   v = max(_GBS_Limits.MIN_V, v)
#   v = min(_GBS_Limits.MAX_V, v)
#
#   # Calculate the value at the estimated vol
#   value, delta, gamma, theta, vega, rho = val_fn(fs, x, t, r, b, v)
#   min_diff = abs(cp - value)
#
#   logger.debug("-----")
#   logger.debug("Debug info for: _Newton_ImpliedVol()")
#   logger.debug("    Vinitial={0}".format(v))
#
#   # Newton-Raphson Search
#   countr = 0
#   while precision <= abs(cp - value) <= min_diff and countr < max_steps:
#
#     v = v - (value - cp) / vega
#     if (v > _GBS_Limits.MAX_V) or (v < _GBS_Limits.MIN_V):
#       logger.debug("    Volatility out of bounds")
#       break
#
#     value, delta, gamma, theta, vega, rho = val_fn(fs, x, t, r, b, v)
#     min_diff = min(abs(cp - value), min_diff)
#
#     # keep track of how many loops
#     countr += 1
#     logger.debug("     IVOL STEP {0}. v={1}".format(countr, v))
#     if abs(cp - value) < precision:
#       # the search function converged
#
#       return float(v)
#     else:
#       # if the search function didn't converge, try a bisection search
#       return _bisection_implied_vol(val_fn, fs, x, t, r, b, cp, precision, max_steps)
#
# def _bisection_implied_vol(val_fn,  fs, x, t, r, b, cp, precision=.00001, max_steps=500):
#   logger.debug("-----")
#   logger.debug("Debug info for: _bisection_implied_vol()")
#
#   # Estimate Upper and Lower bounds on volatility
#   # Assume American Implied vol is within +/- 50% of the GBS Implied Vol
#   v_mid = _approx_implied_vol( fs, x, t, r, b, cp)
#
#   if (v_mid <= _GBS_Limits.MIN_V) or (v_mid >= _GBS_Limits.MAX_V):
#     # if the volatility estimate is out of bounds, search entire allowed vol space
#     v_low = _GBS_Limits.MIN_V
#     v_high = _GBS_Limits.MAX_V
#     v_mid = (v_low + v_high) / 2
#   else:
#     # reduce the size of the vol space
#     v_low = max(_GBS_Limits.MIN_V, v_mid * .5)
#     v_high = min(_GBS_Limits.MAX_V, v_mid * 1.5)
#
#   # Estimate the high/low bounds on price
#   cp_mid = val_fn( fs, x, t, r, b, v_mid)[0]
#
#   # initialize bisection loop
#   current_step = 0
#   diff = abs(cp - cp_mid)
#
#   logger.debug("     American IVOL starting conditions: CP={0} cp_mid={1}".format(cp, cp_mid))
#   logger.debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))
#   while (diff > precision) and (current_step < max_steps):
#     current_step += 1
#
#     # Cut the search area in half
#     if cp_mid < cp:
#       v_low = v_mid
#     else:
#       v_high = v_mid
#
#     cp_low = val_fn( fs, x, t, r, b, v_low)[0]
#     cp_high = val_fn( fs, x, t, r, b, v_high)[0]
#
#     v_mid = v_low + (cp - cp_low) * (v_high - v_low) / (cp_high - cp_low)
#     v_mid = max(_GBS_Limits.MIN_V, v_mid)  # enforce high/low bounds
#     v_mid = min(_GBS_Limits.MAX_V, v_mid)  # enforce high/low bounds
#
#     cp_mid = val_fn(fs, x, t, r, b, v_mid)[0]
#     diff = abs(cp - cp_mid)
#
#     logger.debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))
#
#     # return output
#   if abs(cp - cp_mid) < precision:
#     return float(v_mid)
#   else:
#     raise GBS_CalculationError(
#       "Implied Vol did not converge. Best Guess={0}, Price diff={1}, Required Precision={2}".format(v_mid, diff,
#                                                                                                     precision))
#
#
# # 计算隐含波动率
# def _gbs_implied_vol(fs, x, t, r, b, cp, precision=.00001, max_steps=500):
#     return _newton_implied_vol(_gbs, x, fs, t, b, r, cp, precision, max_steps)
#
# # 注册 UDF
# @udf(returnType=DoubleType())
# def calculate_iv(fs, x, t, r, b, cp):
#     try:
#         return _gbs_implied_vol(fs, x, t, r, b, cp)
#     except:
#         return 0.0  # 返回默认值
#
# @udf(returnType=DoubleType())
# def calculate_bs(fs, x, t, r, v):
#     try:
#         value, delta, gamma, theta, vega, rho = _gbs(fs, x, t, r, r, v)
#         return float(value)  # 返回期权价格
#     except:
#         return 0.0  # 返回默认值
#
#
# @udf(returnType=DoubleType())
# def implied_stock_price(fs, x, t, r, v, Call_market_price):
#     try:
#         # 定义目标函数
#         def target_function(fs_target):
#             return float(calculate_bs(fs_target, x, t, r, v) - Call_market_price)
#         # 使用 brentq 求解
#         return brentq(target_function, 50, 200)
#     except:
#         return 0.0
# # 主函数
# if __name__ == "__main__":
#     # 读取数据
#     # path = "/Volumes/PortableSSD/strike_price/BTC"
#     # df = spark.read.csv(f"{path}/*.csv", header=True, inferSchema=True)
#     df = spark.read.csv("/Volumes/PortableSSD/strike_price/BTC/part-00067-243da852-1383-47e1-ad58-589df428520d-c000.csv", header=True, inferSchema=True)
#     df = df.filter(~col("option_symbol").contains("-P"))
#
#     # 过滤无效数据
#     # df = df.filter((col("spot_close") > 0) & (col("strike_price") > 0) & (col("Time_to_Expiration") > 0))
#
#     # 计算 prev_spot_close 和 prev_option_close
#     window_spec = Window.partitionBy("option_symbol").orderBy("close_time")
#     df = df.withColumn("prev_spot_close", lag("spot_close").over(window_spec))
#     df = df.withColumn("prev_option_close", lag("option_close").over(window_spec))
#     df = df.fillna(0, subset=["prev_spot_close", "prev_option_close"])
#
#     # 计算 IV 和 BS_price
#     df = df.withColumn("rate", col("interpolated_rate") * 0.01)
#     df = df.withColumn("IV", calculate_iv(col("prev_spot_close"), col("strike_price"), col("Time_to_Expiration"), col("rate"), col("rate"), col("prev_option_close")))
#     # df = df.withColumn("BS_price", calculate_bs(col("spot_close"), col("strike_price"), col("Time_to_Expiration"), col("rate"), col("IV")))
#     # df = df.withColumn('Implied_stock_price',implied_stock_price(col("spot_close"), col("strike_price"), col("Time_to_Expiration"), col("rate"), col("IV"), col('option_close')))
#     df = df.withColumn('Implied_stock_price',brentq(implied_stock_price, 50, 200, args=(col("strike_price"), col("Time_to_Expiration"), col("rate"), col("IV"), col('option_close'))))
#
#     # 保存结果
#     # df.write.csv("/Volumes/PortableSSD/BSM/BTC/Call.csv", header=True)
#     # df.write.csv("/Users/ruocheng.gu/Desktop/call.csv", header=True)
#     # 打印结果
#     df.select("option_symbol", "strike_price", "Time_to_Expiration", "IV", "Implied_stock_price").show()
#
#
#     # 停止 SparkSession
#     spark.stop()



# 定义 GBS 模型的输入限制
class _GBS_Limits:
    MAX32 = 2147483248.0
    MIN_T = 1.0 / 1000.0
    MIN_X = 0.01
    MIN_FS = 0.01
    MIN_V = 0.005
    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32
    MIN_TA = 0
    MIN_b = -1
    MIN_r = -1
    MAX_b = 1
    MAX_r = 2
    MAX_V = 2

# 定义异常类
class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

class GBS_CalculationError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

# 测试输入是否合法
def _gbs_test_inputs(fs, x, t, r, b, v):
    if (x < _GBS_Limits.MIN_X) | (x > _GBS_Limits.MAX_X):
        raise GBS_InputError("Invalid Input Strike Price (X). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X, _GBS_Limits.MAX_X))
    if (fs < _GBS_Limits.MIN_FS) | (fs > _GBS_Limits.MAX_FS):
        raise GBS_InputError("Invalid Input Forward/Spot Price (FS). Acceptable range for inputs is {1} to {2}".format(fs, _GBS_Limits.MIN_FS, _GBS_Limits.MAX_FS))
    if (t < _GBS_Limits.MIN_T) | (t > _GBS_Limits.MAX_T):
        raise GBS_InputError("Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T, _GBS_Limits.MAX_T))
    if (b < _GBS_Limits.MIN_b) | (b > _GBS_Limits.MAX_b):
        raise GBS_InputError("Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b, _GBS_Limits.MIN_b, _GBS_Limits.MAX_b))
    if (r < _GBS_Limits.MIN_r) | (r > _GBS_Limits.MAX_r):
        raise GBS_InputError("Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r, _GBS_Limits.MIN_r, _GBS_Limits.MAX_r))
    if (v < _GBS_Limits.MIN_V) | (v > _GBS_Limits.MAX_V):
        raise GBS_InputError("Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v, _GBS_Limits.MIN_V, _GBS_Limits.MAX_V))



def _gbs(fs, x, t, r, b, v):
#
      logger.debug("Debugging Information: _gbs()")
      # -----------
      # Test Inputs (throwing an exception on failure)
      _gbs_test_inputs( fs, x, t, r, b, v)

      # -----------
      # Create preliminary calculations
      t__sqrt = math.sqrt(t)
      d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
      d2 = d1 - v * t__sqrt
      # logger.debug("     Call Option")
      # value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
      # delta = math.exp((b - r) * t) * norm.cdf(d1)
      # gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
      # theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
      #     (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
      # vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
      # rho = x * t * math.exp(-r * t) * norm.cdf(d2)
      logger.debug("     Put Option")
      value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
      delta = -math.exp((b - r) * t) * norm.cdf(-d1)
      gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
      theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
        (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
      vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
      rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

      logger.debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
      logger.debug("     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
                                                                                                          theta, vega,
                                                                                                          rho))


      return float(value), float(delta), float(gamma), float(theta), float(vega), float(rho)


# 近似隐含波动率
# call
def _approx_implied_vol(fs, x, t, r, b, cp):

    b = r
    ebrt = math.exp((b - r) * t)
    ert = math.exp(-r * t)

    a = math.sqrt(2 * math.pi) / (fs * ebrt + x * ert)

    # if option_type == "c":
    payoff = fs * ebrt - x * ert
    # else:
    # payoff = x * ert - fs * ebrt

    b = cp - payoff / 2
    # b = r
    c = (payoff ** 2) / math.pi

    v = (a * (b + math.sqrt(b ** 2 + c))) / math.sqrt(t)
    return float(v)

# 牛顿法计算隐含波动率
def _newton_implied_vol(val_fn, x, fs, t, b, r, cp, precision=.00001, max_steps=500):

    #  Estimate starting Vol, making sure it is allowable range
    v = _approx_implied_vol(fs, x, t, r, b, cp)
    v = max(_GBS_Limits.MIN_V, v)
    v = min(_GBS_Limits.MAX_V, v)

    # Calculate the value at the estimated vol
    value, delta, gamma, theta, vega, rho = val_fn(fs, x, t, r, b, v)
    min_diff = abs(cp - value)

    logger.debug("-----")
    logger.debug("Debug info for: _Newton_ImpliedVol()")
    logger.debug("    Vinitial={0}".format(v))

    # Newton-Raphson Search
    countr = 0
    while precision <= abs(cp - value) <= min_diff and countr < max_steps:

      v = v - (value - cp) / vega
      if (v > _GBS_Limits.MAX_V) or (v < _GBS_Limits.MIN_V):
        logger.debug("    Volatility out of bounds")
        break

      value, delta, gamma, theta, vega, rho = val_fn(fs, x, t, r, b, v)
      min_diff = min(abs(cp - value), min_diff)

      # keep track of how many loops
      countr += 1
      logger.debug("     IVOL STEP {0}. v={1}".format(countr, v))
      if abs(cp - value) < precision:
        # the search function converged

        return float(v)
      else:
        # if the search function didn't converge, try a bisection search
        return _bisection_implied_vol(val_fn, fs, x, t, r, b, cp, precision, max_steps)



# 二分法计算隐含波动率
def _bisection_implied_vol(val_fn, fs, x, t, r, b, cp, precision=.00001, max_steps=500):

    logger.debug("-----")
    logger.debug("Debug info for: _bisection_implied_vol()")

    # Estimate Upper and Lower bounds on volatility
    # Assume American Implied vol is within +/- 50% of the GBS Implied Vol
    v_mid = _approx_implied_vol( fs, x, t, r, b, cp)

    if (v_mid <= _GBS_Limits.MIN_V) or (v_mid >= _GBS_Limits.MAX_V):
      # if the volatility estimate is out of bounds, search entire allowed vol space
      v_low = _GBS_Limits.MIN_V
      v_high = _GBS_Limits.MAX_V
      v_mid = (v_low + v_high) / 2
    else:
      # reduce the size of the vol space
      v_low = max(_GBS_Limits.MIN_V, v_mid * .5)
      v_high = min(_GBS_Limits.MAX_V, v_mid * 1.5)

    # Estimate the high/low bounds on price
    cp_mid = val_fn( fs, x, t, r, b, v_mid)[0]

    # initialize bisection loop
    current_step = 0
    diff = abs(cp - cp_mid)

    logger.debug("     American IVOL starting conditions: CP={0} cp_mid={1}".format(cp, cp_mid))
    logger.debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))
    while (diff > precision) and (current_step < max_steps):
      current_step += 1

      # Cut the search area in half
      if cp_mid < cp:
        v_low = v_mid
      else:
        v_high = v_mid

      cp_low = val_fn( fs, x, t, r, b, v_low)[0]
      cp_high = val_fn( fs, x, t, r, b, v_high)[0]

      v_mid = v_low + (cp - cp_low) * (v_high - v_low) / (cp_high - cp_low)
      v_mid = max(_GBS_Limits.MIN_V, v_mid)  # enforce high/low bounds
      v_mid = min(_GBS_Limits.MAX_V, v_mid)  # enforce high/low bounds

      cp_mid = val_fn(fs, x, t, r, b, v_mid)[0]
      diff = abs(cp - cp_mid)

      logger.debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))

      # return output
    if abs(cp - cp_mid) < precision:
      return float(v_mid)
    else:
      raise GBS_CalculationError(
        "Implied Vol did not converge. Best Guess={0}, Price diff={1}, Required Precision={2}".format(v_mid, diff,
                                                                                                    precision))

# call
# def implied_stock_price(C_model, C_market, K, T, r, sigma, tolerance=1e-6, max_iterations=100):
#   # if sigma <= 0 or T <= 0:
#   #   return None
#   S0 = K  # Initial guess
#   for i in range(max_iterations):
#     epsilon = C_market - C_model
#     if abs(epsilon) <= tolerance:
#       return S0
#     d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     delta = stats.norm.cdf(d1)
#     # if delta == 0:
#     #   return None
#     S0 += epsilon / delta
#   return float(S0)  # Return the last estimate if max iterations reached

# put
def implied_stock_price(P_model, P_market, K, T, r, sigma, tolerance=1e-6, max_iterations=100):
  """
  使用牛顿迭代法计算看跌期权的隐含股价
  :param P_model:  模型计算出的看跌期权价格（Black-Scholes公式结果）
  :param P_market: 市场观察到的看跌期权价格
  :param K:        行权价
  :param T:        到期时间（年）
  :param r:        无风险利率
  :param sigma:    波动率
  :param tolerance: 收敛容忍度
  :param max_iterations: 最大迭代次数
  :return: 隐含股价 S0
  """
  S0 = K
  for i in range(max_iterations):
    epsilon = P_market - P_model
    if abs(epsilon) <= tolerance:
      return S0
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_put = stats.norm.cdf(d1) - 1

    # if abs(delta_put) < 1e-10:
    #   break
    if delta_put == 0:
      return None

    S0 += epsilon / delta_put

  return float(S0)

# ILS

def process_group(data):
  price_discovery_measures = []
  window_data_list = []

  # 示例：测试不同窗口大小（如1和2天，假设数据按分钟记录）
  for j in range(1,6):
    window_size = j * 60  # 根据实际时间单位调整
    for i in range(len(data) - window_size + 1):
      window_data = data.iloc[i:i + window_size]

      # 检查缺失值
      if window_data[['implied_stock_price', 'spot_close']].isnull().any().any():
        continue

      # 计算对数价格
      log_S1 = np.log(window_data['implied_stock_price'])
      log_S2 = np.log(window_data['spot_close'])
      df_2 = pd.DataFrame({'log_S1': log_S1, 'log_S2': log_S2}).dropna()

      if df_2.empty:
        continue

      try:
        # 选择滞后阶数
        lag_order = select_order(df_2, maxlags=5, deterministic="ci").aic  # 降低maxlags
        vecm = VECM(df_2, k_ar_diff=lag_order, coint_rank=1, deterministic="ci")
        vecm_fit = vecm.fit()

        # 提取参数并计算指标
        beta = vecm_fit.beta
        alpha = vecm_fit.alpha
        sigma_u = vecm_fit.sigma_u
        pi_matrix = np.dot(alpha, beta.T)

        # 确保sigma_u正定
        if np.any(np.linalg.eigvals(sigma_u) <= 0):
          raise ValueError("sigma_u is not positive definite")

        omega_inv = np.linalg.inv(sigma_u)
        cholesky_omega = np.linalg.cholesky(sigma_u)
        numerator = (cholesky_omega @ pi_matrix) ** 2
        denominator = np.trace(omega_inv @ pi_matrix @ pi_matrix.T)

        if denominator == 0:
          continue  # 避免除以零

        IS1 = numerator[0, 0] / denominator
        IS2 = numerator[1, 0] / denominator
        total_variance = np.trace(sigma_u)
        CS1 = sigma_u[0, 0] / total_variance
        CS2 = sigma_u[1, 1] / total_variance

        IL1 = (IS1 / IS2) * (CS2 / CS1) if IS2 != 0 else 0
        IL2 = (IS2 / IS1) * (CS1 / CS2) if IS1 != 0 else 0
        ILS1 = IL1 / (IL1 + IL2) if (IL1 + IL2) != 0 else 0.5
        ILS2 = IL2 / (IL1 + IL2) if (IL1 + IL2) != 0 else 0.5

        # 标记原数据窗口的结果
        result_row = window_data.copy()
        result_row['ILS1'] = ILS1
        result_row['ILS2'] = ILS2
        result_row['window_size'] = window_size
        window_data_list.append(result_row)

      except Exception as e:
        print(f"窗口 {i} 处理失败: {e}")

  return pd.concat(window_data_list) if window_data_list else pd.DataFrame()


# 计算隐含波动率
def _gbs_implied_vol(fs, x, t, r, b, cp, precision=.00001, max_steps=500):
    return _newton_implied_vol(_gbs, x, fs, t, b, r, cp, precision, max_steps)

# 注册 UDF
@udf(returnType=DoubleType())
def calculate_iv(fs, x, t, r, b, cp):
    try:
        return _gbs_implied_vol(fs, x, t, r, b, cp)
    except:
        return 0.0  # 返回默认值

@udf(returnType=DoubleType())
def calculate_bs(fs, x, t, r, v):
    try:
        value, delta, gamma, theta, vega, rho = _gbs(fs, x, t, r, r, v)
        return float(value)  # 返回期权价格
    except:
        return 0.0  # 返回默认值

@udf(returnType=DoubleType())
def calculate_ims(C_model, C_market, K, T, r, sigma):
  try:
    return implied_stock_price(C_model, C_market, K, T, r, sigma)
  except:
    return 0.0

import psutil
import time
def monitor_memory():
  process = psutil.Process()
  print("Monitoring memeory useage...")
  while True:
    print(f"Memory (RSS):"
          f"{process.memory_info().rss/(1024*1024)}MB")
    time.sleep(1)
# 主函数
if __name__ == "__main__":
  from pyspark.sql import SparkSession
  from pyspark.storagelevel import StorageLevel
  import pyspark.sql.functions as F
  from pyspark.sql.functions import pandas_udf, PandasUDFType

  try:

      path = "/home/featurize/work/RuochengG/data_classification/Put/ETH/short_itm.csv"
      df = spark.read.csv(f"{path}/*.csv", header=True, inferSchema=True)
      # df = df.filter(~col("option_symbol").contains("-P"))
      df.persist(StorageLevel.MEMORY_AND_DISK)

      # 过滤无效数据
      # df = df.filter((col("spot_close") > 0) & (col("strike_price") > 0) & (col("Time_to_Expiration") > 0))

      # 计算 prev_spot_close 和 prev_option_close
      window_spec = Window.partitionBy("option_symbol").orderBy("close_time")
      df = df.withColumn("prev_spot_close", lag("spot_close").over(window_spec))
      df = df.withColumn("prev_option_close", lag("true_option_price").over(window_spec))

      df = df.fillna(0, subset=["prev_spot_close", "prev_option_close"])

      # 计算 IV 和 BS_price
      df = df.withColumn("rate", col("interpolated_rate") * 0.01)
      df = df.withColumn("implied_volatility", calculate_iv(col("prev_spot_close"), col("strike_price"), col("Time_to_Expiration"), col("rate"), col("rate"), col("prev_option_close")))
      df = df.withColumn("implied_option_price", calculate_bs(col("spot_close"), col("strike_price"), col("Time_to_Expiration"), col("rate"), col("implied_volatility")))
      df = df.withColumn("implied_stock_price", calculate_ims(col("implied_option_price"),col("true_option_price"), col("strike_price"),col("Time_to_Expiration"), col("rate"), col("implied_volatility")))
      df.write.csv("/home/featurize/work/RuochengG/data_bsm/Put/ETH/short_itm",header = True)
      df.show()
  finally:
    spark.stop()




# df_measures = pd.DataFrame(price_discovery_measures)
# print(df_measures)





