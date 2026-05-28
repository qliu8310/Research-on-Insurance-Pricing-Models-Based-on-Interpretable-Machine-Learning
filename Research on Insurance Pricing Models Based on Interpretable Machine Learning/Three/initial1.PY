# sys and os imports
# 导入系统和操作系统相关的库，用于文件路径操作、系统命令执行等
import os
import sys
import platform
import subprocess
import re           # 正则表达式，用于文本处理
import warnings     # 用于控制警告信息的显示

# display and plotting
# 导入显示和绘图工具
from IPython.display import Image  # 在Jupyter Notebook中显示图片
import plotly.express as px        # Plotly的高级接口，用于快速画交互式图表
import plotly.graph_objects as go  # Plotly的底层接口，用于定制化图表

# data
# 导入数据处理的核心库
import numpy as np                 # 数值计算
import pandas as pd                # 表格数据处理
from dataclasses import dataclass, field # 用于创建结构化的数据类（Python 3.7+特性）

# modelling
# 导入建模相关的库
import random                      # 生成随机数（常用于设置随机种子）

# modelling scikit-learn
# 导入传统的机器学习库 Scikit-learn
import sklearn as sk
from sklearn.linear_model import PoissonRegressor # 泊松回归模型，保险定价中常用的广义线性模型(GLM)，常用于索赔频率建模
from sklearn.preprocessing import StandardScaler  # 数据标准化工具

# modelling tensorflow
# 导入深度学习框架 TensorFlow 和 Keras
import tensorflow as tf
import keras
# 导入激活函数：
# tanh: 双曲正切
# exponential: 指数激活函数（对于输出必须为正数的模型如泊松回归非常重要）
# gelu: 高斯误差线性单元（Transformer模型中常用的激活函数）
from keras.activations import (tanh, exponential, gelu)

# saveing & time
# 导入保存和时间处理库
import pickle      # 用于序列化和反序列化Python对象（保存模型或数据）
import time        # 计时工具
import datetime    # 日期处理

# for multiprocessing
# 导入多进程处理库，用于并行计算加速
from multiprocessing import Process
import logging     # 日志记录工具

# 导入 Keras 的回调函数
from tensorflow.keras.callbacks import EarlyStopping # 早停法：当模型在验证集上不再提升时停止训练，防止过拟合

# set the path to the storage folder:
# 设置项目的主存储路径（注意：这里使用的是 Windows 风格的路径）
storage_path = "D:/法国第三方/Enhancing_actuarial_non-life_pricing_models_via_transformers_Public-main"

# import my classes for the ft-transformer models:
# ----------------------
# 将 helper 文件夹添加到系统路径中，以便能导入自定义的模块
sys.path.insert(1, f'{storage_path}/helper')

# 导入自定义的模块（位于 helper 文件夹下）
# Main model classes: 包含主要的模型定义，推测是关于 Actuarial (精算) 增强模型的类
import main_model_classes as EnhActuar 
# Helper: 包含辅助函数（如数据清洗、预处理等）
import helper as helper

# 设置 Pandas 显示选项：显示所有列（不折叠），方便查看宽表数据
pd.set_option('display.max_columns', None)


