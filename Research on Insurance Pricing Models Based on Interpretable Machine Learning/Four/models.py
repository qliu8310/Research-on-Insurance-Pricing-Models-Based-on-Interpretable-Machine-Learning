import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# ===================== 全局配置（统一管理）=====================
# 1. 绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.titlesize"] = 27  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 17  
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['axes.titleweight'] = 'bold'   # 标题加粗
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 箱体颜色
model_colors = {  # 模型配色（统一）
    'GLM': 'dodgerblue',
    'XGBoost': 'orange',
    'Wide&Deep': 'limegreen',
    'NN-Reg': 'red',
    'ResNet': 'yellow',
    'Transformer': 'blueviolet',
    'EBM': 'black'
}
model_labels = ['GLM', 'NN-Reg', 'ResNet', 'Transformer', 'Wide&Deep', 'XGBoost', 'EBM']  # 统一标签

# 2. 路径配置（集中管理，便于修改）
BASE_PATH = r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本'
PRED_PATH = f'{BASE_PATH}\\predictions'
SAVE_PATH = BASE_PATH
PAPER_SAVE_PATH = r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\论文'

# 3. 分组配置
groups = {
    '1_3': {'name': 'H->D', 'weight_col': 'sum_hyear'},
    '1_2': {'name': 'H->L', 'weight_col': 'sum_hyear'},
    '2_3': {'name': 'L->D', 'weight_col': 'sum_lyear'},
    '2_1': {'name': 'L->H', 'weight_col': 'sum_lyear'}
}

# ===================== 通用函数封装（避免重复）=====================
def load_group_data(group_key):
    """加载单个分组的所有数据"""
    try:
        # 原始数据
        df = pd.read_csv(f'{BASE_PATH}\\分组{group_key}.csv')
        # 拟合/预测数据
        fitted = pd.read_excel(f'{PRED_PATH}\\fitted_{group_key}.xlsx')
        predictions = pd.read_excel(f'{PRED_PATH}\\predictions_{group_key}.xlsx')
        fitted_predictions = pd.read_excel(f'{PRED_PATH}\\fitted_predictions_{group_key}.xlsx')
        
        # 拆分训练/测试集
        train = df.loc[df['time'].isin([1, 3, 5.5, 8.5, 11.5])]
        test = df.loc[df['time'].isin([14.5, 18])]
        test_reset = test.reset_index(drop=True)
        
        # 拼接数据
        new_df = pd.concat([df, fitted_predictions], axis=1)
        new_train = pd.concat([train, fitted], axis=1)
        new_test = pd.concat([test_reset, predictions], axis=1)
        
        return {
            'df': df, 'train': train, 'test': test, 'test_reset': test_reset,
            'new_df': new_df, 'new_train': new_train, 'new_test': new_test
        }
    except Exception as e:
        print(f'加载分组{group_key}数据失败：{e}')
        return None

def calculate_metrics(y_true, y_pred, sample_weight=None, eps=1e-10):
    """
    计算模型评估指标：MSE + 加权泊松偏差
    eps: 防止log(0)或除以0的极小值
    """
    # 转换为数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # 2. 泊松偏差（处理边界情况）
    y_true_safe = np.maximum(y_true, eps)
    y_pred_safe = np.maximum(y_pred, eps)
    
    deviance_per_obs = 2 * (y_true * np.log(y_true_safe / y_pred_safe) - (y_true - y_pred))
    deviance_per_obs[y_true == 0] = 2 * y_pred[y_true == 0]  # 真实值为0时的修正
    
    # 3. 加权计算
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        weighted_sum = tf.reduce_sum(tf.multiply(deviance_per_obs, sample_weight)).numpy()
        weighted_avg = weighted_sum / sample_weight.sum()
        multiply = (deviance_per_obs * sample_weight) / sample_weight.sum()
    else:
        weighted_sum = None
        weighted_avg = None
        multiply = None
    
    return {
        'mse': mse,
        'deviance_per_obs': deviance_per_obs,
        'weighted_sum': weighted_sum,
        'weighted_avg': weighted_avg,
        'multiply': multiply
    }

def plot_boxplot(data, group_name, save=False):
    """绘制模型性能箱线图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.boxplot(data, labels=model_labels, patch_artist=True)
    
    # 设置箱体颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 图表样式
    ax.set_title(f'模型性能对比 ({group_name})', fontsize=20)
    ax.set_ylabel('性能指标（加权泊松偏差）', fontsize=15)
    ax.set_xlabel('模型', fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=14)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fitted_by_age(data_dict, group_name, save=False):
    """绘制按年龄分组的拟合值对比图"""
    new_train = data_dict['new_train']
    # 按年龄聚合
    y_true = new_train.groupby("age")['ny'].mean()
    fitted_data = {
        'GLM': new_train.groupby('age')['poisson_fitted'].mean(),
        'XGBoost': new_train.groupby('age')['xgboost_fitted'].mean(),
        'Wide&Deep': new_train.groupby('age')['wd_fitted'].mean(),
        'NN-Reg': new_train.groupby('age')['net_fitted'].mean(),
        'ResNet': new_train.groupby('age')['res_fitted'].mean(),
        'Transformer': new_train.groupby('age')['transformer_fitted'].mean(),
        'EBM': new_train.groupby('age')['ebm_fitted'].mean()
    }
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(y_true.index, y_true, label='真实值', color='black', alpha=0.8, s=50)
    for model, color in model_colors.items():
        ax.plot(fitted_data[model].index, fitted_data[model], label=model, 
                linestyle='-', color=color, linewidth=2)
    
    # 样式
    ax.set_title(f'按年龄分组的拟合值对比 ({group_name})', fontsize=20)
    ax.set_xlabel('年龄', fontsize=17)
    ax.set_ylabel(f'{group_name} 转移强度', fontsize=17)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_fitted.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predicted_by_age(data_dict, group_name, save=False):
    """绘制按年龄分组的预测值对比图"""
    new_test = data_dict['new_test']
    # 按年龄聚合
    y_true = new_test.groupby("age")['ny'].mean()
    pred_data = {
        'GLM': new_test.groupby('age')['poisson_predictions'].mean(),
        'XGBoost': new_test.groupby('age')['xgboost_predictions'].mean(),
        'Wide&Deep': new_test.groupby('age')['wd_predictions'].mean(),
        'NN-Reg': new_test.groupby('age')['net_predictions'].mean(),
        'ResNet': new_test.groupby('age')['res_predictions'].mean(),  # 修正原代码的变量错误
        'Transformer': new_test.groupby('age')['transformer_predictions'].mean(),
        'EBM': new_test.groupby('age')['ebm_predictions'].mean()
    }
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(y_true.index, y_true, label='真实值', color='black', alpha=0.8, s=50)
    for model, color in model_colors.items():
        ax.plot(pred_data[model].index, pred_data[model], label=model, 
                linestyle='-', color=color, linewidth=2)
    
    # 样式
    ax.set_title(f'按年龄分组的预测值对比 ({group_name})', fontsize=20)
    ax.set_xlabel('年龄', fontsize=17)
    ax.set_ylabel(f'{group_name} 转移强度', fontsize=17)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_by_age(data_dict, group_name, save=False):
    """绘制按年龄分组的残差图"""
    new_test = data_dict['new_test']
    # 按年龄聚合
    y_true = new_test.groupby("age")['ny'].mean()
    pred_data = {
        'GLM': new_test.groupby('age')['poisson_predictions'].mean(),
        'XGBoost': new_test.groupby('age')['xgboost_predictions'].mean(),
        'Wide&Deep': new_test.groupby('age')['wd_predictions'].mean(),
        'NN-Reg': new_test.groupby('age')['net_predictions'].mean(),
        'ResNet': new_test.groupby('age')['res_predictions'].mean(),
        'Transformer': new_test.groupby('age')['transformer_predictions'].mean(),
        'EBM': new_test.groupby('age')['ebm_predictions'].mean()
    }
    
    # 计算残差
    residuals = {}
    for model in model_colors.keys():
        residuals[model] = y_true - pred_data[model]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = y_true.index
    for model, color in model_colors.items():
        ax.scatter(x_axis, residuals[model], label=model, marker='o', color=color, alpha=0.5, s=50)
    
    # 参考线
    ax.axhline(y=0, linestyle='--', color='black', alpha=0.8, linewidth=2)
    
    # 样式
    ax.set_title(f'残差图 ({group_name})', fontsize=20)
    ax.set_xlabel('年龄', fontsize=17)
    ax.set_ylabel(f'{group_name} 残差值', fontsize=17)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_prediction_res.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predicted_by_year(data_dict, group_name, save=False):
    """绘制按年份分组的预测值对比图"""
    new_df = data_dict['new_df']
    # 按年份聚合
    y_true = new_df.groupby("t")['ny'].mean()
    pred_data = {
        'GLM': new_df.groupby('t')['poisson_fitted_predictions'].mean(),
        'XGBoost': new_df.groupby('t')['xgboost_fitted_predictions'].mean(),
        'Wide&Deep': new_df.groupby('t')['wd_fitted_predictions'].mean(),
        'NN-Reg': new_df.groupby('t')['net_fitted_predictions'].mean(),
        'ResNet': new_df.groupby('t')['res_fitted_predictions'].mean(),
        'Transformer': new_df.groupby('t')['transformer_fitted_predictions'].mean(),
        'EBM': new_df.groupby('t')['ebm_fitted_predictions'].mean()
    }
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(y_true.index, y_true, label='真实值', color='black', alpha=0.8, s=50)
    for model, color in model_colors.items():
        ax.plot(pred_data[model].index, pred_data[model], label=model, 
                linestyle='-', color=color, linewidth=2)
    
    # 样式
    ax.set_title(f'按年份分组的预测值对比 ({group_name})', fontsize=20)
    ax.set_xlabel('年份（1998-2018）', fontsize=17)
    ax.set_ylabel(f'{group_name} 转移强度', fontsize=17)
    ax.set_xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_prediction_fitted.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_by_year(data_dict, group_name, save=False):
    """绘制按年份分组的残差图"""
    new_df = data_dict['new_df']
    # 按年份聚合
    y_true = new_df.groupby("t")['ny'].mean()
    pred_data = {
        'GLM': new_df.groupby('t')['poisson_fitted_predictions'].mean(),
        'XGBoost': new_df.groupby('t')['xgboost_fitted_predictions'].mean(),
        'Wide&Deep': new_df.groupby('t')['wd_fitted_predictions'].mean(),
        'NN-Reg': new_df.groupby('t')['net_fitted_predictions'].mean(),
        'ResNet': new_df.groupby('t')['res_fitted_predictions'].mean(),
        'Transformer': new_df.groupby('t')['transformer_fitted_predictions'].mean(),
        'EBM': new_df.groupby('t')['ebm_fitted_predictions'].mean()
    }
    
    # 计算残差
    residuals = {}
    for model in model_colors.keys():
        residuals[model] = y_true - pred_data[model]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = y_true.index
    for model, color in model_colors.items():
        ax.scatter(x_axis, residuals[model], label=model, marker='o', color=color, alpha=0.5, s=50)
    
    # 参考线
    ax.axhline(y=0, linestyle='--', color='black', alpha=0.8, linewidth=2)
    
    # 样式
    ax.set_title(f'残差图 ({group_name})', fontsize=20)
    ax.set_xlabel('年份（1998-2018）', fontsize=17)
    ax.set_ylabel(f'{group_name} 残差值', fontsize=17)
    ax.set_xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{SAVE_PATH}\\{group_name}_prediction_fitted_res.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 主流程执行 =====================
# 存储所有分组的指标结果
all_metrics = {}
all_weighted_avg = {k: [] for k in groups.keys()}
all_multiply_data = {k: {} for k in groups.keys()}

# 1. 遍历所有分组，加载数据并计算指标
for group_key, group_info in groups.items():
    print(f'===== 处理分组 {group_key} ({group_info["name"]}) =====')
    
    # 加载数据
    data_dict = load_group_data(group_key)
    if not data_dict:
        continue
    
    # 提取关键数据
    test = data_dict['test']
    new_test = data_dict['new_test']
    sample_weight = test[group_info['weight_col']]
    y_test = test['ny']
    
    # 模型列表（对应列名）
    model_cols = {
        'GLM': 'poisson_predictions',
        'NN-Reg': 'net_predictions',
        'ResNet': 'res_predictions',
        'Transformer': 'transformer_predictions',
        'Wide&Deep': 'wd_predictions',
        'XGBoost': 'xgboost_predictions',
        'EBM': 'ebm_predictions'
    }
    
    # 计算每个模型的指标
    group_metrics = {}
    multiply_data = {}
    weighted_avg_list = []
    
    for model_name, col_name in model_cols.items():
        y_pred = new_test[col_name]
        metrics = calculate_metrics(y_test, y_pred, sample_weight)
        
        group_metrics[model_name] = metrics
        multiply_data[model_name] = metrics['multiply']
        weighted_avg_list.append(metrics['weighted_avg'])
        
        print(f'{model_name} MSE: {metrics["mse"]:.6f}, 加权泊松偏差: {metrics["weighted_avg"]:.6f}')
    
    # 存储结果
    all_metrics[group_key] = group_metrics
    all_weighted_avg[group_key] = weighted_avg_list
    all_multiply_data[group_key] = pd.DataFrame(multiply_data)
    
    # 2. 生成可视化图表（保存到指定路径）
    plot_boxplot(all_multiply_data[group_key], group_info['name'], save=True)
    plot_fitted_by_age(data_dict, group_info['name'], save=True)
    plot_predicted_by_age(data_dict, group_info['name'], save=True)
    plot_residual_by_age(data_dict, group_info['name'], save=True)
    plot_predicted_by_year(data_dict, group_info['name'], save=True)
    plot_residual_by_year(data_dict, group_info['name'], save=True)

# 3. 绘制所有分组的泊松偏差对比折线图
plt.figure(figsize=(15, 8))
model_names = model_labels
for group_key, group_info in groups.items():
    plt.plot(model_names, all_weighted_avg[group_key], marker='o', label=group_info['name'], linewidth=2)

# 添加数值标签
for i, model in enumerate(model_names):
    for group_key, group_info in groups.items():
        val = all_weighted_avg[group_key][i]
        plt.text(i, val, f'({val:.4f})', ha='center', va='bottom', fontsize=12)

# 样式设置
plt.ylim(0.005, 0.14)
plt.xlabel('模型', fontsize=23)
plt.ylabel('泊松偏差', fontsize=23)
plt.legend(fontsize=17)
plt.grid(True, alpha=0.5)
plt.tight_layout()

# 保存到论文文件夹
plt.savefig(f'{PAPER_SAVE_PATH}\\Comparison of Poisson Deviance values across models.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 4. 整理并输出所有分组的泊松偏差结果
result_df = pd.DataFrame({
    groups[k]['name']: all_weighted_avg[k] for k in groups.keys()
}, index=model_names)
print('\n===== 所有分组的加权泊松偏差汇总 =====')
print(result_df)