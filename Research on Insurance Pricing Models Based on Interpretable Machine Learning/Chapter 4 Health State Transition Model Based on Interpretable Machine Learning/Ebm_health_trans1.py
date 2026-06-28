# 统一导入依赖（精简冗余依赖）
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')  # 屏蔽pandas视图赋值警告

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.titlesize"] = 20  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 15  
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
set_visualize_provider(InlineProvider())  # 设置EBM可视化

# 配置常量（统一管理路径和参数）
BASE_PATH = r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本'
EBM_PARAMS = {
    'objective': 'poisson_deviance',
    'learning_rate': 0.08,
    'inner_bags': 25,
    'outer_bags': 10,
    'max_bins': 32
}
FEATURES = {
    'X': ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink'],
    'Y': 'ny',
    'num_features': ['time', 'age'],
    'cat_features': ['sex', 'residenc', 'marry', 'smoke', 'drink'],
    'train_time': [1, 3, 5.5, 8.5, 11.5],
    'test_time': [14.5, 18]
}
# 健康状态转移配置（统一管理标签和权重列）
TRANSITION_CONFIG = {
    '1_3': {'name': 'H->D', 'weight_col': 'sum_hyear'},
    '1_2': {'name': 'H->L', 'weight_col': 'sum_hyear'},
    '2_3': {'name': 'L->D', 'weight_col': 'sum_lyear'},
    '2_1': {'name': 'L->H', 'weight_col': 'sum_lyear'}
}

def create_preprocessor(num_features, cat_features):
    """创建特征预处理管道（避免重复定义）"""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    return preprocessor

def load_and_split_data(file_suffix, base_path, features):
    """加载数据并划分训练/测试集"""
    file_path = f'{base_path}\\分组{file_suffix}.csv'
    df = pd.read_csv(file_path)
    train = df.loc[df['time'].isin(features['train_time'])].copy()  # copy避免SettingWithCopyWarning
    test = df.loc[df['time'].isin(features['test_time'])].copy()
    return df, train, test

def train_ebm_model(train, test, features, weight_col, ebm_params):
    """训练EBM模型并返回预测结果"""
    # 提取特征和目标变量
    X_train = train[features['X']]
    y_train = train[features['Y']]
    w_train = train[weight_col]
    X_test = test[features['X']]
    y_test = test[features['Y']]
    w_test = test[weight_col]
    
    # 预处理
    preprocessor = create_preprocessor(features['num_features'], features['cat_features'])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 训练模型
    ebm = ExplainableBoostingRegressor(**ebm_params)
    ebm.fit(X_train_processed, y_train, sample_weight=w_train)
    
    # 预测
    train['fitted'] = ebm.predict(X_train_processed)
    test['predictions'] = ebm.predict(X_test_processed)
    
    return ebm, preprocessor, X_train_processed, X_test_processed, train, test, w_test

def plot_age_analysis(train, test, transition_name):
    """绘制按年龄分组的真实值vs预测值、残差图"""
    # 训练集拟合值对比
    y_true_train = train.groupby("age")[FEATURES['Y']].mean()
    y_fitted_train = train.groupby('age')['fitted'].mean()
    plt.plot(y_true_train.index, y_true_train, label='True Values', color='blue')
    plt.plot(y_fitted_train.index, y_fitted_train, label='Fitted Values', color='red')
    plt.title(f'Actual vs Fitted by Age({transition_name})')
    plt.xlabel('Age(65-105)')
    plt.ylabel(f'{transition_name} intensity')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 测试集预测值对比
    y_true_test = test.groupby("age")[FEATURES['Y']].mean()
    y_pred_test = test.groupby('age')['predictions'].mean()
    plt.scatter(y_true_test.index, y_true_test, label='True Values')
    plt.plot(y_pred_test.index, y_pred_test, label='Predicted Values', color='red')
    plt.title(f'Actual vs Predicted by Age({transition_name})')
    plt.xlabel('Age(65-105)')
    plt.ylabel(f'{transition_name} intensity')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 残差图
    residuals = y_true_test - y_pred_test
    plt.scatter(y_true_test.index, residuals, label='Residuals')
    plt.axhline(y=0, linestyle='--', label='y=0', color='red')
    plt.xlabel('Age(65-105)')
    plt.ylabel(f'{transition_name} Residuals')
    plt.title(f'Residual Plot({transition_name})')
    plt.grid(True)
    plt.show()

def plot_demographic_analysis(data, group_col, label_map, transition_name, x_col='age'):
    """绘制按人口特征（性别/居住地等）分组的预测值"""
    data[f'{group_col}_label'] = data[group_col].map(label_map)
    pred_by_group = data.groupby([x_col, f'{group_col}_label'])['predictions'].mean().reset_index()
    
    plt.figure()
    for label, color in zip(label_map.values(), ['blue', 'red']):
        subset = pred_by_group[pred_by_group[f'{group_col}_label'] == label]
        plt.plot(subset[x_col], subset['predictions'], color=color, linestyle='--', label=label)
    
    # 平均值
    mean_pred = data.groupby(x_col)['predictions'].mean()
    plt.plot(mean_pred.index, mean_pred, label='Mean', color='black')
    plt.title(f'{transition_name} by {group_col.replace("_", " ")}')
    plt.xlabel(x_col if x_col == 'age' else 'Year (1998-2018)')
    plt.ylabel(f'{transition_name} intensity')
    plt.legend()
    plt.grid(True)
    if x_col == 't':
        plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.show()

def plot_year_analysis(df, transition_name):
    """绘制按年份分组的真实值vs预测值、残差图"""
    y_true = df.groupby("t")[FEATURES['Y']].mean()
    y_pred = df.groupby('t')['predictions'].mean()
    
    # 真实值vs预测值
    plt.scatter(y_true.index, y_true, label='True Values')
    plt.plot(y_pred.index, y_pred, label='Predicted Values', color='red')
    plt.title(f'Actual vs Predicted by Year({transition_name})')
    plt.xlabel('Year(1998-2018)')
    plt.ylabel(f'{transition_name} intensity')
    plt.legend()
    plt.grid(True)
    plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.show()
    
    # 残差图
    residuals = y_true - y_pred
    plt.scatter(y_true.index, residuals, label='Residuals')
    plt.axhline(y=0, linestyle='--', label='y=0', color='red')
    plt.xlabel('Year(1998-2018)')
    plt.ylabel(f'{transition_name} Residuals')
    plt.title(f'Residual Plot({transition_name})')
    plt.grid(True)
    plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.show()

def calculate_metrics(y_test, y_pred, w_test):
    """计算MSE和加权泊松偏差（替换TensorFlow为numpy）"""
    # MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # 加权泊松偏差（鲁棒处理零值）
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)
    w_test = np.array(w_test)
    
    # 避免log(0)或负数
    y_pred = np.clip(y_pred, 1e-8, None)  # 预测值最小设为1e-8
    mask = y_true > 0
    deviance_per_obs = np.zeros_like(y_true)
    
    # 非零真实值的偏差计算
    deviance_per_obs[mask] = 2 * (y_true[mask] * np.log(y_true[mask] / y_pred[mask]) - (y_true[mask] - y_pred[mask]))
    # 零真实值的偏差计算
    deviance_per_obs[~mask] = 2 * y_pred[~mask]
    
    # numpy计算加权和（替代tf）
    weighted_sum = np.sum(deviance_per_obs * w_test)
    exposure_weighted_average = weighted_sum / w_test.sum()
    print(f"Exposure Weighted Poisson Deviance: {exposure_weighted_average:.6f}")
    return mse, exposure_weighted_average

def save_predictions(train, test, df, file_suffix, base_path):
    """保存拟合值和预测值到Excel"""
    pred_path = f'{base_path}\\predictions\\predictions_{file_suffix}.xlsx'
    fitted_path = f'{base_path}\\predictions\\fitted_{file_suffix}.xlsx'
    full_pred_path = f'{base_path}\\predictions\\fitted_predictions_{file_suffix}.xlsx'
    
    # 读取或创建空DataFrame
    for path in [pred_path, fitted_path, full_pred_path]:
        if not pd.io.common.file_exists(path):
            pd.DataFrame().to_excel(path, index=False)
    
    # 保存预测结果
    predictions_df = pd.read_excel(pred_path)
    predictions_df['ebm_predictions'] = test['predictions'].values
    predictions_df.to_excel(pred_path, index=False)
    
    fitted_df = pd.read_excel(fitted_path)
    fitted_df['ebm_fitted'] = train['fitted'].values
    fitted_df.to_excel(fitted_path, index=False)
    
    full_pred_df = pd.read_excel(full_pred_path)
    full_pred_df['ebm_fitted_predictions'] = df['predictions'].values
    full_pred_df.to_excel(full_pred_path, index=False)

def run_transition_analysis(file_suffix, transition_config, base_path, features, ebm_params):
    """执行单类健康状态转移的完整分析流程"""
    transition_name = transition_config['name']
    weight_col = transition_config['weight_col']
    
    # 1. 加载数据
    df, train, test = load_and_split_data(file_suffix, base_path, features)
    
    # 2. 训练模型
    ebm, preprocessor, X_train_proc, X_test_proc, train, test, w_test = train_ebm_model(
        train, test, features, weight_col, ebm_params
    )
    
    # 3. 合并预测结果到全量数据
    df['predictions'] = pd.concat([train['fitted'], test['predictions']], ignore_index=True)
    
    # 4. 可视化分析
    # 按年龄分析
    plot_age_analysis(train, test, transition_name)
    
    # 按人口特征（年龄维度）分析
    demo_configs = {
        'sex': {1: 'male', 2: 'female'},
        'residenc': {1: 'rural', 2: 'urban'},
        'marry': {1: 'with spouse', 2: 'without spouse'},
        'smoke': {1: 'smoking', 2: 'not smoking'},
        'drink': {1: 'drinking', 2: 'not drinking'}
    }
    for col, label_map in demo_configs.items():
        plot_demographic_analysis(test, col, label_map, transition_name, x_col='age')
    
    # 按年份分析
    plot_year_analysis(df, transition_name)
    
    # 按人口特征（年份维度）分析
    for col, label_map in demo_configs.items():
        plot_demographic_analysis(df, col, label_map, transition_name, x_col='t')
    
    # 5. 模型评估
    calculate_metrics(test[FEATURES['Y']], test['predictions'], w_test)
    
    # 6. 保存结果
    save_predictions(train, test, df, file_suffix, base_path)
    
    # 7. EBM可解释性可视化
    show(ebm.explain_global())
    show(ebm.explain_local(X_test_proc, test[FEATURES['Y']]))
    
    return ebm, df, train, test

# 执行所有健康状态转移分析
if __name__ == '__main__':
    for suffix, config in TRANSITION_CONFIG.items():
        print(f"\n========== 分析 {config['name']} 健康状态转移 ==========")
        run_transition_analysis(suffix, config, BASE_PATH, FEATURES, EBM_PARAMS)