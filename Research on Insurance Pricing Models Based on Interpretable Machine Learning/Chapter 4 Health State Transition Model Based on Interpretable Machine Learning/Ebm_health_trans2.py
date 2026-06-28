# --------------- 1. 环境配置与依赖导入 ---------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpret import set_visualize_provider, show
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei']        # 显示中文
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示
plt.rcParams["figure.figsize"] = (15, 8)           # 默认图大小
plt.rcParams["axes.titlesize"] = 27                # 标题字体
plt.rcParams['xtick.labelsize'] = 15               # x轴标签
plt.rcParams['ytick.labelsize'] = 15               # y轴标签

# 设置可视化提供者
set_visualize_provider(InlineProvider())

# --------------- 2. 常量定义（统一管理） ---------------
# 模型参数
EBM_PARAMS = {
    'objective': 'poisson_deviance',
    'learning_rate': 0.08,
    'inner_bags': 25,
    'outer_bags': 10
}

# 文件路径（使用正斜杠避免转义，或原始字符串）
BASE_PATH = r'D:/PG/Rpaper/基于可解释机器学习的中国老年人健康状态转移模型及应用/'
DATA_PATH = BASE_PATH + '数据 - 副本/'
FIG_PATH = BASE_PATH + '论文/'
EXCEL_PATH = BASE_PATH + '数据 - 副本/ebm_matrix/'

# 特征与目标列
FEATURES = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']
TARGET = 'ny'

# 类别特征映射字典
MAPPINGS = {
    'sex': {1: 'male', 2: 'female'},
    'residenc': {1: 'rural', 2: 'urban'},
    'marry': {1: 'with spouse', 2: 'without spouse'},
    'smoke': {1: 'smoking', 2: 'not smoking'},
    'drink': {1: 'drinking', 2: 'not drinking'}
}

# 健康转移类型映射（用于图表标签）
TRANSFER_LABELS = {
    '1_3': 'H->D',
    '1_2': 'H->L',
    '2_3': 'L->D',
    '2_1': 'L->H'
}

# --------------- 3. 工具函数封装（减少重复） ---------------
def load_data(file_name):
    """加载数据并处理类别特征"""
    try:
        df = pd.read_csv(DATA_PATH + file_name)
        # 统一处理类别特征
        for col in ['sex', 'residenc', 'marry', 'smoke', 'drink']:
            df[col] = df[col].astype('category')
        return df
    except FileNotFoundError:
        print(f"错误：文件 {file_name} 未找到！")
        return None
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        return None

def train_ebm(df, weight_col):
    """训练EBM模型"""
    X = df[FEATURES]
    y = df[TARGET]
    sample_weight = df[weight_col]
    
    ebm = ExplainableBoostingRegressor(**EBM_PARAMS)
    ebm.fit(X, y, sample_weight=sample_weight)
    
    # 预测并计算评估指标
    df['fitted'] = ebm.predict(X)
    mse = mean_squared_error(y, df['fitted'])
    mae = mean_absolute_error(y, df['fitted'])
    print(f"模型训练完成 | MSE: {mse:.6f} | MAE: {mae:.6f}")
    
    return ebm, df

def plot_fit_residual_by_age(df, transfer_type, save_name):
    """绘制按年龄分组的拟合值+残差图（双子图）"""
    # 按年龄聚合真实值和预测值
    y_true = df.groupby("age")[TARGET].mean()
    y_fitted = df.groupby('age')['fitted'].mean()
    residuals = y_true - y_fitted
    transfer_label = TRANSFER_LABELS[transfer_type]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.3)
    
    # 子图1：真实值vs预测值
    ax1.plot(y_true.index, y_true, label='True Values', color='#1F77B4', linewidth=2)
    ax1.plot(y_true.index, y_fitted, label='Fitted Values', color='orange', linewidth=2)
    ax1.set_title(f'Actual vs Fitted by Age ({transfer_label})', fontsize=20)
    ax1.set_xlabel('Age', fontsize=15)
    ax1.set_ylabel(f'{transfer_label} intensity', fontsize=15)
    ax1.legend()
    ax1.grid(True)
    
    # 子图2：残差图
    ax2.scatter(y_true.index, residuals, label='Residuals', alpha=0.7)
    ax2.axhline(y=0, linestyle='--', label='y=0', color='k', linewidth=1.5)
    ax2.set_xlabel('Age', fontsize=15)
    ax2.set_ylabel(f'{transfer_label} Residuals', fontsize=15)
    ax2.set_title(f'Residual Plot ({transfer_label})', fontsize=20)
    ax2.grid(True)
    ax2.legend()
    
    # 调整宽高比
    for ax in [ax1, ax2]:
        x_range = max(ax.get_xlim()) - min(ax.get_xlim())
        y_range = max(ax.get_ylim()) - min(ax.get_ylim())
        ax.set_aspect(x_range / y_range)
    
    # 保存图表（先保存再show，避免空白）
    plt.tight_layout()
    plt.savefig(FIG_PATH + save_name, dpi=100, bbox_inches='tight')
    # plt.show()  # 根据需要开启
    plt.close()  # 关闭画布释放内存

def plot_group_by_feature(df, transfer_type, feature, save_name):
    """绘制按指定特征（性别/居住地等）分组的预测值曲线"""
    transfer_label = TRANSFER_LABELS[transfer_type]
    feature_label = f'{feature}_label'
    
    # 添加特征标签列
    df[feature_label] = df[feature].map(MAPPINGS[feature])
    
    # 按年龄+特征聚合
    fitted_data = df.groupby(['age', feature_label])['fitted'].mean().reset_index()
    overall_mean = df.groupby('age')['fitted'].mean()
    
    # 获取特征的两个类别
    categories = list(MAPPINGS[feature].values())
    data1 = fitted_data[fitted_data[feature_label] == categories[0]]
    data2 = fitted_data[fitted_data[feature_label] == categories[1]]
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(data1['age'], data1['fitted'], color='royalblue', linestyle='--', 
             label=categories[0], linewidth=3.5)
    plt.plot(data2['age'], data2['fitted'], color='orangered', linestyle='--', 
             label=categories[1], linewidth=3.5)
    plt.plot(overall_mean.index, overall_mean, label='Mean', color='black', linewidth=3.5)
    
    # 图表样式
    plt.title(f'{transfer_label} by {feature}', fontsize=20)
    plt.xlabel('Age', fontsize=20)
    plt.ylabel(f'{transfer_label} intensity', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    
    # 保存
    plt.savefig(FIG_PATH + save_name, dpi=100, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_by_year(df, transfer_type):
    """绘制按年份分组的真实值vs预测值+残差图"""
    transfer_label = TRANSFER_LABELS[transfer_type]
    
    # 按年份聚合
    y_true = df.groupby("t")[TARGET].mean()
    y_fitted = df.groupby('t')['fitted'].mean()
    residuals = y_true - y_fitted
    
    # 真实值vs预测值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.index, y_true, label='True Values')
    plt.plot(y_fitted.index, y_fitted, label='Fitted Values', color='red')
    plt.title(f'Actual vs Fitted by Year ({transfer_label})', fontsize=18)
    plt.xlabel('Year (1998-2018)', fontsize=15)
    plt.ylabel(f'{transfer_label} intensity', fontsize=15)
    plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 残差图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.index, residuals, label='Residuals')
    plt.axhline(y=0, linestyle='--', label='y=0', color='red')
    plt.title(f'Residual Plot ({transfer_label})', fontsize=18)
    plt.xlabel('Year (1998-2018)', fontsize=15)
    plt.ylabel(f'{transfer_label} Residuals', fontsize=15)
    plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_year_by_feature(df, transfer_type, feature):
    """绘制按年份+特征分组的预测值曲线"""
    transfer_label = TRANSFER_LABELS[transfer_type]
    feature_label = f'{feature}_label'
    
    # 添加特征标签
    df[feature_label] = df[feature].map(MAPPINGS[feature])
    
    # 按年份+特征聚合
    fitted_data = df.groupby(['t', feature_label])['fitted'].mean().reset_index()
    overall_mean = df.groupby('t')['fitted'].mean()
    
    # 获取特征类别
    categories = list(MAPPINGS[feature].values())
    data1 = fitted_data[fitted_data[feature_label] == categories[0]]
    data2 = fitted_data[fitted_data[feature_label] == categories[1]]
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(data1['t'], data1['fitted'], color='blue', linestyle='--', label=categories[0])
    plt.plot(data2['t'], data2['fitted'], color='red', linestyle='--', label=categories[1])
    plt.plot(overall_mean.index, overall_mean, label='Mean', color='black')
    
    plt.title(f'{transfer_label} by {feature}', fontsize=18)
    plt.xlabel('Year (1998-2018)', fontsize=15)
    plt.ylabel(f'{transfer_label} intensity', fontsize=15)
    plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------- 4. 主流程执行 ---------------
def main():
    # 存储各分组的性别聚合数据（用于最后合并）
    gender_results = {
        'male': {},
        'female': {}
    }
    
    # ========== 分组1: 11_3.csv (H→D) ==========
    print("\n----- 处理分组11_3 (H→D) -----")
    df_1_3 = load_data('分组11_3.csv')
    if df_1_3 is not None:
        ebm_1_3, df_1_3 = train_ebm(df_1_3, 'sum_hyear')
        
        # 年龄拟合+残差图
        plot_fit_residual_by_age(
            df_1_3, 
            '1_3', 
            'H-D Actual and fitted values and their residuals grouped by age.png'
        )
        
        # 按特征分组绘图（性别/居住地/婚姻/吸烟/饮酒）
        features_to_plot = ['sex', 'residenc', 'marry', 'smoke', 'drink']
        for feat in features_to_plot:
            plot_group_by_feature(
                df_1_3, 
                '1_3', 
                feat, 
                f'H-D Comparison of transfer intensities for each health state grouped by {feat}.png'
            )
        
        # 按年份绘图
        plot_by_year(df_1_3, '1_3')
        for feat in features_to_plot:
            plot_year_by_feature(df_1_3, '1_3', feat)
        
        # 保存EBM解释结果
        show(ebm_1_3.explain_global())
        show(ebm_1_3.explain_local(df_1_3[FEATURES], df_1_3[TARGET]))
        
        # 提取性别聚合数据
        df_1_3['sex_label'] = df_1_3['sex'].map(MAPPINGS['sex'])
        gender_data = df_1_3.groupby(['age', 'sex_label'])['fitted'].mean().reset_index()
        gender_results['male']['1_3'] = gender_data[gender_data['sex_label']=='male'].rename(columns={'fitted': 'fitted_1_3_m'})
        gender_results['female']['1_3'] = gender_data[gender_data['sex_label']=='female'].rename(columns={'fitted': 'fitted_1_3_f'})
    
    # ========== 分组2: 11_2.csv (H→L) ==========
    print("\n----- 处理分组11_2 (H→L) -----")
    df_1_2 = load_data('分组11_2.csv')
    if df_1_2 is not None:
        ebm_1_2, df_1_2 = train_ebm(df_1_2, 'sum_hyear')
        
        plot_fit_residual_by_age(
            df_1_2, 
            '1_2', 
            'H-L Actual and fitted values and their residuals grouped by age.png'
        )
        
        for feat in features_to_plot:
            plot_group_by_feature(
                df_1_2, 
                '1_2', 
                feat, 
                f'H-L Comparison of transfer intensities for each health state grouped by {feat}.png'
            )
        
        plot_by_year(df_1_2, '1_2')
        for feat in features_to_plot:
            plot_year_by_feature(df_1_2, '1_2', feat)
        
        show(ebm_1_2.explain_global())
        show(ebm_1_2.explain_local(df_1_2[FEATURES], df_1_2[TARGET]))
        
        # 提取性别聚合数据
        df_1_2['sex_label'] = df_1_2['sex'].map(MAPPINGS['sex'])
        gender_data = df_1_2.groupby(['age', 'sex_label'])['fitted'].mean().reset_index()
        gender_results['male']['1_2'] = gender_data[gender_data['sex_label']=='male'].rename(columns={'fitted': 'fitted_1_2_m'})
        gender_results['female']['1_2'] = gender_data[gender_data['sex_label']=='female'].rename(columns={'fitted': 'fitted_1_2_f'})
    
    # ========== 分组3: 22_3.csv (L→D) ==========
    print("\n----- 处理分组22_3 (L→D) -----")
    df_2_3 = load_data('分组22_3.csv')
    if df_2_3 is not None:
        ebm_2_3, df_2_3 = train_ebm(df_2_3, 'sum_lyear')
        
        plot_fit_residual_by_age(
            df_2_3, 
            '2_3', 
            'L-D Actual and fitted values and their residuals grouped by age.png'
        )
        
        for feat in features_to_plot:
            plot_group_by_feature(
                df_2_3, 
                '2_3', 
                feat, 
                f'L-D Comparison of transfer intensities for each health state grouped by {feat}.png'
            )
        
        plot_by_year(df_2_3, '2_3')
        for feat in features_to_plot:
            plot_year_by_feature(df_2_3, '2_3', feat)
        
        show(ebm_2_3.explain_global())
        show(ebm_2_3.explain_local(df_2_3[FEATURES], df_2_3[TARGET]))
        
        # 提取性别聚合数据
        df_2_3['sex_label'] = df_2_3['sex'].map(MAPPINGS['sex'])
        gender_data = df_2_3.groupby(['age', 'sex_label'])['fitted'].mean().reset_index()
        gender_results['male']['2_3'] = gender_data[gender_data['sex_label']=='male'].rename(columns={'fitted': 'fitted_2_3_m'})
        gender_results['female']['2_3'] = gender_data[gender_data['sex_label']=='female'].rename(columns={'fitted': 'fitted_2_3_f'})
    
    # ========== 分组4: 22_1.csv (L→H) ==========
    print("\n----- 处理分组22_1 (L→H) -----")
    df_2_1 = load_data('分组22_1.csv')
    if df_2_1 is not None:
        ebm_2_1, df_2_1 = train_ebm(df_2_1, 'sum_lyear')
        
        plot_fit_residual_by_age(
            df_2_1, 
            '2_1', 
            'L-H Actual and fitted values and their residuals grouped by age.png'
        )
        
        for feat in features_to_plot:
            plot_group_by_feature(
                df_2_1, 
                '2_1', 
                feat, 
                f'L-H Comparison of transfer intensities for each health state grouped by {feat}.png'
            )
        
        plot_by_year(df_2_1, '2_1')
        for feat in features_to_plot:
            plot_year_by_feature(df_2_1, '2_1', feat)
        
        show(ebm_2_1.explain_global())
        show(ebm_2_1.explain_local(df_2_1[FEATURES], df_2_1[TARGET]))
        
        # 提取性别聚合数据
        df_2_1['sex_label'] = df_2_1['sex'].map(MAPPINGS['sex'])
        gender_data = df_2_1.groupby(['age', 'sex_label'])['fitted'].mean().reset_index()
        gender_results['male']['2_1'] = gender_data[gender_data['sex_label']=='male'].rename(columns={'fitted': 'fitted_2_1_m'})
        gender_results['female']['2_1'] = gender_data[gender_data['sex_label']=='female'].rename(columns={'fitted': 'fitted_2_1_f'})
    
    # ========== 合并并保存性别数据 ==========
    print("\n----- 合并并保存性别数据 -----")
    # 合并男性数据
    male_df = gender_results['male']['1_3']
    for key in ['1_2', '2_3', '2_1']:
        male_df = pd.merge(male_df, gender_results['male'][key], on=['age', 'sex_label'], how='outer')
    
    # 合并女性数据
    female_df = gender_results['female']['1_3']
    for key in ['1_2', '2_3', '2_1']:
        female_df = pd.merge(female_df, gender_results['female'][key], on=['age', 'sex_label'], how='outer')
    
    # 保存到Excel
    try:
        male_df.rename(columns={'age': '年龄'}).to_excel(EXCEL_PATH + '转移强度-M1111.xlsx', index=False)
        female_df.rename(columns={'age': '年龄'}).to_excel(EXCEL_PATH + '转移强度-F1111.xlsx', index=False)
        print("数据保存成功！")
    except Exception as e:
        print(f"数据保存失败：{str(e)}")

if __name__ == '__main__':
    main()