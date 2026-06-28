import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time

df_freq_prep_xgb = df_freq.copy()

# 将 VehGas 转换为二进制
df_freq_prep_xgb["VehGas"] = df_freq_prep_xgb["VehGas"].map({"Diesel": 1, "Regular": 0}).astype(int)

# 定义数值特征和分类特征
nr_col = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehGas", "Density"]
cat_col = ["Area", "VehBrand", "Region"]

# 标准化数值特征
prep_standardscaler = StandardScaler()
prep_standardscaler.fit(df_freq_prep_xgb[bool_in_learn][nr_col])  # 只用训练数据拟合标准化器
df_freq_prep_xgb[nr_col] = prep_standardscaler.transform(df_freq_prep_xgb[nr_col])

# 将分类特征转换为整数编码
for col in cat_col:
    df_freq_prep_xgb[col] = df_freq_prep_xgb[col].astype('category').cat.codes

# 准备数据
X_xgb = df_freq_prep_xgb.drop(columns=['Exposure', 'ClaimNb', 'IDpol'])
X_xgb_learn = X_xgb[bool_in_learn]
X_xgb_test = X_xgb[bool_in_test]

# 转换为 XGBoost 的 DMatrix 格式
dtrain = xgb.DMatrix(X_xgb_learn, label=y_true["train"] / exposure["train"])
dtest = xgb.DMatrix(X_xgb_test, label=y_true["test"] / exposure["test"])

# 主循环
for run_index in range(15):
    print(f"Model: {run_index}")

    # 设置随机种子
    np.random.seed(random_seeds[run_index])
    
    # 创建XGBoost模型
    model = xgb.XGBRegressor(
        objective='count:poisson',  # 泊松回归目标函数
        max_depth=4,                # 减小树的深度
        learning_rate=0.05,         # 降低学习率
        n_estimators=200,           # 增加迭代次数
        subsample=0.8,              # 减少样本采样比例
        colsample_bytree=0.8,       # 减少特征采样比例
        gamma=0.1,                  # 增加 gamma
        random_state=random_seeds[run_index]  # 设置随机种子
    )
    
    # 训练模型
    start_time = time.time()
    model.fit(
        X_xgb_learn, 
        y_true["train"] / exposure["train"], 
        sample_weight=exposure["train"]
    )
    end_time = time.time()
    execution_time_xgb = end_time - start_time
    
    # 预测
    y_pred_train = model.predict(X_xgb_learn) * exposure["train"]
    y_pred_test = model.predict(X_xgb_test) * exposure["test"]
    
    # 计算泊松偏差损失
    poisson_deviance_loss_train = poisson_deviance_loss(y_true["train"], y_pred_train)
    poisson_deviance_loss_test = poisson_deviance_loss(y_true["test"], y_pred_test)
    
    # 计算平均频率
    pred_avg_freq_train = y_pred_train.sum() / exposure["train"].sum()
    pred_avg_freq_test = y_pred_test.sum() / exposure["test"].sum()
    
    # 评估模型
    xgb_results = Results(
        model=f"XGB (run: {run_index})",
        epochs=0,  # 使用固定的迭代次数
        run_time=execution_time_xgb,
        poisson_deviance_loss_train=poisson_deviance_loss_train,
        poisson_deviance_loss_test=poisson_deviance_loss_test,
        pred_avg_freq_train=pred_avg_freq_train,
        pred_avg_freq_test=pred_avg_freq_test
    )
    
    # 将结果存入 DataFrame
    store_results_in_df(xgb_results)

# 显示结果
display(df_results)

import shap
import matplotlib.pyplot as plt

# 训练 SHAP 解释器
explainer = shap.TreeExplainer(model, X_xgb_learn)
shap_values = explainer.shap_values(X_xgb_test)

# 准备绘图数据
# 将独热编码特征的 SHAP 值合并
shap.summary_plot_data = shap.summary_plot(shap_values, X_xgb_test)

# 绘制 SHAP 值总结图（所有特征）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_xgb_test, feature_names=X_xgb_test.columns.tolist(), plot_type="bar", max_display=20)
plt.show()