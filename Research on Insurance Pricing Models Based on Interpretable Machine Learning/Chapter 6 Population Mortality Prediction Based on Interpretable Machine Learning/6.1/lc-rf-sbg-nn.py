import torch
import shap
from alibi.explainers import ALE, plot_ale

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import optuna

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Multiply, Lambda, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# 加载数据
data = pd.read_csv("E:/xiazai/Human Mortality Database/Chinese/female.csv")

# 向上填充0值和inf值
data.iloc[:, 2:3] = data.iloc[:, 2:3].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data.iloc[:, 2:3] = data.iloc[:, 2:3].fillna(method='ffill')
# 将年份小于2015的数据筛选为训练集,# 将年份大于等于2015的数据筛选为测试集
train_data1 = data[(data['year'] >= 1994) & (data['year'] <= 2015)]
test_data1 = data[(data['year'] >= 2016) & (data['year'] <= 2021)]
# 将训练集数据化为矩阵形式
train_data1['mx_adj'] = np.log(train_data1['mx'])
dcast_data = train_data1.groupby(['age', 'year'])['mx_adj'].mean().reset_index()
rates_mat = dcast_data.pivot(index='age', columns='year', values='mx_adj').fillna(0).values
# 将测试集数据化为矩阵形式
test_data1['mx_adj'] = np.log(test_data1['mx'])
dcast_data1 = test_data1.groupby(['age', 'year'])['mx_adj'].mean().reset_index()
rates_mat1 = dcast_data1.pivot(index='age', columns='year', values='mx_adj').fillna(0).values

# 初始化参数
ax = np.zeros(rates_mat.shape[0])
bx = np.full(rates_mat.shape[0], 1 / (1 + 0.5))
kt = np.zeros(rates_mat.shape[1])
oldobj = 100000


# 定义损失函数
def loss(ax, bx, kt, weight_factor):
    mx = np.outer(bx, kt)
    predicted_mx = mx + ax[:, np.newaxis]
    weights = np.where(rates_mat > predicted_mx, weight_factor, 1 - weight_factor)
    weighted_mse = np.sum(weights * (rates_mat - predicted_mx) ** 2) / np.sum(weights)
    return weighted_mse


# 梯度下降迭代
epsilon = 0.000001
i = 0
weight_factor = 0.5
while True:
    mx = np.outer(bx, kt)
    predicted_mx = mx + ax[:, np.newaxis]

    gradient_kt = -2 * np.dot(bx, (rates_mat - predicted_mx)) / rates_mat.shape[0]
    kt = kt - 0.1 * gradient_kt

    mx = np.outer(bx, kt)
    predicted_mx = mx + ax[:, np.newaxis]

    gradient_bx = -2 * np.dot((rates_mat - predicted_mx), kt) / rates_mat.shape[1]
    bx = bx - 0.1 * gradient_bx

    mx = np.outer(bx, kt)
    predicted_mx = mx + ax[:, np.newaxis]

    gradient_ax = -2 * np.mean(rates_mat - predicted_mx, axis=1) / rates_mat.shape[1]
    ax = ax - 0.1 * gradient_ax

    # 计算新的损失函数值
    newobj = loss(ax, bx, kt, weight_factor)
    i += 1

    if oldobj - newobj <= epsilon:
        break

    oldobj = newobj

c1 = np.mean(kt)
c2 = np.sum(bx)
ax = ax + c1 * bx
bx = bx / c2
kt = (kt - c1) * c2

mx = np.outer(bx, kt)
fitting_mx = mx + ax[:, np.newaxis]
fittingmse = np.mean((rates_mat - fitting_mx) ** 2)
ax_data = pd.DataFrame({'age': range(91), 'ax': ax})
plt.figure(figsize=(8, 6))
plt.scatter(ax_data['age'], ax_data['ax'], label='Data Points')
plt.plot(ax_data['age'], ax_data['ax'], label='Line')
plt.xlabel('Age')
plt.ylabel('αx')
plt.legend()
plt.savefig('ax_plot.png') 

bx_data = pd.DataFrame({'age': range(91), 'bx': bx})
plt.figure(figsize=(8, 6))
plt.scatter(bx_data['age'], bx_data['bx'], label='Data Points')
plt.plot(bx_data['age'], bx_data['bx'], label='Line')
plt.xlabel('Age')
plt.ylabel('βx')
plt.legend()
plt.savefig('bx_plot.png') 
plt.show()

kt_data = pd.DataFrame({'year': range(1994, 2016), 'kt': kt})
plt.figure(figsize=(10, 6))
plt.scatter(kt_data['year'], kt_data['kt'], label='Data Points')
plt.plot(kt_data['year'], kt_data['kt'], label='Line')
plt.xlabel('Year')
plt.ylabel('kt')
plt.legend()
plt.savefig('kt_plot.png') 
plt.show()

kt_data['kt_diff'] = kt_data['kt'].diff().dropna()
# 移除包含无穷大或NaN的行
kt_data = kt_data.replace([np.inf, -np.inf], np.nan).dropna()
# 进行迪基-富勒检验
result_diff = adfuller(kt_data['kt_diff'])

# 检验差分后的序列的平稳性
result_diff = adfuller(kt_data['kt_diff'])
print("ADF Statistic (after differencing):", result_diff[0])
print("p-value (after differencing):", result_diff[1])

# 根据 ACF 和 PACF 绘制图来选择 ARIMA(p, d, q) 的参数
plot_acf(kt_data['kt_diff'], lags=20)
plot_pacf(kt_data['kt_diff'], lags=10)
plt.show()

# 转换时间索引
kt_data['year'] = pd.to_datetime(kt_data['year'], format='%Y')
kt_data.set_index('year', inplace=True)
# 用ARIMA模型拟合时间序列
# p是自回归阶数，d是差分阶数，q是移动平均阶数

model = ARIMA(kt_data['kt'], order=(5,1,2))
result = model.fit()

#ARIMA参数
#male：（3,1,1）
#female（5,1,2）

# 使用模型进行未来10年的预测
forecast = result.get_forecast(steps=6)
# 将预测结果转换为数据框
kt_forecast = forecast.predicted_mean.values
forecast_df = pd.DataFrame({'year': range(2016, 2022), 'kt_forecast': kt_forecast})
# 计算预测的死亡率
fmx = np.outer(bx, kt_forecast)
forecast_mx = fmx + ax[:, np.newaxis]
# 2012
year_2016 = forecast_mx[:, 0]
year_2016_true = rates_mat1[:, 0]
prediction_2016_mse = np.mean((year_2016 - year_2016_true) ** 2)
year_2017 = forecast_mx[:, 1]
year_2018 = forecast_mx[:, 2]
year_2019 = forecast_mx[:, 3]
year_2020 = forecast_mx[:, 4]

residual = rates_mat - fitting_mx

# 数据归一化
scaler = MinMaxScaler()
residual_scaled = scaler.fit_transform(residual.reshape(-1, 1)).reshape(91, 22)

ax_data = pd.DataFrame({'age': range(91), 'ax': ax})
plt.figure(figsize=(8, 6))
plt.scatter(ax_data['age'], ax_data['ax'], label='Data Points')
plt.plot(ax_data['age'], ax_data['ax'], label='Line')
plt.xlabel('Age')
plt.ylabel('αx')
plt.legend()
plt.savefig('ax_plot.png') 
plt.show()

bx_data = pd.DataFrame({'age': range(91), 'bx': bx})
plt.figure(figsize=(8, 6))
plt.scatter(bx_data['age'], bx_data['bx'], label='Data Points')
plt.plot(bx_data['age'], bx_data['bx'], label='Line')
plt.xlabel('Age')
plt.ylabel('βx')
plt.legend()
plt.savefig('bx_plot.png') 
plt.show()

# 首先绘制历史 kt 数据
kt_data = pd.DataFrame({'year': range(1994, 2016), 'kt': kt})
plt.figure(figsize=(10, 6))
plt.scatter(kt_data['year'], kt_data['kt'], label='Historical Data Points')
plt.plot(kt_data['year'], kt_data['kt'], label='Historical Line')

# 添加预测的 kt 数据
forecast_years = range(2016, 2022)  # 2017-2021
plt.scatter(forecast_years, kt_forecast, color='red', label='Forecast Data Points')
plt.plot(forecast_years, kt_forecast, color='red', linestyle='--', label='Forecast Line')

plt.xlabel('Year')
plt.ylabel('kt')
plt.legend()
plt.savefig('kt_plot_with_forecast.png') 
plt.show()

import pandas as pd
import numpy as np

# 残差比值：ψ = m_true / m_LC
m_true = np.exp(rates_mat)            # shape (91, 22)
m_lc = np.exp(fitting_mx)             # shape (91, 22)
psi = m_true / (m_lc + 1e-10)

# 构造训练集 (age, year, cohort, ψ)
ages = np.arange(91)
years = np.arange(1994, 2016)

records = []
for i, age in enumerate(ages):
    for j, year in enumerate(years):
        cohort = year - age
        psi_val = psi[i, j]
        records.append([age, year, cohort, psi_val])

df_train = pd.DataFrame(records, columns=["age", "year", "cohort", "psi"])


# LC 预测：forecast_mx → exp
m_lc_test = np.exp(forecast_mx[:, :6])    # shape (91, 5)
m_true_test = np.exp(rates_mat1[:, :6])   # shape (91, 5)
years_test = np.arange(2016, 2022)

# 构造测试集
records_test = []
for i, age in enumerate(ages):
    for j, year in enumerate(years_test):
        cohort = year - age
        m_lc_val = m_lc_test[i, j]
        m_true_val = m_true_test[i, j]
        psi_val = m_true_val / (m_lc_val + 1e-10)
        records_test.append([age, year, cohort, m_lc_val, psi_val])

df_test = pd.DataFrame(records_test, columns=["age", "year", "cohort", "m_lc", "psi_true"])


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(df_train[["age", "year", "cohort"]], df_train["psi"])

df_test["psi_pred_rf"] = rf.predict(df_test[["age", "year", "cohort"]])


from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(df_train[["age", "year", "cohort"]], df_train["psi"])

df_test["psi_pred_xgb"] = xgb.predict(df_test[["age", "year", "cohort"]])


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_nn = scaler_x.fit_transform(df_train[["age", "year", "cohort"]])
y_train_nn = scaler_y.fit_transform(df_train[["psi"]])

mlp = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(64, activation='relu'),
    Dense(1)
])
mlp.compile(optimizer='adam', loss='mse')
mlp.fit(X_train_nn, y_train_nn, epochs=100, batch_size=64, verbose=0)

X_test_nn = scaler_x.transform(df_test[["age", "year", "cohort"]])
psi_pred_nn = mlp.predict(X_test_nn)
df_test["psi_pred_nn"] = scaler_y.inverse_transform(psi_pred_nn)


# 真实 log(m)
df_test["log_m_true"] = np.log(df_test["m_lc"] * df_test["psi_true"] + 1e-10)

# Lee-Carter 原始预测
df_test["log_m_lc"] = np.log(df_test["m_lc"] + 1e-10)

# 各模型预测
df_test["log_m_rf"]  = np.log(df_test["m_lc"] * df_test["psi_pred_rf"] + 1e-10)
df_test["log_m_xgb"] = np.log(df_test["m_lc"] * df_test["psi_pred_xgb"] + 1e-10)
df_test["log_m_nn"]  = np.log(df_test["m_lc"] * df_test["psi_pred_nn"] + 1e-10)
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def eval_log_errors(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n📊 {model_name} - log(mortality) 评估指标：")
    print(f"MAE :  {mae:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAPE:  {mape:.4%}")

# 调用
eval_log_errors(df_test["log_m_true"], df_test["log_m_lc"], "Lee-Carter")
eval_log_errors(df_test["log_m_true"], df_test["log_m_rf"], "Random Forest")
eval_log_errors(df_test["log_m_true"], df_test["log_m_xgb"], "XGBoost")
eval_log_errors(df_test["log_m_true"], df_test["log_m_nn"],  "Neural Network (MLP)")

import matplotlib.pyplot as plt
import numpy as np

# 筛选 2020 年数据
df_2021 = df_test[df_test['year'] == 2021].copy()

# 计算预测死亡率
df_2021['m_true'] = df_2021['m_lc'] * df_2021['psi_true']
df_2021['m_rf'] = df_2021['m_lc'] * df_2021['psi_pred_rf']
df_2021['m_xgb'] = df_2021['m_lc'] * df_2021['psi_pred_xgb']
df_2021['m_nn'] = df_2021['m_lc'] * df_2021['psi_pred_nn']

# 取 log(死亡率)
df_2021['log_m_true'] = np.log(df_2021['m_true'] + 1e-10)
df_2021['log_m_lc'] = np.log(df_2021['m_lc'] + 1e-10)
df_2021['log_m_rf'] = np.log(df_2021['m_rf'] + 1e-10)
df_2021['log_m_xgb'] = np.log(df_2021['m_xgb'] + 1e-10)
df_2021['log_m_nn'] = np.log(df_2021['m_nn'] + 1e-10)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(df_2021['age'], df_2021['log_m_true'], label='True 2021', linestyle='--', color='black')
plt.plot(df_2021['age'], df_2021['log_m_lc'], label='LC 2021', linestyle=':', color='blue')
plt.plot(df_2021['age'], df_2021['log_m_rf'], label='LC + RF', linestyle='-', color='green')
plt.plot(df_2021['age'], df_2021['log_m_xgb'], label='LC + XGBoost', linestyle='-', color='orange')
plt.plot(df_2021['age'], df_2021['log_m_nn'], label='LC + MLP', linestyle='-', color='red')

plt.xlabel('Age')
plt.ylabel('log(Mortality Rate)')
plt.title('Log Mortality Rate by Age in 2021 - Model Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


male:(3,1,1)
female:(5,1,2)

# 打印变量重要性
# 随机森林模型的特征重要性
rf_importances = rf.feature_importances_

# XGBoost模型的特征重要性
xgb_importances = xgb.feature_importances_

print(f"随机森林特征重要性: {rf_importances}")
print(f"XGBoost特征重要性: {xgb_importances}")

# 特征名称
features = df_test[["age", "year", "cohort"]].columns

# 绘制水平柱状图
plt.figure(figsize=(8, 3))
plt.barh(features, rf_importances,height=0.45,  label='Random Forest')
plt.xlabel('Feature Importance')
plt.title('Feature Importance - Random Forest')
plt.legend()
plt.gca().invert_yaxis()  # 反转y轴，使得重要性最高的特征在顶部
plt.tight_layout()
plt.savefig('rf_importance.png')  # 保存图像
plt.show()

plt.figure(figsize=(8, 3))
plt.barh(features, xgb_importances,height=0.45,  label='XGBoost')
plt.xlabel('Feature Importance')
plt.title('Feature Importance - XGBoost')
plt.legend()
plt.gca().invert_yaxis()  # 反转y轴
plt.tight_layout()
plt.savefig('xgb_importance.png')  # 保存图像
plt.show()

X_train=df_train[["age", "year", "cohort"]]

X_test=df_test[["age", "year", "cohort"]]

# 使用SHAP分析随机森林模型
explainer_rf = shap.Explainer(rf, X_train)
shap_values_rf = explainer_rf(X_test)

# 使用SHAP分析XGBoost模型
explainer_xgb = shap.Explainer(xgb, X_train)
shap_values_xgb = explainer_xgb(X_test)

#部分依赖图
shap.plots.partial_dependence( 
    "age", rf.predict, X_test, ice=False, 
    model_expected_value=True, feature_expected_value=True 
)

shap.plots.partial_dependence( 
    "year", rf.predict, X_test, ice=False, 
    model_expected_value=True, feature_expected_value=True 
)

shap.plots.partial_dependence( 
    "cohort", rf.predict, X_test, ice=False, 
    model_expected_value=True, feature_expected_value=True 
)

shap.plots.partial_dependence( 
    "age", rf.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)
shap.plots.partial_dependence( 
    "year", rf.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)
shap.plots.partial_dependence( 
    "cohort", rf.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)

shap.plots.partial_dependence( 
    "age", xgb.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)

shap.plots.partial_dependence( 
    "year", xgb.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)


shap.plots.partial_dependence( 
    "cohort", xgb.predict, X_test, ice=True, 
    model_expected_value=True, feature_expected_value=True 
)

# 随机森林模型的部分依赖曲线
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    rf, 
    X_test, 
    features=['age', 'year', 'cohort'], 
    feature_names=['age', 'year', 'cohort'],  # 确保这里的名称与数据中的列名一致
    grid_resolution=20,
    ax=ax
)
ax.set_title('Random Forest Partial Dependence')
plt.show()

# XGBoost模型的部分依赖曲线
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    xgb, 
    X_test, 
    features=['age', 'year', 'cohort'], 
    feature_names=['age', 'year', 'cohort'],  # 确保这里的名称与数据中的列名一致
    grid_resolution=20,
    ax=ax
)
ax.set_title('XGBoost Partial Dependence')
plt.show()

from alepython import ale_plot
#随机森林ALE图
ale_plot(rf, X_test, 'age', monte_carlo=True)
ale_plot(rf, X_test, 'year', monte_carlo=True)
ale_plot(rf, X_test, 'cohort', monte_carlo=True)

#xgboostALE图
ale_plot(xgb, X_test, 'age', monte_carlo=True)
ale_plot(xgb, X_test, 'year', monte_carlo=True)
ale_plot(xgb, X_test, 'cohort', monte_carlo=True)

shap.summary_plot(shap_values_rf, X_test, feature_names=X_test.columns)

shap.summary_plot(shap_values_xgb, X_test, feature_names=X_test.columns)

shap.summary_plot(shap_values_rf, X_test, feature_names=X_test.columns,plot_type="bar")

shap.summary_plot(shap_values_xgb, X_test, feature_names=X_test.columns,plot_type="bar")

sample_idx = 100  # 或你想查看的其他索引

import shap

# 初始化 JS 可视化环境（仅在 notebook 中生效）
shap.initjs()

# 使用 Random Forest 模型的 force plot
shap.force_plot(
    base_value=explainer_rf.expected_value,
    shap_values=shap_values_rf[sample_idx].values,
    features=X_test.iloc[sample_idx],
    feature_names=X_test.columns
)


shap.force_plot(
    base_value=explainer_xgb.expected_value,
    shap_values=shap_values_xgb[sample_idx].values,
    features=X_test.iloc[sample_idx],
    feature_names=X_test.columns
)


shap.dependence_plot(
    ind="age",                        # 要分析的特征
    shap_values=shap_values_rf.values,  # SHAP 值（二维数组）
    features=X_test,                 # 输入特征（DataFrame）
    feature_names=X_test.columns
)

shap.dependence_plot(
    ind="age",                        # 要分析的特征
    shap_values=shap_values_rf.values,  # SHAP 值（二维数组）
    features=X_test,                 # 输入特征（DataFrame）
    feature_names=X_test.columns,
    interaction_index='year'
)

shap.dependence_plot(
    ind="cohort",                        # 要分析的特征
    shap_values=shap_values_rf.values,  # SHAP 值（二维数组）
    features=X_test,                 # 输入特征（DataFrame）
    feature_names=X_test.columns,
    interaction_index='year'
)

shap.dependence_plot(
    ind="age",
    shap_values=shap_values_xgb.values,
    features=X_test,
    feature_names=X_test.columns
)


shap.dependence_plot(
    ind="age",                        # 要分析的特征
    shap_values=shap_values_xgb.values,  # SHAP 值（二维数组）
    features=X_test,                 # 输入特征（DataFrame）
    feature_names=X_test.columns,
    interaction_index='year'
)

shap.dependence_plot(
    ind="cohort",                        # 要分析的特征
    shap_values=shap_values_xgb.values,  # SHAP 值（二维数组）
    features=X_test,                 # 输入特征（DataFrame）
    feature_names=X_test.columns,
    interaction_index='year'
)

