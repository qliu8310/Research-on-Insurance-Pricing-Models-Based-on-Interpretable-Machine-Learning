import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("论文数据.csv").drop(columns='id')

# 定义原始变量名和新变量名的映射关系
rename_columns = {
    "性别": "Gender",
    "最高学历": "Highest Education Level",
    "婚姻状态": "Marital Status",
    "残疾个数": "Number of Disabilities",
    "事故种类": "Type of Accident",
    "过去两年有没有摔倒": "Falls in the Past Two Years",
    "疼痛部位个数": "Number of Pain Locations",
    "平均每晚睡眠时间": "Average Hours of Sleep per Night",
    "平均午睡时间": "Average Nap Duration",
    "每周激烈活动天数": "Days of Intense Activity per Week",
    "每周中等强度活动天数": "Days of Moderate Activity per Week",
    "每周轻微活动天数": "Days of Light Activity per Week",
    "吸烟状态": "Smoking Status",
    "平均一天抽多少支烟": "Average Number of Cigarettes Smoked per Day",
    "过去一年喝酒频率": "Alcohol Consumption Frequency in the Past Year"
}

# 使用Pandas的rename方法来替换DataFrame中的列名
data.rename(columns=rename_columns, inplace=True)

# 设置随机种子以确保结果的可重复性
np.random.seed(123)

# 将分类变量转换为因子（在pandas中称为类别类型）
categorical_columns = ['Gender', 'Highest Education Level', 'Marital Status', 'Type of Accident', 'Falls in the Past Two Years', 'Smoking Status', 'Alcohol Consumption Frequency in the Past Year']
data[categorical_columns] = data[categorical_columns].apply(lambda x: x.astype('category'))

# 划分数据集
train, test = train_test_split(data, test_size=0.3, random_state=42)

# 显示划分后的数据集大小
print("训练集大小:", train.shape)
print("测试集大小:", test.shape)

# 假设data是一个Pandas DataFrame，包含所有数据
train_ratio = 0.7
test_ratio = 1 - train_ratio
train_size, test_size = int(train_ratio * len(data)), int(test_ratio * len(data))

# 随机拆分数据集
train_data, test_data = train_test_split(data, train_size=train_size, random_state=42)

# 构建随机生存森林模型
rfsrc = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rfsrc.fit(train_data.drop(['time', 'status'], axis=1), train_data['time'])

# 预测生存时间
train_pred = rfsrc.predict(train_data.drop(['time', 'status'], axis=1))
test_pred = rfsrc.predict(test_data.drop(['time', 'status'], axis=1))

# 绘制生存曲线
kmf = KaplanMeierFitter()
# 注意：这里使用event_observed参数而不是groups
kmf.fit(train_data['time'], event_observed=train_data['status'])
kmf.plot_survival_function()

# 手动计算Brier分数
def brier_score(event_times, predicted_probabilities, event_observed):
    """
    Calculate Brier score.

    Parameters:
    - event_times: array-like, the observed event times.
    - predicted_probabilities: array-like, the predicted probabilities of the event occurring.
    - event_observed: array-like, the observed event indicator (1 if event occurred, 0 if censored).

    Returns:
    - The Brier score.
    """
    # Ensure the inputs are numpy arrays
    event_times = np.asarray(event_times)
    predicted_probabilities = np.asarray(predicted_probabilities)
    event_observed = np.asarray(event_observed)

    # Calculate the Brier score
    bs = np.mean((predicted_probabilities - event_observed) ** 2)
    return bs

# 我们需要将预测结果转换为生存概率
# 这里假设预测结果是生存时间，我们将其转换为生存概率
train_probs = np.exp(-train_pred / train_data['time'])
test_probs = np.exp(-test_pred / test_data['time'])

# 计算Brier分数
bs_km = brier_score(test_data['time'], test_probs, test_data['status'])
print(f"Brier Score (KM): {bs_km}")


from lifelines.utils import concordance_index

# 计算C-index
c_index = concordance_index(test_data['time'], test_probs, test_data['status'])
print(f"Concordance Index: {c_index}")

# 变量重要性
importances = rfsrc.feature_importances_
indices = np.argsort(importances)[::-1]
# 绘制变量重要性
plt.figure()
plt.title('Feature Importances')
plt.bar(range(train_data.drop(['time', 'status'], axis=1).shape[1]), importances[indices], color="blue", align="center")
plt.xticks(range(train_data.drop(['time', 'status'], axis=1).shape[1]), train_data.drop(['time', 'status'], axis=1).columns[indices], rotation=90)
plt.xlim([-1, train_data.drop(['time', 'status'], axis=1).shape[1]])
plt.show()

# 绘制训练集和测试集的生存曲线
kmf_train = KaplanMeierFitter()
kmf_train.fit(train_data['time'], event_observed=train_data['status'])
kmf_test = KaplanMeierFitter()
kmf_test.fit(test_data['time'], event_observed=test_data['status'])

plt.figure(figsize=(10, 6))
kmf_train.plot_survival_function(label='Training Set')
kmf_test.plot_survival_function(label='Test Set')
plt.title('Survival Curves for Training and Test Sets')
plt.legend()
plt.show()

# 绘制随机生存森林预测的生存曲线
# 我们需要将预测结果转换为生存概率
train_probs = np.exp(-train_pred / train_data['time'])
test_probs = np.exp(-test_pred / test_data['time'])


# 绘制预测生存时间与实际生存时间的比较图
plt.figure(figsize=(10, 6))
plt.scatter(train_data['time'], train_pred, color='blue', label='Training Data', alpha=0.5)
plt.scatter(test_data['time'], test_pred, color='green', label='Test Data', alpha=0.5)
plt.plot([train_data['time'].min(), train_data['time'].max()], [train_data['time'].min(), train_data['time'].max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Survival Time')
plt.ylabel('Predicted Survival Time')
plt.title('Actual vs Predicted Survival Time')
plt.legend()
plt.show()

# 绘制预测生存概率与实际生存状态的比较图
plt.figure(figsize=(10, 6))
plt.hist(train_probs, bins=20, color='blue', alpha=0.5, label='Training Data')
plt.hist(test_probs, bins=20, color='green', alpha=0.5, label='Test Data')
plt.xlabel('Predicted Survival Probability')
plt.ylabel('Frequency')
plt.title('Predicted Survival Probability Distribution')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import coxnam.train.train as ct
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('E:\\workspace\\项目文件\\项目11\\论文数据.csv').drop(columns=['id'])
# 划分数据集
train_df,test_df = train_test_split(
    df, test_size=0.2, random_state=42
)
train_X_tensor, train_y_tensor, train_duration_tensor, train_event_tensor, X=ct.load_and_prepare_data(train_df)
test_X_tensor, test_y_tensor, test_duration_tensor, test_event_tensor, X=ct.load_and_prepare_data(test_df)

coxnam_model=ct.train_model(train_X_tensor, train_duration_tensor, train_event_tensor, device=device,num_epochs=50,model_name='coxnam')
c_index,b_score=ct.evaluate_model(coxnam_model.to('cpu'), test_X_tensor, test_duration_tensor, test_event_tensor)
print('C-index: {:.4f}'.format(c_index), 'Brier Score: {:.4f}'.format(b_score))

coxnet_model=ct.train_model(train_X_tensor, train_duration_tensor, train_event_tensor, device=device,num_epochs=50,model_name='coxnet')

c_index,b_score=ct.evaluate_model(coxnet_model.to('cpu'), test_X_tensor, test_duration_tensor, test_event_tensor)
print('C-index: {:.4f}'.format(c_index), 'Brier Score: {:.4f}'.format(b_score))

lgb_model=lgb.LGBMRegressor()
lgb_model.fit(train_df.drop(['time', 'status'], axis=1),train_df['time'])
# 预测生存时间
train_pred = lgb_model.predict(train_df.drop(['time', 'status'], axis=1))
test_pred = lgb_model.predict(test_df.drop(['time', 'status'], axis=1))
# 计算Brier分数
bs_km = ct.brier_score(test_df['time'], test_pred, test_df['status'],time_point=np.inf)
print(f"Brier Score (KM): {bs_km}")
c_index = concordance_index(test_df['time'], test_pred, test_df['status'])
print(f"Concordance Index: {c_index}")

# 训练Cox模型
cph = CoxPHFitter()
cph.fit(train_df, duration_col='time', event_col='status')
# 获取每个个体在每个时间点的预测生存概率
predicted_survival = cph.predict_survival_function(test_df)
# 计算Brier分数
bs_km = ct.brier_score(test_df['time'], predicted_survival.loc[97].values, test_df['status'],time_point=np.inf)
print(f"Brier Score (KM): {bs_km}")
# 输出C-index
print(f'Concordance Index: {cph.concordance_index_}')