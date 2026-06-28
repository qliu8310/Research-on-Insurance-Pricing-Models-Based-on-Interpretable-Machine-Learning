import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.models import Sequential, Model
import numpy as np
import math
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Embedding
from tensorflow.keras.layers import Layer
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

#读取数据
df = pd.read_csv('E:/xiazai/Human Mortality Database/Chinese/female.csv')

#死亡率换成对数死亡率
df['mx'] = np.log(df['mx'] )
print(df.head())

# 分割训练集和测试集
train_df = df[df['year'] < 2012]  
test_df = df[df['year'] >= 2012]

train_df = train_df.groupby(['year', 'age'])['mx'].first().reset_index()
# 将数据转换为矩阵，其中行代表不同的年份，列代表不同的年龄
train_mat = pd.pivot_table(train_df, values='mx', index='year', columns='age', fill_value=0).values
test_df = test_df.groupby(['year', 'age'])['mx'].first().reset_index()
# 将数据转换为矩阵，其中行代表不同的年份，列代表不同的年龄
test_mat = pd.pivot_table(test_df, values='mx', index='year', columns='age', fill_value=0).values


#LSTM
# 数据标准化（LSTM通常对数据进行标准化或归一化）
scaler = MinMaxScaler(feature_range=(0, 1))
train_mat_scaled = scaler.fit_transform(train_mat)
test_mat_scaled = scaler.transform(test_mat)

# 创建时间序列数据 (samples, time_steps, features)
def create_dataset(data, time_step=7):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])  # 选择当前时间步和前几个时间步的数据
        y.append(data[i + time_step, :])  # 预测下一个时间步的死亡率
    return np.array(X), np.array(y)

# 假设使用过去5年的数据来预测未来1年的死亡率
time_step = 7
X_train, y_train = create_dataset(train_mat_scaled, time_step)
X_test, y_test = create_dataset(test_mat_scaled, time_step)

# LSTM的输入需要是三维数据： (samples, time_steps, features)
print("Train data shape:", X_train.shape)  # (samples, time_steps, features)
print("Test data shape:", X_test.shape)

# 建立LSTM模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False, activation='relu'))
model.add(Dense(units=y_train.shape[1]))  # 输出的单元数等于年龄段的数量
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 进行预测
predictions = model.predict(X_test)

# 将预测结果反标准化回原始值
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算 MAE（Mean Absolute Error）
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
print(f'MAE (Mean Absolute Error): {mae}')

# 计算 RMSE（Root Mean Squared Error）
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
print(f'RMSE (Root Mean Squared Error): {rmse}')

# 计算 MAPE（Mean Absolute Percentage Error）
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 防止除以0的情况
    non_zero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) 
    return mape

mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
print(f'MAPE (Mean Absolute Percentage Error): {mape}')

# 获取从第六年开始的数据，行数应该是 910 行
# 选择 test_df 的最后 910 行
result_df = test_df.iloc[-273:].copy()
# 将预测结果合并到新的 DataFrame 中
result_df['predicted_log_mortality'] = predictions_rescaled.flatten()
result_df['mx_log'] =y_test_rescaled.flatten()
# 保存合并的数据到 csv 文件
result_df.to_csv('mortality_predictions_all_years.csv', index=False)


# 读取 CSV 文件
df = pd.read_csv('mortality_predictions_all_years.csv')

# 准备颜色列表，每个年份一个颜色
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

# 绘制散点图
plt.figure(figsize=(14, 8))

# 遍历每个年份，并绘制实际和预测的对数死亡率
for i, year in enumerate([2017, 2018, 2019, 2020, 2021]):
    # 提取特定年份的数据
    year_data = df[(df['year'] == year)]
    
    # 检查year_data是否为空
    if not year_data.empty:
        # 提取对应的年龄
        ages = year_data['age'].values
        
        # 提取实际对数死亡率
        actual_log_mx = year_data['mx'].values
        
        # 提取预测对数死亡率
        predicted_log_mx = year_data['predicted_log_mortality'].values
        
        # 绘制实际对数死亡率的曲线
        plt.plot(ages, actual_log_mx, label=f'Actual {year}', color=colors[i % len(colors)], alpha=0.8)
        
        # 绘制预测对数死亡率的虚线
        plt.plot(ages, predicted_log_mx, linestyle='--', label=f'Predicted {year}', color=colors[i % len(colors)], alpha=0.8)
    else:
        print(f"No data for year {year}.")

# 设置图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Log Actual vs Log Predicted Mortality Rate by Age')
plt.xlabel('Age')
plt.ylabel('Log Mortality Rate (log(mx))')

# 显示图形
plt.show()

# 选择第一个样本进行可视化
sample_idx = 2

# 获取贡献值和输入数据（形状变为 (7, 110)）
contributions_sample = contributions_np[sample_idx]  # (7, 110)
X_test_sample = X_test[sample_idx]  # (7, 110)

# 反标准化并转换回mx原始值
X_test_sample_logmx = scaler.inverse_transform(X_test_sample.reshape(7, -1))  # 注意修改为7
X_test_sample_mx = np.exp(X_test_sample_logmx)  # 实际死亡率值

# 计算总贡献并选择前10个年龄
total_contrib = np.sum(contributions_sample, axis=0)  # 沿时间步求和
top10_ages = np.argsort(total_contrib)[-10:][::-1]

# 准备可视化数据
time_steps = np.arange(7)  # 修改为7个时间步
x = np.tile(time_steps, len(top10_ages))
y = np.repeat(top10_ages, 7)  # 修改为7

# 调整点大小范围
contrib_values = contributions_sample[:, top10_ages].T.flatten()  # 形状 (7*10,)
sizes = np.abs(contrib_values)
sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 900 + 100

# 颜色数值（mx值）
colors = X_test_sample_mx[:, top10_ages].T.flatten()  # 形状 (7*10,)

# 创建可视化
plt.figure(figsize=(10, 6))
sc = plt.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)

# 添加年龄连线
for age in top10_ages:
    plt.plot(time_steps, [age]*7, c='gray', ls='--', lw=0.5, alpha=0.5)  # 修改为7

# 坐标轴设置
plt.xticks(time_steps, labels=[f'T-{7-i}' for i in time_steps])  # 修改标签
plt.yticks(top10_ages)
plt.xlabel('Years Before Prediction', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.title('Top 10 Age-Specific Mortality Predictors (7-Year Window)', fontsize=14)  # 更新标题

# 颜色条设置
cbar = plt.colorbar(sc)
cbar.set_label('Mortality Rate (mx)', fontsize=12)

# 网格设置
plt.grid(True, alpha=0.2, ls='--')
plt.tight_layout()
plt.savefig('LSTM解释')
plt.show()

def compute_age_specific_dtd(model, input_data, reference_point, target_age):
    """
    计算输入对特定目标年龄预测的贡献
    :param target_age: 需要分析的输出年龄（0-109）
    """
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    reference_point = tf.convert_to_tensor(reference_point, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_data)
        predictions = model(input_data)
        # 仅保留目标年龄的预测值
        target_predictions = predictions[:, target_age]  # 形状：(batch,)

    # 计算梯度（仅对目标年龄）
    gradients = tape.gradient(target_predictions, input_data)  # 形状：(batch, 7, 110)
    contributions = gradients * (input_data - reference_point)
    return contributions.numpy()

# 分析对80岁死亡率预测的贡献
contributions_80 = compute_age_specific_dtd(model, X_test, reference_point, target_age=80)

# 可视化该年龄的贡献模式
plt.figure(figsize=(10, 6))
plt.imshow(contributions_80[0].T, cmap='viridis', aspect='auto')  # 第一个样本
plt.xlabel('Time Step (T-6 to T-0)')
plt.ylabel('Age')
plt.title('Contribution to Age 80 Prediction')
plt.colorbar()
plt.show()

def analyze_top_contrib_ages(target_age, sample_idx=0, time_steps=7):
    # 计算针对目标年龄的贡献
    contributions = compute_age_specific_dtd(model, X_test, reference_point, target_age)
    
    # 选择指定样本
    contrib_sample = contributions[sample_idx]  # 形状 (7, 110)
    X_test_sample = X_test[sample_idx]         # 形状 (7, 110)

    # ==== 修正部分开始 ====
    # 反标准化时保持特征维度
    X_test_logmx = scaler.inverse_transform(X_test_sample)  # 直接使用原始形状(7,110)
    X_test_mx = np.exp(X_test_logmx)  # 得到实际mx值
    # ==== 修正部分结束 ====

    # 计算每个输入年龄的总贡献（跨时间步求和）
    total_contrib = np.sum(contrib_sample, axis=0)  # 形状 (110,)
    
    # 获取前10个影响最大的输入年龄
    top10_idx = np.argsort(total_contrib)[-10:][::-1]
    top10_ages = top10_idx.tolist()
    top10_contrib = contrib_sample[:, top10_idx]     # 形状 (7, 10)

    # 准备可视化数据
    time_points = np.arange(time_steps)
    ages_grid, time_grid = np.meshgrid(top10_ages, time_points)
    
    # 将数据展平
    x = time_grid.flatten()  # 时间坐标
    y = ages_grid.flatten()  # 年龄坐标
    sizes = np.abs(top10_contrib.T.flatten())  # 贡献绝对值
    colors = X_test_mx[:, top10_idx].T.flatten()  # 实际死亡率值

    # 创建可视化
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        x, y, 
        s=(sizes/np.max(sizes))*800 + 100,  # 动态调整气泡大小
        c=colors, 
        cmap='viridis', 
        alpha=0.7, 
        edgecolors='w', 
        linewidth=0.5
    )

    # 添加年龄连线
    for age in top10_ages:
        plt.plot(time_points, [age]*time_steps, c='gray', ls='--', lw=0.5, alpha=0.5)

    # 坐标轴设置
    plt.xticks(time_points, labels=[f'T-{7-i}' for i in range(time_steps)])
    plt.yticks(top10_ages)
    plt.xlabel('Years Before Prediction', fontsize=12)
    plt.ylabel('Input Age', fontsize=12)
    plt.title(f'Top 10 Contributing Ages for Predicting Age {target_age} Mortality', fontsize=14)

    # 颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Actual Mortality Rate (mx)', fontsize=12)
    
    # 显示关键数值
    #for i, age in enumerate(top10_ages):
     #   plt.text(
      #      x=time_points[-1] + 0.5, 
       #     y=age, 
        #    s=f"Total: {total_contrib[age]:.2f}",
         #   va='center',
          #  ha='left',
           # fontsize=9
        #)

    plt.grid(True, alpha=0.2, ls='--')
    plt.tight_layout()
    plt.savefig('LSTM特定年龄解释.png', dpi=300)
    plt.show()

# 使用示例：分析80岁死亡率预测的前10贡献年龄
analyze_top_contrib_ages(target_age=80, sample_idx=2)

import numpy as np
import matplotlib.pyplot as plt

# 假设 contributions 是 DTD 计算得到的贡献值，形状为 (batch_size, time_steps, features)
# contributions = compute_dtd(model, X_test, reference_point)

# 计算每个时间步的总贡献
time_step_contributions = np.sum(contributions, axis=2)  # 沿特征维度求和

# 计算每个特征的总贡献
feature_contributions = np.sum(contributions, axis=1)  # 沿时间步维度求和

# 可视化每个时间步的总贡献
plt.figure(figsize=(10, 6))
plt.plot(range(time_step_contributions.shape[1]), time_step_contributions[0, :], marker='o')
plt.xlabel('Time Step')
plt.ylabel('Total Contribution')
plt.title('Total Contribution by Time Step')
plt.grid(True)
plt.show()



# 可视化每个特征的总贡献
plt.figure(figsize=(10, 6))
plt.bar(range(feature_contributions.shape[1]), feature_contributions[0, :])
plt.xlabel('Feature (Age)')
plt.ylabel('Total Contribution')
plt.title('Total Contribution by Feature (Age)')
plt.show()

# 分析不同年龄段对预测的影响
age_importance = np.mean(np.abs(predictions_rescaled - y_test_rescaled), axis=0)

# 可视化不同年龄段的平均绝对误差
plt.figure(figsize=(10, 6))
plt.bar(range(len(age_importance)), age_importance)
plt.title('Average Absolute Error by Age')
plt.xlabel('Age')
plt.ylabel('Average Absolute Error')
plt.show()

#RNN
# Reshaping X to be 3D as RNN expects (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# 建立RNN模型
model = Sequential()
model.add(SimpleRNN(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))  # 使用RNN
model.add(Dense(train_mat_scaled.shape[1]))  # 输出层，数量等于年龄的数量

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 预测并反标准化
predictions = model.predict(X_test)

# 反标准化
predictions_rescaled_rnn = scaler.inverse_transform(predictions)

# 计算 MAE（Mean Absolute Error）
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled_rnn)
print(f'MAE (Mean Absolute Error): {mae}')

# 计算 RMSE（Root Mean Squared Error）
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled_rnn))
print(f'RMSE (Root Mean Squared Error): {rmse}')

# 计算 MAPE（Mean Absolute Percentage Error）
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 防止除以0的情况
    non_zero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) 
    return mape

mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled_rnn)
print(f'MAPE (Mean Absolute Percentage Error): {mape}')

# 获取从第六年开始的数据，行数应该是 455 行
# 选择 test_df 的最后 455 行
result_df_rnn = test_df.iloc[-273:].copy()
# 将预测结果合并到新的 DataFrame 中
result_df_rnn['predicted_log_mortality'] = predictions_rescaled_rnn.flatten()
result_df_rnn['mx_log'] =y_test_rescaled.flatten()
# 保存合并的数据到 csv 文件
result_df_rnn.to_csv('mortality_predictions_rnn.csv', index=False)

# 读取 CSV 文件
df = pd.read_csv('mortality_predictions_rnn.csv')

# 准备颜色列表，每个年份一个颜色
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

# 绘制散点图
plt.figure(figsize=(14, 8))

# 遍历每个年份，并绘制实际和预测的对数死亡率
for i, year in enumerate([2017, 2018, 2019, 2020, 2021]):
    # 提取特定年份的数据
    year_data = df[(df['year'] == year)]
    
    # 检查year_data是否为空
    if not year_data.empty:
        # 提取对应的年龄
        ages = year_data['age'].values
        
        # 提取实际对数死亡率
        actual_log_mx = year_data['mx'].values
        
        # 提取预测对数死亡率
        predicted_log_mx = year_data['predicted_log_mortality'].values
        
        # 绘制实际对数死亡率的曲线
        plt.plot(ages, actual_log_mx, label=f'Actual {year}', color=colors[i % len(colors)], alpha=0.8)
        
        # 绘制预测对数死亡率的虚线
        plt.plot(ages, predicted_log_mx, linestyle='--', label=f'Predicted {year}', color=colors[i % len(colors)], alpha=0.8)
    else:
        print(f"No data for year {year}.")

# 设置图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Log Actual vs Log Predicted Mortality Rate by Age')
plt.xlabel('Age')
plt.ylabel('Log Mortality Rate (log(mx))')

# 显示图形
plt.show()

# Transformer
# 读取数据
df = pd.read_csv('E:/xiazai/Human Mortality Database/Chinese/female.csv')

# 死亡率换成对数死亡率
df['mx'] = np.log(df['mx'])
print(df.head())

# 分割训练集和测试集
train_df = df[df['year'] < 2017]
test_df = df[df['year'] >= 2012]

# 对训练集按年和年龄分组并获取第一个值
train_df = train_df.groupby(['year', 'age'])['mx'].first().reset_index()

# 将数据转换为矩阵，其中行代表不同的年份，列代表不同的年龄
train_mat = pd.pivot_table(train_df, values='mx', index='year', columns='age', fill_value=0).values

# 对测试集按年和年龄分组并获取第一个值
test_df = test_df.groupby(['year', 'age'])['mx'].first().reset_index()

# 将数据转换为矩阵，其中行代表不同的年份，列代表不同的年龄
test_mat = pd.pivot_table(test_df, values='mx', index='year', columns='age', fill_value=0).values

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_mat_scaled = scaler.fit_transform(train_mat)
test_mat_scaled = scaler.transform(test_mat)

# 创建时间序列数据 (samples, time_steps, features)
def create_dataset(data, time_step=7):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])  # 选择当前时间步和前几个时间步的数据
        y.append(data[i + time_step, :])  # 预测下一个时间步的死亡率
    return np.array(X), np.array(y)

# 假设使用过去5年的数据来预测未来1年的死亡率
time_step = 7
X_train, y_train = create_dataset(train_mat_scaled, time_step)
X_test, y_test = create_dataset(test_mat_scaled, time_step)

import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, output_dim):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)  # 将输入映射到d_model维度
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, d_model))  # 假设最大长度500
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff) 
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_dim)  # 输出维度是年龄的数量（即预测的死亡率）

    def forward(self, x):
        # 输入数据 (batch_size, seq_len, input_dim) -> (batch_size, seq_len, d_model)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer编码层处理
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 只返回最后一个时间步的输出 (batch_size, d_model)
        x = x[:, -1, :]  # 选取最后一个时间步的输出
        
        # 将编码后的输出传入全连接层进行预测
        x = self.fc_out(x)
        return x

# 定义超参数
input_dim = X_train.shape[2]  # 每个时间步的特征数量，即年龄的数量
d_model = 128  # Transformer模型的隐藏层维度
num_heads = 8  # 多头自注意力机制的头数
num_layers = 4  # Transformer编码器的层数
d_ff = 512  # 前馈网络的隐藏层维度
output_dim = y_train.shape[1]  # 预测的维度是年龄数量，即y_train的第二维


# 创建模型
model = TransformerModel(input_dim=input_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, output_dim=output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换为Tensor
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 训练一个batch
    idx = np.random.choice(X_train_tensor.size(0), batch_size)
    X_batch = X_train_tensor[idx]
    y_batch = y_train_tensor[idx]
    
    # 前向传播
    y_pred = model(X_batch)
    
    # 计算损失
    loss = criterion(y_pred, y_batch)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# 评估模型
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    
# 将预测结果反标准化
y_test_pred = scaler.inverse_transform(y_test_pred.numpy().reshape(-1, y_test_pred.shape[-1]))
y_test = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1, y_test_tensor.shape[-1]))

# 计算预测误差
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE: {mse:.4f}")

# 计算 MAE
mae = mean_absolute_error(y_test, y_test_pred)
print(f"MAE: {mae:.4f}")

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"RMSE: {rmse:.4f}")

# 计算 MAPE
epsilon = 1e-10  # 防止除零错误
mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + epsilon))) 
print(f"MAPE: {mape:.4f}")

# 获取从第六年开始的数据，行数应该是 455 行
# 选择 test_df 的最后 455 行
result_df = test_df.iloc[-273:].copy()
# 将预测结果合并到新的 DataFrame 中
result_df['predicted_log_mortality'] = y_test_pred.flatten()
result_df['mx_log'] =y_test.flatten()
# 保存合并的数据到 csv 文件
result_df.to_csv('mortality_predictions_trans.csv', index=False)

# 读取 CSV 文件
df = pd.read_csv('mortality_predictions_trans.csv')

# 准备颜色列表，每个年份一个颜色
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

# 绘制散点图
plt.figure(figsize=(14, 8))

# 遍历每个年份，并绘制实际和预测的对数死亡率
for i, year in enumerate([2017, 2018, 2019, 2020, 2021]):
    # 提取特定年份的数据
    year_data = df[(df['year'] == year)]
    
    # 检查year_data是否为空
    if not year_data.empty:
        # 提取对应的年龄
        ages = year_data['age'].values
        
        # 提取实际对数死亡率
        actual_log_mx = year_data['mx'].values
        
        # 提取预测对数死亡率
        predicted_log_mx = year_data['predicted_log_mortality'].values
        
        # 绘制实际对数死亡率的曲线
        plt.plot(ages, actual_log_mx, label=f'Actual {year}', color=colors[i % len(colors)], alpha=0.8)
        
        # 绘制预测对数死亡率的虚线
        plt.plot(ages, predicted_log_mx, linestyle='--', label=f'Predicted {year}', color=colors[i % len(colors)], alpha=0.8)
    else:
        print(f"No data for year {year}.")

# 设置图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Log Actual vs Log Predicted Mortality Rate by Age')
plt.xlabel('Age')
plt.ylabel('Log Mortality Rate (log(mx))')

# 显示图形
plt.show()

