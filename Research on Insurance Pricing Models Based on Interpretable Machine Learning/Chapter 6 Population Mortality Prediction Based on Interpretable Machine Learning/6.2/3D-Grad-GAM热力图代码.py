import numpy as np
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import khatri_rao
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Reshape, Concatenate, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("C:/Users/32052/Desktop/mortalitydata.csv")

# 输出0值和inf值的位置
zero_inf_locations = (data.iloc[:, 2:33] == 0) | np.isinf(data.iloc[:, 2:9])
zero_inf_indices = np.where(zero_inf_locations)
print("0值和inf值的位置：")
for row, col in zip(*zero_inf_indices):
    print(f"行 {row}，列 {col + 2}")

# 向上填充0值和inf值
data.iloc[:, 2:33] = data.iloc[:, 2:33].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data.iloc[:, 2:33] = data.iloc[:, 2:33].fillna(method='ffill')

countries = ["Norway_female", "Norway_male", "Finland_female", "Finland_male",
             "Sweden_female", "Sweden_male", "Denmark_female", "Denmark_male",
             "Ireland_female", "Ireland_male", "UK_female", "UK_male",
             "Netherlands_female", "Netherlands_male", "Belgium_female", "Belgium_male",
             "France_female", "France_male", "Switzerland_female", "Switzerland_male",
             "Italy_female", "Italy_male", "Spain_female", "Spain_male",
             "Portugal_female", "Portugal_male", "Czechia_female", "Czechia_male",
             "Hungary_female", "Hungary_male", "Bulgaria_female", "Bulgaria_male"
             ]

# 取对数并更新数据框
data[countries] = data[countries].apply(lambda x: np.log(x))
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]
# 训练集
years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values
# 计算数据的维度
num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)
# 将数据塑造成三维空间数据
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))
# 测试集
years_test = test_data['year'].values
ages_test = test_data['age'].values
mortality_rates_test = test_data[countries].values
# 计算数据的维度
num_years_test = len(np.unique(years_test))
num_ages_test = len(np.unique(ages_test))
num_countries = len(countries)
# 将数据塑造成三维空间数据
mortality_test = mortality_rates_test.reshape((num_years_test, num_ages_test, num_countries))

ax_h = train_data.groupby('age')[countries].mean()

ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h
ax_t = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, 5))
ax_t = np.transpose(ax_t, (2, 0, 1))


def manual_khatri_rao(matrices):
    """手动实现 Khatri-Rao 积"""
    if len(matrices) == 0:
        return np.array([])

    result = matrices[0]
    for i in range(1, len(matrices)):
        # Khatri-Rao 积是列向量的 Kronecker 积
        r, c = result.shape
        r2, c2 = matrices[i].shape

        if c != c2:
            raise ValueError("All matrices must have the same number of columns for Khatri-Rao product")

        new_result = np.zeros((r * r2, c))
        for col in range(c):
            new_result[:, col] = np.kron(result[:, col], matrices[i][:, col])
        result = new_result

    return result


def cp_als(tensor: np.ndarray, R=1, max_iter=100, svd_method='randomized_svd'):
    N = tl.ndim(tensor)
    tensor_shape = tensor.shape

    # 第一步：SVD 初始化
    lbd, A = initialize_cp(tensor, R, init='svd', svd=svd_method,
                           random_state=0,
                           normalize_factors=True)

    for epoch in tqdm(range(max_iter)):
        for n in range(N):
            # 第二步：计算共享张量 V
            V = np.ones((R, R))
            for i in range(N):
                if i != n:
                    V = V * np.matmul(A[i].T, A[i])

            # 第三步：计算 Khatri-Rao 积（不包括第n个矩阵）
            matrices = [A[i] for i in range(N) if i != n]
            T = manual_khatri_rao(matrices)

            # 第四步：展开张量并更新因子矩阵
            X_unfolded = tl.unfold(tensor, mode=n)

            # 计算伪逆并确保维度匹配
            V_pinv = np.linalg.pinv(V)

            # 更新因子矩阵
            A[n] = np.dot(np.dot(X_unfolded, T), V_pinv)

            # 第五步：归一化因子矩阵
            norms = np.linalg.norm(A[n], axis=0)
            lbd = norms.copy()
            A[n] = A[n] / norms

        # 第六步：检查收敛性
        # 重构张量来检查误差
        reconstructed = np.zeros(tensor_shape)
        for r in range(R):
            factor = A[0][:, r][:, np.newaxis, np.newaxis]
            for i in range(1, N):
                factor = factor * A[i][:, r][np.newaxis, :, np.newaxis] if i == 1 else factor * A[i][:, r][np.newaxis,
                                                                                                np.newaxis, :]
            reconstructed += lbd[r] * factor

        error = np.linalg.norm(tensor - reconstructed)
        if error <= 1e-7:
            print(f"Converged at iteration {epoch}")
            return A, lbd, epoch

    return A, lbd, max_iter


# 主函数
if __name__ == '__main__':
    # 调用 cp_als 函数，将 mx_t_h 作为输入，并指定有效的 SVD 方法
    A, lbd, epoch = cp_als(mx_t_h, R=5, max_iter=1000, svd_method='randomized_svd')

    # 输出分解的张量
    reconstructed_tensor = np.zeros(mx_t_h.shape)
    for r in range(len(lbd)):
        factor = A[0][:, r][:, np.newaxis, np.newaxis] * A[1][:, r][np.newaxis, :, np.newaxis] * A[2][:, r][np.newaxis,
                                                                                                 np.newaxis, :]
        reconstructed_tensor += lbd[r] * factor

    # 打印分解的张量
    print("Decomposed tensor shape:", reconstructed_tensor.shape)

    # 打印每个 lambda_r 的值
    print("Lambda values:")
    print(lbd)

    # 打印重构误差和迭代次数
    print("Reconstruction error:", np.linalg.norm(mx_t_h - reconstructed_tensor))
    print("Number of epochs:", epoch)

A0 = A[0]
A1 = A[1]
A2 = A[2]

# 初始化重构张量
tensor_pred = np.zeros((66, 101, 32))

# 计算 tensor_pred
for r in range(5):
    # 计算每个因子矩阵的第 r 列，reshape 成三维张量
    factor = np.outer(A[0][:, r], np.outer(A[1][:, r], A[2][:, r])).reshape(66, 101, 32)
    # 将每个 factor 乘以对应的 lambda_r，并累加到 tensor_pred 中
    tensor_pred += lbd[r] * factor

# 打印 tensor_pred
print("tensor_pred shape:", tensor_pred.shape)

fitting = tensor_pred + ax_t_h

# 预测
# 存储预测结果的矩阵
prediction_matrix = np.zeros((5, 5))

# 对每一列数据进行预测
for i in range(5):
    # 使用ARIMA模型进行预测
    try:
        model = ARIMA(A0[:, i], order=(1, 1, 1))
        model_fit = model.fit()
        # 预测未来5年的数据
        forecast = model_fit.forecast(steps=5)
        # 将预测结果存储在预测矩阵中的对应列
        prediction_matrix[:, i] = forecast
    except:
        # 如果ARIMA失败，使用简单线性外推
        prediction_matrix[:, i] = np.linspace(A0[-1, i], A0[-1, i] + (A0[-1, i] - A0[-2, i]), 5)

print("预测结果矩阵：")
print(prediction_matrix)

tensor_pred2 = np.zeros((5, 101, 32))
# 计算预测张量
for r in range(5):
    # 计算每个因子矩阵的第 r 列，reshape 成三维张量
    factor1 = np.outer(prediction_matrix[:, r], np.outer(A[1][:, r], A[2][:, r])).reshape(5, 101, 32)
    # 将每个 factor 乘以对应的 lambda_r，并累加到 tensor_pred 中
    tensor_pred2 += lbd[r] * factor1

pred = tensor_pred2 + ax_t
mse_cpd = np.mean((mortality_test - pred) ** 2)
residual = mortality_train - fitting


# ------------------------------------------------------
# 构造训练集和标签
def create_dataset(data, time_steps=5, age_window=3):
    X, y, countries = [], [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            for country in range(data.shape[2]):
                X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, country])
                y.append(data[year, age, country])
                countries.append(country)
    return np.array(X), np.array(y), np.array(countries)


X, y, countries = create_dataset(residual)
X = X[..., np.newaxis, np.newaxis]  # 添加两个通道维度

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries, test_size=0.2, random_state=42
)


# 构建3D CNN模型带嵌入层
def build_model(num_countries):
    input_data = Input(shape=(5, 7, 1, 1))
    country_input = Input(shape=(1,))

    embedding = Embedding(input_dim=num_countries, output_dim=5, input_length=1)(country_input)
    embedding = Reshape((1, 1, 1, 5))(embedding)

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_data)
    pool1 = MaxPooling3D((2, 2, 1))(conv1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D((2, 2, 1))(conv2)
    flat = Flatten()(pool2)

    concat = Concatenate()([flat, Flatten()(embedding)])
    dense1 = Dense(64, activation='relu')(concat)
    output = Dense(1)(dense1)

    model = Model(inputs=[input_data, country_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


model = build_model(num_countries=32)

# 训练模型
model.fit([X_train, countries_train], y_train, epochs=10, batch_size=32,
          validation_data=([X_test, countries_test], y_test))


# 预测2016年的死亡率残差
def predict_next_year(data, model, time_steps=5, age_window=3):
    predicted = np.zeros((101, 32))
    for age in range(101):
        for country in range(32):
            # 处理边界年龄，使用零填充
            if age < age_window:
                padded_data = np.pad(data[-time_steps:, :age + age_window + 1, country],
                                     ((0, 0), (age_window - age, 0)), mode='constant')
            elif age >= 101 - age_window:
                padded_data = np.pad(data[-time_steps:, age - age_window:, country],
                                     ((0, 0), (0, age_window - (100 - age))), mode='constant')
            else:
                padded_data = data[-time_steps:, age - age_window:age + age_window + 1, country]

            input_data = padded_data[np.newaxis, ..., np.newaxis, np.newaxis]
            country_input = np.array([country])[np.newaxis]
            predicted[age, country] = model.predict([input_data, country_input])
    return predicted


# 预测2016-2020年的死亡率残差
predictions = []
residual_padded = residual.copy()

for year in range(2016, 2021):
    # 预测下一年
    next_year_residual = predict_next_year(residual_padded, model)
    predictions.append(next_year_residual)

    # 更新残差数据，加入新预测的数据
    residual = np.concatenate((residual, next_year_residual[np.newaxis, ...]), axis=0)
    residual_padded = residual

predictions_residual_2016_2020 = np.stack(predictions, axis=0)

pred_sum = pred + predictions_residual_2016_2020

forecast_mse_2016_2020 = np.mean((pred_sum - mortality_test) ** 2)
forecast_mse1_2016_2020 = np.mean((pred - mortality_test) ** 2)

print(f"Forecast MSE 2016-2020: {forecast_mse_2016_2020}")
print(f"Baseline MSE 2016-2020: {forecast_mse1_2016_2020}")

# 初始化列表存储每个国家的 MSE
MSE_CP_list = []
MSE_CNN3_list = []

# 计算每个国家的 MSE
for i in range(32):
    female_CNN3 = pred_sum[:, :, i]
    female_CP = pred[:, :, i]
    female = mortality_test[:, :, i]

    MSE_CNN3 = np.mean((female_CNN3 - female) ** 2)
    MSE_CP = np.mean((female_CP - female) ** 2)

    MSE_CP_list.append(MSE_CP)
    MSE_CNN3_list.append(MSE_CNN3)

    print(f"Country {i + 1} - MSE_CP: {MSE_CP}")
    print(f"Country {i + 1} - MSE_CNN3: {MSE_CNN3}")

# 初始化列表存储每个国家的 MAE
MAE_CP_list = []
MAE_CNN3_list = []

# 计算每个国家的 MSE
for i in range(32):
    female_CNN3 = pred_sum[:, :, i]
    female_CP = pred[:, :, i]
    female = mortality_test[:, :, i]

    MAE_CNN3 = np.mean(np.abs(female_CNN3 - female) )
    MAE_CP = np.mean(np.abs(female_CP - female) )
    MAE_CP_list.append(MAE_CP)
    MAE_CNN3_list.append(MAE_CNN3)

    print(f"Country {i + 1} - MAE_CP: {MAE_CP}")
    print(f"Country {i + 1} - MAE_CNN3: {MAE_CNN3}")

# 初始化列表存储每个国家的 MAE
MB_CP_list = []
MB_CNN3_list = []

# 计算每个国家的 MSE
for i in range(32):
    female_CNN3 = pred_sum[:, :, i]
    female_CP = pred[:, :, i]
    female = mortality_test[:, :, i]

    MB_CNN3 = np.mean(female_CNN3 - female)
    MB_CP = np.mean(female_CP - female)

    MB_CP_list.append(MB_CP)
    MB_CNN3_list.append(MB_CNN3)

    print(f"Country {i + 1} - MAE_CP: {MB_CP}")
    print(f"Country {i + 1} - MAE_CNN3: {MB_CNN3}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

last_conv_layer_name = "conv3d_1"

# 2️⃣ 构建 Grad-CAM 解释模型
grad_model = Model(inputs=[model.input[0], model.input[1]],
                   outputs=[model.get_layer(last_conv_layer_name).output, model.output])

# 3️⃣ 计算 Grad-CAM
def compute_gradcam(input_data, country_index, model, layer_name):
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    country_index = tf.convert_to_tensor(country_index, dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(input_data)  # 监视输入
        conv_outputs, predictions = grad_model([input_data, country_index])
        loss = tf.reduce_mean(predictions)  # 计算损失

    grads = tape.gradient(loss, conv_outputs)  # 计算梯度

    if grads is None:
        print("⚠️ Warning: Gradients are None. Possible vanishing gradient problem.")
        return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]))

    pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))  # 平均梯度
    conv_outputs = conv_outputs[0].numpy()
    heatmap = np.mean(conv_outputs * pooled_grads.numpy(), axis=-1)

    # 5️⃣ 修正热力图归一化
    if np.max(heatmap) != np.min(heatmap):
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    else:
        heatmap = np.zeros_like(heatmap)  # 避免 NaN

    return heatmap

# 4️⃣ 选择要分析的国家
country_index = np.array([1])  # 选择第一个国家
input_data = X_test[0:1]  # 选择一个样本

# 5️⃣ 生成 Grad-CAM 热力图
heatmap = compute_gradcam(input_data, country_index, model, last_conv_layer_name)

# 6️⃣ 可视化 Grad-CAM，修正 X 轴（时间）和 Y 轴（年龄）
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap='jet', extent=[0, 5, -3, 3], origin="lower", aspect="auto", interpolation="bicubic")
plt.colorbar(label="Importance")

# 设置 X 轴（时间）和 Y 轴（年龄）的刻度
plt.xticks(np.linspace(0, 5, 5))
plt.yticks(np.linspace(-3, 3, 7))

# 轴标签 & 标题
plt.xlabel("Years (time sequence)")
plt.ylabel("Ages")
plt.title(f"Grad-CAM Heatmap for Country {country_index[0]}")

# 显示图像
plt.show()