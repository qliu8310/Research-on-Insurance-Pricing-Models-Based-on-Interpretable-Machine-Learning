'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Embedding, Reshape, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# ===============================
# 1. 数据预处理
# ===============================
data = pd.read_csv("C:/Users/32052/Desktop/mortalitydata.csv")

countries = ["Norway_female", "Norway_male", "Finland_female", "Finland_male",
             "Sweden_female", "Sweden_male", "Denmark_female", "Denmark_male",
             "Ireland_female", "Ireland_male", "UK_female", "UK_male",
             "Netherlands_female", "Netherlands_male", "Belgium_female", "Belgium_male",
             "France_female", "France_male", "Switzerland_female", "Switzerland_male",
             "Italy_female", "Italy_male", "Spain_female", "Spain_male",
             "Portugal_female", "Portugal_male", "Czechia_female", "Czechia_male",
             "Hungary_female", "Hungary_male", "Bulgaria_female", "Bulgaria_male"
             ]

# 数据清洗：替换 0 和 inf，向前填充，取对数
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 划分训练集 & 测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values

num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)

# 转换为 (年 × 年龄 × 国家) 矩阵
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))

# ===============================
# 2. 构造训练数据（时间窗口=5，年龄窗口=3）
# ===============================
def improved_create_dataset(data, time_steps=5, age_window=3, stride=1):
    X, y, countries_idx, years_idx, ages_idx = [], [], [], [], []

    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window, stride):
            for country in range(data.shape[2]):
                # 提取时间-年龄窗口
                time_slice = data[year - time_steps:year,
                             age - age_window:age + age_window + 1,
                             country]

                # 检查数据有效性
                if np.any(np.isnan(time_slice)) or np.any(np.isinf(time_slice)):
                    continue

                X.append(time_slice)
                y.append(data[year, age, country])
                countries_idx.append(country)
                years_idx.append(year)
                ages_idx.append(age)

    return (np.array(X)[..., np.newaxis], np.array(y),
            np.array(countries_idx), np.array(years_idx), np.array(ages_idx))


# 使用改进的数据生成
X, y, countries_idx, years_idx, ages_idx = improved_create_dataset(mortality_train)

# 划分训练集和测试集
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries_idx, test_size=0.2, random_state=42
)

# ===============================
# 3. 构建 3D-CNN 模型
# ===============================
def build_model(num_countries):
    input_data = Input(shape=(5, 7, 1, 1))   # 时间×年龄窗口
    country_input = Input(shape=(1,))

    # 国家嵌入
    embedding = Embedding(input_dim=num_countries, output_dim=5, input_length=1)(country_input)
    embedding = Reshape((1, 1, 1, 5))(embedding)

    # 3D-CNN 主干
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

model = build_model(num_countries=num_countries)
model.fit([X_train, countries_train], y_train, epochs=10, batch_size=32,
          validation_data=([X_test, countries_test], y_test))

# ===============================
# 评估函数
# ===============================
def evaluate_model_performance(model, X_test, countries_test, y_test):
    # 预测
    y_pred = model.predict([X_test, countries_test])

    # 多种评估指标
    mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    r2 = 1 - mse / np.var(y_test)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 可视化预测 vs 真实值
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred.flatten()
    plt.hist(residuals, bins=50)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return y_pred, residuals


# ===============================
# 改进 Saliency Map
# ===============================
def improved_saliency_map(model, input_data, country_index, target_layer_idx=-2):
    """
    改进的Saliency Map计算，可以选择中间层
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.output, model.layers[target_layer_idx].output]
    )

    with tf.GradientTape(persistent=True) as tape:
        input_data_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        country_tensor = tf.convert_to_tensor(country_index, dtype=tf.int32)

        tape.watch(input_data_tensor)
        predictions, layer_output = grad_model([input_data_tensor, country_tensor])

        # 对预测值求梯度
        loss = predictions[:, 0]

    grads = tape.gradient(loss, input_data_tensor)
    saliency = tf.reduce_mean(tf.abs(grads), axis=-1)  # 平均掉最后的通道维度

    return saliency.numpy().squeeze()


# ===============================
# 示例：计算 Saliency Map
# ===============================
country_index = np.array([1])  # 假设 Norway_female 在列表第0个
input_data = X_test[0:1]

# 计算 saliency map
saliency_map = improved_saliency_map(model, input_data, country_index)

# ===============================
# 聚合特征贡献 (时间 × 年龄)
# ===============================
# 年龄贡献（平均时间维度）
age_contrib = np.mean(saliency_map, axis=0)

# 时间贡献（平均年龄维度，并翻转顺序）
# 注意：原本 [0]=最早, [4]=最近 -> 翻转后 [0]=最近, [4]=最早
time_contrib = np.mean(saliency_map, axis=1)[::-1]

# ===============================
# 可视化
# ===============================
# 年龄索引：-3 到 3
age_indices = np.arange(-3, 4)  # 共7个点
# 时间索引：0 到 4 (0=最近, 4=最久)
time_indices = np.arange(0, 5)

plt.figure(figsize=(12, 5))

# 年龄贡献
plt.subplot(1, 2, 1)
plt.bar(age_indices, age_contrib)
plt.xlabel("Age Window (-3 ~ +3)")
plt.ylabel("Importance")
plt.title("Feature Contribution by Age")

# 时间贡献
plt.subplot(1, 2, 2)
plt.bar(time_indices, time_contrib)
plt.xlabel("Time Window (0=Recent, 4=Oldest)")
plt.ylabel("Importance")
plt.title("Feature Contribution by Time")

plt.tight_layout()
plt.show()
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# ===============================
# 1. 数据预处理（保持不变）
# ===============================
data = pd.read_csv("C:/Users/32052/Desktop/mortalitydata.csv")

countries = ["Norway_female", "Norway_male", "Finland_female", "Finland_male",
             "Sweden_female", "Sweden_male", "Denmark_female", "Denmark_male",
             "Ireland_female", "Ireland_male", "UK_female", "UK_male",
             "Netherlands_female", "Netherlands_male", "Belgium_female", "Belgium_male",
             "France_female", "France_male", "Switzerland_female", "Switzerland_male",
             "Italy_female", "Italy_male", "Spain_female", "Spain_male",
             "Portugal_female", "Portugal_male", "Czechia_female", "Czechia_male",
             "Hungary_female", "Hungary_male", "Bulgaria_female", "Bulgaria_male"
             ]

# 数据清洗：替换 0 和 inf，向前填充，取对数
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 划分训练集 & 测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values

num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)

# 转换为 (年 × 年龄 × 国家) 矩阵
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))


# ===============================
# 2. 构造训练数据（保持不变）
# ===============================
def improved_create_dataset(data, time_steps=5, age_window=3, stride=1):
    X, y, countries_idx, years_idx, ages_idx = [], [], [], [], []

    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window, stride):
            for country in range(data.shape[2]):
                # 提取时间-年龄窗口
                time_slice = data[year - time_steps:year,
                             age - age_window:age + age_window + 1,
                             country]

                # 检查数据有效性
                if np.any(np.isnan(time_slice)) or np.any(np.isinf(time_slice)):
                    continue

                X.append(time_slice)
                y.append(data[year, age, country])
                countries_idx.append(country)
                years_idx.append(year)
                ages_idx.append(age)

    return (np.array(X)[..., np.newaxis], np.array(y),
            np.array(countries_idx), np.array(years_idx), np.array(ages_idx))


# 使用改进的数据生成
X, y, countries_idx, years_idx, ages_idx = improved_create_dataset(mortality_train)

# 划分训练集和测试集
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries_idx, test_size=0.2, random_state=42
)


# ===============================
# 3. 构建 2D-CNN 模型（带国家嵌入层）
# ===============================
def build_2d_cnn_model(num_countries, params):
    # 主输入：时间×年龄窗口特征
    input_data = Input(shape=(5, 7, 1))  # 5个时间步 × 7个年龄窗口 × 1个通道

    # 国家输入：用于嵌入层
    country_input = Input(shape=(1,))

    # 国家嵌入层
    embedding = Embedding(
        input_dim=num_countries,
        output_dim=5,
        input_length=1
    )(country_input)
    embedding = Reshape((5,))(embedding)  # 展平嵌入向量

    # 2D-CNN 主干网络
    x = Conv2D(int(params['filters1']), (3, 3), activation='relu', padding='same')(input_data)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(int(params['filters2']), (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # 拼接CNN特征和国家嵌入特征
    concat = Concatenate()([x, embedding])

    # 全连接层
    x = Dense(int(params['dense_units']), activation='relu')(concat)
    output = Dense(1)(x)  # 输出死亡率预测值

    # 构建模型
    model = Model(inputs=[input_data, country_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error'
    )
    return model


# 模型参数（女性和男性分别设置）
params_female = {'filters1': 112, 'filters2': 64, 'dense_units': 200, 'learning_rate': 0.00019}
params_male = {'filters1': 96, 'filters2': 224, 'dense_units': 250, 'learning_rate': 0.00010}

# 根据需要选择参数（这里以女性为例）
#params = params_female
params = params_male
# 构建并训练模型
model = build_2d_cnn_model(num_countries=num_countries, params=params)
model.fit(
    [X_train, countries_train],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test, countries_test], y_test)
)


# ===============================
# 评估函数（保持不变）
# ===============================
def evaluate_model_performance(model, X_test, countries_test, y_test):
    # 预测
    y_pred = model.predict([X_test, countries_test])

    # 确保y_pred是一维数组
    y_pred = y_pred.flatten()

    # 多种评估指标 - 计算标量值而不是数组
    mse = np.mean(tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy())
    mae = np.mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy())
    r2 = 1 - mse / np.var(y_test)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 可视化预测 vs 真实值
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=50)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return y_pred, residuals


# 评估模型
y_pred, residuals = evaluate_model_performance(model, X_test, countries_test, y_test)


# ===============================
# Saliency Map 调整为2D
# ===============================
def improved_saliency_map(model, input_data, country_index, target_layer_idx=-2):
    """
    改进的Saliency Map计算，适配2D-CNN
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.output, model.layers[target_layer_idx].output]
    )

    with tf.GradientTape(persistent=True) as tape:
        input_data_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        country_tensor = tf.convert_to_tensor(country_index, dtype=tf.int32)

        tape.watch(input_data_tensor)
        predictions, layer_output = grad_model([input_data_tensor, country_tensor])

        # 对预测值求梯度
        loss = predictions[:, 0]

    grads = tape.gradient(loss, input_data_tensor)
    saliency = tf.reduce_mean(tf.abs(grads), axis=-1)  # 平均掉最后的通道维度

    return saliency.numpy().squeeze()


# ===============================
# 示例：计算并可视化Saliency Map
# ===============================
country_index = np.array([1])  # 选择一个国家索引
input_data = X_test[0:1]

# 计算saliency map
saliency_map = improved_saliency_map(model, input_data, country_index)

# 聚合特征贡献 (时间 × 年龄)
age_contrib = np.mean(saliency_map, axis=0)  # 年龄贡献（平均时间维度）
time_contrib = np.mean(saliency_map, axis=1)[::-1]  # 时间贡献（平均年龄维度，并翻转顺序）

# 可视化
age_indices = np.arange(-3, 4)  # 共7个点
time_indices = np.arange(0, 5)  # 0=最近, 4=最久

plt.figure(figsize=(12, 5))

# 年龄贡献
plt.subplot(1, 2, 1)
plt.bar(age_indices, age_contrib)
plt.xlabel("Age Window (-3 ~ +3)")
plt.ylabel("Importance")
plt.title("Feature Contribution by Age")

# 时间贡献
plt.subplot(1, 2, 2)
plt.bar(time_indices, time_contrib)
plt.xlabel("Time Window (0=Recent, 4=Oldest)")
plt.ylabel("Importance")
plt.title("Feature Contribution by Time")

plt.tight_layout()
plt.show()
