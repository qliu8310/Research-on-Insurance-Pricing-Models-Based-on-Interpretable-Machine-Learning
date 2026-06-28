'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Embedding, Reshape, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# 1. 数据处理部分（假定数据文件已存在）
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
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]
years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values
num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))

years_test = test_data['year'].values
ages_test = test_data['age'].values
mortality_rates_test = test_data[countries].values
num_years_test = len(np.unique(years_test))
num_ages_test = len(np.unique(ages_test))
mortality_test = mortality_rates_test.reshape((num_years_test, num_ages_test, num_countries))

ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# 2. 构造训练集和标签
def create_dataset(data, time_steps=5, age_window=3):
    X, y, countries = [], [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            for country in range(data.shape[2]):
                X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, country])
                y.append(data[year, age, country])
                countries.append(country)
    return np.array(X), np.array(y), np.array(countries)

X, y, countries_idx = create_dataset(mx_t_h)  # 使用 mx_t_h 作为残差
X = X[..., np.newaxis, np.newaxis]  # 添加两个通道维度

# 3. 划分训练和测试集
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries_idx, test_size=0.2, random_state=42
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
model.fit([X_train, countries_train], y_train, epochs=10, batch_size=32,
          validation_data=([X_test, countries_test], y_test))

# 5. Grad-CAM部分
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
country_index = np.array([0])  # 选择第一个国家
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
'''

#LIME
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from lime import lime_tabular
import seaborn as sns

# 设置中文字体，确保可视化正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# ===============================
# 1. 数据加载与预处理
# ===============================
# 加载数据
data = pd.read_csv("C:/Users/32052/Desktop/mortalitydata.csv")

# 定义国家列表，明确Norway_female在索引0位置（核心：挪威女性是第1个元素，索引=0）
countries = ["Norway_female", "Norway_male", "Finland_female", "Finland_male",
             "Sweden_female", "Sweden_male", "Denmark_female", "Denmark_male",
             "Ireland_female", "Ireland_male", "UK_female", "UK_male",
             "Netherlands_female", "Netherlands_male", "Belgium_female", "Belgium_male",
             "France_female", "France_male", "Switzerland_female", "Switzerland_male",
             "Italy_female", "Italy_male", "Spain_female", "Spain_male",
             "Portugal_female", "Portugal_male", "Czechia_female", "Czechia_male",
             "Hungary_female", "Hungary_male", "Bulgaria_female", "Bulgaria_male"
             ]

# 数据清洗：处理缺失值和异常值
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()  # 向前填充

# 对数转换前的有效性检查
for country in countries:
    valid_range = (data[country] > 0) & (data[country] < 1e6)  # 设定合理范围
    data.loc[~valid_range, country] = np.nan

# 对数转换和最终填充
data[countries] = data[countries].apply(lambda x: np.log(x) if x.notna().any() else x)
data[countries] = data[countries].fillna(method='ffill').fillna(method='bfill')  # 双向填充

# 划分训练集和测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

# 数据重塑为(年 × 年龄 × 国家)三维矩阵
years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values
num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))

# 计算残差（调整年龄效应）
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h  # 残差 = 原始数据 - 年龄效应


# ===============================
# 2. 构建数据集（时间窗口处理）
# ===============================
def create_dataset(data, time_steps=5, age_window=3):
    """
    构建用于模型训练的数据集
    time_steps: 时间窗口大小（5年）
    age_window: 年龄窗口大小（±3岁，共7个年龄点）
    """
    X, y, country_indices = [], [], []  # 存储特征、标签和国家索引

    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            for country_idx in range(data.shape[2]):
                # 提取时间-年龄窗口数据
                window_data = data[year - time_steps:year,
                              age - age_window:age + age_window + 1,
                              country_idx]

                # 反转时间维度，使T0表示最近一年，T4表示最远一年
                window_data = window_data[::-1]

                # 跳过包含缺失值或异常值的样本
                if np.isnan(window_data).any() or np.isinf(window_data).any():
                    continue

                X.append(window_data)
                y.append(data[year, age, country_idx])  # 目标值：当前时间点的死亡率残差
                country_indices.append(country_idx)  # 记录样本对应的国家索引

    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    country_indices = np.array(country_indices)

    # 打印数据集信息（核心：显示Norway_female样本数，索引=0）
    print(f"数据集构建完成 - 总样本数: {len(X)}")
    print(f"Norway_female样本数: {sum(country_indices == 0)} (国家索引0)")
    return X, y, country_indices


# 生成训练数据
X, y, country_indices = create_dataset(mx_t_h)
# 修改数据形状以适应3D-CNN：添加两个维度 (样本数, 5, 7, 1, 1)
X = X[..., np.newaxis, np.newaxis]  # 添加两个维度，适应3D-CNN输入要求 (样本数, 5, 7, 1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, country_indices, test_size=0.2, random_state=42
)


# ===============================
# 3. 构建和训练3D CNN模型
# ===============================
def build_3d_cnn():
    input_data = Input(shape=(5, 7, 1, 1))
    conv1 = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(input_data)
    pool1 = MaxPooling3D((1, 2, 1))(conv1)
    conv2 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D((1, 2, 1))(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(64, activation='relu')(flat)
    output = Dense(1)(dense1)
    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 创建并训练3D-CNN模型
model = build_3d_cnn()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,  # 从训练集中划分20%作为验证集
    verbose=1  # 显示训练过程
)

# 绘制训练过程中的损失曲线
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练损失曲线')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('均方误差 (MSE)')
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 4. 模型评估
# ===============================
def evaluate_model(model, X_test, y_test):
    """评估模型性能并可视化结果"""
    y_pred = model.predict(X_test, verbose=0).flatten()  # 获取预测结果并展平

    # 计算评估指标
    mse = np.mean((y_test - y_pred) **2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred)** 2) / np.sum((y_test - np.mean(y_test)) **2))

    # 打印评估结果
    print(f"\n模型评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 可视化预测效果
    plt.figure(figsize=(10, 5))

    # 子图1：预测值 vs 真实值
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 理想线
    plt.title('预测值 vs 真实值')
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    # 子图2：残差分布
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=30, color='green')
    plt.title('残差分布')
    plt.xlabel('残差 (真实值-预测值)')
    plt.ylabel('频率')

    plt.tight_layout()
    plt.show()

    return y_pred


# 评估模型在测试集上的表现
y_pred = evaluate_model(model, X_test, y_test)


# ===============================
# 5. LIME解释（核心：针对Norway_female，索引=0）
# ===============================
def predict_fn(inputs):
    """LIME所需的预测函数，将输入转换为3D-CNN模型所需形状"""
    # 重塑为(样本数, 5, 7, 1, 1) - 3D-CNN输入形状
    reshaped = inputs.reshape((-1, 5, 7, 1, 1))
    preds = model.predict(reshaped, verbose=0)
    return preds


# 定义特征名称，明确时间步含义（T0=最近一年）
feature_names = [
    f"T{t}_A{a} (t={t}: {t}年前)"
    for t in range(5)  # 时间步：0-4（0=最近）
    for a in range(-3, 4)  # 年龄偏移：-3到+3
]

# 创建LIME解释器
# 注意：需要将训练数据展平为2D格式供LIME使用
X_train_flat = X_train.reshape((X_train.shape[0], -1))
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_flat,  # 使用展平的训练数据
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=False,
    random_state=42
)

# 筛选测试集中的Norway_female样本（核心：挪威女性的国家索引=0）
norway_female_test_indices = np.where(countries_test == 1)[0]

if len(norway_female_test_indices) == 0:
    print("\n警告：测试集中没有找到Norway_female样本！")
    print("建议：1. 修改train_test_split的random_state参数；2. 增大test_size比例")
else:
    # 选择测试集中第一个Norway_female样本
    sample_idx = norway_female_test_indices[0]
    sample = X_test[sample_idx:sample_idx + 1]
    sample_flat = sample.reshape(1, -1)  # 展平用于LIME
    country_name = countries[countries_test[sample_idx]]  # 确认国家名称（应为Norway_female）

    print(f"\n已选择测试集中的Norway_female样本（索引：{sample_idx}）")
    print(f"解释样本 {sample_idx} (国家: {country_name}) 的预测结果...")

    # 生成LIME解释 - 获取所有特征
    exp = explainer.explain_instance(
        data_row=sample_flat[0],
        predict_fn=predict_fn,
        num_features=len(feature_names)  # 获取所有特征
    )

    # ===============================
    # 6. 可视化LIME解释结果 - 正面和负面影响各前8个特征
    # ===============================
    # 提取所有特征的重要性
    feature_importance = exp.as_list()

    # 分离正面和负面影响
    positive_features = [(f, v) for f, v in feature_importance if v > 0]
    negative_features = [(f, v) for f, v in feature_importance if v < 0]

    # 排序并取前8个
    top_positive = sorted(positive_features, key=lambda x: x[1], reverse=True)[:8]
    top_negative = sorted(negative_features, key=lambda x: x[1])[:8]

    # 合并并排序（负面在前，正面在后）
    selected_features = top_negative + top_positive

    # 准备绘图数据
    features = [item[0] for item in selected_features]
    importance_values = [item[1] for item in selected_features]

    # 简化特征名称 - 去掉括号及内容
    simplified_features = []
    for feature in features:
        # 去掉括号及括号内的所有内容
        if " (" in feature:
            simplified_features.append(feature.split(" (")[0])
        else:
            simplified_features.append(feature)

    # 使用深绿色和正红色
    colors = ['#FF0000' if x < 0 else '#006400' for x in importance_values]  # 深绿色: #006400

    # 创建水平条形图
    plt.figure(figsize=(10, 8))

    # 创建条形图 - 不再添加数值标签
    bars = plt.barh(simplified_features, importance_values, color=colors, alpha=0.8)

    plt.title('Local explanation for Norway_male\n(Red: Negative impact, Green: Positive impact)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=12)

    # 添加垂直线在x=0处
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # 调整布局
    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)
    plt.show()

    # 打印统计信息
    print(f"\n样本预测值: {model.predict(sample, verbose=0)[0][0]:.4f}")
    print(f"显示特征: {len(selected_features)}个 (前8个正面 + 前8个负面)")

    # 打印最重要的特征
    print("\n最重要的正面影响特征:")
    for feature, value in top_positive:
        simplified = feature.split(" (")[0] if " (" in feature else feature
        print(f"  {simplified}: {value:.4f}")

    print("\n最重要的负面影响特征:")
    for feature, value in top_negative:
        simplified = feature.split(" (")[0] if " (" in feature else feature
        print(f"  {simplified}: {value:.4f}")


'''
#全局解释SHAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Embedding, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam

# ===============================
# 1. 读取数据
# ===============================
data = pd.read_csv("C:/Users/32052/Desktop/mortalitydata.csv")

countries = [
    "Norway_female", "Norway_male", "Finland_female", "Finland_male",
    "Sweden_female", "Sweden_male", "Denmark_female", "Denmark_male",
    "Ireland_female", "Ireland_male", "UK_female", "UK_male",
    "Netherlands_female", "Netherlands_male", "Belgium_female", "Belgium_male",
    "France_female", "France_male", "Switzerland_female", "Switzerland_male",
    "Italy_female", "Italy_male", "Spain_female", "Spain_male",
    "Portugal_female", "Portugal_male", "Czechia_female", "Czechia_male",
    "Hungary_female", "Hungary_male", "Bulgaria_female", "Bulgaria_male"
]

data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()

# ===============================
# 2. 残差计算
# ===============================
def compute_residual(country_series):
    df = pd.DataFrame({
        "year": data["year"],
        "age": data["age"],
        "mx": np.log(country_series.values)
    })
    mat = df.pivot(index="age", columns="year", values="mx").fillna(0).values
    ax = np.mean(mat, axis=1)
    kt = np.mean(mat, axis=0)
    bx = np.ones_like(ax) * 0.1
    fitting = np.outer(bx, kt) + ax[:, np.newaxis]
    residual = mat - fitting
    scaler = MinMaxScaler()
    return scaler.fit_transform(residual.reshape(-1,1)).reshape(mat.shape)

# ===============================
# 3. 数据窗口构造
# ===============================
def prepare_data(residual, country_idx, age_window=3, time_steps=5):
    residual_padded = np.pad(residual, ((age_window, age_window),(0,0)), mode='constant')
    X, y, c_ids = [], [], []
    for year in range(residual_padded.shape[1]-time_steps):
        for age in range(age_window, residual_padded.shape[0]-age_window):
            neigh = residual_padded[age-age_window:age+age_window+1, year:year+time_steps]
            X.append(neigh.reshape(time_steps, 2*age_window+1,1,1))
            y.append(residual_padded[age, year+time_steps])
            c_ids.append(country_idx)
    return np.array(X), np.array(y), np.array(c_ids)

# ===============================
# 4. 3D-CNN 模型
# ===============================
def build_model(num_countries):
    input_data = Input(shape=(5,7,1,1))
    country_input = Input(shape=(1,))
    embedding = Embedding(input_dim=num_countries, output_dim=5)(country_input)
    embedding = Reshape((5,))(embedding)

    conv1 = Conv3D(32,(3,3,3),activation='relu',padding='same')(input_data)
    flat = Flatten()(conv1)
    concat = Concatenate()([flat, embedding])
    dense1 = Dense(64, activation='relu')(concat)
    output = Dense(1)(dense1)

    model = Model(inputs=[input_data, country_input], outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model

# ===============================
# 5. 获取 SHAP 值函数 (KernelExplainer)
# ===============================
def get_shap_values(countries_list):
    shap_age, shap_time = [], []
    for c in countries_list:
        idx = countries.index(c)
        residual = compute_residual(data[c])
        X, y, c_ids = prepare_data(residual, idx)
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c_ids, test_size=0.2, random_state=42)

        model = build_model(num_countries=len(countries))
        model.fit([X_train, c_train], y_train, epochs=3, batch_size=32, verbose=0)

        # Kernel SHAP: reshape输入
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        def f(x):
            x_reshaped = x.reshape(-1,5,7,1,1)
            return model.predict([x_reshaped, np.full((x_reshaped.shape[0],1), idx)], verbose=0).flatten()

        explainer = shap.KernelExplainer(f, X_train_flat[:50])
        shap_vals = explainer.shap_values(X_test_flat[:10], nsamples=100)  # 样本少一些加速

        # reshape 回 (time, age)
        shap_matrix = np.abs(np.array(shap_vals)).reshape(-1,5,7)
        shap_age.append(np.mean(np.mean(shap_matrix, axis=0), axis=0))    # 聚合到年龄
        shap_time.append(np.mean(np.mean(shap_matrix, axis=0), axis=1))   # 聚合到时间

    return np.mean(shap_age, axis=0), np.mean(shap_time, axis=0)

# ===============================
# 6. 分男女组计算
# ===============================
female_countries = [c for c in countries if "female" in c]
male_countries = [c for c in countries if "male" in c]

female_shap_age, female_shap_time = get_shap_values(female_countries)
male_shap_age, male_shap_time = get_shap_values(male_countries)

# ===============================
# 7. 可视化柱状图
# ===============================
fig, axes = plt.subplots(1,2,figsize=(14,6))
width = 0.35

# 年龄维度
age_labels = [f"A{i}" for i in range(-3,4)]
x = np.arange(len(age_labels))
axes[0].bar(x - width/2, female_shap_age, width, label="Female")
axes[0].bar(x + width/2, male_shap_age, width, label="Male")
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_labels)
axes[0].set_ylabel("Mean SHAP Importance")
axes[0].set_title("Age Importance")

# 时间维度
time_labels = [f"T{i}" for i in range(5)]
x = np.arange(len(time_labels))
axes[1].bar(x - width/2, female_shap_time, width, label="Female")
axes[1].bar(x + width/2, male_shap_time, width, label="Male")
axes[1].set_xticks(x)
axes[1].set_xticklabels(time_labels)
axes[1].set_ylabel("Mean SHAP Importance")
axes[1].set_title("Time Importance")

plt.suptitle("3D-CNN Kernel SHAP Analysis")
plt.tight_layout()
plt.show()
'''