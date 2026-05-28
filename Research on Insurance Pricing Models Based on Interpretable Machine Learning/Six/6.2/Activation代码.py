'''
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Embedding, Reshape, Concatenate
from sklearn.model_selection import train_test_split

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

# 数据清洗
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 训练 / 测试划分
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values

num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)

# 转换为 (年 × 年龄 × 国家)
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))


# ===============================
# 2. 构造训练数据
# ===============================
def create_dataset(data, time_steps=5, age_window=3):
    X, y, countries_idx = [], [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            for country in range(data.shape[2]):
                X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, country])
                y.append(data[year, age, country])
                countries_idx.append(country)
    return np.array(X), np.array(y), np.array(countries_idx)


X, y, countries_idx = create_dataset(mortality_train)
X = X[..., np.newaxis, np.newaxis]  # (样本, 5, 7, 1, 1)

X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries_idx, test_size=0.2, random_state=42
)


# ===============================
# 3. 构建 3D-CNN 模型
# ===============================
def build_model(num_countries):
    input_data = Input(shape=(5, 7, 1, 1))
    country_input = Input(shape=(1,))

    embedding = Embedding(input_dim=num_countries, output_dim=5, input_length=1)(country_input)
    embedding = Reshape((1, 1, 1, 5))(embedding)

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name="conv3d_1")(input_data)
    pool1 = MaxPooling3D((2, 2, 1))(conv1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="conv3d_2")(pool1)
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
          validation_data=([X_test, countries_test], y_test), verbose=1)

# ===============================
# 4. Activation Map
# ===============================
# 定义中间模型：提取 conv3d_1 或 conv3d_2 的特征图
activation_model = Model(inputs=model.inputs,
                         outputs=model.get_layer("conv3d_1").output)

# 选择 Norway_female 的一个测试样本
country_name = "Norway_male"
country_index = np.array([countries.index(country_name)])
sample_idx = np.where(countries_test == country_index[0])[0][0]
input_data = X_test[sample_idx:sample_idx + 1]

# 提取激活图
activations = activation_model.predict([input_data, country_index])
print("激活图形状:", activations.shape)  # (1, H, W, D, filters)

# ===============================
# 5. 可视化部分滤波器激活
# ===============================
num_filters = activations.shape[-1]
n_cols = 8
n_rows = min(num_filters // n_cols, 4)  # 最多展示 32 个滤波器

plt.figure(figsize=(15, 8))
for i in range(n_rows * n_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    # 取某个深度切片（例如第 0 个时间/年龄层）
    plt.imshow(activations[0, :, :, 0, i], cmap="viridis", aspect="auto")
    plt.axis("off")
    plt.title(f"F{i}")
plt.suptitle(f"Activation Maps ({country_name}, conv3d_1)", fontsize=16)
plt.show()


# ===============================
# 6. 【修正】综合分析激活图在时间和年龄维度上的模式（修正时间轴方向）
# ===============================

def analyze_spatiotemporal_preference(activations):
    """
    综合分析每个卷积核在时间和年龄维度上的激活偏好
    修正：T0是最近时间（前一年），T4是最早时间（前5年）
    activations: 形状为 (1, 5, 7, 1, 32) 的激活图
    """
    results = []

    # 移除批次维度，得到 (5, 7, 1, 32)
    activation_maps = activations[0]

    for filter_idx in range(activation_maps.shape[-1]):
        # 获取单个滤波器的激活图 (5, 7, 1)
        filter_activation = activation_maps[:, :, :, filter_idx]

        # ========== 年龄维度分析 ==========
        # 计算年龄维度的平均激活 (7个年龄点)
        age_profile = np.mean(filter_activation, axis=(0, 2))  # 形状: (7,)

        # 计算年龄维度的三个区域
        previous_ages = np.mean(age_profile[:3])  # 前三年 (年龄-3到-1)
        current_age = age_profile[3]  # 当前年龄
        next_ages = np.mean(age_profile[4:])  # 后三年 (年龄+1到+3)

        # 计算年龄偏好比率
        if next_ages > 0:
            age_preference_ratio = previous_ages / next_ages
        else:
            age_preference_ratio = previous_ages / (next_ages + 1e-8)

        # ========== 时间维度分析 ==========
        # 计算时间维度的平均激活 (5个时间点)
        time_profile = np.mean(filter_activation, axis=(1, 2))  # 形状: (5,)

        # 【修正】时间轴方向：T0是最近时间，T4是最早时间
        # 所以索引0对应T4（最早），索引4对应T0（最近）
        recent_times = np.mean(time_profile[:2])  # 近期时间 (T4, T3) - 实际上是较早的时间
        middle_times = np.mean(time_profile[2:3])  # 中期时间 (T2)
        early_times = np.mean(time_profile[3:])  # 早期时间 (T1, T0) - 实际上是较近的时间

        # 计算时间偏好比率 (近期 vs 早期)
        # 【修正】现在比较的是：较近时间(T1,T0) vs 较远时间(T4,T3)
        if recent_times > 0:
            time_preference_ratio = early_times / recent_times
        else:
            time_preference_ratio = early_times / (recent_times + 1e-8)

        # ========== 综合指标 ==========
        total_activation = np.mean(filter_activation)

        results.append({
            'filter': f'F{filter_idx}',
            # 年龄维度指标
            'age_previous': previous_ages,
            'age_current': current_age,
            'age_next': next_ages,
            'age_preference_ratio': age_preference_ratio,
            'prefers_previous_age': age_preference_ratio > 1.0,
            # 时间维度指标
            'time_early': early_times,  # 较近时间 (T1, T0)
            'time_middle': middle_times,  # 中期时间 (T2)
            'time_recent': recent_times,  # 较远时间 (T4, T3)
            'time_preference_ratio': time_preference_ratio,
            'prefers_early_time': time_preference_ratio > 1.0,  # 偏好较近时间
            # 综合指标
            'total_activation': total_activation,
            # 分类标签
            'age_time_pattern': classify_age_time_pattern(age_preference_ratio, time_preference_ratio)
        })

    return pd.DataFrame(results)


def classify_age_time_pattern(age_ratio, time_ratio):
    """
    根据年龄和时间偏好对滤波器进行分类
    【修正】时间偏好现在表示：较近时间 vs 较远时间
    """
    if age_ratio > 1.2 and time_ratio > 1.2:
        return "前年龄-近时间型"  # 偏好前三年年龄和较近时间
    elif age_ratio > 1.2 and time_ratio < 0.8:
        return "前年龄-远时间型"  # 偏好前三年年龄和较远时间
    elif age_ratio < 0.8 and time_ratio > 1.2:
        return "后年龄-近时间型"  # 偏好后三年年龄和较近时间
    elif age_ratio < 0.8 and time_ratio < 0.8:
        return "后年龄-远时间型"  # 偏好后三年年龄和较远时间
    elif age_ratio > 1.0 and time_ratio > 1.0:
        return "平衡偏前型"
    elif age_ratio < 1.0 and time_ratio < 1.0:
        return "平衡偏后型"
    else:
        return "混合型"


# 运行时空偏好分析
spatiotemporal_df = analyze_spatiotemporal_preference(activations)

# 按年龄偏好比率排序
age_preference_sorted = spatiotemporal_df.sort_values('age_preference_ratio', ascending=False)

# 按时间偏好比率排序
time_preference_sorted = spatiotemporal_df.sort_values('time_preference_ratio', ascending=False)

print("=" * 70)
print("时空维度激活偏好综合分析（修正时间轴方向）")
print("=" * 70)
print("时间轴说明：T0=最近时间（前1年），T4=最早时间（前5年）")
print("时间偏好比率 = 较近时间(T1,T0) / 较远时间(T4,T3)")

print(f"年龄维度统计:")
print(f"  - 偏好前三年(ratio>1)的滤波器数量: {sum(spatiotemporal_df['prefers_previous_age'])}")
print(f"  - 偏好后三年(ratio<1)的滤波器数量: {sum(~spatiotemporal_df['prefers_previous_age'])}")
print(f"  - 平均年龄偏好比率: {spatiotemporal_df['age_preference_ratio'].mean():.3f}")

print(f"\n时间维度统计:")
print(f"  - 偏好较近时间(ratio>1)的滤波器数量: {sum(spatiotemporal_df['prefers_early_time'])}")
print(f"  - 偏好较远时间(ratio<1)的滤波器数量: {sum(~spatiotemporal_df['prefers_early_time'])}")
print(f"  - 平均时间偏好比率: {spatiotemporal_df['time_preference_ratio'].mean():.3f}")

print(f"\n滤波器模式分类:")
pattern_counts = spatiotemporal_df['age_time_pattern'].value_counts()
for pattern, count in pattern_counts.items():
    print(f"  - {pattern}: {count}个滤波器")

print("\n前10个最偏好前三年年龄的滤波器:")
print(age_preference_sorted[['filter', 'age_preference_ratio', 'time_preference_ratio', 'age_time_pattern']].head(10))

print("\n前10个最偏好较近时间的滤波器:")
print(time_preference_sorted[['filter', 'time_preference_ratio', 'age_preference_ratio', 'age_time_pattern']].head(10))

# ===============================
# 7. 【修正】可视化时空偏好分析结果（修正时间轴标签）
# ===============================

# 创建综合可视化
plt.figure(figsize=(15, 5))

# 子图1: 年龄偏好 vs 时间偏好散点图
plt.subplot(1, 3, 1)
colors = []
for pattern in spatiotemporal_df['age_time_pattern']:
    if "前年龄" in pattern and "近时间" in pattern:
        colors.append('red')
    elif "前年龄" in pattern and "远时间" in pattern:
        colors.append('orange')
    elif "后年龄" in pattern and "近时间" in pattern:
        colors.append('blue')
    elif "后年龄" in pattern and "远时间" in pattern:
        colors.append('green')
    else:
        colors.append('gray')

scatter = plt.scatter(spatiotemporal_df['age_preference_ratio'],
                      spatiotemporal_df['time_preference_ratio'],
                      c=colors, alpha=0.7, s=60)
plt.axvline(1.0, color='black', linestyle='--', alpha=0.5)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('年龄偏好比率 (前三年/后三年)')
plt.ylabel('时间偏好比率 (较近时间/较远时间)\n(T1,T0)/(T4,T3)')
plt.title('滤波器时空偏好分布')
plt.grid(True, alpha=0.3)

# 添加图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='前年龄-近时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='前年龄-远时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='后年龄-近时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='后年龄-远时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='其他类型')
]
plt.legend(handles=legend_elements, loc='upper right')

# 子图2: 年龄偏好分布
plt.subplot(1, 3, 2)
plt.hist(spatiotemporal_df['age_preference_ratio'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='平衡点')
plt.axvline(spatiotemporal_df['age_preference_ratio'].mean(), color='blue', linestyle='-', label='平均值')
plt.xlabel('年龄偏好比率')
plt.ylabel('滤波器数量')
plt.title('年龄偏好分布')
plt.legend()

# 子图3: 时间偏好分布
plt.subplot(1, 3, 3)
plt.hist(spatiotemporal_df['time_preference_ratio'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='平衡点')
plt.axvline(spatiotemporal_df['time_preference_ratio'].mean(), color='blue', linestyle='-', label='平均值')
plt.xlabel('时间偏好比率\n(较近时间/较远时间)')
plt.ylabel('滤波器数量')
plt.title('时间偏好分布')
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 8. 【修正】可视化代表性滤波器的详细时空模式（修正时间轴标签）
# ===============================

# 选择各类代表性滤波器
representative_filters = []

# 添加各类典型模式的滤波器
for pattern in ["前年龄-近时间型", "前年龄-远时间型", "后年龄-近时间型", "后年龄-远时间型", "平衡偏前型"]:
    pattern_filters = spatiotemporal_df[spatiotemporal_df['age_time_pattern'] == pattern]
    if len(pattern_filters) > 0:
        # 选择该类中总激活最强的滤波器
        top_filter = pattern_filters.nlargest(1, 'total_activation')
        representative_filters.append(top_filter.iloc[0]['filter'])

# 如果代表性滤波器不足6个，补充其他高激活滤波器
if len(representative_filters) < 6:
    additional_filters = spatiotemporal_df.nlargest(6 - len(representative_filters), 'total_activation')
    for _, row in additional_filters.iterrows():
        if row['filter'] not in representative_filters:
            representative_filters.append(row['filter'])

# 创建详细时空模式可视化 - 减小图形尺寸但保持适当间距
fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # 减小图形尺寸
axes = axes.ravel()

for idx, filter_name in enumerate(representative_filters[:6]):  # 最多显示6个
    filter_idx = int(filter_name[1:])  # 从"F0"中提取数字0

    # 获取该滤波器的激活图
    filter_activation = activations[0, :, :, 0, filter_idx]

    # 绘制热力图
    im = axes[idx].imshow(filter_activation, cmap='viridis', aspect='auto')

    # 获取该滤波器的分类信息
    filter_info = spatiotemporal_df[spatiotemporal_df['filter'] == filter_name].iloc[0]

    axes[idx].set_title(f'{filter_name} - {filter_info["age_time_pattern"]}\n'
                        f'年龄比: {filter_info["age_preference_ratio"]:.2f}, '
                        f'时间比: {filter_info["time_preference_ratio"]:.2f}',
                        fontsize=12, pad=15)  # 减小字体和间距

    axes[idx].set_xlabel('年龄偏移 (年)', fontsize=10)
    axes[idx].set_ylabel('时间 (年)', fontsize=10)

    # 【修正】设置坐标轴刻度 - 时间轴方向：T4最早，T0最近
    axes[idx].set_xticks(range(7))
    axes[idx].set_xticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'], fontsize=9)
    axes[idx].set_yticks(range(5))
    axes[idx].set_yticklabels(['T4', 'T3', 'T2', 'T1', 'T0'], fontsize=9)  # T4最早，T0最近

    # 添加颜色条 - 调整颜色条位置和大小
    cbar = plt.colorbar(im, ax=axes[idx], shrink=0.7, aspect=15)
    cbar.ax.tick_params(labelsize=8)

# 隐藏多余的子图
for idx in range(len(representative_filters), 6):
    axes[idx].set_visible(False)

# 保持适当的子图间距
plt.subplots_adjust(wspace=0.4, hspace=0.5)  # 适当减小间距

plt.suptitle('代表性滤波器的时空激活模式分析（T4:最早, T0:最近）', fontsize=14, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为总标题留出空间
plt.show()

# ===============================
# 9. 【修正】输出综合分析结论（修正时间轴解释）
# ===============================

print("\n" + "=" * 70)
print("综合分析结论（修正时间轴方向）")
print("=" * 70)
print("时间轴说明：T0=最近时间（前1年），T4=最早时间（前5年）")

mean_age_ratio = spatiotemporal_df['age_preference_ratio'].mean()
mean_time_ratio = spatiotemporal_df['time_preference_ratio'].mean()

# 年龄维度结论
if mean_age_ratio > 1.2:
    age_conclusion = "模型明显更关注年龄维度的前三年数据"
elif mean_age_ratio > 1.0:
    age_conclusion = "模型略微更关注年龄维度的前三年数据"
elif mean_age_ratio < 0.8:
    age_conclusion = "模型明显更关注年龄维度的后三年数据"
elif mean_age_ratio < 1.0:
    age_conclusion = "模型略微更关注年龄维度的后三年数据"
else:
    age_conclusion = "模型对前后三年年龄数据的关注度相对平衡"

# 时间维度结论
if mean_time_ratio > 1.2:
    time_conclusion = "模型明显更关注较近时间数据（T1,T0）"
elif mean_time_ratio > 1.0:
    time_conclusion = "模型略微更关注较近时间数据（T1,T0）"
elif mean_time_ratio < 0.8:
    time_conclusion = "模型明显更关注较远时间数据（T4,T3）"
elif mean_time_ratio < 1.0:
    time_conclusion = "模型略微更关注较远时间数据（T4,T3）"
else:
    time_conclusion = "模型对较近和较远时间数据的关注度相对平衡"

print(f"年龄维度: {age_conclusion}")
print(f"  平均年龄偏好比率: {mean_age_ratio:.3f}")
print(
    f"  偏好前三年年龄的滤波器比例: {sum(spatiotemporal_df['prefers_previous_age']) / len(spatiotemporal_df) * 100:.1f}%")

print(f"\n时间维度: {time_conclusion}")
print(f"  平均时间偏好比率: {mean_time_ratio:.3f}")
print(
    f"  偏好较近时间的滤波器比例: {sum(spatiotemporal_df['prefers_early_time']) / len(spatiotemporal_df) * 100:.1f}%")

print(f"\n主要模式分布:")
for pattern, count in pattern_counts.items():
    percentage = count / len(spatiotemporal_df) * 100
    print(f"  - {pattern}: {count}个滤波器 ({percentage:.1f}%)")

# 综合学术解释
print(f"\n学术解释:")
print("基于3D-CNN激活图的时空模式分析表明：")
print(
    f"1. 在年龄维度上，模型{'更依赖于前三年年龄的历史模式' if mean_age_ratio > 1.0 else '更依赖于后三年年龄的模式'}来预测当前年龄的死亡率")
print(
    f"2. 在时间维度上，模型{'更关注较近的时间模式（T1,T0）' if mean_time_ratio > 1.0 else '更关注较远的时间模式（T4,T3）'}")
print(
    f"3. 最主要的滤波器模式是'{pattern_counts.index[0]}'，占{pattern_counts.iloc[0] / len(spatiotemporal_df) * 100:.1f}%")

if mean_age_ratio > 1.0 and mean_time_ratio > 1.0:
    print("4. 综合来看，模型表现出'前年龄-近时间'的偏好模式，这符合死亡率预测中的近期效应和累积风险理论")
elif mean_age_ratio > 1.0 and mean_time_ratio < 1.0:
    print("4. 综合来看，模型表现出'前年龄-远时间'的偏好模式，表明长期历史模式和年龄累积效应共同作用")
elif mean_age_ratio < 1.0 and mean_time_ratio > 1.0:
    print("4. 综合来看，模型表现出'后年龄-近时间'的偏好模式，可能反映了对未来风险的预期")
else:
    print("4. 综合来看，模型表现出复杂的时空依赖模式")
'''

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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

# 数据清洗
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 训练 / 测试划分
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

years = train_data['year'].values
ages = train_data['age'].values
mortality_rates = train_data[countries].values

num_years = len(np.unique(years))
num_ages = len(np.unique(ages))
num_countries = len(countries)

# 转换为 (年 × 年龄 × 国家)
mortality_train = mortality_rates.reshape((num_years, num_ages, num_countries))


# ===============================
# 2. 构造训练数据
# ===============================
def create_dataset(data, time_steps=5, age_window=3):
    X, y, countries_idx = [], [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            for country in range(data.shape[2]):
                X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, country])
                y.append(data[year, age, country])
                countries_idx.append(country)
    return np.array(X), np.array(y), np.array(countries_idx)


X, y, countries_idx = create_dataset(mortality_train)
X = X[..., np.newaxis]  # (样本, 5, 7, 1) - 修改：去掉一个维度，适应2D-CNN

X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries_idx, test_size=0.2, random_state=42
)


# ===============================
# 3. 构建 2D-CNN 模型
# ===============================
def cnn2d_model(params):
    """构建2D卷积神经网络模型"""
    model = Sequential([
        Input(shape=(5, 7, 1)),  # 输入形状：5时间步 × 7年龄窗口 × 1通道
        Conv2D(int(params['filters1']), (3, 3), activation='relu', padding='same', name="conv2d_1"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(int(params['filters2']), (3, 3), activation='relu', padding='same', name="conv2d_2"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(int(params['dense_units']), activation='relu'),
        Dense(1)  # 输出层：预测死亡率残差
    ])

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae']  # 监控平均绝对误差
    )
    return model


# 模型参数（使用挪威男性参数）
#params = {'filters1': 112, 'filters2': 64, 'dense_units': 200, 'learning_rate': 0.00019}#女
params = {'filters1': 96, 'filters2': 224, 'dense_units': 250, 'learning_rate': 0.00010}#男#
model = cnn2d_model(params)
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

# ===============================
# 4. Activation Map
# ===============================
# 定义中间模型：提取 conv2d_1 的特征图
activation_model = Model(inputs=model.inputs,
                         outputs=model.get_layer("conv2d_1").output)

# 选择 Norway_male 的一个测试样本
country_name = "Norway_male"
country_index = np.array([countries.index(country_name)])
sample_idx = np.where(countries_test == country_index[0])[0][0]
input_data = X_test[sample_idx:sample_idx + 1]

# 提取激活图
activations = activation_model.predict(input_data)  # 修改：去掉国家输入
print("激活图形状:", activations.shape)  # (1, H, W, filters) - 2D激活图

# ===============================
# 5. 可视化部分滤波器激活
# ===============================
num_filters = activations.shape[-1]
n_cols = 8
n_rows = min(num_filters // n_cols, 4)  # 最多展示 32 个滤波器

plt.figure(figsize=(15, 8))
for i in range(n_rows * n_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    # 取激活图的第一个样本
    plt.imshow(activations[0, :, :, i], cmap="viridis", aspect="auto")
    plt.axis("off")
    plt.title(f"F{i}")
plt.suptitle(f"Activation Maps ({country_name}, conv2d_1)", fontsize=16)
plt.show()


# ===============================
# 6. 【修正】综合分析激活图在时间和年龄维度上的模式（修正时间轴方向）
# ===============================

def analyze_spatiotemporal_preference(activations):
    """
    综合分析每个卷积核在时间和年龄维度上的激活偏好
    修正：T0是最近时间（前一年），T4是最早时间（前5年）
    activations: 形状为 (1, H, W, filters) 的2D激活图
    """
    results = []

    # 移除批次维度，得到 (H, W, filters)
    activation_maps = activations[0]

    for filter_idx in range(activation_maps.shape[-1]):
        # 获取单个滤波器的激活图 (H, W)
        filter_activation = activation_maps[:, :, filter_idx]

        # ========== 年龄维度分析 ==========
        # 计算年龄维度的平均激活 (7个年龄点)
        age_profile = np.mean(filter_activation, axis=0)  # 形状: (W,) - 修改：沿时间维度平均

        # 计算年龄维度的三个区域
        previous_ages = np.mean(age_profile[:3])  # 前三年 (年龄-3到-1)
        current_age = age_profile[3]  # 当前年龄
        next_ages = np.mean(age_profile[4:])  # 后三年 (年龄+1到+3)

        # 计算年龄偏好比率
        if next_ages > 0:
            age_preference_ratio = previous_ages / next_ages
        else:
            age_preference_ratio = previous_ages / (next_ages + 1e-8)

        # ========== 时间维度分析 ==========
        # 计算时间维度的平均激活 (5个时间点)
        time_profile = np.mean(filter_activation, axis=1)  # 形状: (H,) - 修改：沿年龄维度平均

        # 【修正】时间轴方向：T0是最近时间，T4是最早时间
        # 所以索引0对应T4（最早），索引4对应T0（最近）
        recent_times = np.mean(time_profile[:2])  # 较远时间 (T4, T3)
        middle_times = np.mean(time_profile[2:3])  # 中期时间 (T2)
        early_times = np.mean(time_profile[3:])  # 较近时间 (T1, T0)

        # 计算时间偏好比率 (较近时间 vs 较远时间)
        if recent_times > 0:
            time_preference_ratio = early_times / recent_times
        else:
            time_preference_ratio = early_times / (recent_times + 1e-8)

        # ========== 综合指标 ==========
        total_activation = np.mean(filter_activation)

        results.append({
            'filter': f'F{filter_idx}',
            # 年龄维度指标
            'age_previous': previous_ages,
            'age_current': current_age,
            'age_next': next_ages,
            'age_preference_ratio': age_preference_ratio,
            'prefers_previous_age': age_preference_ratio > 1.0,
            # 时间维度指标
            'time_early': early_times,  # 较近时间 (T1, T0)
            'time_middle': middle_times,  # 中期时间 (T2)
            'time_recent': recent_times,  # 较远时间 (T4, T3)
            'time_preference_ratio': time_preference_ratio,
            'prefers_early_time': time_preference_ratio > 1.0,  # 偏好较近时间
            # 综合指标
            'total_activation': total_activation,
            # 分类标签
            'age_time_pattern': classify_age_time_pattern(age_preference_ratio, time_preference_ratio)
        })

    return pd.DataFrame(results)


def classify_age_time_pattern(age_ratio, time_ratio):
    """
    根据年龄和时间偏好对滤波器进行分类
    【修正】时间偏好现在表示：较近时间 vs 较远时间
    """
    if age_ratio > 1.2 and time_ratio > 1.2:
        return "前年龄-近时间型"  # 偏好前三年年龄和较近时间
    elif age_ratio > 1.2 and time_ratio < 0.8:
        return "前年龄-远时间型"  # 偏好前三年年龄和较远时间
    elif age_ratio < 0.8 and time_ratio > 1.2:
        return "后年龄-近时间型"  # 偏好后三年年龄和较近时间
    elif age_ratio < 0.8 and time_ratio < 0.8:
        return "后年龄-远时间型"  # 偏好后三年年龄和较远时间
    elif age_ratio > 1.0 and time_ratio > 1.0:
        return "平衡偏前型"
    elif age_ratio < 1.0 and time_ratio < 1.0:
        return "平衡偏后型"
    else:
        return "混合型"


# 运行时空偏好分析
spatiotemporal_df = analyze_spatiotemporal_preference(activations)

# 按年龄偏好比率排序
age_preference_sorted = spatiotemporal_df.sort_values('age_preference_ratio', ascending=False)

# 按时间偏好比率排序
time_preference_sorted = spatiotemporal_df.sort_values('time_preference_ratio', ascending=False)

print("=" * 70)
print("时空维度激活偏好综合分析（2D-CNN模型）")
print("=" * 70)
print("时间轴说明：T0=最近时间（前1年），T4=最早时间（前5年）")
print("时间偏好比率 = 较近时间(T1,T0) / 较远时间(T4,T3)")

print(f"年龄维度统计:")
print(f"  - 偏好前三年(ratio>1)的滤波器数量: {sum(spatiotemporal_df['prefers_previous_age'])}")
print(f"  - 偏好后三年(ratio<1)的滤波器数量: {sum(~spatiotemporal_df['prefers_previous_age'])}")
print(f"  - 平均年龄偏好比率: {spatiotemporal_df['age_preference_ratio'].mean():.3f}")

print(f"\n时间维度统计:")
print(f"  - 偏好较近时间(ratio>1)的滤波器数量: {sum(spatiotemporal_df['prefers_early_time'])}")
print(f"  - 偏好较远时间(ratio<1)的滤波器数量: {sum(~spatiotemporal_df['prefers_early_time'])}")
print(f"  - 平均时间偏好比率: {spatiotemporal_df['time_preference_ratio'].mean():.3f}")

print(f"\n滤波器模式分类:")
pattern_counts = spatiotemporal_df['age_time_pattern'].value_counts()
for pattern, count in pattern_counts.items():
    print(f"  - {pattern}: {count}个滤波器")

print("\n前10个最偏好前三年年龄的滤波器:")
print(age_preference_sorted[['filter', 'age_preference_ratio', 'time_preference_ratio', 'age_time_pattern']].head(10))

print("\n前10个最偏好较近时间的滤波器:")
print(time_preference_sorted[['filter', 'time_preference_ratio', 'age_preference_ratio', 'age_time_pattern']].head(10))

# ===============================
# 7. 【修正】可视化时空偏好分析结果（修正时间轴标签）
# ===============================

# 创建综合可视化
plt.figure(figsize=(15, 5))

# 子图1: 年龄偏好 vs 时间偏好散点图
plt.subplot(1, 3, 1)
colors = []
for pattern in spatiotemporal_df['age_time_pattern']:
    if "前年龄" in pattern and "近时间" in pattern:
        colors.append('red')
    elif "前年龄" in pattern and "远时间" in pattern:
        colors.append('orange')
    elif "后年龄" in pattern and "近时间" in pattern:
        colors.append('blue')
    elif "后年龄" in pattern and "远时间" in pattern:
        colors.append('green')
    else:
        colors.append('gray')

scatter = plt.scatter(spatiotemporal_df['age_preference_ratio'],
                      spatiotemporal_df['time_preference_ratio'],
                      c=colors, alpha=0.7, s=60)
plt.axvline(1.0, color='black', linestyle='--', alpha=0.5)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('年龄偏好比率 (前三年/后三年)')
plt.ylabel('时间偏好比率 (较近时间/较远时间)\n(T1,T0)/(T4,T3)')
plt.title('滤波器时空偏好分布')
plt.grid(True, alpha=0.3)

# 添加图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='前年龄-近时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='前年龄-远时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='后年龄-近时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='后年龄-远时间型'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='其他类型')
]
plt.legend(handles=legend_elements, loc='upper right')

# 子图2: 年龄偏好分布
plt.subplot(1, 3, 2)
plt.hist(spatiotemporal_df['age_preference_ratio'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='平衡点')
plt.axvline(spatiotemporal_df['age_preference_ratio'].mean(), color='blue', linestyle='-', label='平均值')
plt.xlabel('年龄偏好比率')
plt.ylabel('滤波器数量')
plt.title('年龄偏好分布')
plt.legend()

# 子图3: 时间偏好分布
plt.subplot(1, 3, 3)
plt.hist(spatiotemporal_df['time_preference_ratio'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='平衡点')
plt.axvline(spatiotemporal_df['time_preference_ratio'].mean(), color='blue', linestyle='-', label='平均值')
plt.xlabel('时间偏好比率\n(较近时间/较远时间)')
plt.ylabel('滤波器数量')
plt.title('时间偏好分布')
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 8. 【修正】可视化代表性滤波器的详细时空模式（修正时间轴标签）
# ===============================

# 选择各类代表性滤波器
representative_filters = []

# 添加各类典型模式的滤波器
for pattern in ["前年龄-近时间型", "前年龄-远时间型", "后年龄-近时间型", "后年龄-远时间型", "平衡偏前型"]:
    pattern_filters = spatiotemporal_df[spatiotemporal_df['age_time_pattern'] == pattern]
    if len(pattern_filters) > 0:
        # 选择该类中总激活最强的滤波器
        top_filter = pattern_filters.nlargest(1, 'total_activation')
        representative_filters.append(top_filter.iloc[0]['filter'])

# 如果代表性滤波器不足6个，补充其他高激活滤波器
if len(representative_filters) < 6:
    additional_filters = spatiotemporal_df.nlargest(6 - len(representative_filters), 'total_activation')
    for _, row in additional_filters.iterrows():
        if row['filter'] not in representative_filters:
            representative_filters.append(row['filter'])

# 创建详细时空模式可视化 - 减小图形尺寸但保持适当间距
fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # 减小图形尺寸
axes = axes.ravel()

for idx, filter_name in enumerate(representative_filters[:6]):  # 最多显示6个
    filter_idx = int(filter_name[1:])  # 从"F0"中提取数字0

    # 获取该滤波器的激活图
    filter_activation = activations[0, :, :, filter_idx]

    # 绘制热力图
    im = axes[idx].imshow(filter_activation, cmap='viridis', aspect='auto')

    # 获取该滤波器的分类信息
    filter_info = spatiotemporal_df[spatiotemporal_df['filter'] == filter_name].iloc[0]

    axes[idx].set_title(f'{filter_name} - {filter_info["age_time_pattern"]}\n'
                        f'年龄比: {filter_info["age_preference_ratio"]:.2f}, '
                        f'时间比: {filter_info["time_preference_ratio"]:.2f}',
                        fontsize=12, pad=15)  # 减小字体和间距

    axes[idx].set_xlabel('年龄偏移 (年)', fontsize=10)
    axes[idx].set_ylabel('时间 (年)', fontsize=10)

    # 【修正】设置坐标轴刻度 - 时间轴方向：T4最早，T0最近
    axes[idx].set_xticks(range(7))
    axes[idx].set_xticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'], fontsize=9)
    axes[idx].set_yticks(range(5))
    axes[idx].set_yticklabels(['T4', 'T3', 'T2', 'T1', 'T0'], fontsize=9)  # T4最早，T0最近

    # 添加颜色条 - 调整颜色条位置和大小
    cbar = plt.colorbar(im, ax=axes[idx], shrink=0.7, aspect=15)
    cbar.ax.tick_params(labelsize=8)

# 隐藏多余的子图
for idx in range(len(representative_filters), 6):
    axes[idx].set_visible(False)

# 保持适当的子图间距
plt.subplots_adjust(wspace=0.4, hspace=0.5)  # 适当减小间距

plt.suptitle('代表性滤波器的时空激活模式分析（2D-CNN，T4:最早, T0:最近）', fontsize=14, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为总标题留出空间
plt.show()

# ===============================
# 9. 【修正】输出综合分析结论（修正时间轴解释）
# ===============================

print("\n" + "=" * 70)
print("综合分析结论（2D-CNN模型）")
print("=" * 70)
print("时间轴说明：T0=最近时间（前1年），T4=最早时间（前5年）")

mean_age_ratio = spatiotemporal_df['age_preference_ratio'].mean()
mean_time_ratio = spatiotemporal_df['time_preference_ratio'].mean()

# 年龄维度结论
if mean_age_ratio > 1.2:
    age_conclusion = "模型明显更关注年龄维度的前三年数据"
elif mean_age_ratio > 1.0:
    age_conclusion = "模型略微更关注年龄维度的前三年数据"
elif mean_age_ratio < 0.8:
    age_conclusion = "模型明显更关注年龄维度的后三年数据"
elif mean_age_ratio < 1.0:
    age_conclusion = "模型略微更关注年龄维度的后三年数据"
else:
    age_conclusion = "模型对前后三年年龄数据的关注度相对平衡"

# 时间维度结论
if mean_time_ratio > 1.2:
    time_conclusion = "模型明显更关注较近时间数据（T1,T0）"
elif mean_time_ratio > 1.0:
    time_conclusion = "模型略微更关注较近时间数据（T1,T0）"
elif mean_time_ratio < 0.8:
    time_conclusion = "模型明显更关注较远时间数据（T4,T3）"
elif mean_time_ratio < 1.0:
    time_conclusion = "模型略微更关注较远时间数据（T4,T3）"
else:
    time_conclusion = "模型对较近和较远时间数据的关注度相对平衡"

print(f"年龄维度: {age_conclusion}")
print(f"  平均年龄偏好比率: {mean_age_ratio:.3f}")
print(
    f"  偏好前三年年龄的滤波器比例: {sum(spatiotemporal_df['prefers_previous_age']) / len(spatiotemporal_df) * 100:.1f}%")

print(f"\n时间维度: {time_conclusion}")
print(f"  平均时间偏好比率: {mean_time_ratio:.3f}")
print(
    f"  偏好较近时间的滤波器比例: {sum(spatiotemporal_df['prefers_early_time']) / len(spatiotemporal_df) * 100:.1f}%")

print(f"\n主要模式分布:")
for pattern, count in pattern_counts.items():
    percentage = count / len(spatiotemporal_df) * 100
    print(f"  - {pattern}: {count}个滤波器 ({percentage:.1f}%)")

# 综合学术解释
print(f"\n学术解释:")
print("基于2D-CNN激活图的时空模式分析表明：")
print(
    f"1. 在年龄维度上，模型{'更依赖于前三年年龄的历史模式' if mean_age_ratio > 1.0 else '更依赖于后三年年龄的模式'}来预测当前年龄的死亡率")
print(
    f"2. 在时间维度上，模型{'更关注较近的时间模式（T1,T0）' if mean_time_ratio > 1.0 else '更关注较远的时间模式（T4,T3）'}")
print(
    f"3. 最主要的滤波器模式是'{pattern_counts.index[0]}'，占{pattern_counts.iloc[0] / len(spatiotemporal_df) * 100:.1f}%")

if mean_age_ratio > 1.0 and mean_time_ratio > 1.0:
    print("4. 综合来看，模型表现出'前年龄-近时间'的偏好模式，这符合死亡率预测中的近期效应和累积风险理论")
elif mean_age_ratio > 1.0 and mean_time_ratio < 1.0:
    print("4. 综合来看，模型表现出'前年龄-远时间'的偏好模式，表明长期历史模式和年龄累积效应共同作用")
elif mean_age_ratio < 1.0 and mean_time_ratio > 1.0:
    print("4. 综合来看，模型表现出'后年龄-近时间'的偏好模式，可能反映了对未来风险的预期")
else:
    print("4. 综合来看，模型表现出复杂的时空依赖模式")