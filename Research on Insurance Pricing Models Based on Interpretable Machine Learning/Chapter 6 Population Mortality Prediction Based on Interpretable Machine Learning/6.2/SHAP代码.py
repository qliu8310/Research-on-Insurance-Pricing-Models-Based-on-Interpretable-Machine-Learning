'''
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

# ===============================
# 1. 数据准备（多国家）
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

# 数据清洗
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()


# ===============================
# 2. 残差计算（以简化Lee-Carter近似为例）
# ===============================
def compute_residual(country_series):
    df = pd.DataFrame({
        "year": data["year"],
        "age": data["age"],
        "mx": np.log(country_series.values)
    })
    # pivot: 年龄×年份
    mat = df.pivot(index="age", columns="year", values="mx").fillna(0).values
    ax = np.mean(mat, axis=1)
    kt = np.mean(mat, axis=0)
    bx = np.ones_like(ax) * 0.1
    fitting = np.outer(bx, kt) + ax[:, np.newaxis]
    residual = mat - fitting
    scaler = MinMaxScaler()
    return scaler.fit_transform(residual.reshape(-1, 1)).reshape(mat.shape)


# ===============================
# 3. 数据准备函数 (窗口构造)
# ===============================
def prepare_data(residual, age_window=3, time_steps=5):
    residual_padded = np.pad(residual, ((age_window, age_window), (0, 0)),
                             mode="constant", constant_values=0)
    X, y = [], []
    for year in range(residual_padded.shape[1] - time_steps):
        for age in range(age_window, residual_padded.shape[0] - age_window):
            neigh = residual_padded[age - age_window:age + age_window + 1,
                    year:year + time_steps]
            X.append(neigh)
            y.append(residual_padded[age, year + time_steps])
    X = np.array(X).transpose(0, 2, 1).reshape(-1, time_steps, 2 * age_window + 1, 1)
    return X, np.array(y)


# ===============================
# 4. 定义CNN模型
# ===============================
def cnn2d_model():
    model = Sequential([
        Input(shape=(5, 7, 1)),
        Conv2D(112, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(200, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1.9e-4), loss="mse")
    return model


# ===============================
# 5. 针对每个国家跑 SHAP
# ===============================
all_shap_values = []
all_labels = []

for c in countries:
    print(f"Processing {c}...")
    residual = compute_residual(data[c])
    X, y = prepare_data(residual)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = cnn2d_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # SHAP
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    X_explain = X_test[:30]
    shap_values = explainer.shap_values(X_explain)
    sv = np.squeeze(shap_values[0] if isinstance(shap_values, list) else shap_values)

    # 保存 (平均到时间/年龄维度)
    age_importance = np.mean(np.abs(sv), axis=(0, 1))  # 聚合到年龄
    all_shap_values.append(age_importance)
    all_labels.append(c)

all_shap_values = np.array(all_shap_values)  # (国家数, 年龄窗口大小=7)

# ===============================
# 6. SHAP Summary Plot
# ===============================
shap.summary_plot(
    all_shap_values,  # SHAP值
    features=all_shap_values,  # 传同样的形状，避免 shape 错误
    feature_names=[f"Age {i}" for i in range(-3, 4)],
    plot_type="dot",
    show=True
)




#柱形图
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

# ===============================
# 1. 数据准备
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
    return scaler.fit_transform(residual.reshape(-1, 1)).reshape(mat.shape)


# ===============================
# 3. 数据准备函数
# ===============================
def prepare_data(residual, age_window=3, time_steps=5):
    residual_padded = np.pad(residual, ((age_window, age_window), (0, 0)),
                             mode="constant", constant_values=0)
    X, y = [], []
    for year in range(residual_padded.shape[1] - time_steps):
        for age in range(age_window, residual_padded.shape[0] - age_window):
            neigh = residual_padded[age - age_window:age + age_window + 1,
                    year:year + time_steps]
            X.append(neigh)
            y.append(residual_padded[age, year + time_steps])
    X = np.array(X).transpose(0, 2, 1).reshape(-1, time_steps, 2 * age_window + 1, 1)
    return X, np.array(y)


# ===============================
# 4. CNN模型
# ===============================
def cnn2d_model():
    model = Sequential([
        Input(shape=(5, 7, 1)),
        Conv2D(112, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(200, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1.9e-4), loss="mse")
    return model


# ===============================
# 5. 分别处理女性组和男性组
# ===============================
female_countries = [c for c in countries if "female" in c]
male_countries = [c for c in countries if "male" in c]

def get_group_shap(countries_list, age_window=3, time_steps=5):
    age_importance_group = []
    time_importance_group = []

    for c in countries_list:
        print(f"Processing {c}...")
        residual = compute_residual(data[c])
        X, y = prepare_data(residual, age_window=age_window, time_steps=time_steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = cnn2d_model()
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # SHAP
        background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        X_explain = X_test[:30]
        shap_values = explainer.shap_values(X_explain)
        sv = np.squeeze(shap_values[0] if isinstance(shap_values, list) else shap_values)

        # 📌 年龄维度聚合
        age_importance = np.mean(np.abs(sv), axis=(0, 1))   # (7,)
        age_importance_group.append(age_importance)

        # 📌 时间维度聚合
        time_importance = np.mean(np.abs(sv), axis=(0, 2))  # (5,)
        time_importance_group.append(time_importance)

    # 返回两个平均结果
    return np.mean(age_importance_group, axis=0), np.mean(time_importance_group, axis=0)

# 1️⃣ 计算男女平均 SHAP
# 计算 SHAP
# ===============================
female_shap_age, female_shap_time = get_group_shap(female_countries)
male_shap_age, male_shap_time = get_group_shap(male_countries)

# 2️⃣ 绘制柱形图对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
width = 0.35
# 左图：年龄维度
# 年龄维度
age_window = 3
age_labels = [f"Age {i}" for i in range(-age_window, age_window + 1)]
x = np.arange(len(age_labels))
axes[0].bar(x - width/2, female_shap_age, width, label="Female")
axes[0].bar(x + width/2, male_shap_age, width, label="Male")
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_labels, rotation=45)
axes[0].set_ylabel("Mean SHAP Importance")
axes[0].set_title("Age Importance (Female vs Male)")
axes[0].legend()

# 时间维度
time_labels = [f"T{i}" for i in range(5)]
x = np.arange(len(time_labels))
axes[1].bar(x - width/2, female_shap_time, width, label="Female")
axes[1].bar(x + width/2, male_shap_time, width, label="Male")
axes[1].set_xticks(x)
axes[1].set_xticklabels(time_labels)
axes[1].set_ylabel("Mean SHAP Importance")
axes[1].set_title("Time Importance (Female vs Male)")
axes[1].legend()

plt.suptitle("SHAP Analysis: Age vs Time Dependence", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.suptitle("SHAP Analysis: Age vs Time Dependence", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
'''


'''
#   2D-蜜蜂图
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ===============================
# 1. 数据准备
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
    return scaler.fit_transform(residual.reshape(-1, 1)).reshape(mat.shape)


# ===============================
# 3. 数据准备函数
# ===============================
def prepare_data(residual, age_window=3, time_steps=5):
    residual_padded = np.pad(residual, ((age_window, age_window), (0, 0)),
                             mode="constant", constant_values=0)
    X, y = [], []
    for year in range(residual_padded.shape[1] - time_steps):
        for age in range(age_window, residual_padded.shape[0] - age_window):
            neigh = residual_padded[age - age_window:age + age_window + 1,
                    year:year + time_steps]
            X.append(neigh)
            y.append(residual_padded[age, year + time_steps])
    X = np.array(X).transpose(0, 2, 1).reshape(-1, time_steps, 2 * age_window + 1, 1)
    return X, np.array(y)


# ===============================
# 4. CNN模型
# ===============================

def cnn2d_model():
    model = Sequential([
        Input(shape=(5, 7, 1)),
        Conv2D(112, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(200, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1.9e-4), loss="mse")
    return model


# ===============================
# 5. 分别处理女性组和男性组
# ===============================
female_countries = [c for c in countries if "female" in c]
male_countries = [c for c in countries if "male" in c]


def get_group_shap(countries_list, age_window=3, time_steps=5):
    all_shap_values = []
    all_feature_values = []

    for c in countries_list:
        print(f"Processing {c}...")
        residual = compute_residual(data[c])
        X, y = prepare_data(residual, age_window=age_window, time_steps=time_steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = cnn2d_model()
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # SHAP
        background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        X_explain = X_test[:30]
        shap_values = explainer.shap_values(X_explain)

        # 获取SHAP值和特征值
        sv = np.squeeze(shap_values[0] if isinstance(shap_values, list) else shap_values)
        feature_vals = np.squeeze(X_explain)

        # 重塑为二维数组 (样本数, 特征数)
        sv_reshaped = sv.reshape(sv.shape[0], -1)
        feature_vals_reshaped = feature_vals.reshape(feature_vals.shape[0], -1)

        all_shap_values.append(sv_reshaped)
        all_feature_values.append(feature_vals_reshaped)

    # 合并所有国家的SHAP值和特征值
    all_shap = np.concatenate(all_shap_values, axis=0)
    all_features = np.concatenate(all_feature_values, axis=0)

    return all_shap, all_features


# 计算女性组和男性组的SHAP值
female_shap, female_features = get_group_shap(female_countries)
male_shap, male_features = get_group_shap(male_countries)


# ===============================
# 6. 创建按最大SHAP值排序的前8个特征的beeswarm图
# ===============================
def create_top8_shap_beeswarm(shap_values, feature_values, feature_names, title):
    """
    创建前8个最重要特征的SHAP beeswarm图，按最大SHAP值排序
    """
    # 计算每个特征的最大绝对SHAP值
    max_abs_shap = np.max(np.abs(shap_values), axis=0)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(max_abs_shap)[-8:][::-1]  # 取最大的8个，并反转顺序（最重要的在前）

    # 提取前8个特征的SHAP值和特征值
    top8_shap = shap_values[:, top8_indices]
    top8_features = feature_values[:, top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 创建Explanation对象
    explanation = shap.Explanation(
        values=top8_shap,
        base_values=np.zeros(top8_shap.shape[0]),
        data=top8_features,
        feature_names=top8_names
    )

    # 绘制beeswarm图
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 打印前8个特征的详细信息
    print(f"\n{title} - 前8个最重要特征:")
    for i, idx in enumerate(top8_indices):
        max_shap = np.max(np.abs(shap_values[:, idx]))
        mean_shap = np.mean(np.abs(shap_values[:, idx]))
        print(f"{i + 1}. {feature_names[idx]}: 最大|SHAP|={max_shap:.4f}, 平均|SHAP|={mean_shap:.4f}")

    return top8_indices, top8_names


# 创建特征名称
age_window = 3
time_steps = 5
feature_names = [f"T{t}_A{a}" for t in range(time_steps) for a in range(-age_window, age_window + 1)]

# 检查SHAP值的维度并相应调整
if female_shap.shape[1] != len(feature_names):
    feature_names = [f"Feature_{i}" for i in range(female_shap.shape[1])]

# 绘制女性组和男性组的前8个最重要特征的SHAP beeswarm图
female_top8_indices, female_top8_names = create_top8_shap_beeswarm(
    female_shap, female_features, feature_names, "女性组 - 前8个最重要特征的SHAP Beeswarm图"
)

male_top8_indices, male_top8_names = create_top8_shap_beeswarm(
    male_shap, male_features, feature_names, "男性组 - 前8个最重要特征的SHAP Beeswarm图"
)


# ===============================
# 7. 创建女性和男性组共同的前8个特征的beeswarm图
# ===============================
def create_common_top8_shap_beeswarm(shap_values_female, shap_values_male,
                                     feature_values_female, feature_values_male,
                                     feature_names, title):
    """
    创建女性和男性组共同的前8个最重要特征的SHAP beeswarm图
    """
    # 计算每个特征在女性组和男性组中的最大绝对SHAP值，并取两者中的较大值
    max_abs_shap_female = np.max(np.abs(shap_values_female), axis=0)
    max_abs_shap_male = np.max(np.abs(shap_values_male), axis=0)
    combined_max_shap = np.maximum(max_abs_shap_female, max_abs_shap_male)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(combined_max_shap)[-8:][::-1]

    # 提取前8个特征的SHAP值和特征值
    top8_shap_female = shap_values_female[:, top8_indices]
    top8_shap_male = shap_values_male[:, top8_indices]
    top8_features_female = feature_values_female[:, top8_indices]
    top8_features_male = feature_values_male[:, top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 合并女性和男性的数据
    combined_shap = np.vstack([top8_shap_female, top8_shap_male])
    combined_features = np.vstack([top8_features_female, top8_features_male])

    # 创建组标签
    group_labels = np.array(["女性"] * top8_shap_female.shape[0] +
                            ["男性"] * top8_shap_male.shape[0])

    # 创建带有组信息的特征名称
    grouped_feature_names = [f"{name}" for name in top8_names]

    # 创建Explanation对象
    explanation = shap.Explanation(
        values=combined_shap,
        base_values=np.zeros(combined_shap.shape[0]),
        data=combined_features,
        feature_names=grouped_feature_names
    )

    # 绘制beeswarm图
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 打印前8个特征的详细信息
    print(f"\n{title}:")
    for i, idx in enumerate(top8_indices):
        max_shap_female = np.max(np.abs(shap_values_female[:, idx]))
        max_shap_male = np.max(np.abs(shap_values_male[:, idx]))
        combined_max = max(max_shap_female, max_shap_male)
        print(
            f"{i + 1}. {feature_names[idx]}: 最大|SHAP|={combined_max:.4f} (女性:{max_shap_female:.4f}, 男性:{max_shap_male:.4f})")

    return top8_indices, top8_names


# 创建共同的前8个特征beeswarm图
common_top8_indices, common_top8_names = create_common_top8_shap_beeswarm(
    female_shap, male_shap, female_features, male_features,
    feature_names, "女性和男性组共同的前8个最重要特征的SHAP Beeswarm图"
)


# ===============================
# 8. 创建特征重要性条形图
# ===============================
def create_top8_shap_barplot(shap_values, feature_names, title, color='skyblue'):
    """
    创建前8个最重要特征的SHAP值条形图
    """
    # 计算每个特征的最大绝对SHAP值
    max_abs_shap = np.max(np.abs(shap_values), axis=0)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(max_abs_shap)[-8:][::-1]
    top8_max_shap = max_abs_shap[top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 创建条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(top8_names)), top8_max_shap, color=color, alpha=0.7)

    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{top8_max_shap[i]:.4f}', ha='center', va='bottom')

    plt.xlabel('特征')
    plt.ylabel('最大|SHAP值|')
    plt.title(title)
    plt.xticks(range(len(top8_names)), top8_names, rotation=45)
    plt.tight_layout()
    plt.show()


# 创建女性组和男性组的特征重要性条形图
create_top8_shap_barplot(female_shap, feature_names, "女性组 - 前8个最重要特征的最大|SHAP值|", 'lightcoral')
create_top8_shap_barplot(male_shap, feature_names, "男性组 - 前8个最重要特征的最大|SHAP值|", 'lightblue')
'''
#3D-蜜蜂图

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ===============================
# 1. 数据准备
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
    return scaler.fit_transform(residual.reshape(-1, 1)).reshape(mat.shape)


# ===============================
# 3. 数据准备函数 - 调整为正确的3D格式
# ===============================
def prepare_data_3d(residual, age_window=3, time_steps=5):
    residual_padded = np.pad(residual, ((age_window, age_window), (0, 0)),
                             mode="constant", constant_values=0)
    X, y = [], []
    for year in range(residual_padded.shape[1] - time_steps):
        for age in range(age_window, residual_padded.shape[0] - age_window):
            # 获取3D窗口: (时间步长, 年龄窗口, 1)
            window = residual_padded[age - age_window:age + age_window + 1,
                     year:year + time_steps]
            # 转置为 (时间步长, 年龄窗口, 1)
            window = window.T.reshape(time_steps, 2 * age_window + 1, 1)
            X.append(window)
            y.append(residual_padded[age, year + time_steps])
    X = np.array(X)
    return X, np.array(y)


# ===============================
# 4. 简化的3D-CNN模型 - 避免使用可能引起问题的层
# ===============================
def build_simple_3d_cnn(time_steps=5, age_window=3):
    input_data = Input(shape=(time_steps, 2 * age_window + 1, 1, 1))

    # 使用简化的3D卷积，避免使用可能引起问题的操作
    conv1 = Conv3D(32, (1, 3, 1), activation='relu', padding='same')(input_data)
    conv2 = Conv3D(64, (1, 3, 1), activation='relu', padding='same')(conv1)

    flat = Flatten()(conv2)
    dense1 = Dense(64, activation='relu')(flat)
    output = Dense(1)(dense1)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model


# ===============================
# 5. 分别处理女性组和男性组
# ===============================
female_countries = [c for c in countries if "female" in c]
male_countries = [c for c in countries if "male" in c]


def get_group_shap_3d(countries_list, age_window=3, time_steps=5):
    all_shap_values = []
    all_feature_values = []

    for c in countries_list:
        print(f"Processing {c}...")
        residual = compute_residual(data[c])
        X, y = prepare_data_3d(residual, age_window=age_window, time_steps=time_steps)

        # 确保数据是5D格式 (样本数, 时间步长, 年龄窗口, 1, 1)
        X = X.reshape(X.shape[0], time_steps, 2 * age_window + 1, 1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_simple_3d_cnn(time_steps=time_steps, age_window=age_window)
        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

        # 使用 GradientExplainer 替代 DeepExplainer
        background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

        # 使用 GradientExplainer
        explainer = shap.GradientExplainer(model, background)
        X_explain = X_test[:30]
        shap_values = explainer.shap_values(X_explain)

        # 获取SHAP值和特征值
        sv = np.squeeze(shap_values[0] if isinstance(shap_values, list) else shap_values)
        feature_vals = np.squeeze(X_explain)

        # 重塑为二维数组 (样本数, 特征数)
        sv_reshaped = sv.reshape(sv.shape[0], -1)
        feature_vals_reshaped = feature_vals.reshape(feature_vals.shape[0], -1)

        all_shap_values.append(sv_reshaped)
        all_feature_values.append(feature_vals_reshaped)

    # 合并所有国家的SHAP值和特征值
    all_shap = np.concatenate(all_shap_values, axis=0)
    all_features = np.concatenate(all_feature_values, axis=0)

    return all_shap, all_features


# 计算女性组和男性组的SHAP值
print("计算女性组SHAP值...")
female_shap, female_features = get_group_shap_3d(female_countries)
print("计算男性组SHAP值...")
male_shap, male_features = get_group_shap_3d(male_countries)


# ===============================
# 6. 创建按最大SHAP值排序的前8个特征的beeswarm图
# ===============================
def create_top8_shap_beeswarm(shap_values, feature_values, feature_names, title):
    """
    创建前8个最重要特征的SHAP beeswarm图，按最大SHAP值排序
    """
    # 计算每个特征的最大绝对SHAP值
    max_abs_shap = np.max(np.abs(shap_values), axis=0)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(max_abs_shap)[-8:][::-1]  # 取最大的8个，并反转顺序（最重要的在前）

    # 提取前8个特征的SHAP值和特征值
    top8_shap = shap_values[:, top8_indices]
    top8_features = feature_values[:, top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 创建Explanation对象
    explanation = shap.Explanation(
        values=top8_shap,
        base_values=np.zeros(top8_shap.shape[0]),
        data=top8_features,
        feature_names=top8_names
    )

    # 绘制beeswarm图
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 打印前8个特征的详细信息
    print(f"\n{title} - 前8个最重要特征:")
    for i, idx in enumerate(top8_indices):
        max_shap = np.max(np.abs(shap_values[:, idx]))
        mean_shap = np.mean(np.abs(shap_values[:, idx]))
        print(f"{i + 1}. {feature_names[idx]}: 最大|SHAP|={max_shap:.4f}, 平均|SHAP|={mean_shap:.4f}")

    return top8_indices, top8_names


# 创建特征名称
age_window = 3
time_steps = 5
feature_names = [f"T{t}_A{a}" for t in range(time_steps) for a in range(-age_window, age_window + 1)]

# 检查SHAP值的维度并相应调整
if female_shap.shape[1] != len(feature_names):
    print(f"警告: 特征名数量({len(feature_names)})与SHAP值特征数({female_shap.shape[1]})不匹配")
    feature_names = [f"Feature_{i}" for i in range(female_shap.shape[1])]

# 绘制女性组和男性组的前8个最重要特征的SHAP beeswarm图
female_top8_indices, female_top8_names = create_top8_shap_beeswarm(
    female_shap, female_features, feature_names, "女性组 - 前8个最重要特征的SHAP Beeswarm图"
)

male_top8_indices, male_top8_names = create_top8_shap_beeswarm(
    male_shap, male_features, feature_names, "男性组 - 前8个最重要特征的SHAP Beeswarm图"
)


# ===============================
# 7. 创建女性和男性组共同的前8个特征的beeswarm图
# ===============================
def create_common_top8_shap_beeswarm(shap_values_female, shap_values_male,
                                     feature_values_female, feature_values_male,
                                     feature_names, title):
    """
    创建女性和男性组共同的前8个最重要特征的SHAP beeswarm图
    """
    # 计算每个特征在女性组和男性组中的最大绝对SHAP值，并取两者中的较大值
    max_abs_shap_female = np.max(np.abs(shap_values_female), axis=0)
    max_abs_shap_male = np.max(np.abs(shap_values_male), axis=0)
    combined_max_shap = np.maximum(max_abs_shap_female, max_abs_shap_male)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(combined_max_shap)[-8:][::-1]

    # 提取前8个特征的SHAP值和特征值
    top8_shap_female = shap_values_female[:, top8_indices]
    top8_shap_male = shap_values_male[:, top8_indices]
    top8_features_female = feature_values_female[:, top8_indices]
    top8_features_male = feature_values_male[:, top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 合并女性和男性的数据
    combined_shap = np.vstack([top8_shap_female, top8_shap_male])
    combined_features = np.vstack([top8_features_female, top8_features_male])

    # 创建Explanation对象
    explanation = shap.Explanation(
        values=combined_shap,
        base_values=np.zeros(combined_shap.shape[0]),
        data=combined_features,
        feature_names=top8_names
    )

    # 绘制beeswarm图
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 打印前8个特征的详细信息
    print(f"\n{title}:")
    for i, idx in enumerate(top8_indices):
        max_shap_female = np.max(np.abs(shap_values_female[:, idx]))
        max_shap_male = np.max(np.abs(shap_values_male[:, idx]))
        combined_max = max(max_shap_female, max_shap_male)
        print(
            f"{i + 1}. {feature_names[idx]}: 最大|SHAP|={combined_max:.4f} (女性:{max_shap_female:.4f}, 男性:{max_shap_male:.4f})")

    return top8_indices, top8_names


# 创建共同的前8个特征beeswarm图
common_top8_indices, common_top8_names = create_common_top8_shap_beeswarm(
    female_shap, male_shap, female_features, male_features,
    feature_names, "女性和男性组共同的前8个最重要特征的SHAP Beeswarm图"
)


# ===============================
# 8. 创建特征重要性条形图
# ===============================
def create_top8_shap_barplot(shap_values, feature_names, title, color='skyblue'):
    """
    创建前8个最重要特征的SHAP值条形图
    """
    # 计算每个特征的最大绝对SHAP值
    max_abs_shap = np.max(np.abs(shap_values), axis=0)

    # 选择前8个最重要的特征
    top8_indices = np.argsort(max_abs_shap)[-8:][::-1]
    top8_max_shap = max_abs_shap[top8_indices]
    top8_names = [feature_names[i] for i in top8_indices]

    # 创建条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(top8_names)), top8_max_shap, color=color, alpha=0.7)

    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{top8_max_shap[i]:.4f}', ha='center', va='bottom')

    plt.xlabel('特征')
    plt.ylabel('最大|SHAP值|')
    plt.title(title)
    plt.xticks(range(len(top8_names)), top8_names, rotation=45)
    plt.tight_layout()
    plt.show()


# 创建女性组和男性组的特征重要性条形图
create_top8_shap_barplot(female_shap, feature_names, "女性组 - 前8个最重要特征的最大|SHAP值|", 'lightcoral')
create_top8_shap_barplot(male_shap, feature_names, "男性组 - 前8个最重要特征的最大|SHAP值|", 'lightblue')




'''
#3D-CNN SHAP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.model_selection import train_test_split
import shap

# =========================
# 1️⃣ 数据加载与预处理
# =========================
data = pd.read_csv("C:/Users/32052/Desktop/Norway_male.csv")

countries = ["mortality"]

# 处理缺失值和取对数
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 划分训练/测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

num_years = len(np.unique(train_data['year']))
num_ages = len(np.unique(train_data['age']))

mortality_train = train_data[countries].values.reshape((num_years, num_ages, 1))
mortality_test = test_data[countries].values.reshape((len(np.unique(test_data['year'])), num_ages, 1))

# =========================
# 2️⃣ 计算残差
# =========================
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# =========================
# 3️⃣ 残差标准化（z-score）
# =========================
mean_age = np.mean(mx_t_h, axis=0, keepdims=True)
std_age = np.std(mx_t_h, axis=0, keepdims=True)
std_age[std_age == 0] = 1.0
mx_t_h_norm = (mx_t_h - mean_age) / std_age

# =========================
# 4️⃣ 构造训练集与标签
# =========================
def create_dataset(data, time_steps=5, age_window=3):
    X, y = [], []
    for year in range(time_steps, data.shape[0]-1):
        for age in range(age_window, data.shape[1]-age_window):
            X.append(data[year-time_steps:year, age-age_window:age+age_window+1, 0])
            y.append(data[year, age, 0])
    X = np.array(X)[..., np.newaxis, np.newaxis]  # shape (samples, time_steps, age_window*2+1, 1, 1)
    y = np.array(y)
    return X, y

X, y = create_dataset(mx_t_h_norm, time_steps=5, age_window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 5️⃣ 构建3D-CNN模型
# =========================
def build_model():
    input_data = Input(shape=(5,7,1,1))
    conv1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(input_data)
    pool1 = MaxPooling3D((2,2,1))(conv1)
    conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D((2,2,1))(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(64, activation='relu')(flat)
    output = Dense(1)(dense1)
    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model()
model.summary()

# =========================
# 6️⃣ 训练模型
# =========================
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# =========================
# 7️⃣ SHAP解释（GradientExplainer）
# =========================
background_size = 100
X_background = X_train[:background_size]

explainer = shap.GradientExplainer(model, X_background)

# 选择一个测试样本
X_sample = X_test[0:1]
shap_values = explainer.shap_values(X_sample)
shap_values_tensor = shap_values[0][0, :, :, 0, 0]  # 去掉批次和通道维度

# =========================
# 8️⃣ 平滑与放大热力图
# =========================
# 高斯平滑
shap_smoothed = gaussian_filter(shap_values_tensor, sigma=1.0)
# 放大矩阵
shap_zoomed = zoom(shap_smoothed, (20,20))  # 可以调整放大倍数

# =========================
# 9️⃣ 可视化热力图（横轴时间，纵轴年龄）
# =========================
plt.figure(figsize=(10,6))
plt.imshow(
    shap_zoomed,
    cmap='jet',
    extent=[0, 5, -3, 3],  # 横轴时间序列, 纵轴年龄窗口
    origin='lower',
    aspect='auto',
    interpolation='bicubic'
)
plt.colorbar(label='SHAP value')
plt.xticks(np.arange(0,6,1))
plt.yticks(np.arange(-3,4,1))
plt.xlabel("Years (time sequence)")
plt.ylabel("Ages (relative to target age)")
plt.title("SHAP Heatmap for Norway_female (Residual z-score)")
plt.show()
'''



'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.model_selection import train_test_split
import shap

# =========================
# 1️⃣ 数据加载与预处理
# =========================
data = pd.read_csv("C:/Users/32052/Desktop/Norway_male.csv")
countries = ["mortality"]

data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

num_years = len(np.unique(train_data['year']))
num_ages = len(np.unique(train_data['age']))

mortality_train = train_data[countries].values.reshape((num_years, num_ages, 1))
mortality_test = test_data[countries].values.reshape((len(np.unique(test_data['year'])), num_ages, 1))

# =========================
# 2️⃣ 计算残差
# =========================
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# =========================
# 3️⃣ 残差标准化
# =========================
mean_age = np.mean(mx_t_h, axis=0, keepdims=True)
std_age = np.std(mx_t_h, axis=0, keepdims=True)
std_age[std_age == 0] = 1.0
mx_t_h_norm = (mx_t_h - mean_age) / std_age


# =========================
# 4️⃣ 构造训练集与标签
# =========================
def create_dataset(data, time_steps=5, age_window=3):
    X, y = [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, 0])
            y.append(data[year, age, 0])
    X = np.array(X)[..., np.newaxis, np.newaxis]  # shape (samples, time_steps, age_window*2+1, 1, 1)
    y = np.array(y)
    return X, y


X, y = create_dataset(mx_t_h_norm, time_steps=5, age_window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 5️⃣ 构建 3D-CNN 模型（块状特征）
# =========================
def build_model_blocky():
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


model = build_model_blocky()
model.summary()

# =========================
# 6️⃣ 训练模型
# =========================
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# =========================
# 7️⃣ SHAP 解释
# =========================
background_size = 50
X_background = X_train[:background_size]

explainer = shap.GradientExplainer(model, X_background)
X_sample = X_test[0:1]
shap_values = explainer.shap_values(X_sample)
shap_tensor = shap_values[0][0, :, :, 0, 0]  # shape (time_steps, age_window*2+1)

# =========================
# 8️⃣ 可视化热力图（块状）
# =========================
plt.figure(figsize=(10, 6))
plt.imshow(
    shap_tensor,
    cmap='jet',
    extent=[0, 5, -3, 3],
    origin='lower',
    aspect='auto',
    interpolation='bicubic'
)
plt.colorbar(label='SHAP value')
# 固定刻度
plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(-3, 4, 1))
plt.xlabel("Years (time sequence)")
plt.ylabel("Ages ")
plt.title("SHAP Heatmap for Norway_male")
plt.show()

# =========================
# 9️⃣ 按年龄窗口聚合 SHAP 重要性
# =========================
age_importance = np.sum(np.abs(shap_tensor), axis=0)
age_window = (shap_tensor.shape[1] - 1) // 2

plt.figure(figsize=(10, 6))
plt.plot(range(-age_window, age_window + 1), age_importance, marker='o', color='blue')
plt.xticks(range(-age_window, age_window + 1))
plt.xlabel("Relative age (years)")
plt.ylabel("Aggregated SHAP importance")
plt.title("Aggregated SHAP importance by age window")
plt.grid(alpha=0.3)
plt.show()

# =========================
# 10️⃣ 按时间步聚合 SHAP 重要性
# =========================
time_importance = np.sum(np.abs(shap_tensor), axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(shap_tensor.shape[0]), time_importance, marker='s', color='orange')
plt.xticks(range(shap_tensor.shape[0]))
plt.xlabel("Relative year (time step)")
plt.ylabel("Aggregated SHAP importance")
plt.title("Aggregated SHAP importance by time window")
plt.grid(alpha=0.3)
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.model_selection import train_test_split
import shap

# =========================
# 1️⃣ 数据加载与预处理
# =========================
data = pd.read_csv("C:/Users/32052/Desktop/Norway_male.csv")
countries = ["mortality"]

data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

num_years = len(np.unique(train_data['year']))
num_ages = len(np.unique(train_data['age']))

mortality_train = train_data[countries].values.reshape((num_years, num_ages, 1))
mortality_test = test_data[countries].values.reshape((len(np.unique(test_data['year'])), num_ages, 1))

# =========================
# 2️⃣ 计算残差
# =========================
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# =========================
# 3️⃣ 残差标准化
# =========================
mean_age = np.mean(mx_t_h, axis=0, keepdims=True)
std_age = np.std(mx_t_h, axis=0, keepdims=True)
std_age[std_age == 0] = 1.0
mx_t_h_norm = (mx_t_h - mean_age) / std_age


# =========================
# 4️⃣ 构造训练集与标签
# =========================
def create_dataset(data, time_steps=5, age_window=3):
    X, y = [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            X.append(data[year - time_steps:year, age - age_window:age + age_window + 1, 0])
            y.append(data[year, age, 0])
    X = np.array(X)[..., np.newaxis, np.newaxis]  # shape (samples, time_steps, age_window*2+1, 1, 1)
    y = np.array(y)
    return X, y


X, y = create_dataset(mx_t_h_norm, time_steps=5, age_window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 5️⃣ 构建 3D-CNN 模型（块状特征）
# =========================
def build_model_blocky():
    input_data = Input(shape=(5, 7, 1, 1))

    # 调整卷积核，不在时间维度上平滑
    conv1 = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(input_data)
    # 调整池化层，不在时间维度上平滑
    pool1 = MaxPooling3D((1, 2, 1))(conv1)

    # 调整卷积核，不在时间维度上平滑
    conv2 = Conv3D(64, (1, 3, 3), activation='relu', padding='same')(pool1)
    # 调整池化层，不在时间维度上平滑
    pool2 = MaxPooling3D((1, 2, 1))(conv2)

    flat = Flatten()(pool2)
    dense1 = Dense(64, activation='relu')(flat)
    output = Dense(1)(dense1)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 为了保证实验的可重复性，固定随机种子
#np.random.seed(42)
#tf.random.set_seed(42)

model = build_model_blocky()
model.summary()

# =========================
# 6️⃣ 训练模型
# =========================
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# =========================
# 7️⃣ SHAP 解释
# =========================
background_size = 50
X_background = X_train[:background_size]

explainer = shap.GradientExplainer(model, X_background)
X_sample = X_test[0:1]
shap_values = explainer.shap_values(X_sample)
shap_tensor = shap_values[0][0, :, :, 0, 0]  # shape (time_steps, age_window*2+1)

# =========================
# 8️⃣ 可视化热力图（块状）
# =========================
plt.figure(figsize=(10, 6))
plt.imshow(
    shap_tensor,
    cmap='jet',
    extent=[0, 5, -3, 3],
    origin='lower',
    aspect='auto',
    interpolation='bicubic'
)
plt.colorbar(label='SHAP value')
plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(-3, 4, 1))
plt.xlabel("Years (time sequence)")
plt.ylabel("Ages ")
plt.title("SHAP Heatmap for Norway_male")
plt.show()

# =========================
# 9️⃣ 按年龄窗口聚合 SHAP 重要性
# =========================
age_importance = np.sum(np.abs(shap_tensor), axis=0)
age_window = (shap_tensor.shape[1] - 1) // 2

plt.figure(figsize=(10, 6))
plt.plot(range(-age_window, age_window + 1), age_importance, marker='o', color='blue')
plt.xticks(range(-age_window, age_window + 1))
plt.xlabel("Relative age (years)")
plt.ylabel("Aggregated SHAP importance")
plt.title("Aggregated SHAP importance by age window")
plt.grid(alpha=0.3)
plt.show()

# =========================
# 10️⃣ 按时间步聚合 SHAP 重要性
# =========================
time_importance = np.sum(np.abs(shap_tensor), axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(shap_tensor.shape[0]), time_importance, marker='s', color='orange')
plt.xticks(range(shap_tensor.shape[0]))
plt.xlabel("Relative year (time step)")
plt.ylabel("Aggregated SHAP importance")
plt.title("Aggregated SHAP importance by time window")
plt.grid(alpha=0.3)
plt.show()
'''

'''
#SHAP-3D
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.model_selection import train_test_split
import shap
import warnings

# 尝试不同的ForcePlot导入方式以兼容不同SHAP版本
try:
    from shap.plots._force import ForcePlot
except ImportError:
    try:
        from shap.plots.force import ForcePlot
    except ImportError:
        ForcePlot = None

warnings.filterwarnings('ignore')

# =========================
# 1️⃣ 数据加载与预处理
# =========================
data = pd.read_csv("C:/Users/32052/Desktop/Norway_female.csv")
countries = ["mortality"]

# 缺失值与异常值处理
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 划分训练/测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

# 3D 数据矩阵
num_years = len(np.unique(train_data['year']))
num_ages = len(np.unique(train_data['age']))
mortality_train = train_data[countries].values.reshape((num_years, num_ages, 1))

# =========================
# 2️⃣ 残差（去除年龄效应）
# =========================
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# =========================
# 3️⃣ 残差标准化
# =========================
mean_age = np.mean(mx_t_h, axis=0, keepdims=True)
std_age = np.std(mx_t_h, axis=0, keepdims=True)
std_age[std_age == 0] = 1.0
mx_t_h_norm = (mx_t_h - mean_age) / std_age


# =========================
# 4️⃣ 构造 3D 数据集
# =========================
def create_dataset(data, time_steps=5, age_window=3):
    X, y = [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            window = data[year - time_steps:year, age - age_window:age + age_window + 1, 0]
            window = window[::-1]  # 最近年份在前
            X.append(window)
            y.append(data[year, age, 0])
    X = np.array(X)[..., np.newaxis, np.newaxis]
    y = np.array(y)
    return X, y


X, y = create_dataset(mx_t_h_norm, time_steps=5, age_window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 5️⃣ 构建 3D-CNN 模型
# =========================
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


model = build_3d_cnn()

# =========================
# 6️⃣ 训练模型
# =========================
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# =========================
# 7️⃣ SHAP 解释（ForcePlot 前8个特征）
# =========================
background_size = 50
X_background = X_train[:background_size]
background_preds = model.predict(X_background)
expected_value = np.mean(background_preds)
expected_value_rounded = round(expected_value, 2)

explainer = shap.GradientExplainer(model, X_background)

sample_idx = 0
X_sample = X_test[sample_idx:sample_idx + 1]
y_true_sample = round(y_test[sample_idx], 2)
y_pred_sample = round(model.predict(X_sample).flatten()[0], 2)

# ===== 计算 SHAP 值 =====
shap_values = explainer.shap_values(X_sample)

feature_count = X_sample.shape[1] * X_sample.shape[2]
X_sample_flat = X_sample.reshape(-1, feature_count)
shap_values_flat = shap_values[0].reshape(-1, feature_count)

# 特征名
feature_names = [f"T{t}_A{a}" for t in range(5) for a in range(-3, 4)]
if len(feature_names) != feature_count:
    feature_names = [f"Feature_{i}" for i in range(feature_count)]

# ===== 取前8个最重要特征 =====
N = 8
shap_importance = np.abs(shap_values_flat[0])
top_idx = np.argsort(shap_importance)[::-1][:N]

# ===== 对SHAP值和特征值进行四舍五入 =====
shap_values_top = np.round(shap_values_flat[0][top_idx], 2)
feature_values_top = np.round(X_sample_flat[0][top_idx], 2)

# ===== 关键修改：使用SHAP值作为特征值传入，标签格式为 T0_A0=+0.22 =====
feature_labels_top = []
for i in top_idx:
    shap_value_rounded = np.round(shap_values_flat[0][i], 2)
    # 格式化SHAP值（带符号）
    if shap_value_rounded >= 0:
        shap_str = f"+{shap_value_rounded:.2f}"
    else:
        shap_str = f"{shap_value_rounded:.2f}"
    feature_labels_top.append(f"{feature_names[i]}={shap_str}")

# ===== 使用SHAP值作为特征值传入 =====
X_sample_top = shap_values_top

# ===== 绘制 ForcePlot =====
plt.figure(figsize=(12, 8))
shap_plot = shap.force_plot(
    expected_value_rounded,
    shap_values_top,
    X_sample_top,  # 使用SHAP值作为特征值
    feature_names=feature_labels_top,  # 标签包含特征名称和SHAP值
    matplotlib=True,
    show=False
)

plt.title(f"Sample {sample_idx} - True: {y_true_sample:.2f}, Pred: {y_pred_sample:.2f}")
plt.tight_layout()
plt.show()

# ===== 额外输出特征值的详细信息 =====
print(f"\n=== 特征值详细信息 ===")
print(f"样本索引: {sample_idx}")
print(f"真实值: {y_true_sample:.2f}")
print(f"预测值: {y_pred_sample:.2f}")
print(f"基准值: {expected_value_rounded:.2f}")
print(f"\n前{N}个最重要特征的特征值:")

# 创建特征值表格
feature_table_data = []
for i, idx in enumerate(top_idx):
    shap_value_rounded = np.round(shap_values_flat[0][idx], 2)
    feature_value_rounded = np.round(X_sample_flat[0][idx], 2)

    # 格式化SHAP值（带符号）
    if shap_value_rounded >= 0:
        shap_str = f"+{shap_value_rounded:.2f}"
    else:
        shap_str = f"{shap_value_rounded:.2f}"

    feature_table_data.append([
        feature_names[idx],
        shap_str,
        f"{feature_value_rounded:.2f}"
    ])

# 使用pandas创建漂亮的表格
feature_df = pd.DataFrame(feature_table_data, columns=["特征", "SHAP值", "特征值"])
print(feature_df.to_string(index=False))

print(f"\nSHAP值总和: {np.round(np.sum(shap_values_flat[0]), 2)}")
print(
    f"预测值验证: 基准值({expected_value_rounded:.2f}) + SHAP总和({np.round(np.sum(shap_values_flat[0]), 2)}) = {np.round(expected_value_rounded + np.sum(shap_values_flat[0]), 2)}")

'''

#SHAP-2D
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shap
import warnings

# 尝试不同的ForcePlot导入方式以兼容不同SHAP版本
try:
    from shap.plots._force import ForcePlot
except ImportError:
    try:
        from shap.plots.force import ForcePlot
    except ImportError:
        ForcePlot = None

warnings.filterwarnings('ignore')

# =========================
# 1️⃣ 数据加载与预处理
# =========================
data = pd.read_csv("C:/Users/32052/Desktop/Norway_male.csv")
countries = ["mortality"]

# 缺失值与异常值处理
data[countries] = data[countries].replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
data[countries] = data[countries].ffill()
data[countries] = data[countries].apply(lambda x: np.log(x))

# 划分训练/测试集
train_data = data[data['year'] <= 2015]
test_data = data[data['year'] > 2015]

# 3D 数据矩阵
num_years = len(np.unique(train_data['year']))
num_ages = len(np.unique(train_data['age']))
mortality_train = train_data[countries].values.reshape((num_years, num_ages, 1))

# =========================
# 2️⃣ 残差（去除年龄效应）
# =========================
ax_h = train_data.groupby('age')[countries].mean()
ax_t_h = np.tile(ax_h.values[:, :, np.newaxis], (1, 1, num_years))
ax_t_h = np.transpose(ax_t_h, (2, 0, 1))
mx_t_h = mortality_train - ax_t_h

# =========================
# 3️⃣ 残差标准化
# =========================
mean_age = np.mean(mx_t_h, axis=0, keepdims=True)
std_age = np.std(mx_t_h, axis=0, keepdims=True)
std_age[std_age == 0] = 1.0
mx_t_h_norm = (mx_t_h - mean_age) / std_age


# =========================
# 4️⃣ 构造 2D 数据集
# =========================
def create_dataset(data, time_steps=5, age_window=3):
    X, y = [], []
    for year in range(time_steps, data.shape[0] - 1):
        for age in range(age_window, data.shape[1] - age_window):
            window = data[year - time_steps:year, age - age_window:age + age_window + 1, 0]
            window = window[::-1]  # 最近年份在前
            X.append(window)
            y.append(data[year, age, 0])
    # 调整为2D CNN输入格式 (样本数, 时间步长, 年龄窗口, 1)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


X, y = create_dataset(mx_t_h_norm, time_steps=5, age_window=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 5️⃣ 构建 2D-CNN 模型
# =========================
def cnn2d_model():
    """构建2D卷积神经网络模型"""
    # 使用挪威女性的参数
    #params = {'filters1': 112, 'filters2': 64, 'dense_units': 200, 'learning_rate': 0.00019}  # 女
    params = {'filters1': 96, 'filters2': 224, 'dense_units': 250, 'learning_rate': 0.00010}#男

    model = Sequential([
        Input(shape=(5, 7, 1)),  # 输入形状：5时间步 × 7年龄窗口 × 1通道
        Conv2D(int(params['filters1']), (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(int(params['filters2']), (3, 3), activation='relu', padding='same'),
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


model = cnn2d_model()

# =========================
# 6️⃣ 训练模型
# =========================
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# =========================
# 7️⃣ SHAP 解释（ForcePlot 前8个特征）
# =========================
background_size = 50
X_background = X_train[:background_size]
background_preds = model.predict(X_background)
expected_value = np.mean(background_preds)
expected_value_rounded = round(expected_value, 2)

explainer = shap.GradientExplainer(model, X_background)

sample_idx = 0
X_sample = X_test[sample_idx:sample_idx + 1]
y_true_sample = round(y_test[sample_idx], 2)
y_pred_sample = round(model.predict(X_sample).flatten()[0], 2)

# ===== 计算 SHAP 值 =====
shap_values = explainer.shap_values(X_sample)

feature_count = X_sample.shape[1] * X_sample.shape[2]
X_sample_flat = X_sample.reshape(-1, feature_count)
shap_values_flat = shap_values[0].reshape(-1, feature_count)

# 特征名
feature_names = [f"T{t}_A{a}" for t in range(5) for a in range(-3, 4)]
if len(feature_names) != feature_count:
    feature_names = [f"Feature_{i}" for i in range(feature_count)]

# ===== 取前8个最重要特征 =====
N = 8
shap_importance = np.abs(shap_values_flat[0])
top_idx = np.argsort(shap_importance)[::-1][:N]

# ===== 对SHAP值和特征值进行四舍五入 =====
shap_values_top = np.round(shap_values_flat[0][top_idx], 2)
feature_values_top = np.round(X_sample_flat[0][top_idx], 2)

# ===== 关键修改：使用SHAP值作为特征值传入，标签格式为 T0_A0=+0.22 =====
feature_labels_top = []
for i in top_idx:
    shap_value_rounded = np.round(shap_values_flat[0][i], 2)
    # 格式化SHAP值（带符号）
    if shap_value_rounded >= 0:
        shap_str = f"+{shap_value_rounded:.2f}"
    else:
        shap_str = f"{shap_value_rounded:.2f}"
    feature_labels_top.append(f"{feature_names[i]}={shap_str}")

# ===== 使用SHAP值作为特征值传入 =====
X_sample_top = shap_values_top

# ===== 绘制 ForcePlot =====
plt.figure(figsize=(12, 8))
shap_plot = shap.force_plot(
    expected_value_rounded,
    shap_values_top,
    X_sample_top,  # 使用SHAP值作为特征值
    feature_names=feature_labels_top,  # 标签包含特征名称和SHAP值
    matplotlib=True,
    show=False
)

plt.title(f"Sample {sample_idx} - True: {y_true_sample:.2f}, Pred: {y_pred_sample:.2f}")
plt.tight_layout()
plt.show()

# ===== 额外输出特征值的详细信息 =====
print(f"\n=== 特征值详细信息 ===")
print(f"样本索引: {sample_idx}")
print(f"真实值: {y_true_sample:.2f}")
print(f"预测值: {y_pred_sample:.2f}")
print(f"基准值: {expected_value_rounded:.2f}")
print(f"\n前{N}个最重要特征的特征值:")

# 创建特征值表格
feature_table_data = []
for i, idx in enumerate(top_idx):
    shap_value_rounded = np.round(shap_values_flat[0][idx], 2)
    feature_value_rounded = np.round(X_sample_flat[0][idx], 2)

    # 格式化SHAP值（带符号）
    if shap_value_rounded >= 0:
        shap_str = f"+{shap_value_rounded:.2f}"
    else:
        shap_str = f"{shap_value_rounded:.2f}"

    feature_table_data.append([
        feature_names[idx],
        shap_str,
        f"{feature_value_rounded:.2f}"
    ])

# 使用pandas创建漂亮的表格
feature_df = pd.DataFrame(feature_table_data, columns=["特征", "SHAP值", "特征值"])
print(feature_df.to_string(index=False))

print(f"\nSHAP值总和: {np.round(np.sum(shap_values_flat[0]), 2)}")
print(
    f"预测值验证: 基准值({expected_value_rounded:.2f}) + SHAP总和({np.round(np.sum(shap_values_flat[0]), 2)}) = {np.round(expected_value_rounded + np.sum(shap_values_flat[0]), 2)}")
