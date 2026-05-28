import os
import gc
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.special import gammaln
import matplotlib.pyplot as plt # 用于后续画图解释

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_lattice as tfl

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Input, Multiply, Embedding, Reshape, 
                                     Concatenate, Dropout, BatchNormalization, Lambda, 
                                     Layer, Activation, LeakyReLU, Add)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as K
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')

SEED_VALUE = 42
tf.keras.utils.set_random_seed(SEED_VALUE)
tf.keras.backend.set_floatx("float32")

# ==========================================
# 1. 下载与加载数据
# ==========================================
print("1. Downloading Authentic French Motor Data...")
from sklearn.datasets import fetch_openml
data_raw = fetch_openml(data_id=41214, as_frame=True, parser='auto')
df = data_raw.frame

# 标准清洗
df.loc[df['ClaimNb'] >= 5, 'ClaimNb'] = 4
df = df[(df['VehAge'] >= 0) & (df['VehAge'] <= 25) &
        (df['DrivAge'] >= 18) & (df['DrivAge'] <= 80)]
df.loc[df["Exposure"] > 1.0, "Exposure"] = 1.0

df['VehGas'] = df['VehGas'].astype(str)
df['Area'] = df['Area'].astype(str)
df['VehBrand'] = df['VehBrand'].astype(str)
df['Region'] = df['Region'].astype(str)

expo_var = 'Exposure'
target_var = 'ClaimNb'
cat_vars = ['Area', 'VehBrand', 'VehGas', 'Region']
num_vars = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
all_vars = cat_vars + num_vars + [expo_var]

# 严格保证单调性逻辑
monotonicity_list = {'VehAge': 'increasing', 'BonusMalus': 'decreasing'}
monotonicity_list_pi = {'VehAge': 'decreasing', 'BonusMalus': 'increasing'}

# 全量数据变量
w_raw = df[expo_var].values.astype(np.float32)
y_raw = df[target_var].values.astype(np.float32)

# 计算全集先验
GLOBAL_MEAN = np.sum(w_raw * y_raw) / np.sum(w_raw)                   
init_mu_bias = np.log(GLOBAL_MEAN + 1e-7)
PRIOR_PI = 0.85 

preprocessor_cat = OrdinalEncoder()
X_cat_encoded = preprocessor_cat.fit_transform(df[cat_vars]).astype(np.int32)
cat_uniques = {var: len(np.unique(X_cat_encoded[:, i])) for i, var in enumerate(cat_vars)}

# 全集数值特征标准化
X_num_raw = df[num_vars].values.astype(np.float32)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_raw)

def prepare_inputs(X_num, X_cat, w):
    inputs = {}
    inputs[expo_var] = w
    for i, var in enumerate(num_vars):
        inputs[var] = X_num[:, i]
    for i, var in enumerate(cat_vars):
        inputs[var] = X_cat[:, i]
    return inputs

# 准备供模型训练的全集输入字典
full_inputs = prepare_inputs(X_num_scaled, X_cat_encoded, w_raw)

# ==========================================
# 2. 自定义层与损失函数
# ==========================================
# ==========================================
# 3. 满血模型构建
# ==========================================
@tf.keras.utils.register_keras_serializable()
class AddBias(Layer):
    def __init__(self, init_val=0.0, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.init_val = init_val
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.b = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(self.init_val), trainable=True, name='bias')
        super().build(input_shape)

    def call(self, inputs):
        subnet_sum = K.sum(inputs, axis=-1, keepdims=True)
        output = subnet_sum + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    # ★ 新增：为了能够顺利保存模型，必须加上 get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_val": self.init_val,
            # 将激活函数序列化保存
            "activation": tf.keras.activations.serialize(self.activation) if self.activation else None,
        })
        return config
    
def create_subnet(num_layers, units_first_layer, activation, model_name, dropout_rate=0.0, l2=0.001):
    model = Sequential(name=model_name)
    for i in range(num_layers):
        num_units = max(1, units_first_layer - i * int(units_first_layer / num_layers))
        if activation == 'leaky_relu':
            model.add(Dense(num_units, kernel_regularizer=L2(l2)))
            model.add(LeakyReLU(alpha=0.1))
        else:
            model.add(Dense(num_units, activation=activation, kernel_regularizer=L2(l2)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, use_bias=False, kernel_regularizer=L2(l2))) 
    return model

def add_calibrate_layer(var, other_var, data_num, data_cat, monotonicity_list, num_keypoints, prefix=""):
    if var in cat_vars:
        var_idx = cat_vars.index(var)
        num_buckets = len(np.unique(data_cat[:, var_idx]))
        return tfl.layers.CategoricalCalibration(
            num_buckets=num_buckets, 
            output_min=0.0, output_max=5.0, dtype=tf.float32, 
            name=f"{prefix}{var}_cal_{other_var}"
        )
    else:
        var_idx = num_vars.index(var)
        kp_array = np.linspace(data_num[:, var_idx].min(), data_num[:, var_idx].max(), num=num_keypoints).astype(np.float32)
        return tfl.layers.PWLCalibration(
            input_keypoints=kp_array, monotonicity=monotonicity_list.get(var, "none"), 
            output_min=0.0, output_max=5.0, dtype=tf.float32, 
            name=f"{prefix}{var}_cal_{other_var}"
        )

@tf.function
def zip_logloss(y_true, mu, pi):
    poisson_log_prob = y_true * tf.math.log(mu + 1e-7) - mu - tf.math.lgamma(y_true + 1)
    non_zero_part = tf.math.log(1 - pi + 1e-7) + poisson_log_prob
    zero_part = tf.math.log(pi + (1 - pi) * tf.exp(-mu) + 1e-7)
    return -tf.where(y_true > 0, non_zero_part, zero_part)

class ZIPLogLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return zip_logloss(y_true, y_pred[:, 0], y_pred[:, 1])

# ==========================================
# 3. 满血 NAM 构建 (全集训练用)
# ==========================================
def build_full_nam(hp, X_num_train, X_cat_train):
    inputs_dict = {}
    embed_inputs = {}
    mu_subnets = []
    pi_subnets = []
    
    interactions = list(combinations(num_vars + cat_vars, 2))
    lattice_sizes = {var: np.unique(X_cat_train[:, cat_vars.index(var)]).shape[0] if var in cat_vars else hp['num_vertices'] for var in num_vars + cat_vars}
    
    expo_in = Input(shape=(1,), name=expo_var)
    inputs_dict[expo_var] = expo_in
    
    for var in num_vars:
        inp = Input(shape=(1,), name=var)
        inputs_dict[var] = inp
        embed_inputs[var] = inp
        
    for var in cat_vars:
        inp = Input(shape=(1,), name=var, dtype=tf.int32)
        inputs_dict[var] = inp
        emb = Embedding(input_dim=cat_uniques[var], output_dim=1)(inp)
        embed_inputs[var] = Reshape((1,))(emb)
    
    for var in num_vars:
        if var in monotonicity_list:
            cal_m = add_calibrate_layer(var, "main_m", X_num_train, X_cat_train, monotonicity_list, hp['kp'], "mu_")
            cal_p = add_calibrate_layer(var, "main_p", X_num_train, X_cat_train, monotonicity_list_pi, hp['kp'], "pi_")
            mu_subnets.append(cal_m(inputs_dict[var]))
            pi_subnets.append(cal_p(inputs_dict[var]))
        else:
            mu_subnets.append(create_subnet(hp['main_layers'], hp['main_units'], 'relu', f"mu_{var}_sub", hp['dropout'], hp['l2'])(inputs_dict[var]))
            pi_subnets.append(create_subnet(hp['main_layers'], max(1, hp['main_units']//2), 'relu', f"pi_{var}_sub", hp['dropout'], hp['l2'])(inputs_dict[var]))
            
    for var in cat_vars:
        mu_subnets.append(embed_inputs[var])
        emb_pi = Embedding(input_dim=cat_uniques[var], output_dim=1)(inputs_dict[var])
        pi_subnets.append(Reshape((1,))(emb_pi))

    for (var1, var2) in interactions:
        v1_in = embed_inputs[var1]
        v2_in = embed_inputs[var2]
        if any(v in monotonicity_list for v in [var1, var2]):
            m_cal1 = add_calibrate_layer(var1, var2, X_num_train, X_cat_train, monotonicity_list, hp['kp'], "mu_")(inputs_dict[var1])
            m_cal2 = add_calibrate_layer(var2, var1, X_num_train, X_cat_train, monotonicity_list, hp['kp'], "mu_")(inputs_dict[var2])
            if var1 in cat_vars: m_cal1 = Lambda(lambda x: tf.cast(x, tf.float32))(m_cal1)
            if var2 in cat_vars: m_cal2 = Lambda(lambda x: tf.cast(x, tf.float32))(m_cal2)
            
            m_lat = tfl.layers.Lattice(lattice_sizes=[lattice_sizes[var1], lattice_sizes[var2]],
                                       monotonicities=["increasing" if var1 in monotonicity_list else 'none', "increasing" if var2 in monotonicity_list else 'none'])([m_cal1, m_cal2])
            mu_subnets.append(m_lat)
            
            p_cal1 = add_calibrate_layer(var1, var2, X_num_train, X_cat_train, monotonicity_list_pi, hp['kp'], "pi_")(inputs_dict[var1])
            p_cal2 = add_calibrate_layer(var2, var1, X_num_train, X_cat_train, monotonicity_list_pi, hp['kp'], "pi_")(inputs_dict[var2])
            if var1 in cat_vars: p_cal1 = Lambda(lambda x: tf.cast(x, tf.float32))(p_cal1)
            if var2 in cat_vars: p_cal2 = Lambda(lambda x: tf.cast(x, tf.float32))(p_cal2)
            
            p_lat = tfl.layers.Lattice(lattice_sizes=[lattice_sizes[var1], lattice_sizes[var2]],
                                       monotonicities=["increasing" if var1 in monotonicity_list_pi else 'none', "increasing" if var2 in monotonicity_list_pi else 'none'])([p_cal1, p_cal2])
            pi_subnets.append(p_lat)
        else:
            concat_in = Concatenate()([v1_in, v2_in])
            mu_subnets.append(create_subnet(hp['int_layers'], hp['int_units'], 'relu', f"mu_{var1}_{var2}_int", hp['dropout'], hp['l2'])(concat_in))
            pi_subnets.append(create_subnet(hp['int_layers'], max(1, hp['int_units']//2), 'relu', f"pi_{var1}_{var2}_int", hp['dropout'], hp['l2'])(concat_in))

    flat_mu = [Reshape((1,))(s) for s in mu_subnets]
    flat_pi = [Reshape((1,))(s) for s in pi_subnets]
    
    z1 = AddBias(init_val=init_mu_bias, name='mu_bias_layer')(Add()(flat_mu))
    z2 = AddBias(init_val=0.0, name='pi_bias_layer')(Add()(flat_pi))
    
    mu = Lambda(lambda x: tf.exp(tf.clip_by_value(x, -10.0, 5.0)), name='mu_output')(z1)
    mu_with_expo = Multiply()([expo_in, mu])
    pi_final = Activation('sigmoid', name='pi_output')(tf.clip_by_value(z2, -10.0, 10.0))
    
    outputs = Concatenate()([mu_with_expo, pi_final])
    model = Model(inputs=inputs_dict, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['lr']), loss=ZIPLogLoss())
    return model

# ==========================================
# 4. 全集训练 (Train on Full Dataset)
# ==========================================
# 使用经过 10 折检验的最佳超参数
BEST_HP = {
    'main_units': 128, 'main_layers': 2,
    'int_units': 64, 'int_layers': 2,
    'num_vertices': 6, 'kp': 40,
    'l2': 1e-4, 'activation': 'relu',
    'dropout': 0.1, 'lr': 0.001
}

print("\n2. Training on FULL Dataset for Interpretation...")
model_full = build_full_nam(BEST_HP, X_num_scaled, X_cat_encoded)

# 在全集上训练不使用 Validation Split 触发 EarlyStopping，而是给定固定的 Epochs
# 假设 10 折时平均在 60 Epochs 停下，我们这里直接训 60 轮
model_full.fit(
    full_inputs, y_raw,
    epochs=60,               
    batch_size=8192,
    verbose=1
)

print("Training completed. Saving final model...")
model_full.save('full_nam_zip_model.keras')

# ==========================================
# 5. 全集评估验证 (验证训练是否成功)
# ==========================================
y_pred_full = model_full.predict(full_inputs, batch_size=8192, verbose=0)
mu_full, pi_full = y_pred_full[:, 0], y_pred_full[:, 1]

sum_zip_log_loss = tf.reduce_sum(zip_logloss(y_raw, mu_full, pi_full)).numpy() 

y_hat = mu_full * (1 - pi_full)
y_hat_safe = np.clip(y_hat, 1e-7, 1e6)
d_model = np.where(y_raw < 1e-3, 2 * y_hat_safe, 2 * (y_raw * np.log(y_raw / y_hat_safe) - y_raw + y_hat_safe))
dev_percent = np.mean(d_model) * 100

y_bar = np.sum(y_raw * w_raw) / np.sum(w_raw)
y_null = np.clip(w_raw * y_bar, 1e-7, 1e6)
d_null = np.where(y_raw < 1e-3, 2 * y_null, 2 * (y_raw * np.log(y_raw / y_null) - y_raw + y_null))
pseudo_r2 = (1 - (np.sum(d_model) / np.sum(d_null))) * 100

print(f"\n=== Full Dataset Verification Metrics ===")
print(f"Total ZIP NLL (All {len(y_raw):,} rows): {sum_zip_log_loss:.2f}")
print(f"-> Equivalent 10-Fold NLL (for baseline): {sum_zip_log_loss/10.0:.2f} (Should be ~13700-14200)")
print(f"Dev(%): {dev_percent:.2f}% (Should be ~30%)")
print(f"Pseudo R²(%): {pseudo_r2:.2f}% (Should be ~8%)")
print("=========================================")

