from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Embedding, Reshape, Concatenate, Layer, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import os

from tensorflow.keras.layers import Layer, Dense, Concatenate, Input, Multiply, Add
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input, Multiply, Add, BatchNormalization
class ProtoLayer(Layer):
    def __init__(self, num_prototypes=64, **kwargs):
        super().__init__(**kwargs)
        self.num_prototypes = num_prototypes  # 每个特征的原型数量

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        # 初始化原型参数 (num_features, num_prototypes)
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.num_features, self.num_prototypes),
            initializer='zeros',  # 初始值为0，之后用分位数覆盖
            trainable=True
        )
        # 线性参数 (num_features, num_prototypes)
        self.a = self.add_weight(
            name='a',
            shape=(self.num_features, self.num_prototypes),
            initializer='ones',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.num_features, self.num_prototypes),
            initializer='zeros',
            trainable=True
        )
        # 动态sigma参数
        self.sigma = self.add_weight(
            name='sigma',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        # 输入形状 (batch_size, num_features)
        inputs_exp = tf.expand_dims(inputs, axis=-1)  # (B,F,1)
        prototypes_exp = tf.expand_dims(self.prototypes, axis=0)  # (1,F,P)
        
        # 计算距离
        distances = (inputs_exp - prototypes_exp)**2  # (B,F,P)
        rbf = tf.exp(-distances / (2 * self.sigma**2 + 1e-8))
        
        # 调整线性参数维度
        a_exp = tf.expand_dims(self.a, axis=0)  # (1,F,P)
        b_exp = tf.expand_dims(self.b, axis=0)  # (1,F,P)
        linear = a_exp * inputs_exp + b_exp  # (B,F,P)
        
        # 加权归一化
        weighted = linear * rbf  # (B,F,P)
        sum_rbf = tf.reduce_sum(rbf, axis=-1, keepdims=True)  # (B,F,1)
        normalized = weighted / (sum_rbf + 1e-8)
        
        # 聚合输出
        return tf.reduce_sum(normalized, axis=-1)  # (B,F)

    def set_prototypes_from_quantiles(self, data):
        """
        使用数据的分位数初始化原型。
        :param data: 输入数据 (batch_size, num_features)
        """
        quantiles = np.percentile(data, np.linspace(0, 100, self.num_prototypes), axis=0).T
        self.prototypes.assign(quantiles.astype(np.float32))

class SigmaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, proto_layer, total_epochs, tau):
        super().__init__()
        self.proto_layer = proto_layer
        self.total_epochs = total_epochs
        self.tau = tau

    def on_epoch_begin(self, epoch, logs=None):
        T = epoch
        Tmax = self.total_epochs
        sigma = 1 / (1 + tf.exp((T - Tmax / 2) / self.tau))
        self.proto_layer.sigma.assign(sigma)


class ProtoFeatureEncoder(Layer):
    def __init__(self, num_layers, units_per_layer, activation='relu', **kwargs):
        """
        使用共享残差块的 Proto 特征编码器。
        """
        super(ProtoFeatureEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation_fn = tf.keras.activations.get(activation)

        # 残差块
        self.shared_dense_layers = [
            Dense(units_per_layer, activation=self.activation_fn)
            for _ in range(num_layers)
        ]
        self.shared_shortcut_layers = [
            Dense(units_per_layer, activation=None)
            for _ in range(num_layers)
        ]

        # 最终预测层
        self.final_predictor = Dense(1, activation='linear')

    def call(self, inputs):
        _, proto_activations = inputs  # proto_activations shape: (B, F)
        num_features = proto_activations.shape[-1]

        outputs = []

        for i in range(num_features):
            x = tf.expand_dims(proto_activations[:, i], axis=-1)  # (B, 1)

            for j in range(self.num_layers):
                shortcut = self.shared_shortcut_layers[j](x)
                x = self.shared_dense_layers[j](x)
                x = tf.keras.activations.relu(x + shortcut)

            pred = self.final_predictor(x)  # (B, 1)
            outputs.append(pred)

        return Concatenate(axis=1)(outputs)  # (B, F)


class SparseFeatureMaskLayer(Layer):
    def __init__(self, num_features, sparsity=0.01, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.sparsity = sparsity  # 稀疏性比例

    def build(self, input_shape):
        self.mask = self.add_weight(
            name='mask',
            shape=(self.num_features,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # 应用稀疏性约束
        sparse_mask = tf.nn.relu(self.mask - self.sparsity)
        return inputs * sparse_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'sparsity': self.sparsity
        })
        return config


def Create_Poisson_FNN_Cat_Emb_Proto(input_dim, emb_dim, cat_vocabulary, num_prototypes=64, 
                                    num_layers=4, units_per_layer=30, sparsity=0.2):  # 添加sparsity参数
    # 输入层
    Input_Matrix_Just_NR = Input(shape=(input_dim,), dtype='float32', name='Input_Matrix_Just_NR')
    Input_Exposure = Input(shape=(1,), dtype='float32', name='Input_Exposure')
    Input_EMB_Area = Input(shape=(1,), dtype='int32', name='Input_EMB_Area')
    Input_EMB_VehBrand = Input(shape=(1,), dtype='int32', name='Input_EMB_VehBrand')
    Input_EMB_Region = Input(shape=(1,), dtype='int32', name='Input_EMB_Region')
    
    # 嵌入层
    emb_Area = Embedding(input_dim=len(cat_vocabulary["Cat_Area"]), output_dim=emb_dim, input_length=1, name='emb_Area')(Input_EMB_Area)
    emb_VehBrand = Embedding(input_dim=len(cat_vocabulary["Cat_VehBrand"]), output_dim=emb_dim, input_length=1, name='emb_VehBrand')(Input_EMB_VehBrand)
    emb_Region = Embedding(input_dim=len(cat_vocabulary["Cat_Region"]), output_dim=emb_dim, input_length=1, name='emb_Region')(Input_EMB_Region)
    
    # 将嵌入层的输出展平
    emb_Area = Reshape((emb_dim,))(emb_Area)
    emb_VehBrand = Reshape((emb_dim,))(emb_VehBrand)
    emb_Region = Reshape((emb_dim,))(emb_Region)

    # 将数值特征和嵌入特征合并
    combined = Concatenate(name='combined')([Input_Matrix_Just_NR, emb_Area, emb_VehBrand, emb_Region])
    
    # 添加稀疏特征归因掩码层，使用传入的sparsity参数
    sparse_combined = SparseFeatureMaskLayer(
        num_features=combined.shape[-1], 
        sparsity=sparsity,  # 使用参数值
        name='sparse_mask'
    )(combined)
    
    proto_output = ProtoLayer(num_prototypes=64, name='proto_layer')(sparse_combined)
    
    encoder_outputs = ProtoFeatureEncoder(num_layers=num_layers, units_per_layer=units_per_layer, name='proto_encoder')([proto_output, proto_output])
    
    hidden1 = Dense(20, activation='relu', name='hidden1')(encoder_outputs)
    hidden2 = Dense(15, activation='relu', name='hidden2')(hidden1)
    hidden3 = Dense(10, activation='relu', name='hidden3')(hidden2)
    
    Result_FNN1 = Dense(1, activation='exponential', name='Result_FFN1', trainable=True)(hidden3)
    
    Response = Multiply(name='Result')([Result_FNN1, Input_Exposure])
    
    model = Model(inputs=[Input_Matrix_Just_NR, Input_EMB_Area, Input_EMB_VehBrand, Input_EMB_Region, Input_Exposure], 
                  outputs=[Response], name='Poisson_FNN_Cat_Emb_Proto')
    
    return model

# 主循环
for run_index in range(15):
    print(f"Model: {run_index}")
    data_nn_cat_emb_learn, y_true_learn = create_ffn_cat_emb_data(bool_in_learn)
    data_nn_cat_emb_learn_test, y_true_test = create_ffn_cat_emb_data(bool_in_test)
    
    # Create the dataframes needed for training:
    data_nn_cat_emb_learn_train, y_true_learn_train = create_ffn_cat_emb_data(train_val_split[f"learn_train_{run_index}"])
    data_nn_cat_emb_learn_val, y_true_learn_val = create_ffn_cat_emb_data(train_val_split[f"learn_val_{run_index}"])
    
    # 设置随机种子
    tf.random.set_seed(random_seeds[run_index])
    np.random.seed(random_seeds[run_index])
    
    # 创建模型
    FNN_Cat_Emb_Proto = Create_Poisson_FNN_Cat_Emb_Proto(input_dim=len(nr_col), emb_dim=1, cat_vocabulary=cat_vocabulary, num_prototypes=64, num_layers=4, units_per_layer=30, sparsity=0.001)

    # 获取数值特征和分类特征
    X_nn_just_nr_train, Input_EMB_Area_train, Input_EMB_VehBrand_train, Input_EMB_Region_train, exposure_train = data_nn_cat_emb_learn_train
    X_nn_just_nr_val, Input_EMB_Area_val, Input_EMB_VehBrand_val, Input_EMB_Region_val, exposure_val = data_nn_cat_emb_learn_val
    
    # 合并数值特征和分类特征
    X_train = np.concatenate([X_nn_just_nr_train, Input_EMB_Area_train.reshape(-1, 1), Input_EMB_VehBrand_train.reshape(-1, 1), Input_EMB_Region_train.reshape(-1, 1)], axis=1)
    X_val = np.concatenate([X_nn_just_nr_val, Input_EMB_Area_val.reshape(-1, 1), Input_EMB_VehBrand_val.reshape(-1, 1), Input_EMB_Region_val.reshape(-1, 1)], axis=1)
    
    # 初始化原型参数
    proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
    proto_layer.set_prototypes_from_quantiles(X_train)
    
    # 编译模型
    FNN_Cat_Emb_Proto.compile(optimizer='adam', loss=poisson_loss_for_tf, metrics=[poisson_loss_for_tf])
    
    # 模型回调
    early_stopping_callback = EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)

    # 模型训练
    start_time = time.time()
    epochs_Cat_Emb_Proto = 100
    # 回调设置
    sigma_scheduler = SigmaScheduler(proto_layer, total_epochs=epochs_Cat_Emb_Proto, tau=16)
    callbacks = [early_stopping_callback, sigma_scheduler]
    FNN_Cat_Emb_Proto_history = FNN_Cat_Emb_Proto.fit(
        x=data_nn_cat_emb_learn_train,
        y=y_true_learn_train,
        validation_data=(data_nn_cat_emb_learn_val, y_true_learn_val),
        epochs=epochs_Cat_Emb_Proto,
        batch_size=5000,
        verbose=2,
        callbacks=callbacks
    )
    end_time = time.time()
    execution_time_nn_cat_emb_proto = end_time - start_time
    best_epoch_FNN_cat_emb_proto = np.argmin(FNN_Cat_Emb_Proto_history.history['val_loss']) + 1

    # 保存模型权重
    FNN_Cat_Emb_Proto.save_weights(f'{storage_path}/saved_models/FNN_Proto_res{run_index}1013.weights.h5')

    # 加载模型权重
    FNN_Cat_Emb_Proto.load_weights(f'{storage_path}/saved_models/FNN_Proto_res{run_index}1013.weights.h5')


    # 预测
    y_pred["train"][f"FNN_Cat_Emb_Proto_{run_index}"] = np.array([x for [x] in FNN_Cat_Emb_Proto.predict(data_nn_cat_emb_learn, batch_size=100000)])
    y_pred["test"][f"FNN_Cat_Emb_Proto_{run_index}"] = np.array([x for [x] in FNN_Cat_Emb_Proto.predict(data_nn_cat_emb_learn_test, batch_size=100000)])

    # 评估模型
    FNN_Cat_Emb_Proto_results = Results(
        model=f"FNN_Cat_Emb_Proto (run: {run_index})",
        epochs=best_epoch_FNN_cat_emb_proto,
        run_time=execution_time_nn_cat_emb_proto,
        nr_parameters=np.sum([np.prod(v.shape.as_list()) for v in FNN_Cat_Emb_Proto.trainable_weights]),
        poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"][f"FNN_Cat_Emb_Proto_{run_index}"]),
        poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"][f"FNN_Cat_Emb_Proto_{run_index}"]),
        pred_avg_freq_train=y_pred["train"][f"FNN_Cat_Emb_Proto_{run_index}"].sum() / exposure["train"].sum(),
        pred_avg_freq_test=y_pred["test"][f"FNN_Cat_Emb_Proto_{run_index}"].sum() / exposure["test"].sum()
    )

    # 将结果存入 DataFrame
    store_results_in_df(FNN_Cat_Emb_Proto_results)

    # 在每轮结束时显示 df_results
    print(f"Results after run {run_index + 1}:")
    display(df_results)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 选择一个模型
run_index = 0
FNN_Cat_Emb_Proto = Create_Poisson_FNN_Cat_Emb_Proto(input_dim=len(nr_col), emb_dim=1, cat_vocabulary=cat_vocabulary, num_prototypes=64, num_layers=4, units_per_layer=30)
weights_path = f'{storage_path}/saved_models/FNN_Proto_res{run_index}1013.weights.h5'
FNN_Cat_Emb_Proto.load_weights(weights_path)

# 提取数值特征的原型参数
proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
prototypes = proto_layer.get_weights()[0]  # 形状: (num_features, num_prototypes)

# 提取分类特征的嵌入权重
emb_layers = {
    'Area': FNN_Cat_Emb_Proto.get_layer('emb_Area'),
    'VehBrand': FNN_Cat_Emb_Proto.get_layer('emb_VehBrand'),
    'Region': FNN_Cat_Emb_Proto.get_layer('emb_Region')
}

# 提取嵌入层的权重
emb_weights = {name: layer.get_weights()[0] for name, layer in emb_layers.items()}

# 特征名称列表
feature_names = [
    "VehPower", 
    "VehAge", 
    "DrivAge", 
    "BonusMalus", 
    "VehGas", 
    "Density", 
    "Area", 
    "VehBrand", 
    "Region"
]

# 将数值特征和分类特征的原型参数合并
all_prototypes = []
for feature_name in feature_names:
    if feature_name in emb_weights:
        # 展平嵌入权重
        emb_weights_flattened = emb_weights[feature_name].flatten()
        # 确保每个特征的原型参数形状一致
        all_prototypes.append(emb_weights_flattened)
    else:
        all_prototypes.append(prototypes[feature_names.index(feature_name)])

# 确保所有特征的原型参数具有相同的形状
max_length = max(len(proto) for proto in all_prototypes)
all_prototypes_padded = [np.pad(proto, (0, max_length - len(proto)), mode='constant') for proto in all_prototypes]

# 转换为 NumPy 数组
all_prototypes_array = np.array(all_prototypes_padded)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(all_prototypes_array, 
            cmap='viridis', 
            xticklabels=np.arange(max_length), 
            yticklabels=feature_names,
            cbar_kws={'label': 'Prototype Value'})

plt.title("Learned Prototype Values for All Features", fontsize=16)
plt.xlabel("Prototype Index", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.show()


# 选择要分析的特征
feature_index = 2 
feature_name = feature_names[feature_index]  # 特征名称

# 提取特征矩阵
all_feature_values_normalized = data_nn_cat_emb_learn_train[0][:, :len(nr_col)]  
all_feature_values = prep_standardscaler.inverse_transform(all_feature_values_normalized)

feature_values = all_feature_values[:, feature_index]

# 提取形状函数值
def extract_shape_functions(model, data, feature_index):
    """
    提取指定特征的形状函数值。
    """
    # 创建一个辅助模型，输出 ProtoLayer 的结果
    input_layer = model.input
    proto_layer_output = model.get_layer('proto_layer').output
    aux_model = Model(inputs=input_layer, outputs=proto_layer_output)
    
    # 提取 ProtoLayer 的输出
    proto_output = aux_model.predict(data)
    
    # 提取指定特征的形状函数值
    shape_function_values = proto_output[:, feature_index]
    
    return shape_function_values

# 绘制形状函数图
def plot_shape_function(feature_values, shape_function_values, feature_name):
    """
    绘制形状函数图。
    """
    plt.figure(figsize=(10, 6))
    
    # 计算特征值的密度
    density, bins = np.histogram(feature_values, bins=20)
    density = density / np.max(density)  # 归一化密度值

    # 绘制形状函数值（折线图）
    sorted_indices = np.argsort(feature_values)  # 按特征值排序
    plt.plot(feature_values[sorted_indices], shape_function_values[sorted_indices], '-', alpha=0.7, label='Shape Function')

    # 获取当前 y 轴的范围
    y_min = np.min(shape_function_values) - 0.1 * np.abs(np.min(shape_function_values))
    y_max = np.max(shape_function_values) + 0.1 * np.abs(np.max(shape_function_values))

    for i in range(len(bins) - 1):
        x_start = bins[i]
        x_end = bins[i + 1]
        alpha = density[i]
        rect = patches.Rectangle(
            (x_start, y_min),
            x_end - x_start,
            y_max - y_min,
            linewidth=0.01,
            edgecolor=[0.9, 0.5, 0.5],
            facecolor=[0.9, 0.5, 0.5],
            alpha=alpha
        )
        plt.gca().add_patch(rect)

    plt.title(f'Shape Function for {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Contribution to Prediction')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)  # 设置 y 轴范围
    plt.show()

# 提取形状函数值
shape_function_values = extract_shape_functions(FNN_Cat_Emb_Proto, data_nn_cat_emb_learn_train, feature_index)

# 绘制形状函数图
plot_shape_function(feature_values, shape_function_values, feature_name)


# 提取 ProtoLayer 中的 a 和 b 参数
proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
a_values = proto_layer.a.numpy()  # a_values 的形状为 (num_features, num_prototypes)
b_values = proto_layer.b.numpy()  # b_values 的形状为 (num_features, num_prototypes)

# 可视化 a 和 b 权重
plt.figure(figsize=(12, 6))

# 绘制 a 权重热力图
plt.subplot(1, 2, 1)
plt.imshow(a_values, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Linear Parameter a\n(num_features x num_prototypes)')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)  # 设置纵坐标为特征名
plt.xlabel("Prototypes")

# 绘制 b 权重热力图
plt.subplot(1, 2, 2)
plt.imshow(b_values, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Linear Parameter b \n(num_features x num_prototypes)')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)  # 设置纵坐标为特征名
plt.xlabel("Prototypes")

plt.tight_layout()
plt.show()

# 获取原型层的原型
proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
prototypes = proto_layer.prototypes.numpy()

# 绘制原型的热力图
plt.figure(figsize=(10, 8))
sns.heatmap(prototypes, annot=False, fmt=".2f", cmap='viridis', xticklabels=False, yticklabels=feature_names)
plt.title("Prototypes Learned by ProtoLayer")
plt.xlabel("Prototypes")
plt.ylabel("Features")
plt.show()


import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.models import Model

# 定义预测函数
def predict_wrapper(x):
    """
    This function takes a sample and splits it into the correct format for the model,
    then makes predictions using the trained model.
    """
    # Ensure x has 9 features
    num_features = x.shape[1]
    if num_features != 9:
        raise ValueError(f"Input data has {num_features} features, but the model expects 9 features.")
    
    # Split x into the components required by the model
    input_matrix_just_nr = x[:, :6]  # Numerical features (first 6 columns)
    input_emb_area = x[:, 6].reshape(-1, 1)  # Area (7th column)
    input_emb_vehbrand = x[:, 7].reshape(-1, 1)  # VehBrand (8th column)
    input_emb_region = x[:, 8].reshape(-1, 1)  # Region (9th column)
    
    # Exposure (assumed to be 1 for each sample)
    input_exposure = np.ones((x.shape[0], 1))
    
    # Define embedding dimension
    emb_dim = 1  # Embedding dimension

    # Embed categorical features using the model's embedding layers
    emb_area = FNN_Cat_Emb_Proto.get_layer('emb_Area')(input_emb_area)
    emb_vehbrand = FNN_Cat_Emb_Proto.get_layer('emb_VehBrand')(input_emb_vehbrand)
    emb_region = FNN_Cat_Emb_Proto.get_layer('emb_Region')(input_emb_region)
    
    # Flatten the embedded features
    emb_area = tf.keras.layers.Reshape((emb_dim,))(emb_area)
    emb_vehbrand = tf.keras.layers.Reshape((emb_dim,))(emb_vehbrand)
    emb_region = tf.keras.layers.Reshape((emb_dim,))(emb_region)
    
    # Predict using the model
    return FNN_Cat_Emb_Proto.predict([input_matrix_just_nr, emb_area, emb_vehbrand, emb_region, input_exposure])

# 准备背景数据集
# 1. 准备数据
background_data_nn_cat_emb_learn_train = [
    data_nn_cat_emb_learn_train[0][:100],  # 数值特征（前 6 列）
    data_nn_cat_emb_learn_train[1][:100].reshape(-1, 1),  # Area
    data_nn_cat_emb_learn_train[2][:100].reshape(-1, 1),  # VehBrand
    data_nn_cat_emb_learn_train[3][:100].reshape(-1, 1)  # Region
]

# 将所有特征（数值 + 分类）合并到一个矩阵中
background_features = np.concatenate(background_data_nn_cat_emb_learn_train, axis=1)

# 创建 SHAP Explainer
explainer = shap.KernelExplainer(predict_wrapper, background_features)

# 1. 准备数据
test_data_nn_cat_emb_learn_test = [
    data_nn_cat_emb_learn_test[0],  # 数值特征（前 6 列）
    data_nn_cat_emb_learn_test[1].reshape(-1, 1),  # Area
    data_nn_cat_emb_learn_test[2].reshape(-1, 1),  # VehBrand
    data_nn_cat_emb_learn_test[3].reshape(-1, 1)  # Region
]

# 将所有特征（数值 + 分类）合并到一个矩阵中
test_features = np.concatenate(test_data_nn_cat_emb_learn_test, axis=1)

# 计算测试集的 SHAP 值
shap_values = explainer.shap_values(test_features[:10000])

# 检查 SHAP 值的形状
print("SHAP values shape:", shap_values.shape)

# 如果 SHAP 值是三维数组，去掉最后一个维度
if len(shap_values.shape) == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values[:, :, 0]

# 检查处理后的 SHAP 值的形状
print("Processed SHAP values shape:", shap_values.shape)

# 可视化整个数据集的 SHAP 值
shap.summary_plot(shap_values, test_features[:10000], feature_names=["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehGas", "Density", "Area", "VehBrand", "Region"])

import numpy as np
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer

# 1. 准备数据
data_nn_cat_emb_learn_train[1] = data_nn_cat_emb_learn_train[1].reshape(-1, 1)  # Area
data_nn_cat_emb_learn_train[2] = data_nn_cat_emb_learn_train[2].reshape(-1, 1)  # VehBrand
data_nn_cat_emb_learn_train[3] = data_nn_cat_emb_learn_train[3].reshape(-1, 1)  # Region

# 2. 反向标准化函数
def reverse_standardization(z, mean, std):
    return z * std + mean

# 3. 准备原始特征数据，数值特征并进行反向标准化
numeric_features_original = np.column_stack([
    reverse_standardization(data_nn_cat_emb_learn_train[0][:, i], means[i], stds[i])
    for i in range(6)
])

# 所有特征
features_original = np.concatenate([
    numeric_features_original,  # 原始数值特征（前 6 列）
    data_nn_cat_emb_learn_train[1],  # Area（分类特征）
    data_nn_cat_emb_learn_train[2],  # VehBrand（分类特征）
    data_nn_cat_emb_learn_train[3],  # Region（分类特征）
], axis=1)  # 特征矩阵的形状应为 (num_samples, 9)

# 4. 定义目标标签
labels = y_true_learn / exposure["train"]  # 确保这是训练时使用的目标

# 5. 定义LIME解释器，使用原始特征数据
explainer = LimeTabularExplainer(
    training_data=features_original,  # 使用原始特征数据
    training_labels=labels,
    mode="regression",
    feature_names=["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehGas", "Density", "Area", "VehBrand", "Region"],
    class_names=["claim_frequency"],
    discretize_continuous=True,
    categorical_features=[6, 7, 8]  # 分类特征的索引
)

# 6. 定义预测包装函数，需要处理原始输入
def predict_wrapper(x):
    """
    该函数接受原始特征值，将其标准化后再传入模型进行预测
    """
    # 确保x有9个特征
    num_features = x.shape[1]
    if num_features != 9:
        raise ValueError(f"输入数据有 {num_features} 个特征，但模型期望9个特征。")
    
    # 提取数值特征并进行标准化（与训练时保持一致）
    numeric_features = x[:, :6].copy()
    for i in range(6):
        numeric_features[:, i] = (numeric_features[:, i] - means[i]) / stds[i]
    
    # 拆分特征
    input_matrix_just_nr = numeric_features  # 标准化后的数值特征
    input_emb_area = x[:, 6].reshape(-1, 1).astype(np.int32)  # Area（第7列）
    input_emb_vehbrand = x[:, 7].reshape(-1, 1).astype(np.int32)  # VehBrand（第8列）
    input_emb_region = x[:, 8].reshape(-1, 1).astype(np.int32)  # Region（第9列）
    
    # 暴露量
    input_exposure = np.ones((x.shape[0], 1))
    
    # 定义嵌入维度
    emb_dim = 1  # 嵌入维度
    
    # 使用模型的嵌入层嵌入分类特征
    emb_area = FNN_Cat_Emb_Proto.get_layer('emb_Area')(input_emb_area)
    emb_vehbrand = FNN_Cat_Emb_Proto.get_layer('emb_VehBrand')(input_emb_vehbrand)
    emb_region = FNN_Cat_Emb_Proto.get_layer('emb_Region')(input_emb_region)
    
    # 展平嵌入特征
    emb_area = tf.keras.layers.Reshape((emb_dim,))(emb_area)
    emb_vehbrand = tf.keras.layers.Reshape((emb_dim,))(emb_vehbrand)
    emb_region = tf.keras.layers.Reshape((emb_dim,))(emb_region)
    
    # 使用模型进行预测
    return FNN_Cat_Emb_Proto.predict([input_matrix_just_nr, emb_area, emb_vehbrand, emb_region, input_exposure])

# 7. 选择一个样本进行解释
sample_index = 10000 
sample_data = features_original[sample_index].reshape(1, -1)  # 使用原始特征数据

# 8. 使用LIME解释该样本的预测
explanation = explainer.explain_instance(
    data_row=sample_data[0],  # 选定的样本（展平为1D数组）
    predict_fn=predict_wrapper,  # 预测函数
    num_features=9  # 要在解释中显示的特征数量
)

# 9. 显示解释
explanation.show_in_notebook()

# 获取解释结果
fig = explanation.as_pyplot_figure()
fig.suptitle('LIME Explanation')  # 设置整个图的标题
plt.tight_layout()

# 保存为图像
plt.savefig('lime_explanation.png', dpi=300)
plt.close(fig)

