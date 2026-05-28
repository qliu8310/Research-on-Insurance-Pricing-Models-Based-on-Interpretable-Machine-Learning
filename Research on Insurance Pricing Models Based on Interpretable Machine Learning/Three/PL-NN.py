from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Embedding, Reshape, Concatenate, Layer, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import os

# 原型层定义
class ProtoLayer(Layer):
    def __init__(self, num_prototypes=16, **kwargs):
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

# 动态sigma调整器
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

# 定义模型创建函数
def Create_Poisson_FNN_Cat_Emb_Proto(input_dim, emb_dim, cat_vocabulary, num_prototypes=32):
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
    
    # 添加 ProtoLayer
    proto_output = ProtoLayer(num_prototypes=num_prototypes, name='proto_layer')(combined)
    
    # 隐藏层
    hidden1 = Dense(20, activation='relu', name='hidden1')(proto_output)
    hidden2 = Dense(15, activation='relu', name='hidden2')(hidden1)
    hidden3 = Dense(10, activation='relu', name='hidden3')(hidden2)
    
    # 输出层
    Result_FNN1 = Dense(1, activation='exponential', name='Result_FFN1', trainable=True)(hidden3)
    
    # 将输出与暴露量相乘
    Response = Multiply(name='Result')([Result_FNN1, Input_Exposure])
    
    # 定义并返回模型
    model = Model(inputs=[Input_Matrix_Just_NR, Input_EMB_Area, Input_EMB_VehBrand, Input_EMB_Region, Input_Exposure], 
                  outputs=[Response], name='Poisson_FNN_Cat_Emb_Proto')
    
    return model


# 主循环
for run_index in range(15):
    print(f"Model: {run_index}")
    # 定义嵌入维度
    emb_dim = 1 
    # 准备数据
    data_nn_cat_emb_learn, y_true_learn = create_ffn_cat_emb_data(bool_in_learn)
    data_nn_cat_emb_learn_test, y_true_test = create_ffn_cat_emb_data(bool_in_test)
    
    # 创建训练和验证数据
    data_nn_cat_emb_learn_train, y_true_learn_train = create_ffn_cat_emb_data(train_val_split[f"learn_train_{run_index}"])
    data_nn_cat_emb_learn_val, y_true_learn_val = create_ffn_cat_emb_data(train_val_split[f"learn_val_{run_index}"])
    
    # 设置随机种子
    tf.random.set_seed(random_seeds[run_index])
    np.random.seed(random_seeds[run_index])
    
    # 创建模型
    FNN_Cat_Emb_Proto = Create_Poisson_FNN_Cat_Emb_Proto(input_dim=len(nr_col), emb_dim=1, cat_vocabulary=cat_vocabulary, num_prototypes=16)
    
    # 获取数值特征和分类特征
    X_nn_just_nr_train, Input_EMB_Area_train, Input_EMB_VehBrand_train, Input_EMB_Region_train, exposure_train = data_nn_cat_emb_learn_train
    X_nn_just_nr_val, Input_EMB_Area_val, Input_EMB_VehBrand_val, Input_EMB_Region_val, exposure_val = data_nn_cat_emb_learn_val
    
    # 获取嵌入层
    emb_layer_Area = FNN_Cat_Emb_Proto.get_layer('emb_Area')
    emb_layer_VehBrand = FNN_Cat_Emb_Proto.get_layer('emb_VehBrand')
    emb_layer_Region = FNN_Cat_Emb_Proto.get_layer('emb_Region')
    
    # 将分类特征的索引值转换为嵌入向量
    emb_Area_train = emb_layer_Area(Input_EMB_Area_train).numpy().reshape(-1, emb_dim)
    emb_VehBrand_train = emb_layer_VehBrand(Input_EMB_VehBrand_train).numpy().reshape(-1, emb_dim)
    emb_Region_train = emb_layer_Region(Input_EMB_Region_train).numpy().reshape(-1, emb_dim)
    
    # 合并数值特征和嵌入特征
    X_train_combined = np.concatenate([X_nn_just_nr_train, emb_Area_train, emb_VehBrand_train, emb_Region_train], axis=1)
    
    # 初始化原型参数
    proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
    proto_layer.set_prototypes_from_quantiles(X_train_combined)
    
    
    # 编译模型
    FNN_Cat_Emb_Proto.compile(optimizer='adam', loss=poisson_loss_for_tf, metrics=[poisson_loss_for_tf])
    
    # 模型回调
    early_stopping_callback = EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)
    
    # 动态sigma调整器
    sigma_scheduler = SigmaScheduler(proto_layer, total_epochs=100, tau=16)
    
    # 模型训练
    start_time = time.time()
    FNN_Cat_Emb_Proto_history = FNN_Cat_Emb_Proto.fit(
        x=data_nn_cat_emb_learn_train,
        y=y_true_learn_train,
        validation_data=(data_nn_cat_emb_learn_val, y_true_learn_val),
        epochs=100,
        batch_size=5000,
        verbose=2,
        callbacks=[early_stopping_callback, sigma_scheduler]
    )
    end_time = time.time()
    
    # 计算训练时间和最佳epoch
    execution_time_nn_cat_emb_proto = end_time - start_time
    best_epoch_FNN_cat_emb_proto = np.argmin(FNN_Cat_Emb_Proto_history.history['val_loss']) + 1
    
    # 保存模型权重
    FNN_Cat_Emb_Proto.save_weights(f'{storage_path}/saved_models/Emb_FNN_Proto{run_index}.weights.h5')
    
    # 加载模型权重
    FNN_Cat_Emb_Proto.load_weights(f'{storage_path}/saved_models/Emb_FNN_Proto{run_index}.weights.h5')

    proto_layer = FNN_Cat_Emb_Proto.get_layer('proto_layer')
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
    
    # 清理内存
    del data_nn_cat_emb_learn_train, data_nn_cat_emb_learn_val
   


data_nn_cat_emb_learn, y_true_learn = create_ffn_cat_emb_data(bool_in_learn)
data_nn_cat_emb_learn_test, y_true_test = create_ffn_cat_emb_data(bool_in_test)
    
# Create the dataframes needed for training:
data_nn_cat_emb_learn_train, y_true_learn_train = create_ffn_cat_emb_data(train_val_split[f"learn_train_{run_index}"])
data_nn_cat_emb_learn_val, y_true_learn_val = create_ffn_cat_emb_data(train_val_split[f"learn_val_{run_index}"])

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import matplotlib.patches as patches

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

# 选择要分析的特征
feature_index = 2 
feature_name = feature_names[feature_index]  # 特征名称

# 提取整个数值特征矩阵
all_feature_values_normalized = data_nn_cat_emb_learn_train[0][:, :len(nr_col)]  

# 对数值特征列进行反归一化
all_feature_values = prep_standardscaler.inverse_transform(all_feature_values_normalized)

# 提取你感兴趣的特征列
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

# 获取原型层的线性参数 a 和 b
a_weights = proto_layer.a.numpy()
b_weights = proto_layer.b.numpy()

# 可视化 a 和 b 权重
plt.figure(figsize=(12, 6))

# 绘制 a 权重
plt.subplot(1, 2, 1)
plt.imshow(a_weights, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Linear Parameter a Weights (num_features x num_prototypes)')

# 绘制 b 权重
plt.subplot(1, 2, 2)
plt.imshow(b_weights, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Linear Parameter b Weights (num_features x num_prototypes)')

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

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

