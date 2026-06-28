import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Activation
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model

# 第一部分：原始一维 CNN 模型代码（添加 Grad-CAM 支持）
# ====================================================

# 读取数据（假设数据格式与原始代码一致）
df = pd.read_csv('E:/xiazai/Human Mortality Database/Chinese/male.csv')
df['mx'] = np.log(df['mx'])

# 分割训练集和测试集
train_df = df[df['year'] < 2012]  
test_df = df[df['year'] >= 2012]

# 转换为矩阵
train_mat = pd.pivot_table(train_df, values='mx', index='year', columns='age', fill_value=0).values
test_mat = pd.pivot_table(test_df, values='mx', index='year', columns='age', fill_value=0).values

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_mat_scaled = scaler.fit_transform(train_mat)
test_mat_scaled = scaler.transform(test_mat)

# 创建时间序列数据
def create_dataset(data, time_step=7):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, :])
    return np.array(X), np.array(y)

time_step = 7
X_train, y_train = create_dataset(train_mat_scaled, time_step)
X_test, y_test = create_dataset(test_mat_scaled, time_step)

from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, Flatten, Dense

def build_cnn_model(input_shape, output_units):
    model = Sequential([
        # 修改这里：添加 padding='same' 保持时间步长度不变
        Conv1D(64, kernel_size=3, activation='relu', 
               input_shape=input_shape, padding='same', name='last_conv1d'),
        # 可以保留或移除池化层
        # 如果保留，需要确保池化后长度仍为7 (pool_size=1 或使用 stride=1)
        AveragePooling1D(pool_size=1),  # 修改为不减少时间步
        Flatten(),
        Dense(output_units, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 模型参数
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps=7, features=年龄数量)
output_units = y_train.shape[1]  # 输出年龄数量

# 重新构建和训练模型
model = build_cnn_model(input_shape, output_units)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 第二部分：Grad-CAM 实现（针对一维 CNN）
# ====================================================
class GradCAM:
    def __init__(self, model, layer_name='last_conv1d'):
        self.model = model
        self.layer = model.get_layer(layer_name)
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[self.layer.output, model.output]
        )

    def compute_heatmap(self, input_sample, class_idx=None):
        # 转换为 Tensor
        input_tensor = tf.convert_to_tensor(input_sample[np.newaxis, ...])
        
        # 计算梯度
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            else:
                class_idx = tf.constant(class_idx, dtype=tf.int64)  # 确保为 Tensor
            loss = predictions[:, class_idx]

        # 获取梯度
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("梯度计算失败！请检查模型结构。")
        
        # 计算权重（全局平均梯度）
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # 沿时间和特征维度平均
        
        # 计算加权激活
        conv_outputs = conv_outputs[0]  # 去除 batch 维度
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # 归一化处理（使用 TensorFlow 操作）
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        # 转换为 numpy 并返回
        return heatmap.numpy(), class_idx.numpy()
    
    # grad_cam_diagnosis.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def diagnose_gradcam(model, input_sample, class_idx):
    # 1. 检查输入数据
    print("\n=== 输入数据检查 ===")
    print("输入样本形状:", input_sample.shape)
    print("时间步均值:", np.mean(input_sample, axis=1))
    
    # 2. 计算原始热力图
    grad_cam = GradCAM(model)
    heatmap, _ = grad_cam.compute_heatmap(input_sample, class_idx)
    print("\n=== 热力图数值 ===")
    print("热力图:", heatmap.round(3))
    
    # 3. 检查梯度
    last_conv_layer = model.get_layer('last_conv1d')
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_sample[np.newaxis, ...])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    print("\n=== 梯度统计 ===")
    print("梯度均值:", np.mean(grads.numpy()))
    print("梯度最大值:", np.max(grads.numpy()))
    
    # 4. 可视化诊断
    plt.figure(figsize=(15, 4))
    

    
    # 热力图可视化
    plt.subplot(132)
    plt.plot(heatmap, marker='o')
    plt.title("时间步重要性曲线")
    plt.xlabel("时间步")
    plt.grid(True)
    plt.savefig("时间步重要性曲线")
    
    # 扰动分析
    plt.subplot(133)
    original_pred = model.predict(input_sample[np.newaxis, ...])[0, class_idx]
    perturbations = []
    for t in range(input_sample.shape[0]):
        perturbed = input_sample.copy()
        perturbed[t, :] += 0.5
        perturbed_pred = model.predict(perturbed[np.newaxis, ...])[0, class_idx]
        perturbations.append(perturbed_pred - original_pred)
    plt.bar(range(len(perturbations)), perturbations)
    plt.title("时间步扰动影响")
    plt.xlabel("时间步")
    plt.ylabel("预测值变化")
    plt.savefig("时间步扰动影响1")
    
    plt.show()

# 使用示例
sample_idx = 0
input_sample = X_test[sample_idx]
diagnose_gradcam(model, input_sample, class_idx=80)

# 修改后的 Grad-CAM 实现（针对整体平均预测）
class GlobalGradCAM:
    def __init__(self, model, layer_name='last_conv1d'):
        self.model = model
        self.layer = model.get_layer(layer_name)
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[self.layer.output, model.output]
        )

    def compute_heatmap(self, input_sample):
        # 转换为 Tensor
        input_tensor = tf.convert_to_tensor(input_sample[np.newaxis, ...])
        
        # 计算梯度 - 修改为对所有年龄预测取平均
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_tensor)
            # 使用所有年龄预测的平均值作为损失
            loss = tf.reduce_mean(predictions)

        # 获取梯度
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("梯度计算失败！请检查模型结构。")
        
        # 计算权重（全局平均梯度）
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # 沿时间和特征维度平均
        
        # 计算加权激活
        conv_outputs = conv_outputs[0]  # 去除 batch 维度
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # 归一化处理（使用 TensorFlow 操作）
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        # 转换为 numpy 并返回
        return heatmap.numpy()
    
    # 修改后的诊断函数（针对整体平均预测）
def diagnose_global_gradcam(model, input_sample):
    # 1. 检查输入数据
    print("\n=== 输入数据检查 ===")
    print("输入样本形状:", input_sample.shape)
    print("时间步均值:", np.mean(input_sample, axis=1))
    
    # 2. 计算全局热力图
    global_grad_cam = GlobalGradCAM(model)
    heatmap = global_grad_cam.compute_heatmap(input_sample)
    print("\n=== 热力图数值 ===")
    print("热力图:", heatmap.round(3))
    
    # 3. 检查梯度
    last_conv_layer = model.get_layer('last_conv1d')
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_sample[np.newaxis, ...])
        # 使用所有年龄预测的平均值作为损失
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    print("\n=== 梯度统计 ===")
    print("梯度均值:", np.mean(grads.numpy()))
    print("梯度最大值:", np.max(grads.numpy()))
    
    # 4. 可视化诊断
    plt.figure(figsize=(15, 4))
    
    # 输入数据可视化
    ax1 = plt.subplot(131)
    im = ax1.imshow(input_sample.T, aspect='auto', cmap='viridis')
    plt.title("输入样本（log死亡率）")
    plt.xlabel("时间步")
    plt.ylabel("年龄")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # 热力图可视化
    plt.subplot(132)
    plt.plot(heatmap, marker='o')
    plt.title("时间步重要性曲线 ")
    plt.xlabel("时间步")
    plt.grid(True)
    
    # 扰动分析 - 修改为对整体平均预测的影响
    plt.subplot(133)
    original_pred = model.predict(input_sample[np.newaxis, ...])
    original_avg_pred = np.mean(original_pred[0])  # 计算所有年龄预测的平均值
    perturbations = []
    for t in range(input_sample.shape[0]):
        perturbed = input_sample.copy()
        perturbed[t, :] += 0.5
        perturbed_pred = model.predict(perturbed[np.newaxis, ...])
        perturbed_avg_pred = np.mean(perturbed_pred[0])  # 计算扰动后的平均值
        perturbations.append(perturbed_avg_pred - original_avg_pred)
    plt.bar(range(len(perturbations)), perturbations)
    plt.title("时间步扰动对整体预测的影响")
    plt.xlabel("时间步")
    plt.ylabel("平均预测值变化")
    
    plt.tight_layout()
    plt.show()
    
    return heatmap, perturbations

# 使用修改后的代码
sample_idx = 2
input_sample = X_test[sample_idx]

# 计算整体平均预测的热力图
heatmap, perturbations = diagnose_global_gradcam(model, input_sample)

# 单独绘制热力图
plt.figure(figsize=(10, 4))
plt.plot(heatmap, color='r', linewidth=2)
plt.title("Grad-CAM 热力图 (整体平均预测)")
plt.xlabel("历史时间步 (年)")
plt.ylabel("重要性权重")
plt.xticks(range(7), labels=[f"t-{6-i}" for i in range(7)])
plt.grid(True)
plt.savefig("全年龄预测影响")
plt.show()

