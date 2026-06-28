import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Embedding, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import os

# Create the dataframes needed for evaluation:
data_nn_cat_emb_learn, y_true_learn = create_ffn_cat_emb_data(bool_in_learn)
data_nn_cat_emb_test, y_true_test = create_ffn_cat_emb_data(bool_in_test)

# 定义一个函数来创建模型
def Create_Poisson_FNN_Cat_Emb(input_dim, emb_dim, cat_vocabulary):
    # 输入层
    Input_Matrix_Just_NR = Input(shape=(input_dim,), dtype='float32', name='Input_Matrix_Just_NR')
    Input_Exposure = Input(shape=(1,), dtype='float32', name='Input_Exposure')
    Input_EMB_VehBrand = Input(shape=(1,), dtype='int32', name='Input_EMB_VehBrand')
    Input_EMB_Region = Input(shape=(1,), dtype='int32', name='Input_EMB_Region')
    
    # 嵌入层
    emb_VehBrand = Embedding(input_dim=len(cat_vocabulary["Cat_VehBrand"]), output_dim=emb_dim, input_length=1, name='emb_VehBrand')(Input_EMB_VehBrand)
    emb_Region = Embedding(input_dim=len(cat_vocabulary["Cat_Region"]), output_dim=emb_dim, input_length=1, name='emb_Region')(Input_EMB_Region)
    
    # 将嵌入层的输出展平
    emb_VehBrand = Reshape((emb_dim,))(emb_VehBrand)
    emb_Region = Reshape((emb_dim,))(emb_Region)
    
    # 将数值特征和嵌入特征合并
    combined = Concatenate(name='combined')([Input_Matrix_Just_NR, emb_VehBrand, emb_Region])
    
    # 隐藏层
    hidden1 = Dense(20, activation='relu', name='hidden1')(combined)
    hidden2 = Dense(15, activation='relu', name='hidden2')(hidden1)
    hidden3 = Dense(10, activation='relu', name='hidden3')(hidden2)
    
    # 输出层
    Result_FNN1 = Dense(1, activation='exponential', name='Result_FFN1', trainable=True)(hidden3)
    
    # 将输出与暴露量相乘
    Response = Multiply(name='Result')([Result_FNN1, Input_Exposure])
    
    # 定义并返回模型
    model = Model(inputs=[Input_Matrix_Just_NR, Input_EMB_VehBrand, Input_EMB_Region, Input_Exposure], 
                  outputs=[Response], name='Poisson_FNN_Cat_Emb')
    
    return model

# 主循环
for run_index in range(15):
    print(f"Model: {run_index}")
    
    # Create the dataframes needed for training:
    data_nn_cat_emb_learn_train, y_true_learn_train = create_ffn_cat_emb_data(train_val_split[f"learn_train_{run_index}"])
    data_nn_cat_emb_learn_val, y_true_learn_val = create_ffn_cat_emb_data(train_val_split[f"learn_val_{run_index}"])
    
    # 设置随机种子
    tf.random.set_seed(random_seeds[run_index])
    np.random.seed(random_seeds[run_index])
    
    # 创建模型
    FNN_Cat_Emb = Create_Poisson_FNN_Cat_Emb(input_dim=len(nr_col), emb_dim=1, cat_vocabulary=cat_vocabulary)
    
    # 编译模型
    FNN_Cat_Emb.compile(optimizer='adam', loss=poisson_loss_for_tf, metrics=[poisson_loss_for_tf])
    
    # 模型回调
    early_stopping_callback = EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)
    
    # 模型训练
    start_time = time.time()
    epochs_Cat_Emb = 500
    FNN_Cat_Emb_history = FNN_Cat_Emb.fit(
        x=data_nn_cat_emb_learn_train,
        y=y_true_learn_train,
        validation_data=(data_nn_cat_emb_learn_val, y_true_learn_val),
        epochs=epochs_Cat_Emb,
        batch_size=5000,
        verbose=2,
        callbacks=[early_stopping_callback]
    )
    end_time = time.time()
    execution_time_nn_cat_emb = end_time - start_time
    best_epoch_FNN_cat_emb = np.argmin(FNN_Cat_Emb_history.history['val_loss']) + 1
    
    # 保存模型权重
    FNN_Cat_Emb.save_weights(f'{storage_path}/saved_models/Poisson_FNN_Cat_Emb_{run_index}.weights.h5')
    
    # 加载模型权重
    FNN_Cat_Emb.load_weights(f'{storage_path}/saved_models/Poisson_FNN_Cat_Emb_{run_index}.weights.h5')
    
    # 预测
    y_pred["train"][f"FNN_Cat_Emb_{run_index}"] = np.array([x for [x] in FNN_Cat_Emb.predict(data_nn_cat_emb_learn, batch_size=100000)])
    y_pred["test"][f"FNN_Cat_Emb_{run_index}"] = np.array([x for [x] in FNN_Cat_Emb.predict(data_nn_cat_emb_test, batch_size=100000)])
    
    # 评估模型
    FNN_Cat_Emb_results = Results(
        model=f"FNN_Cat_Emb (run: {run_index})",
        epochs=best_epoch_FNN_cat_emb,
        run_time=execution_time_nn_cat_emb,
        nr_parameters=np.sum([np.prod(v.shape.as_list()) for v in FNN_Cat_Emb.trainable_weights]),
        poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"][f"FNN_Cat_Emb_{run_index}"]),
        poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"][f"FNN_Cat_Emb_{run_index}"]),
        pred_avg_freq_train=y_pred["train"][f"FNN_Cat_Emb_{run_index}"].sum() / exposure["train"].sum(),
        pred_avg_freq_test=y_pred["test"][f"FNN_Cat_Emb_{run_index}"].sum() / exposure["test"].sum()
    )
    
    # 将结果存入 DataFrame
    store_results_in_df(FNN_Cat_Emb_results)

# 显示结果
display(df_results)

# 清理内存
del data_nn_cat_emb_learn, data_nn_cat_emb_test, data_nn_cat_emb_learn_train, data_nn_cat_emb_learn_val