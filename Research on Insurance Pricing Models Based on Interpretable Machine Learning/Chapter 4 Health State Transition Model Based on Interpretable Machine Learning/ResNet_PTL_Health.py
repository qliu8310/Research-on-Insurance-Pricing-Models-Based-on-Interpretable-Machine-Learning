import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.activations import exponential
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.titlesize"] = 20  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 15  
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_3.csv')

# 确定训练集和测试集
train = df.loc[df['time'].isin([1, 3, 5.5, 8.5, 11.5])]
test = df.loc[df['time'].isin([14.5, 18])]

# 定义特征和目标变量
Y = 'ny'
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']

# 提取特征和目标变量
X_train = train[X]
y_train = train[Y]
w_train = train['sum_hyear']
X_test = test[X]
y_test = test[Y]
w_test = test['sum_hyear']

# 数值型和分类变量的列
num_features = ['time', 'age']
cat_features = ['sex', 'residenc', 'marry', 'smoke', 'drink']

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)])

# 对训练集和测试集进行预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# 处理后的特征维度
input_shape = X_train_processed.shape[1]
# input_shape = X_train.shape[1]


# 构建ResNet块

#I型ResNet块
def type_i_block(input_tensor, dimension):
    # 主路径  
    x = Dense(dimension)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = Dense(dimension)(x)
    x = BatchNormalization()(x)
    # 快捷路径（直接传递）
    x = Add()([input_tensor, x])
    x = Activation('selu')(x)
    return x

#II型ResNet块
def type_ii_block(input_tensor,dimension):
    # 主路径  
    x = Dense(dimension)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = Dense(dimension)(x)
    x = BatchNormalization()(x)
     # 快捷路径
    shortcut = Dense(dimension)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([shortcut, x])
    x = Activation('selu')(x)
    return x

def build_resnet(L, dimension):
    inputs = Input(shape=(dimension,))
    x = type_ii_block(inputs, dimension)
    for _ in range(L - 1):
        x = type_i_block(x, dimension)
    model = Model(inputs, x)
    return model


def build_expectile_nn(L, p, input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(p)(inputs)  # 将输入特征调整到第一个ResNet子网络的维度p
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    
    resnet1 = build_resnet(L, p)(x)
    transition = Dense(p // 2)(resnet1)  # 调整维度到p/2
    resnet2 = build_resnet(L, p // 2)(transition)
    outputs = Dense(1,activation='exponential')(resnet2)
    model = Model(inputs, outputs)
    return model


# 定义模型参数
L = 3
p = 64

# 构建模型
model = build_expectile_nn(L, p, input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='poisson')

# 模型训练
history=model.fit(X_train_processed, y_train, sample_weight=w_train, epochs=100, batch_size=64)
# history=model.fit(X_train, y_train, sample_weight=w_train, epochs=100, batch_size=64)

original_weights=model.get_weights()

# 模型预测
y_pred = model.predict(X_test_processed)
# y_pred = model.predict(X_test)

# 输出预测结果
print(y_pred)

model.summary()

plt.plot(history.history['loss'])

mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse}")
test["predictions"]=y_pred
sample_weight=w_test
y_true=y_test
y_pred=test["predictions"].values
y_true = np.array(y_true)  
y_pred = np.array(y_pred)
deviance_per_obs = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
deviance_per_obs[y_true == 0] =2 * y_pred[y_true == 0]
weighted_sum = tf.reduce_sum(tf.multiply(deviance_per_obs, sample_weight))  
poisson_deviance=weighted_sum/sample_weight.sum() 

poisson_deviance

# 拟合值
# y_fitted = model.predict(X_train)
y_fitted = model.predict(X_train_processed)
train["fitted"] = y_fitted  
# train["y_fitted"] = y_fitted  
# train['fitted']=train["y_fitted"] /train['sum_hyear']
y_true_train=train.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train= train.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train.index, y_true_train, label='True Values',color='blue')
plt.plot(y_fitted_train.index, y_fitted_train, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(H->D)')
plt.xlabel('Age')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)
plt.show()

#预测值
test["predictions"]=y_pred
# ['sum_health1_3']/['sum_year']
y_true2=test.groupby("age")['ny'].mean()   #测试集真实值
y_pred2= test.groupby('age')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true2.index, y_true2, label='True Values')
plt.plot(y_true2.index, y_pred2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(H->D)')
plt.xlabel('Age')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差  
residuals1 = y_true2 - y_pred2

x_axis1 = y_true2.index
  
# 绘制残差图  

plt.scatter(x_axis1,residuals1,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线  
plt.xlabel('age(65-105)')  
plt.ylabel('H->D Residuals')  
plt.title('Residual Plot(H->D)')  
plt.grid(True)  
plt.show()

predictions_list= [train['fitted'], test['predictions']]
df['predictions'] = pd.concat(predictions_list)

y_true3 = df.groupby("t")['ny'].mean()   # Testing set true values
y_pred3 = df.groupby('t')['predictions'].mean()   # Predicted values


plt.scatter(y_true3.index, y_true3, label='True Values')
plt.plot(y_true3.index, y_pred3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->D)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2 = y_true3 - y_pred3
x_axis2 = y_true3.index  
# 绘制残差图  
 
plt.scatter(x_axis2,residuals2,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('H->D Residuals')  
plt.title('Residual Plot(H->D)')  
plt.grid(True)  
plt.show()

predictions_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx')
fitted_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx')
fitted_predictions_1_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx') 

test_reset_index = test.reset_index(drop=True)   
predictions_1_3['res_predictions'] = test_reset_index["predictions"]

fitted_1_3['res_fitted']=train["fitted"]
fitted_predictions_1_3['res_fitted_predictions']=df['predictions']


fitted_predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx', index=False)
predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx', index=False)
fitted_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx', index=False)

df_1_2 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_2.csv')  
train_1_2 = df_1_2 .loc[df_1_2['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_1_2= df_1_2.loc[df_1_2['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_1_2  = train_1_2 [X]
y_train_1_2  = train_1_2 [Y]
w_train_1_2  = train_1_2 ['sum_hyear']
X_test_1_2  = test_1_2 [X]
y_test_1_2  = test_1_2 [Y]
w_test_1_2  = test_1_2 ['sum_hyear']

# 对训练集和测试集进行预处理
X_train_processed_1_2  = preprocessor.fit_transform(X_train_1_2 )
X_test_processed_1_2  = preprocessor.transform(X_test_1_2 )
# 处理后的特征维度
input_shape = X_train_processed_1_2 .shape[1]
# input_shape = X_train_1_2.shape[1]

L = 3
p = 64

# 构建模型
model_1_2 = build_expectile_nn(L, p, input_shape)
model_1_2.compile(optimizer=Adam(learning_rate=0.001), loss='poisson')
model_1_2.set_weights(original_weights)

# 模型训练
history_1_2=model_1_2.fit(X_train_processed_1_2, y_train_1_2, sample_weight=w_train_1_2, epochs=100, batch_size=64)
# history_1_2=model_1_2.fit(X_train_1_2, y_train_1_2, sample_weight=w_train_1_2, epochs=100, batch_size=64)


# 模型预测
y_pred_1_2 = model_1_2.predict(X_test_processed_1_2)
# y_pred_1_2 = model_1_2.predict(X_test_1_2)

# 输出预测结果
print(y_pred_1_2)

plt.plot(history_1_2.history['loss'])

mse_1_2 = mean_squared_error(y_test_1_2,y_pred_1_2)
print(f"Mean Squared Error: {mse_1_2}")

test_1_2["predictions"]=y_pred_1_2
sample_weight_1_2=w_test_1_2
y_true_1_2=y_test_1_2.values
y_pred_1_2=test_1_2["predictions"].values
# y_true_1_2 = np.array(y_true_1_2)  
# y_pred_1_2 = np.array(y_pred_1_2) 
deviance_per_obs_1_2 = 2 * (y_true_1_2 * np.log(y_true_1_2 / y_pred_1_2) - (y_true_1_2 - y_pred_1_2))
deviance_per_obs_1_2[y_true_1_2 == 0] = 2 * y_pred_1_2[y_true_1_2 == 0]
weighted_sum_1_2 = tf.reduce_sum(tf.multiply(deviance_per_obs_1_2, sample_weight_1_2))  
exposure_weighted_average_1_2=weighted_sum_1_2/sample_weight_1_2.sum() 
exposure_weighted_average_1_2

# 对训练集进行预测
# y_fitted_1_2 = model_1_2.predict(X_train_1_2)
y_fitted_1_2 = model_1_2.predict(X_train_processed_1_2)
train_1_2["fitted"] = y_fitted_1_2
y_true_train_1_2=train_1_2.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_1_2= train_1_2.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_1_2.index, y_true_train_1_2, label='True Values',color='blue')
plt.plot(y_fitted_train_1_2.index, y_fitted_train_1_2, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(H->L)')
plt.xlabel('Age(65-105)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.show()

test_1_2["predictions"] = y_pred_1_2
# ['sum_health1_3']/['sum_year']
y_true1_1_2 = test_1_2.groupby("age")['ny'].mean()  # 测试集真实值
y_pred1_1_2 = test_1_2.groupby('age')['predictions'].mean()  # 预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_1_2.index, y_true1_1_2, label='True Values')
plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(H->L)')
plt.xlabel('Age(65-105)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差
residuals1_1_2 = y_true1_1_2 - y_pred1_1_2

x_axis1_1_2 = y_true1_1_2.index

# 绘制残差图

plt.scatter(x_axis1_1_2, residuals1_1_2, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xlabel('age(65-105)')
plt.ylabel('H->L Residuals')
plt.title('Residual Plot(H->L)')
plt.grid(True)
plt.show()

predictions_list_1_2 = [train_1_2['fitted'], test_1_2['predictions']]
df_1_2['predictions'] = pd.concat(predictions_list_1_2)

y_true2_1_2=df_1_2.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_1_2= df_1_2.groupby('t')['predictions'].mean()   #预测值
##绘制按年份分组的转移次数真实值和预测值

plt.scatter(y_true2_1_2.index, y_true2_1_2, label='True Values')
plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->L)')
plt.xlabel('Year(1998-2018)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_1_2 = y_true2_1_2 - y_pred2_1_2

x_axis2_1_2 = y_true2_1_2.index
  
# 绘制残差图  

plt.scatter(x_axis2_1_2,residuals2_1_2, label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('Year(1998-2018)')  
plt.ylabel('H->L Residuals')  
plt.title('Residual Plot(H->L)')  
plt.grid(True)  
plt.show()

predictions_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx')
fitted_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx')
fitted_predictions_1_2 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx') 

test_1_2_reset_index = test_1_2.reset_index(drop=True)   
predictions_1_2['res_predictions'] = test_1_2_reset_index["predictions"]

fitted_1_2['res_fitted']=train_1_2["fitted"]
fitted_predictions_1_2['res_fitted_predictions']=df_1_2['predictions']


fitted_predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx', index=False)
predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx', index=False)
fitted_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx', index=False)

df_2_3 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_3.csv')  
train_2_3 = df_2_3 .loc[df_2_3['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_3= df_2_3.loc[df_2_3['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_2_3  = train_2_3 [X]
y_train_2_3  = train_2_3 [Y]
w_train_2_3  = train_2_3 ['sum_lyear']
X_test_2_3  = test_2_3 [X]
y_test_2_3  = test_2_3 [Y]
w_test_2_3  = test_2_3 ['sum_lyear']

# 对训练集和测试集进行预处理
X_train_processed_2_3  = preprocessor.fit_transform(X_train_2_3 )
X_test_processed_2_3  = preprocessor.transform(X_test_2_3 )
# 处理后的特征维度
input_shape = X_train_processed_2_3 .shape[1]
# input_shape = X_train_2_3 .shape[1]

L = 3
p = 64

# 构建模型
model_2_3 = build_expectile_nn(L, p, input_shape)
model_2_3.compile(optimizer=Adam(learning_rate=0.001), loss='poisson')
model_2_3.set_weights(original_weights)

# 模型训练
history_2_3=model_2_3.fit(X_train_processed_2_3, y_train_2_3, sample_weight=w_train_2_3, epochs=100, batch_size=64)
# history_2_3=model_2_3.fit(X_train_2_3, y_train_2_3, sample_weight=w_train_2_3, epochs=200, batch_size=32)

# 模型预测
y_pred_2_3 = model_2_3.predict(X_test_processed_2_3)
# y_pred_2_3 = model_2_3.predict(X_test_2_3)

# 输出预测结果
print(y_pred_2_3)

plt.plot(history_2_3.history['loss'])

mse_2_3 = mean_squared_error(y_test_2_3,y_pred_2_3)
print(f"Mean Squared Error: {mse_2_3}")

test_2_3["predictions"]=y_pred_2_3
sample_weight_2_3=w_test_2_3
y_true_2_3=y_test_2_3.values
y_pred_2_3=test_2_3["predictions"].values
# y_true_2_3 = np.array(y_true_2_3)  
# y_pred_2_3 = np.array(y_pred_2_3) 
deviance_per_obs_2_3 = 2 * (y_true_2_3 * np.log(y_true_2_3 / y_pred_2_3) - (y_true_2_3 - y_pred_2_3))
deviance_per_obs_2_3[y_true_2_3 == 0] = 2*y_pred_2_3[y_true_2_3 == 0]
weighted_sum_2_3 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_3, sample_weight_2_3))  
exposure_weighted_average_2_3=weighted_sum_2_3/sample_weight_2_3.sum() 
exposure_weighted_average_2_3

#拟合图
# y_fitted_2_3 = model_2_3.predict(X_train_2_3)
y_fitted_2_3 = model_2_3.predict(X_train_processed_2_3)
train_2_3["fitted"] = y_fitted_2_3  
y_true_train_2_3=train_2_3.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_3= train_2_3.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_3.index, y_true_train_2_3, label='True Values',color='blue')
plt.plot(y_fitted_train_2_3.index, y_fitted_train_2_3, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->D)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

#预测图
test_2_3["predictions"]=y_pred_2_3
y_true1_2_3=test_2_3.groupby("age")['ny'].mean()   #测试集真实值
y_pred1_2_3= test_2_3.groupby('age')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_2_3.index, y_true1_2_3, label='True Values')
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(L->D)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差  
residuals1_2_3 = y_true1_2_3 - y_pred1_2_3
x_axis1_2_3 = y_true1_2_3.index
  
# 绘制残差图  
 
plt.scatter(x_axis1_2_3,residuals1_2_3,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xlabel('age(65-105)')  
plt.ylabel('L->D Residuals')  
plt.title('Residual Plot(L->D)')  
plt.grid(True)  
plt.show()

predictions_list_2_3 = [train_2_3['fitted'], test_2_3['predictions']]
df_2_3['predictions'] = pd.concat(predictions_list_2_3)

y_true2_2_3=df_2_3.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_3= df_2_3.groupby('t')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true2_2_3.index, y_true2_2_3, label='True Values')
plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by year(L->D)')
plt.xlabel('year(1998-2018)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_2_3 = y_true2_2_3 - y_pred2_2_3

x_axis2_2_3 = y_true2_2_3.index
  
# 绘制残差图  
 
plt.scatter(x_axis2_2_3,residuals2_2_3,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('L->D Residuals')  
plt.title('Residual Plot(L->D)')  
plt.grid(True)  
plt.show()

predictions_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx')
fitted_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx')
fitted_predictions_2_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx') 

test_2_3_reset_index = test_2_3.reset_index(drop=True)   
predictions_2_3['res_predictions'] = test_2_3_reset_index["predictions"]

fitted_2_3['res_fitted']=train_2_3["fitted"]
fitted_predictions_2_3['res_fitted_predictions']=df_2_3['predictions']


fitted_predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx', index=False)
predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx', index=False)
fitted_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx', index=False)

df_2_1 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_1.csv')  
train_2_1 = df_2_1 .loc[df_2_1['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_1= df_2_1.loc[df_2_1['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_2_1  = train_2_1 [X]
y_train_2_1  = train_2_1 [Y]
w_train_2_1  = train_2_1 ['sum_lyear']
X_test_2_1  = test_2_1 [X]
y_test_2_1  = test_2_1 [Y]
w_test_2_1  = test_2_1 ['sum_lyear']

# 对训练集和测试集进行预处理
X_train_processed_2_1  = preprocessor.fit_transform(X_train_2_1 )
X_test_processed_2_1  = preprocessor.transform(X_test_2_1 )
# 处理后的特征维度
input_shape = X_train_processed_2_1 .shape[1]
# input_shape = X_train_2_1 .shape[1]

L = 3
p = 64

# 构建模型
model_2_1 = build_expectile_nn(L, p, input_shape)
model_2_1.compile(optimizer=Adam(learning_rate=0.001), loss='poisson')
model_2_1.set_weights(original_weights)

# 模型训练
history_2_1=model_2_1.fit(X_train_processed_2_1, y_train_2_1, sample_weight=w_train_2_1, epochs=100, batch_size=64)
# history_2_1=model_2_1.fit(X_train_2_1, y_train_2_1, sample_weight=w_train_2_1, epochs=100, batch_size=32)

# 模型预测
# y_pred_2_1 = model_2_1.predict(X_test_2_1)
y_pred_2_1 = model_2_1.predict(X_test_processed_2_1)

# 输出预测结果
print(y_pred_2_1)

plt.plot(history_2_1.history['loss'])

mse_2_1 = mean_squared_error(y_test_2_1,y_pred_2_1)
print(f"Mean Squared Error: {mse_2_1}")


test_2_1["predictions"]=y_pred_2_1
sample_weight_2_1=w_test_2_1
y_true_2_1=y_test_2_1.values
y_pred_2_1=test_2_1["predictions"].values
# y_true_2_1 = np.array(y_true_2_1)  
# y_pred_2_1 = np.array(y_pred_2_1) 
deviance_per_obs_2_1 = 2 * (y_true_2_1 * np.log(y_true_2_1 / y_pred_2_1) - (y_true_2_1 - y_pred_2_1))
deviance_per_obs_2_1[y_true_2_1 == 0] =2*y_pred_2_1[y_true_2_1 == 0]
weighted_sum_2_1 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_1, sample_weight_2_1))  
exposure_weighted_average_2_1=weighted_sum_2_1/sample_weight_2_1.sum() 
exposure_weighted_average_2_1

# 拟合图
# y_fitted_2_1 = model_2_1.predict(X_train_2_1)
y_fitted_2_1 = model_2_1.predict(X_train_processed_2_1)

train_2_1["fitted"] = y_fitted_2_1  
y_true_train_2_1=train_2_1.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_1= train_2_1.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_1.index, y_true_train_2_1, label='True Values',color='blue')
plt.plot(y_fitted_train_2_1.index, y_fitted_train_2_1, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->H)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_1["predictions"]=y_pred_2_1
y_true1_2_1=test_2_1.groupby("age")['ny'].mean()   
y_pred1_2_1= test_2_1.groupby('age')['predictions'].mean()   
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_2_1.index, y_true1_2_1, label='True Values')
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(L->H)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差  
residuals1_2_1 = y_true1_2_1 - y_pred1_2_1

x_axis1_2_1 = y_true1_2_1.index
  
# 绘制残差图  
 
plt.scatter(x_axis1_2_1,residuals1_2_1,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xlabel('age(65-105)')  
plt.ylabel('L->H Residuals')  
plt.title('Residual Plot(L->H)')  
plt.grid(True)  
plt.show()

predictions_list_2_1 = [train_2_1['fitted'], test_2_1['predictions']]
df_2_1['predictions'] = pd.concat(predictions_list_2_1)

y_true2_2_1=df_2_1.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_1= df_2_1.groupby('t')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true2_2_1.index, y_true2_2_1, label='True Values')
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by year(L->H)')
plt.xlabel('year(1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_2_1 = y_true2_2_1 - y_pred2_2_1
x_axis2_2_1 = y_true2_2_1.index
  
# 绘制残差图  
  
plt.scatter(x_axis2_2_1,residuals2_2_1,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('L->H Residuals')  
plt.title('Residual Plot(L->H)')  
plt.grid(True)  
plt.show()

predictions_2_1= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_1.xlsx')
fitted_2_1= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_1.xlsx')
fitted_predictions_2_1 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_1.xlsx') 

test_2_1_reset_index = test_2_1.reset_index(drop=True)   
predictions_2_1['res_predictions'] = test_2_1_reset_index["predictions"]

fitted_2_1['res_fitted']=train_2_1["fitted"]
fitted_predictions_2_1['res_fitted_predictions']=df_2_1['predictions']


fitted_predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_1.xlsx', index=False)
predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_1.xlsx', index=False)
fitted_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_1.xlsx', index=False)

print(poisson_deviance)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

print(poisson_deviance)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

predictions_list= [train['fitted'], test['predictions']]
df['predictions'] = pd.concat(predictions_list)

y_true3 = df.groupby("t")['ny'].mean()   # Testing set true values
y_pred3 = df.groupby('t')['predictions'].mean()   # Predicted values


plt.scatter(y_true3.index, y_true3, label='True Values')
plt.plot(y_true3.index, y_pred3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->D)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2 = y_true3 - y_pred3
x_axis2 = y_true3.index  
# 绘制残差图  

plt.scatter(x_axis2,residuals2,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('H->D Residuals')  
plt.title('Residual Plot(H->D)')  
plt.grid(True)  
plt.show()


# 拟合值
y_fitted = model.predict(X_train_processed)

train["fitted"] = y_fitted  
# train["y_fitted"] = y_fitted  
# train['fitted']=train["y_fitted"] /train['sum_hyear']
y_true_train=train.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train= train.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train.index, y_true_train, label='True Values',color='blue')
plt.plot(y_fitted_train.index, y_fitted_train, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(H->D)')
plt.xlabel('Age')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)
plt.show()

test_1_2["predictions"] = y_pred_1_2
# ['sum_health1_3']/['sum_year']
y_true1_1_2 = test_1_2.groupby("age")['ny'].mean()  # 测试集真实值
y_pred1_1_2 = test_1_2.groupby('age')['predictions'].mean()  # 预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_1_2.index, y_true1_1_2, label='True Values')
plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(H->L)')
plt.xlabel('Age(65-105)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差
residuals1_1_2 = y_true1_1_2 - y_pred1_1_2

x_axis1_1_2 = y_true1_1_2.index

# 绘制残差图

plt.scatter(x_axis1_1_2, residuals1_1_2, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xlabel('age(65-105)')
plt.ylabel('H->L Residuals')
plt.title('Residual Plot(H->L)')
plt.grid(True)
plt.show()

# 对训练集进行预测
y_fitted_1_2 = model_1_2.predict(X_train_processed_1_2)
train_1_2["fitted"] = y_fitted_1_2
y_true_train_1_2=train_1_2.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_1_2= train_1_2.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_1_2.index, y_true_train_1_2, label='True Values',color='blue')
plt.plot(y_fitted_train_1_2.index, y_fitted_train_1_2, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(H->L)')
plt.xlabel('Age(65-105)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.show()

predictions_list_1_2 = [train_1_2['fitted'], test_1_2['predictions']]
df_1_2['predictions'] = pd.concat(predictions_list_1_2)

y_true2_1_2=df_1_2.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_1_2= df_1_2.groupby('t')['predictions'].mean()   #预测值
##绘制按年份分组的转移次数真实值和预测值

plt.scatter(y_true2_1_2.index, y_true2_1_2, label='True Values')
plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->L)')
plt.xlabel('Year(1998-2018)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_1_2 = y_true2_1_2 - y_pred2_1_2

x_axis2_1_2 = y_true2_1_2.index
  
# 绘制残差图  

plt.scatter(x_axis2_1_2,residuals2_1_2, label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('Year(1998-2018)')  
plt.ylabel('H->L Residuals')  
plt.title('Residual Plot(H->L)')  
plt.grid(True)  
plt.show()

#拟合图
y_fitted_2_3 = model_2_3.predict(X_train_processed_2_3)
train_2_3["fitted"] = y_fitted_2_3  
y_true_train_2_3=train_2_3.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_3= train_2_3.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_3.index, y_true_train_2_3, label='True Values',color='blue')
plt.plot(y_fitted_train_2_3.index, y_fitted_train_2_3, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->D)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

#预测图
test_2_3["predictions"]=y_pred_2_3
y_true1_2_3=test_2_3.groupby("age")['ny'].mean()   #测试集真实值
y_pred1_2_3= test_2_3.groupby('age')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_2_3.index, y_true1_2_3, label='True Values')
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(L->D)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差  
residuals1_2_3 = y_true1_2_3 - y_pred1_2_3
x_axis1_2_3 = y_true1_2_3.index
  
# 绘制残差图  

plt.scatter(x_axis1_2_3,residuals1_2_3,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xlabel('age(65-105)')  
plt.ylabel('L->D Residuals')  
plt.title('Residual Plot(L->D)')  
plt.grid(True)  
plt.show()

predictions_list_2_3 = [train_2_3['fitted'], test_2_3['predictions']]
df_2_3['predictions'] = pd.concat(predictions_list_2_3)

y_true2_2_3=df_2_3.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_3= df_2_3.groupby('t')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true2_2_3.index, y_true2_2_3, label='True Values')
plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by year(L->D)')
plt.xlabel('year(1998-2018)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_2_3 = y_true2_2_3 - y_pred2_2_3

x_axis2_2_3 = y_true2_2_3.index
  
# 绘制残差图  
 
plt.scatter(x_axis2_2_3,residuals2_2_3,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('L->D Residuals')  
plt.title('Residual Plot(L->D)')  
plt.grid(True)  
plt.show()

# 拟合图
y_fitted_2_1 = model_2_1.predict(X_train_processed_2_1)
train_2_1["fitted"] = y_fitted_2_1  
y_true_train_2_1=train_2_1.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_1= train_2_1.groupby('age')['fitted'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_1.index, y_true_train_2_1, label='True Values',color='blue')
plt.plot(y_fitted_train_2_1.index, y_fitted_train_2_1, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->H)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_1["predictions"]=y_pred_2_1
y_true1_2_1=test_2_1.groupby("age")['ny'].mean()   
y_pred1_2_1= test_2_1.groupby('age')['predictions'].mean()   
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true1_2_1.index, y_true1_2_1, label='True Values')
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Age(L->H)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

# 计算残差  
residuals1_2_1 = y_true1_2_1 - y_pred1_2_1

x_axis1_2_1 = y_true1_2_1.index
  
# 绘制残差图  
 
plt.scatter(x_axis1_2_1,residuals1_2_1,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xlabel('age(65-105)')  
plt.ylabel('L->H Residuals')  
plt.title('Residual Plot(L->H)')  
plt.grid(True)  
plt.show()

predictions_list_2_1 = [train_2_1['fitted'], test_2_1['predictions']]
df_2_1['predictions'] = pd.concat(predictions_list_2_1)

y_true2_2_1=df_2_1.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_1= df_2_1.groupby('t')['predictions'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.scatter(y_true2_2_1.index, y_true2_2_1, label='True Values')
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by year(L->H)')
plt.xlabel('year(1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差  
residuals2_2_1 = y_true2_2_1 - y_pred2_2_1
x_axis2_2_1 = y_true2_2_1.index
  
# 绘制残差图  
 
plt.scatter(x_axis2_2_1,residuals2_2_1,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('L->H Residuals')  
plt.title('Residual Plot(L->H)')  
plt.grid(True)  
plt.show()

print(poisson_deviance)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

print(poisson_deviance)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)