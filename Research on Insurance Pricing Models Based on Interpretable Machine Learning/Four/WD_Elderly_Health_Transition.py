import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families import family, Poisson
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error  
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder 
from keras import backend as K  
from keras import metrics 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.losses import poisson  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Normalization, Resizing, RandomFlip  #from tensorflow.keras.layers.experimental import preprocessing
import keras_tuner as kt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.titlesize"] = 20  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 15  
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_3.csv')  
train = df.loc[df['time'].isin([1, 3, 5.5, 8.5, 11.5])]
test = df.loc[df['time'].isin([14.5, 18])]
# 定义特征和目标变量  
Y = 'ny'  
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']  
  
# 特征工程：对分类变量和数值变量分别进行编码  
categorical_cols = ['sex', 'residenc', 'marry', 'smoke', 'drink']  
numeric_cols = ['time', 'age']  
  
# 使用ColumnTransformer进行特征预处理  
preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', StandardScaler(), numeric_cols),  
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  
    ]  
)  
  
# 预处理训练和测试数据  
X_train_preprocessed = preprocessor.fit_transform(train[X])  
X_test_preprocessed = preprocessor.transform(test[X])  

 
# 分割数值特征和分类特征（用于构建模型）  
time_train = X_train_preprocessed[:, :1]  
age_train = X_train_preprocessed[:, 1:2]  
time_test = X_test_preprocessed[:, :1]  
age_test = X_test_preprocessed[:, 1:2]  

X_train_wide = X_train_preprocessed[:, len(numeric_cols):]  
X_test_wide = X_test_preprocessed[:, len(numeric_cols):]  
X_train_deep=X_train_preprocessed[:, len(numeric_cols):]  
X_test_deep= X_test_preprocessed[:, len(numeric_cols):]  
# 提取目标变量和样本权重
y_train = train[Y]  
w_train = train['sum_hyear']  
w_test = test['sum_hyear']  
y_test = test[Y]  

time_train,age_train,

class WideAndDeepModel:
    def __init__(self, wide_input_shape, deep_input_shape):
        self.wide_input_shape = wide_input_shape
        self.deep_input_shape = deep_input_shape
        self.model = None

    def build_model(self):
        # Wide Component
        wide_input = Input(shape=self.wide_input_shape, name='wide_input')
        wide_output = Dense(5, activation='exponential')(wide_input)
      

        # Deep Component: Neural Network
        deep_input = Input(shape=self.deep_input_shape, name='deep_input')
        deep_layer = BatchNormalization()(deep_input)
        deep_layer = Dense(80, activation='selu')(deep_layer)
        deep_layer = Dense(80, activation='selu')(deep_layer)
        deep_layer = Dense(80, activation='selu')(deep_layer)
        deep_layer = Dropout(0.1)(deep_layer)

        # Combine Wide and Deep Components
        deep_combined = Concatenate()([wide_output, deep_layer])

        # Age and Time Features
        age_input = Input(shape=(1,), name='age_input')
        time_input = Input(shape=(1,), name='time_input')

        # Connect Age and Time to the Deep Component
        deep_combined_with_age_time = Concatenate()([deep_combined, age_input, time_input])

        # Output Layer
        output_layer = Dense(1, activation='exponential', name='output')(deep_combined_with_age_time)

        # Final Model
        model = Model(inputs=[wide_input, deep_input, age_input, time_input], outputs=output_layer)
        return model

    def compile_model(self, optimizer):
        self.model = self.build_model()
        self.model.compile(optimizer=optimizer, loss='poisson', metrics=['poisson'])

    def train_model(self, wide_input, deep_input, age_input, time_input, y_train, epochs, sample_weight, batch_size):
        history = self.model.fit(
            {'wide_input': wide_input, 'deep_input': deep_input, 'age_input': age_input, 'time_input': time_input},
            y_train,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def predict(self, wide_input, deep_input, age_input, time_input):
        return self.model.predict(
            {'wide_input': wide_input, 'deep_input': deep_input, 'age_input': age_input, 'time_input': time_input})

    def summary(self):
        self.model.summary()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

# Initialize and compile model
wide_deep_model = WideAndDeepModel(wide_input_shape=(X_train_deep.shape[1],), deep_input_shape=(X_train_deep.shape[1],))
wide_deep_model.compile_model(optimizer=Adam())
wide_deep_model.summary()
# Train the model
history = wide_deep_model.train_model(
    wide_input=X_train_deep,
    deep_input=X_train_deep,
    age_input=age_train,
    time_input=time_train,
    y_train=y_train,
    sample_weight=w_train,
    epochs=100,
    batch_size=32
)

weights_1_3 = wide_deep_model.get_weights()
plt.plot(history.history['loss'])
plt.show()

y_pred1 = wide_deep_model.predict(
    wide_input=X_test_wide,
    deep_input=X_test_deep,
    age_input=age_test,
    time_input=time_test)
y_pred1

mse = mean_squared_error(y_test,y_pred1)
print(f"Mean Squared Error: {mse}")

# 对训练集进行预测
y_fitted = wide_deep_model.predict(
    wide_input=X_train_wide,
    deep_input=X_train_deep,
    age_input=age_train,
    time_input=time_train)
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

test["predictions"] = y_pred1

# test["y_pred1"]=y_pred1
# test['predictions']=test["y_pred1"]/test['sum_hyear']
y_true2 = test.groupby("age")['ny'].mean()  # 测试集真实值
y_pred2 = test.groupby('age')['predictions'].mean()  # 预测值
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

plt.scatter(x_axis1, residuals1, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xlabel('age(65-105)')
plt.ylabel('H->D Residuals')
plt.title('Residual Plot(H->D)')
plt.grid(True)
plt.show()

predictions_list = [train['fitted'], test['predictions']]
df['predictions'] = pd.concat(predictions_list)

y_true3 = df.groupby("t")['ny'].mean()   # Testing set true values
y_pred3= df.groupby('t')['predictions'].mean()   # Predicted values


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

sample_weight=w_test
y_true=y_test
y_pred=test["predictions"].values
y_true = np.array(y_true)  
y_pred = np.array(y_pred) 
deviance_per_obs = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
deviance_per_obs[y_true == 0] =2 * y_pred[y_true == 0]

weighted_sum = tf.reduce_sum(tf.multiply(deviance_per_obs, sample_weight))  
exposure_weighted_average=weighted_sum/sample_weight.sum() 
exposure_weighted_average

# predictions_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx')
# fitted_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx')
# fitted_predictions_1_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx') 

# test_reset_index = test.reset_index(drop=True)   
# predictions_1_3['wd_predictions'] = test_reset_index["predictions"]

# fitted_1_3['wd_fitted']=train["fitted"]
# fitted_predictions_1_3['wd_fitted_predictions']=df['predictions']


# fitted_predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx', index=False)
# predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx', index=False)
# fitted_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx', index=False)


df_1_2 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_2.csv')  
train_1_2 = df_1_2 .loc[df_1_2['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_1_2= df_1_2.loc[df_1_2['time'].isin([14.5, 18])] 

# 预处理训练和测试数据  
X_train_preprocessed_1_2 = preprocessor.fit_transform(train_1_2[X])  
X_test_preprocessed_1_2= preprocessor.transform(test_1_2[X])  

# 分割数值特征和分类特征（用于构建模型）  

X_train_1_2_wide = X_train_preprocessed_1_2[:, len(numeric_cols):]  
X_train_1_2_deep= X_train_preprocessed_1_2[:, len(numeric_cols):] 
X_test_1_2_wide = X_test_preprocessed_1_2[:, len(numeric_cols):]  
X_test_1_2_deep = X_test_preprocessed_1_2[:, len(numeric_cols):]  

time_train_1_2 = X_train_preprocessed_1_2[:, :1]  
age_train_1_2 = X_train_preprocessed_1_2[:, 1:2]  
time_test_1_2 = X_test_preprocessed_1_2[:, :1]  
age_test_1_2 = X_test_preprocessed_1_2[:, 1:2]  

# 提取目标变量和样本权重（这部分保持不变）  
y_train_1_2 = train_1_2[Y]  
w_train_1_2 = train_1_2['sum_hyear']  
w_test_1_2 = test_1_2['sum_hyear']  
y_test_1_2 = test_1_2[Y]  
  

model_1_2= WideAndDeepModel(wide_input_shape=(X_train_1_2_wide.shape[1],), deep_input_shape=(X_train_1_2_deep.shape[1],))
model_1_2.compile_model(optimizer=Adam())
# 设置初始权重
model_1_2.set_weights(weights_1_3)

history_1_2=model_1_2.train_model(
    wide_input=X_train_1_2_wide,
    deep_input=X_train_1_2_deep,
    age_input=age_train_1_2,
    time_input=time_train_1_2,
    y_train=y_train_1_2,
    sample_weight=w_train_1_2,
    epochs=100,
    batch_size=64
)

plt.plot(history_1_2.history['loss'])
plt.show()

# 对训练集进行预测
y_fitted_1_2 = model_1_2.predict(
    wide_input=X_train_1_2_wide,
    deep_input=X_train_1_2_deep,
    age_input=age_train_1_2,
    time_input=time_train_1_2)
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


predictions_sigma = model_1_2.predict(
    wide_input=X_test_1_2_wide,
    deep_input=X_test_1_2_deep,
    age_input=age_test_1_2,
    time_input=time_test_1_2)
predictions_sigma

test_1_2["predictions"] = predictions_sigma
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

y_true2_1_2 = df_1_2.groupby("t")['ny'].mean()  # Testing set true values
y_pred2_1_2 = df_1_2.groupby('t')['predictions'].mean()  # Predicted values

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

plt.scatter(x_axis2_1_2, residuals2_1_2, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('Year(1998-2018)')
plt.ylabel('H->L Residuals')
plt.title('Residual Plot(H->L)')
plt.grid(True)
plt.show()


sample_weight_1_2=w_test_1_2
y_true_1_2=y_test_1_2
y_pred_1_2=test_1_2["predictions"].values
# y_true_1_2 = np.array(y_true_1_2)  
# y_pred_1_2 = np.array(y_pred_1_2) 
deviance_per_obs_1_2 = 2 * (y_true_1_2 * np.log(y_true_1_2 / y_pred_1_2) - (y_true_1_2 - y_pred_1_2))
deviance_per_obs_1_2[y_true_1_2 == 0] = 2* y_pred_1_2[y_true_1_2 == 0]
weighted_sum_1_2 = tf.reduce_sum(tf.multiply(deviance_per_obs_1_2, sample_weight_1_2))  
exposure_weighted_average_1_2=weighted_sum_1_2/sample_weight_1_2.sum() 
exposure_weighted_average_1_2

mse = mean_squared_error(y_test_1_2,y_pred_1_2)
print(f"Mean Squared Error: {mse}")

# predictions_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx')
# fitted_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx')
# fitted_predictions_1_2 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx') 

# test_1_2_reset_index = test_1_2.reset_index(drop=True)   
# predictions_1_2['wd_predictions'] = test_1_2_reset_index["predictions"]

# fitted_1_2['wd_fitted']=train_1_2["fitted"]
# fitted_predictions_1_2['wd_fitted_predictions']=df_1_2['predictions']


# fitted_predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx', index=False)
# predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx', index=False)
# fitted_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx', index=False)

df_2_3 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_3.csv')  
train_2_3 = df_2_3 .loc[df_2_3['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_3= df_2_3.loc[df_2_3['time'].isin([14.5, 18])] 

# 预处理训练和测试数据  
X_train_preprocessed_2_3 = preprocessor.fit_transform(train_2_3[X])  
X_test_preprocessed_2_3= preprocessor.transform(test_2_3[X])  

# 分割数值特征和分类特征（用于构建模型）  

X_train_2_3_wide = X_train_preprocessed_2_3[:, len(numeric_cols):]  
X_train_2_3_deep= X_train_preprocessed_2_3[:, len(numeric_cols):] 
X_test_2_3_wide = X_test_preprocessed_2_3[:, len(numeric_cols):]  
X_test_2_3_deep = X_test_preprocessed_2_3[:, len(numeric_cols):]  

time_train_2_3 = X_train_preprocessed_2_3[:, :1]  
age_train_2_3 = X_train_preprocessed_2_3[:, 1:2]  
time_test_2_3 = X_test_preprocessed_2_3[:, :1]  
age_test_2_3 = X_test_preprocessed_2_3[:, 1:2]  

# 提取目标变量和样本权重（这部分保持不变）  
y_train_2_3 = train_2_3[Y]  
w_train_2_3 = train_2_3['sum_lyear']  
w_test_2_3 = test_2_3['sum_lyear']  
y_test_2_3 = test_2_3[Y]  

model_2_3= WideAndDeepModel(wide_input_shape=(X_train_2_3_wide.shape[1],), deep_input_shape=(X_train_2_3_deep.shape[1],))
model_2_3.compile_model(optimizer=Adam())
# 设置初始权重
model_2_3.set_weights(weights_1_3)
history_2_3=model_2_3.train_model(
    wide_input=X_train_2_3_wide,
    deep_input=X_train_2_3_deep,
    age_input=age_train_2_3,
    time_input=time_train_2_3,
    y_train=y_train_2_3,
    sample_weight=w_train_2_3,
    epochs=100,
    batch_size=64
)


plt.plot(history_2_3.history['loss'])
plt.show()

# 对训练集进行预测
y_fitted_2_3 = model_2_3.predict(
    wide_input=X_train_2_3_wide,
    deep_input=X_train_2_3_deep,
    age_input=age_train_2_3,
    time_input=time_train_2_3)
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


predictions_nu = model_2_3.predict(
    wide_input=X_test_2_3_wide,
    deep_input=X_test_2_3_deep,
    age_input=age_test_2_3,
    time_input=time_test_2_3)
predictions_nu


test_2_3["predictions"] = predictions_nu
# ['sum_health1_3']/['sum_year']
y_true1_2_3 = test_2_3.groupby("age")['ny'].mean()  # 测试集真实值
y_pred1_2_3 = test_2_3.groupby('age')['predictions'].mean()  # 预测值
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

plt.scatter(x_axis1_2_3, residuals1_2_3, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xlabel('age(65-105)')
plt.ylabel('L->D Residuals')
plt.title('Residual Plot(L->D)')
plt.grid(True)
plt.show()


predictions_list_2_3 = [train_2_3['fitted'], test_2_3['predictions']]
df_2_3['predictions'] = pd.concat(predictions_list_2_3)

y_true2_2_3 = df_2_3.groupby("t")['ny'].mean()  # Testing set true values
y_pred2_2_3 = df_2_3.groupby('t')['predictions'].mean()  # Predicted values

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

plt.scatter(x_axis2_2_3, residuals2_2_3, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')
plt.ylabel('L->D Residuals')
plt.title('Residual Plot(L->D)')
plt.grid(True)
plt.show()

sample_weight_2_3=w_test_2_3
y_true_2_3=y_test_2_3
y_pred_2_3=test_2_3["predictions"].values
# y_true_2_3 = np.array(y_true_2_3)  
# y_pred_2_3 = np.array(y_pred_2_3) 
deviance_per_obs_2_3 = 2 * (y_true_2_3 * np.log(y_true_2_3 / y_pred_2_3) - (y_true_2_3 - y_pred_2_3))
deviance_per_obs_2_3[y_true_2_3 == 0] =2* y_pred_2_3[y_true_2_3 == 0]
weighted_sum_2_3 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_3, sample_weight_2_3))  
exposure_weighted_average_2_3=weighted_sum_2_3/sample_weight_2_3.sum() 
exposure_weighted_average_2_3

mse = mean_squared_error(y_test_2_3,y_pred_2_3)
print(f"Mean Squared Error: {mse}")

predictions_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx')
fitted_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx')
fitted_predictions_2_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx') 

test_2_3_reset_index = test_2_3.reset_index(drop=True)   
predictions_2_3['wd_predictions'] = test_2_3_reset_index["predictions"]

fitted_2_3['wd_fitted']=train_2_3["fitted"]
fitted_predictions_2_3['wd_fitted_predictions']=df_2_3['predictions']


fitted_predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx', index=False)
predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx', index=False)
fitted_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx', index=False)

df_2_1 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_1.csv')  
train_2_1 = df_2_1 .loc[df_2_1['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_1= df_2_1.loc[df_2_1['time'].isin([14.5, 18])] 

# 预处理训练和测试数据  
X_train_preprocessed_2_1 = preprocessor.fit_transform(train_2_1[X])  
X_test_preprocessed_2_1= preprocessor.transform(test_2_1[X])  

# 分割数值特征和分类特征（用于构建模型）  

X_train_2_1_wide = X_train_preprocessed_2_1[:, len(numeric_cols):]  
X_train_2_1_deep= X_train_preprocessed_2_1[:, len(numeric_cols):] 
X_test_2_1_wide = X_test_preprocessed_2_1[:, len(numeric_cols):]  
X_test_2_1_deep = X_test_preprocessed_2_1[:, len(numeric_cols):]  

time_train_2_1 = X_train_preprocessed_2_1[:, :1]  
age_train_2_1 = X_train_preprocessed_2_1[:, 1:2]  
time_test_2_1 = X_test_preprocessed_2_1[:, :1]  
age_test_2_1 = X_test_preprocessed_2_1[:, 1:2]  

# 提取目标变量和样本权重（这部分保持不变）  
y_train_2_1 = train_2_1[Y]  
w_train_2_1 = train_2_1['sum_lyear']  
w_test_2_1 = test_2_1['sum_lyear']  
y_test_2_1 = test_2_1[Y]  
  
model_2_1= WideAndDeepModel(wide_input_shape=(X_train_2_1_wide.shape[1],), deep_input_shape=(X_train_2_1_deep.shape[1],))
model_2_1.compile_model(optimizer=Adam())
# 设置初始权重
model_2_1.set_weights(weights_1_3)

history_2_1=model_2_1.train_model(
    wide_input=X_train_2_1_wide,
    deep_input=X_train_2_1_deep,
    age_input=age_train_2_1,
    time_input=time_train_2_1,
    y_train=y_train_2_1,
    sample_weight=w_train_2_1,
    epochs=100,
    batch_size=32
)


plt.plot(history_2_1.history['loss'])
plt.show()

# 对训练集进行预测
y_fitted_2_1 = model_2_1.predict(
    wide_input=X_train_2_1_wide,
    deep_input=X_train_2_1_deep,
    age_input=age_train_2_1,
    time_input=time_train_2_1)
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

predictions_2_1= model_2_1.predict(
    wide_input=X_test_2_1_wide,
    deep_input=X_test_2_1_deep,
    age_input=age_test_2_1,
    time_input=time_test_2_1)
predictions_2_1

test_2_1["predictions"] = predictions_2_1
# ['sum_health1_3']/['sum_year']
y_true1_2_1 = test_2_1.groupby("age")['ny'].mean()  # 测试集真实值
y_pred1_2_1 = test_2_1.groupby('age')['predictions'].mean()  # 预测值
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

plt.scatter(x_axis1_2_1, residuals1_2_1, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xlabel('age(65-105)')
plt.ylabel('L->H Residuals')
plt.title('Residual Plot(L->H)')
plt.grid(True)
plt.show()

predictions_list_2_1 = [train_2_1['fitted'], test_2_1['predictions']]
df_2_1['predictions'] = pd.concat(predictions_list_2_1)

y_true2_2_1 = df_2_1.groupby("t")['ny'].mean()  # Testing set true values
y_pred2_2_1 = df_2_1.groupby('t')['predictions'].mean()  # Predicted values

##绘制按年份分组的转移次数真实值和预测值

plt.scatter(y_true2_2_1.index, y_true2_2_1, label='True Values')
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->L)')
plt.xlabel('Year(1998-2018)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

# 计算残差
residuals2_2_1 = y_true2_2_1 - y_pred2_2_1

x_axis2_2_1 = y_true2_2_1.index

# 绘制残差图

plt.scatter(x_axis2_2_1, residuals2_2_1, label='Residuals')
plt.axhline(y=0, linestyle='--', label='y=0', color='red')  # 绘制y=0的直线
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('Year(1998-2018)')
plt.ylabel('H->L Residuals')
plt.title('Residual Plot(H->L)')
plt.grid(True)
plt.show()

sample_weight_2_1=w_test_2_1
y_true_2_1=y_test_2_1
y_pred_2_1=test_2_1["predictions"].values
# y_true_2_1 = np.array(y_true_2_1)  
# y_pred_2_1 = np.array(y_pred_2_1) 
deviance_per_obs_2_1 = 2 * (y_true_2_1 * np.log(y_true_2_1 / y_pred_2_1) - (y_true_2_1 - y_pred_2_1))
deviance_per_obs_2_1[y_true_2_1 == 0] = 2*y_pred_2_1[y_true_2_1 == 0]
weighted_sum_2_1 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_1, sample_weight_2_1))  
exposure_weighted_average_2_1=weighted_sum_2_1/sample_weight_2_1.sum() 
exposure_weighted_average_2_1

mse = mean_squared_error(y_test_2_1,y_pred_2_1)
print(f"Mean Squared Error: {mse}")


predictions_2_1= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_1.xlsx')
fitted_2_1= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_1.xlsx')
fitted_predictions_2_1 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_1.xlsx') 

test_2_1_reset_index = test_2_1.reset_index(drop=True)   
predictions_2_1['wd_predictions'] = test_2_1_reset_index["predictions"]

fitted_2_1['wd_fitted']=train_2_1["fitted"]
fitted_predictions_2_1['wd_fitted_predictions']=df_2_1['predictions']


fitted_predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_1.xlsx', index=False)
predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_1.xlsx', index=False)
fitted_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_1.xlsx', index=False)

print(exposure_weighted_average)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

print(exposure_weighted_average)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

print(exposure_weighted_average)
print(exposure_weighted_average_1_2)
print(exposure_weighted_average_2_3)
print(exposure_weighted_average_2_1)

