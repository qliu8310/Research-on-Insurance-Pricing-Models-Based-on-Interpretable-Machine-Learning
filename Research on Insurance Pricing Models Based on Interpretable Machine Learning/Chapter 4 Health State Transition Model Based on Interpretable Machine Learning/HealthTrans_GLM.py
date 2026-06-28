import numpy as np  
import pandas as pd  
import statsmodels.api as sm  
import matplotlib.pyplot as plt  
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.titlesize"] = 20  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 15  
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error 
import tensorflow as tf 
from keras.activations import exponential
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_3.csv')  
data

# 确定训练集和测试集
train_data = data.loc[data['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_data = data.loc[data['time'].isin([14.5, 18])]  

# 定义特征和目标变量
Y = 'sum_health1_3'
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']

# 提取特征和目标变量
X_train = train_data[X]
y_train = train_data[Y]
X_test = test_data[X]
y_test = test_data[Y]

# 提取暴露量  
exposure_train = train_data['sum_hyear']  
exposure_test = test_data['sum_hyear']  

# 数值型和分类变量的列
num_features = ['time', 'age']
cat_features = ['sex', 'residenc', 'marry', 'smoke', 'drink']

# 创建预处理管道，这样才能保证特征处理一致性
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

# 读取数据
df = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据\分组11_3.csv')

# 分割训练集和测试集
train = df.loc[df['time'].isin([1, 3, 5.5, 8.5, 11.5])]
test = df.loc[df['time'].isin([14.5, 18])]

# 定义特征和目标变量
Y = 'sum_health1_3'
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

# # 分割数值特征和分类特征（用于构建模型）
# num_train = X_train_preprocessed[:, :len(numeric_cols)]
# cat_train = X_train_preprocessed[:, len(numeric_cols):]
# num_test = X_test_preprocessed[:, :len(numeric_cols)]
# cat_test = X_test_preprocessed[:, len(numeric_cols):]

# 提取目标变量和样本权重
y_train = train[Y]
w_train = train['sum_hyear']
w_test = test['sum_hyear']
y_test = test[Y]

# X_train_preprocessed

#创建Poisson回归模型  
poisson_model = sm.Poisson(train_data[Y], sm.add_constant(X_train_processed), exposure=exposure_train)  
  
# 拟合模型  
poisson_results = poisson_model.fit()  
print(poisson_results.summary())  

# 对训练集进行预测  
fitted = poisson_results.predict(sm.add_constant(X_train_preprocessed), exposure=exposure_train)  
train_data["fitted"] = fitted/exposure_train
  
# 对测试集进行预测  
predictions = poisson_results.predict(sm.add_constant(X_test_preprocessed), exposure=exposure_test)  
test_data["predictions"] = predictions/exposure_test

train_true_ny=train_data.groupby("age")['ny'].mean()  #训练集真实值
fitted_ny=train_data.groupby('age')['fitted'].mean()  #拟合值
test_true_ny=test_data.groupby("age")['ny'].mean()   #测试集真实值
pred_ny = test_data.groupby('age')['predictions'].mean()   #预测值

# 计算预测值的均方误差
# deviance = poisson_results.deviance  
# print(f"Deviance: {deviance}")

mse = mean_squared_error(test_data['ny'], test_data['predictions'])
print(f"Mean Squared Error: {mse}")

sample_weight=exposure_test.values
y_true=test_data['ny'].values
y_pred=test_data["predictions"].values
y_true = np.array(y_true)  
y_pred = np.array(y_pred) 
deviance_per_obs = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
deviance_per_obs[y_true == 0] =2*y_pred[y_true == 0]
weighted_sum = tf.reduce_sum(tf.multiply(deviance_per_obs, sample_weight))  
exposure_weighted_average=weighted_sum/sample_weight.sum() 
exposure_weighted_average

# 绘制折线图
# 真实值vs拟合值
plt.figure(figsize=(15, 8))
plt.plot(train_true_ny.index, train_true_ny.values,color='blue', label='Number of real transfers')  #折线图
plt.plot(fitted_ny.index, fitted_ny.values,color='red', label='Fitted number of transfers')  #折线图
plt.xlabel('年龄',fontsize=15)
plt.ylabel('转移强度',fontsize=15)
plt.title('泊松回归模型——训练集真实值与拟合值对比',fontsize=20)
plt.legend()

# 真实值vs预测值
plt.figure(figsize=(15, 8))
plt.plot(pred_ny.index, pred_ny.values,color='red', label='Predicted number of transfers')  #折线图
plt.plot(test_true_ny.index, test_true_ny.values,color='blue', label='Number of real transfers')  #折线图
plt.xlabel('年龄',fontsize=15)
plt.ylabel('转移强度',fontsize=15)
plt.title('泊松回归模型——测试集真实值与预测值对比')
plt.legend(fontsize=13)
plt.show()

# 计算残差  
residuals1 = test_true_ny - pred_ny

x_axis1 = test_true_ny.index
  
# 绘制残差图   
plt.scatter(x_axis1,residuals1, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线  
plt.xlabel('age(65-105)')  
plt.ylabel('H->D Residuals')  
plt.title('Residual Plot(H->D)')  
plt.grid(True)  
plt.show()

predictions_list = [train_data['fitted'], test_data['predictions']]

data['predictions'] = pd.concat(predictions_list)


test_true_ny1 = data.groupby("t")['ny'].mean()   # Testing set true values
pred_ny1 = data.groupby('t')['predictions'].mean()   # Predicted values


plt.scatter(test_true_ny1.index, test_true_ny1, label='True Values', color='black')
plt.plot(pred_ny1.index, pred_ny1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->D)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('H->D intensity')
plt.legend()
plt.grid(True)

# Set x-axis ticks to specific years
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])

plt.show()

# 计算残差  
residuals2 = test_true_ny1 - pred_ny1

x_axis2 = test_true_ny1.index
  
# 绘制残差图  
 
plt.scatter(x_axis2,residuals2, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('H->D Residuals')  
plt.title('Residual Plot(H->D)')  
plt.grid(True)  
plt.show()

# poisson_predictions_1_3 = pd.DataFrame( {'poisson_Predictions': test_data['predictions']})
# poisson_fitted_1_3 = pd.DataFrame({ 'poisson_fitted': train_data['fitted']}  )
# poisson_fitted_predictions_1_3 = pd.DataFrame({'poisson_fitted_predictions': data['predictions']})
  

# poisson_fitted_predictions_1_3.to_excel(r'E:\Administrator\Desktop\predictions\fitted_predictions_1_3.xlsx', index=False)
# poisson_predictions_1_3.to_excel(r'E:\Administrator\Desktop\predictions\predictions_1_3.xlsx', index=False)
# poisson_fitted_1_3.to_excel(r'E:\Administrator\Desktop\predictions\fitted_1_3.xlsx', index=False)


predictions_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx')
fitted_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx')
fitted_predictions_1_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx') 

test_reset_index = test_data.reset_index(drop=True)   
predictions_1_3['poisson_predictions'] = test_reset_index["predictions"]

fitted_1_3['poisson_fitted']=train_data["fitted"]
fitted_predictions_1_3['poisson_fitted_predictions']=data['predictions']


fitted_predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx', index=False)
predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx', index=False)
fitted_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx', index=False)

data_1_2= pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据\分组11_2.csv')  

Y = 'sum_health1_2'  
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']  
  
# train_data_1_2, test_data_1_2 = train_test_split(data_1_2, test_size=0.2, random_state=42)  

train_data_1_2 = data_1_2 .loc[data_1_2['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_data_1_2= data_1_2.loc[data_1_2['time'].isin([14.5, 18])]  
   
exposure_train_1_2 = train_data_1_2['sum_hyear']  
exposure_test_1_2 = test_data_1_2['sum_hyear']  


# 提取特征和目标变量
X_train_1_2 = train_data_1_2[X]
y_train_1_2 = train_data_1_2[Y]
X_test_1_2 = test_data_1_2[X]
y_test_1_2 = test_data_1_2[Y]


# 对训练集和测试集进行预处理
X_train_processed_1_2 = preprocessor.fit_transform(X_train_1_2)
X_test_processed_1_2  = preprocessor.transform(X_test_1_2)

#创建Poisson回归模型  
poisson_model_1_2 = sm.Poisson(train_data_1_2[Y], sm.add_constant(X_train_processed_1_2), exposure=exposure_train_1_2)  
# 拟合模型  
poisson_results_1_2 = poisson_model_1_2.fit()  
print(poisson_results_1_2.summary())  

# 对训练集进行预测  
fitted_1_2 = poisson_results_1_2.predict(sm.add_constant(X_train_processed_1_2), exposure=exposure_train_1_2)  
train_data_1_2["fitted"] = fitted_1_2/exposure_train_1_2
# 对测试集进行预测  
predictions_1_2 = poisson_results_1_2.predict(sm.add_constant(X_test_processed_1_2), exposure=exposure_test_1_2)  
test_data_1_2["predictions"] = predictions_1_2/exposure_test_1_2

train_true_ny_1_2=train_data_1_2.groupby("age")['ny'].mean()  #训练集真实值
fitted_ny_1_2=train_data_1_2.groupby('age')['fitted'].mean()  #拟合值
test_true_ny_1_2=test_data_1_2.groupby("age")['ny'].mean()   #测试集真实值
pred_ny_1_2= test_data_1_2.groupby('age')['predictions'].mean()   #预测值

mse = mean_squared_error(test_data_1_2['ny'], test_data_1_2['predictions'])
print(f"Mean Squared Error: {mse}")

sample_weight_1_2=exposure_test_1_2.values
y_true_1_2=test_data_1_2['ny'].values
y_pred_1_2=test_data_1_2["predictions"].values
y_true_1_2 = np.array(y_true_1_2)  
y_pred_1_2 = np.array(y_pred_1_2) 
deviance_per_obs_1_2 = 2 * (y_true_1_2 * np.log(y_true_1_2 / y_pred_1_2) - (y_true_1_2 - y_pred_1_2))
deviance_per_obs_1_2[y_true_1_2 == 0] =2* y_pred_1_2[y_true_1_2 == 0]
weighted_sum_1_2 = tf.reduce_sum(tf.multiply(deviance_per_obs_1_2, sample_weight_1_2))  
exposure_weighted_average_1_2=weighted_sum_1_2/sample_weight_1_2.sum() 
exposure_weighted_average_1_2

# 真实值vs拟合值
plt.plot(train_true_ny_1_2.index, train_true_ny_1_2.values,color='blue', label='Number of real transfers')  #折线图
plt.plot(fitted_ny_1_2.index, fitted_ny_1_2.values,color='red', label='Fitted number of transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——训练集真实值与拟合值对比')
plt.legend()


data_1_2= pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据\分组11_2.csv')  

Y = 'sum_health1_2'  
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']  
  
# train_data_1_2, test_data_1_2 = train_test_split(data_1_2, test_size=0.2, random_state=42)  

train_data_1_2 = data_1_2 .loc[data_1_2['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_data_1_2= data_1_2.loc[data_1_2['time'].isin([14.5, 18])]  
   
exposure_train_1_2 = train_data_1_2['sum_hyear']  
exposure_test_1_2 = test_data_1_2['sum_hyear']  


# 提取特征和目标变量
X_train_1_2 = train_data_1_2[X]
y_train_1_2 = train_data_1_2[Y]
X_test_1_2 = test_data_1_2[X]
y_test_1_2 = test_data_1_2[Y]


# 对训练集和测试集进行预处理
X_train_processed_1_2 = preprocessor.fit_transform(X_train_1_2)
X_test_processed_1_2  = preprocessor.transform(X_test_1_2)

#创建Poisson回归模型  
poisson_model_1_2 = sm.Poisson(train_data_1_2[Y], sm.add_constant(X_train_processed_1_2), exposure=exposure_train_1_2)  
# 拟合模型  
poisson_results_1_2 = poisson_model_1_2.fit()  
print(poisson_results_1_2.summary())  

# 对训练集进行预测  
fitted_1_2 = poisson_results_1_2.predict(sm.add_constant(X_train_processed_1_2), exposure=exposure_train_1_2)  
train_data_1_2["fitted"] = fitted_1_2/exposure_train_1_2
# 对测试集进行预测  
predictions_1_2 = poisson_results_1_2.predict(sm.add_constant(X_test_processed_1_2), exposure=exposure_test_1_2)  
test_data_1_2["predictions"] = predictions_1_2/exposure_test_1_2


train_true_ny_1_2=train_data_1_2.groupby("age")['ny'].mean()  #训练集真实值
fitted_ny_1_2=train_data_1_2.groupby('age')['fitted'].mean()  #拟合值
test_true_ny_1_2=test_data_1_2.groupby("age")['ny'].mean()   #测试集真实值
pred_ny_1_2= test_data_1_2.groupby('age')['predictions'].mean()   #预测值


mse = mean_squared_error(test_data_1_2['ny'], test_data_1_2['predictions'])
print(f"Mean Squared Error: {mse}")

sample_weight_1_2=exposure_test_1_2.values
y_true_1_2=test_data_1_2['ny'].values
y_pred_1_2=test_data_1_2["predictions"].values
y_true_1_2 = np.array(y_true_1_2)  
y_pred_1_2 = np.array(y_pred_1_2) 
deviance_per_obs_1_2 = 2 * (y_true_1_2 * np.log(y_true_1_2 / y_pred_1_2) - (y_true_1_2 - y_pred_1_2))
deviance_per_obs_1_2[y_true_1_2 == 0] =2* y_pred_1_2[y_true_1_2 == 0]
weighted_sum_1_2 = tf.reduce_sum(tf.multiply(deviance_per_obs_1_2, sample_weight_1_2))  
exposure_weighted_average_1_2=weighted_sum_1_2/sample_weight_1_2.sum() 
exposure_weighted_average_1_2

# 真实值vs拟合值
plt.plot(train_true_ny_1_2.index, train_true_ny_1_2.values,color='blue', label='Number of real transfers')  #折线图
plt.plot(fitted_ny_1_2.index, fitted_ny_1_2.values,color='red', label='Fitted number of transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——训练集真实值与拟合值对比')
plt.legend()


# 真实值vs预测值

plt.plot(pred_ny_1_2.index, pred_ny_1_2.values,color='red', label='Predicted number of transfers')  #折线图
plt.plot(test_true_ny_1_2.index, test_true_ny_1_2.values,color='blue', label='Number of real transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——测试集真实值与预测值对比')
plt.legend()
plt.show()

# 计算残差  
residuals1_1_2 = test_true_ny_1_2 - pred_ny_1_2

x_axis1_1_2 = test_true_ny_1_2.index
  
# 绘制残差图  
plt.figure(figsize=(10, 6))  
plt.scatter(x_axis1_1_2,residuals1_1_2, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线  
plt.xlabel('age(65-105)')  
plt.ylabel('H->L Residuals')  
plt.title('Residual Plot(H->L)')  
plt.grid(True)  
plt.show()

predictions_list_1_2 = [train_data_1_2['fitted'], test_data_1_2['predictions']]

data_1_2['predictions'] = pd.concat(predictions_list_1_2)


test_true_ny1_1_2 = data_1_2.groupby("t")['ny'].mean()   # Testing set true values
pred_ny1_1_2= data_1_2.groupby('t')['predictions'].mean()   # Predicted values

plt.figure(figsize=(10, 6))
plt.scatter(test_true_ny1_1_2.index, test_true_ny1_1_2, label='True Values', color='black')
plt.plot(pred_ny1_1_2.index, pred_ny1_1_2, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(H->L)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)

# Set x-axis ticks to specific years
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])

plt.show()

# 计算残差  
residuals2_1_2 = test_true_ny1_1_2 - pred_ny1_1_2

x_axis2_1_2 = test_true_ny1_1_2.index
  
# 绘制残差图  
plt.figure(figsize=(10, 6))  
plt.scatter(x_axis2_1_2,residuals2_1_2, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.xlabel('year(1998-2018)')  
plt.ylabel('H->L Residuals')  
plt.title('Residual Plot(H->L)')  
plt.grid(True)  
plt.show()


predictions_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx')
fitted_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx')
fitted_predictions_1_2 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx') 

test_data_1_2_reset_index = test_data_1_2.reset_index(drop=True)   
predictions_1_2['poisson_predictions'] = test_data_1_2_reset_index["predictions"]

fitted_1_2['poisson_fitted']=train_data_1_2["fitted"]
fitted_predictions_1_2['poisson_fitted_predictions']=data_1_2['predictions']


fitted_predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx', index=False)
predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx', index=False)
fitted_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx', index=False)


data_2_3= pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_3.csv')  

Y = 'sum_health2_3'  
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']  
  
# train_data_2_3, test_data_2_3 = train_test_split(data_2_3, test_size=0.2, random_state=42)  

train_data_2_3 = data_2_3 .loc[data_2_3['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_data_2_3= data_2_3.loc[data_2_3['time'].isin([14.5, 18])]  
   
exposure_train_2_3 = train_data_2_3['sum_lyear']  
exposure_test_2_3 = test_data_2_3['sum_lyear']  

X_train_2_3  = train_data_2_3 [X]
y_train_2_3  = train_data_2_3 [Y]
X_test_2_3  = test_data_2_3 [X]
y_test_2_3  = test_data_2_3 [Y]

# 对训练集和测试集进行预处理
X_train_processed_2_3  = preprocessor.fit_transform(X_train_2_3 )
X_test_processed_2_3  = preprocessor.transform(X_test_2_3 )

#创建Poisson回归模型  
poisson_model_2_3 = sm.Poisson(train_data_2_3[Y], sm.add_constant(X_train_processed_2_3), exposure=exposure_train_2_3)  
  
# 拟合模型  
poisson_results_2_3 = poisson_model_2_3.fit()  
print(poisson_results_2_3.summary())  

# 对训练集进行预测  
fitted_2_3 = poisson_results_2_3.predict(sm.add_constant(X_train_processed_2_3), exposure=exposure_train_2_3)  
train_data_2_3["fitted"] = fitted_2_3/exposure_train_2_3
  
# 对测试集进行预测  
predictions_2_3 = poisson_results_2_3.predict(sm.add_constant(X_test_processed_2_3), exposure=exposure_test_2_3)  
test_data_2_3["predictions"] = predictions_2_3/exposure_test_2_3

train_true_ny_2_3=train_data_2_3.groupby("age")['ny'].mean()  #训练集真实值
fitted_ny_2_3=train_data_2_3.groupby('age')['fitted'].mean()  #拟合值
test_true_ny_2_3=test_data_2_3.groupby("age")['ny'].mean()   #测试集真实值
pred_ny_2_3= test_data_2_3.groupby('age')['predictions'].mean()   #预测值

mse_2_3 = mean_squared_error(test_data_2_3['ny'], test_data_2_3['predictions'])
print(f"Mean Squared Error: {mse_2_3}")

sample_weight_2_3=exposure_test_2_3.values
y_true_2_3=test_data_2_3['ny'].values
y_pred_2_3=test_data_2_3["predictions"].values
# y_true_2_3 = np.array(y_true_2_3)  
# y_pred_2_3 = np.array(y_pred_2_3) 
deviance_per_obs_2_3 = 2 * (y_true_2_3 * np.log(y_true_2_3 / y_pred_2_3) - (y_true_2_3 - y_pred_2_3))
deviance_per_obs_2_3[y_true_2_3 == 0] =2*y_pred_2_3[y_true_2_3 == 0]
weighted_sum_2_3 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_3, sample_weight_2_3))  
exposure_weighted_average_2_3=weighted_sum_2_3/sample_weight_2_3.sum() 
exposure_weighted_average_2_3

# 真实值vs拟合值

plt.plot(train_true_ny_2_3.index, train_true_ny_2_3.values,color='blue', label='Number of real transfers')  #折线图
plt.plot(fitted_ny_2_3.index, fitted_ny_2_3.values,color='red', label='Fitted number of transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——训练集真实值与拟合值对比')
plt.legend()

# 真实值vs预测值

plt.plot(pred_ny_2_3.index, pred_ny_2_3.values,color='red', label='Predicted number of transfers')  #折线图
plt.plot(test_true_ny_2_3.index, test_true_ny_2_3.values,color='blue', label='Number of real transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——测试集真实值与预测值对比')
plt.legend()
plt.show()

# 计算残差  
residuals1_2_3 = test_true_ny_2_3 - pred_ny_2_3

x_axis1_2_3 = test_true_ny_2_3.index
  
# 绘制残差图  

plt.scatter(x_axis1_2_3,residuals1_2_3, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线  
plt.xlabel('age(65-105)')  
plt.ylabel('L->D Residuals')  
plt.title('Residual Plot(L->D)')  
plt.grid(True)  
plt.show()

predictions_list_2_3 = [train_data_2_3['fitted'], test_data_2_3['predictions']]

data_2_3['predictions'] = pd.concat(predictions_list_2_3)


test_true_ny1_2_3 = data_2_3.groupby("t")['ny'].mean()   # Testing set true values
pred_ny1_2_3= data_2_3.groupby('t')['predictions'].mean()   # Predicted values


plt.scatter(test_true_ny1_2_3.index, test_true_ny1_2_3, label='True Values', color='black')
plt.plot(pred_ny1_2_3.index, pred_ny1_2_3, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(L->D)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)

# Set x-axis ticks to specific years
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])

plt.show()

# 计算残差  
residuals2_2_3 = test_true_ny1_2_3 - pred_ny1_2_3

x_axis2_2_3 = test_true_ny1_2_3.index
  
# 绘制残差图  
 
plt.scatter(x_axis2_2_3,residuals2_2_3, alpha=0.5,label='Residuals')  
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

test_data_2_3_reset_index = test_data_2_3.reset_index(drop=True)   
predictions_2_3['poisson_predictions'] = test_data_2_3_reset_index["predictions"]

fitted_2_3['poisson_fitted']=train_data_2_3["fitted"]
fitted_predictions_2_3['poisson_fitted_predictions']=data_2_3['predictions']


fitted_predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx', index=False)
predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx', index=False)
fitted_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx', index=False)

data_2_1= pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_1.csv')  

Y = 'sum_health2_1'  
X = ['time', 'age', 'sex', 'residenc', 'marry', 'smoke', 'drink']  
  
# train_data_2_1, test_data_2_1 = train_test_split(data_2_1, test_size=0.2, random_state=42)  

train_data_2_1 = data_2_1 .loc[data_2_1['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_data_2_1= data_2_1.loc[data_2_1['time'].isin([14.5, 18])]  
   
exposure_train_2_1 = train_data_2_1['sum_lyear']  
exposure_test_2_1 = test_data_2_1['sum_lyear'] 

# 提取特征和目标变量
X_train_2_1  = train_data_2_1 [X]
y_train_2_1  = train_data_2_1 [Y]
X_test_2_1  = test_data_2_1 [X]
y_test_2_1  = test_data_2_1 [Y]

# 对训练集和测试集进行预处理
X_train_processed_2_1  = preprocessor.fit_transform(X_train_2_1 )
X_test_processed_2_1  = preprocessor.transform(X_test_2_1 )

#创建Poisson回归模型  
poisson_model_2_1 = sm.Poisson(train_data_2_1[Y], sm.add_constant(X_train_processed_2_1), exposure=exposure_train_2_1)  
  
# 拟合模型  
poisson_results_2_1 = poisson_model_2_1.fit()  
print(poisson_results_2_1.summary())  

# 对训练集进行预测  
fitted_2_1 = poisson_results_2_1.predict(sm.add_constant(X_train_processed_2_1), exposure=exposure_train_2_1)  
train_data_2_1["fitted"] = fitted_2_1/exposure_train_2_1
  
# 对测试集进行预测  
predictions_2_1 = poisson_results_2_1.predict(sm.add_constant(X_test_processed_2_1), exposure=exposure_test_2_1)  
test_data_2_1["predictions"] = predictions_2_1/exposure_test_2_1

train_true_ny_2_1=train_data_2_1.groupby("age")['ny'].mean()  #训练集真实值
fitted_ny_2_1=train_data_2_1.groupby('age')['fitted'].mean()  #拟合值
test_true_ny_2_1=test_data_2_1.groupby("age")['ny'].mean()   #测试集真实值
pred_ny_2_1= test_data_2_1.groupby('age')['predictions'].mean()   #预测值

mse_2_1 = mean_squared_error(test_data_2_1['ny'], test_data_2_1['predictions'])
print(f"Mean Squared Error: {mse_2_1}")

sample_weight_2_1=exposure_test_2_1.values
y_true_2_1=test_data_2_1['ny'].values
y_pred_2_1=test_data_2_1["predictions"].values
# y_true_2_1 = np.array(y_true_2_1)  
# y_pred_2_1 = np.array(y_pred_2_1) 
deviance_per_obs_2_1 = 2 * (y_true_2_1 * np.log(y_true_2_1 / y_pred_2_1) - (y_true_2_1 - y_pred_2_1))
deviance_per_obs_2_1[y_true_2_1 == 0] = 2*y_pred_2_1[y_true_2_1 == 0]
weighted_sum_2_1 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_1, sample_weight_2_1))  
exposure_weighted_average_2_1=weighted_sum_2_1/sample_weight_2_1.sum() 
exposure_weighted_average_2_1

# 真实值vs拟合值

plt.plot(train_true_ny_2_1.index, train_true_ny_2_1.values,color='blue', label='Number of real transfers')  #折线图
plt.plot(fitted_ny_2_1.index, fitted_ny_2_1.values,color='red', label='Fitted number of transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——训练集真实值与拟合值对比')
plt.legend()

# 真实值vs预测值

plt.plot(pred_ny_2_1.index, pred_ny_2_1.values,color='red', label='Predicted number of transfers')  #折线图
plt.plot(test_true_ny_2_1.index, test_true_ny_2_1.values,color='blue', label='Number of real transfers')  #折线图
plt.xlabel('年龄')
plt.ylabel('转移强度')
plt.title('泊松回归模型——测试集真实值与预测值对比')
plt.legend()
plt.show()

# 计算残差  
residuals1_2_1 = test_true_ny_2_1 - pred_ny_2_1

x_axis1_2_1 = test_true_ny_2_1.index
  
# 绘制残差图  
 
plt.scatter(x_axis1_2_1,residuals1_2_1, alpha=0.5,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线  
plt.xlabel('age(65-105)')  
plt.ylabel('L->H Residuals')  
plt.title('Residual Plot(L->H)')  
plt.grid(True)  
plt.show()

predictions_list_2_1 = [train_data_2_1['fitted'], test_data_2_1['predictions']]

data_2_1['predictions'] = pd.concat(predictions_list_2_1)


test_true_ny1_2_1 = data_2_1.groupby("t")['ny'].mean()   # Testing set true values
pred_ny1_2_1= data_2_1.groupby('t')['predictions'].mean()   # Predicted values


plt.scatter(test_true_ny1_2_1.index, test_true_ny1_2_1, label='True Values', color='black')
plt.plot(pred_ny1_2_1.index, pred_ny1_2_1, label='Predicted Values', color='red')
plt.title('Actual vs Predicted by Year(L->H)')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)

# Set x-axis ticks to specific years
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])

plt.show()

# 计算残差  
residuals2_2_1 = test_true_ny1_2_1 - pred_ny1_2_1

x_axis2_2_1 = test_true_ny1_2_1.index
  
# 绘制残差图  
 
plt.scatter(x_axis2_2_1,residuals2_2_1, alpha=0.5,label='Residuals')  
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

test_data_2_1_reset_index = test_data_2_1.reset_index(drop=True)   
predictions_2_1['poisson_predictions'] = test_data_2_1_reset_index["predictions"]

fitted_2_1['poisson_fitted']=train_data_2_1["fitted"]
fitted_predictions_2_1['poisson_fitted_predictions']=data_2_1["predictions"]


fitted_predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_1.xlsx', index=False)
predictions_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_1.xlsx', index=False)
fitted_2_1.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_1.xlsx', index=False)
