import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import poisson
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder  
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split  
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families import family, Poisson
import numpy as np  
import pandas as pd  
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
train_final_df = preprocessor.fit_transform(X_train)
test_final_df = preprocessor.transform(X_test)


# # 手动创建训练集的特征名称  
# train_feature_names = ['time', 'age']  
# for col in cat_features:  
#     train_unique_vals = train[col].unique()  
#     for val in train_unique_vals:  
#         train_feature_names.append(f'{col}_{val}') 
# #手动创建测试集的特征名称
# test_feature_names =['time', 'age'] 
# for col in cat_features:  
#     test_unique_vals = test[col].unique()  
#     for val in test_unique_vals:  
#         test_feature_names.append(f'{col}_{val}')
        
# # # 将编码后的数据转换为DataFrame  
# train_final_df = pd.DataFrame(X_train_processed, columns=train_feature_names)  
# test_final_df = pd.DataFrame(X_test_processed, columns=test_feature_names) 


# # 处理后的特征维度
input_shape = train_final_df.shape[1]
# # input_shape = X_train.shape[1]
# X_train_processed,train_feature_names,train_final_df

input_dim = train_final_df.shape[1]
# 定义模型  
K.clear_session()
tf.random.set_seed(42) 
model = Sequential()  
# model.add(Input(shape=(input_dim,)))  
# model.add(tf.keras.layers.Input(shape=(input_dim,)))
model.add(BatchNormalization(input_shape=(input_dim,)))
# model.add(Dense(80, input_dim=train_final_df.shape[1], activation='selu'))  
model.add(Dense(80, activation='selu'))  
model.add(Dense(80, activation='selu'))  
model.add(Dense(80, activation='selu'))  
model.add(Dropout(0.1))  # 10% dropout  
# model.add(Dense(1))  # 回归问题，不需要激活函数  
model.add(Dense(1,activation="exponential")) 
# model.add(Dense(1,activation="linear"))   

# # # 编译模型  编译神经网络模型，定义模型训练时的优化器、损失函数和评估指标
model.compile(optimizer=Adam(), loss='poisson',metrics=['poisson'])  
# # 训练模型  
history=model.fit(train_final_df, y_train, sample_weight=w_train, epochs=100, batch_size=64, verbose=1)  

# model.save_weights('model_weights.h5')


plt.plot(history.history['loss'])
plt.show()

#拟合值
y_fitted = model.predict(train_final_df)
train["fitted"] = y_fitted  
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

y_pred1=model.predict(test_final_df)
# y_pred1

test["predictions"]=y_pred1
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

sample_weight=w_test
y_true=y_test.values
y_pred=test["predictions"].values
y_true = np.array(y_true)  
y_pred = np.array(y_pred) 
deviance_per_obs = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
deviance_per_obs[y_true == 0] = 2*y_pred[y_true == 0]
weighted_sum = tf.reduce_sum(tf.multiply(deviance_per_obs, sample_weight))  
exposure_weighted_average=weighted_sum/sample_weight.sum() 
exposure_weighted_average

mse = mean_squared_error(y_test,y_pred1)
print(f"Mean Squared Error: {mse}")

predictions_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx')
fitted_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx')
fitted_predictions_1_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx') 

test_reset_index = test.reset_index(drop=True)   
predictions_1_3['net_predictions'] = test_reset_index["predictions"]

fitted_1_3['net_fitted']=train["fitted"]
fitted_predictions_1_3['net_fitted_predictions']=df['predictions']


fitted_predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx', index=False)
predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx', index=False)
fitted_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx', index=False)

new_data_sigma = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组11_2.csv')  
train_1_2 = new_data_sigma .loc[new_data_sigma['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_1_2= new_data_sigma.loc[new_data_sigma['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_1_2  = train_1_2 [X]
y_train_1_2  = train_1_2 [Y]
w_train_1_2  = train_1_2 ['sum_hyear']
X_test_1_2  = test_1_2 [X]
y_test_1_2  = test_1_2 [Y]
w_test_1_2  = test_1_2 ['sum_hyear']

# 对训练集和测试集进行预处理
train_final_df_1_2  = preprocessor.fit_transform(X_train_1_2 )
test_final_df_1_2  = preprocessor.transform(X_test_1_2 )
# 处理后的特征维度
input_shape = train_final_df_1_2 .shape[1]
# input_shape = X_train_1_2.shape[1]


# train_feature_names_1_2 =['time', 'age']  
# for col in cat_features:  
#     train_unique_vals = train_1_2[col].unique()  
#     for val in train_unique_vals:  
#         train_feature_names_1_2.append(f'{col}_{val}') 
# #手动创建测试集的特征名称
# test_feature_names_1_2 = ['time', 'age'] 
# for col in cat_features:  
#     test_unique_vals = test_1_2[col].unique()  
#     for val in test_unique_vals:  
#         test_feature_names_1_2.append(f'{col}_{val}')
        
# # 将编码后的数据转换为DataFrame  
# train_final_df_1_2 = pd.DataFrame(X_train_processed_1_2, columns=train_feature_names_1_2)  
# test_final_df_1_2 = pd.DataFrame(X_test_processed_1_2, columns=test_feature_names_1_2) 
train_final_df_1_2


#1_2
# 使用迁移学习的权重初始化新模型  
tf.random.set_seed(42) 
input_dim = train_final_df_1_2.shape[1]
model_sigma = Sequential()  
# model_sigma.add(Input(shape=(input_dim,)))  
model_sigma.add(BatchNormalization(input_shape=(input_dim,)))
# model_sigma.add(Dense(80, input_dim=train_final_df_1_2.shape[1], activation='selu'))  
model_sigma.add(Dense(80, activation='selu')) 
model_sigma.add(Dense(80, activation='selu'))  
model_sigma.add(Dense(80, activation='selu'))  
model_sigma.add(Dropout(0.1))  
model_sigma.add(Dense(1,activation="exponential")) 

# 设置权重    迁移  
original_weights = model.get_weights() 
model_sigma.set_weights(original_weights)  

# 编译新模型 
model_sigma.compile(optimizer=Adam(), loss='poisson',metrics=['poisson'])  # 或者其他合适的损失函数  
  
# 训练新模型  
history_sigma = model_sigma.fit(train_final_df_1_2, y_train_1_2, sample_weight=w_train_1_2, epochs=100, batch_size=64, verbose=1)


# 评估新模型  
test_loss_sigma = model_sigma.evaluate(test_final_df_1_2, y_test_1_2)  
print(f'Test Loss for Sigma Model: {test_loss_sigma}') 

# 可视化 sigma 模型的训练损失  
plt.plot(history_sigma.history['loss'], label='Training Loss')  
plt.show() 

# 对训练集进行预测  
y_fitted_1_2 = model_sigma.predict(train_final_df_1_2)
train_1_2["fitted_1_2"] = y_fitted_1_2  
y_true_train_1_2=train_1_2.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_1_2= train_1_2.groupby('age')['fitted_1_2'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_1_2.index, y_true_train_1_2, label='True Values',color='blue')
plt.plot(y_fitted_train_1_2.index, y_fitted_train_1_2, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(H->L)')
plt.xlabel('Age(65-105)')
plt.ylabel('H->L intensity')
plt.legend()
plt.grid(True)
plt.show()

 # 进行预测
predictions_sigma = model_sigma.predict(test_final_df_1_2)  
predictions_sigma

sample_weight_1_2=w_test_1_2
test_1_2["predictions"]=predictions_sigma
# ['sum_health1_3']/['sum_year']
y_true1_1_2=test_1_2.groupby("age")['ny'].mean()   #测试集真实值
y_pred1_1_2= test_1_2.groupby('age')['predictions'].mean()   #预测值
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
 
plt.scatter(x_axis1_1_2,residuals1_1_2,label='Residuals')  
plt.axhline(y=0, linestyle='--', label='y=0',color='red') # 绘制y=0的直线 
plt.xlabel('age(65-105)')  
plt.ylabel('H->L Residuals')  
plt.title('Residual Plot(H->L)')  
plt.grid(True)  
plt.show()

# test_1_2['sex_label'] = test_1_2['sex'].map({1: 'male', 2: 'female'}) 
# y_pred1_1_2_sex= test_1_2.groupby(['age','sex_label'])['predictions'].mean().reset_index()   #预测值

# male_data_1_2 = y_pred1_1_2_sex[y_pred1_1_2_sex['sex_label'] == 'male']  
# female_data_1_2 = y_pred1_1_2_sex[y_pred1_1_2_sex['sex_label'] == 'female'] 

# # 绘制男性的预测值
# plt.plot(male_data_1_2['age'], male_data_1_2['predictions'], color='blue', linestyle='--',label='male')  
# # 绘制女性的预测值   
# plt.plot(female_data_1_2['age'], female_data_1_2['predictions'], color='red', linestyle='--',label='female')  
# # 绘制平均值  
# plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Mean', color='black')
# plt.title('H->L by sex')
# plt.xlabel('t')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.show()

# test_1_2['residenc_label'] = test_1_2['residenc'].map({1: 'rural', 2: 'urban'}) 
# y_pred1_1_2_residenc= test_1_2.groupby(['age','residenc_label'])['predictions'].mean().reset_index()   #预测值

# rural_data_1_2 = y_pred1_1_2_residenc[y_pred1_1_2_residenc['residenc_label'] == 'rural']  
# urban_data_1_2 = y_pred1_1_2_residenc[y_pred1_1_2_residenc['residenc_label'] == 'urban'] 


# # 绘制农村的预测值
# plt.plot(rural_data_1_2['age'], rural_data_1_2['predictions'], color='blue', linestyle='--',label='rural')  
# # 绘制城镇的预测值   
# plt.plot(urban_data_1_2['age'], urban_data_1_2['predictions'], color='red', linestyle='--',label='urban')  
# # 绘制平均值  
# plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Mean', color='black')

# plt.title('H->L by residency')
# plt.xlabel('Age')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.show()

# test_1_2['marry_label'] = test_1_2['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
# y_pred1_1_2_marry= test_1_2.groupby(['age','marry_label'])['predictions'].mean().reset_index()   #预测值

# with_spouse_data_1_2 = y_pred1_1_2_marry[y_pred1_1_2_marry['marry_label'] == 'with spouse']  
# without_spouse_data_1_2 = y_pred1_1_2_marry[y_pred1_1_2_marry['marry_label'] == 'without spouse'] 


# # 绘制有配偶的预测值
# plt.plot(with_spouse_data_1_2['age'],with_spouse_data_1_2['predictions'], color='blue', linestyle='--',label='with spouse')  
# # 绘制没有配偶的预测值   
# plt.plot(without_spouse_data_1_2['age'],without_spouse_data_1_2['predictions'], color='red', linestyle='--',label='without spouse')  
# # 绘制平均值  
# plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Mean', color='black')

# plt.title('H->L by marital status')
# plt.xlabel('Age')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.show()

# test_1_2['smoking_label'] = test_1_2['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
# y_pred1_1_2_smoking= test_1_2.groupby(['age','smoking_label'])['predictions'].mean().reset_index()   #预测值

# smoking_data_1_2 = y_pred1_1_2_smoking[y_pred1_1_2_smoking['smoking_label'] == 'smoking']  
# not_smoking_data_1_2 = y_pred1_1_2_smoking[y_pred1_1_2_smoking['smoking_label'] == 'not smoking'] 


# # 绘制吸烟的预测值
# plt.plot(smoking_data_1_2['age'], smoking_data_1_2['predictions'], color='blue', linestyle='--',label='smoking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_smoking_data_1_2['age'], not_smoking_data_1_2['predictions'], color='red', linestyle='--',label='not smoking')  
# # 绘制平均值  
# plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Mean', color='black')

# plt.title('H->L by smoking')
# plt.xlabel('Age')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.show()

# test_1_2['drinking_label'] = test_1_2['drink'].map({1: 'drinking', 2: 'not drinking'}) 
# y_pred1_1_2_drinking= test_1_2.groupby(['age','drinking_label'])['predictions'].mean().reset_index()   #预测值

# drinking_data_1_2 = y_pred1_1_2_drinking[y_pred1_1_2_drinking['drinking_label'] == 'drinking']  
# not_drinking_data_1_2 = y_pred1_1_2_drinking[y_pred1_1_2_drinking['drinking_label'] == 'not drinking']


# # 绘制吸烟的预测值
# plt.plot(drinking_data_1_2['age'], drinking_data_1_2['predictions'], color='blue', linestyle='--',label='drinking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_drinking_data_1_2['age'], not_drinking_data_1_2['predictions'], color='red', linestyle='--',label='not drinking')  
# # 绘制平均值  
# plt.plot(y_pred1_1_2.index, y_pred1_1_2, label='Mean', color='black')

# plt.title('H->L by drinking')
# plt.xlabel('Age')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.show()

predictions_list_1_2 = [train_1_2['fitted_1_2'], test_1_2['predictions']]
new_data_sigma['predictions'] = pd.concat(predictions_list_1_2)

y_true2_1_2=new_data_sigma.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_1_2= new_data_sigma.groupby('t')['predictions'].mean()   #预测值
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

# #性别
# new_data_sigma['sex_label'] = new_data_sigma['sex'].map({1: 'male', 2: 'female'}) 
# y_pred2_1_2_sex= new_data_sigma.groupby(['t','sex_label'])['predictions'].mean().reset_index()   #预测值
# male_data1_1_2 = y_pred2_1_2_sex[y_pred2_1_2_sex['sex_label'] == 'male']  
# female_data1_1_2 = y_pred2_1_2_sex[y_pred2_1_2_sex['sex_label'] == 'female'] 

# # 绘制男性的预测值
# plt.plot(male_data1_1_2['t'], male_data1_1_2['predictions'], color='blue', linestyle='--',label='male')  
# # 绘制女性的预测值   
# plt.plot(female_data1_1_2['t'], female_data1_1_2['predictions'], color='red', linestyle='--',label='female')  
# # 绘制平均值  
# plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Mean', color='black')
# plt.title('H->L by sex')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()


# #居住地
# new_data_sigma['residenc_label'] = new_data_sigma['residenc'].map({1: 'rural', 2: 'urban'}) 
# y_pred2_1_2_residenc= new_data_sigma.groupby(['t','residenc_label'])['predictions'].mean().reset_index()   #预测值
# rural_data1_1_2 = y_pred2_1_2_residenc[y_pred2_1_2_residenc['residenc_label'] == 'rural']  
# urban_data1_1_2 = y_pred2_1_2_residenc[y_pred2_1_2_residenc['residenc_label'] == 'urban'] 


# # 绘制农村的预测值
# plt.plot(rural_data1_1_2['t'], rural_data1_1_2['predictions'], color='blue', linestyle='--',label='rural')  
# # 绘制城镇的预测值   
# plt.plot(urban_data1_1_2['t'], urban_data1_1_2['predictions'], color='red', linestyle='--',label='urban')  
# # 绘制平均值  
# plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Mean', color='black')
# plt.title('H->L by residency')
# plt.xlabel('t')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #婚姻情况
# new_data_sigma['marry_label'] = new_data_sigma['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
# y_pred2_1_2_marry= new_data_sigma.groupby(['t','marry_label'])['predictions'].mean().reset_index()   #预测值
# with_spouse_data1_1_2 = y_pred2_1_2_marry[y_pred2_1_2_marry['marry_label'] == 'with spouse']  
# without_spouse_data1_1_2 = y_pred2_1_2_marry[y_pred2_1_2_marry['marry_label'] == 'without spouse'] 


# # 绘制有配偶的预测值
# plt.plot(with_spouse_data1_1_2['t'],with_spouse_data1_1_2['predictions'], color='blue', linestyle='--',label='with spouse')  
# # 绘制没有配偶的预测值   
# plt.plot(without_spouse_data1_1_2['t'],without_spouse_data1_1_2['predictions'], color='red', linestyle='--',label='without spouse')  
# # 绘制平均值  
# plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Mean', color='black')
# plt.title('H->L by marital status')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #吸烟情况
# new_data_sigma['smoking_label'] = new_data_sigma['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
# y_pred2_1_2_smoking= new_data_sigma.groupby(['t','smoking_label'])['predictions'].mean().reset_index()   #预测值
# smoking_data1_1_2 = y_pred2_1_2_smoking[y_pred2_1_2_smoking['smoking_label'] == 'smoking']  
# not_smoking_data1_1_2 = y_pred2_1_2_smoking[y_pred2_1_2_smoking['smoking_label'] == 'not smoking'] 


# # 绘制吸烟的预测值
# plt.plot(smoking_data1_1_2['t'], smoking_data1_1_2['predictions'], color='blue', linestyle='--',label='smoking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_smoking_data1_1_2['t'], not_smoking_data1_1_2['predictions'], color='red', linestyle='--',label='not smoking')  
# # 绘制平均值  
# plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Mean', color='black')
# plt.title('H->L by smoking')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #饮酒情况
# new_data_sigma['drinking_label'] = new_data_sigma['drink'].map({1: 'drinking', 2: 'not drinking'}) 
# y_pred2_1_2_drinking= new_data_sigma.groupby(['t','drinking_label'])['predictions'].mean().reset_index()   #预测值
# drinking_data1_1_2 = y_pred2_1_2_drinking[y_pred2_1_2_drinking['drinking_label'] == 'drinking']  
# not_drinking_data1_1_2 = y_pred2_1_2_drinking[y_pred2_1_2_drinking['drinking_label'] == 'not drinking']

# # 绘制吸烟的预测值
# plt.plot(drinking_data1_1_2['t'], drinking_data1_1_2['predictions'], color='blue', linestyle='--',label='drinking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_drinking_data1_1_2['t'], not_drinking_data1_1_2['predictions'], color='red', linestyle='--',label='not drinking')  
# # 绘制平均值  
# plt.plot(y_pred2_1_2.index, y_pred2_1_2, label='Mean', color='black')
# plt.title('H->L by drinking')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('H->L intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

sample_weight_1_2=w_test_1_2
y_true_1_2=y_test_1_2.values
y_pred_1_2=test_1_2["predictions"].values
# y_true_1_2 = np.array(y_true_1_2)  
# y_pred_1_2 = np.array(y_pred_1_2) 
deviance_per_obs_1_2 = 2 * (y_true_1_2 * np.log(y_true_1_2 / y_pred_1_2) - (y_true_1_2 - y_pred_1_2))
deviance_per_obs_1_2[y_true_1_2 == 0] =2* y_pred_1_2[y_true_1_2 == 0]
weighted_sum_1_2 = tf.reduce_sum(tf.multiply(deviance_per_obs_1_2, sample_weight_1_2))  
exposure_weighted_average_1_2=weighted_sum_1_2/sample_weight_1_2.sum() 
exposure_weighted_average_1_2

mse = mean_squared_error(y_test_1_2,y_pred_1_2)
print(f"Mean Squared Error: {mse}")

predictions_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx')
fitted_1_2= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx')
fitted_predictions_1_2 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx') 

test_1_2_reset_index = test_1_2.reset_index(drop=True)   
predictions_1_2['net_predictions'] = test_1_2_reset_index["predictions"]

fitted_1_2['net_fitted']=train_1_2["fitted_1_2"]
fitted_predictions_1_2['net_fitted_predictions']=new_data_sigma['predictions']


fitted_predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_2.xlsx', index=False)
predictions_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_2.xlsx', index=False)
fitted_1_2.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_2.xlsx', index=False)

new_data_nu = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_3.csv')  
train_2_3 = new_data_nu .loc[new_data_nu['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_3= new_data_nu.loc[new_data_nu['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_2_3  = train_2_3 [X]
y_train_2_3  = train_2_3 [Y]
w_train_2_3  = train_2_3 ['sum_lyear']
X_test_2_3  = test_2_3 [X]
y_test_2_3  = test_2_3 [Y]
w_test_2_3  = test_2_3 ['sum_lyear']

# 对训练集和测试集进行预处理
train_final_df_2_3  = preprocessor.fit_transform(X_train_2_3 )
test_final_df_2_3  = preprocessor.transform(X_test_2_3 )
# 处理后的特征维度
input_shape = train_final_df_2_3 .shape[1]
# input_shape = X_train_2_3.shape[1]


# train_feature_names_2_3 =['time', 'age']  
# for col in cat_features:  
#     train_unique_vals = train_2_3[col].unique()  
#     for val in train_unique_vals:  
#         train_feature_names_2_3.append(f'{col}_{val}') 
# #手动创建测试集的特征名称
# test_feature_names_2_3 = ['time', 'age'] 
# for col in cat_features:  
#     test_unique_vals = test_2_3[col].unique()  
#     for val in test_unique_vals:  
#         test_feature_names_2_3.append(f'{col}_{val}')
        
# # 将编码后的数据转换为DataFrame  
# train_final_df_2_3 = pd.DataFrame(X_train_processed_2_3, columns=train_feature_names_2_3)  
# test_final_df_2_3 = pd.DataFrame(X_test_processed_2_3, columns=test_feature_names_2_3) 
# X_train_processed_2_3,train_feature_names_2_3,train_final_df_2_3


#2_3
# 对ν模型做类似的操作 
tf.random.set_seed(42) 
input_dim = train_final_df_2_3.shape[1]

model_nu = Sequential()  
# model_sigma.add(Input(shape=(input_dim,)))  
model_nu.add(BatchNormalization(input_shape=(input_dim,)))
# model_sigma.add(Dense(80, input_dim=train_final_df_2_3.shape[1], activation='selu'))  
model_nu.add(Dense(80, activation='selu')) 
model_nu.add(Dense(80, activation='selu'))  
model_nu.add(Dense(80, activation='selu'))  
model_nu.add(Dropout(0.1))  
model_nu.add(Dense(1,activation="exponential"))  

# 设置权重  
original_weights = model.get_weights()  
model_nu.set_weights(original_weights) 



# 编译新模型  
model_nu.compile(optimizer=Adam(), loss='poisson',metrics=['poisson'])  # 或者其他合适的损失函数  
  
# 训练新模型  
history_nu = model_nu.fit(train_final_df_2_3, y_train_2_3, sample_weight=w_train_2_3, epochs=100, batch_size=64, verbose=1)


# 评估新模型  
test_loss_nu = model_nu.evaluate(test_final_df_2_3,y_test_2_3)  
print(f'Test Loss for Nu Model: {test_loss_nu}')

# 可视化nu模型的训练损失  
plt.plot(history_nu.history['loss'], label='Training Loss')  
plt.show() 

# 对训练集进行预测  
y_fitted_2_3 = model_nu.predict(train_final_df_2_3)
train_2_3["fitted_2_3"] = y_fitted_2_3  
y_true_train_2_3=train_2_3.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_3= train_2_3.groupby('age')['fitted_2_3'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_3.index, y_true_train_2_3, label='True Values',color='blue')
plt.plot(y_fitted_train_2_3.index, y_fitted_train_2_3, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->D)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

 # 进行预测
predictions_nu = model_nu.predict(test_final_df_2_3)  
predictions_nu

sample_weight_2_3=w_test_2_3
test_2_3["predictions"]=predictions_nu
# ['sum_health1_3']/['sum_year']
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

test_2_3['sex_label'] = test_2_3['sex'].map({1: 'male', 2: 'female'}) 
y_pred1_2_3_sex= test_2_3.groupby(['age','sex_label'])['predictions'].mean().reset_index()   #预测值

male_data_2_3 = y_pred1_2_3_sex[y_pred1_2_3_sex['sex_label'] == 'male']  
female_data_2_3 = y_pred1_2_3_sex[y_pred1_2_3_sex['sex_label'] == 'female'] 

# 绘制男性的预测值
plt.plot(male_data_2_3['age'], male_data_2_3['predictions'], color='blue', linestyle='--',label='male')  
# 绘制女性的预测值   
plt.plot(female_data_2_3['age'], female_data_2_3['predictions'], color='red', linestyle='--',label='female')  
# 绘制平均值  
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Mean', color='black')
plt.title('L->D by sex')
plt.xlabel('Age')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()


test_2_3['residenc_label'] = test_2_3['residenc'].map({1: 'rural', 2: 'urban'}) 
y_pred1_2_3_residenc= test_2_3.groupby(['age','residenc_label'])['predictions'].mean().reset_index()   #预测值

rural_data_2_3 = y_pred1_2_3_residenc[y_pred1_2_3_residenc['residenc_label'] == 'rural']  
urban_data_2_3 = y_pred1_2_3_residenc[y_pred1_2_3_residenc['residenc_label'] == 'urban'] 


# 绘制农村的预测值
plt.plot(rural_data_2_3['age'], rural_data_2_3['predictions'], color='blue', linestyle='--',label='rural')  
# 绘制城镇的预测值   
plt.plot(urban_data_2_3['age'], urban_data_2_3['predictions'], color='red', linestyle='--',label='urban')  
# 绘制平均值  
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Mean', color='black')

plt.title('L->D by residency')
plt.xlabel('Age')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_3['marry_label'] = test_2_3['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
y_pred1_2_3_marry= test_2_3.groupby(['age','marry_label'])['predictions'].mean().reset_index()   #预测值

with_spouse_data_2_3 = y_pred1_2_3_marry[y_pred1_2_3_marry['marry_label'] == 'with spouse']  
without_spouse_data_2_3 = y_pred1_2_3_marry[y_pred1_2_3_marry['marry_label'] == 'without spouse'] 


# 绘制有配偶的预测值
plt.plot(with_spouse_data_2_3['age'],with_spouse_data_2_3['predictions'], color='blue', linestyle='--',label='with spouse')  
# 绘制没有配偶的预测值   
plt.plot(without_spouse_data_2_3['age'],without_spouse_data_2_3['predictions'], color='red', linestyle='--',label='without spouse')  
# 绘制平均值  
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Mean', color='black')

plt.title('L->D by marital status')
plt.xlabel('Age')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_3['smoking_label'] = test_2_3['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
y_pred1_2_3_smoking= test_2_3.groupby(['age','smoking_label'])['predictions'].mean().reset_index()   #预测值

smoking_data_2_3 = y_pred1_2_3_smoking[y_pred1_2_3_smoking['smoking_label'] == 'smoking']  
not_smoking_data_2_3 = y_pred1_2_3_smoking[y_pred1_2_3_smoking['smoking_label'] == 'not smoking'] 


# 绘制吸烟的预测值
plt.plot(smoking_data_2_3['age'], smoking_data_2_3['predictions'], color='blue', linestyle='--',label='smoking')  
# 绘制不吸烟的预测值   
plt.plot(not_smoking_data_2_3['age'], not_smoking_data_2_3['predictions'], color='red', linestyle='--',label='not smoking')  
# 绘制平均值  
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Mean', color='black')

plt.title('L->D by smoking')
plt.xlabel('Age')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_3['drinking_label'] = test_2_3['drink'].map({1: 'drinking', 2: 'not drinking'}) 
y_pred1_2_3_drinking= test_2_3.groupby(['age','drinking_label'])['predictions'].mean().reset_index()   #预测值

drinking_data_2_3 = y_pred1_2_3_drinking[y_pred1_2_3_drinking['drinking_label'] == 'drinking']  
not_drinking_data_2_3 = y_pred1_2_3_drinking[y_pred1_2_3_drinking['drinking_label'] == 'not drinking']


# 绘制吸烟的预测值
plt.plot(drinking_data_2_3['age'], drinking_data_2_3['predictions'], color='blue', linestyle='--',label='drinking')  
# 绘制不吸烟的预测值   
plt.plot(not_drinking_data_2_3['age'], not_drinking_data_2_3['predictions'], color='red', linestyle='--',label='not drinking')  
# 绘制平均值  
plt.plot(y_pred1_2_3.index, y_pred1_2_3, label='Mean', color='black')

plt.title('L->D by drinking')
plt.xlabel('Age')
plt.ylabel('L->D intensity')
plt.legend()
plt.grid(True)
plt.show()

predictions_list_2_3 = [train_2_3["fitted_2_3"] , test_2_3['predictions']]
new_data_nu['predictions'] = pd.concat(predictions_list_2_3)

y_true2_2_3=new_data_nu.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_3= new_data_nu.groupby('t')['predictions'].mean()   #预测值
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

# #性别
# new_data_nu['sex_label'] = new_data_nu['sex'].map({1: 'male', 2: 'female'}) 
# y_pred2_2_3_sex= new_data_nu.groupby(['t','sex_label'])['predictions'].mean().reset_index()   #预测值
# male_data1_2_3 = y_pred2_2_3_sex[y_pred2_2_3_sex['sex_label'] == 'male']  
# female_data1_2_3 = y_pred2_2_3_sex[y_pred2_2_3_sex['sex_label'] == 'female'] 

# 
# # 绘制男性的预测值
# plt.plot(male_data1_2_3['t'], male_data1_2_3['predictions'], color='blue', linestyle='--',label='male')  
# # 绘制女性的预测值   
# plt.plot(female_data1_2_3['t'], female_data1_2_3['predictions'], color='red', linestyle='--',label='female')  
# # 绘制平均值  
# plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Mean', color='black')
# plt.title('L->D by sex')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('L->D intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()


# #居住地
# new_data_nu['residenc_label'] = new_data_nu['residenc'].map({1: 'rural', 2: 'urban'}) 
# y_pred2_2_3_residenc= new_data_nu.groupby(['t','residenc_label'])['predictions'].mean().reset_index()   #预测值
# rural_data1_2_3 = y_pred2_2_3_residenc[y_pred2_2_3_residenc['residenc_label'] == 'rural']  
# urban_data1_2_3 = y_pred2_2_3_residenc[y_pred2_2_3_residenc['residenc_label'] == 'urban'] 

# 
# # 绘制农村的预测值
# plt.plot(rural_data1_2_3['t'], rural_data1_2_3['predictions'], color='blue', linestyle='--',label='rural')  
# # 绘制城镇的预测值   
# plt.plot(urban_data1_2_3['t'], urban_data1_2_3['predictions'], color='red', linestyle='--',label='urban')  
# # 绘制平均值  
# plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Mean', color='black')
# plt.title('L->D by residency')
# plt.xlabel('t')
# plt.ylabel('L->D intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #婚姻情况
# new_data_nu['marry_label'] = new_data_nu['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
# y_pred2_2_3_marry= new_data_nu.groupby(['t','marry_label'])['predictions'].mean().reset_index()   #预测值
# with_spouse_data1_2_3 = y_pred2_2_3_marry[y_pred2_2_3_marry['marry_label'] == 'with spouse']  
# without_spouse_data1_2_3 = y_pred2_2_3_marry[y_pred2_2_3_marry['marry_label'] == 'without spouse'] 

# 
# # 绘制有配偶的预测值
# plt.plot(with_spouse_data1_2_3['t'],with_spouse_data1_2_3['predictions'], color='blue', linestyle='--',label='with spouse')  
# # 绘制没有配偶的预测值   
# plt.plot(without_spouse_data1_2_3['t'],without_spouse_data1_2_3['predictions'], color='red', linestyle='--',label='without spouse')  
# # 绘制平均值  
# plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Mean', color='black')
# plt.title('L->D by marital status')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('L->D intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #吸烟情况
# new_data_nu['smoking_label'] = new_data_nu['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
# y_pred2_2_3_smoking= new_data_nu.groupby(['t','smoking_label'])['predictions'].mean().reset_index()   #预测值
# smoking_data1_2_3 = y_pred2_2_3_smoking[y_pred2_2_3_smoking['smoking_label'] == 'smoking']  
# not_smoking_data1_2_3 = y_pred2_2_3_smoking[y_pred2_2_3_smoking['smoking_label'] == 'not smoking'] 

# 
# # 绘制吸烟的预测值
# plt.plot(smoking_data1_2_3['t'], smoking_data1_2_3['predictions'], color='blue', linestyle='--',label='smoking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_smoking_data1_2_3['t'], not_smoking_data1_2_3['predictions'], color='red', linestyle='--',label='not smoking')  
# # 绘制平均值  
# plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Mean', color='black')
# plt.title('L->D by smoking')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('L->D intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

# #饮酒情况
# new_data_nu['drinking_label'] = new_data_nu['drink'].map({1: 'drinking', 2: 'not drinking'}) 
# y_pred2_2_3_drinking= new_data_nu.groupby(['t','drinking_label'])['predictions'].mean().reset_index()   #预测值
# drinking_data1_2_3 = y_pred2_2_3_drinking[y_pred2_2_3_drinking['drinking_label'] == 'drinking']  
# not_drinking_data1_2_3 = y_pred2_2_3_drinking[y_pred2_2_3_drinking['drinking_label'] == 'not drinking']
# 
# # 绘制吸烟的预测值
# plt.plot(drinking_data1_2_3['t'], drinking_data1_2_3['predictions'], color='blue', linestyle='--',label='drinking')  
# # 绘制不吸烟的预测值   
# plt.plot(not_drinking_data1_2_3['t'], not_drinking_data1_2_3['predictions'], color='red', linestyle='--',label='not drinking')  
# # 绘制平均值  
# plt.plot(y_pred2_2_3.index, y_pred2_2_3, label='Mean', color='black')
# plt.title('L->D by drinking')
# plt.xlabel('Year (1998-2018)')
# plt.ylabel('L->D intensity')
# plt.legend()
# plt.grid(True)
# plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
# plt.show()

sample_weight_2_3=w_test_2_3
y_true_2_3=y_test_2_3.values
y_pred_2_3=test_2_3["predictions"].values
# y_true_2_3 = np.array(y_true_2_3)  
# y_pred_2_3 = np.array(y_pred_2_3) 
deviance_per_obs_2_3 = 2 * (y_true_2_3 * np.log(y_true_2_3 / y_pred_2_3) - (y_true_2_3 - y_pred_2_3))
deviance_per_obs_2_3[y_true_2_3 == 0] =2*y_pred_2_3[y_true_2_3 == 0]
weighted_sum_2_3 = tf.reduce_sum(tf.multiply(deviance_per_obs_2_3, sample_weight_2_3))  
exposure_weighted_average_2_3=weighted_sum_2_3/sample_weight_2_3.sum() 
exposure_weighted_average_2_3

mse = mean_squared_error(y_test_2_3,y_pred_2_3)
print(f"Mean Squared Error: {mse}")

predictions_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx')
fitted_2_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx')
fitted_predictions_2_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx') 

test_2_3_reset_index = test_2_3.reset_index(drop=True)   
predictions_2_3['net_predictions'] = test_2_3_reset_index["predictions"]

fitted_2_3['net_fitted']=train_2_3["fitted_2_3"]
fitted_predictions_2_3['net_fitted_predictions']=new_data_nu['predictions']


fitted_predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_2_3.xlsx', index=False)
predictions_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_2_3.xlsx', index=False)
fitted_2_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_2_3.xlsx', index=False)

new_data_2_1 = pd.read_csv(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\分组22_1.csv')  
train_2_1 = new_data_2_1 .loc[new_data_2_1['time'].isin([1, 3, 5.5, 8.5, 11.5])]  
test_2_1= new_data_2_1.loc[new_data_2_1['time'].isin([14.5, 18])] 

# 提取特征和目标变量
X_train_2_1  = train_2_1 [X]
y_train_2_1  = train_2_1 [Y]
w_train_2_1  = train_2_1 ['sum_lyear']
X_test_2_1  = test_2_1 [X]
y_test_2_1  = test_2_1 [Y]
w_test_2_1  = test_2_1 ['sum_lyear']

# 对训练集和测试集进行预处理
train_final_df_2_1  = preprocessor.fit_transform(X_train_2_1 )
test_final_df_2_1  = preprocessor.transform(X_test_2_1 )
# 处理后的特征维度
input_shape = train_final_df_2_1 .shape[1]
# input_shape = X_train_2_1.shape[1]


# train_feature_names_2_1 =['time', 'age']  
# for col in cat_features:  
#     train_unique_vals = train_2_1[col].unique()  
#     for val in train_unique_vals:  
#         train_feature_names_2_1.append(f'{col}_{val}') 
# #手动创建测试集的特征名称
# test_feature_names_2_1 = ['time', 'age'] 
# for col in cat_features:  
#     test_unique_vals = test_2_1[col].unique()  
#     for val in test_unique_vals:  
#         test_feature_names_2_1.append(f'{col}_{val}')
        
# # 将编码后的数据转换为DataFrame  
# train_final_df_2_1 = pd.DataFrame(X_train_processed_2_1, columns=train_feature_names_2_1)  
# test_final_df_2_1 = pd.DataFrame(X_test_processed_2_1, columns=test_feature_names_2_1) 
# X_train_processed_2_1,train_feature_names_2_1,train_final_df_2_1


#2_1
tf.random.set_seed(42) 
input_dim = train_final_df_2_1.shape[1]

model_2_1 = Sequential()  
# model_2_1.add(Input(shape=(input_dim,)))  
model_2_1.add(BatchNormalization(input_shape=(input_dim,)))
# model_2_1.add(Dense(80, input_dim=train_final_df_2_1.shape[1], activation='selu'))  
model_2_1.add(Dense(80, activation='selu')) 
model_2_1.add(Dense(80, activation='selu'))  
model_2_1.add(Dense(80, activation='selu'))  
model_2_1.add(Dropout(0.1))  
model_2_1.add(Dense(1,activation="exponential"))  

# 设置权重  
original_weights = model.get_weights()  
model_2_1.set_weights(original_weights) 

# 编译新模型  
model_2_1.compile(optimizer=Adam(), loss='poisson',metrics=['poisson'])  # 或者其他合适的损失函数  
  
# 训练新模型  
history_2_1 = model_2_1.fit(train_final_df_2_1, y_train_2_1, sample_weight=w_train_2_1, epochs=100, batch_size=64, verbose=1)

# 评估新模型  
test_loss_2_1 = model_2_1.evaluate(test_final_df_2_1,y_test_2_1)  
print(f'Test Loss for Nu Model: {test_loss_2_1}')

# 可视化nu模型的训练损失  
plt.plot(history_2_1.history['loss'], label='Training Loss')  
plt.show() 

# 对训练集进行预测  
y_fitted_2_1 = model_2_1.predict(train_final_df_2_1)
train_2_1["fitted_2_1"] = y_fitted_2_1  
y_true_train_2_1=train_2_1.groupby("age")['ny'].mean()   #测试集真实值
y_fitted_train_2_1= train_2_1.groupby('age')['fitted_2_1'].mean()   #预测值
##绘制按年龄分组的转移次数真实值和预测值

plt.plot(y_true_train_2_1.index, y_true_train_2_1, label='True Values',color='blue')
plt.plot(y_fitted_train_2_1.index, y_fitted_train_2_1, label='fitted Values', color='red')
plt.title('Actual vs Fitted by Age(L->H)')
plt.xlabel('Age(65-105)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

 # 进行预测
predictions_2_1 = model_2_1.predict(test_final_df_2_1)  
predictions_2_1

sample_weight_2_1=w_test_2_1
test_2_1["predictions"]=predictions_2_1
# ['sum_health1_3']/['sum_year']
y_true1_2_1=test_2_1.groupby("age")['ny'].mean()   #测试集真实值
y_pred1_2_1= test_2_1.groupby('age')['predictions'].mean()   #预测值
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

test_2_1['sex_label'] = test_2_1['sex'].map({1: 'male', 2: 'female'}) 
y_pred1_2_1_sex= test_2_1.groupby(['age','sex_label'])['predictions'].mean().reset_index()   #预测值

male_data_2_1 = y_pred1_2_1_sex[y_pred1_2_1_sex['sex_label'] == 'male']  
female_data_2_1 = y_pred1_2_1_sex[y_pred1_2_1_sex['sex_label'] == 'female'] 

# 绘制男性的预测值
plt.plot(male_data_2_1['age'], male_data_2_1['predictions'], color='blue', linestyle='--',label='male')  
# 绘制女性的预测值   
plt.plot(female_data_2_1['age'], female_data_2_1['predictions'], color='red', linestyle='--',label='female')  
# 绘制平均值  
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Mean', color='black')
plt.title('L->H by sex')
plt.xlabel('Age')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()


test_2_1['residenc_label'] = test_2_1['residenc'].map({1: 'rural', 2: 'urban'}) 
y_pred1_2_1_residenc= test_2_1.groupby(['age','residenc_label'])['predictions'].mean().reset_index()   #预测值

rural_data_2_1 = y_pred1_2_1_residenc[y_pred1_2_1_residenc['residenc_label'] == 'rural']  
urban_data_2_1 = y_pred1_2_1_residenc[y_pred1_2_1_residenc['residenc_label'] == 'urban'] 

# 绘制农村的预测值
plt.plot(rural_data_2_1['age'], rural_data_2_1['predictions'], color='blue', linestyle='--',label='rural')  
# 绘制城镇的预测值   
plt.plot(urban_data_2_1['age'], urban_data_2_1['predictions'], color='red', linestyle='--',label='urban')  
# 绘制平均值  
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Mean', color='black')

plt.title('L->H by residency')
plt.xlabel('Age')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_1['marry_label'] = test_2_1['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
y_pred1_2_1_marry= test_2_1.groupby(['age','marry_label'])['predictions'].mean().reset_index()   #预测值

with_spouse_data_2_1 = y_pred1_2_1_marry[y_pred1_2_1_marry['marry_label'] == 'with spouse']  
without_spouse_data_2_1 = y_pred1_2_1_marry[y_pred1_2_1_marry['marry_label'] == 'without spouse'] 


# 绘制有配偶的预测值
plt.plot(with_spouse_data_2_1['age'],with_spouse_data_2_1['predictions'], color='blue', linestyle='--',label='with spouse')  
# 绘制没有配偶的预测值   
plt.plot(without_spouse_data_2_1['age'],without_spouse_data_2_1['predictions'], color='red', linestyle='--',label='without spouse')  
# 绘制平均值  
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Mean', color='black')

plt.title('L->H by marital status')
plt.xlabel('Age')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_1['smoking_label'] = test_2_1['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
y_pred1_2_1_smoking= test_2_1.groupby(['age','smoking_label'])['predictions'].mean().reset_index()   #预测值

smoking_data_2_1 = y_pred1_2_1_smoking[y_pred1_2_1_smoking['smoking_label'] == 'smoking']  
not_smoking_data_2_1 = y_pred1_2_1_smoking[y_pred1_2_1_smoking['smoking_label'] == 'not smoking'] 


# 绘制吸烟的预测值
plt.plot(smoking_data_2_1['age'], smoking_data_2_1['predictions'], color='blue', linestyle='--',label='smoking')  
# 绘制不吸烟的预测值   
plt.plot(not_smoking_data_2_1['age'], not_smoking_data_2_1['predictions'], color='red', linestyle='--',label='not smoking')  
# 绘制平均值  
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Mean', color='black')

plt.title('L->H by smoking')
plt.xlabel('Age')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

test_2_1['drinking_label'] = test_2_1['drink'].map({1: 'drinking', 2: 'not drinking'}) 
y_pred1_2_1_drinking= test_2_1.groupby(['age','drinking_label'])['predictions'].mean().reset_index()   #预测值

drinking_data_2_1 = y_pred1_2_1_drinking[y_pred1_2_1_drinking['drinking_label'] == 'drinking']  
not_drinking_data_2_1 = y_pred1_2_1_drinking[y_pred1_2_1_drinking['drinking_label'] == 'not drinking']


# 绘制吸烟的预测值
plt.plot(drinking_data_2_1['age'], drinking_data_2_1['predictions'], color='blue', linestyle='--',label='drinking')  
# 绘制不吸烟的预测值   
plt.plot(not_drinking_data_2_1['age'], not_drinking_data_2_1['predictions'], color='red', linestyle='--',label='not drinking')  
# 绘制平均值  
plt.plot(y_pred1_2_1.index, y_pred1_2_1, label='Mean', color='black')

plt.title('L->H by drinking')
plt.xlabel('Age')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.show()

predictions_list_2_1 = [train_2_1['fitted_2_1'], test_2_1['predictions']]
new_data_2_1['predictions'] = pd.concat(predictions_list_2_1)

y_true2_2_1=new_data_2_1.groupby("t")['ny'].mean()   #测试集真实值
y_pred2_2_1= new_data_2_1.groupby('t')['predictions'].mean()   #预测值
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

#性别
new_data_2_1['sex_label'] = new_data_2_1['sex'].map({1: 'male', 2: 'female'}) 
y_pred2_2_1_sex= new_data_2_1.groupby(['t','sex_label'])['predictions'].mean().reset_index()   #预测值
male_data1_2_1 = y_pred2_2_1_sex[y_pred2_2_1_sex['sex_label'] == 'male']  
female_data1_2_1 = y_pred2_2_1_sex[y_pred2_2_1_sex['sex_label'] == 'female'] 


# 绘制男性的预测值
plt.plot(male_data1_2_1['t'], male_data1_2_1['predictions'], color='blue', linestyle='--',label='male')  
# 绘制女性的预测值   
plt.plot(female_data1_2_1['t'], female_data1_2_1['predictions'], color='red', linestyle='--',label='female')  
# 绘制平均值  
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Mean', color='black')
plt.title('L->H by sex')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()


#居住地
new_data_2_1['residenc_label'] = new_data_2_1['residenc'].map({1: 'rural', 2: 'urban'}) 
y_pred2_2_1_residenc= new_data_2_1.groupby(['t','residenc_label'])['predictions'].mean().reset_index()   #预测值
rural_data1_2_1 = y_pred2_2_1_residenc[y_pred2_2_1_residenc['residenc_label'] == 'rural']  
urban_data1_2_1 = y_pred2_2_1_residenc[y_pred2_2_1_residenc['residenc_label'] == 'urban'] 


# 绘制农村的预测值
plt.plot(rural_data1_2_1['t'], rural_data1_2_1['predictions'], color='blue', linestyle='--',label='rural')  
# 绘制城镇的预测值   
plt.plot(urban_data1_2_1['t'], urban_data1_2_1['predictions'], color='red', linestyle='--',label='urban')  
# 绘制平均值  
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Mean', color='black')
plt.title('L->H by residency')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

#婚姻情况
new_data_2_1['marry_label'] = new_data_2_1['marry'].map({1: 'with spouse', 2: 'without spouse'}) 
y_pred2_2_1_marry= new_data_2_1.groupby(['t','marry_label'])['predictions'].mean().reset_index()   #预测值
with_spouse_data1_2_1 = y_pred2_2_1_marry[y_pred2_2_1_marry['marry_label'] == 'with spouse']  
without_spouse_data1_2_1 = y_pred2_2_1_marry[y_pred2_2_1_marry['marry_label'] == 'without spouse'] 


# 绘制有配偶的预测值
plt.plot(with_spouse_data1_2_1['t'],with_spouse_data1_2_1['predictions'], color='blue', linestyle='--',label='with spouse')  
# 绘制没有配偶的预测值   
plt.plot(without_spouse_data1_2_1['t'],without_spouse_data1_2_1['predictions'], color='red', linestyle='--',label='without spouse')  
# 绘制平均值  
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Mean', color='black')
plt.title('L->H by marital status')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

#吸烟情况
new_data_2_1['smoking_label'] = new_data_2_1['smoke'].map({1: 'smoking', 2: 'not smoking'}) 
y_pred2_2_1_smoking= new_data_2_1.groupby(['t','smoking_label'])['predictions'].mean().reset_index()   #预测值
smoking_data1_2_1 = y_pred2_2_1_smoking[y_pred2_2_1_smoking['smoking_label'] == 'smoking']  
not_smoking_data1_2_1 = y_pred2_2_1_smoking[y_pred2_2_1_smoking['smoking_label'] == 'not smoking'] 


# 绘制吸烟的预测值
plt.plot(smoking_data1_2_1['t'], smoking_data1_2_1['predictions'], color='blue', linestyle='--',label='smoking')  
# 绘制不吸烟的预测值   
plt.plot(not_smoking_data1_2_1['t'], not_smoking_data1_2_1['predictions'], color='red', linestyle='--',label='not smoking')  
# 绘制平均值  
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Mean', color='black')
plt.title('L->H by smoking')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()

#饮酒情况
new_data_2_1['drinking_label'] = new_data_2_1['drink'].map({1: 'drinking', 2: 'not drinking'}) 
y_pred2_2_1_drinking= new_data_2_1.groupby(['t','drinking_label'])['predictions'].mean().reset_index()   #预测值
drinking_data1_2_1 = y_pred2_2_1_drinking[y_pred2_2_1_drinking['drinking_label'] == 'drinking']  
not_drinking_data1_2_1 = y_pred2_2_1_drinking[y_pred2_2_1_drinking['drinking_label'] == 'not drinking']

# 绘制吸烟的预测值
plt.plot(drinking_data1_2_1['t'], drinking_data1_2_1['predictions'], color='blue', linestyle='--',label='drinking')  
# 绘制不吸烟的预测值   
plt.plot(not_drinking_data1_2_1['t'], not_drinking_data1_2_1['predictions'], color='red', linestyle='--',label='not drinking')  
# 绘制平均值  
plt.plot(y_pred2_2_1.index, y_pred2_2_1, label='Mean', color='black')
plt.title('L->H by drinking')
plt.xlabel('Year (1998-2018)')
plt.ylabel('L->H intensity')
plt.legend()
plt.grid(True)
plt.xticks([2000, 2002, 2005, 2008, 2010, 2012, 2015])
plt.show()


sample_weight_2_1=w_test_2_1
y_true_2_1=y_test_2_1.values
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
predictions_2_1['net_predictions'] = test_2_1_reset_index["predictions"]

fitted_2_1['net_fitted']=train_2_1["fitted_2_1"]
fitted_predictions_2_1['net_fitted_predictions']=new_data_2_1['predictions']


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