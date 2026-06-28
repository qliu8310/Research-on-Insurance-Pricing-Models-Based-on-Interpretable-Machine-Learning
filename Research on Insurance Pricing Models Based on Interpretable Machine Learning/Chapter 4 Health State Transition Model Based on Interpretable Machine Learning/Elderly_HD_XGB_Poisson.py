import tensorflow as tf
print(tf.__version__)
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Resizing, RandomFlip
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, optimizers, metrics
#from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential  
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.losses import poisson  
from statsmodels.genmod.families import family, Poisson
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import keras_tuner as kt   #import kerastuner(kerastuner 已被集成到 KerasTuner（独立库）或 TensorFlow 2.10+ 中，原 kerastuner 包不再维护)
from keras import backend as K  
from keras import metrics 
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
# input_shape = X_train_processed.shape[1]
input_shape = X_train.shape[1]


# model = xgb.XGBRegressor(objective='count:poisson')
model = xgb.XGBRegressor(objective='count:poisson',n_estimators=150,learning_rate=0.1,subsample=0.6,max_depth=3,random_state=420,colsample_bytree=0.8,num_leaves=100)
# model = xgb.XGBRegressor(objective='count:poisson',n_estimators=150,learning_rate=0.1,random_state=420,subsample=0.5,colsample_bytree=0.8)

model.fit(X_train_processed,y_train,sample_weight=w_train)


# 拟合值
y_fitted = model.predict( X_train_processed)
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

y_pred = model.predict(X_test_processed)
y_pred

mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse}")

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

# df['predictions'] = pd.concat(train['fitted'],test['predictions'])  
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



predictions_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx')
fitted_1_3= pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx')
fitted_predictions_1_3 = pd.read_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx') 

test_reset_index = test.reset_index(drop=True)   
predictions_1_3['xgboost_predictions'] = test_reset_index["predictions"]

fitted_1_3['xgboost_fitted']=train["fitted"]
fitted_predictions_1_3['xgboost_fitted_predictions']=df['predictions']


fitted_predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_predictions_1_3.xlsx', index=False)
predictions_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\predictions_1_3.xlsx', index=False)
fitted_1_3.to_excel(r'D:\PG\Rpaper\基于可解释机器学习的中国老年人健康状态转移模型及应用\数据 - 副本\predictions\fitted_1_3.xlsx', index=False)
