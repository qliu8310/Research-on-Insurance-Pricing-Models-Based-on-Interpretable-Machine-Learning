import torch
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

# 导入数据
dat = pd.read_csv('processed_dat.csv')
feature_names=list(('Area','VehPower','VehAge','DrivAge','BonusMalus','VehBrand','VehGas','Density','Region'))
X = dat[feature_names]
w = dat['Exposure']
y = dat['ClaimNb']
kf = KFold(n_splits=10, shuffle=True,random_state=111)
# 定义模型
class NetworkMU(nn.Module):
    def __init__(self):
        super(NetworkMU, self).__init__()
        # 添加网络结构的定义
        self.hidden1 = nn.Linear(9, 100)
        self.dropout1 = nn.Dropout(0.3)  # 第一个 Dropout 层，合适的Dropout率可以在0.2到0.5之间
        self.hidden2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.3)  # 第二个 Dropout 层
        self.hidden3 = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.1)  # 第三个 Dropout 层
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        # 网络的前向传播
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = F.relu(self.hidden3(x))
        x = self.dropout3(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = self.out(x)
        return x

class NetworkP(nn.Module):
    def __init__(self):
        super(NetworkP, self).__init__()
        # 添加网络结构的定义
        self.hidden1 = nn.Linear(9, 100)
        self.dropout1 = nn.Dropout(0.3)  # 第一个 Dropout 层，合适的Dropout率可以在0.2到0.5之间
        self.hidden2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.3)  # 第二个 Dropout 层
        self.hidden3 = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.1)  # 第三个 Dropout 层
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        # 网络的前向传播
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = F.relu(self.hidden3(x))
        x = self.dropout3(x)  # 在需要应用 Dropout 的地方调用 Dropout 层
        x = self.out(x)
        return x

# 初始化网络并加载网络
network_mu = NetworkMU()
network_p = NetworkP()
# 加载网络参数
network_mu.load_state_dict(torch.load('network_mu(泊松分布)'))
network_p.load_state_dict(torch.load('network_p(泊松分布)'))

# Saliency 敏感性分析
from captum.attr import Saliency

model = network_mu
model.eval()
X_tensor = torch.Tensor(X.values)
input_data = X_tensor.requires_grad_(True)

# Create a Saliency object for the model
saliency = Saliency(model)

# Perform Saliency Sensitivity Analysis
saliency_scores = saliency.attribute(input_data)

# Visualize the Saliency scores
saliency_scores = saliency_scores.reshape(-1, X_tensor.shape[1])

# Take the absolute mean across the examples to get a single sensitivity score per feature
sensitivity_scores = torch.abs(saliency_scores).mean(dim=0).detach().numpy()

# Visualize the Saliency scores
plt.bar(feature_names, sensitivity_scores)
plt.xlabel('Feature Name')
plt.ylabel('Sensitivity Score')
plt.title('Saliency Sensitivity Analysis')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

model =network_p
model.eval()
X_tensor = torch.Tensor(X.values)
input_data = X_tensor.requires_grad_(True)

# Create a Saliency object for the model
saliency = Saliency(model)

# Perform Saliency Sensitivity Analysis
saliency_scores = saliency.attribute(input_data)

# Visualize the Saliency scores
saliency_scores = saliency_scores.reshape(-1, X_tensor.shape[1])

# Take the absolute mean across the examples to get a single sensitivity score per feature
sensitivity_scores = torch.abs(saliency_scores).mean(dim=0).detach().numpy()

# Visualize the Saliency scores
plt.bar(feature_names, sensitivity_scores)
plt.xlabel('Feature Name')
plt.ylabel('Sensitivity Score')
plt.title('Saliency Sensitivity Analysis')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

# shap 解释
import shap
from torch.autograd import Variable

X_numpy = X.values
# 将 NumPy 数组包装成 PyTorch Variable
X_variable = Variable(torch.Tensor(X_numpy), requires_grad=True)
# 使用 DeepExplainer 进行 SHAP 值计算
TOLERANCE = 0.1
explainer = shap.DeepExplainer(network_mu, X_variable)
shap_values = explainer.shap_values(X_variable, ranked_outputs=0, check_additivity=False)
# 显示整体特征重要性
shap.summary_plot(shap_values, X_numpy, feature_names=["Feature{}".format(i) for i in range(X_numpy.shape[1])])
# 使用 DeepExplainer 进行 SHAP 值计算
TOLERANCE = 0.1
explainer1 = shap.DeepExplainer(network_p, X_variable)
shap_values1 = explainer1.shap_values(X_variable, ranked_outputs=0, check_additivity=False)
# 显示整体特征重要性
shap.summary_plot(shap_values1, X_numpy, feature_names=["Feature{}".format(i) for i in range(X_numpy.shape[1])])

# 特征重要性
# 将模型设置为评估模式
network_mu.eval()
# 将DataFrame转换为 PyTorch 的 Tensor
X_tensor = torch.Tensor(X.values)

# Wrap the input data in a Variable (no longer necessary in recent PyTorch versions, but can still be used for backward compatibility)
input_data = torch.autograd.Variable(X_tensor, requires_grad=True)

# Forward pass
output = network_mu(input_data)

# Create a tensor for the gradient with the same shape as the output
grad_output = torch.ones_like(output)

# Backward pass with specified grad_output
output.backward(gradient=grad_output)

# Calculate the gradient for each feature
feature_importance = input_data.grad.data.abs().numpy()

# Print feature importance
print("Feature Importance:", feature_importance)
import matplotlib.pyplot as plt
import numpy as np
# Calculate the mean importance across all samples for each feature
average_feature_importance = np.mean(feature_importance, axis=0)

# Get the indices that would sort the feature importance in descending order
sorted_indices = np.argsort(average_feature_importance)[::-1]

# Sort the feature names and importance accordingly
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_average_feature_importance = average_feature_importance[sorted_indices]

# Plot the sorted average feature importance with customized x-axis labels and rotated ticks
plt.bar(range(len(sorted_average_feature_importance)), sorted_average_feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Average Importance')
plt.title('Average Feature Importance')

# Set custom x-axis labels and rotate them by 45 degrees
plt.xticks(range(len(sorted_average_feature_importance)), sorted_feature_names, rotation=45, ha="right")

plt.show()

# network_p()进行特征重要性分析
# 将模型设置为评估模式
network_p.eval()
# 将DataFrame转换为 PyTorch 的 Tensor
X_tensor = torch.Tensor(X.values)

# Wrap the input data in a Variable (no longer necessary in recent PyTorch versions, but can still be used for backward compatibility)
input_data = torch.autograd.Variable(X_tensor, requires_grad=True)

# Forward pass
output = network_p(input_data)

# Create a tensor for the gradient with the same shape as the output
grad_output = torch.ones_like(output)

# Backward pass with specified grad_output
output.backward(gradient=grad_output)

# Calculate the gradient for each feature
feature_importance_p = input_data.grad.data.abs().numpy()

# Print feature importance
print("Feature Importance_p:", feature_importance_p)

# 可视化
# Calculate the mean importance across all samples for each feature
average_feature_importance = np.mean(feature_importance_p, axis=0)

# Get the indices that would sort the feature importance in descending order
sorted_indices = np.argsort(average_feature_importance)[::-1]

# Sort the feature names and importance accordingly
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_average_feature_importance = average_feature_importance[sorted_indices]

# Plot the sorted average feature importance with customized x-axis labels and rotated ticks
plt.bar(range(len(sorted_average_feature_importance)), sorted_average_feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Average Importance')
plt.title('Average Feature Importance')

# Set custom x-axis labels and rotate them by 45 degrees
plt.xticks(range(len(sorted_average_feature_importance)), sorted_feature_names, rotation=45, ha="right")

plt.show()

# DeepLift解释
from captum.attr import DeepLift
# Create a DeepLift object for the model
model = network_mu
model.eval()
X_tensor = torch.Tensor(X.values)
input_data = X_tensor.requires_grad_(True)
deep_lift = DeepLift(model)

# Perform DeepLift analysis
attributions = deep_lift.attribute(input_data)
print(attributions.shape)
mean_attributions = attributions.mean(dim=0)
# 将 PyTorch 张量转换为 NumPy 数组
mean_attributions_np = mean_attributions.detach().numpy()
plt.bar(feature_names, mean_attributions_np)
plt.xlabel('Feature Name')
plt.ylabel('Attribution Score')
plt.title('DeepLift Analysis')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

model = network_p
model.eval()
X_tensor = torch.Tensor(X.values)
input_data = X_tensor.requires_grad_(True)
deep_lift = DeepLift(model)

# Perform DeepLift analysis
attributions = deep_lift.attribute(input_data)
print(attributions.shape)
mean_attributions = attributions.mean(dim=0)
# 将 PyTorch 张量转换为 NumPy 数组
mean_attributions_np = mean_attributions.detach().numpy()
plt.bar(feature_names, mean_attributions_np)
plt.xlabel('Feature Name')
plt.ylabel('Attribution Score')
plt.title('DeepLift Analysis')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()