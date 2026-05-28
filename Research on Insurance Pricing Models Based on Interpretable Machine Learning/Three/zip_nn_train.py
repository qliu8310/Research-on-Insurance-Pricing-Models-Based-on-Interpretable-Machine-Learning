# 模型训练部分代码
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

dat = pd.read_csv('processed_dat.csv')
feature_names=list(('Area','VehPower','VehAge','DrivAge','BonusMalus','VehBrand','VehGas','Density','Region'))
X = dat[feature_names]
w = dat['Exposure']
y = dat['ClaimNb']
kf = KFold(n_splits=10, shuffle=True,random_state=111)

# 创建网络
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

# 定义损失函数
class ZIP_logloss(nn.Module):
    def __init__(self):
        super(ZIP_logloss, self).__init__()

    def forward(self, y_true, mu, pi):
        # 计算非零部分的损失
        non_zero_part = torch.log((1 - pi) * (torch.distributions.Poisson(mu).log_prob(y_true).exp()))

        # 计算零部分的损失
        zero_part = torch.log(pi + (1-pi) * torch.exp(-mu))

        # 组合非零和零部分的损失
        loss = -torch.where(y_true > 0, non_zero_part, zero_part)

        return loss.sum()

# 定义偏差函数
class CustomLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 防止除以零或对数负数

    def forward(self, y_pred, y_true):
        y_pred_safe = torch.clamp(y_pred, min=self.eps)  # 确保预测值为正
        loss = torch.where(
            y_true == 0,
            2 * y_pred_safe,  # y_i=0时的损失
            2 * (y_true * torch.log(y_true / y_pred_safe) - y_true + y_pred_safe)  # y_i>0时的损失
        )
        return loss.mean()  # 返回平均值作为最终损失

deviance = CustomLoss()

loss_fn = ZIP_logloss()
optimizer_mu = torch.optim.Adam(network_mu.parameters(), lr=0.0000001)
optimizer_p = torch.optim.Adam(network_p.parameters(), lr=0.0000001)

loss_list = list()
deviance_list = list()
r_list = list()
# 模型训练
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    # 将模型设置为训练模式
    network_mu.train()
    network_p.train()
    # Split the data into training and validation sets for this fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    w_train, w_val = w.iloc[train_index], w.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Convert the data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    w_train_tensor = torch.tensor(w_train.values, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    w_val_tensor = torch.tensor(w_val.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, w_train_tensor, y_train_tensor)
    # Create DataLoader for training and validation
    batch_size = 128  # Set the desired batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Training loop for this fold
    count = 0
    for epoch in range(8):
        for batch_X, batch_w, batch_y in train_dataloader:

            count += 1
            mu_train = network_mu(batch_X)
            ZIP_mu = (torch.exp(mu_train)) * batch_w
            p_train = network_p(batch_X)
            ZIP_p = torch.sigmoid(p_train)

            # Calculate the loss
            loss = loss_fn(batch_y, ZIP_mu, ZIP_p)
            # 交替更新 ZIP_mu 和 ZIP_p
            if count % 2 == 0:
                # 固定 ZIP_p，训练 ZIP_mu
                optimizer_mu.zero_grad()
                loss.backward()
                optimizer_mu.step()
            else:
                # 固定 ZIP_mu，训练 ZIP_p
                optimizer_p.zero_grad()
                loss.backward()
                optimizer_p.step()

    # Evaluation on validation set for this fold
    with torch.no_grad():
        network_mu.eval()
        network_p.eval()
        mu_val = network_mu(X_val_tensor)
        ZIP_mu_val = w_val_tensor * torch.exp(mu_val)
        p_val = network_p(X_val_tensor)
        ZIP_p_val = torch.sigmoid(p_val)

        mu_p_pred = ZIP_mu_val * (1 - ZIP_p_val)

        # Calculate the validation loss
        val_loss = loss_fn(y_val_tensor, ZIP_mu_val, ZIP_p_val)

        deviance_loss = deviance(mu_p_pred, y_val_tensor)
        deviance_list.append(deviance_loss)
        print(deviance_loss)

        y_val_mean = sum(y_val_tensor) / sum(w_val_tensor)

        dev_loss = deviance(w_val_tensor * y_val_mean, y_val_tensor)
        r = 1 - (deviance_loss / dev_loss)
        #         r = 1 - (dev_loss / deviance_loss)
        r_list.append(r)
        print(r)

        loss_list.append(val_loss)
        fold += 1
    print(f"fold{fold}: {val_loss}")

# Calculate the average
loss_average = sum(loss_list) / len(loss_list)
print(f"loss_average:{loss_average}")
deviance_average = sum(deviance_list) / len(deviance_list)
print(f"loss_average:{deviance_average }")
r_average = sum(r_list) / len(r_list)
print(f"loss_average:{r_average }")

