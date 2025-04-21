import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

train_data = pd.read_csv('D:/Pythontest/pytorch/data/california-house-prices/train.csv')
test_data = pd.read_csv('D:/Pythontest/pytorch/data/california-house-prices/test.csv')

train_data_tmp = train_data.iloc[:, 3:39]
test_data_tmp = test_data.iloc[:, 3:39]
train_data_numeric = train_data_tmp.select_dtypes(exclude=['object'])
test_data_numeric = test_data_tmp.select_dtypes(exclude=['object'])
all_features = pd.concat([train_data_numeric, test_data_numeric])

# all_features = pd.concat((train_data.iloc[:, [10, 11, 12, 13, 14, 15, 16, 33, 34, 36]], test_data.iloc[:, [10, 11, 12, 13, 33, 34, 36]]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)



# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

all_features = all_features * 1.0

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data["Sold Price"].values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch衰减为原来的0.1倍

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        # 更新学习率
        # scheduler.step()
        # 可选：打印当前学习率
        # print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")  # 输出当前学习率

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    all_train_ls, all_valid_ls = [], []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        all_train_ls.append(train_ls)  # 收集每个fold的train误差
        all_valid_ls.append(valid_ls)  # 收集每个fold的valid误差

        if i == 0:
            plt.plot(range(1, num_epochs + 1), train_ls, label='Train RMSE')
            plt.plot(range(1, num_epochs + 1), valid_ls, label='Validation RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('Log RMSE')
            plt.title('Train and Validation Log RMSE (Fold 1)')
            plt.legend()
            plt.yscale('log')  # 使用对数坐标轴
            plt.xlim([1, num_epochs])
            plt.show()  # 显示图形
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k, all_train_ls, all_valid_ls

k, num_epochs, lr, weight_decay, batch_size = 6, 200, 35, 0, 32
train_l, valid_l, all_train_ls, all_valid_ls= k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)

print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')