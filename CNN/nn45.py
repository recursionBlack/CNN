import torch

# 波士顿房价预测模型搭建

# data
import numpy as np
import re
ff = open("housing.data").readlines()
data = []
for item in ff:
    # 数据间的多个空格合并为一个空格
    out = re.sub(r"\s{2, }", " ", item).strip()
    print(out)
    data.append(out.split(" "))
data = np.array(data).astype(np.float)
print(data.shape)

Y = data[:, -1]
X = data[:, 0:-1]

X_train = [0:496, ...]
Y_train = [0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# net
# 只定义一个简单的回归网络
# 自定义的网络结构必须继承自torch.nn.module
class Net(torch.nn.module):
    # 自定义网络结构必须要重写的两个函数：__init__和forward
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 只有一个线性层的简单初始网络
        self.predict = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        out = self.predict(x)
        return out
net = Net(13, 1)
# loss
# 采用均方损失作为loss
loss_func = torch.nn.MSELoss()

# optimizer
# 学习率0.01
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# training
# 定义训练的过程
# 训练次数
for i in range(1000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    # 样本标签
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_train = loss_func(pred, y_data) * 1000
    # print(pred.shape)
    # print(y_data.shape)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    print("ite:{}, loss_train:{}".format(i ,loss_train))
    print(pred[0:10])
    print(y_data[0:10])

    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    # 样本标签
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 1000
    print("ite:{}, loss_test:{}".format(i ,loss_test))

# 保存网络
torch.save(net, "model/model.pkl")
# 方式2，只保存参数，再次加载时需要先定义net对象
# torch.save(net.state_dict(), "params.pkl")