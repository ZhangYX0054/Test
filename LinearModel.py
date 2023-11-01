import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0,6.0]

# 模型
def forward(x):
    return x * w

# 损失函数
def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) * (y_hat - y)

# 权重保存
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):       # 从0到4.1取值，间隔为0.1
    print("w = ", w)
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print('\t', x_val, y_val, y_pred_val,loss_val)
    print('MSE=', loss_sum)
    w_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')    # 纵坐标
plt.xlabel('w')    # 横坐标
plt.show()


