import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 利用torch实现反向传播Backward
# 1.计算损失 2.backward 3.梯度下降的持续更新
x_data = [1.0]
y_data = [2.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2

# 打印学习之前的值，.item()表示输出张量的值
print("Predict (before training)", 4, forward(4).item())
learning_rate = 0.01
epoch_list = []
loss_list = []

# 训练过程
for epoch in range(500):
    for x, y in zip(x_data, y_data):
        loss_value = loss(x, y)      # forward前馈
        loss_list.append(loss_value.item())    # 将梯度存到w之中,随后释放计算图 item()可以直接将梯度变成标量
        loss_value.backward()       # 成员函数backward()向后传播 自动求出所有需要的梯度
        w.data -= learning_rate * w.grad.data    # w的grad也是张量，计算应该取data 不去建立计算图
        w.grad.data.zero_()     #释放data
        epoch_list.append(epoch)
    print("progress:", epoch, loss_value.item())
print("Predict (after training)", 4, forward(4).item())

# 绘图
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



