import matplotlib.pyplot as plt

"""
随机梯度下降法在神经网络中被证明是有效的。效率较低(时间复杂度较高)，学习性能较好。

随机梯度下降法和梯度下降法的主要区别在于：

1、损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练数据的损失。对应于源代码则是少了两个for循环。

2、梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度。

3、本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。梯度下降法中梯度总共更新100(epoch)次
"""

x_all = [1.0, 2.0, 3.0]
y_all = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def lost(x, y):
    y_pre = forward(x)
    return (y_pre - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))

for epoch in range(116):
    for x, y in zip(x_all, y_all):
        grad = gradient(x, y)
        w -= 0.01 * grad
        print("\t gard: ", x, y, grad)
        loss = lost(x, y)
    print("process:", epoch, "w=", w, "loss=", loss)
    epoch_list.append(epoch)
    loss_list.append(loss)

print('predict (after training)', 4.0, forward(4.0))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')    # 执行次数
plt.show()

