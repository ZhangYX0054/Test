import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Design Model  重点目标在于构造计算图
"""
所有模型都要继承自Model
最少实现两个成员方法
    构造函数 初始化：__init__()
    前馈：forward()
Model自动实现backward
可以在Functions中构建自己的计算块
"""
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)     # 构造了一个包含 w和 b的对象

    def forward(self, x):
        y_hat = self.linear(x)      # linear成为了可调用的对象 直接计算forward
        return y_hat

model = LinearModel()     # 创建的实体类

criterion = torch.nn.MSELoss(reduction='sum')  #需要的参数是y_pred和y  该函数默认用于计算两个输入对应元素差值平方和的均值。在深度学习中，可以使用该函数用来计算两个特征图的相似性。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)   # 随机梯度下降的优化器

# 训练
for epoch in range(1271):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)



