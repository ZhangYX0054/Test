import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(loss.item())

x_test = torch.Tensor([1.0])
y_test = model(x_test)
print('y_pred = ', y_test.data)

x = np.linspace(0, 10, 500)       # x轴坐标值 从0-10，均匀的分为500个元素
x_t = torch.Tensor(x).view(500, 1)     # 返回一个新的张量，其数据与自张量相同，但形状不同
y_t = model(x_t)
y = y_t.data.numpy()

"""
plt.rcParams['font.family'] = 'SimHei'       # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号
"""
plt.plot(x, y)   # plot显示的是整个图像
plt.plot([0, 10], [0.5, 0.5], c='lime') # c代表的是颜色  [0.5, 0.5]从0.5-0.5画线
plt.xlabel('hours')
plt.ylabel('Probability Of Pass')
plt.grid()       # 网格线
plt.show()

"""
plt.xticks()、plt.yticks()分别表示横纵轴的刻度
labels参数调整刻度为自己想要的字符或文字
plt.xlim(),plt.ylim()分别表示横纵轴的刻度范围
给图设置标题：plt.title()，plt.xlabel(),plt.ylabel()
显示图例：plt.legend(  )
保存图片：plt.savefig(fname)
"""




