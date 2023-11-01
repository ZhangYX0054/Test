import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 参数 delimiter 可以指定各种分隔符、针对特定列的转换器函数、需要跳过的行数等
# usecol是指只使用0,2两列
xy = np.loadtxt('diabetes.csv', delimiter=' ', dtype=np.float32)
x_data = torch.from_numpy(xy[ : , :-1])
y_data = torch.from_numpy(xy[ : , [-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        """
        self.fc1 = torch.nn.Linear(9, 6)
        self.fc2 = torch.nn.Linear(6, 4)
        self.fc3 = torch.nn.Linear(4, 1)
        """
        # 效果更好
        self.fc1 = torch.nn.Linear(9, 32)
        self.fc2 = torch.nn.Linear(32, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = Model()


criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.plot([0, 100], [0.2, 0.2], c='lime')
plt.xlabel('epoch_list')
plt.ylabel('loss_list')
plt.grid()
plt.show()


