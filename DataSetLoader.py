import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
Dataset是一个抽象函数，不可以直接实例化，所以我们要创建一个自己的类，继承Dataset
继承之后必须要实现三个函数
    __init__(self,filepath)：初始化函数，之后我们可以提供数据集路径进行数据的加载
    __getitem__(self,index)：帮助我们通过索引找到某个样本
    __len__(self)：返回数据集的大小
"""
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=' ', dtype=np.float32)
        # shape本身是一个二元组（x，y）对应数据集的行数和列数，这里[0]我们取行数,即样本数
        self.length = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length
# 定义好DiabetesDataset后我们就可以实例化他了
dataset = DiabetesDataset('diabetes.csv')
# 我们用DataLoader为数据进行分组，batch_size是一个组中有多少个样本，shuffle表示要不要对样本进行随机排列
# 一般来说，训练集我们随机排列，测试集不需要。num_workers表示我们可以用多少进程并行的运算
train_loaders = DataLoader(dataset=dataset, batch_size=32, shuffle=True,num_workers=0)   # num_workers线程

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(9, 16)
        self.fc2 = torch.nn.Linear(16, 6)
        self.fc3 = torch.nn.Linear(6, 4)
        self.fc4 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
# model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':       # if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(100):
        for i, (x, y) in enumerate(train_loaders):   # 取出一个bath
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'epoch: {epoch}, loss: {loss.item()}')










