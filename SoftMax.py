import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
# 一般用Compose把多个步骤整合到一起
# transforms.ToTensor()能够把灰度范围从0-255变换到0-1之间      transform.Normalize()则把0-1变换到(-1,1)
transforms  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root = '../dataset/mnist', train = True, download = True, transform = transforms)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transforms)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x         # 最后一层不做激活，不进行非线性变换

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            """
            选用下划线代表不需要用到的变量。比如在图像分类任务中，值所对应的index就对应着相应的类别class
        当我们只关心网络预测的类别是什么，而不关心该类别的预测概率是多少时，就选择使用下划线_。
            dim=0表示计算每列的最大值，dim=1表示每行的最大值
            """
            _, predicted = torch.max(outputs.data, dim=1) # 是计算模型中每个类别的最大值并返回其索引值，即该类别的标签值。

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  #张量之间的比较运算
    print('accuracy no test set: %d %%' % (100*correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()


