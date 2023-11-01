import torch
# 加载数据集
from torch.utils.data import DataLoader
# 数据的原始处理
from torchvision import transforms
# pytorch直接准备好的数据集
from torchvision import datasets
# 激活函数
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
# 我们拿到的图片是pillow，我们要将其转变成模型里可以训练的tensor，也就是张量的格式
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试集
test_dataset = datasets.MNIST(root='../data', train = False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 做模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义我们第一个需要使用的卷积层，一位内图片输入的通道为1，第一个参数直接设置成1
        # 输出的通道为10，kernel_size为卷积核的大小
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 定义了一个池化层
        self.pooling = torch.nn.MaxPool2d(2)
        # 最后是做分类用的线性层
        self.fc1 = torch.nn.Linear(320, 10)

    # 计算过程
    def forward(self, x):
        batch_size = x.size(0)   # 自动获取batch的大小
        # 输入x经过一个卷积层，之后经历一个池化层，最后用relu做激活
        x = F.relu(self.pooling(self.conv1(x)))
        # 在经历上面的过程
        x = F.relu(self.pooling((self.conv2(x))))
        # 为了可以让全连接层使用，我们需要将一个二维的图片张量变成一维的
        x = x.view(batch_size, -1)
        # 经过线性层，确定他是一个0~9的概率
        x = self.fc1(x)
        return x

model = Net()    # 实例化一个模型
# 使用GPU进行计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义的损失函数，来计算我们模型输出的值和标准值的差距
criterion = torch.nn.CrossEntropyLoss()
# 定义的优化器，训练模型怎么做的全靠这个，它可以反向的更改相应层的权重
# lr过大会出现过拟合现象
optimizer = optim.SGD(model.parameters(), lr=0.15, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):    # 每次取出一个样本
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 优化器清零
        optimizer.zero_grad()
        # 正向计算
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向梯度计算
        loss.backward()
        # 更新权重
        optimizer.step()
        # 损失求和
        running_loss += loss.item()
        # 每300次输出一次数据
        if batch_index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 2000))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要算梯度
        for data in test_loader:
            imputs, labels = data
            imputs, labels = imputs.to(device), labels.to(device)
            outputs = model(imputs)
            # 取概率最大的数作为输出
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # 计算正确率
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()

