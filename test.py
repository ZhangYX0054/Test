# import time
# from datetime import datetime
#
#
# end_time = time.time() + 60  # 一分钟后的时间
#
# while time.time() < end_time:
#     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#     time.sleep(1)
#


# x = input('输入两个数：')
# a, b = map(int, x.split())
# if a > b:
#     a, b = b, a
# print(a, b)


# def a(y):
#     return y*2
#
# list = [1, 3, 5, 7]
# list_a = map(a, list)
# for num in list_a:
#     print(num)


# 判断今天是今年的第几天
# import time
#
# data  =time.localtime()
# year, month, day = data[:3]
# day_month = [31,28,31,30,31,30,31,31,30,31,30,31]
# if year % 400 ==0 or (year % 4 == 0 and year % 100 != 0):
#     day_month[1] = 29
# if month == 1:
#     print(day)
# else:
#     print(sum(day_month[:month-1]) + day)


# import datetime # 直接显示今天是今年的
#
# today = datetime.date.today()
# print(today)

# import calendar
# print(calendar.calendar(2023))
# print(calendar.month(2023,9))
# print(calendar.isleap(2023))
# print(calendar.weekday(2023, 8, 30))


# import time
# digits = (1,2,3,4)
# start = time.time()
# for i in range(1000000):
#     result = []
#     for i in digits:
#         for j in digits:
#             for k in digits:
#                 result.append(i*100 + j*10 + k)
# print(time.time() - start)
#
# start = time.time()
# digits = (1,2,3,4)
# for i in range(1000000):
#     result = []
#     for i in digits:
#         i *= 100
#         for j in digits:
#             j *= 10
#             for k in digits:
#                 result.append(i + j + k)
# print(time.time() - start)

# class linear:
#     def __init__(self, a, b):
#         self.a, self.b = a, b
#     def __call__(self, x):
#         return self.a * x + self.b
#
# taxes = linear(0.3, 2)
#
# print(taxes)


# f = lambda x, y, z : x + y + z
# print(f(1,2,3))
#
# g = lambda x, y = 2, z = 3 : x + y + z
# print(g(1))
# print(g(2, z=5, y=6))
# L = [(lambda x : x**2), (lambda x : x**3), (lambda x : x**4)]
# print(L[0](2), L[1](3), L[2](2))
#
# D = { 'f1':(lambda:2+3 ), 'f2':(lambda :2*3), 'f3':(lambda :2**3) }
# print( D['f1'](), D['f2'](), D['f3']() )

# def demo(str):
#     result = [0, 0]
#     for ch in str:
#         if 'a' <= ch <= 'z':
#             result[1] += 1
#         elif 'A' <= ch <= 'Z':
#             result[0] += 1
#     return result
# print(demo('fhwjASOIFHfjejkf'))

# def demo(lst, k):
#     x = lst[:k]
#     x.reverse()
#     y = lst[k:]
#     y.reverse()
#     r = x+y
#     return list(reversed(r))
# lst = list(range(1, 21))
# print(lst)
# print(demo(lst,5))

# import random
# def demo(lst):
#     m = min(lst)
#     results = (m, )
#     positions = [index for index,value in enumerate(lst) if value == m]
#     results += tuple(positions)
#     return results
# x = [random.randint(1, 20) for i in range(50)] # randint 生成一个随机的整数
# print(x)
# print(demo(x))

# from random import randint
# def guess():
#     value = randint(1,1000)
#     maxtimes = 5
#     for i in range(maxtimes):
#         prompt = 'start to guess' if i == 0 else 'guess again:'
#         try:
#             x = int(input(prompt))
#             if x == value:
#                 print("猜对了")
#                 break
#             elif x > value:
#                 print("太大了")
#             else:
#                 print("太小了")
#         except:
#             print("1-999")
#     else:
#         print("游戏结束")
#         print("this value is:",value)
# guess()


# class Test:
#     ''' 这只是一个测试 '''
#     pass
# Test.__doc__

"""
import types
class Car(object):
    price = 100000

    def __init__(self, color):
        self.color = color

car1 = Car("Red")
car2 = Car("Blue")
print(car1.color, Car.price)
Car.price = 110000
Car.name = "QQ"
car1.color = "yellow"
print(car2.color, Car.price, Car.name)
print(car1.color, Car.price, Car.name)

def setSpeed(self,speed):
    self.speed = speed
car1.setSpeed = types.MethodType(setSpeed, car1)
car1.setSpeed(50)
print(car1.speed)

"""



"""
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())


print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred = ', y_test.data)

"""











