import torch
from torch.autograd import Variable
import model_mlp as m  # 导入自定义模型 这里还可以是model_conv1d等等
import os
import numpy as np

LEARNING_RATE = 1E-3  # 学习率
BATCH_SIZE = 128  # 批大小
EPOCH = 50  # 循环次数
TRAIN = 0.8  # 训练集占全体数据的比例，剩下为测试集
SAVE_FREQ = 10  # 每几次循环保存

path = "data/" + m.name
if not os.path.exists(path):
    os.mkdir(path)

print("正在准备数据...")
data = np.load("data/stock.npz")  # 加载价格数据
price = data["price"]
label = data["label"]

count = np.size(price, 0)
count_train = int(count * TRAIN)
count_test = count - count_train

price_train = price[0:count_train]
label_train = label[0:count_train]
price_test = price[count_train:count]
label_test = label[count_train:count]

shuffle = np.random.RandomState(seed=10).permutation(count_train)  # 随机排列，用于打乱数据与标签
price_train = price_train[shuffle]
label_train = label_train[shuffle]

loss_fn = torch.nn.MSELoss()  # 均方误差

model = m.Model()  # 初始化模型
model.cuda()  # 使用GPU
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1E-4)  # 最优化器，SGD+Momentum

batches_train = int(count_train / BATCH_SIZE)
batches_test = int(count_test / BATCH_SIZE)

price_train = torch.from_numpy(price_train).float().cuda()  # 将训练集移动到GPU
label_train = torch.from_numpy(label_train).float().cuda()

price_test = torch.from_numpy(price_test).float().cuda()  # 将测试集移动到GPU
label_test = torch.from_numpy(label_test).float().cuda()

for epoch in range(1, EPOCH + 1):  # 遍历EPOCH次数据
    print("Epoch={0}".format(epoch))

    for k in range(batches_train):  # 每次遍历数据分成若干批

        model.train()  # 让模型进入训练模式

        start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
        x = Variable(price_train[start:end])
        y_ = Variable(label_train[start:end])

        optimizer.zero_grad()  # 清零梯度
        y = model.forward(x)  # 正向传播，计算结果
        loss = loss_fn.forward(y, y_)  # 计算误差
        loss.backward()  # 反向传播，计算对误差的梯度
        optimizer.step()  # 最优化器根据梯度调整参数

    test_correct = np.zeros([5], dtype=int)

    for k in range(batches_test):

        model.eval()  # 让模型进入预测模式

        start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
        x = Variable(price_test[start:end])
        y_ = Variable(label_test[start:end])

        y = model.forward(x)  # 正向传播，计算结果
        test_correct += torch.sum(((y_ * y) >= 0).int(), 0).data.cpu().numpy()  # 统计正确预测的个数

    print("test_acc={0}".format(test_correct / count_test))
    if epoch % SAVE_FREQ == 0:
        torch.save(model.state_dict(), "{path}/epoch{epoch}.pkl".format(
            path=path, epoch=epoch))  # 保存模型
