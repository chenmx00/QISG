import torch
from torch.autograd import Variable
import model_mlp as m
import os
import numpy as np

LEARNING_RATE = 1E-3
BATCH_SIZE = 128
EPOCH = 50
TRAIN = 0.8
SAVE_FREQ = 10

path = "data/" + m.name
if not os.path.exists(path):
    os.mkdir(path)

print("正在准备数据...")
data = np.load("data/stock.npz")
price = data["price"]
label = data["label"]

print(np.sum(label[:, 4] == 0))
exit()
count = np.size(price, 0)
count_train = int(count * TRAIN)
count_test = count - count_train

price_train = price[0:count_train]
label_train = label[0:count_train]
price_test = price[count_train:count]
label_test = label[count_train:count]

shuffle = np.random.RandomState(seed=10).permutation(count_train)
price_train = price_train[shuffle]
label_train = label_train[shuffle]

loss_fn = torch.nn.MSELoss()

model = m.Model()
model.cuda()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1E-4)

batches_train = int(count_train / BATCH_SIZE)
batches_test = int(count_test / BATCH_SIZE)

price_train = torch.from_numpy(price_train).float().cuda()
label_train = torch.from_numpy(label_train).float().cuda()

price_test = torch.from_numpy(price_test).float().cuda()
label_test = torch.from_numpy(label_test).float().cuda()

for epoch in range(1, EPOCH + 1):
    print("Epoch={0}".format(epoch))

    for k in range(batches_train):

        model.train()

        start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
        x = Variable(price_train[start:end])
        y_ = Variable(label_train[start:end])

        optimizer.zero_grad()
        y = model.forward(x)
        loss = loss_fn.forward(y, y_)
        loss.backward()
        optimizer.step()

    test_correct = np.zeros([5], dtype=int)

    for k in range(batches_test):

        model.eval()

        start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
        x = Variable(price_test[start:end])
        y_ = Variable(label_test[start:end])

        y = model.forward(x)
        test_correct += torch.sum(((y_ * y) >= 0).int(), 0).data.cpu().numpy()

    print("test_acc={0}".format(test_correct / count_test))
    if epoch % SAVE_FREQ == 0:
        torch.save(model.state_dict(), "{path}/epoch{epoch}.pkl".format(
            path=path, epoch=epoch))
