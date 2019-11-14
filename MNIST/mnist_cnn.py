import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain
import matplotlib.pyplot as plt


train, test = chainer.datasets.get_mnist(ndim=3)


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(1, 20, 5),
            conv2 = L.Convolution2D(20, 50, 5),
            fc1 = L.Linear(800, 500),
            fc2 = L.Linear(500, 10),
        )
    def __call__(self, x):
        cv1 = self.conv1(x)
        h = F.max_pooling_2d(F.relu(cv1), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)


model = L.Classifier(Model())
optimizer = optimizers.MomentumSGD(lr = 0.01, momentum = 0.9)
optimizer.setup(model)

batchsize = 1000
epoch_max = 10 #最大エポック数

def conv(batch, batchsize):
    x = []
    t = []
    for j in range(batchsize):
        x.append(batch[j][0])
        t.append(batch[j][1])
    return Variable(np.array(x)), Variable(np.array(t))

#グラフを作成する
def create_graph(x, y):
    plt.plot(x, y)
    plt.show()


plx = []
ply = []

#MNISTのテストをする
def test_MNIST(data):
    i = chainer.iterators.SerialIterator(test, batchsize).next()
    x, t = conv(i, batchsize)
    for i in range(10):
        result = model(x)
        print('result = {}'.format(result))

for epoch in range(epoch_max):
    for i in chainer.iterators.SerialIterator(train, batchsize, repeat = False):
        x, t = conv(i, batchsize)

        model.zerograds()
        loss = model(x, t)
        loss.backward()
        optimizer.update()

    i = chainer.iterators.SerialIterator(test, batchsize).next()
    x, t = conv(i, batchsize)
    loss = model(x, t)

    plx.append(epoch + 1)
    ply.append(loss.data)
    print('epoch = {}, loss = {}'.format(epoch + 1, loss.data))
    
create_graph(plx, ply)

test_MNIST(test)
