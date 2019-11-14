import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1 = L.Linear(3, 64),
            l2 = L.Linear(64, 64),
            l3 = L.Linear(64, 64),
            l4 = L.Linear(64, 64),
            l5 = L.Linear(64, 64),
            l6 = L.Linear(64, 48),
            l7 = L.Linear(48, 24),
            l8 = L.Linear(24, 1),
        )
    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = F.sigmoid(self.l4(h))
        h = F.sigmoid(self.l5(h))
        h = F.sigmoid(self.l6(h))
        h = F.sigmoid(self.l7(h))
        return self.l8(h)

model = Model()
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

x = Variable(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype = np.float32))
t = Variable(np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype = np.float32))

for i in range(0,3000):
    model.cleargrads()
    y = model(x)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()

    print("loss : ", loss.data)

print(y.data)
