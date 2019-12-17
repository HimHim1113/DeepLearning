from keras.layers import UpSampling2D, Input
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.datasets import cifar10
from keras.models import Model

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
input = Input((32,32,3))
x = UpSampling2D(4)(input)
vgg = VGG16(include_top=False, weights="imagenet", input_tensor=x)
model = Model(input, vgg.output)

model.summary()

y_pred = model.predict(preprocess_input(X_test[:16]))
print(y_pred.shape)
