from keras.datasets import mnist

#mnistデータセットのダウンロード
#x~が画像データ、y~が0~9のラベル
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784) #二次元配列を一次元に変換
y_train = y_train.reshape(60000, 784)
x_train = x_train.astype('float32') # int型をfloat32型に変換
x_test = x_test.astype('float32')
x_train /= 255 #[0~255]の値を[0.0~1.0]に変換


model = Sequential()
model.add(Dense(128), activation='relu', )
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()