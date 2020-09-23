import numpy as np
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D

from keras.datasets import mnist

from PIL import Image

epochs = 1
rotate_deg = 45

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:][0:6000]
y_train = y_train[0:6000]

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(32,32,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

train_resize = np.reshape(x_train, (-1, 28, 28, 1))
test_resize = np.reshape(y_train, (-1, 28, 28, 1))

train_resize = np.pad(train_resize, [(0,),(2, ), (2, ),(0,)], mode='constant')
test_resize = np.pad(test_resize, [(0,),(2, ), (2, ),(0,)], mode='constant')

train_images = np.reshape(x_train, (-1, 28, 28, 1))
train_images = np.pad(train_images, [(0,),(2, ), (2, ),(0,)], mode='constant')
test_images = np.reshape(x_test, (-1, 28, 28))

for x in range(0,5999):
    im = Image.fromarray(test_images[x], 'F')
    im = im.rotate(rotate_deg)
    test_images[x] = np.asarray(im)

test_images = np.reshape(x_train, (-1, 28, 28, 1))
test_images = np.pad(test_images, [(0,),(2, ), (2, ),(0,)], mode='constant')

model.fit(train_resize, x_test,validation_data=(x_test, y_test),epochs=epochs, batch_size=256, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(x_train, y_train, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])