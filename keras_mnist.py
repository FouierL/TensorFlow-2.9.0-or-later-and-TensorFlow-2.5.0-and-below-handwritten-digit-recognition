import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Sequential
(input_train, target_train), (input_test, target_test)=mnist.load_data()
batch_size=600
img_width, img_height=28, 28
loss_function=sparse_categorical_crossentropy
no_classes=10
no_epochs=1
optimizer=adam_v2.Adam
validation_split=0.2
verbosity=1
input_train=input_train.reshape((input_train.shape[0], img_width, img_height, 1))
input_test=input_test.reshape((input_test.shape[0], img_width, img_height, 1))
input_shape=(img_width, img_height, 1)
input_train=input_train.astype('float32')
input_test=input_test.astype('float32')
input_train=input_train/255
input_test=input_test/255
model=Sequential()
model.add=(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add=(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add=(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))
model.compile(
    loss=loss_function,
    optimizer=optimizer,
    metrics=(['accuracy'])
)
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split
)
scores=model.evaluate(input_train, target_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))