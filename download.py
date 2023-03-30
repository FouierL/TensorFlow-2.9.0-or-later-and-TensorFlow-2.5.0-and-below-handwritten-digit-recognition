# 手写数字识别 -- CNN神经网络训练
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import adam_v2
def print_history(history):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
    plt.show()
# 1、载入数据
mnist = tf.keras.datasets.mnist
(train_data, train_target), (test_data, test_target) = mnist.load_data()
# 2、改变数据维度
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
# 3、归一化
train_data = train_data / 255.0
test_data = test_data / 255.0
# 4、独热编码
train_target = tf.keras.utils.to_categorical(train_target, num_classes=10)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=10)  # 10种结果
# 5、搭建CNN卷积神经网络
model = Sequential()
# 5-1、第一层：卷积层+池化层
# 第一个卷积层
model.add(Convolution2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
# 第一个池化层
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', ))
# 5-2、第二层：卷积层+池化层
# 第二个卷积层
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
# 第二个池化层
model.add(MaxPooling2D(2, 2, 'same'))
# 5-3、扁平化
model.add(Flatten())
# 5-4、第三层：第一个全连接层
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
# 5-5、第四层：第二个全连接层（输出层）
model.add(Dense(10, activation='softmax'))
# 6、编译
model.compile(optimizer=adam_v2.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# 7、训练
history=model.fit(train_data, train_target, batch_size=64, epochs=10, validation_data=(test_data, test_target))
# 8、保存模型
model.summary()
model.save('mnist.h5')
# 9、画图
print(history.params)
print_history(history) #调用绘图函数