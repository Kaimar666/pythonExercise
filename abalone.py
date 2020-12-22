import os
import csv
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize

# #参数提取中设置格式
# np.set_printoptions(threshold=np.inf)

###数据预处理
train_url = "C:/Users/81275/.keras/datasets/abalone/abalone_train.data"
test_url = "C:/Users/81275/.keras/datasets/abalone/abalone_test.data"

def load_data(data_url):
    raw_data = np.genfromtxt(data_url, dtype='str', comments='%', delimiter=',')
    # raw_data = pd.read_csv(data_url, delimiter=',', dtype='str')
    raw_data[raw_data == 'M'] = '0'
    raw_data[raw_data == 'F'] = '1'
    raw_data[raw_data == 'I'] = '2'
    label = raw_data[:, 0].astype('int')
    feature = raw_data[:, 1:].astype('float')
    return np.array(feature), label

x_train_data, y_train_data = load_data(train_url)
x_test_data, y_test_data = load_data(test_url)



###数据预处理
#查看数据分布
x_train_data = pd.DataFrame(x_train_data)
y_train_data = pd.DataFrame(y_train_data)
x_train_data.columns = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
y_train_data.columns = ['Sex']
sn.countplot(x=y_train_data['Sex'])
plt.show()
# print(x_train_data)
print(y_train_data['Sex'].value_counts())


# #数据打乱
# np.random.seed(116)
# np.random.shuffle(x_train_data)
# np.random.seed(116)
# np.random.shuffle(y_train_data)
# np.random.seed(116)
# np.random.shuffle(x_test_data)
# np.random.seed(116)
# np.random.shuffle(y_test_data)
# tf.random.set_seed(116)
#
# #独热码标签处理
# y_train_data = label_binarize(y_train_data, classes=[0, 1, 2])
# y_test_data = label_binarize(y_test_data, classes=[0, 1, 2])
# print(y_train_data)
# ###数据预处理结束 包含训练集x_train_data,训练集标签y_train_data 测试集x_test_data,测试集标签y_test_data
#
# # #定义网络结构为64*128*3的三层全连接网络结构
# # model = tf.keras.Sequential([
# #     #tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(64, activation='relu'),
# #     tf.keras.layers.Dense(128, activation='relu'),
# #     tf.keras.layers.Dense(3, activation='tanh')
# # ])
#
# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(64, activation='relu')
#         self.d3 = Dense(3, activation='softmax')
#
#     def call(self, x):
#         x = self.flatten(x)
#         x = self.d1(x)
#         x = self.d2(x)
#         y = self.d3(x)
#         return y
#
# model = MyModel()
#
# #定义优化器、损失函数、评价指标
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#               metrics=['categorical_accuracy'])
#
# # #断点续训
# # checkpoint_save_path="./checkpoint/abalone.ckpt"
# # if os.path.exists(checkpoint_save_path+'.index'):
# #     print('-------------load the model-----------------')
# #     model.load_weights(checkpoint_save_path)
# #
# # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
# #                                                  save_weights_only=True,
# #                                                  save_best_only=True)
#
#
# #输入数据
# # history = model.fit(x_train_data, y_train_data, epochs=150,
# #                     validation_data=(x_test_data,y_test_data), validation_freq=1, callbacks=[cp_callback])
# history = model.fit(x_train_data, y_train_data, epochs=150, validation_data=(x_test_data,y_test_data), validation_freq=1)
#
# #计算图
# model.summary()
#
# # #参数提取
# # #print(model.trainable_variables)
# # file=open('./weights.txt','w')
# # for v in model.trainable_variables:
# #     file.write(str(v.name)+'\n')
# #     file.write(str(v.shape)+'\n')
# #     file.write(str(v.numpy())+'\n')
# # file.close()
#
# #可视化
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# accuracy = history.history['categorical_accuracy']
# val_accuracy = history.history['val_categorical_accuracy']
#
# plt.subplot(1, 2, 1)
# plt.plot(accuracy, label='train_accuracy')
# plt.plot(val_accuracy, label='validation_accuracy')
# plt.title('Train and Validation Accuracy')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(loss, label='train_loss')
# plt.plot(val_loss, label='validation_loss')
# plt.title('Train and Validation Loss')
# plt.legend()
# plt.show()