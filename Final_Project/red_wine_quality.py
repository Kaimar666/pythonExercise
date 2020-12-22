import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

data_url = r"C:/Users/81275/.keras/datasets/wine_red/wine_red.xlsx"

#数据预处理
def data_process(data_url):
    data = pd.read_excel(data_url)
    x_train = data.iloc[:, 1:11]
    y_train = data.iloc[:, -1].astype('int')
    y_train[y_train <= 5] = 0
    y_train[y_train >= 6] = 1
    return np.array(x_train).astype('float'), np.array(y_train).astype('int')

#参数提取格式设置
np.set_printoptions(threshold=np.inf)

#数据读取
x_train, y_train = data_process(data_url)
print(y_train)

#定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#定义优化器、损失函数、评测指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#断点续训
checkpoint_save_path="./checkpoint/red_wine_quality_1.ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

#输入数据
history = model.fit(x_train, y_train, epochs=500, validation_split=0.2, validation_freq=1, callbacks=[cp_callback])

#打印计算图
model.summary()

#参数提取
file=open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()

#数据可视化
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['sparse_categorical_accuracy']
val_accuracy = history.history['val_sparse_categorical_accuracy']

plt.subplot(1, 2, 1)
plt.plot(loss,label='train_loss')
plt.plot(val_loss,label='validation_loss')
plt.title('Loss_Caparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy,label='train_accuracy')
plt.plot(val_accuracy,label='validation_accuracy')
plt.title('Accuracy_Caparison')
plt.legend()
plt.show()