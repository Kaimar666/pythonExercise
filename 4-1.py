import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import linear_model

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)#使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train_data = x_data[:-20]
y_train_data = y_data[:-20]
x_test_data = x_data[-20:]
y_test_data = y_data[-20:]

model = linear_model.LogisticRegression(multi_class='auto')

model.fit(x_train_data, y_train_data)
print('the weight of Logistic Regression:\n',model.coef_)
print('the intercept(b0) of Logistic Regression:\n',model.intercept_)
pred = model.predict(x_test_data)
print("真实分类:", y_test_data)
print("预测分类:", pred)
print("模型准确率:",model.score(x_test_data, y_test_data))


#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#数据可视化
x1_min, x1_max = x_data[:, 0].min(), x_data[:, 0].max()   # 第1列的范围,花萼长度
x2_min, x2_max = x_data[:, 1].min(), x_data[:, 1].max()   # 第2列的范围,花萼宽度
x3_min, x3_max = x_data[:, 2].min(), x_data[:, 2].max()   # 第3列的范围,花瓣长度
x4_min, x4_max = x_data[:, 3].min(), x_data[:, 3].max()   # 第4列的范围,花瓣宽度
t1 = np.linspace(x1_min, x1_max, 2)
t2 = np.linspace(x2_min, x2_max, 2)
t3 = np.linspace(x3_min, x3_max, 2)
t4 = np.linspace(x4_min, x4_max, 2)


'''plt.subplot(1, 2, 1)'''
X1, X2 = np.meshgrid(t1, t2)
fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.scatter(x_data[:, 0], x_data[:, 1], y_data, color="c")
ax1.scatter(x_test_data[:, 0], x_test_data[:, 1], y_test_data, color="r")
#ax1.plot_surface(X1, X2, np.array(pred.reshape(1,20)), rstride=1, cstride=1, cmap=plt.cm.rainbow, alpha=0.5)
'''plt.title("回归模型")
plt.scatter(x_data[:, 0], x_data[:, 1], color='red')
plt.plot(x_test_data, pred, color='blue', label='预测')
'''
plt.title("花萼长度and花萼宽度")

'''plt.subplot(1, 2, 2)'''
X3, X4 = np.meshgrid(t3, t4)
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.scatter(x_data[:, 2], x_data[:, 3], y_data, color="r")
ax2.scatter(x_test_data[:, 2], x_test_data[:, 3], y_test_data, color="g")
#ax2.plot_surface(X1, X2, pred.reshape(X1.shape()), rstride=1, cstride=1, cmap=plt.cm.rainbow, alpha=0.5)
'''plt.scatter(x_data[:, 2], x_data[:, 3], color='red')
plt.plot(x_test_data, pred, color='blue')
'''
plt.title("花瓣长度and花瓣宽度")
plt.show()


"""
###方法二###
#定义网络结构，第一层为64个神经元；第二层为64个神经元；输出为三分类，所以输出节点层为3
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

#定义优化器，损失函数，
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=100, validation_freq=1, validation_split=0.1)

model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.title('accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('loss')
plt.legend()
plt.show()
"""