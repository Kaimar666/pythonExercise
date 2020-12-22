import numpy as np #处理输入数据
from sklearn import linear_model
import matplotlib.pyplot as plt
#Axes3D 用来绘制3D图形的类。
from mpl_toolkits.mplot3d import Axes3D

#输入数据(x1,x2,x3,x4,y)
data = np.array([[7,26,6,60,78.5],
                [1,29,15,52,74.3],
                [11,56,8,20,104.3],
                [11,31,8,47,87.6],
                [7,52,6,33,95.9],
                [11,55,9,22,109.2],
                [3,71,17,6,102.7],
                [1,31,22,44,72.5],
                [2,54,18,22,93.1],
                [21,47,4,26,115.9],
                [1,40,23,34,83.8],
                [11,66,9,12,113.3],
                [10,68,8,12,109.4]])

test_data = np.array([[7,26,6,60,78.5],
                    [1,29,15,52,74.3],
                    [11,56,8,20,104.3]])

x_data = data[:,0:4]
y_data = data[:,-1]

x_test = test_data[:,0:4]
y_test = test_data[:,-1]

#数据训练(使用逻辑回归函数)
regr = linear_model.LinearRegression()
regr.fit(x_data,y_data)

#y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4
print("线性回归方程为 : y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4")
print("模型拟合后相关参数：")
print("b1 : %f, b2 : %f, b3 : %f, b4 : %f"%(regr.coef_[0],regr.coef_[1],regr.coef_[2],regr.coef_[3]))
print("b0 : %f"%regr.intercept_)

#预测
x_pred = [[9,55,10,42]]
y_pred = regr.predict(x_pred)
print("预测值:%f"%y_pred)
score = regr.score(x_test, y_test)
print(f"模型的score为：{score}")
