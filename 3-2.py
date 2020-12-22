import numpy as np #处理输入数据
import matplotlib as mat
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt #可视化

#输入数据(y,x1,x2)
data = np.array([[450.5,4,171.2],
                 [507.7,4,174.2],
                 [613.9,5,204.3],
                 [563.4,4,218.7],
                 [501.5,4,219.4],
                 [781.5,7,240.4],
                 [541.8,4,273.5],
                 [611.1,5,294.8],
                 [1222.1,10,330.2],
                 [793.2,7,333.1],
                 [660.8,5,366.0],
                 [792.7,6,350.9],
                 [580.8,4,357.9],
                 [612.7,5,359.0],
                 [890.8,7,371.9],
                 [1121.0,9,435.3],
                 [1094.2,8,523.9],
                 [1253,10,604.1]])

test_data = np.array([[450.5,4,171.2],
                    [507.7,4,174.2],
                    [613.9,5,204.3]])

x_data = data[:,1:]
y_data = data[:,0]

x_test = test_data[:,1:]
y_test = test_data[:,0]

#数据训练(使用逻辑回归函数)
regr = linear_model.LinearRegression()
regr.fit(x_data,y_data)
#y=b0+b1*x1+b2*x2
print("线性回归方程为 : y=b0+b1*x1+b2*x2")
print("模型拟合后相关参数：")
print("b1 : %f, b2 : %f"%(regr.coef_[0],regr.coef_[1]))
print("b0 : %f"%regr.intercept_)

#预测
x_pred = [[8,403.5]]
y_pred = regr.predict(x_pred)
print("预测值:%f"%y_pred)
score = regr.score(x_test, y_test)
print(f"模型的score为：{score}")

#可视化
#plot
mat.rcParams['font.family'] = 'SimHei'

max1, max2 = np.max(x_data, axis=0)
min1, min2 = np.min(x_data, axis=0)
x1 = np.linspace(min1, max1, 30)
x2 = np.linspace(min2, max2, 30)
X1, X2 = np.meshgrid(x1, x2)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, color="b")
surf = ax.plot_surface(X1, X2, regr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        rstride=5, cstride=5, cmap=plt.cm.rainbow, alpha=0.5)
fig.colorbar(surf)
plt.title("回归模型")
plt.show()